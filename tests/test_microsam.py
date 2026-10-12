"""micro-SAM in the app: the Kaggle package, loading a trained model file, segmenting slices and stacks with it.
micro-SAM itself is replaced by a stub here (training needs a GPU); boneseg/microsam_core.py is checked for real at the
end when micro-sam and its weights are installed."""
import io
import json
import pickle
import zipfile

import numpy as np
import pytest
import tifffile
import torch
from PIL import Image

from tests.test_api import SETTINGS, client, upload_stack  # noqa: F401  (client is a fixture)
from tests.test_bugfixes import wait


def labelled(client):
    ds, gt, centers = upload_stack(client)
    client.patch(f"/api/datasets/{ds['id']}", json={"reference_channel": 1})
    r = client.post(f"/api/datasets/{ds['id']}/labels/from-reference", json={"channel": 0, "n": 3})
    assert r.status_code == 200, r.text
    return ds, gt


def slim_model(path, recipe="full"):
    """A model file shaped like the notebook's output (weights only, tiny)."""
    key = "sam.image_encoder.blocks.0.attn.qkv.w_a_linear_q.weight" if recipe == "lora" else "sam.image_encoder.patch_embed.proj.weight"
    torch.save({"model_state": {key: torch.zeros(2, 2)}, "decoder_state": {"base.0.weight": torch.zeros(2)},
                "boneseg": {"model": "vit_b_lm", "recipe": recipe, "val_dice": 0.81, "n_train": 2, "n_val": 1, "clip": [1.0, 99.5]}}, path)
    return path.read_bytes()


def test_kaggle_package(client):
    ds, gt = labelled(client)
    url = f"/api/datasets/{ds['id']}/microsam/kaggle"
    assert client.post(url, json={"channel": 0, "username": ""}).status_code == 400            # User name needed
    assert client.post(url, json={"channel": 0, "username": "me", "recipe": "qlora"}).status_code == 400
    r = client.post(url, json={"channel": 0, "username": "Some-User", "recipe": "lora", "settings": SETTINGS})
    assert r.status_code == 200, r.text
    assert "attachment" in r.headers["content-disposition"] and r.headers["content-disposition"].endswith('.zip"')
    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = z.namelist()
    top = names[0].split("/")[0]
    meta = json.loads(z.read(f"{top}/dataset/dataset-metadata.json"))
    kmeta = json.loads(z.read(f"{top}/notebook/kernel-metadata.json"))
    split = json.loads(z.read(f"{top}/dataset/boneseg_split.json"))
    assert meta["id"] == f"some-user/{top}" and kmeta["dataset_sources"] == [meta["id"]] and len(kmeta["title"]) <= 50
    assert kmeta["enable_gpu"] == "true" and kmeta["is_private"] == "true"
    assert len(split["train"]) == 2 and len(split["val"]) == 1 and split["recipe"] == "lora"   # Every second label validates
    for n in split["train"] + split["val"]:
        img = np.asarray(Image.open(io.BytesIO(z.read(f"{top}/dataset/images/{n}.png"))))
        inst = np.asarray(Image.open(io.BytesIO(z.read(f"{top}/dataset/instances/{n}.png"))))
        assert img.dtype == np.uint8 and img.shape == inst.shape == gt.shape
        assert inst.max() >= 3 and ((inst > 0) == gt).mean() > 0.99    # One id per blob, covering the label
    nb = json.loads(z.read(f"{top}/notebook/notebook.ipynb"))
    import ast
    for cell in nb["cells"]:
        ast.parse(cell["source"])                                       # Every cell is valid Python
    assert "kaggle datasets create" in z.read(f"{top}/README.txt").decode()


def test_kaggle_package_needs_two_labels(client):
    ds, gt, centers = upload_stack(client)
    r = client.post(f"/api/datasets/{ds['id']}/microsam/kaggle", json={"channel": 0, "username": "me"})
    assert r.status_code == 400 and "2 labels" in r.json()["detail"]


def test_model_file_load_check_and_remove(client, tmp_path):
    ds, gt = labelled(client)
    base = f"/api/datasets/{ds['id']}/microsam"
    assert client.get(f"{base}?c=0").json()["model"] is None
    # Not a model, and a pickle that would run code: both refused without unpickling
    assert client.post(f"{base}/model?c=0", content=b"not a model").status_code == 400

    class Evil:
        def __reduce__(self):
            return (print, ("ran",))
    bad = tmp_path / "evil.pt"
    torch.save({"model_state": Evil(), "decoder_state": {}}, bad, pickle_module=pickle)
    r = client.post(f"{base}/model?c=0", content=bad.read_bytes())
    assert r.status_code == 400 and "Not a micro-SAM model" in r.json()["detail"]
    r = client.post(f"{base}/model?c=0", content=slim_model(tmp_path / "m.pt", "lora"))
    assert r.status_code == 200, r.text
    info = client.get(f"{base}?c=0").json()["model"]
    assert info["recipe"] == "lora" and info["val_dice"] == 0.81
    assert client.delete(f"{base}/model?c=0").json()["ok"] and client.get(f"{base}?c=0").json()["model"] is None


def test_segment_slices_and_stack_with_microsam(client, tmp_path, monkeypatch):
    from boneseg import microsam

    ds, gt = labelled(client)
    body = {"method": "microsam", "channel": 0, "z": 1, "settings": SETTINGS}
    r = client.post(f"/api/datasets/{ds['id']}/segment", json=body)
    assert r.status_code == 400 and "No micro-SAM model" in r.json()["detail"]
    client.post(f"/api/datasets/{ds['id']}/microsam/model?c=0", content=slim_model(tmp_path / "m.pt"))
    seen = []

    def fake_foreground(path, img):   # The "model" finds the blobs, with soft edges
        seen.append(img.shape)
        return np.where(gt, 0.9, 0.1).astype(np.float32)

    monkeypatch.setattr(microsam, "foreground", fake_foreground)
    monkeypatch.setattr(microsam, "require", lambda: None)
    r = client.post(f"/api/datasets/{ds['id']}/segment", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["evaluation"]["dice"] > 0.99 and out["threshold"] == 0.5 and seen == [gt.shape]
    rep = client.get(f"/api/datasets/{ds['id']}/report?c=0&z=1")
    assert rep.status_code == 200 and "micro-SAM" in rep.text and "DINOv2" not in rep.text.split("micro-SAM", 1)[1][:400]
    job = client.post(f"/api/datasets/{ds['id']}/stack", json={"method": "microsam", "channel": 0, "settings": SETTINGS}).json()
    job = wait(client, job)
    assert job["status"] == "done", job
    assert job["meta"]["method"] == "microsam" and job["result"]["summary"]["mean_dice_vs_reference"] > 0.99
    masks = tifffile.imread(io.BytesIO(client.get(f"/api/jobs/{job['id']}/files/masks.tif").content))
    assert masks.shape == (4, *gt.shape)
    # Deleting the dataset removes its micro-SAM model
    client.delete(f"/api/datasets/{ds['id']}")
    assert not list((tmp_path / "data" / "models").glob("microsam_*"))


def test_microsam_core_real_model():
    """micro-SAM's generalist with its decoder on a small image: runs, gives probabilities, finds bright blobs."""
    pytest.importorskip("micro_sam")
    from micro_sam.util import microsam_cachedir
    if not (microsam_cachedir() / "models" / "vit_b_lm_decoder").exists():
        pytest.skip("micro-SAM's vit_b_lm weights are not downloaded")
    from boneseg import microsam_core as MC
    from tests.conftest import make_blobs

    img, gt, _ = make_blobs(h=256, w=256, seed=1)
    m = MC.Model(None, device="cpu")
    fg = m.foreground(img)
    assert fg.shape == gt.shape and 0 <= fg.min() <= fg.max() <= 1
    p = fg >= 0.5
    assert 2 * (p & gt).sum() / (p.sum() + gt.sum()) > 0.6
    m.set_image(img)
    ys, xs = np.nonzero(gt)
    assert m.points([(ys[0], xs[0])], []).any()
