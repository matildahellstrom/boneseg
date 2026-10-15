"""Regression tests for the bugs found by the bug-hunting run of 10 October 2026 (numbers as in its report).
Front-end bugs (lost clicks, undo, slice races, page switching) are tested in tests/e2e/test_browser.py."""
import io
import time

import numpy as np
import pytest
import torch
from PIL import Image

from tests.test_api import SETTINGS, bg_points, client, upload_stack  # noqa: F401  (client is a fixture)


def wait(client, job, statuses=("done", "failed", "cancelled")):
    for _ in range(400):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] in statuses:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job still {job['status']}")


def test_bug09_cancelled_work_is_not_done(tmp_path):
    from boneseg.store import Store

    store = Store(tmp_path)
    assert store.record_job("stack", {}, lambda d: {"x": 1}, cancelled=lambda: True).status == "cancelled"
    assert store.record_job("stack", {}, lambda d: {"x": 1}, cancelled=lambda: False).status == "done"


def test_bug13_batch_runs_single_channel_files(client):
    a, gt, centers = upload_stack(client, n_z=1, with_reference=False)   # One channel, never chosen in the app
    prof = client.post("/api/profiles", json={"name": "p", "dataset_id": a["id"], "channel": 0, "z": 0, "pos": [list(x) for x in centers],
                                              "neg": bg_points(gt), "settings": SETTINGS}).json()
    job = wait(client, client.post("/api/batch", json={"profile_id": prof["id"], "channel": None, "dataset_ids": [a["id"]], "settings": SETTINGS}).json())
    assert [s["status"] for s in job["result"]["samples"]] == ["done"]


def test_bug14_running_jobs_listed_and_one_stack_run_at_a_time(client, monkeypatch):
    import boneseg.api.segmentation as seg

    ds, gt, centers = upload_stack(client, n_z=4)
    did = ds["id"]
    gate = {"open": False}
    real = seg.run_stack

    def slow(*a, **k):   # Holds the job open until the test lets it finish
        while not gate["open"]:
            time.sleep(0.02)
        return real(*a, **k)
    monkeypatch.setattr(seg, "run_stack", slow)
    body = {"channel": 0, "pos": [list(c) for c in centers], "neg": bg_points(gt), "settings": SETTINGS}
    job = client.post(f"/api/datasets/{did}/stack", json=body).json()
    running = client.get(f"/api/datasets/{did}/jobs", params={"status": "running"}).json()
    assert [j["id"] for j in running] == [job["id"]]
    assert client.post(f"/api/datasets/{did}/stack", json=body).status_code == 409
    gate["open"] = True
    assert wait(client, job)["status"] == "done"
    assert client.get(f"/api/datasets/{did}/jobs", params={"status": "running"}).json() == []


def test_bug16_labels_inside_a_region_ignore_the_outside(client):
    ds, gt, centers = upload_stack(client, n_z=3)
    did = ds["id"]
    h, w = gt.shape
    client.put(f"/api/datasets/{did}/roi", json={"polygon": [[0, 0], [0, w // 2], [h, w // 2], [h, 0]]})
    body = {"channel": 0, "z": 1, "pos": [list(c) for c in centers], "neg": bg_points(gt), "settings": SETTINGS}
    client.post(f"/api/datasets/{did}/segment", json=body)
    client.post(f"/api/datasets/{did}/labels", json={"channel": 0, "z": 1})
    store = client.app.state.store
    region = store.label_region(did, 0, 1)
    assert region is not None and region[:, : w // 2 - 2].all() and not region[:, w // 2 + 2:].any()
    store.delete_label(did, 0, 1)
    assert store.label_region(did, 0, 1) is None


def test_bug16_training_skips_patches_outside_the_valid_region():
    from boneseg.head import fit
    from boneseg.segment import Embedding

    g = torch.nn.functional.normalize(torch.rand(10, 10, 8), dim=-1)
    emb = Embedding(g, 140, 140)
    mask = np.zeros((140, 140), bool)
    valid = np.zeros((140, 140), bool)
    valid[:, :70] = True                 # Only the left half is labelled
    mask[20:50, 10:40] = True
    # With a valid region only the left half's 50 patches are used; the result must still train
    model, dim = fit([(emb, mask, valid)], "linear", epochs=5)
    assert dim == 8 and model(torch.zeros(1, 8)).shape == (1, 1)


def test_bug19_inputs_are_checked(client):
    ds, gt, centers = upload_stack(client, n_z=1)
    did = ds["id"]
    seg = lambda st: client.post(f"/api/datasets/{did}/segment", json={"channel": 0, "z": 0, "pos": [list(centers[0])], "neg": bg_points(gt), "settings": st})  # noqa: E731
    for bad in ({"vit_size": "abc"}, {"neg_weight": "x"}, {"vit_size": 0}, {"shift_passes": 0}, {"refiner": "nope.pt"}, {"clip_low": 99, "clip_high": 1}):
        r = seg({**SETTINGS, **bad})
        assert r.status_code == 400 and r.headers["content-type"].startswith("application/json"), (bad, r.text)
    assert seg({**SETTINGS, "vit_size": "252"}).status_code == 200   # A number given as text is fine
    assert client.put(f"/api/datasets/{did}/annotations", json={"channel": 0, "z": 99, "pos": [[1, 1]], "neg": []}).status_code == 400
    assert client.put(f"/api/datasets/{did}/annotations", json={"channel": 7, "z": 0, "pos": [[1, 1]], "neg": []}).status_code == 400
    assert client.post(f"/api/datasets/{did}/labels", json={"channel": 0, "z": 42}).status_code == 400
    r = client.post(f"/api/datasets/{did}/labels", json={"channel": 0, "z": 0, "mask_png": "data:image/png;base64,bm90IGEgcG5n"})
    assert r.status_code == 400 and "PNG" in r.json()["detail"]
    assert client.put(f"/api/datasets/{did}/roi", content='{"polygon": [[0, 0], [NaN, 5], [9, 9]]}',
                      headers={"content-type": "application/json"}).status_code in (400, 422)


def test_bug20_downloads_are_named_after_the_image(client):
    ds, gt, centers = upload_stack(client, n_z=2)
    job = client.post(f"/api/datasets/{ds['id']}/stack", json={"channel": 0, "pos": [list(c) for c in centers], "neg": bg_points(gt), "settings": SETTINGS}).json()
    wait(client, job)
    r = client.get(f"/api/jobs/{job['id']}/files/slices.csv")
    assert r.status_code == 200 and 'filename="stack_slices.csv"' in r.headers["content-disposition"]


def test_bug21_errors_without_server_paths(client):
    r = client.post("/api/datasets", files={"file": ("broken.png", b"not really a png", "image/png")})
    assert r.status_code == 400 and "broken.png" in r.json()["detail"]
    assert str(client.app.state.store.root) not in r.json()["detail"]
    r = client.post("/api/datasets/from-path", json={"path": "  "})
    assert r.status_code == 400 and "full path" in r.json()["detail"]


def test_bug22_training_as_a_background_job(client):
    ds, gt, centers = upload_stack(client, n_z=4)
    did = ds["id"]
    client.patch(f"/api/datasets/{did}", json={"reference_channel": 1})
    client.post(f"/api/datasets/{did}/labels/from-reference", json={"channel": 0, "n": 2})
    job = client.post(f"/api/datasets/{did}/head", json={"channel": 0, "settings": SETTINGS, "background": True}).json()
    assert job["kind"] == "train"
    job = wait(client, job)
    assert job["status"] == "done" and job["result"]["kind"] in ("linear", "mlp") and "threshold" in job["result"]
    assert client.get(f"/api/datasets/{did}/head", params={"c": 0}).json() is not None


def test_bug23_finetune_cancel_before_the_first_check(monkeypatch):
    import boneseg.finetune as fmod

    class Tiny(torch.nn.Module):   # A stand-in for DINOv2 with the methods SegNet uses
        patch_size, embed_dim = 14, 4

        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
            self.norm = torch.nn.LayerNorm(4)
            self.proj = torch.nn.Conv2d(3, 4, 14, 14)

        def get_intermediate_layers(self, x, n=1, reshape=True, norm=True):
            return [self.proj(x)]

    img = np.random.default_rng(0).random((56, 56)).astype(np.float32)
    gt = img > 0.5
    steps = []
    ft = fmod.finetune([(img, gt)], [(img, gt)], train_blocks=1, steps=500, crop=56, batch=1, pretrained=Tiny(), device=torch.device("cpu"),
                       progress=lambda p, m: steps.append(p), cancelled=lambda: len(steps) >= 3)
    assert ft.info["best_step"] <= 5 and len(steps) <= 5   # Stopped right after cancelling, not at step 100


# Code analysis of 11 October 2026 (code-analysis.md): B = bugs, W = weaknesses

def test_b1_concurrent_annotation_saves_are_all_kept(tmp_path):
    import threading

    from boneseg.store import Store

    store = Store(tmp_path)
    ds = store.add_dataset_file("a.npy", lambda p: np.save(p, np.zeros((1, 40, 10, 10), np.float32)))
    errors = []

    def save(z):
        try:
            store.set_annotation(ds.id, 0, z, [[1, 1]], [])
        except Exception as e:  # noqa: BLE001
            errors.append(e)
    threads = [threading.Thread(target=save, args=(z,)) for z in range(40)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and len(store.get_annotations(ds.id)) == 40


def test_b2_one_structure_alone_and_in_the_multi_path_agree():
    from boneseg import segment
    from boneseg.backbone import get_backbone
    from tests.conftest import make_blobs

    img, gt, centers = make_blobs(seed=0)
    s = segment.SegmentationSettings(backbone="classic", vit_size=252)
    emb = segment.embed_image(get_backbone("classic"), img, s)
    ys, xs = np.nonzero(~gt)
    neg = [[int(ys[i]), int(xs[i])] for i in np.linspace(0, len(ys) - 1, 6).astype(int)]
    pos = [list(c) for c in centers[:4]]
    single = segment.segment(emb, pos, neg, s).mask
    multi = segment.segment_multi(emb, [{"name": "cells", "pos": pos}], neg, s).labels == 1
    assert (single == multi).mean() > 0.995   # Same edge snapping and threshold rule


def test_b3_png_modes(tmp_path):
    from boneseg import io as bio

    Image.fromarray(np.dstack([np.full((20, 30), 100, np.uint8), np.full((20, 30), 255, np.uint8)]), "LA").save(tmp_path / "la.png")
    rgb = np.zeros((20, 30, 3), np.uint8)
    rgb[:, :15] = (200, 30, 30)
    Image.fromarray(rgb).convert("P", palette=Image.ADAPTIVE, colors=4).save(tmp_path / "p.png")
    la = bio.load_volume(tmp_path / "la.png")
    assert la.n_channels == 1 and la.get_plane(0, 0).max() == 100
    p = bio.load_volume(tmp_path / "p.png")
    assert p.channel_names[:3] == ["Red", "Green", "Blue"] and p.get_plane(0, 0).max() == 200


def test_b4_z_context_in_micrometres():
    from boneseg.head import Head, head_input
    from boneseg.segment import Embedding

    calls = []
    embed = lambda z: (calls.append(z), Embedding(torch.nn.functional.normalize(torch.rand(2, 2, 3), dim=-1), 28, 28))[1]  # noqa: E731
    h = Head("classic", 1, 252, "linear", {}, 6, z_context=2, z_context_um=4.0)   # Trained on 2 um slices
    head_input(h, embed, 10, 30, z_um=1.0)                                        # A 1 um stack: 4 slices away
    assert calls == [10, 6, 14]
    with pytest.raises(ValueError, match="needs a stack"):
        head_input(h, embed, 0, 1, z_um=1.0)


def test_b6_prefetch_stops_when_settings_change(tmp_path):
    from boneseg.segment import SegmentationSettings
    from boneseg.store import Store

    store = Store(tmp_path)
    ds = store.add_dataset_file("a.npy", lambda p: np.save(p, np.random.default_rng(0).random((1, 6, 60, 60)).astype(np.float32)))
    a, b = SegmentationSettings(backbone="classic", vit_size=56), SegmentationSettings(backbone="classic", vit_size=84)
    store.prefetch(ds.id, 0, [1, 2], a)
    gen = store._prefetch_gen
    store.prefetch(ds.id, 0, [3], b)
    assert store._prefetch_gen == gen + 1   # Queued slices of the old settings are skipped


def test_w1_large_stacks_label_3d_objects_at_reduced_resolution(client, monkeypatch):
    import boneseg.pipeline as pl

    monkeypatch.setattr(pl, "VOXEL_BUDGET", 1000)
    ds, gt, centers = upload_stack(client, n_z=4)
    job = client.post(f"/api/datasets/{ds['id']}/stack", json={"channel": 0, "pos": [list(c) for c in centers], "neg": bg_points(gt), "settings": SETTINGS}).json()
    job = wait(client, job)
    assert job["status"] == "done" and job["result"]["summary"]["objects_3d_downsampled"] > 1
    import tifffile
    masks = tifffile.imread(io.BytesIO(client.get(f"/api/jobs/{job['id']}/files/masks.tif").content))
    assert masks.shape == (4, *gt.shape)   # Mask stacks stay at full resolution


def test_w2_tiffs_are_read_lazily(tmp_path):
    import tifffile

    from boneseg import io as bio

    arr = (np.random.default_rng(0).random((5, 30, 40)) * 60000).astype(np.uint16)
    tifffile.imwrite(tmp_path / "s.tif", arr, compression="zlib")
    v = bio.load_volume(tmp_path / "s.tif")
    assert hasattr(v, "_file") and all(np.array_equal(v.get_plane(0, z), arr[z]) for z in range(5))


def test_w3_host_origin_and_upload_limits(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from boneseg.api import create_app

    monkeypatch.setenv("BONESEG_MAX_UPLOAD_GB", "0.000001")   # 1 kB
    c = TestClient(create_app(tmp_path / "d", allowed_hosts=["testserver"]))
    assert c.get("/api/health").status_code == 200
    assert c.get("/api/health", headers={"host": "evil.example"}).status_code == 400
    assert c.post("/api/datasets/demo", headers={"origin": "http://evil.example"}).status_code == 403
    assert c.post("/api/datasets/demo", headers={"origin": "http://testserver"}).status_code == 200
    big = io.BytesIO()
    np.save(big, np.zeros((1, 1, 40, 40), np.float32))
    assert c.post("/api/datasets/stream", params={"filename": "big.npy"}, content=big.getvalue()).status_code == 413
    assert c.post("/api/datasets", files={"file": ("big.npy", big.getvalue(), "application/octet-stream")}).status_code == 413


def test_w4_project_export_has_only_this_images_profiles(client):
    import zipfile

    a, gt, centers = upload_stack(client, n_z=1)
    b, _, _ = upload_stack(client, n_z=1)
    body = lambda ds: {"name": "p", "channel": 0, "z": 0, "pos": [list(c) for c in centers], "neg": bg_points(gt), "settings": SETTINGS, "dataset_id": ds["id"]}  # noqa: E731
    pa = client.post("/api/profiles", json=body(a)).json()["id"]
    client.post("/api/profiles", json=body(b))
    z = zipfile.ZipFile(io.BytesIO(client.get(f"/api/datasets/{a['id']}/export/project.zip").content))
    assert [n for n in z.namelist() if n.startswith("profiles/")] == [f"profiles/{pa}.npz"]
    assert str(client.app.state.store.root) not in z.read("dataset.json").decode()


def test_w5_learned_models_load_safely(client, tmp_path):
    ds, gt, centers = upload_stack(client, n_z=3)
    client.patch(f"/api/datasets/{ds['id']}", json={"reference_channel": 1})
    client.post(f"/api/datasets/{ds['id']}/labels/from-reference", json={"channel": 0, "n": 2})
    client.post(f"/api/datasets/{ds['id']}/head", json={"channel": 0, "settings": SETTINGS})
    d = torch.load(client.app.state.store.head_path(ds["id"], 0), weights_only=True)
    assert d["kind"] in ("linear", "mlp")


def test_w7_old_runs_are_pruned(tmp_path):
    from boneseg.store import Store

    store = Store(tmp_path)
    ids = [store.record_job("stack", {"dataset_id": "x"}, lambda d: {}).id for _ in range(5)]
    for i, jid in enumerate(ids):
        store.jobs[jid].created = i
    store.prune_jobs("x", keep=2)
    assert sorted(j.id for j in store.jobs.values()) == sorted(ids[-2:])
    assert not (tmp_path / "jobs" / ids[0]).exists()


def test_w8_two_backbones_stay_loaded(monkeypatch):
    import boneseg.backbone as bmod

    class Fake(bmod.Backbone):
        def __init__(self, name, device):
            self.name = name
    monkeypatch.setattr(bmod, "DinoBackbone", Fake)
    monkeypatch.setattr(bmod, "_cache", {})
    a = bmod.get_backbone("dinov2_s14")
    bmod.get_backbone("dinov2_b14")
    assert bmod.get_backbone("dinov2_s14") is a   # Still loaded, not reloaded
    bmod.get_backbone("dinov2_l14")
    assert set(bmod._cache) == {"dinov2_s14", "dinov2_l14"}   # The least recently used went
