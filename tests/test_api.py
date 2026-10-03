import io
import time

import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient

from boneseg.api import create_app
from tests.conftest import make_blobs

SETTINGS = {"backbone": "classic", "vit_size": 252}


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(tmp_path / "data"))


def upload_stack(client, n_z=4, with_reference=True):
    planes, refs = [], []
    for z in range(n_z):
        img, gt, centers = make_blobs(seed=0)
        planes.append(img * (0.9 + 0.02 * z))
        refs.append(gt.astype(np.float32))
    chans = [np.stack(planes)] + ([np.stack(refs)] if with_reference else [])
    arr = np.stack(chans).transpose(1, 0, 2, 3)  # Z, C, Y, X for ImageJ
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr.astype(np.float32), imagej=True, metadata={"axes": "ZCYX", "spacing": 2.0, "unit": "um"},
                     resolution=(2.0, 2.0))
    r = client.post("/api/datasets", files={"file": ("stack.tif", buf.getvalue(), "image/tiff")})
    assert r.status_code == 200, r.text
    _, gt, centers = make_blobs(seed=0)
    return r.json(), gt, centers


def bg_points(gt):
    ys, xs = np.nonzero(~gt)
    idx = np.linspace(0, len(ys) - 1, 6).astype(int)
    return [[int(ys[i]), int(xs[i])] for i in idx]


def test_health_and_index(client):
    assert client.get("/api/health").json()["backbones"][0]["id"] == "classic"
    assert "boneseg" in client.get("/").text.lower()


def test_upload_plane_segment_export(client):
    ds, gt, centers = upload_stack(client)
    assert ds["n_z"] == 4 and ds["n_channels"] == 2 and ds["voxel_um"] == pytest.approx([2.0, 0.5, 0.5])
    r = client.get(f"/api/datasets/{ds['id']}/plane", params={"c": 0, "z": 1})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"

    # Point the app at the reference channel so it reports Dice
    client.patch(f"/api/datasets/{ds['id']}", json={"reference_channel": 1})
    body = {"channel": 0, "z": 1, "pos": [list(c) for c in centers[:3]], "neg": bg_points(gt), "settings": SETTINGS, "uncertainty": True}
    r = client.post(f"/api/datasets/{ds['id']}/segment", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["mask_png"].startswith("data:image/png") and out["evaluation"]["dice"] > 0.6
    assert out["stats"]["n_objects"] >= 1 and "uncertainty_png" in out

    r = client.get(f"/api/datasets/{ds['id']}/export/mask", params={"c": 0, "z": 1, "fmt": "tif"})
    assert tifffile.imread(io.BytesIO(r.content)).shape == gt.shape
    r = client.get(f"/api/datasets/{ds['id']}/export/objects.csv", params={"c": 0, "z": 1})
    assert r.text.startswith("label,area_um2")
    assert client.get(f"/api/datasets/{ds['id']}/export/mask", params={"c": 0, "z": 3}).status_code == 404


def test_errors(client):
    ds, gt, _ = upload_stack(client, n_z=1, with_reference=False)
    r = client.post(f"/api/datasets/{ds['id']}/segment", json={"channel": 0, "z": 0, "pos": [], "neg": [], "settings": SETTINGS})
    assert r.status_code == 400 and "positive" in r.json()["detail"]
    r = client.post(f"/api/datasets/{ds['id']}/segment", json={"channel": 5, "z": 0, "pos": [[10, 10]], "settings": SETTINGS})
    assert r.status_code == 400
    assert client.get("/api/datasets/nope").status_code == 404
    r = client.post("/api/datasets", files={"file": ("x.bmp", b"123", "image/bmp")})
    assert r.status_code == 400


def test_profiles_and_stack_job(client):
    ds, gt, centers = upload_stack(client)
    client.patch(f"/api/datasets/{ds['id']}", json={"reference_channel": 1})
    r = client.post("/api/profiles", json={"name": "TRAP+ cells", "dataset_id": ds["id"], "channel": 0, "z": 0,
                                           "pos": [list(c) for c in centers], "neg": bg_points(gt), "settings": SETTINGS})
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    assert [p["name"] for p in client.get("/api/profiles").json()] == ["TRAP+ cells"]

    # A profile segments a slice without new clicks
    r = client.post(f"/api/datasets/{ds['id']}/segment", json={"channel": 0, "z": 2, "profile_id": pid, "settings": SETTINGS})
    assert r.status_code == 200 and r.json()["evaluation"]["dice"] > 0.6
    # Mismatched backbone gives a clear error
    r = client.post(f"/api/datasets/{ds['id']}/segment", json={"channel": 0, "z": 2, "profile_id": pid, "settings": {"backbone": "dinov2_s14"}})
    assert r.status_code == 400 and "backbone" in r.json()["detail"]

    for body in ({"profile_id": pid}, {"pos": [list(c) for c in centers], "neg": bg_points(gt)}):
        r = client.post(f"/api/datasets/{ds['id']}/stack", json={"channel": 0, "ref_z": 1, "settings": SETTINGS, **body})
        job = r.json()
        for _ in range(100):
            job = client.get(f"/api/jobs/{job['id']}").json()
            if job["status"] in ("done", "failed"):
                break
            time.sleep(0.05)
        assert job["status"] == "done", job
        assert job["result"]["summary"]["n_slices"] == 4 and job["result"]["summary"]["mean_dice_vs_reference"] > 0.6
        masks = tifffile.imread(io.BytesIO(client.get(f"/api/jobs/{job['id']}/files/masks.tif").content))
        assert masks.shape == (4, *gt.shape)
        assert client.get(f"/api/jobs/{job['id']}/files/slices.csv").text.startswith("z,")
    assert client.delete(f"/api/profiles/{pid}").json()["ok"]


def test_datasets_persist_across_restart(tmp_path):
    c1 = TestClient(create_app(tmp_path / "data"))
    ds, _, _ = upload_stack(c1, n_z=1)
    c1.patch(f"/api/datasets/{ds['id']}", json={"reference_channel": 1})
    c2 = TestClient(create_app(tmp_path / "data"))
    listed = c2.get("/api/datasets").json()
    assert [d["id"] for d in listed] == [ds["id"]] and listed[0]["reference_channel"] == 1
    assert c2.delete(f"/api/datasets/{ds['id']}").json()["ok"]
    assert c2.get("/api/datasets").json() == []
