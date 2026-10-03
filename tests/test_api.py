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
        assert job["result"]["summary"]["n_objects_3d"] >= 1
        assert client.get(f"/api/jobs/{job['id']}/files/objects_3d.csv").text.startswith("label,volume_um3")
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


def test_annotations_labels_and_learned_model(client):
    import base64
    from PIL import Image

    ds, gt, centers = upload_stack(client, n_z=4, with_reference=False)
    did = ds["id"]
    # Clicks are saved per slice and survive a reload
    client.put(f"/api/datasets/{did}/annotations", json={"channel": 0, "z": 1, "pos": [list(c) for c in centers[:2]], "neg": bg_points(gt)})
    assert client.get(f"/api/datasets/{did}/annotations").json()["0:1"]["pos"] == [list(c) for c in centers[:2]]
    client.put(f"/api/datasets/{did}/annotations", json={"channel": 0, "z": 1, "pos": [], "neg": []})
    assert client.get(f"/api/datasets/{did}/annotations").json() == {}

    # Label slice 0 from a segmentation result, and slice 1 from a painted mask
    body = {"channel": 0, "z": 0, "pos": [list(c) for c in centers], "neg": bg_points(gt), "settings": SETTINGS}
    assert client.post(f"/api/datasets/{did}/segment", json=body).status_code == 200
    assert client.post(f"/api/datasets/{did}/labels", json={"channel": 0, "z": 0}).status_code == 200
    rgba = np.zeros(gt.shape + (4,), np.uint8)
    rgba[gt] = (255, 255, 255, 255)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    r = client.post(f"/api/datasets/{did}/labels", json={"channel": 0, "z": 1, "mask_png": url})
    assert r.json()["labels"] == [{"channel": 0, "z": 0}, {"channel": 0, "z": 1}]
    assert client.get(f"/api/datasets/{did}/labels/png", params={"c": 0, "z": 1}).status_code == 200

    # Without a reference channel, results are scored against the saved label
    r = client.post(f"/api/datasets/{did}/segment", json={**body, "z": 1}).json()
    assert r["evaluation"]["against"] == "your saved label"

    # Learned model: needs labels, reports cross-validation, then segments and runs stacks
    assert client.post(f"/api/datasets/{did}/segment", json={"method": "learned", "channel": 0, "z": 2, "settings": SETTINGS}).status_code == 400
    r = client.post(f"/api/datasets/{did}/head", json={"channel": 0, "settings": SETTINGS})
    assert r.status_code == 200, r.text
    info = r.json()
    assert info["cv"]["chosen"] in ("linear", "mlp") and 0 <= info["cv"][info["cv"]["chosen"]]["mean_dice"] <= 1
    r = client.post(f"/api/datasets/{did}/segment", json={"method": "learned", "channel": 0, "z": 2, "settings": SETTINGS, "uncertainty": True})
    assert r.status_code == 200 and r.json()["threshold_source"].startswith("learned")
    r = client.post(f"/api/datasets/{did}/segment", json={"method": "learned", "channel": 0, "z": 2, "settings": {**SETTINGS, "vit_size": 140}})
    assert r.status_code == 400 and "retrain" in r.json()["detail"]
    job = client.post(f"/api/datasets/{did}/stack", json={"method": "learned", "channel": 0, "settings": SETTINGS}).json()
    for _ in range(100):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert job["status"] == "done", job
    assert client.delete(f"/api/datasets/{did}/labels", params={"c": 0, "z": 1}).json()["labels"] == [{"channel": 0, "z": 0}]


def test_histomorphometry_endpoint(client):
    ds = client.post("/api/datasets/demo").json()
    did = ds["id"]
    import glob
    from pathlib import Path
    stack = tifffile.imread(glob.glob(str(Path(client.app.state.store.root) / "datasets" / did / "*.tif"))[0])
    z = 6
    gt_cells = stack[z, 2] > 0
    bone = stack[z, 0] > np.percentile(stack[z, 0], 60)
    ys, xs = np.nonzero(gt_cells)
    by, bx = np.nonzero(bone & ~gt_cells)
    ny, nx = np.nonzero(~bone & ~gt_cells)
    pick = lambda a, b, k: [[int(a[i]), int(b[i])] for i in np.linspace(0, len(a) - 1, k).astype(int)]
    s = {"backbone": "classic", "vit_size": 252}
    # Segment bone on channel 0 and cells on channel 1, then measure
    assert client.post(f"/api/datasets/{did}/segment", json={"channel": 0, "z": z, "pos": pick(by, bx, 8), "neg": pick(ny, nx, 8), "settings": s}).status_code == 200
    assert client.post(f"/api/datasets/{did}/segment", json={"channel": 1, "z": z, "pos": pick(ys, xs, 6), "neg": pick(ny, nx, 8), "settings": s}).status_code == 200
    r = client.post(f"/api/datasets/{did}/histomorphometry", json={"z": z, "bone": {"channel": 0}, "cells": {"channel": 1}, "settings": s})
    assert r.status_code == 200, r.text
    out = r.json()
    assert 0 < out["summary"]["B.Ar/T.Ar_%"] < 100 and out["summary"]["B.Pm_mm"] > 0
    assert out["overlay_png"].startswith("data:image/png")
    assert client.get(f"/api/datasets/{did}/histomorphometry/cells.csv", params={"z": z}).text.startswith("label,")
    # Reference channel as the cell source, and a clear error for a missing result
    assert client.post(f"/api/datasets/{did}/histomorphometry", json={"z": z, "bone": {"channel": 0}, "cells": {"channel": 1, "source": "reference"}, "settings": s}).status_code == 200
    r = client.post(f"/api/datasets/{did}/histomorphometry", json={"z": 2, "bone": {"channel": 0}, "cells": {"channel": 1}, "settings": s})
    assert r.status_code == 400 and "Segment channel" in r.json()["detail"]


def test_region_of_interest(client):
    ds, gt, centers = upload_stack(client, n_z=2)
    did = ds["id"]
    h, w = gt.shape
    body = {"channel": 0, "z": 0, "pos": [list(c) for c in centers], "neg": bg_points(gt), "settings": SETTINGS}
    full = client.post(f"/api/datasets/{did}/segment", json=body).json()["stats"]
    # Left half only
    r = client.put(f"/api/datasets/{did}/roi", json={"polygon": [[0, 0], [0, w // 2], [h, w // 2], [h, 0]]})
    assert r.status_code == 200 and r.json()["area_um2"] == pytest.approx(ds["voxel_um"][1] * ds["voxel_um"][2] * h * (w // 2), rel=0.05)
    half = client.post(f"/api/datasets/{did}/segment", json=body).json()["stats"]
    assert half["image_area_um2"] < 0.55 * full["image_area_um2"] and half["area_um2"] <= full["area_um2"]
    assert client.get("/api/datasets/" + did).json()["roi"] is not None
    assert client.put(f"/api/datasets/{did}/roi", json={"polygon": [[0, 0], [5, 5]]}).status_code == 400
    assert client.put(f"/api/datasets/{did}/roi", json={"polygon": None}).json()["roi"] is None


def test_histomorphometry_roi_edge_is_not_surface():
    from boneseg import histo

    bone = np.zeros((100, 100), bool)
    bone[:, :60] = True
    roi = np.zeros_like(bone)
    roi[:, 20:80] = True                       # Region cuts the bone at x = 20
    s, _ = histo.histomorphometry(bone, np.zeros_like(bone), (1.0, 1.0), roi=roi)
    assert 90 < s["B.Pm_mm"] * 1000 < 110      # Only the real edge at x = 59
    assert s["B.Ar/T.Ar_%"] == pytest.approx(100 * 40 / 60)


def test_pixel_size_override(client, tmp_path):
    from PIL import Image

    img, gt, centers = make_blobs()
    buf = io.BytesIO()
    Image.fromarray((img * 255).astype(np.uint8)).save(buf, format="PNG")
    ds = client.post("/api/datasets", files={"file": ("cells.png", buf.getvalue(), "image/png")}).json()
    assert not ds["voxel_size_known"]
    body = {"channel": 0, "z": 0, "pos": [list(c) for c in centers], "neg": bg_points(gt), "settings": SETTINGS}
    area_px = client.post(f"/api/datasets/{ds['id']}/segment", json=body).json()["stats"]["area_um2"]
    r = client.patch(f"/api/datasets/{ds['id']}", json={"voxel_um_override": [1.0, 0.5, 0.5]})
    assert r.json()["voxel_size_known"] and r.json()["voxel_um"] == [1.0, 0.5, 0.5]
    area_um = client.post(f"/api/datasets/{ds['id']}/segment", json=body).json()["stats"]["area_um2"]
    assert area_um == pytest.approx(area_px * 0.25, rel=0.02)
    assert client.patch(f"/api/datasets/{ds['id']}", json={"voxel_um_override": [1.0, 0.0, 0.5]}).status_code == 400
    # The override survives a restart
    c2 = TestClient(create_app(client.app.state.store.root))
    assert c2.get(f"/api/datasets/{ds['id']}").json()["voxel_um"] == [1.0, 0.5, 0.5]


def test_report(client):
    ds, gt, centers = upload_stack(client, n_z=3)
    did = ds["id"]
    client.patch(f"/api/datasets/{did}", json={"reference_channel": 1})
    assert client.get(f"/api/datasets/{did}/report", params={"c": 0, "z": 1}).status_code == 404
    body = {"channel": 0, "z": 1, "pos": [list(c) for c in centers], "neg": bg_points(gt), "settings": SETTINGS}
    client.post(f"/api/datasets/{did}/segment", json=body)
    job = client.post(f"/api/datasets/{did}/stack", json={**body, "ref_z": 1}).json()
    for _ in range(100):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] == "done":
            break
        time.sleep(0.05)
    assert job["meta"]["dataset_id"] == did
    r = client.get(f"/api/datasets/{did}/report", params={"c": 0, "z": 1})
    assert r.status_code == 200
    page = r.text
    for text in ("Segmentation", "Dice", "Latest stack run", "Methods", "data:image/png;base64", "6 background clicks"):
        assert text in page, text
    r = client.get(f"/api/datasets/{did}/report", params={"c": 0, "z": 1, "download": True})
    assert "attachment" in r.headers["content-disposition"]
