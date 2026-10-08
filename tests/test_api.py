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
    assert out["error_png"].startswith("data:image/png") and out["evaluation"]["false_positive_um2"] >= 0
    assert out["stats"]["n_objects"] >= 1 and "uncertainty_png" in out

    r = client.get(f"/api/datasets/{ds['id']}/export/mask", params={"c": 0, "z": 1, "fmt": "tif"})
    assert tifffile.imread(io.BytesIO(r.content)).shape == gt.shape
    r = client.get(f"/api/datasets/{ds['id']}/export/objects.csv", params={"c": 0, "z": 1})
    assert r.text.startswith("label,area_um2")
    header = r.text.split("\n")[0]
    assert "ch0_Channel_0_mean" in header and "ch1_" not in header  # The reference channel is left out
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
        labels = tifffile.imread(io.BytesIO(client.get(f"/api/jobs/{job['id']}/files/labels_3d.tif").content))
        assert labels.shape == masks.shape and labels.max() == job["result"]["summary"]["n_objects_3d"]
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
    chosen = info["cv"][info["cv"]["chosen"]]
    assert chosen["kind"] in ("linear", "mlp") and chosen["context"] in (1, 5) and 0 <= chosen["mean_dice"] <= 1
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
    # Results made before the region was drawn stay available, clipped to it
    r = client.get(f"/api/datasets/{did}/export/mask", params={"c": 0, "z": 0, "fmt": "tif"})
    assert r.status_code == 200 and not tifffile.imread(io.BytesIO(r.content))[:, w // 2 + 2:].any()
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
    client.patch(f"/api/datasets/{did}", json={"notes": "TRAP stain, <b>batch 2</b>"})
    page_with_notes = client.get(f"/api/datasets/{did}/report", params={"c": 0, "z": 1}).text
    assert "TRAP stain, &lt;b&gt;batch 2&lt;/b&gt;" in page_with_notes and "DINOv2: Learning robust" in page_with_notes
    for text in ("Segmentation", "Dice", "Latest stack run", "Methods", "data:image/png;base64", "6 background clicks"):
        assert text in page, text
    r = client.get(f"/api/datasets/{did}/report", params={"c": 0, "z": 1, "download": True})
    assert "attachment" in r.headers["content-disposition"]


def test_open_by_path_can_be_disabled(tmp_path):
    np.save(tmp_path / "a.npy", np.zeros((20, 20), np.float32))
    on = TestClient(create_app(tmp_path / "d1"))
    first = on.post("/api/datasets/from-path", json={"path": str(tmp_path / "a.npy")}).json()
    again = on.post("/api/datasets/from-path", json={"path": str(tmp_path / "a.npy")}).json()
    assert first["id"] == again["id"] and len(on.get("/api/datasets").json()) == 1
    assert on.post("/api/datasets/from-path", json={"path": str(tmp_path / "nope.npy")}).status_code == 404
    off = TestClient(create_app(tmp_path / "d2", allow_paths=False))
    assert off.post("/api/datasets/from-path", json={"path": str(tmp_path / "a.npy")}).status_code == 403
    assert off.get("/api/health").json()["allow_paths"] is False


def test_training_checks_against_the_reference(client):
    ds, gt, centers = upload_stack(client, n_z=4)
    did = ds["id"]
    client.patch(f"/api/datasets/{did}", json={"reference_channel": 1})
    body = {"channel": 0, "z": 0, "pos": [list(c) for c in centers], "neg": bg_points(gt), "settings": SETTINGS}
    client.post(f"/api/datasets/{did}/segment", json=body)
    client.post(f"/api/datasets/{did}/labels", json={"channel": 0, "z": 0})
    out = client.post(f"/api/datasets/{did}/head", json={"channel": 0, "settings": SETTINGS}).json()
    assert out["reference_check"]["n_slices"] == 3 and 0 <= out["reference_check"]["mean_dice"] <= 1


def test_learned_model_as_profile(client, tmp_path):
    ds, gt, centers = upload_stack(client, n_z=3)
    did = ds["id"]
    client.patch(f"/api/datasets/{did}", json={"reference_channel": 1})
    body = {"channel": 0, "z": 0, "pos": [list(c) for c in centers], "neg": bg_points(gt), "settings": SETTINGS}
    client.post(f"/api/datasets/{did}/segment", json=body)
    client.post(f"/api/datasets/{did}/labels", json={"channel": 0, "z": 0})
    assert client.post(f"/api/datasets/{did}/head/export", json={"channel": 0, "name": "x"}).status_code == 400
    client.post(f"/api/datasets/{did}/head", json={"channel": 0, "settings": SETTINGS})
    r = client.post(f"/api/datasets/{did}/head/export", json={"channel": 0, "name": "Learned cells"})
    assert r.status_code == 200, r.text
    prof = r.json()
    assert prof["kind"] == "learned"
    # A second dataset segments with the learned profile and no clicks
    ds2, _, _ = upload_stack(client, n_z=2)
    client.patch(f"/api/datasets/{ds2['id']}", json={"reference_channel": 1})
    r = client.post(f"/api/datasets/{ds2['id']}/segment", json={"channel": 0, "z": 1, "profile_id": prof["id"], "settings": SETTINGS})
    assert r.status_code == 200 and r.json()["threshold_source"].startswith("learned") and r.json()["evaluation"]["dice"] > 0.5
    # Batch with the learned profile from the command line
    from boneseg.__main__ import main
    store_root = client.app.state.store.root
    src = next((store_root / "datasets" / ds2["id"]).glob("*.tif"))
    main(["batch", str(src), "--profile", prof["id"], "--channel", "0", "--reference", "1", "--out", str(tmp_path / "o"), "--data-dir", str(store_root)])
    import pandas as pd
    out = pd.read_csv(tmp_path / "o" / "summary.csv")
    assert out["status"].iloc[0] == "ok" and out["mean_dice_vs_reference"].iloc[0] > 0.5


def test_side_view(client):
    ds, gt, centers = upload_stack(client, n_z=4)
    did = ds["id"]
    r = client.get(f"/api/datasets/{did}/xz", params={"c": 0, "y": 50})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert float(r.headers["x-stretch"]) == pytest.approx(4.0)  # 2 um slices, 0.5 um pixels
    body = {"channel": 0, "ref_z": 0, "pos": [list(c) for c in centers], "neg": bg_points(gt), "settings": SETTINGS}
    job = client.post(f"/api/datasets/{did}/stack", json=body).json()
    for _ in range(100):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] == "done":
            break
        time.sleep(0.05)
    assert client.get(f"/api/datasets/{did}/xz", params={"c": 0, "y": int(centers[0][0]), "job_id": job["id"]}).status_code == 200
    assert client.get(f"/api/datasets/{did}/xz", params={"c": 0, "y": 9999}).status_code == 400


def test_streaming_upload(client):
    img, gt, _ = make_blobs()
    buf = io.BytesIO()
    tifffile.imwrite(buf, np.stack([img, img]).astype(np.float32))
    r = client.post("/api/datasets/stream", params={"filename": "../../evil name.tif"}, content=buf.getvalue(),
                    headers={"Content-Type": "application/octet-stream"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["name"] == "evil name.tif" and d["n_z"] == 2
    root = client.app.state.store.root / "datasets"
    assert not list(root.glob(".incoming-*"))   # No temporary file left behind
    r = client.post("/api/datasets/stream", params={"filename": "x.bmp"}, content=b"abc")
    assert r.status_code == 400 and not list(root.glob(".incoming-*"))


def test_profile_download_and_import(client, tmp_path):
    ds, gt, centers = upload_stack(client, n_z=1)
    prof = client.post("/api/profiles", json={"name": "Shared", "dataset_id": ds["id"], "channel": 0, "z": 0,
                                              "pos": [list(c) for c in centers], "neg": bg_points(gt), "settings": SETTINGS}).json()
    blob = client.get(f"/api/profiles/{prof['id']}/download").content
    other = TestClient(create_app(tmp_path / "other"))
    r = other.post("/api/profiles/import", files={"file": ("p.npz", blob, "application/octet-stream")})
    assert r.status_code == 200 and r.json()["name"] == "Shared"
    assert other.post("/api/profiles/import", files={"file": ("p.npz", b"not a profile", "application/octet-stream")}).status_code == 400


def test_study_comparison_and_job_persistence(client):
    ids = []
    for i in range(4):
        ds, gt, centers = upload_stack(client, n_z=2)
        ids.append(ds["id"])
        client.patch(f"/api/datasets/{ds['id']}", json={"group": "control" if i < 2 else "treated"})
        body = {"channel": 0, "ref_z": 0, "pos": [list(c) for c in centers], "neg": bg_points(gt), "settings": SETTINGS}
        job = client.post(f"/api/datasets/{ds['id']}/stack", json=body).json()
        for _ in range(100):
            if client.get(f"/api/jobs/{job['id']}").json()["status"] == "done":
                break
            time.sleep(0.05)
    out = client.get("/api/study", params={"metric": "mean_area_fraction"}).json()
    assert len(out["rows"]) == 4 and all(r["has_stack_run"] for r in out["rows"])
    assert out["comparison"]["test"] == "Mann-Whitney U" and set(out["comparison"]["groups"]) == {"control", "treated"}
    assert client.get("/api/study.csv").text.startswith("dataset_id,")
    assert client.get("/api/study", params={"metric": "nope"}).status_code == 400
    # Stack results survive a restart
    c2 = TestClient(create_app(client.app.state.store.root))
    assert all(r["has_stack_run"] for r in c2.get("/api/study").json()["rows"])


def test_empty_mask_gives_null_not_error(client):
    ds, gt, centers = upload_stack(client, n_z=1)
    client.patch(f"/api/datasets/{ds['id']}", json={"reference_channel": 1})
    # A manual threshold of 1.0 keeps only the single top pixel or nothing, so HD95 is undefined
    body = {"channel": 0, "z": 0, "pos": [list(c) for c in centers], "neg": bg_points(gt),
            "settings": {**SETTINGS, "threshold_mode": "manual", "manual_threshold": 1.01}}
    r = client.post(f"/api/datasets/{ds['id']}/segment", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["stats"]["n_objects"] == 0 and r.json()["evaluation"]["hd95_um"] is None


def test_multi_structure_endpoint(client):
    ds, gt, centers = upload_stack(client, n_z=1, with_reference=False)
    did = ds["id"]
    body = {"channel": 0, "z": 0, "neg": bg_points(gt), "settings": SETTINGS,
            "structures": [{"name": "big cells", "color": "#ff8800", "pos": [list(c) for c in centers[:2]]},
                           {"name": "small cells", "color": "#00ff88", "pos": [list(c) for c in centers[2:4]]},
                           {"name": "empty", "pos": []}]}
    r = client.post(f"/api/datasets/{did}/segment_multi", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert [s["name"] for s in out["structures"]] == ["big cells", "small cells"] and out["labels_png"].startswith("data:image/png")
    labels = tifffile.imread(io.BytesIO(client.get(f"/api/datasets/{did}/export/labels", params={"c": 0, "z": 0}).content))
    assert labels.shape == gt.shape and set(np.unique(labels)) <= {0, 1, 2}
    # Clicks for extra structures are saved with the slice
    client.put(f"/api/datasets/{did}/annotations", json={"channel": 0, "z": 0, "pos": [[1, 2]], "neg": [], "extra": [{"name": "s2", "color": "#fff", "pos": [[3, 4]]}]})
    assert client.get(f"/api/datasets/{did}/annotations").json()["0:0"]["extra"][0]["pos"] == [[3, 4]]


def test_histomorphometry_from_structures(client):
    ds, gt, centers = upload_stack(client, n_z=1, with_reference=False)
    did = ds["id"]
    ys, xs = np.nonzero(~ndi_dilate(gt))
    bone_like = [[int(ys[i]), int(xs[i])] for i in np.linspace(0, len(ys) - 1, 4).astype(int)]
    body = {"channel": 0, "z": 0, "neg": bg_points(gt)[:3], "settings": SETTINGS,
            "structures": [{"name": "cells", "pos": [list(c) for c in centers[:3]]}, {"name": "bone", "pos": bone_like}]}
    assert client.post(f"/api/datasets/{did}/segment_multi", json=body).status_code == 200
    r = client.post(f"/api/datasets/{did}/histomorphometry", json={"z": 0, "bone": {"channel": 0, "source": "structure:1"},
                                                                  "cells": {"channel": 0, "source": "structure:0"}, "settings": SETTINGS})
    assert r.status_code == 200, r.text
    assert r.json()["summary"]["B.Pm_mm"] >= 0 and r.json()["summary"]["cells_total"] >= 1
    r = client.post(f"/api/datasets/{did}/histomorphometry", json={"z": 0, "bone": {"channel": 0, "source": "structure:5"},
                                                                  "cells": {"channel": 0, "source": "structure:0"}, "settings": SETTINGS})
    assert r.status_code == 400


def ndi_dilate(m):
    import scipy.ndimage as ndi
    return ndi.binary_dilation(m, iterations=6)


def test_download_names_are_safe():
    from boneseg.api.context import attachment

    h = attachment('evil"\r\nX-Injected: 1 näme.csv')["Content-Disposition"]
    assert "\n" not in h and "\r" not in h and h.count('"') == 2 and h.endswith('.csv"')
    assert attachment(".csv")["Content-Disposition"] == 'attachment; filename="csv"'


def test_stack_with_two_structures(client):
    ds, gt, centers = upload_stack(client, n_z=3, with_reference=False)
    did = ds["id"]
    ys, xs = np.nonzero(~ndi_dilate(gt))
    other = [[int(ys[i]), int(xs[i])] for i in np.linspace(0, len(ys) - 1, 4).astype(int)]
    body = {"channel": 0, "ref_z": 1, "neg": bg_points(gt)[:3], "settings": SETTINGS,
            "structures": [{"name": "cells", "pos": [list(c) for c in centers[:3]]}, {"name": "matrix", "pos": other}]}
    client.patch(f"/api/datasets/{did}", json={"group": "control"})
    job = client.post(f"/api/datasets/{did}/stack", json=body).json()
    for _ in range(200):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert job["status"] == "done", job
    s = job["result"]["summary"]
    assert set(s["structures"]) == {"cells", "matrix"} and s["structures"]["cells"]["n_objects_3d"] >= 1
    labels = tifffile.imread(io.BytesIO(client.get(f"/api/jobs/{job['id']}/files/labels.tif").content))
    assert labels.shape == (3, *gt.shape) and set(np.unique(labels)) <= {0, 1, 2}
    assert client.get(f"/api/jobs/{job['id']}/files/objects_3d.csv").text.startswith("structure,label")
    assert client.get(f"/api/datasets/{did}/xz", params={"c": 0, "y": 40, "job_id": job["id"]}).status_code == 200
    # A structure named like bone ("matrix") turns on stack-level histomorphometry, which Compare samples can use
    hm = s["histomorphometry"]
    assert hm["bone"] == "matrix" and hm["cells"] == "cells" and hm["Oc.Pm/B.Pm_%"] is not None
    assert client.get(f"/api/jobs/{job['id']}/files/histomorphometry.csv").text.startswith("z,")
    row = [r for r in client.get("/api/study", params={"metric": "Oc.Pm/B.Pm_%"}).json()["rows"] if r["dataset_id"] == did][0]
    assert row["Oc.Pm/B.Pm_%"] == pytest.approx(hm["Oc.Pm/B.Pm_%"])


def test_project_export(client):
    import zipfile

    ds, gt, centers = upload_stack(client, n_z=2)
    did = ds["id"]
    body = {"channel": 0, "z": 0, "pos": [list(c) for c in centers], "neg": bg_points(gt), "settings": SETTINGS}
    client.put(f"/api/datasets/{did}/annotations", json={"channel": 0, "z": 0, "pos": body["pos"], "neg": body["neg"]})
    client.post(f"/api/datasets/{did}/segment", json=body)
    client.post(f"/api/datasets/{did}/labels", json={"channel": 0, "z": 0})
    job = client.post(f"/api/datasets/{did}/stack", json={**body, "ref_z": 0}).json()
    for _ in range(100):
        if client.get(f"/api/jobs/{job['id']}").json()["status"] == "done":
            break
        time.sleep(0.05)
    r = client.get(f"/api/datasets/{did}/export/project.zip")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert {"dataset.json", "annotations.json", "README.txt", "labels/c0_z0.png"} <= set(names)
    assert any(n.endswith("/slices.csv") for n in names) and not any(n.endswith(".tif") for n in names)
    with_masks = zipfile.ZipFile(io.BytesIO(client.get(f"/api/datasets/{did}/export/project.zip", params={"include_masks": True}).content)).namelist()
    assert any(n.endswith("masks.tif") for n in with_masks)


def test_multi_structure_profile(client, tmp_path):
    ds, gt, centers = upload_stack(client, n_z=2, with_reference=False)
    ys, xs = np.nonzero(~ndi_dilate(gt))
    other = [[int(ys[i]), int(xs[i])] for i in np.linspace(0, len(ys) - 1, 4).astype(int)]
    structures = [{"name": "cells", "color": "#ff8800", "pos": [list(c) for c in centers[:3]]}, {"name": "bone matrix", "color": "#00ff88", "pos": other}]
    r = client.post("/api/profiles", json={"name": "Two things", "dataset_id": ds["id"], "channel": 0, "z": 0, "pos": [],
                                           "neg": bg_points(gt)[:3], "structures": structures, "settings": SETTINGS})
    assert r.status_code == 200, r.text
    prof = r.json()
    assert prof["kind"] == "structures" and [s["name"] for s in prof["structures"]] == ["cells", "bone matrix"]
    # Another dataset, no clicks: segment and run the stack with the profile
    ds2, _, _ = upload_stack(client, n_z=2, with_reference=False)
    r = client.post(f"/api/datasets/{ds2['id']}/segment_multi", json={"channel": 0, "z": 1, "profile_id": prof["id"], "settings": SETTINGS})
    assert r.status_code == 200, r.text
    assert [s["name"] for s in r.json()["structures"]] == ["cells", "bone matrix"]
    job = client.post(f"/api/datasets/{ds2['id']}/stack", json={"channel": 0, "profile_id": prof["id"], "settings": SETTINGS}).json()
    for _ in range(100):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert job["status"] == "done" and set(job["result"]["summary"]["structures"]) == {"cells", "bone matrix"}
    assert "histomorphometry" in job["result"]["summary"]
    # Using it as a single-structure profile gives a clear error
    r = client.post(f"/api/datasets/{ds2['id']}/segment", json={"channel": 0, "z": 0, "profile_id": prof["id"], "settings": SETTINGS})
    assert r.status_code == 400 and "several structures" in r.json()["detail"]
    # Batch from the command line
    from boneseg.__main__ import main
    root = client.app.state.store.root
    src = next((root / "datasets" / ds2["id"]).glob("*.tif"))
    main(["batch", str(src), "--profile", prof["id"], "--channel", "0", "--out", str(tmp_path / "o"), "--data-dir", str(root)])
    import pandas as pd
    out = pd.read_csv(tmp_path / "o" / "summary.csv")
    assert out["status"].iloc[0] == "ok" and "cells_volume_um3" in out and "Oc.Pm/B.Pm_%" in out


def test_batch_from_the_app_feeds_compare(client):
    ids = []
    for i in range(3):
        ds, gt, centers = upload_stack(client, n_z=2)
        ids.append(ds["id"])
        client.patch(f"/api/datasets/{ds['id']}", json={"group": "a" if i < 2 else "b"})
    prof = client.post("/api/profiles", json={"name": "cells", "dataset_id": ids[0], "channel": 0, "z": 0, "pos": [list(c) for c in centers],
                                              "neg": bg_points(gt), "settings": SETTINGS}).json()
    job = client.post("/api/batch", json={"profile_id": prof["id"], "channel": 0, "settings": SETTINGS}).json()
    for _ in range(200):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert job["status"] == "done", job
    assert [s["status"] for s in job["result"]["samples"]] == ["done"] * 3
    rows = client.get("/api/study", params={"metric": "mean_area_fraction"}).json()["rows"]
    assert all(r["has_stack_run"] for r in rows if r["dataset_id"] in ids)
    assert client.post("/api/batch", json={"profile_id": prof["id"], "channel": 7, "settings": SETTINGS}).status_code == 400
    # A sample without the channel is skipped, not fatal
    np.save(client.app.state.store.root / "one_channel.npy", np.zeros((40, 40), np.float32))
    client.post("/api/datasets/from-path", json={"path": str(client.app.state.store.root / "one_channel.npy")})
    job = client.post("/api/batch", json={"profile_id": prof["id"], "channel": 1, "settings": SETTINGS}).json()
    for _ in range(200):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert job["status"] == "done" and [s["status"] for s in job["result"]["samples"]].count("skipped") == 1


def test_labels_and_learned_model_with_several_structures(client, tmp_path):
    import base64
    from PIL import Image

    ds, gt, centers = upload_stack(client, n_z=3, with_reference=False)
    did = ds["id"]
    ys, xs = np.nonzero(~ndi_dilate(gt))
    other = [[int(ys[i]), int(xs[i])] for i in np.linspace(0, len(ys) - 1, 4).astype(int)]
    body = {"channel": 0, "z": 0, "neg": bg_points(gt)[:3], "settings": SETTINGS,
            "structures": [{"name": "cells", "pos": [list(c) for c in centers[:3]]}, {"name": "matrix", "pos": other}]}
    assert client.post(f"/api/datasets/{did}/segment_multi", json=body).status_code == 200
    # Label slice 0 from the multi result, and slice 1 from a painted map (red channel = structure index)
    assert client.post(f"/api/datasets/{did}/labels", json={"channel": 0, "z": 0, "structures": ["cells", "matrix"]}).status_code == 200
    lab = np.zeros(gt.shape, np.uint8)
    lab[gt] = 1
    lab[:20, :] = 2
    rgba = np.zeros(gt.shape + (4,), np.uint8)
    rgba[..., 0] = lab
    rgba[..., 3] = np.where(lab > 0, 255, 0)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    assert client.post(f"/api/datasets/{did}/labels", json={"channel": 0, "z": 1, "mask_png": url, "structures": ["cells", "matrix"]}).status_code == 200
    assert client.get(f"/api/datasets/{did}").json()["label_structures"]["0"] == ["cells", "matrix"]
    info = client.post(f"/api/datasets/{did}/head", json={"channel": 0, "settings": SETTINGS}).json()
    assert info["names"] == ["cells", "matrix"]
    r = client.post(f"/api/datasets/{did}/segment_multi", json={"method": "learned", "channel": 0, "z": 2, "settings": SETTINGS})
    assert r.status_code == 200, r.text
    assert [s["name"] for s in r.json()["structures"]] == ["cells", "matrix"]
    r = client.post(f"/api/datasets/{did}/segment", json={"method": "learned", "channel": 0, "z": 2, "settings": SETTINGS})
    assert r.status_code == 400 and "several structures" in r.json()["detail"]
    job = client.post(f"/api/datasets/{did}/stack", json={"method": "learned", "channel": 0, "settings": SETTINGS}).json()
    for _ in range(200):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert job["status"] == "done", job
    assert set(job["result"]["summary"]["structures"]) == {"cells", "matrix"} and "histomorphometry" in job["result"]["summary"]
    # Exported as a profile, it segments other files in batch
    prof = client.post(f"/api/datasets/{did}/head/export", json={"channel": 0, "name": "learned two"}).json()
    from boneseg.__main__ import main
    root = client.app.state.store.root
    main(["batch", str(next((root / "datasets" / did).glob("*.tif"))), "--profile", prof["id"], "--channel", "0",
          "--out", str(tmp_path / "o"), "--data-dir", str(root)])
    import pandas as pd
    assert "cells_volume_um3" in pd.read_csv(tmp_path / "o" / "summary.csv")


def test_settings_are_remembered(client):
    ds, _, _ = upload_stack(client, n_z=1)
    client.patch(f"/api/datasets/{ds['id']}", json={"settings": {"backbone": "classic", "neg_weight": 1.2}})
    c2 = TestClient(create_app(client.app.state.store.root))
    assert c2.get(f"/api/datasets/{ds['id']}").json()["settings"]["neg_weight"] == 1.2


def test_labels_from_reference(client):
    ds, gt, _ = upload_stack(client, n_z=4)
    did = ds["id"]
    assert client.post(f"/api/datasets/{did}/labels/from-reference", json={"channel": 0, "n": 2}).status_code == 400
    client.patch(f"/api/datasets/{did}", json={"reference_channel": 1})
    out = client.post(f"/api/datasets/{did}/labels/from-reference", json={"channel": 0, "n": 2}).json()
    assert out["saved"] == [0, 3] and len(out["labels"]) == 2, out
    assert client.post(f"/api/datasets/{did}/labels/from-reference", json={"channel": 1, "n": 2}).status_code == 400
    info = client.post(f"/api/datasets/{did}/head", json={"channel": 0, "settings": SETTINGS}).json()
    assert info["reference_check"]["mean_dice"] > 0.5


def test_histomorphometry_stack_across_channels(client):
    ds = client.post("/api/datasets/demo").json()
    did = ds["id"]
    import glob
    from pathlib import Path
    stack = tifffile.imread(glob.glob(str(Path(client.app.state.store.root) / "datasets" / did / "*.tif"))[0])
    z = 6
    s = {"backbone": "classic", "vit_size": 252}
    m = stack[z, 0].astype(float)
    hi, lo = np.argwhere(m > np.percentile(m, 85)), np.argwhere(m < np.percentile(m, 15))
    rng = np.random.default_rng(0)
    bone_prof = client.post("/api/profiles", json={"name": "bone", "dataset_id": did, "channel": 0, "z": z,
                                                   "pos": hi[rng.choice(len(hi), 6)].tolist(), "neg": lo[rng.choice(len(lo), 6)].tolist(), "settings": s}).json()
    client.patch(f"/api/datasets/{did}", json={"group": "control"})
    body = {"bone": {"channel": 0, "source": "profile", "profile_id": bone_prof["id"]}, "cells": {"channel": 1, "source": "reference"},
            "z_step": 3, "settings": s}
    assert client.post(f"/api/datasets/{did}/histomorphometry/stack", json={**body, "cells": {"channel": 1, "source": "current"}}).status_code == 400
    job = client.post(f"/api/datasets/{did}/histomorphometry/stack", json=body).json()
    for _ in range(200):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert job["status"] == "done", job
    hm = job["result"]["summary"]["histomorphometry"]
    assert hm["Oc.Pm/B.Pm_%"] is not None and job["result"]["summary"]["n_slices"] == 4
    assert client.get(f"/api/jobs/{job['id']}/files/histomorphometry.csv").text.startswith("z,")
    row = [r for r in client.get("/api/study", params={"metric": "Oc.Pm/B.Pm_%"}).json()["rows"] if r["dataset_id"] == did][0]
    assert row["Oc.Pm/B.Pm_%"] == pytest.approx(hm["Oc.Pm/B.Pm_%"])


def test_plane_with_second_channel(client):
    from PIL import Image

    ds, gt, _ = upload_stack(client, n_z=1)
    grey = Image.open(io.BytesIO(client.get(f"/api/datasets/{ds['id']}/plane", params={"c": 0, "z": 0}).content))
    both = Image.open(io.BytesIO(client.get(f"/api/datasets/{ds['id']}/plane", params={"c": 0, "z": 0, "overlay": 1, "color": "00ff00"}).content))
    assert grey.mode == "L" and both.mode == "RGB"
    arr = np.asarray(both).astype(int)
    assert (arr[..., 1] - arr[..., 0]).max() > 50   # Green where the second channel is bright


def test_access_token(tmp_path):
    c = TestClient(create_app(tmp_path / "d", token="s3cret"))
    assert c.get("/").status_code == 200                          # The page loads, so it can explain what is needed
    assert c.get("/api/health").status_code == 401
    assert c.get("/api/health", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.get("/api/health", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    c.cookies.set("boneseg_token", "s3cret")
    assert c.get("/api/datasets").status_code == 200
    assert TestClient(create_app(tmp_path / "e")).get("/api/health").status_code == 200   # No token, open as before


def test_report_for_several_structures(client):
    ds, gt, centers = upload_stack(client, n_z=2, with_reference=False)
    did = ds["id"]
    ys, xs = np.nonzero(~ndi_dilate(gt))
    other = [[int(ys[i]), int(xs[i])] for i in np.linspace(0, len(ys) - 1, 4).astype(int)]
    body = {"channel": 0, "z": 0, "neg": bg_points(gt)[:3], "settings": SETTINGS,
            "structures": [{"name": "cells", "color": "#ff8800", "pos": [list(c) for c in centers[:3]]}, {"name": "bone matrix", "color": "#00ff88", "pos": other}]}
    client.post(f"/api/datasets/{did}/segment_multi", json=body)
    job = client.post(f"/api/datasets/{did}/stack", json={**body, "ref_z": 0}).json()
    for _ in range(200):
        if client.get(f"/api/jobs/{job['id']}").json()["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    page = client.get(f"/api/datasets/{did}/report", params={"c": 0, "z": 0}).text
    for text in ("<h2>Structures</h2>", "bone matrix", "Several structures were segmented together", "Latest stack run (2 slices)", "Histomorphometry"):
        assert text in page, text
    # A single-structure segmentation afterwards makes the report about that one again
    client.post(f"/api/datasets/{did}/segment", json={"channel": 0, "z": 0, "pos": [list(c) for c in centers], "neg": bg_points(gt), "settings": SETTINGS})
    assert "<h2>Structures</h2>" not in client.get(f"/api/datasets/{did}/report", params={"c": 0, "z": 0}).text


def test_batch_uses_each_samples_own_channel(client):
    a, gt, centers = upload_stack(client, n_z=1)
    b, _, _ = upload_stack(client, n_z=1)
    c, _, _ = upload_stack(client, n_z=1)
    client.patch(f"/api/datasets/{a['id']}", json={"default_channel": 0})
    client.patch(f"/api/datasets/{b['id']}", json={"default_channel": 1})
    assert client.patch(f"/api/datasets/{c['id']}", json={"default_channel": 9}).status_code == 400
    prof = client.post("/api/profiles", json={"name": "p", "dataset_id": a["id"], "channel": 0, "z": 0, "pos": [list(x) for x in centers],
                                              "neg": bg_points(gt), "settings": SETTINGS}).json()
    job = client.post("/api/batch", json={"profile_id": prof["id"], "channel": None, "dataset_ids": [a["id"], b["id"], c["id"]], "settings": SETTINGS}).json()
    for _ in range(200):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert job["meta"]["channels"] == {a["id"]: 0, b["id"]: 1}
    status = {s["dataset_id"]: s["status"] for s in job["result"]["samples"]}
    assert status == {a["id"]: "done", b["id"]: "done", c["id"]: "skipped"}


def test_boundary_settings_through_the_api(client):
    ds, gt, centers = upload_stack(client)
    client.patch(f"/api/datasets/{ds['id']}", json={"reference_channel": 1})
    st = {**SETTINGS, "threshold_position": 0.6, "edge_refine": "guided", "shift_passes": 2}
    body = {"channel": 0, "z": 1, "pos": [list(c) for c in centers[:3]], "neg": bg_points(gt), "settings": st}
    r = client.post(f"/api/datasets/{ds['id']}/segment", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["evaluation"]["dice"] > 0.6
    # A missing refiner file is a clear error, not a crash
    r = client.post(f"/api/datasets/{ds['id']}/segment", json={**body, "settings": {**st, "refiner": "/no/such/refiner.pt"}})
    assert r.status_code == 400 and "Refiner file not found" in r.text


def test_model_files_are_confined_without_path_access(tmp_path):
    from boneseg import backbone
    c = TestClient(create_app(tmp_path / "data", allow_paths=False))
    try:
        with pytest.raises(ValueError, match="models folder"):
            backbone.get_backbone("dinov2_s14@/etc/hosts")
        (tmp_path / "data" / "models" / "dinov2_s14_mine.pt").write_bytes(b"x")
        ids = [b["id"] for b in c.get("/api/health").json()["backbones"]]
        assert any(i.endswith("dinov2_s14_mine.pt") for i in ids)
    finally:
        backbone.MODEL_DIRS = None


def test_colour_images_get_a_colour_channel(client, tmp_path):
    from PIL import Image
    img, gt, centers = make_blobs(seed=0)
    # A brightfield-like colour image: purple objects on a light background
    rgb = np.stack([1 - 0.5 * img, 1 - 0.8 * img, 1 - 0.3 * img], -1)
    buf = io.BytesIO()
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(buf, format="PNG")
    r = client.post("/api/datasets", files={"file": ("cells.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200, r.text
    ds = r.json()
    assert ds["n_channels"] == 4 and ds["rgb_channel"] == 3 and ds["channel_names"][3] == "Colour (RGB)"
    png = client.get(f"/api/datasets/{ds['id']}/plane", params={"c": 3, "z": 0})
    assert png.status_code == 200 and Image.open(io.BytesIO(png.content)).mode == "RGB"
    body = {"channel": 3, "z": 0, "pos": [list(c) for c in centers[:3]], "neg": bg_points(gt), "settings": SETTINGS}
    r = client.post(f"/api/datasets/{ds['id']}/segment", json=body)
    assert r.status_code == 200, r.text
    csv = client.get(f"/api/datasets/{ds['id']}/export/objects.csv", params={"c": 3, "z": 0, "all_channels": True}).text
    assert "Red_mean" in csv and "Colour" not in csv.split("\n")[0]
    r = client.patch(f"/api/datasets/{ds['id']}", json={"reference_channel": 3})
    assert r.status_code == 400 and "colour channel" in r.text
    side = client.get(f"/api/datasets/{ds['id']}/xz", params={"c": 3, "y": 10})
    assert side.status_code == 200


def test_missed_bone_suggestions(client):
    # A blob far from every click and outside the mask should be offered as possibly missed
    ds, gt, centers = upload_stack(client)
    body = {"channel": 0, "z": 1, "pos": [list(centers[0])], "neg": bg_points(gt), "settings": {**SETTINGS, "shift_passes": 1}}
    out = client.post(f"/api/datasets/{ds['id']}/segment", json=body).json()
    assert "missed" in out and isinstance(out["missed"], list)
    for m in out["missed"]:
        assert len(m["point"]) == 2 and m["area_px"] > 0
