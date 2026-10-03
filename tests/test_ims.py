"""The Imaris reader path, on small files written with the same layout as real Imaris exports."""
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from boneseg import io as bio
from boneseg.api import create_app
from boneseg.store import Store
from tests.conftest import make_blobs
from tests.ims_writer import write_ims


@pytest.fixture
def ims_file(tmp_path):
    img, gt, centers = make_blobs(h=150, w=190, seed=6)
    zs = 5
    arr = np.zeros((3, zs, *img.shape), np.uint16)
    for z in range(zs):
        arr[0, z] = (img * 3000 * (1 - 0.05 * z)).astype(np.uint16)
        arr[1, z] = (np.random.default_rng(z).random(img.shape) * 500).astype(np.uint16)
        arr[2, z] = gt.astype(np.uint16) * 200
    path = tmp_path / "sample.ims"
    write_ims(path, arr, voxel_um=(2.0, 0.65, 0.65), names=["TRAP", "SOST", "Masked Channel 1"])
    return path, arr, gt, centers


def test_load_ims(ims_file):
    path, arr, gt, _ = ims_file
    v = bio.load_volume(path)
    assert (v.n_channels, v.n_z, v.height, v.width) == (3, 5, *gt.shape)
    assert v.voxel_um == pytest.approx((2.0, 0.65, 0.65)) and v.voxel_size_known
    assert v.channel_names == ["TRAP", "SOST", "Masked Channel 1"]
    assert np.array_equal(v.get_plane(0, 3), arr[0, 3])
    assert np.array_equal(v.get_plane(np.int64(2), np.int64(4)), arr[2, 4])  # NumPy indices
    assert np.array_equal(v.get_xz(0, 40), arr[0, :, 40, :])
    assert Store._guess_reference(v) == {"reference_channel": 2, "reference_guessed": True}


def test_ims_through_the_app(ims_file, tmp_path):
    path, arr, gt, centers = ims_file
    client = TestClient(create_app(tmp_path / "data"))
    ds = client.post("/api/datasets/from-path", json={"path": str(path)}).json()
    assert ds["reference_channel"] == 2 and ds["channel_names"][0] == "TRAP"
    ys, xs = np.nonzero(~gt)
    neg = [[int(ys[i]), int(xs[i])] for i in np.linspace(0, len(ys) - 1, 6).astype(int)]
    body = {"channel": 0, "z": 2, "pos": [list(c) for c in centers[:3]], "neg": neg, "settings": {"backbone": "classic", "vit_size": 252}}
    r = client.post(f"/api/datasets/{ds['id']}/segment", json=body).json()
    assert r["evaluation"]["dice"] > 0.5 and r["stats"]["area_um2"] > 0
    job = client.post(f"/api/datasets/{ds['id']}/stack", json={**body, "ref_z": 2}).json()
    for _ in range(200):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert job["status"] == "done" and job["result"]["summary"]["mean_dice_vs_reference"] > 0.5
    csv = client.get(f"/api/datasets/{ds['id']}/export/objects.csv", params={"c": 0, "z": 2}).text
    assert "ch0_TRAP_mean" in csv.split("\n")[0] and "ch1_SOST_mean" in csv.split("\n")[0]
