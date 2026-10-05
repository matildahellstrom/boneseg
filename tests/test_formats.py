"""Every kind of file a user might drop on the app goes through upload, segmentation, a stack run and export."""
import io
import time

import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient
from PIL import Image

from boneseg.api import create_app
from tests.conftest import make_blobs

SETTINGS = {"backbone": "classic", "vit_size": 252}


def _png(arr, mode=None):
    buf = io.BytesIO()
    Image.fromarray(arr, mode).save(buf, format="PNG")
    return buf.getvalue()


def _tif(arr, **kw):
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr, **kw)
    return buf.getvalue()


def _npy(arr):
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


img, gt, centers = make_blobs(h=180, w=220, seed=4)
u8 = (img * 255).astype(np.uint8)
u16 = (img * 60000).astype(np.uint16)

CASES = {
    "grey PNG": ("a.png", lambda: _png(u8), (1, 1)),
    # Colour images get a fourth, virtual "Colour (RGB)" channel
    "RGB PNG": ("a.png", lambda: _png(np.stack([u8, u8 // 2, u8 // 3], -1)), (4, 1)),
    "RGBA PNG": ("a.png", lambda: _png(np.stack([u8, u8, u8, np.full_like(u8, 255)], -1), "RGBA"), (4, 1)),
    "JPEG": ("a.jpg", lambda: (lambda b: (Image.fromarray(u8).save(b, format="JPEG"), b.getvalue())[1])(io.BytesIO()), (1, 1)),
    "16-bit 2D TIFF": ("a.tif", lambda: _tif(u16), (1, 1)),
    "3D TIFF, z only": ("a.tif", lambda: _tif(np.stack([u16] * 4), photometric="minisblack", metadata={"axes": "ZYX"}), (1, 4)),
    "ImageJ ZCYX TIFF": ("a.tif", lambda: _tif(np.stack([np.stack([u16, u16])] * 3), imagej=True, metadata={"axes": "ZCYX"}), (2, 3)),
    "time series TIFF": ("a.tif", lambda: _tif(np.stack([np.stack([u16] * 3)] * 2), imagej=True, metadata={"axes": "TZYX"}), (1, 3)),
    "float32 TIFF": ("a.tif", lambda: _tif(img.astype(np.float32)), (1, 1)),
    "2D npy": ("a.npy", lambda: _npy(img), (1, 1)),
    "3D npy": ("a.npy", lambda: _npy(np.stack([img] * 3)), (1, 3)),
    "4D npy": ("a.npy", lambda: _npy(np.stack([np.stack([img] * 2)] * 2)), (2, 2)),
}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    return TestClient(create_app(tmp_path_factory.mktemp("formats")))


@pytest.mark.parametrize("name", list(CASES))
def test_format_end_to_end(client, name):
    fname, make, (n_c, n_z) = CASES[name]
    r = client.post("/api/datasets/stream", params={"filename": fname}, content=make())
    assert r.status_code == 200, r.text
    ds = r.json()
    assert (ds["n_channels"], ds["n_z"], ds["height"], ds["width"]) == (n_c, n_z, *gt.shape), ds
    did = ds["id"]
    assert client.get(f"/api/datasets/{did}/plane", params={"c": n_c - 1, "z": n_z - 1}).status_code == 200
    ys, xs = np.nonzero(~gt)
    neg = [[int(ys[i]), int(xs[i])] for i in np.linspace(0, len(ys) - 1, 6).astype(int)]
    body = {"channel": 0, "z": 0, "pos": [list(c) for c in centers[:3]], "neg": neg, "settings": SETTINGS, "uncertainty": True}
    r = client.post(f"/api/datasets/{did}/segment", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["stats"]["n_objects"] >= 1
    for fmt in ("png", "tif"):
        assert client.get(f"/api/datasets/{did}/export/mask", params={"c": 0, "z": 0, "fmt": fmt}).status_code == 200
    assert client.get(f"/api/datasets/{did}/export/objects.csv", params={"c": 0, "z": 0}).status_code == 200
    assert client.get(f"/api/datasets/{did}/report", params={"c": 0, "z": 0}).status_code == 200
    if n_z > 1:
        job = client.post(f"/api/datasets/{did}/stack", json={**body, "ref_z": 0}).json()
        for _ in range(200):
            job = client.get(f"/api/jobs/{job['id']}").json()
            if job["status"] in ("done", "failed"):
                break
            time.sleep(0.05)
        assert job["status"] == "done", job
        assert client.get(f"/api/datasets/{did}/xz", params={"c": 0, "y": 50}).status_code == 200


def test_unreadable_files_give_clear_errors(client):
    r = client.post("/api/datasets/stream", params={"filename": "broken.tif"}, content=b"not a tiff at all")
    assert r.status_code == 400 and "Could not read broken.tif" in r.json()["detail"]
    r = client.post("/api/datasets/stream", params={"filename": "broken.ims"}, content=b"not hdf5")
    assert r.status_code == 400 and "Could not read broken.ims" in r.json()["detail"]
    r = client.post("/api/datasets/stream", params={"filename": "five_d.npy"}, content=_npy(np.zeros((2, 2, 2, 2, 2))))
    assert r.status_code == 400 and "shape" in r.json()["detail"]


def test_colour_volume_planes(tmp_path):
    from PIL import Image
    from boneseg import io as bio
    a = (np.random.default_rng(0).random((40, 50, 3)) * 255).astype(np.uint8)
    Image.fromarray(a).save(tmp_path / "c.png")
    v = bio.load_volume(tmp_path / "c.png")
    assert v.n_channels == 4 and v.rgb_channel == 3
    assert v.get_plane(3, 0).shape == (40, 50, 3) and np.array_equal(v.get_plane(3, 0)[..., 1], a[..., 1])
    assert v.get_xz(3, 5).shape == (1, 50)
    g = (np.random.default_rng(0).random((40, 50)) * 255).astype(np.uint8)
    Image.fromarray(g).save(tmp_path / "g.png")
    assert bio.load_volume(tmp_path / "g.png").rgb_channel is None
