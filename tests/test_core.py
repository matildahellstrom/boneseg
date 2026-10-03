import numpy as np
import pytest
import tifffile

from boneseg import io, metrics, quantify, segment
from boneseg.backbone import get_backbone
from boneseg.segment import Profile, SegmentationSettings
from tests.conftest import make_blobs


def classic_settings(**kw):
    return SegmentationSettings(backbone="classic", vit_size=252, **kw)


def background_points(gt, n=6, seed=1):
    rng = np.random.default_rng(seed)
    ys, xs = np.nonzero(~gt)
    idx = rng.choice(len(ys), n, replace=False)
    return [(int(ys[i]), int(xs[i])) for i in idx]


def test_segment_blobs_classic(blobs):
    img, gt, centers = blobs
    s = classic_settings()
    emb = segment.embed_image(get_backbone("classic"), img, s)
    res = segment.segment(emb, centers[:3], background_points(gt), s)
    assert res.heat.shape == img.shape and res.mask.shape == img.shape
    assert 0 <= res.heat.min() and res.heat.max() <= 1
    assert metrics.dice(res.mask, gt) > 0.6


def test_threshold_modes(blobs):
    img, gt, _ = blobs
    heat = segment.rescale01(img)
    m, _ = segment.threshold_heatmap(heat, classic_settings(threshold_mode="top_percent", top_percent=20))
    assert abs(m.mean() - 0.2) < 0.02
    m, thr = segment.threshold_heatmap(heat, classic_settings(threshold_mode="manual", manual_threshold=0.7))
    assert thr == 0.7 and m.mean() < 0.5
    with pytest.raises(ValueError):
        segment.threshold_heatmap(heat, classic_settings(threshold_mode="nope"))


def test_needs_positive_point(blobs):
    img, gt, _ = blobs
    s = classic_settings()
    emb = segment.embed_image(get_backbone("classic"), img, s)
    with pytest.raises(ValueError):
        segment.segment(emb, [], background_points(gt), s)


def test_postprocess_removes_small_objects():
    mask = np.zeros((100, 100), bool)
    mask[10:40, 10:40] = True
    mask[70:72, 70:72] = True
    s = classic_settings(min_object_um2=10)
    out = segment.postprocess(mask, s, (1.0, 1.0))
    assert out[20, 20] and not out[71, 71]


def test_uncertainty_and_suggestion(blobs):
    img, gt, centers = blobs
    s = classic_settings()
    emb = segment.embed_image(get_backbone("classic"), img, s)
    pos, neg = segment.prototypes(emb, centers[:2]), segment.prototypes(emb, background_points(gt, 3))
    u = segment.uncertainty_map(emb, pos, neg, s, n_boot=8)
    assert u.shape == img.shape and u.min() >= 0 and u.max() <= 0.5
    sug = segment.suggest_click(u + 0.01, centers[:2])
    assert sug is not None and all((sug[0] - y) ** 2 + (sug[1] - x) ** 2 > 1 for y, x in centers[:2])


def test_profile_roundtrip_and_transfer(tmp_path):
    img_a, gt_a, centers_a = make_blobs(seed=0)
    img_b, gt_b, _ = make_blobs(seed=5)
    s = classic_settings()
    bb = get_backbone("classic")
    emb_a = segment.embed_image(bb, img_a, s)
    prof = Profile(name="cells", backbone="classic", layer_from_end=1,
                   pos=segment.prototypes(emb_a, centers_a).numpy(), neg=segment.prototypes(emb_a, background_points(gt_a)).numpy(),
                   settings=s.to_dict())
    prof.save(tmp_path / "p.npz")
    loaded = Profile.load(tmp_path / "p.npz")
    assert loaded.name == "cells" and loaded.pos.shape == prof.pos.shape
    pos, neg = loaded.tensors()
    res = segment.segment_with_prototypes(segment.embed_image(bb, img_b, s), pos, neg, s)
    assert metrics.dice(res.mask, gt_b) > 0.5


def test_metrics_edge_cases():
    z = np.zeros((10, 10), bool)
    assert metrics.dice(z, z) == 1.0 and metrics.iou(z, z) == 1.0
    assert np.isnan(metrics.hd95(z, ~z))
    a = z.copy(); a[2:5, 2:5] = True
    assert metrics.hd95(a, a, (0.5, 0.5)) == 0.0


def test_quantify(blobs):
    img, gt, _ = blobs
    summary = quantify.summarize_mask(gt, img, (0.5, 0.5))
    assert summary["n_objects"] >= 1
    assert summary["area_um2"] == pytest.approx(gt.sum() * 0.25)
    assert summary["contrast_ratio"] > 1.5
    table = quantify.object_table(gt, img, (0.5, 0.5))
    assert table["area_um2"].sum() == pytest.approx(summary["area_um2"])
    empty = quantify.summarize_mask(np.zeros_like(gt), img)
    assert empty["n_objects"] == 0


def test_load_formats(tmp_path, blobs):
    img, gt, _ = blobs
    stack = np.stack([np.stack([img, img * 0.5]), np.stack([gt.astype(np.float32)] * 2)])  # C, Z, Y, X
    tifffile.imwrite(tmp_path / "s.tif", stack.transpose(1, 0, 2, 3).astype(np.float32), imagej=True, metadata={"axes": "ZCYX", "spacing": 2.0, "unit": "um"},
                     resolution=(1 / 0.5, 1 / 0.5))
    v = io.load_volume(tmp_path / "s.tif")
    assert (v.n_channels, v.n_z, v.height, v.width) == (2, 2, *img.shape)
    assert v.voxel_um == pytest.approx((2.0, 0.5, 0.5)) and v.voxel_size_known
    assert np.allclose(v.get_plane(0, 1), img * 0.5)
    np.save(tmp_path / "a.npy", img)
    v = io.load_volume(tmp_path / "a.npy")
    assert (v.n_channels, v.n_z) == (1, 1)
    from PIL import Image
    Image.fromarray((img * 255).astype(np.uint8)).save(tmp_path / "a.png")
    assert io.load_volume(tmp_path / "a.png").n_channels == 1
    with pytest.raises(ValueError):
        io.load_volume(tmp_path / "a.bmp")


def test_vit_input_size_keeps_aspect():
    h, w = segment.vit_input_size(3000, 1500, 980, 14)
    assert h == 980 and w % 14 == 0 and abs(w - 490) <= 14


def test_fill_holes_only_fills_small_interior_holes():
    mask = np.zeros((60, 60), bool)
    mask[10:50, 10:50] = True
    mask[20:23, 20:23] = False      # Small interior hole, 9 px
    mask[30:45, 30:45] = False      # Big interior hole, 225 px
    out = segment.postprocess(mask, classic_settings(fill_holes_um2=20), (1.0, 1.0))
    assert out[21, 21] and not out[37, 37] and not out[0, 0]
