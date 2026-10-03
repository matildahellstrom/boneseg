"""Benchmarks threshold rules and stack propagation on a real file with an expert mask channel.

Usage:
  python scripts/benchmark_file.py data/liudata/FILE.ims --channel 3 --reference 4 --slices 12

Clicks are simulated from the expert mask: object clicks at random pixels well inside the mask,
background clicks well outside it. Two protocols are reported:
  per-slice  clicks on every slice, scored on the same slice
  stack      clicks on the middle slice only, carried to the other slices
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from boneseg import io as bio, metrics  # noqa: E402
from boneseg.backbone import get_backbone  # noqa: E402
from boneseg.pipeline import StackRequest, run_stack  # noqa: E402
from boneseg.segment import SegmentationSettings, embed_image, prototypes, segment_with_prototypes  # noqa: E402


def clicks(gt, n_pos, n_neg, seed, margin=10):
    rng = np.random.default_rng(seed)
    inner = ndi.binary_erosion(gt, iterations=margin)
    inner = inner if inner.any() else gt
    outer = ~ndi.binary_dilation(gt, iterations=margin)
    ys, xs = np.nonzero(inner)
    i = rng.choice(len(ys), n_pos, replace=len(ys) < n_pos)
    pos = list(zip(ys[i].tolist(), xs[i].tolist()))
    ys, xs = np.nonzero(outer)
    i = rng.choice(len(ys), n_neg, replace=len(ys) < n_neg)
    return pos, list(zip(ys[i].tolist(), xs[i].tolist()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--channel", type=int, required=True)
    ap.add_argument("--reference", type=int, required=True)
    ap.add_argument("--slices", type=int, default=12)
    ap.add_argument("--z-margin", type=int, default=40)
    ap.add_argument("--backbone", default="dinov2_s14")
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args(argv)

    vol = bio.load_volume(args.path)
    zs = np.linspace(args.z_margin, vol.n_z - 1 - args.z_margin, args.slices).round().astype(int).tolist()
    planes = {z: bio.normalize_plane(vol.get_plane(args.channel, z)) for z in zs}
    refs = {z: vol.get_plane(args.reference, z) > 0 for z in zs}
    zs = [z for z in zs if refs[z].any()]
    bb = get_backbone(args.backbone)
    base = SegmentationSettings(backbone=args.backbone)
    embs = {z: embed_image(bb, planes[z], base) for z in zs}
    print(f"{Path(args.path).name}: channel {args.channel}, reference {args.reference}, {len(zs)} slices, "
          f"pixel {vol.pixel_um[0]:.3f} um, mean foreground {np.mean([refs[z].mean() for z in zs]):.3f}")

    rows = []
    for n_pos, n_neg in ((3, 6), (25, 25)):
        for mode in ("clicks", "otsu", "top_percent"):
            s = SegmentationSettings(backbone=args.backbone, threshold_mode=mode, top_percent=10)
            for seed in range(args.seeds):
                for z in zs:
                    pp, nn = clicks(refs[z], n_pos, n_neg, seed + 100 * z)
                    res = segment_with_prototypes(embs[z], prototypes(embs[z], pp), prototypes(embs[z], nn), s, vol.pixel_um,
                                                  pos_points=pp, neg_points=nn)
                    rows.append({"protocol": "per-slice", "clicks": f"{n_pos}+{n_neg}", "threshold": mode, "seed": seed, "z": z,
                                 "dice": metrics.dice(res.mask, refs[z])})
                ref_z = zs[len(zs) // 2]
                pp, nn = clicks(refs[ref_z], n_pos, n_neg, seed)
                pos, neg = prototypes(embs[ref_z], pp), prototypes(embs[ref_z], nn)
                ref_res = segment_with_prototypes(embs[ref_z], pos, neg, s, vol.pixel_um, pos_points=pp, neg_points=nn)
                out = run_stack(StackRequest(z_list=zs, ref_z=ref_z), pos, neg, s, ref_res.raw_threshold,
                                get_embedding=embs.__getitem__, get_image=planes.__getitem__, get_reference=refs.__getitem__,
                                voxel_um=vol.voxel_um, out_dir=Path(tempfile.mkdtemp()))
                for r in out["slices"]:
                    rows.append({"protocol": "stack", "clicks": f"{n_pos}+{n_neg}", "threshold": mode, "seed": seed, "z": r["z"], "dice": r["dice"]})
    df = pd.DataFrame(rows)
    table = df.groupby(["protocol", "clicks", "threshold"])["dice"].mean().unstack("threshold").round(3)
    print(table.to_string())
    return table


if __name__ == "__main__":
    main()
