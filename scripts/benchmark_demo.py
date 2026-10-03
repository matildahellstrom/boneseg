"""Benchmarks stack propagation on the synthetic demo stack against its expert masks.

Usage: python scripts/benchmark_demo.py [--backbone dinov2_s14] [--seeds 3]
Clicks are simulated: object clicks at cell centres and background clicks at random
background pixels, all on the middle slice.
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
from boneseg import io as bio  # noqa: E402
from boneseg.backbone import get_backbone  # noqa: E402
from boneseg.demo import write_demo_tiff  # noqa: E402
from boneseg.pipeline import StackRequest, run_stack  # noqa: E402
from boneseg.segment import SegmentationSettings, embed_image, prototypes, segment_with_prototypes  # noqa: E402


def simulated_clicks(gt: np.ndarray, n_pos: int, n_neg: int, seed: int):
    rng = np.random.default_rng(seed)
    lab, n = ndi.label(gt)
    cents = ndi.center_of_mass(gt, lab, range(1, n + 1))
    order = rng.permutation(n)[:n_pos]
    pos = [tuple(int(v) for v in cents[i]) for i in order]
    ys, xs = np.nonzero(~ndi.binary_dilation(gt, iterations=8))
    idx = rng.choice(len(ys), n_neg, replace=False)
    return pos, [(int(ys[i]), int(xs[i])) for i in idx]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dinov2_s14")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-pos", type=int, default=3)
    ap.add_argument("--n-neg", type=int, default=6)
    ap.add_argument("--depth-degradation", type=float, default=0.0, help="0 to 1, how much deeper slices degrade")
    ap.add_argument("--ref", choices=["middle", "top"], default="middle", help="Which slice gets the clicks")
    ap.add_argument("--modes", default="clicks,otsu,top_percent")
    ap.add_argument("--score-norm", default="robust")
    args = ap.parse_args(argv)

    tmp = Path(tempfile.mkdtemp())
    vol = bio.load_volume(write_demo_tiff(tmp / "demo.tif", depth_degradation=args.depth_degradation))
    c, ref_c = 1, 2
    planes = {z: bio.normalize_plane(vol.get_plane(c, z)) for z in range(vol.n_z)}
    refs = {z: vol.get_plane(ref_c, z) > 0 for z in range(vol.n_z)}
    ref_z = vol.n_z // 2 if args.ref == "middle" else int(np.argmax([refs[z].sum() for z in range(3)]))
    rows = []
    for mode in args.modes.split(","):
        s = SegmentationSettings(backbone=args.backbone, threshold_mode=mode, top_percent=6, score_norm=args.score_norm)
        bb = get_backbone(args.backbone)
        embs = {z: embed_image(bb, planes[z], s) for z in planes}
        for seed in range(args.seeds):
            pos_pts, neg_pts = simulated_clicks(refs[ref_z], args.n_pos, args.n_neg, seed)
            pos, neg = prototypes(embs[ref_z], pos_pts), prototypes(embs[ref_z], neg_pts)
            ref_res = segment_with_prototypes(embs[ref_z], pos, neg, s, pos_points=pos_pts, neg_points=neg_pts)
            out = run_stack(StackRequest(z_list=list(planes), ref_z=ref_z), pos, neg, s, ref_res.raw_threshold,
                            get_embedding=embs.__getitem__, get_image=planes.__getitem__, get_reference=refs.__getitem__,
                            voxel_um=vol.voxel_um, out_dir=tmp)
            df = pd.DataFrame(out["slices"])
            rows.append({"threshold": mode, "seed": seed, "dice_ref_slice": df.loc[df.z == ref_z, "dice"].item(),
                         "dice_mean": df["dice"].mean(), "dice_far": df.loc[(df.z - ref_z).abs() >= 4, "dice"].mean(),
                         "area_fraction": df["area_fraction"].mean()})
    res = pd.DataFrame(rows).groupby(["threshold"]).mean(numeric_only=True).drop(columns="seed")
    print(f"Backbone {args.backbone}, {args.n_pos} object and {args.n_neg} background clicks, {args.seeds} seeds")
    print(res.round(3).to_string())
    return res


if __name__ == "__main__":
    main()
