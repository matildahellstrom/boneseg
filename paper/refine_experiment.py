"""Development-slice experiment for the boundary improvements (never touches test slices).

Sweeps, for boneseg from clicks:
  * threshold_position: where between background and object clicks the threshold sits (0.5 = method-v1)
  * edge_refine: none, or a guided filter on the image with several eps values
  * shift_passes: 1 (method-v1) or n x n sub-patch shifts
and reports Dice and the bias of B.Ar/T.Ar and Tb.Th against the expert masks.
Writes paper/results/refine_dev.csv. Usage: python paper/refine_experiment.py [--samples A E F]
"""
from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from common import available_samples, bone_measures, load_sample, simulated_clicks
from boneseg import metrics
from boneseg.backbone import get_backbone
from boneseg.segment import (SegmentationSettings, calibrate_threshold, embed_image, guided_filter, normalize_scores,
                             prototypes, sample_points, score_grid, upsample)

OUT = Path(__file__).resolve().parent / "results"
BUDGETS = [(3, 6), (25, 25)]
POSITIONS = [0.5, 0.6, 0.7, 0.8]
EDGES = [("none", None), ("guided", 1e-3), ("guided", 1e-2), ("guided", 1e-1)]
SHIFTS = [1, 2, 3]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--neg-weight", type=float, default=0.8)
    args = ap.parse_args(argv)
    names = args.samples or available_samples()
    t0 = time.time()
    base = SegmentationSettings(backbone="dinov2_s14", neg_weight=args.neg_weight)
    bb = get_backbone(base.backbone)
    rows = []
    for n in names:
        s = load_sample(n)
        print(f"[{time.time() - t0:5.0f}s] {n}: {len(s.dev)} development slices", flush=True)
        for sl in s.dev:
            gtm = bone_measures(sl.gt, sl.pixel_um)
            for shift in SHIFTS:
                emb = embed_image(bb, sl.img, replace(base, shift_passes=shift))
                cell = max(emb.height / emb.grid.shape[0], emb.width / emb.grid.shape[1])
                for budget, seed in itertools.product(BUDGETS, range(args.seeds)):
                    pos, neg = simulated_clicks(sl.gt, *budget, seed)
                    score = normalize_scores(score_grid(emb.grid, prototypes(emb, pos), prototypes(emb, neg), base.neg_weight), base.score_norm)
                    raw = upsample(score, (emb.height, emb.width))
                    for edge, eps in EDGES:
                        m = raw if edge == "none" else guided_filter(sl.img, raw, max(1, int(round(cell))), eps)
                        ps, ns = sample_points(m, pos), sample_points(m, neg)
                        for position in POSITIONS:
                            pred = m >= calibrate_threshold(ps, ns, position)
                            pm = bone_measures(pred, sl.pixel_um)
                            rows.append({"sample": n, "z": sl.z, "budget": f"{budget[0]}+{budget[1]}", "seed": seed, "shift_passes": shift,
                                         "edge": edge if eps is None else f"{edge} eps={eps:g}", "position": position,
                                         "dice": metrics.dice(pred, sl.gt),
                                         "bar_bias_pp": pm["B.Ar/T.Ar_%"] - gtm["B.Ar/T.Ar_%"],
                                         "tbth_bias_um": pm["Tb.Th_um"] - gtm["Tb.Th_um"]})
        pd.DataFrame(rows).to_csv(OUT / "refine_dev.csv", index=False)
    df = pd.DataFrame(rows)
    # Mean over seeds per slice, then per sample, then over samples
    per = df.groupby(["budget", "shift_passes", "edge", "position", "sample"])[["dice", "bar_bias_pp", "tbth_bias_um"]].mean()
    summ = per.groupby(["budget", "shift_passes", "edge", "position"]).mean().round(3)
    for b in summ.index.get_level_values(0).unique():
        print(f"\n== {b} clicks, development slices of {', '.join(names)} ==")
        print(summ.loc[b].sort_values("dice", ascending=False).head(15).to_string())
        print("method-v1:", summ.loc[(b, 1, "none", 0.5)].to_dict())
    print(f"\n{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
