"""Development-only check of the learned refiner, leave-one-sample-out on development slices.

For each sample, a refiner is trained on the development slices of the other samples (several simulated click
draws per slice) and scored on this sample's development slices, against the plain click threshold with the same
settings. Test slices are never touched. Writes paper/results/refiner_dev.csv.
Usage: python paper/refiner_experiment.py [--samples A E F] [--shift-passes 2] [--edge none] [--steps 1500]
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
from boneseg.refine import train_refiner
from boneseg.segment import SegmentationSettings, embed_image, segment

OUT = Path(__file__).resolve().parent / "results"
TRAIN_BUDGETS = [(3, 6), (10, 10), (25, 25)]
EVAL_BUDGETS = [(3, 6), (25, 25)]


def raw_and_threshold(emb, pos, neg, st):
    """The (edge-refined) raw score map and the click-calibrated threshold, without the refiner."""
    r = segment(emb, pos, neg, replace(st, refiner=""))
    lo, hi = r.raw_score_range
    return lo + r.heat.astype(np.float64) * (hi - lo), r.raw_threshold, r.mask


def examples_for(slices, embs, st, seeds):
    for sl in slices:
        for budget, seed in itertools.product(TRAIN_BUDGETS, seeds):
            pos, neg = simulated_clicks(sl.gt, *budget, 1000 + seed, noisy=seed % 2 == 1)
            raw, thr, _ = raw_and_threshold(embs[(sl.sample, sl.z)], pos, neg, st)
            yield sl.img, raw, thr, sl.gt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="*", default=None)
    ap.add_argument("--shift-passes", type=int, default=1)
    ap.add_argument("--edge", default="none")
    ap.add_argument("--guided-eps", type=float, default=0.01)
    ap.add_argument("--position", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    names = args.samples or available_samples()
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)  # noqa: E731
    st = SegmentationSettings(backbone="dinov2_s14", neg_weight=0.8, shift_passes=args.shift_passes, edge_refine=args.edge,
                              guided_eps=args.guided_eps, threshold_position=args.position, refiner="x")  # Keeps the image
    bb = get_backbone(st.backbone)
    samples = {n: load_sample(n) for n in names}
    embs = {(sl.sample, sl.z): embed_image(bb, sl.img, st) for s in samples.values() for sl in s.dev}
    log(f"embedded {len(embs)} development slices")
    rows = []
    for held in names:
        train = [sl for n in names if n != held for sl in samples[n].dev]
        ref = train_refiner(examples_for(train, embs, st, range(3)), steps=args.steps, log=log)
        log(f"held-out {held}: refiner trained on {len(train)} slices")
        for sl, budget, seed in itertools.product(samples[held].dev, EVAL_BUDGETS, range(2)):
            pos, neg = simulated_clicks(sl.gt, *budget, seed)
            raw, thr, plain = raw_and_threshold(embs[(sl.sample, sl.z)], pos, neg, st)
            refined = ref.predict(sl.img, raw, thr) >= 0.5
            gtm = bone_measures(sl.gt, sl.pixel_um)
            for method, pred in (("threshold", plain), ("refiner", refined)):
                pm = bone_measures(pred, sl.pixel_um)
                rows.append({"sample": held, "z": sl.z, "budget": f"{budget[0]}+{budget[1]}", "seed": seed, "method": method,
                             "dice": metrics.dice(pred, sl.gt), "bar_bias_pp": pm["B.Ar/T.Ar_%"] - gtm["B.Ar/T.Ar_%"],
                             "tbth_bias_um": pm["Tb.Th_um"] - gtm["Tb.Th_um"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"refiner_dev{args.tag}.csv", index=False)
    per = df.groupby(["budget", "method", "sample"])[["dice", "bar_bias_pp", "tbth_bias_um"]].mean()
    print(per.round(3).to_string())
    print(per.groupby(["budget", "method"]).mean().round(3).to_string())


if __name__ == "__main__":
    main()
