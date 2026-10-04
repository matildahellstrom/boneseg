"""Nested leave-one-sample-out evaluation of boneseg against baselines.

For each held-out sample:
  * every tunable setting is chosen on the development slices of the OTHER samples only
    (boneseg: background weight and backbone input size; random forest: largest filter scale);
  * interactive methods are scored on the held-out sample's test slices with simulated clicks
    (3 object + 6 background, and 25 + 25), clean and noisy, three click seeds each;
  * label-based methods are trained either on 5 labelled development slices of the held-out sample
    ("same sample") or on 5 labelled development slices of each other sample ("other samples").
Test slices are never used for tuning or training.

Writes paper/results/slices.csv (one row per test slice, method and condition) and tuning.csv.
Usage: python paper/evaluate.py [--samples A E F] [--quick]
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import methods as M
from common import ROOT, available_samples, bone_measures, load_sample, simulated_clicks
from boneseg import metrics
from boneseg.segment import SegmentationSettings

OUT = Path(__file__).resolve().parent / "results"
BUDGETS = [(3, 6), (25, 25)]
DINO_GRID = {"neg_weight": [0.4, 0.8, 1.2], "vit_size": [644, 980]}
RF_GRID = {"sigma_max": [8.0, 16.0]}


def score(pred, sl) -> dict:
    out = {"dice": metrics.dice(pred, sl.gt), "iou": metrics.iou(pred, sl.gt), "hd95_um": metrics.hd95(pred, sl.gt, sl.pixel_um, 2048)}
    out.update({f"pred_{k}": v for k, v in bone_measures(pred, sl.pixel_um).items()})
    return out


def tune(inner, budget, seeds, log):
    """Best boneseg and random-forest settings on the development slices of the inner samples, clean clicks."""
    base = SegmentationSettings(backbone="dinov2_s14")
    best = {}
    rows = []
    for nw, vs in itertools.product(DINO_GRID["neg_weight"], DINO_GRID["vit_size"]):
        st = replace(base, neg_weight=nw, vit_size=vs)
        d = [metrics.dice(M.dino_clicks((sl.sample, sl.z), sl.img, *simulated_clicks(sl.gt, *budget, seed), st, sl.pixel_um), sl.gt)
             for s in inner for sl in s.dev for seed in seeds]
        rows.append({"method": "dino_clicks", "setting": f"neg_weight={nw}, vit_size={vs}", "dice": float(np.mean(d))})
        if "dino" not in best or np.mean(d) > best["dino"][0]:
            best["dino"] = (float(np.mean(d)), st)
    for sm in RF_GRID["sigma_max"]:
        d = [metrics.dice(M.rf_clicks((sl.sample, sl.z), sl.img, *simulated_clicks(sl.gt, *budget, seed), sigma_max=sm, seed=seed), sl.gt)
             for s in inner for sl in s.dev for seed in seeds]
        rows.append({"method": "rf_clicks", "setting": f"sigma_max={sm}", "dice": float(np.mean(d))})
        if "rf" not in best or np.mean(d) > best["rf"][0]:
            best["rf"] = (float(np.mean(d)), sm)
    log(f"  tuned for {budget}: boneseg neg_weight={best['dino'][1].neg_weight}, vit_size={best['dino'][1].vit_size}; rf sigma_max={best['rf'][1]}")
    return best["dino"][1], best["rf"][1], rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="*", default=None)
    ap.add_argument("--quick", action="store_true", help="Fewer slices and seeds, for a smoke test")
    args = ap.parse_args(argv)
    names = args.samples or available_samples()
    seeds = [0] if args.quick else [0, 1, 2]
    n_slices = 8 if args.quick else 20
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
    samples = {n: load_sample(n, n_slices) for n in names}
    for s in samples.values():
        log(f"{s.name}: channel {s.channel}, {len(s.dev)} development and {len(s.test)} test slices")
    rows, tuning_rows = [], []
    for held in names:
        log(f"held-out sample {held}")
        tgt = samples[held]
        inner = [samples[n] for n in names if n != held]
        gt_measures = {sl.z: bone_measures(sl.gt, sl.pixel_um) for sl in tgt.test}
        base = lambda sl, method, **kw: {"sample": held, "z": sl.z, "method": method, **kw,  # noqa: E731
                                         **{f"gt_{k}": v for k, v in gt_measures[sl.z].items()}}
        for sl in tgt.test:
            rows.append({**base(sl, "otsu", budget="none", noisy=False, seed=0), **score(M.otsu(sl.img), sl)})
        for budget in BUDGETS:
            st, sigma, trows = tune(inner, budget, seeds[:2], log)
            tuning_rows += [{"held_out": held, "budget": f"{budget[0]}+{budget[1]}", **r} for r in trows]
            for sl, noisy, seed in itertools.product(tgt.test, (False, True), seeds):
                pos, neg = simulated_clicks(sl.gt, *budget, seed + 100, noisy)  # Different seeds from tuning
                key = (sl.sample, sl.z)
                b = f"{budget[0]}+{budget[1]}"
                for method, pred in (("rf_clicks", M.rf_clicks(key, sl.img, pos, neg, sigma_max=sigma, seed=seed)),
                                     ("sam", M.sam_points("sam", key, sl.img, pos, neg)),
                                     ("microsam", M.sam_points("microsam", key, sl.img, pos, neg)),
                                     ("dino_clicks", M.dino_clicks(key, sl.img, pos, neg, st, sl.pixel_um))):
                    rows.append({**base(sl, method, budget=b, noisy=noisy, seed=seed), **score(pred, sl)})
            log(f"  clicks {budget} done")
        # Label-based methods: 5 labelled development slices of this sample, or of each other sample
        st = SegmentationSettings(backbone="dinov2_s14")
        for source, train in (("same sample", [(("" + held, sl.z), sl.img, sl.gt) for sl in tgt.dev[:5]]),
                              ("other samples", [((s.name, sl.z), sl.img, sl.gt) for s in inner for sl in s.dev[:5]])):
            head = M.dino_labels_fit(train, st, tgt.test[0].pixel_um)
            rf = M.rf_labels_fit(train)
            for sl in tgt.test:
                key = (sl.sample, sl.z)
                rows.append({**base(sl, "dino_labels", budget=source, noisy=False, seed=0), **score(M.dino_labels_predict(head, key, sl.img, st, sl.pixel_um), sl)})
                rows.append({**base(sl, "rf_labels", budget=source, noisy=False, seed=0), **score(M.rf_labels_predict(rf, key, sl.img), sl)})
            log(f"  labels from {source} done (boneseg chose {head.kind}, context {head.context})")
        pd.DataFrame(rows).to_csv(OUT / "slices.csv", index=False)  # Saved after every sample
        pd.DataFrame(tuning_rows).to_csv(OUT / "tuning.csv", index=False)
    meta = {"samples": names, "seeds": seeds, "n_slices": n_slices, "budgets": BUDGETS, "dino_grid": DINO_GRID, "rf_grid": RF_GRID,
            "method_tag": "method-v1", "minutes": round((time.time() - t0) / 60, 1)}
    (OUT / "run.json").write_text(json.dumps(meta, indent=1))
    log(f"done, {len(rows)} rows")


if __name__ == "__main__":
    main()
