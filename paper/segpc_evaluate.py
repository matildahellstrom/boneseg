"""Plasma cells on SegPC-2021: boneseg against baselines, settings tuned on training images, scored on validation.

Protocol
  * Tuning: 60 training images (seeded draw). Test: all 200 validation images, which have expert outlines.
  * Simulated clicks: object clicks inside the outlined plasma cells (spread so every cell gets one while clicks
    last), background clicks at least 10 px from any outlined cell. Unoutlined cells can be plasma cells that were
    not chosen as "cells of interest", so background clicks are not aimed at them. 3 + 6 and 10 + 20 clicks, three seeds.
  * Whole cells: boneseg from clicks (background weight, edge refinement, threshold position and minimum cell size
    tuned on the training images by the official score), SAM, micro-SAM, the random forest (minimum cell size tuned),
    Otsu, and boneseg's learned model from 5 labelled training images.
  * Nucleus and cytoplasm: boneseg's several-structure mode with nucleus and cytoplasm clicks.
  * Metrics: the official challenge score (mean over expert cells of the best IoU of any predicted cell, which does
    not penalize extra cells); cells matched one-to-one at IoU >= 0.5 (precision counts unoutlined cells as errors,
    so it is a lower bound); Dice of the cell, nucleus and cytoplasm masks within 25 px of the outlined cells.
Writes paper/results/segpc_slices.csv and segpc_tuning.csv. Usage: python paper/segpc_evaluate.py
"""
from __future__ import annotations

import itertools
import json
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi

import methods as M
from noise_common import instances_from_mask, match_instances
from noise_evaluate import click_masks
from segpc_common import load, official_miou
from boneseg import metrics
from boneseg.backbone import get_backbone
from boneseg.head import segment_with_head, train_head
from boneseg.segment import SegmentationSettings, embed_image, segment, segment_multi

OUT = Path(__file__).resolve().parent / "results"
BUDGETS = [(3, 6), (10, 20)]
MIN_PX = [0, 1000, 3000, 6000]
GRID = {"neg_weight": [0.4, 0.8, 1.2], "edge_refine": ["none", "guided"], "threshold_position": [-1, 0.5, 0.7, 0.9]}


def clicks(item, n_pos, n_neg, seed, region="cell"):
    """Object clicks in the outlined cells (region 'cell', 'nucleus' or 'cytoplasm'), background clicks away from them."""
    rng = np.random.default_rng(seed)
    cls = item.classes
    target = {"cell": cls > 0, "nucleus": cls == 2, "cytoplasm": cls == 1}[region]
    inner = ndi.binary_erosion(target, iterations=4)
    inner = inner if inner.any() else target
    inst = item.instances
    far = ~ndi.binary_dilation(cls > 0, iterations=10)

    def pick(mask, k):
        ys, xs = np.nonzero(mask)
        if not len(ys) or k <= 0:
            return []
        i = rng.choice(len(ys), k, replace=len(ys) < k)
        return [(int(ys[j]), int(xs[j])) for j in i]

    pos = []
    cells = [c for c in range(1, int(inst.max()) + 1) if (inner & (inst == c)).any()]
    rng.shuffle(cells)
    for c in cells[:n_pos]:
        pos += pick(inner & (inst == c), 1)
    pos += pick(inner, n_pos - len(pos))
    return pos, pick(far, n_neg)


def near(item):
    return ndi.binary_dilation(item.cell_mask, iterations=25)


def score(inst, item) -> dict:
    o = official_miou(inst, item)
    m = match_instances(inst, item.instances)
    roi = near(item)
    return {"official": float(np.mean(o)), "n_cells": len(item.cells), **m, "n_pred": int(inst.max()),
            "dice_local": metrics.dice((inst > 0) & roi, item.cell_mask & roi)}


def scores_by_min(mask, item) -> dict:
    lab, n = ndi.label(ndi.binary_fill_holes(mask))
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    out = {}
    for mp in MIN_PX:
        keep = sizes >= mp
        keep[0] = False
        new = np.zeros(n + 1, np.int32)
        new[keep] = np.arange(1, keep.sum() + 1)
        out[mp] = score(new[lab], item)
    return out


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
    bb = get_backbone("dinov2_s14")
    st = SegmentationSettings()
    train_all = load("train")
    random.Random(0).shuffle(train_all)
    tune, label_pool = train_all[:60], train_all[60:]
    test = [it for it in load("val") if len(it.cells)]   # Image 610 has no outlined cell, so nothing to score
    log(f"{len(tune)} tuning and {len(test)} test images")
    embs = {}

    def prep(it):
        k = (it.split, it.name)
        if k not in embs:
            img = it.rgb.astype(np.float32) / 255
            embs[k] = (img, embed_image(bb, img, st))
        return embs[k]

    tuned_path = OUT / "segpc_tuned.json"
    if tuned_path.exists():   # Resume: the tuned settings are reused
        t = json.loads(tuned_path.read_text())
        best = {}
        for k, v in t["best"].items():
            name, budget = json.loads(k)
            best[(name, tuple(budget))] = (tuple(v[0]), v[1]) if isinstance(v, list) else v
        mp_otsu = t["mp_otsu"]
        log("tuned settings loaded from " + tuned_path.name)
    else:
        best, mp_otsu = tune_settings(tune, prep, log)
        tuned_path.write_text(json.dumps({"best": {json.dumps([k[0], list(k[1])]): ([list(v[0]), v[1]] if isinstance(v, tuple) else v) for k, v in best.items()},
                                          "mp_otsu": mp_otsu}, indent=1))
    rows = []
    done = set()
    if (OUT / "segpc_slices.csv").exists():
        prev = pd.read_csv(OUT / "segpc_slices.csv", dtype={"image": str})
        done = set(prev["image"])
        rows = prev.to_dict("records")
        log(f"resuming after {len(done)} finished test images")
    run_test(test, done, rows, best, mp_otsu, prep, label_pool, st, log)


def tune_settings(tune, prep, log):
    """Settings for every method on the training images, by the official score."""
    dino, base = {}, {}
    for it in tune:
        img, emb = prep(it)
        key = (it.split, it.name)
        for budget, seed in itertools.product(BUDGETS, range(2)):
            pos, neg = clicks(it, *budget, seed)
            for k, mask in click_masks(emb, img.mean(-1), pos, neg, GRID).items():
                dino[(budget, seed, key, k)] = scores_by_min(mask, it)
            if seed == 0:
                for meth, mask in (("sam", M.sam_points("sam", key, img, pos, neg)), ("microsam", M.sam_points("microsam", key, img, pos, neg)),
                                   ("rf_clicks", M.rf_clicks(key, img.mean(-1), pos, neg, seed=0))):
                    base[(meth, budget, key)] = scores_by_min(mask, it)
        base[("otsu", None, key)] = scores_by_min(M.otsu(1 - img.mean(-1)), it)
    keys = [(it.split, it.name) for it in tune]
    best, tuning = {}, []
    for budget in BUDGETS:
        cand = {(k, mp): np.mean([dino[(budget, s, key, k)][mp]["official"] for key in keys for s in range(2)])
                for k in itertools.product(*GRID.values()) for mp in MIN_PX}
        (k, mp), val = max(cand.items(), key=lambda kv: kv[1])
        best[("dino", budget)] = (k, mp)
        tuning.append({"method": "dino_clicks", "budget": f"{budget[0]}+{budget[1]}", "setting": f"neg_weight={k[0]}, edge={k[1]}, position={k[2]}, min_px={mp}", "official": val})
        for meth in ("sam", "microsam", "rf_clicks"):
            best[(meth, budget)] = max(MIN_PX, key=lambda m: np.mean([base[(meth, budget, key)][m]["official"] for key in keys]))
            tuning.append({"method": meth, "budget": f"{budget[0]}+{budget[1]}", "setting": f"min_px={best[(meth, budget)]}"})
    mp_otsu = max(MIN_PX, key=lambda m: np.mean([base[("otsu", None, key)][m]["official"] for key in keys]))
    log("tuned: " + "; ".join(f"{a} {b}: {v}" for (a, b), v in best.items()))
    pd.DataFrame(tuning).to_csv(OUT / "segpc_tuning.csv", index=False)
    return best, mp_otsu


def run_test(test, done, rows, best, mp_otsu, prep, label_pool, st, log):
    head = train_head([(prep(it)[1], it.cell_mask) for it in label_pool[:5]], st, [(0, i) for i in range(5)])
    log(f"learned model on 5 training images: {head.kind}, context {head.context}")
    for n_done, it in enumerate(test, start=1):
        if it.name in done:
            continue
        img, emb = prep(it)
        key = (it.split, it.name)
        b = {"image": it.name, "camera": it.camera}
        rows.append({**b, "method": "otsu", "condition": "no input", "seed": 0, **score(instances_from_mask(M.otsu(1 - img.mean(-1)), mp_otsu), it)})
        mp_lab = best[("dino", (10, 20))][1]
        rows.append({**b, "method": "dino_labels", "condition": "5 labelled training images", "seed": 0,
                     **score(instances_from_mask(segment_with_head(head, emb, st).mask, mp_lab), it)})
        for budget, seed in itertools.product(BUDGETS, range(3)):
            cond = f"{budget[0]}+{budget[1]} clicks"
            pos, neg = clicks(it, *budget, seed + 100)
            (nw, edge, tpos), mp = best[("dino", budget)]
            s2 = replace(st, neg_weight=nw, edge_refine=edge, threshold_position=tpos)
            rows.append({**b, "method": "dino_clicks", "condition": cond, "seed": seed, **score(instances_from_mask(segment(emb, pos, neg, s2).mask, mp), it)})
            for meth, mask in (("sam", M.sam_points("sam", key, img, pos, neg)), ("microsam", M.sam_points("microsam", key, img, pos, neg)),
                               ("rf_clicks", M.rf_clicks(key, img.mean(-1), pos, neg, seed=seed))):
                rows.append({**b, "method": meth, "condition": cond, "seed": seed, **score(instances_from_mask(mask, best[(meth, budget)]), it)})
            # Nucleus and cytoplasm as two structures, same number of object clicks split between them
            npos = max(1, budget[0] // 2)
            pn, _ = clicks(it, npos, 0, seed + 200, "nucleus")
            pc, _ = clicks(it, budget[0] - npos if budget[0] > 1 else 1, 0, seed + 300, "cytoplasm")
            res = segment_multi(emb, [{"name": "cytoplasm", "pos": pc}, {"name": "nucleus", "pos": pn}], neg, s2)
            lab = res.labels   # 1 cytoplasm, 2 nucleus
            roi = near(it)
            cls = it.classes
            rows.append({**b, "method": "dino_multi", "condition": cond, "seed": seed,
                         **score(instances_from_mask(lab > 0, mp), it),
                         "dice_nucleus": metrics.dice((lab == 2) & roi, (cls == 2) & roi),
                         "dice_cytoplasm": metrics.dice((lab == 1) & roi, (cls == 1) & roi)})
        if n_done % 20 == 0:
            log(f"{n_done} of {len(test)} test images")
            pd.DataFrame(rows).to_csv(OUT / "segpc_slices.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "segpc_slices.csv", index=False)
    (OUT / "segpc_run.json").write_text(json.dumps({"tuning_images": 60, "test_images": len(test), "budgets": BUDGETS, "grid": GRID,
                                                     "min_px": MIN_PX, "best": {f"{a} {b}": str(v) for (a, b), v in best.items()}}, indent=1))
    log("done")


if __name__ == "__main__":
    main()
