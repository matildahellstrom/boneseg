"""Does fine-tuning DINOv2 help on SegPC-2021? Scored on the 199 validation images with outlines.

DINOv2 Small (last 4 blocks + output layer, colour input) is fine-tuned to predict whole plasma cells on 20 or 100
labelled training images, none of them among the 60 images used to tune the click settings, with early stopping on 20
more. The baseline is the same code with DINOv2 frozen (train_blocks=0). Each model is scored two ways: its own
output layer, and clicks on its features (3 + 6 and 10 + 20, three seeds, the evaluation's clicks and tuned settings).
Metrics: the official challenge score (no size filter, as tuned), and cell F1 and counts with the 16000 px minimum
cell size chosen in segpc_counts.py.
Writes paper/results/segpc_finetune.csv and segpc_finetune_summary.md.
"""
from __future__ import annotations

import copy
import itertools
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import icc_a1
from noise_common import instances_from_mask, match_instances
from segpc_common import load, official_miou
from segpc_evaluate import clicks
from boneseg.backbone import get_backbone, pick_device
from boneseg.finetune import build, finetune, load_dino, predict, prepare
from boneseg.segment import SegmentationSettings, embed_image, segment

OUT = Path(__file__).resolve().parent / "results"
BUDGETS = [(3, 6), (10, 20)]
TUNED = {(3, 6): (0.8, "none", 0.5), (10, 20): (0.4, "none", 0.7)}
MIN_COUNT = 16000


def score(mask, it):
    full = instances_from_mask(mask, 0)
    filt = instances_from_mask(mask, MIN_COUNT)
    m = match_instances(filt, it.instances)
    return {"official": float(np.mean(official_miou(full, it))), **m, "n_pred": int(filt.max()), "n_cells": len(it.cells)}


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
    device = pick_device()
    dino = load_dino()
    st = SegmentationSettings()
    train_all = load("train")
    random.Random(0).shuffle(train_all)
    pool = train_all[60:]                       # The first 60 tuned the click settings
    val_ft = pool[:20]                          # Early stopping
    test = [it for it in load("val") if len(it.cells)]
    rgb = lambda it: it.rgb.astype(np.float32) / 255  # noqa: E731
    ft_dir = OUT / "finetuned"
    ft_dir.mkdir(exist_ok=True)
    models = {}
    for n in (20, 100):
        train = [(rgb(it), it.cell_mask) for it in pool[20:20 + n]]
        val = [(rgb(it), it.cell_mask) for it in val_ft]
        for blocks in (0, 4):
            ft = finetune(train, val, train_blocks=blocks, steps=1000, pretrained=dino, device=device)
            name = f"{'frozen' if blocks == 0 else 'finetuned'}_{n}"
            log(f"{name}: best validation Dice {ft.info['best_val_dice']:.3f} at step {ft.info['best_step']}")
            path = ft_dir / f"segpc_{name}.pt"
            ft.save(path)
            models[name] = (ft, path)
    rows = []
    nets = {k: build(ft, dino=copy.deepcopy(dino), device=device) for k, (ft, _) in models.items()}
    backbones = {"frozen": "dinov2_s14", "finetuned_20": f"dinov2_s14@{models['finetuned_20'][1]}", "finetuned_100": f"dinov2_s14@{models['finetuned_100'][1]}"}
    for bname, bspec in backbones.items():
        bb = get_backbone(bspec)
        for n_done, it in enumerate(test, start=1):
            img = rgb(it)
            emb = embed_image(bb, img, st)
            for budget, seed in itertools.product(BUDGETS, range(3)):
                pos, neg = clicks(it, *budget, seed + 100)
                nw, edge, tpos = TUNED[budget]
                mask = segment(emb, pos, neg, replace(st, neg_weight=nw, edge_refine=edge, threshold_position=tpos)).mask
                rows.append({"image": it.name, "camera": it.camera, "method": f"clicks_{bname}", "condition": f"{budget[0]}+{budget[1]} clicks", "seed": seed, **score(mask, it)})
            if bname == "frozen":   # Output layers, once per image
                for k, net in nets.items():
                    prob = predict(net, prepare(img, it.cell_mask, 980), device)
                    rows.append({"image": it.name, "camera": it.camera, "method": f"layer_{k}", "condition": f"{k.split('_')[1]} labelled images", "seed": 0, **score(prob >= 0.5, it)})
            if n_done % 50 == 0:
                log(f"{bname}: {n_done} of {len(test)} test images")
                pd.DataFrame(rows).to_csv(OUT / "segpc_finetune.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "segpc_finetune.csv", index=False)
    summarize()
    log("done")


def boot(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    b = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def summarize():
    raw = pd.read_csv(OUT / "segpc_finetune.csv", dtype={"image": str})
    df = raw.groupby(["image", "method", "condition"], as_index=False).mean(numeric_only=True)
    names = {"clicks_frozen": "Clicks, frozen DINOv2", "clicks_finetuned_20": "Clicks, DINOv2 fine-tuned on 20 images",
             "clicks_finetuned_100": "Clicks, DINOv2 fine-tuned on 100 images", "layer_frozen_20": "Output layer on frozen DINOv2, 20 images",
             "layer_finetuned_20": "Fine-tuned DINOv2, 20 images", "layer_frozen_100": "Output layer on frozen DINOv2, 100 images",
             "layer_finetuned_100": "Fine-tuned DINOv2, 100 images"}
    lines = ["# Fine-tuning DINOv2 on SegPC-2021\n", "199 validation images. Official score without a size filter; F1 and counts with cells of at least 16000 px.\n",
             "| Method | Condition | Official score (95% CI) | Cell F1 | Count bias per image | Count ICC |", "|---|---|---|---|---|---|"]
    for (m, c), g in df.groupby(["method", "condition"], sort=False):
        r = raw[(raw["method"] == m) & (raw["condition"] == c)]
        tp, fp, fn = r["tp"].sum(), r["fp"].sum(), r["fn"].sum()
        est, lo, hi = boot(g["official"])
        lines.append(f"| {names[m]} | {c} | {est:.3f} ({lo:.3f}–{hi:.3f}) | {2 * tp / max(1, 2 * tp + fp + fn):.3f} | "
                     f"{(g['n_pred'] - g['n_cells']).mean():+.2f} | {icc_a1(g['n_pred'].to_numpy(float), g['n_cells'].to_numpy(float)):.2f} |")
    lines += ["\n| Comparison | Condition | Difference in official score (95% CI) | Images improved |", "|---|---|---|---|"]
    pairs = [("clicks_finetuned_20", "clicks_frozen"), ("clicks_finetuned_100", "clicks_frozen"), ("layer_finetuned_20", "layer_frozen_20"),
             ("layer_finetuned_100", "layer_frozen_100"), ("layer_finetuned_100", "layer_finetuned_20")]
    for a, b in pairs:
        for c in sorted(df.loc[df["method"] == a, "condition"].unique()):
            A = df[(df["method"] == a) & (df["condition"] == c)].set_index("image")["official"]
            bc = c if b.startswith("clicks") or b.split("_")[-1] == a.split("_")[-1] else df.loc[df["method"] == b, "condition"].iloc[0]
            B = df[(df["method"] == b) & (df["condition"] == bc)].set_index("image")["official"]
            d = (A - B).dropna()
            if len(d):
                est, lo, hi = boot(d)
                lines.append(f"| {names[a]} vs {names[b]} | {c} | {est:+.3f} ({lo:+.3f} to {hi:+.3f}) | {(d > 0).mean():.0%} |")
    (OUT / "segpc_finetune_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
