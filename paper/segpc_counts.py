"""Counting plasma cells on SegPC-2021, with a minimum cell size chosen for finding cells.

The official score ignores extra predicted cells, so tuning on it (segpc_evaluate.py) chose no size filter, and every
speck counted as a cell. Here, for each method and click budget, the minimum cell size is chosen on the same 60
training images by cell F1 at IoU >= 0.5, with the segmentation settings segpc_evaluate.py chose, and the validation
images are scored with it: precision, recall, F1, the per-image count against the experts, and the official score.
Writes paper/results/segpc_counts.csv and segpc_counts_summary.md.
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

import methods as M
from analyze import icc_a1
from segpc_common import load
from segpc_evaluate import BUDGETS, clicks, scores_by_min
from boneseg.backbone import get_backbone
from boneseg.head import segment_with_head, train_head
from boneseg.segment import SegmentationSettings, embed_image, segment

OUT = Path(__file__).resolve().parent / "results"
SIZES = [0, 2000, 4000, 8000, 12000, 16000]
TUNED = {(3, 6): (0.8, "none", 0.5), (10, 20): (0.4, "none", 0.7)}   # As chosen by segpc_evaluate.py (segpc_run.json)


def f1(rows):
    tp, fp, fn = sum(r["tp"] for r in rows), sum(r["fp"] for r in rows), sum(r["fn"] for r in rows)
    return 2 * tp / max(1, 2 * tp + fp + fn)


def main():
    import segpc_evaluate as E
    E.MIN_PX = SIZES
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
    bb = get_backbone("dinov2_s14")
    st = SegmentationSettings()
    train_all = load("train")
    random.Random(0).shuffle(train_all)
    tune, label_pool = train_all[:60], train_all[60:]
    test = [it for it in load("val") if len(it.cells)]   # Image 610 has no outlined cell
    head = train_head([(embed_image(bb, it.rgb.astype(np.float32) / 255, st), it.cell_mask) for it in label_pool[:5]], st, [(0, i) for i in range(5)])

    def masks(it, budget, seed):
        img = it.rgb.astype(np.float32) / 255
        emb = embed_image(bb, img, st)
        pos, neg = clicks(it, *budget, seed)
        nw, edge, tpos = TUNED[budget]
        key = (it.split, it.name)
        return {"dino_clicks": segment(emb, pos, neg, replace(st, neg_weight=nw, edge_refine=edge, threshold_position=tpos)).mask,
                "sam": M.sam_points("sam", key, img, pos, neg), "rf_clicks": M.rf_clicks(key, img.mean(-1), pos, neg, seed=seed),
                "dino_labels": segment_with_head(head, emb, st).mask}

    sizes_path = OUT / "segpc_count_sizes.json"
    if sizes_path.exists():   # Resume: the chosen minimum cell sizes are reused
        best = {(k.split("|")[0], tuple(json.loads(k.split("|")[1]))): v for k, v in json.loads(sizes_path.read_text()).items()}
        log("minimum cell sizes loaded from " + sizes_path.name)
    else:
        best = choose_sizes(tune, masks, log)
        sizes_path.write_text(json.dumps({f"{k[0]}|{json.dumps(list(k[1]))}": v for k, v in best.items()}, indent=1))
    rows, done = [], set()
    if (OUT / "segpc_counts.csv").exists():
        prev = pd.read_csv(OUT / "segpc_counts.csv", dtype={"image": str})
        # Only images with every row are kept; a half-finished image is redone
        full = prev.groupby("image").size()
        done = set(full[full == full.max()].index)
        rows = prev[prev["image"].isin(done)].to_dict("records")
        log(f"resuming after {len(done)} finished test images")
    finish(test, done, rows, best, masks, log)


def choose_sizes(tune, masks, log):
    tr = {}
    for it in tune:
        for budget in BUDGETS:
            for meth, m in masks(it, budget, 0).items():
                tr[(meth, budget, it.name)] = scores_by_min(m, it)
    best = {(meth, b): max(SIZES, key=lambda s: f1([tr[(meth, b, it.name)][s] for it in tune])) for meth in ("dino_clicks", "sam", "rf_clicks", "dino_labels") for b in BUDGETS}
    log("minimum cell sizes chosen by F1: " + "; ".join(f"{k[0]} {k[1]}: {v} px" for k, v in best.items()))
    return best


def finish(test, done, rows, best, masks, log):
    for n, it in enumerate(test, start=1):
        if it.name in done:
            continue
        for budget, seed in itertools.product(BUDGETS, range(3)):
            for meth, m in masks(it, budget, seed + 100).items():
                if meth == "dino_labels" and (budget != (10, 20) or seed):
                    continue
                cond = "5 labelled training images" if meth == "dino_labels" else f"{budget[0]}+{budget[1]} clicks"
                rows.append({"image": it.name, "camera": it.camera, "method": meth, "condition": cond, "seed": seed, "min_px": best[(meth, budget)],
                             **scores_by_min(m, it)[best[(meth, budget)]]})
        if n % 25 == 0:
            log(f"{n} of {len(test)} test images")
            pd.DataFrame(rows).to_csv(OUT / "segpc_counts.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "segpc_counts.csv", index=False)
    lines = ["# Counting plasma cells on SegPC-2021\n", "Minimum cell size chosen on 60 training images by cell F1; 200 validation images.\n",
             "| Method | Condition | Min. cell size | Precision | Recall | F1 | Count bias per image | Count ICC | Official score |", "|---|---|---|---|---|---|---|---|---|"]
    names = {"dino_clicks": "boneseg, clicks", "sam": "SAM ViT-B", "rf_clicks": "Random forest", "dino_labels": "boneseg learned model"}
    for (meth, cond), g in df.groupby(["method", "condition"]):
        tp, fp, fn = g["tp"].sum(), g["fp"].sum(), g["fn"].sum()
        per = g.groupby("image")[["n_pred", "n_cells", "official"]].mean()
        lines.append(f"| {names[meth]} | {cond} | {int(g['min_px'].iloc[0])} px | {tp / max(1, tp + fp):.3f} | {tp / max(1, tp + fn):.3f} | "
                     f"{2 * tp / max(1, 2 * tp + fp + fn):.3f} | {(per['n_pred'] - per['n_cells']).mean():+.2f} | "
                     f"{icc_a1(per['n_pred'].to_numpy(float), per['n_cells'].to_numpy(float)):.2f} | {per['official'].mean():.3f} |")
    (OUT / "segpc_counts_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    log("done")


if __name__ == "__main__":
    main()
