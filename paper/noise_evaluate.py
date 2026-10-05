"""Osteoclast segmentation and counting on the NOISe mouse data, leave-one-batch-out.

Protocol (mirrors evaluate.py, with the five NOISe batches in place of samples):
  * 20 development and 20 test patches per batch (paper/noise_common.py), split by well.
  * pilot: on development patches only, chooses the image given to DINOv2: colour, or one stain channel.
  * full: for each held-out batch, every setting is tuned on the development patches of the other four batches:
      boneseg from clicks: background weight, edge refinement, threshold position and minimum cell size
      SAM, micro-SAM, the random forest and Otsu: minimum cell size
    and all methods are scored on the held-out batch's test patches with the same simulated clicks (3 + 6 and
    10 + 20, three seeds). Label-based methods train on 5 development patches of the same batch, or of each other batch.
  * Metrics per patch: Dice of the osteoclast area; cells matched one-to-one at IoU >= 0.5 (precision, recall, F1);
    the osteoclast count and the area fraction against the experts.

Writes paper/results/noise_slices.csv, noise_tuning.csv and noise_pilot.csv.
Usage: python paper/noise_evaluate.py pilot | full [--channel rgb] [--passes 2]
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
from noise_common import (BATCHES, instances_from_mask, load_batch, match_instances, osteoclast_clicks, to_channel)
from boneseg import metrics
from boneseg.backbone import get_backbone
from boneseg.head import segment_with_head, train_head
from boneseg.segment import (SegmentationSettings, auto_position, calibrate_threshold, embed_image, guided_filter,
                             normalize_scores, prototypes, sample_points, score_grid, upsample)

OUT = Path(__file__).resolve().parent / "results"
BUDGETS = [(3, 6), (10, 20)]
MIN_PX = [0, 1000, 2500, 5000]
GRID = {"neg_weight": [0.4, 0.8, 1.2], "edge_refine": ["none", "guided"], "threshold_position": [-1, 0.5, 0.7, 0.9]}
PX = (1.0, 1.0)   # NOISe gives no pixel size; areas are in pixels


def image_for(p, kind):
    return p.rgb.astype(np.float32) / 255.0 if kind == "rgb" else to_channel(p.rgb, kind)


def score_instances(inst, p) -> dict:
    m = match_instances(inst, p.instances)
    return {"dice": metrics.dice(inst > 0, p.gt), **m, "n_pred": int(inst.max()), "n_gt": p.n_cells,
            "area_pred": 100 * float((inst > 0).mean()), "area_gt": 100 * float(p.gt.mean())}


def scores_by_min(mask, p) -> dict:
    """Scores of one mask for every minimum cell size; the labelling is done once."""
    import scipy.ndimage as ndi
    lab, n = ndi.label(ndi.binary_fill_holes(mask))
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    out = {}
    for mp in MIN_PX:
        keep = sizes >= mp
        keep[0] = False
        new = np.zeros(n + 1, np.int32)
        new[keep] = np.arange(1, keep.sum() + 1)
        out[mp] = score_instances(new[lab], p)
    return out


def f1(rows) -> float:
    tp, fp, fn = sum(r["tp"] for r in rows), sum(r["fp"] for r in rows), sum(r["fn"] for r in rows)
    return 2 * tp / max(1, 2 * tp + fp + fn)


def click_masks(emb, img_guide, pos, neg, settings_grid):
    """boneseg masks for every combination of background weight, edge refinement and threshold position."""
    P, N = prototypes(emb, pos), prototypes(emb, neg)
    cell = max(emb.height / emb.grid.shape[0], emb.width / emb.grid.shape[1])
    out = {}
    for nw in settings_grid["neg_weight"]:
        raw = upsample(normalize_scores(score_grid(emb.grid, P, N, nw), "robust"), (emb.height, emb.width))
        for edge in settings_grid["edge_refine"]:
            m = raw if edge == "none" else guided_filter(img_guide, raw, max(1, int(round(cell))), 0.01)
            ps, ns = sample_points(m, pos), sample_points(m, neg)
            for tp_ in settings_grid["threshold_position"]:
                out[(nw, edge, tp_)] = m >= calibrate_threshold(ps, ns, auto_position(tp_, len(pos), len(neg)))
    return out


def pilot(args, log):
    bb = get_backbone("dinov2_s14")
    rows = []
    fixed = {"neg_weight": [0.8], "edge_refine": ["guided"], "threshold_position": [-1]}
    for b in BATCHES:
        for p in load_batch(b)["dev"]:
            od = to_channel(p.rgb, "od")
            for kind, passes in itertools.product(["rgb", "green", "od", "hema"], [1, 2]):
                img = image_for(p, kind)
                emb = embed_image(bb, img, SegmentationSettings(shift_passes=passes))
                guide = img.mean(-1) if img.ndim == 3 else img
                for budget, seed in itertools.product(BUDGETS, range(2)):
                    pos, neg = osteoclast_clicks(p, od, *budget, seed)
                    mask = click_masks(emb, guide, pos, neg, fixed)[(0.8, "guided", -1)]
                    for mp, sc in scores_by_min(mask, p).items():
                        rows.append({"batch": b, "patch": p.name, "channel": kind, "passes": passes, "budget": f"{budget[0]}+{budget[1]}",
                                     "seed": seed, "min_px": mp, **sc})
        log(f"pilot: {b} done")
        pd.DataFrame(rows).to_csv(OUT / "noise_pilot.csv", index=False)
    df = pd.DataFrame(rows)
    summ = df.groupby(["budget", "channel", "passes", "min_px"]).apply(
        lambda g: pd.Series({"dice": g["dice"].mean(), "f1": f1(g.to_dict("records")),
                             "count_err": (g["n_pred"] - g["n_gt"]).abs().mean()}), include_groups=False).round(3)
    print(summ.sort_values("f1", ascending=False).groupby(level=0).head(8).to_string())


def full(args, log):
    bb = get_backbone("dinov2_s14")
    st = SegmentationSettings(shift_passes=args.passes)
    data = {b: load_batch(b) for b in BATCHES}
    seeds = [0, 1, 2]
    emb_cache, od_cache = {}, {}

    def prep(p):
        k = (p.batch, p.name)
        if k not in emb_cache:
            img = image_for(p, args.channel)
            emb_cache[k] = (img, embed_image(bb, img, st))
            od_cache[k] = to_channel(p.rgb, "od")
        return emb_cache[k][0], emb_cache[k][1], od_cache[k]

    # Baseline masks on every development patch once (seed 0), to tune their minimum cell size per fold
    log("baseline masks on development patches")
    base_dev = {}
    for b in BATCHES:
        for p in data[b]["dev"]:
            img, emb, od = prep(p)
            key = (p.batch, p.name)
            for budget in BUDGETS:
                pos, neg = osteoclast_clicks(p, od, *budget, 0)
                for meth, mask in (("sam", M.sam_points("sam", key, img, pos, neg)), ("microsam", M.sam_points("microsam", key, img, pos, neg)),
                                   ("rf_clicks", M.rf_clicks(key, od, pos, neg, seed=0))):
                    base_dev[(meth, budget, key)] = scores_by_min(mask, p)
            base_dev[("otsu", None, key)] = scores_by_min(M.otsu(od), p)
    log("boneseg masks on development patches")
    dino_dev = {}
    for b in BATCHES:
        for p in data[b]["dev"]:
            img, emb, od = prep(p)
            guide = img.mean(-1) if img.ndim == 3 else img
            for budget, seed in itertools.product(BUDGETS, range(2)):
                pos, neg = osteoclast_clicks(p, od, *budget, seed)
                for k, mask in click_masks(emb, guide, pos, neg, GRID).items():
                    dino_dev[(budget, seed, (p.batch, p.name), k)] = scores_by_min(mask, p)

    rows, tuning = [], []
    for held in BATCHES:
        log(f"held-out batch {held}")
        inner = [b for b in BATCHES if b != held]
        inner_keys = [(b, p.name) for b in inner for p in data[b]["dev"]]
        best = {}
        for budget in BUDGETS:
            cand = {}
            for k in itertools.product(*GRID.values()):
                for mp in MIN_PX:
                    cand[(k, mp)] = f1([dino_dev[(budget, s, key, k)][mp] for key in inner_keys for s in range(2)])
            (k, mp), val = max(cand.items(), key=lambda kv: kv[1])
            best[("dino", budget)] = (k, mp)
            tuning.append({"held_out": held, "method": "dino_clicks", "budget": f"{budget[0]}+{budget[1]}", "setting": f"neg_weight={k[0]}, edge={k[1]}, position={k[2]}, min_px={mp}", "f1": val})
            for meth in ("sam", "microsam", "rf_clicks"):
                mp_b = max(MIN_PX, key=lambda m: f1([base_dev[(meth, budget, key)][m] for key in inner_keys]))
                best[(meth, budget)] = mp_b
                tuning.append({"held_out": held, "method": meth, "budget": f"{budget[0]}+{budget[1]}", "setting": f"min_px={mp_b}", "f1": None})
        mp_otsu = max(MIN_PX, key=lambda m: f1([base_dev[("otsu", None, key)][m] for key in inner_keys]))
        log(f"  tuned: " + "; ".join(f"{k[0]} {k[1]}: {v}" for k, v in best.items()))

        def finish(mask, mp):
            return score_instances(instances_from_mask(mask, mp), p)

        for p in data[held]["test"]:
            img, emb, od = prep(p)
            key = (p.batch, p.name)
            base = {"batch": held, "patch": p.name}
            rows.append({**base, "method": "otsu", "condition": "no input", "seed": 0, **finish(M.otsu(od), mp_otsu)})
            for budget, seed in itertools.product(BUDGETS, seeds):
                pos, neg = osteoclast_clicks(p, od, *budget, seed + 100)
                cond = f"{budget[0]}+{budget[1]} clicks"
                (nw, edge, tpos), mp = best[("dino", budget)]
                s2 = replace(st, neg_weight=nw, edge_refine=edge, threshold_position=tpos)
                from boneseg.segment import segment
                rows.append({**base, "method": "dino_clicks", "condition": cond, "seed": seed, **finish(segment(emb, pos, neg, s2).mask, mp)})
                for meth, mask in (("sam", M.sam_points("sam", key, img, pos, neg)), ("microsam", M.sam_points("microsam", key, img, pos, neg)),
                                   ("rf_clicks", M.rf_clicks(key, od, pos, neg, seed=seed))):
                    rows.append({**base, "method": meth, "condition": cond, "seed": seed, **finish(mask, best[(meth, budget)])})
        log("  clicks done")
        # Learned models from 5 labelled development patches, same batch or each other batch
        mp_lab = best[("dino", (10, 20))][1]
        for source, train in (("same batch", data[held]["dev"][:5]), ("other batches", [p for b in inner for p in data[b]["dev"][:5]])):
            samples = [(prep(p)[1], p.gt) for p in train]
            head = train_head(samples, st, [(0, i) for i in range(len(samples))], pixel_um=PX)
            rf = M.rf_labels_fit([((p.batch, p.name), prep(p)[2], p.gt) for p in train])
            for p in data[held]["test"]:
                img, emb, od = prep(p)
                base = {"batch": held, "patch": p.name, "condition": f"labels from {source}", "seed": 0}
                rows.append({**base, "method": "dino_labels", **finish(segment_with_head(head, emb, st, PX).mask, mp_lab)})
                rows.append({**base, "method": "rf_labels", **finish(M.rf_labels_predict(rf, (p.batch, p.name), od), mp_lab)})
            log(f"  labels from {source} done (boneseg chose {head.kind}, context {head.context})")
        pd.DataFrame(rows).to_csv(OUT / "noise_slices.csv", index=False)
        pd.DataFrame(tuning).to_csv(OUT / "noise_tuning.csv", index=False)
    (OUT / "noise_run.json").write_text(json.dumps({"channel": args.channel, "passes": args.passes, "budgets": BUDGETS, "grid": GRID,
                                                     "min_px": MIN_PX, "seeds": seeds}, indent=1))
    log("done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pilot", "full"])
    ap.add_argument("--channel", default="rgb")
    ap.add_argument("--passes", type=int, default=2)
    args = ap.parse_args()
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
    (pilot if args.mode == "pilot" else full)(args, log)


if __name__ == "__main__":
    main()
