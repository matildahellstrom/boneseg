"""Does fine-tuning DINOv2 help with osteoclasts? Leave-one-batch-out on the NOISe test patches.

For each held-out batch, DINOv2 Small is fine-tuned (last 4 blocks + output layer) on 5 development patches of each
other batch, with early stopping on 5 more of each, in colour. The baseline is the same code with DINOv2 frozen.
Scored on the held-out batch's test patches: the output layer's own masks, and clicks (3 + 6 and 10 + 20, three seeds,
the evaluation's clicks) on frozen against fine-tuned features, with the settings the main evaluation chose in every
fold (lambda 0.8, guided filter, automatic threshold position, cells of at least 1000 px).
Writes paper/results/noise_finetune.csv and noise_finetune_summary.md.
"""
from __future__ import annotations

import copy
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import hier_boot
from noise_common import BATCHES, instances_from_mask, load_batch, osteoclast_clicks, to_channel
from noise_evaluate import score_instances
from boneseg.backbone import get_backbone, pick_device
from boneseg.finetune import build, finetune, load_dino, predict, prepare
from boneseg.segment import SegmentationSettings, embed_image, segment

OUT = Path(__file__).resolve().parent / "results"
MIN_PX = 1000
BUDGETS = [(3, 6), (10, 20)]


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
    data = {b: load_batch(b) for b in BATCHES}
    rgb = lambda p: p.rgb.astype(np.float32) / 255.0  # noqa: E731
    device = pick_device()
    dino = load_dino()
    st = SegmentationSettings()
    ft_dir = OUT / "finetuned"
    ft_dir.mkdir(exist_ok=True)
    rows = []
    for held in BATCHES:
        inner = [b for b in BATCHES if b != held]
        train = [(rgb(p), p.gt) for b in inner for p in data[b]["dev"][:5]]
        val = [(rgb(p), p.gt) for b in inner for p in data[b]["dev"][5:10]]
        test = data[held]["test"]
        for blocks in (0, 4):
            method = "probe_frozen" if blocks == 0 else "finetune_b4"
            ft = finetune(train, val, train_blocks=blocks, steps=1000, pretrained=dino, device=device)
            log(f"held-out {held}, {method}: best validation Dice {ft.info['best_val_dice']:.3f} at step {ft.info['best_step']}")
            net = build(ft, dino=copy.deepcopy(dino), device=device)
            for p in test:
                prob = predict(net, prepare(rgb(p), p.gt, 980), device)
                rows.append({"batch": held, "patch": p.name, "method": method, "condition": "labels from other batches", "seed": 0,
                             **score_instances(instances_from_mask(prob >= 0.5, MIN_PX), p)})
            del net
            if blocks:
                path = ft_dir / f"noise_finetune_b4_without_{held}.pt"
                ft.save(path)
                for cm, bname in (("clicks_frozen", "dinov2_s14"), ("clicks_finetuned_b4", f"dinov2_s14@{path}")):
                    bb = get_backbone(bname)
                    for p in test:
                        emb = embed_image(bb, rgb(p), st)
                        od = to_channel(p.rgb, "od")
                        for budget, seed in itertools.product(BUDGETS, range(3)):
                            pos, neg = osteoclast_clicks(p, od, *budget, seed + 100)
                            rows.append({"batch": held, "patch": p.name, "method": cm, "condition": f"{budget[0]}+{budget[1]} clicks",
                                         "seed": seed, **score_instances(instances_from_mask(segment(emb, pos, neg, st).mask, MIN_PX), p)})
                log("  clicks on frozen and fine-tuned features done")
        pd.DataFrame(rows).to_csv(OUT / "noise_finetune.csv", index=False)
    summarize()


def summarize():
    raw = pd.read_csv(OUT / "noise_finetune.csv")
    df = raw.groupby(["batch", "patch", "method", "condition"], as_index=False).mean(numeric_only=True).rename(columns={"batch": "sample"})
    lines = ["# Fine-tuning DINOv2 on the NOISe osteoclasts\n", "Leave-one-batch-out, test patches only, colour input.\n",
             "| Condition | Method | Dice (95% CI) | F1 at IoU 0.5 | Count bias |", "|---|---|---|---|---|"]
    pairs = [("labels from other batches", "finetune_b4", "probe_frozen"), ("3+6 clicks", "clicks_finetuned_b4", "clicks_frozen"),
             ("10+20 clicks", "clicks_finetuned_b4", "clicks_frozen")]
    for cond, a, b in pairs:
        for m in (b, a):
            g = df[(df["condition"] == cond) & (df["method"] == m)]
            r = raw[(raw["condition"] == cond) & (raw["method"] == m)]
            f1 = np.mean([2 * x["tp"].sum() / max(1, 2 * x["tp"].sum() + x["fp"].sum() + x["fn"].sum()) for _, x in r.groupby("batch")])
            est, lo, hi = hier_boot(g, "dice")
            lines.append(f"| {cond} | {m} | {est:.3f} ({lo:.3f}–{hi:.3f}) | {f1:.3f} | {(g['n_pred'] - g['n_gt']).mean():+.2f} |")
    lines += ["\n| Condition | Fine-tuned minus frozen, Dice (95% CI) | Patches improved |", "|---|---|---|"]
    for cond, a, b in pairs:
        A = df[(df["condition"] == cond) & (df["method"] == a)].set_index(["sample", "patch"])["dice"]
        B = df[(df["condition"] == cond) & (df["method"] == b)].set_index(["sample", "patch"])["dice"]
        d = (A - B).dropna().reset_index().rename(columns={"dice": "diff"})
        est, lo, hi = hier_boot(d, "diff")
        lines.append(f"| {cond} | {est:+.3f} ({lo:+.3f} to {hi:+.3f}) | {(d['diff'] > 0).mean():.0%} |")
    (OUT / "noise_finetune_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
