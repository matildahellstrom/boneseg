"""Fine-tune on several samples, then adapt: does pre-fine-tuning DINOv2 on the other samples help when only 5
labelled slices of a new sample are available? Leave-one-sample-out over the five Liu samples, test slices only.

For each held-out sample, three fine-tunings of DINOv2 Small (last 4 blocks, 1000 steps, best validation step):
  same           5 labelled development slices of the held-out sample (its other 5 development slices validate),
                 as in finetune_experiment.py
  other          5 labelled development slices of each other sample (their other development slices validate);
                 the held-out sample is never seen
  other_then_same  "other", then fine-tuned further on the held-out sample's 5 slices, as "same"
and the frozen backbone for reference. Each is scored three ways on the held-out test slices:
  own            the fine-tuned network's own output layer (frozen: an output layer trained on the 5 slices)
  learned        boneseg's learned model (calibrated threshold) on the backbone's features, from the 5 slices
  clicks_3_6     3 + 6 clean simulated clicks on the backbone's features (app defaults, three seeds)
Writes paper/results/finetune_adapt.csv and finetune_adapt_summary.md.
"""
from __future__ import annotations

import copy
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import hier_boot
from common import available_samples, bone_measures, load_sample, simulated_clicks
from boneseg import metrics
from boneseg.backbone import get_backbone, pick_device
from boneseg.finetune import build, finetune, load_dino, predict, prepare
from boneseg.head import segment_with_head, train_head
from boneseg.segment import SegmentationSettings, embed_image, segment

OUT = Path(__file__).resolve().parent / "results"
LABELS = {"frozen": "Frozen DINOv2", "same": "Fine-tuned on the sample's 5 slices", "other": "Fine-tuned on the other samples",
          "other_then_same": "Fine-tuned on the other samples, then the sample's 5 slices"}
SCORING = {"own": "own output layer", "learned": "learned model (calibrated)", "clicks_3_6": "3 + 6 clicks"}


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
    names = available_samples()
    samples = {n: load_sample(n) for n in names}
    device = pick_device()
    dino = load_dino("dinov2_s14")
    st = SegmentationSettings()
    ftdir = OUT / "finetuned"
    ftdir.mkdir(exist_ok=True)
    rows = []

    def add(held, sl, backbone, scoring, pred, seed=0):
        pm, gm = bone_measures(pred, sl.pixel_um), bone_measures(sl.gt, sl.pixel_um)
        rows.append({"sample": held, "z": sl.z, "backbone": backbone, "scoring": scoring, "seed": seed, "dice": metrics.dice(pred, sl.gt),
                     "bar_bias": pm["B.Ar/T.Ar_%"] - gm["B.Ar/T.Ar_%"]})

    for held in names:
        tgt = samples[held]
        same = ([(sl.img, sl.gt) for sl in tgt.dev[:5]], [(sl.img, sl.gt) for sl in tgt.dev[5:]])
        others = [samples[n] for n in names if n != held]
        other = ([(sl.img, sl.gt) for s in others for sl in s.dev[:5]], [(sl.img, sl.gt) for s in others for sl in s.dev[5:]])
        fts = {}
        fts["same"] = finetune(*same, train_blocks=4, steps=1000, pretrained=dino, device=device)
        fts["other"] = finetune(*other, train_blocks=4, steps=1000, pretrained=dino, device=device)
        fts["other_then_same"] = finetune(*same, train_blocks=4, steps=1000, pretrained=dino, device=device, init=fts["other"])
        fts["frozen"] = finetune(*same, train_blocks=0, steps=1000, pretrained=dino, device=device)   # Output layer only
        log(f"{held}: best validation Dice " + ", ".join(f"{k} {v.info['best_val_dice']:.3f}" for k, v in fts.items()))
        for name, ft in fts.items():
            net = build(ft, dino=copy.deepcopy(dino), device=device)
            for sl in tgt.test:
                add(held, sl, name, "own", predict(net, prepare(sl.img, sl.gt, 980), device) >= 0.5)
            del net
            if name == "frozen":
                bb = get_backbone("dinov2_s14")
            else:
                path = ftdir / f"adapt_{name}_{held}.pt"
                ft.save(path)
                bb = get_backbone(f"dinov2_s14@{path}")
            head = train_head([(embed_image(bb, sl.img, st), sl.gt) for sl in tgt.dev[:5]], st, [(0, i) for i in range(5)], pixel_um=tgt.test[0].pixel_um)
            for sl in tgt.test:
                emb = embed_image(bb, sl.img, st)
                add(held, sl, name, "learned", segment_with_head(head, emb, st, sl.pixel_um).mask)
                for seed in range(3):
                    pos, neg = simulated_clicks(sl.gt, 3, 6, seed + 100)
                    add(held, sl, name, "clicks_3_6", segment(emb, pos, neg, st, sl.pixel_um).mask, seed)
        log(f"{held}: scored")
        pd.DataFrame(rows).to_csv(OUT / "finetune_adapt.csv", index=False)
    summarize()
    log("done")


def summarize():
    df = pd.read_csv(OUT / "finetune_adapt.csv")
    d = df.groupby(["sample", "z", "backbone", "scoring"], as_index=False)[["dice", "bar_bias"]].mean()
    samples = list(dict.fromkeys(d["sample"]))
    lines = ["# Fine-tune on several samples, then adapt\n",
             f"Leave-one-sample-out over {', '.join(samples)}; test slices of the held-out sample. 5 labelled slices of it (plus 5 "
             "for validation) are the only target labels. Dice with 95% intervals (hierarchical bootstrap over samples and slices), "
             "and the bias in bone area (B.Ar/T.Ar, percentage points).\n"]
    for scoring, label in SCORING.items():
        lines += [f"\n## Scored by the {label}\n", "| Backbone | " + " | ".join(samples) + " | Mean (95% CI) | B.Ar/T.Ar bias |",
                  "|---|" + "---|" * len(samples) + "---|---|"]
        g = d[d.scoring == scoring]
        for bb, lab in LABELS.items():
            x = g[g.backbone == bb]
            if x.empty:
                continue
            e, lo, hi = hier_boot(x, "dice")
            per = x.groupby("sample")["dice"].mean()
            lines.append(f"| {lab} | " + " | ".join(f"{per.get(s, np.nan):.3f}" for s in samples) + f" | {e:.3f} ({lo:.3f}–{hi:.3f}) | {x['bar_bias'].mean():+.1f} |")
        base = g[g.backbone == "same"].set_index(["sample", "z"])["dice"]
        for bb in ("other_then_same", "other", "frozen"):
            x = (g[g.backbone == bb].set_index(["sample", "z"])["dice"] - base).dropna().reset_index().rename(columns={"dice": "diff"})
            if len(x):
                e, lo, hi = hier_boot(x, "diff")
                lines.append(f"\n{LABELS[bb]} minus fine-tuned on the sample's 5 slices: {e:+.3f} ({lo:+.3f} to {hi:+.3f}), "
                             f"better on {(x['diff'] > 0).mean():.0%} of slices")
    (OUT / "finetune_adapt_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
