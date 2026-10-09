"""Larger and newer backbones: does a bigger DINOv2 (Base, Large), or DINOv3 when its weights are available, segment
bone better than DINOv2 Small? The five Liu samples, test slices only.

Each backbone runs with the app's default settings (the defaults were chosen for Small, so this answers "what does
switching the backbone in the app do", not the best each backbone could do with its own tuning):
  clicks_3_6, clicks_25_25   clean simulated clicks, three seeds each
  learned                    the learned model from 5 labelled development slices of the same sample; it chooses its
                             variant and calibrates its threshold by cross-validation, so it adapts to each backbone
Also records the embedding time per slice. Writes paper/results/backbones.csv and backbones_summary.md.
Usage: python paper/backbones_evaluate.py [--backbones dinov2_s14 dinov2_b14 dinov2_l14]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import hier_boot
from common import available_samples, bone_measures, load_sample, simulated_clicks
from boneseg import metrics
from boneseg.backbone import backbone_label, get_backbone
from boneseg.head import segment_with_head, train_head
from boneseg.segment import SegmentationSettings, embed_image, segment

OUT = Path(__file__).resolve().parent / "results"
BUDGETS = [(3, 6), (25, 25)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="*", default=["dinov2_s14", "dinov2_b14", "dinov2_l14"])
    ap.add_argument("--append", action="store_true", help="Keep the rows of backbones not run now")
    args = ap.parse_args(argv)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
    samples = {n: load_sample(n) for n in available_samples()}
    rows = []
    if args.append and (OUT / "backbones.csv").exists():
        rows = [r for r in pd.read_csv(OUT / "backbones.csv").to_dict("records") if r["backbone"] not in args.backbones]

    def add(bb, sl, method, pred, seed, secs):
        pm, gm = bone_measures(pred, sl.pixel_um), bone_measures(sl.gt, sl.pixel_um)
        rows.append({"backbone": bb, "sample": sl.sample, "z": sl.z, "method": method, "seed": seed, "dice": metrics.dice(pred, sl.gt),
                     "bar_bias": pm["B.Ar/T.Ar_%"] - gm["B.Ar/T.Ar_%"], "embed_s": secs})

    for name in args.backbones:
        st = SegmentationSettings(backbone=name)
        bb = get_backbone(name)
        for s in samples.values():
            head = train_head([(embed_image(bb, sl.img, st), sl.gt) for sl in s.dev[:5]], st, [(0, i) for i in range(5)],
                              pixel_um=s.test[0].pixel_um)
            for sl in s.test:
                t = time.time()
                emb = embed_image(bb, sl.img, st)
                secs = time.time() - t
                add(name, sl, "learned", segment_with_head(head, emb, st, sl.pixel_um).mask, 0, secs)
                for (n_pos, n_neg) in BUDGETS:
                    for seed in range(3):
                        pos, neg = simulated_clicks(sl.gt, n_pos, n_neg, seed + 100)
                        add(name, sl, f"clicks_{n_pos}_{n_neg}", segment(emb, pos, neg, st, sl.pixel_um).mask, seed, secs)
            log(f"{name} {s.name} done (learned-model threshold {head.threshold:.2f})")
            pd.DataFrame(rows).to_csv(OUT / "backbones.csv", index=False)
    summarize()
    log("done")


def summarize():
    df = pd.read_csv(OUT / "backbones.csv")
    d = df.groupby(["backbone", "method", "sample", "z"], as_index=False)[["dice", "bar_bias", "embed_s"]].mean()
    samples = list(dict.fromkeys(d["sample"]))
    lines = ["# Larger and newer backbones\n", f"Liu samples {', '.join(samples)}, test slices; app default settings for every backbone. "
             "Dice with 95% intervals (hierarchical bootstrap), bone-area bias in percentage points, and the time to compute "
             "one slice's features on this Mac.\n"]
    for method, label in (("clicks_3_6", "3 + 6 clicks"), ("clicks_25_25", "25 + 25 clicks"), ("learned", "Learned model, 5 labelled slices")):
        lines += [f"\n## {label}\n", "| Backbone | " + " | ".join(samples) + " | Mean (95% CI) | Bias | Seconds per slice | vs Small |",
                  "|---|" + "---|" * len(samples) + "---|---|---|---|"]
        g = d[d.method == method]
        small = g[g.backbone == "dinov2_s14"].set_index(["sample", "z"])["dice"]
        for bb in dict.fromkeys(g.backbone):
            x = g[g.backbone == bb]
            e, lo, hi = hier_boot(x, "dice")
            per = x.groupby("sample")["dice"].mean()
            diff = (x.set_index(["sample", "z"])["dice"] - small).dropna().reset_index().rename(columns={"dice": "diff"})
            vs = "–" if bb == "dinov2_s14" or diff.empty else "{:+.3f} ({:+.3f} to {:+.3f})".format(*hier_boot(diff, "diff"))
            lines.append(f"| {backbone_label(bb)} | " + " | ".join(f"{per.get(s, np.nan):.3f}" for s in samples)
                         + f" | {e:.3f} ({lo:.3f}–{hi:.3f}) | {x['bar_bias'].mean():+.1f} | {x['embed_s'].mean():.1f} | {vs} |")
    (OUT / "backbones_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
