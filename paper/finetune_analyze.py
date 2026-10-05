"""Summarizes paper/results/finetune.csv into results/finetune_summary.md: Dice per sample, paired differences
(fine-tuned minus frozen, same slices and clicks) with hierarchical bootstrap CIs, and the bias of B.Ar/T.Ar."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import hier_boot

RES = Path(__file__).resolve().parent / "results"
PAIRS = [("labels from same sample", "finetune_b4", "probe_frozen"), ("labels from other samples", "finetune_b4", "probe_frozen"),
         ("3+6 clicks, clean", "clicks_finetuned_b4", "clicks_frozen"), ("25+25 clicks, clean", "clicks_finetuned_b4", "clicks_frozen")]
LABELS = {"probe_frozen": "Frozen DINOv2 + trained output layer", "finetune_b4": "Fine-tuned DINOv2 (last 4 blocks) + output layer",
          "clicks_frozen": "Clicks on frozen DINOv2 features", "clicks_finetuned_b4": "Clicks on fine-tuned DINOv2 features"}


def main():
    df = pd.read_csv(RES / "finetune.csv")
    df = df.groupby(["sample", "z", "method", "condition"], as_index=False).mean(numeric_only=True)   # Seeds averaged per slice
    samples = sorted(df["sample"].unique())
    lines = ["# Fine-tuning DINOv2: results\n",
             f"Samples {', '.join(samples)}, leave-one-sample-out, test slices only. Click conditions use the backbone fine-tuned "
             "on the other samples, so the held-out sample was never seen in training.\n",
             "| Condition | Method | " + " | ".join(samples) + " | Mean (95% CI) | B.Ar/T.Ar bias (points) |", "|---|---|" + "---|" * len(samples) + "---|---|"]
    for cond, a, b in PAIRS:
        for m in (b, a):
            sub = df[(df["condition"] == cond) & (df["method"] == m)]
            if sub.empty:
                continue
            per = sub.groupby("sample")["dice"].mean()
            est, lo, hi = hier_boot(sub, "dice")
            bias = (sub["pred_B.Ar/T.Ar_%"] - sub["gt_B.Ar/T.Ar_%"]).mean()
            lines.append(f"| {cond} | {LABELS[m]} | " + " | ".join(f"{per.get(s, np.nan):.3f}" for s in samples) + f" | {est:.3f} ({lo:.3f}–{hi:.3f}) | {bias:+.1f} |")
    lines += ["\n## Fine-tuned minus frozen, paired on the same slices\n", "| Condition | Difference in Dice (95% CI) | Slices improved |", "|---|---|---|"]
    for cond, a, b in PAIRS:
        A = df[(df["condition"] == cond) & (df["method"] == a)].set_index(["sample", "z"])["dice"]
        B = df[(df["condition"] == cond) & (df["method"] == b)].set_index(["sample", "z"])["dice"]
        d = (A - B).dropna().reset_index().rename(columns={"dice": "diff"})
        if d.empty:
            continue
        est, lo, hi = hier_boot(d, "diff")
        lines.append(f"| {cond} | {est:+.3f} ({lo:+.3f} to {hi:+.3f}) | {(d['diff'] > 0).mean():.0%} |")
    runs = json.loads((RES / "finetune_runs.json").read_text()) if (RES / "finetune_runs.json").exists() else []
    if runs:
        lines += ["\n## Training\n", "| Held out | Labels from | Method | Best validation Dice | At step |", "|---|---|---|---|---|"]
        lines += [f"| {r['held_out']} | {r['source']} | {r['method']} | {r['best_val_dice']:.3f} | {r['best_step']} |" for r in runs]
    (RES / "finetune_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
