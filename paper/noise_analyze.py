"""Summarizes paper/results/noise_slices.csv into results/noise_summary.md.

Per method and condition: Dice (mean of per-batch means, hierarchical bootstrap CI over batches and patches),
cell detection at IoU >= 0.5 pooled per batch (precision, recall, F1; mean over batches), and agreement of the
per-patch osteoclast count and area fraction with the experts (bias, 95% limits of agreement, ICC(A,1), r).
Paired Dice differences against boneseg use the same patches and clicks.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analyze import hier_boot, icc_a1

RES = Path(__file__).resolve().parent / "results"
LABELS = {"otsu": "Otsu threshold", "rf_clicks": "Random forest, clicks (ilastik-style)", "sam": "SAM ViT-B, point prompts",
          "microsam": "micro-SAM ViT-B LM, point prompts", "dino_clicks": "boneseg, clicks",
          "rf_labels": "Random forest, 5 labelled patches", "dino_labels": "boneseg learned model, 5 labelled patches"}
ORDER = ["dino_clicks", "sam", "microsam", "rf_clicks", "otsu", "dino_labels", "rf_labels"]


def detection(g: pd.DataFrame) -> dict:
    per = []
    for _, b in g.groupby("batch"):
        tp, fp, fn = b["tp"].sum(), b["fp"].sum(), b["fn"].sum()
        per.append((tp / max(1, tp + fp), tp / max(1, tp + fn), 2 * tp / max(1, 2 * tp + fp + fn)))
    p, r, f = np.mean(per, axis=0)
    return {"precision": p, "recall": r, "f1": f, "f1_min": min(x[2] for x in per), "f1_max": max(x[2] for x in per)}


def agreement(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    d = x[ok] - y[ok]
    return d.mean(), d.mean() - 1.96 * d.std(ddof=1), d.mean() + 1.96 * d.std(ddof=1), icc_a1(x, y), np.corrcoef(x[ok], y[ok])[0, 1]


def main():
    raw = pd.read_csv(RES / "noise_slices.csv")
    # Seeds averaged per patch for Dice, counts and areas; detection counts are summed over seeds within a batch
    df = raw.groupby(["batch", "patch", "method", "condition"], as_index=False).mean(numeric_only=True).rename(columns={"batch": "sample"})
    batches = sorted(df["sample"].unique())
    lines = ["# Osteoclasts on the NOISe mouse data\n",
             f"Batches {', '.join(batches)}, 20 test patches each (832 x 832 px brightfield, TRAP stain), leave-one-batch-out: "
             "every setting tuned on the other four batches. Cells count as found when a predicted cell overlaps an expert "
             "outline with IoU >= 0.5, one-to-one. Counts and areas are per patch.\n",
             "| Method | Condition | Dice (95% CI) | Precision | Recall | F1 (range over batches) | Count bias | Count ICC | Area bias (points) | Area ICC |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for cond in sorted(df["condition"].unique(), key=lambda c: ("labels" in c, c != "no input", c)):
        for m in ORDER:
            g = df[(df["method"] == m) & (df["condition"] == cond)]
            if g.empty:
                continue
            est, lo, hi = hier_boot(g, "dice")
            det = detection(raw[(raw["method"] == m) & (raw["condition"] == cond)])
            cb, _, _, cicc, _ = agreement(g["n_pred"].to_numpy(float), g["n_gt"].to_numpy(float))
            ab, _, _, aicc, _ = agreement(g["area_pred"].to_numpy(float), g["area_gt"].to_numpy(float))
            rows.append({"method": m, "condition": cond, "dice": est, "dice_lo": lo, "dice_hi": hi, **det, "count_bias": cb, "count_icc": cicc,
                         "area_bias": ab, "area_icc": aicc, **{f"dice_{b}": g[g["sample"] == b]["dice"].mean() for b in batches}})
            lines.append(f"| {LABELS[m]} | {cond} | {est:.3f} ({lo:.3f}–{hi:.3f}) | {det['precision']:.3f} | {det['recall']:.3f} | "
                         f"{det['f1']:.3f} ({det['f1_min']:.2f}–{det['f1_max']:.2f}) | {cb:+.2f} | {cicc:.2f} | {ab:+.2f} | {aicc:.2f} |")
    pd.DataFrame(rows).to_csv(RES / "noise_summary.csv", index=False)
    lines += ["\n## Paired differences in Dice against boneseg, same patches and clicks\n", "| Compared with | Condition | Difference (95% CI) | Wins |", "|---|---|---|---|"]
    ref = df[df["method"] == "dino_clicks"].set_index(["sample", "patch", "condition"])["dice"]
    for m in ("sam", "microsam", "rf_clicks"):
        for cond in sorted(df.loc[df["method"] == m, "condition"].unique()):
            o = df[(df["method"] == m) & (df["condition"] == cond)].set_index(["sample", "patch", "condition"])["dice"]
            d = (ref - o).dropna().reset_index().rename(columns={"dice": "diff"})
            if not d.empty:
                est, lo, hi = hier_boot(d, "diff")
                lines.append(f"| {LABELS[m]} | {cond} | {est:+.3f} ({lo:+.3f} to {hi:+.3f}) | {(d['diff'] > 0).mean():.0%} |")
    lines += ["\n## Dice per batch\n", "| Method | Condition | " + " | ".join(batches) + " |", "|---|---|" + "---|" * len(batches)]
    for r in rows:
        lines.append(f"| {LABELS[r['method']]} | {r['condition']} | " + " | ".join(f"{r[f'dice_{b}']:.3f}" for b in batches) + " |")
    (RES / "noise_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
