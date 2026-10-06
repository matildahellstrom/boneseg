"""Summarizes paper/results/segpc_slices.csv into results/segpc_summary.md and segpc_summary.csv.

Per method and condition, on the 200 validation images (click seeds averaged per image): the official challenge
score with a bootstrap 95% CI over images, cells matched at IoU >= 0.5 (precision is a lower bound, since
unoutlined cells count as errors), Dice near the outlined cells, the per-image cell count against the experts,
nucleus and cytoplasm Dice for the several-structure mode, and the official score per camera. Paired differences
against boneseg use the same images and clicks.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analyze import icc_a1

RES = Path(__file__).resolve().parent / "results"
LABELS = {"dino_clicks": "boneseg, clicks", "dino_multi": "boneseg, nucleus + cytoplasm clicks", "sam": "SAM ViT-B, point prompts",
          "microsam": "micro-SAM ViT-B LM, point prompts", "rf_clicks": "Random forest, clicks (ilastik-style)", "otsu": "Otsu threshold",
          "dino_labels": "boneseg learned model, 5 labelled images"}
ORDER = ["dino_clicks", "dino_multi", "sam", "microsam", "rf_clicks", "otsu", "dino_labels"]


def boot(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    b = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    raw = pd.read_csv(RES / "segpc_slices.csv")
    df = raw.groupby(["image", "camera", "method", "condition"], as_index=False).mean(numeric_only=True)
    cams = sorted(df["camera"].unique())
    lines = ["# Plasma cells on SegPC-2021\n",
             f"Settings tuned on 60 training images; scored on the {df['image'].nunique()} validation images "
             f"({', '.join(f'{(df.drop_duplicates('image')['camera'] == c).sum()} {c}' for c in cams)}). Official score: mean over "
             "expert cells of the best IoU of any predicted whole cell (extra cells are not penalized). Precision counts "
             "unoutlined cells as errors, so it is a lower bound.\n",
             "| Method | Condition | Official score (95% CI) | " + " | ".join(f"Official, {c}" for c in cams) +
             " | Precision | Recall | F1 | Dice near cells | Nucleus Dice | Cytoplasm Dice | Count bias | Count ICC |",
             "|---|---|---|" + "---|" * len(cams) + "---|---|---|---|---|---|---|---|"]
    rows = []
    for cond in sorted(df["condition"].unique(), key=lambda c: ("labelled" in c, c != "no input", c)):
        for m in ORDER:
            g = df[(df["method"] == m) & (df["condition"] == cond)]
            if g.empty:
                continue
            r = raw[(raw["method"] == m) & (raw["condition"] == cond)]
            tp, fp, fn = r["tp"].sum(), r["fp"].sum(), r["fn"].sum()
            est, lo, hi = boot(g["official"])
            nuc = g["dice_nucleus"].mean() if "dice_nucleus" in g and g["dice_nucleus"].notna().any() else np.nan
            cyt = g["dice_cytoplasm"].mean() if "dice_cytoplasm" in g and g["dice_cytoplasm"].notna().any() else np.nan
            row = {"method": m, "condition": cond, "official": est, "lo": lo, "hi": hi, "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
                   "f1": 2 * tp / max(1, 2 * tp + fp + fn), "dice_local": g["dice_local"].mean(), "dice_nucleus": nuc, "dice_cytoplasm": cyt,
                   "count_bias": (g["n_pred"] - g["n_cells"]).mean(), "count_icc": icc_a1(g["n_pred"].to_numpy(float), g["n_cells"].to_numpy(float)),
                   **{f"official_{c}": g[g["camera"] == c]["official"].mean() for c in cams}}
            rows.append(row)
            fmt = lambda v: "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"  # noqa: E731
            lines.append(f"| {LABELS[m]} | {cond} | {est:.3f} ({lo:.3f}–{hi:.3f}) | " + " | ".join(fmt(row[f'official_{c}']) for c in cams) +
                         f" | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['dice_local']:.3f} | {fmt(nuc)} | {fmt(cyt)} | "
                         f"{row['count_bias']:+.2f} | {row['count_icc']:.2f} |")
    pd.DataFrame(rows).to_csv(RES / "segpc_summary.csv", index=False)
    lines += ["\n## Paired differences in the official score against boneseg, same images and clicks\n",
              "| Compared with | Condition | Difference (95% CI) | Images where boneseg is better |", "|---|---|---|---|"]
    ref = df[df["method"] == "dino_clicks"].set_index(["image", "condition"])["official"]
    for m in ("sam", "microsam", "rf_clicks", "dino_multi"):
        for cond in sorted(df.loc[df["method"] == m, "condition"].unique()):
            o = df[(df["method"] == m) & (df["condition"] == cond)].set_index(["image", "condition"])["official"]
            d = (ref - o).dropna()
            if len(d):
                est, lo, hi = boot(d)
                lines.append(f"| {LABELS[m]} | {cond} | {est:+.3f} ({lo:+.3f} to {hi:+.3f}) | {(d > 0).mean():.0%} |")
    (RES / "segpc_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
