"""Summarizes the whole-stack 3D evaluation (results/stack3d.csv, stack3d_slices.csv) into results/stack3d_summary.md
and figures/stack3d_profiles.png: 3D measures per sample against the expert, their relative differences, and how the
bone area per slice follows the expert through each stack (Pearson r, mean per-slice Dice)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
LABELS = {"clicks": "boneseg, 25 + 25 clicks on one slice", "labels_same": "boneseg learned model, 5 slices of the sample",
          "labels_same_cal": "  + calibrated threshold", "labels_same_cal_zs": "  + calibrated, smoothed along z",
          "labels_same_zf": "  + calibrated, features of neighbouring slices", "labels_same_zf_zs": "  + calibrated, neighbouring slices, smoothed",
          "labels_other": "boneseg learned model, slices of other samples", "labels_other_cal": "  + calibrated threshold"}
MEAS = ["BV/TV_%", "BS/BV_per_mm", "Tb.Th_um", "Tb.N_per_mm", "Tb.Sp_um"]


def main():
    df = pd.read_csv(RES / "stack3d.csv")
    sl = pd.read_csv(RES / "stack3d_slices.csv")
    samples = list(dict.fromkeys(df["sample"]))
    lines = ["# Whole stacks in 3D\n", "Central 80% of each stack; measures on volumes reduced to 4 x 6.5 x 6.5 um voxels, the same for "
             "method and expert. Tb.Th, Tb.N and Tb.Sp use the plate model (Tb.Th = 2 BV/BS).\n",
             "| Sample | Method | 3D Dice | " + " | ".join(MEAS) + " | Slice r | Mean slice Dice |", "|---|---|---|" + "---|" * len(MEAS) + "---|---|"]
    rel = []
    for s in samples:
        e = df[(df["sample"] == s) & (df["method"] == "expert")].iloc[0]
        lines.append(f"| {s} ({int(e['n_slices'])} slices) | expert | – | " + " | ".join(f"{e[m]:.3g}" for m in MEAS) + " | – | – |")
        g = sl[sl["sample"] == s]
        for meth in LABELS:
            r = df[(df["sample"] == s) & (df["method"] == meth)]
            if r.empty:
                continue
            r = r.iloc[0]
            corr = np.corrcoef(g[f"area_{meth}"], g["area_expert"])[0, 1]
            cells = [f"{r[m]:.3g} ({100 * (r[m] - e[m]) / e[m]:+.0f}%)" for m in MEAS]
            lines.append(f"| {s} | {LABELS[meth]} | {r['dice_3d']:.3f} | " + " | ".join(cells) + f" | {corr:.2f} | {g[f'dice_{meth}'].mean():.3f} |")
            rel.append({"sample": s, "method": meth, "dice_3d": r["dice_3d"], "slice_r": corr, **{m: 100 * (r[m] - e[m]) / e[m] for m in MEAS}})
    rel = pd.DataFrame(rel)
    lines += ["\n## Mean over samples\n", "| Method | 3D Dice | " + " | ".join(f"{m} difference" for m in MEAS) + " | Slice r |", "|---|---|" + "---|" * len(MEAS) + "---|"]
    for meth, g in rel.groupby("method", sort=False):
        lines.append(f"| {LABELS[meth]} | {g['dice_3d'].mean():.3f} | " + " | ".join(f"{g[m].mean():+.0f}% (|{g[m].abs().mean():.0f}%|)" for m in MEAS) +
                     f" | {g['slice_r'].mean():.2f} |")
    (RES / "stack3d_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    FIG.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, len(samples), figsize=(3.6 * len(samples), 3.0), sharey=False)
    for ax, s in zip(np.atleast_1d(axes), samples):
        g = sl[sl["sample"] == s]
        ax.plot(g["z"], g["area_expert"], color="k", lw=1.6, label="Expert")
        for meth, col in (("clicks", "#008a87"), ("labels_same", "#c27c0e"), ("labels_same_zf", "#7a4fc9")):
            if f"area_{meth}" in g:
                ax.plot(g["z"], g[f"area_{meth}"], color=col, lw=1.1, label=LABELS[meth].strip(" +").replace("boneseg, ", "").replace("boneseg ", ""))
        ax.set_title(f"Sample {s}", fontsize=9)
        ax.set_xlabel("Slice", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("Bone area per slice (%)", fontsize=8)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIG / "stack3d_profiles.png", dpi=170)


if __name__ == "__main__":
    main()
