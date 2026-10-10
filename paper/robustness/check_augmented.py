"""Checks the augmented dataset written by make_augmented.py and summarises its intensity statistics.

  1. Every image and mask in manifest.csv loads; shapes match; images are float32 in [0, 1].
  2. Photometric augmentations: the mask is bit-for-bit the clean mask.
  3. Geometric augmentations: the clean mask is warped again in float64 without thresholding; the stored mask's area
     must be within 2% of that float area (rot90_flip: exactly the clean area), and the stored mask should equal the
     float warp thresholded at one half (number of differing pixels reported).
  4. Visual check: one grid figure per augmentation (clean and levels 1-3, mask outline in red) for the first test
     slice of sample A (or the first sample), in data/augmented/figures/ (git-ignored: it shows Liu images).
Writes data/augmented/checks.csv (one row per item) and data/augmented/stats.csv (per augmentation and level), and
prints a Markdown table for README.md.
Usage (from paper/): python robustness/check_augmented.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))

from robustness.common import load_augmented, resolve_root  # noqa: E402
from robustness.make_augmented import GEOMETRIC, warp  # noqa: E402


def figure(items, out: Path):
    """Clean slice and levels 1-3 side by side, mask outline in red."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(items), figsize=(4 * len(items), 4.6))
    for ax, it in zip(np.atleast_1d(axes), items):
        ax.imshow(it.img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.contour(it.gt, levels=[0.5], colors="red", linewidths=0.6)
        ax.set_title(f"{it.augmentation} L{it.level}\n{it.params}" if it.level else "clean", fontsize=7)
        ax.axis("off")
    fig.suptitle(f"sample {items[0].sample}, z = {items[0].z}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/augmented")
    args = ap.parse_args()
    root = resolve_root(args.root)
    items = sorted(load_augmented(root), key=lambda it: (it.sample, it.z, it.augmentation != "clean"))
    rows, clean, clean_key = [], None, None
    for i, it in enumerate(items):
        img, mask = it.img, it.gt
        # The clean item of each slice comes first in the sorted list; keep its mask for comparison
        if it.augmentation == "clean":
            clean, clean_key = mask, (it.sample, it.z)
        assert clean is not None and clean_key == (it.sample, it.z), f"no clean item for {it.sample} z={it.z}"
        r = dict(sample=it.sample, z=it.z, augmentation=it.augmentation, level=it.level,
                 shape_ok=img.shape == mask.shape, dtype_ok=img.dtype == np.float32,
                 range_ok=bool(img.min() >= 0 and img.max() <= 1), mean=float(img.mean()), std=float(img.std()),
                 bone_mean=float(img[mask].mean()), bg_mean=float(img[~mask].mean()), area=int(mask.sum()))
        if it.augmentation == "clean" or it.augmentation not in GEOMETRIC:
            # Photometric: the mask must not change at all
            r["mask_identical"] = bool(np.array_equal(mask, clean))
        else:
            # Geometric: warp the clean mask in float64 and compare areas
            ref = np.clip(warp(clean.astype(np.float64), it.augmentation, it.params, order=1), 0, 1)
            r["float_area"] = float(ref.sum())
            r["area_rel_diff"] = (r["area"] - r["float_area"]) / r["float_area"]
            r["pixels_differ_from_float_threshold"] = int(((ref >= 0.5) != mask).sum())
            if it.augmentation == "rot90_flip":
                r["area_exact"] = r["area"] == int(clean.sum())
        rows.append(r)
        if i % 200 == 0:
            print(f"{i}/{len(items)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(root / "checks.csv", index=False)

    # Pass/fail summary
    geo = df[df.augmentation.isin(GEOMETRIC)]
    pho = df[~df.augmentation.isin(GEOMETRIC)]
    print(f"items: {len(df)}; files load, shapes match: {df.shape_ok.all()}; float32: {df.dtype_ok.all()}; "
          f"in [0, 1]: {df.range_ok.all()}")
    print(f"photometric and clean masks identical to the clean mask: {pho.mask_identical.sum()}/{len(pho)}")
    print(f"geometric |area - float area| / float area: max {geo.area_rel_diff.abs().max():.4%}, "
          f"within 2%: {(geo.area_rel_diff.abs() <= 0.02).sum()}/{len(geo)}")
    print(geo.groupby(["augmentation", "level"]).agg(max_abs_rel_diff=("area_rel_diff", lambda v: v.abs().max()),
                                                    max_px_differ=("pixels_differ_from_float_threshold", "max")))
    rf = df[df.augmentation == "rot90_flip"]
    print(f"rot90_flip area exactly equal: {rf.area_exact.sum()}/{len(rf)}")

    # Intensity statistics per augmentation and level (averaged over items)
    st = df.groupby(["augmentation", "level"])[["mean", "std", "bone_mean", "bg_mean"]].mean().round(3)
    st["n"] = df.groupby(["augmentation", "level"]).size()
    st.to_csv(root / "stats.csv")
    print("\n| augmentation | level | mean | std | bone mean | background mean |\n|---|---|---|---|---|---|")
    for (a, l), s in st.iterrows():
        print(f"| {a} | {l} | {s['mean']:.3f} | {s['std']:.3f} | {s.bone_mean:.3f} | {s.bg_mean:.3f} |")

    # Visual check figures for one slice
    figs = root / "figures"
    figs.mkdir(exist_ok=True)
    first = "A" if any(it.sample == "A" for it in items) else items[0].sample
    z0 = min(it.z for it in items if it.sample == first)
    one = [it for it in items if it.sample == first and it.z == z0]
    c = [it for it in one if it.augmentation == "clean"]
    for aug in sorted({it.augmentation for it in one} - {"clean"}):
        figure(c + sorted([it for it in one if it.augmentation == aug], key=lambda it: it.level), figs / f"{aug}.png")
    print(f"figures: {figs}")


if __name__ == "__main__":
    main()
