"""Robustness of boneseg under realistic image changes, on the augmented Liu test slices (make_augmented.py).

For every item (5 samples x 10 test slices x 13 augmentations x 3 levels, plus the clean slices):
  clicks_3_6, clicks_25_25   boneseg with clean simulated clicks (three seeds, as evaluate.py), placed on the
                             item's own mask, so clicks follow the image a user would see
  learned                    boneseg's learned model trained on 5 CLEAN development slices of the same sample
                             (calibrated threshold): does a model trained on normal images cope with changed ones?
  sam_25_25                  SAM ViT-B with the same 25 + 25 clicks, as the reference
The result is how much each change costs (Dice and bone-area bias against the clean slice), which is robustness, not
accuracy on new samples: every item is derived from the same 50 test slices.

Writes paper/results/robustness.csv (resumable: finished sample/augmentation/level groups are skipped) and
robustness_summary.md, figures/robustness.png.
Run from paper/:  python robustness/robustness_evaluate.py [--samples A C] [--augs blur gamma_hi] [--no-sam]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # paper/ first, so `common` is paper/common.py

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import methods as M  # noqa: E402
from analyze import hier_boot  # noqa: E402
from common import available_samples, bone_measures, load_sample, simulated_clicks  # noqa: E402
from boneseg import metrics  # noqa: E402
from boneseg.backbone import get_backbone  # noqa: E402
from boneseg.head import segment_with_head, train_head  # noqa: E402
from boneseg.segment import SegmentationSettings, embed_image, segment  # noqa: E402
from robustness.common import load_augmented  # noqa: E402

OUT = HERE.parent / "results"
FIG = HERE.parent / "figures"
BUDGETS = [(3, 6), (25, 25)]
SEEDS = range(3)
METHODS = {"clicks_3_6": "boneseg, 3 + 6 clicks", "clicks_25_25": "boneseg, 25 + 25 clicks",
           "learned": "boneseg learned model (trained on clean slices)", "sam_25_25": "SAM ViT-B, 25 + 25 clicks"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="*", default=None)
    ap.add_argument("--augs", nargs="*", default=None, help="Augmentations to run (default: all, plus clean)")
    ap.add_argument("--no-sam", action="store_true")
    ap.add_argument("--out", default="robustness")
    args = ap.parse_args(argv)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
    st = SegmentationSettings()
    bb = get_backbone(st.backbone)
    path = OUT / f"{args.out}.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(r["sample"], r["augmentation"], int(r["level"])) for r in rows}   # Groups already finished

    for name in args.samples or available_samples():
        items = load_augmented(samples=name, augmentation=(["clean"] + args.augs) if args.augs else None)
        todo = [it for it in items if (it.sample, it.augmentation, int(it.level)) not in done]
        if not todo:
            continue
        # The learned model sees only clean development slices: the same model for every augmentation
        dev = load_sample(name).dev[:5]
        head = train_head([(embed_image(bb, sl.img, st), sl.gt) for sl in dev], st, [(0, i) for i in range(5)], pixel_um=dev[0].pixel_um)
        log(f"{name}: {len(todo)} items, learned-model threshold {head.threshold:.2f}")
        for i, it in enumerate(todo):
            img, gt = it.img, it.gt   # Read from disk once
            emb = embed_image(bb, img, st)
            g_meas = bone_measures(gt, it.pixel_um)

            def add(method, pred, seed=0):
                rows.append({"sample": it.sample, "z": it.z, "augmentation": it.augmentation, "level": int(it.level), "method": method,
                             "seed": seed, "dice": metrics.dice(pred, gt),
                             "bar_bias": bone_measures(pred, it.pixel_um)["B.Ar/T.Ar_%"] - g_meas["B.Ar/T.Ar_%"]})

            add("learned", segment_with_head(head, emb, st, it.pixel_um).mask)
            for (n_pos, n_neg) in BUDGETS:
                for seed in SEEDS:
                    pos, neg = simulated_clicks(gt, n_pos, n_neg, seed + 100)
                    add(f"clicks_{n_pos}_{n_neg}", segment(emb, pos, neg, st, it.pixel_um).mask, seed)
                    if (n_pos, n_neg) == (25, 25) and not args.no_sam:
                        add("sam_25_25", M.sam_points("sam", (it.sample, it.z, it.augmentation, it.level), img, pos, neg), seed)
            if (i + 1) % 50 == 0:
                log(f"  {name}: {i + 1} of {len(todo)}")
                pd.DataFrame(rows).to_csv(path, index=False)
        pd.DataFrame(rows).to_csv(path, index=False)
        log(f"{name} done")
    summarize(args.out)
    log("done")


def summarize(name="robustness"):
    df = pd.read_csv(OUT / f"{name}.csv")
    # One value per slice: mean over click seeds
    d = df.groupby(["sample", "z", "augmentation", "level", "method"], as_index=False)[["dice", "bar_bias"]].mean()
    clean = d[d.augmentation == "clean"].set_index(["sample", "z", "method"])[["dice", "bar_bias"]]
    lines = ["# Robustness to image changes\n",
             "The 50 Liu test slices, each changed by 13 augmentations at three levels (paper/robustness/README.md). Dice on the "
             "changed slice and its drop against the same slice unchanged (95% interval, hierarchical bootstrap over samples and "
             "slices). The learned model was trained on clean slices only. This measures robustness, not accuracy on new samples.\n"]
    for method, label in METHODS.items():
        g = d[d.method == method]
        if g.empty:
            continue
        c = clean.xs(method, level="method")
        lines += [f"\n## {label}\n", f"Clean slices: Dice {g[g.augmentation == 'clean']['dice'].mean():.3f}\n",
                  "| Augmentation | Level 1 | Level 2 | Level 3 | Bias at level 3 (points) |", "|---|---|---|---|---|"]
        for aug in sorted(set(g.augmentation) - {"clean"}):
            cells = []
            for level in (1, 2, 3):
                x = g[(g.augmentation == aug) & (g.level == level)].set_index(["sample", "z"])
                diff = (x["dice"] - c["dice"]).dropna().reset_index().rename(columns={"dice": "diff"})
                if diff.empty:
                    cells.append("–")
                    continue
                e, lo, hi = hier_boot(diff, "diff", n=1000)
                cells.append(f"{x['dice'].mean():.3f} ({e:+.3f}{' *' if hi < 0 else ''})")
            b3 = g[(g.augmentation == aug) & (g.level == 3)]["bar_bias"].mean()
            lines.append(f"| {aug} | " + " | ".join(cells) + f" | {b3:+.1f} |")
    lines.append("\nIn brackets: change against the clean slice; * marks a drop whose 95% interval excludes zero.")
    (OUT / f"{name}_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    _figure(d, clean, name)


def _figure(d, clean, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    augs = sorted(set(d.augmentation) - {"clean"})
    methods = [m for m in METHODS if m in set(d.method)]
    fig, axes = plt.subplots(1, len(methods), figsize=(4.2 * len(methods), 5), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), methods):
        c = clean.xs(m, level="method")["dice"]
        mat = np.array([[(d[(d.method == m) & (d.augmentation == a) & (d.level == lv)].set_index(["sample", "z"])["dice"] - c).mean()
                         for lv in (1, 2, 3)] for a in augs])
        im = ax.imshow(mat, cmap="RdBu", vmin=-0.3, vmax=0.3, aspect="auto")
        ax.set_xticks(range(3), ["L1", "L2", "L3"])
        ax.set_yticks(range(len(augs)), augs, fontsize=8)
        ax.set_title(METHODS[m], fontsize=9)
        for (r, k), v in np.ndenumerate(mat):
            ax.text(k, r, f"{v:+.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=axes, shrink=0.6, label="Dice change against the clean slice")
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    main()
