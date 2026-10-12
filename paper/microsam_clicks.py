"""micro-SAM on the clicks, as the app offers it, on the test slices of the Liu samples with the evaluation's clicks.

  boneseg            boneseg from clicks, app defaults
  microsam           micro-SAM ViT-B LM (vit_b_lm) with the clicks as point prompts: "Use its mask instead"
  microsam_agree     pixels both boneseg and micro-SAM call bone: "Keep only what it agrees on"
  sam, sam_agree     the same with the original SAM ViT-B, for comparison
Same slices, clicks (three seeds, as combine_sam.py) and settings as the earlier SAM comparison.
Writes paper/results/microsam_clicks.csv and microsam_clicks_summary.md (Dice, B.Ar/T.Ar bias, paired differences).
"""
from __future__ import annotations

import itertools
import time
from pathlib import Path

import pandas as pd

import methods as M
from analyze import hier_boot
from common import available_samples, bone_measures, load_sample, simulated_clicks
from boneseg import metrics
from boneseg.segment import SegmentationSettings, segment

OUT = Path(__file__).resolve().parent / "results"
BUDGETS = [(3, 6), (25, 25)]
METHODS = ("boneseg", "microsam", "microsam_agree", "sam", "sam_agree")


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
    st = SegmentationSettings()
    rows = []
    for name in available_samples():
        for sl in load_sample(name).test:
            key = (sl.sample, sl.z)
            emb = M.dino_embedding(key, sl.img, st)
            gtm = bone_measures(sl.gt, sl.pixel_um)
            for budget, seed in itertools.product(BUDGETS, range(3)):
                pos, neg = simulated_clicks(sl.gt, *budget, seed + 100)      # The clicks of combine_sam.py
                b = segment(emb, pos, neg, st, sl.pixel_um).mask
                ms = M.sam_points("microsam", key, sl.img, pos, neg)
                s = M.sam_points("sam", key, sl.img, pos, neg)
                for m, p in {"boneseg": b, "microsam": ms, "microsam_agree": b & ms, "sam": s, "sam_agree": b & s}.items():
                    pm = bone_measures(p, sl.pixel_um)
                    rows.append({"sample": name, "z": sl.z, "budget": f"{budget[0]}+{budget[1]}", "seed": seed, "method": m,
                                 "dice": metrics.dice(p, sl.gt), "bar_bias": pm["B.Ar/T.Ar_%"] - gtm["B.Ar/T.Ar_%"]})
        log(f"{name} done")
        pd.DataFrame(rows).to_csv(OUT / "microsam_clicks.csv", index=False)
    summarize()


def summarize():
    df = pd.read_csv(OUT / "microsam_clicks.csv").groupby(["sample", "z", "budget", "method"], as_index=False).mean(numeric_only=True)
    lines = ["# micro-SAM on the clicks\n", "Test slices of the Liu samples, the evaluation's clicks (three seeds), app default settings.\n",
             "| Method | Budget | Dice (95% CI) | B.Ar/T.Ar bias (points) |", "|---|---|---|---|"]
    budgets = [x for x in ("3+6", "25+25") if x in set(df.budget)]
    for b in budgets:
        for m in METHODS:
            g = df[(df.budget == b) & (df.method == m)]
            e, lo, hi = hier_boot(g, "dice")
            lines.append(f"| {m} | {b} | {e:.3f} ({lo:.3f}–{hi:.3f}) | {g['bar_bias'].mean():+.2f} |")
    lines += ["\n| Comparison | Budget | Difference in Dice (95% CI) | Slices better |", "|---|---|---|---|"]
    for b in budgets:
        w = df[df.budget == b].set_index(["sample", "z", "method"])["dice"].unstack("method")
        for m, ref in (("microsam", "boneseg"), ("microsam_agree", "boneseg"), ("microsam", "sam"), ("microsam_agree", "sam_agree")):
            d = pd.DataFrame({"sample": w.index.get_level_values(0), "diff": (w[m] - w[ref]).to_numpy()})
            e, lo, hi = hier_boot(d, "diff")
            lines.append(f"| {m} vs {ref} | {b} | {e:+.3f} ({lo:+.3f} to {hi:+.3f}) | {(d['diff'] > 0).mean():.0%} |")
    (OUT / "microsam_clicks_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
