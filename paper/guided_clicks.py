"""Do click suggestions beat extra random clicks? Three more clicks after 3 + 6, on every test slice of the Liu samples.

Starting from 3 + 6 careful clicks (the evaluation's seeds), a simulated user adds three clicks one at a time:
  missed       boneseg's top "possibly missed bone" region; the user answers with the expert mask: a bone click inside
               the region's bone if at least half of it is bone, otherwise a background click at its suggested point
  uncertainty  the app's existing suggestion (most uncertain place under leave-one-click-out); the user labels it
  random       a random careful click, alternating bone, background, bone
Every strategy ends with 6 + 9 or so clicks in total; if a suggestion method has nothing to suggest, a random click
is used instead and counted. App default settings. Writes paper/results/guided_clicks.csv and guided_clicks_summary.md.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi

from analyze import hier_boot
from common import available_samples, load_sample, simulated_clicks
from boneseg import metrics
from boneseg.backbone import get_backbone
from boneseg.segment import (SegmentationSettings, embed_image, prototypes, sample_points, segment, suggest_click,
                             uncertainty_map)

OUT = Path(__file__).resolve().parent / "results"
EXTRA = 3


def random_click(gt, kind, rng):
    region = ndi.binary_erosion(gt, iterations=10) if kind == "pos" else ~ndi.binary_dilation(gt, iterations=10)
    region = region if region.any() else (gt if kind == "pos" else ~gt)
    ys, xs = np.nonzero(region)
    i = rng.integers(len(ys))
    return (int(ys[i]), int(xs[i]))


def raw_of(res):
    lo, hi = res.raw_score_range
    return lo + res.heat.astype(np.float64) * (hi - lo)


def candidate_region(raw, res, pos, neg, point):
    """The loose-threshold region that holds the suggested point (as in segment.missed_candidates)."""
    neg_scores = sample_points(raw, neg)
    loose = res.raw_threshold - 0.5 * (res.raw_threshold - float(np.median(neg_scores)))
    lab, _ = ndi.label((raw >= loose) & ~ndi.binary_dilation(res.mask, iterations=2))
    return lab == lab[point[0], point[1]]


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
    st = SegmentationSettings()
    bb = get_backbone(st.backbone)
    rows = []
    for name in available_samples():
        s = load_sample(name)
        for sl in s.test:
            emb = embed_image(bb, sl.img, st)
            for seed in range(3):
                pos0, neg0 = simulated_clicks(sl.gt, 3, 6, seed + 100)
                base = segment(emb, pos0, neg0, st, sl.pixel_um)
                for strategy in ("missed", "uncertainty", "random"):
                    rng = np.random.default_rng(1000 + seed)
                    pos, neg = list(pos0), list(neg0)
                    res = base
                    rows.append({"sample": name, "z": sl.z, "seed": seed, "strategy": strategy, "step": 0, "dice": metrics.dice(res.mask, sl.gt), "fallback": False})
                    for step in range(1, EXTRA + 1):
                        fallback = False
                        if strategy == "missed":
                            cands = res.extra.get("missed", [])
                            if cands:
                                pt = tuple(cands[0]["point"])
                                region = candidate_region(raw_of(res), res, pos, neg, pt)
                                inside = region & sl.gt
                                if inside.sum() >= 0.5 * region.sum():
                                    ys, xs = np.nonzero(inside)
                                    j = int(np.argmax(ndi.distance_transform_edt(inside)[ys, xs]))
                                    pos.append((int(ys[j]), int(xs[j])))
                                else:
                                    neg.append(pt)
                            else:
                                fallback = True
                        elif strategy == "uncertainty":
                            u = uncertainty_map(emb, prototypes(emb, pos), prototypes(emb, neg), st, threshold=res.threshold)
                            pt = suggest_click(u, pos + neg)
                            if pt is not None:
                                (pos if sl.gt[pt] else neg).append(pt)
                            else:
                                fallback = True
                        if strategy == "random" or fallback:
                            kind = "pos" if step % 2 == 1 else "neg"
                            (pos if kind == "pos" else neg).append(random_click(sl.gt, kind, rng))
                        res = segment(emb, pos, neg, st, sl.pixel_um)
                        rows.append({"sample": name, "z": sl.z, "seed": seed, "strategy": strategy, "step": step, "dice": metrics.dice(res.mask, sl.gt), "fallback": fallback})
        log(f"{name} done")
        pd.DataFrame(rows).to_csv(OUT / "guided_clicks.csv", index=False)
    summarize()


def summarize():
    df = pd.read_csv(OUT / "guided_clicks.csv")
    g = df.groupby(["sample", "z", "strategy", "step"], as_index=False)["dice"].mean()
    lines = ["# Click suggestions against extra random clicks\n", "Test slices of the Liu samples, starting from 3 + 6 careful clicks; mean Dice after each extra click.\n",
             "| Strategy | 0 extra | 1 extra | 2 extra | 3 extra | Fallback clicks |", "|---|---|---|---|---|---|"]
    for strat in ("missed", "uncertainty", "random"):
        vals = [hier_boot(g[(g.strategy == strat) & (g.step == k)], "dice")[0] for k in range(EXTRA + 1)]
        fb = df[(df.strategy == strat) & (df.step > 0)]["fallback"].mean()
        lines.append(f"| {strat} | " + " | ".join(f"{v:.3f}" for v in vals) + f" | {fb:.0%} |")
    lines += ["\n| After 3 extra clicks | Difference in Dice (95% CI) | Slices better |", "|---|---|---|"]
    end = g[g.step == EXTRA].set_index(["sample", "z", "strategy"])["dice"].unstack("strategy")
    for a, b in (("missed", "random"), ("uncertainty", "random"), ("missed", "uncertainty")):
        d = (end[a] - end[b]).reset_index().rename(columns={0: "diff"})
        d.columns = ["sample", "z", "diff"]
        e, lo, hi = hier_boot(d, "diff")
        lines.append(f"| {a} vs {b} | {e:+.3f} ({lo:+.3f} to {hi:+.3f}) | {(d['diff'] > 0).mean():.0%} |")
    (OUT / "guided_clicks_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
