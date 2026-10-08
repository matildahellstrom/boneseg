"""boneseg and SAM together, on the test slices of the Liu samples with the evaluation's clicks.

  boneseg        boneseg from clicks, app defaults
  sam            SAM ViT-B with the clicks as point prompts (as in the evaluation)
  sam_boxes      SAM prompted per boneseg region: one box around each region of the boneseg mask, with the clicks
                 that fall inside it; the regions' masks are joined. Uses SAM's edges and boneseg's "where"
  sam_maskprompt SAM with the clicks and boneseg's mask as its mask prompt
  intersection   pixels both boneseg and SAM call bone
  union          pixels either calls bone
Writes paper/results/combine_sam.csv and combine_sam_summary.md (Dice and B.Ar/T.Ar bias, paired differences).
"""
from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import torch.nn.functional as F
import torch

import methods as M
from analyze import hier_boot
from common import available_samples, bone_measures, load_sample, simulated_clicks
from boneseg import metrics
from boneseg.segment import SegmentationSettings, segment

OUT = Path(__file__).resolve().parent / "results"
BUDGETS = [(3, 6), (25, 25)]


def sam_low_res(prob: np.ndarray, pred) -> np.ndarray:
    """A probability map in SAM's 256 x 256 low-resolution mask-prompt frame, as logits."""
    h, w = prob.shape
    s = 1024 / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    t = torch.from_numpy(prob.astype(np.float32))[None, None]
    t = F.interpolate(t, (nh, nw), mode="bilinear", align_corners=False)
    t = F.pad(t, (0, 1024 - nw, 0, 1024 - nh))
    t = F.interpolate(t, (256, 256), mode="bilinear", align_corners=False)[0]
    p = t.clamp(1e-4, 1 - 1e-4)
    return torch.log(p / (1 - p)).numpy() * 4.0


def sam_boxes(pred, mask, pos, neg, min_frac=5e-4):
    lab, n = ndi.label(mask)
    out = np.zeros_like(mask)
    for k, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None or (lab[sl] == k).sum() < min_frac * mask.size:
            continue
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        inside = lambda pts: [(y, x) for y, x in pts if y0 <= y < y1 and x0 <= x < x1]  # noqa: E731
        p, q = inside(pos), inside(neg)
        kw = {"box": np.array([x0, y0, x1, y1], np.float32), "multimask_output": False}
        if p or q:
            kw["point_coords"] = np.array([(x, y) for y, x in p + q], np.float32)
            kw["point_labels"] = np.r_[np.ones(len(p)), np.zeros(len(q))].astype(np.int32)
        m, _, _ = pred.predict(**kw)
        out |= m[0].astype(bool)
    return out


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
                pos, neg = simulated_clicks(sl.gt, *budget, seed + 100)
                r = segment(emb, pos, neg, st, sl.pixel_um)
                b = r.mask
                s = M.sam_points("sam", key, sl.img, pos, neg)
                pred = M._sam_predictor("sam")
                pts = np.array([(x, y) for y, x in list(pos) + list(neg)], np.float32)
                lbl = np.r_[np.ones(len(pos)), np.zeros(len(neg))].astype(np.int32)
                mp, _, _ = pred.predict(point_coords=pts, point_labels=lbl, mask_input=sam_low_res(b.astype(np.float32), pred), multimask_output=False)
                preds = {"boneseg": b, "sam": s, "sam_boxes": sam_boxes(pred, b, pos, neg), "sam_maskprompt": mp[0].astype(bool),
                         "intersection": b & s, "union": b | s}
                for m, p in preds.items():
                    pm = bone_measures(p, sl.pixel_um)
                    rows.append({"sample": name, "z": sl.z, "budget": f"{budget[0]}+{budget[1]}", "seed": seed, "method": m,
                                 "dice": metrics.dice(p, sl.gt), "bar_bias": pm["B.Ar/T.Ar_%"] - gtm["B.Ar/T.Ar_%"]})
        log(f"{name} done")
        pd.DataFrame(rows).to_csv(OUT / "combine_sam.csv", index=False)
    summarize()


def summarize():
    df = pd.read_csv(OUT / "combine_sam.csv").groupby(["sample", "z", "budget", "method"], as_index=False).mean(numeric_only=True)
    lines = ["# boneseg and SAM together\n", "Test slices of the Liu samples, the evaluation's clicks (three seeds), app default settings.\n",
             "| Method | Budget | Dice (95% CI) | B.Ar/T.Ar bias (points) |", "|---|---|---|---|"]
    for b in [x for x in ("3+6", "25+25") if x in set(df.budget)]:
        for m in ("boneseg", "sam", "sam_boxes", "sam_maskprompt", "intersection", "union"):
            g = df[(df.budget == b) & (df.method == m)]
            e, lo, hi = hier_boot(g, "dice")
            lines.append(f"| {m} | {b} | {e:.3f} ({lo:.3f}–{hi:.3f}) | {g['bar_bias'].mean():+.2f} |")
    lines += ["\n| Comparison | Budget | Difference in Dice (95% CI) | Slices better |", "|---|---|---|---|"]
    for b in [x for x in ("3+6", "25+25") if x in set(df.budget)]:
        w = df[df.budget == b].set_index(["sample", "z", "method"])["dice"].unstack("method")
        for m in ("sam_boxes", "sam_maskprompt", "intersection", "union"):
            d = pd.DataFrame({"sample": w.index.get_level_values(0), "diff": (w[m] - w["boneseg"]).to_numpy()})
            e, lo, hi = hier_boot(d, "diff")
            lines.append(f"| {m} vs boneseg | {b} | {e:+.3f} ({lo:+.3f} to {hi:+.3f}) | {(d['diff'] > 0).mean():.0%} |")
    (OUT / "combine_sam_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
