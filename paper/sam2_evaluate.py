"""SAM 2 next to SAM: alone with the clicks, and as the partner in boneseg's "Agree with SAM" (intersection).

Same test slices of the Liu samples and the same clicks as combine_sam.py (3 + 6 and 25 + 25 clean clicks, three
seeds), so the rows pair with paper/results/combine_sam.csv (boneseg, SAM ViT-B and their intersection):
  sam2_bplus, sam2_large           SAM 2.1 Hiera Base+ / Large with the clicks as point prompts
  intersection_sam2_bplus, ..._large  pixels both boneseg (app defaults) and SAM 2 call bone
Writes paper/results/sam2.csv and sam2_summary.md. SAM 2 weights in ~/.cache/boneseg-paper (sam2.1_hiera_*.pt).
"""
from __future__ import annotations

import itertools
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analyze import hier_boot
from common import available_samples, bone_measures, load_sample, simulated_clicks
from boneseg import metrics
from boneseg.backbone import get_backbone
from boneseg.segment import SegmentationSettings, embed_image, segment

OUT = Path(__file__).resolve().parent / "results"
WEIGHTS = Path(os.environ.get("BONESEG_PAPER_WEIGHTS", Path.home() / ".cache" / "boneseg-paper"))
MODELS = {"sam2_bplus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
          "sam2_large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt")}
BUDGETS = [(3, 6), (25, 25)]
LABELS = {"boneseg": "boneseg", "sam": "SAM ViT-B", "intersection": "boneseg ∩ SAM ViT-B (the app's Agree with SAM)",
          "sam2_bplus": "SAM 2.1 Base+", "sam2_large": "SAM 2.1 Large",
          "intersection_sam2_bplus": "boneseg ∩ SAM 2.1 Base+", "intersection_sam2_large": "boneseg ∩ SAM 2.1 Large"}


def predictors():
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    return {k: SAM2ImagePredictor(build_sam2(cfg, str(WEIGHTS / ck), device="cpu")) for k, (cfg, ck) in MODELS.items()}


def rgb8(img):
    img = np.clip(img, 0, 1)
    return ((np.repeat(img[..., None], 3, -1) if img.ndim == 2 else img) * 255).astype(np.uint8)


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
    st = SegmentationSettings()
    bb = get_backbone(st.backbone)
    preds = predictors()
    rows = []
    for name in available_samples():
        for sl in load_sample(name).test:
            emb = embed_image(bb, sl.img, st)
            for p in preds.values():
                p.set_image(rgb8(sl.img))
            gtm = bone_measures(sl.gt, sl.pixel_um)
            for budget, seed in itertools.product(BUDGETS, range(3)):
                pos, neg = simulated_clicks(sl.gt, *budget, seed + 100)
                b = segment(emb, pos, neg, st, sl.pixel_um).mask
                pts = np.array([(x, y) for y, x in list(pos) + list(neg)], np.float32)
                lbl = np.r_[np.ones(len(pos)), np.zeros(len(neg))].astype(np.int32)
                out = {"boneseg": b}
                with torch.inference_mode():
                    for k, p in preds.items():
                        m = p.predict(point_coords=pts, point_labels=lbl, multimask_output=False)[0][0].astype(bool)
                        out[k], out[f"intersection_{k}"] = m, b & m
                for m, pred in out.items():
                    rows.append({"sample": name, "z": sl.z, "budget": f"{budget[0]}+{budget[1]}", "seed": seed, "method": m,
                                 "dice": metrics.dice(pred, sl.gt), "bar_bias": bone_measures(pred, sl.pixel_um)["B.Ar/T.Ar_%"] - gtm["B.Ar/T.Ar_%"]})
        log(f"{name} done")
        pd.DataFrame(rows).to_csv(OUT / "sam2.csv", index=False)
    summarize()
    log("done")


def summarize():
    new = pd.read_csv(OUT / "sam2.csv")
    old = pd.read_csv(OUT / "combine_sam.csv")
    old = old[old.method.isin(["sam", "intersection"])]
    df = pd.concat([new, old], ignore_index=True)
    d = df.groupby(["budget", "method", "sample", "z"], as_index=False)[["dice", "bar_bias"]].mean()
    lines = ["# SAM 2 next to SAM\n", "Liu test slices, the same clean clicks as combine_sam.py (three seeds). boneseg recomputed with the "
             "current defaults; SAM ViT-B rows from combine_sam.csv. Dice with 95% intervals, bone-area bias in points, and the paired "
             "difference to boneseg alone.\n"]
    for budget in ("3+6", "25+25"):
        lines += [f"\n## {budget} clicks\n", "| Method | Dice (95% CI) | B.Ar/T.Ar bias | Minus boneseg (95% CI) | Slices better |", "|---|---|---|---|---|"]
        g = d[d.budget == budget]
        base = g[g.method == "boneseg"].set_index(["sample", "z"])["dice"]
        for m, lab in LABELS.items():
            x = g[g.method == m]
            if x.empty:
                continue
            e, lo, hi = hier_boot(x, "dice")
            diff = (x.set_index(["sample", "z"])["dice"] - base).dropna().reset_index().rename(columns={"dice": "diff"})
            vs = ("–", "–") if m == "boneseg" else ("{:+.3f} ({:+.3f} to {:+.3f})".format(*hier_boot(diff, "diff")), f"{(diff['diff'] > 0).mean():.0%}")
            lines.append(f"| {lab} | {e:.3f} ({lo:.3f}–{hi:.3f}) | {x['bar_bias'].mean():+.1f} | {vs[0]} | {vs[1]} |")
    (OUT / "sam2_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
