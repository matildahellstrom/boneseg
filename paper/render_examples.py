"""Renders the worst, median and best test slice of every sample for boneseg v2 with 25 + 25 clean clicks.

The slices are ranked by the Dice recorded in results/slices.csv for click seed 0, and the masks are recomputed with
the same clicks and the settings every fold chose (lambda 0.8, guided filter, threshold position 0.9, 2 x 2 passes),
so the shown masks reproduce the recorded Dice. Each image is the slice next to an overlay: cyan where boneseg and
the expert agree, amber where only boneseg marks bone, magenta where only the expert does, a white expert outline,
and the clicks. Writes paper/figures/examples/ex_<sample>_<worst|median|best>.jpg and cases.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from PIL import Image

from common import load_sample, simulated_clicks
from boneseg import metrics
from boneseg.backbone import get_backbone
from boneseg.segment import SegmentationSettings, embed_image, segment

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures" / "examples"


def overlay(img, mask, gt, pos, neg):
    g = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([g] * 3, -1).astype(float)
    ov = rgb * 0.55
    for m, col in ((mask & gt, (40, 210, 205)), (mask & ~gt, (255, 170, 30)), (~mask & gt, (235, 60, 140))):
        ov[m] = ov[m] * 0.25 + np.array(col) * 0.75
    ov[gt ^ ndi.binary_erosion(gt, iterations=2)] = (255, 255, 255)
    for (y, x), col in [(p, (60, 255, 90)) for p in pos] + [(p, (255, 60, 60)) for p in neg]:
        ov[max(0, y - 4):y + 5, max(0, x - 4):x + 5] = col
    pair = np.concatenate([rgb, np.full((rgb.shape[0], 12, 3), 255.0), ov], 1).astype(np.uint8)
    im = Image.fromarray(pair)
    return im.resize((1100, int(im.height * 1100 / im.width)), Image.LANCZOS)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(HERE / "results" / "slices.csv")
    sel = df[(df.method == "dino_clicks_v2") & (df.budget == "25+25") & (~df.noisy) & (df.seed == 0)]
    st = SegmentationSettings(backbone="dinov2_s14", vit_size=980, shift_passes=2, edge_refine="guided", guided_eps=0.01,
                              neg_weight=0.8, threshold_position=0.9)
    bb = get_backbone(st.backbone)
    cases = []
    for name in sorted(sel["sample"].unique()):
        s = sel[sel["sample"] == name].sort_values("dice")
        by_z = {sl.z: sl for sl in load_sample(name).test}
        for kind, row in (("worst", s.iloc[0]), ("median", s.iloc[len(s) // 2]), ("best", s.iloc[-1])):
            sl = by_z[int(row.z)]
            pos, neg = simulated_clicks(sl.gt, 25, 25, 100, False)   # Seed 0 of the evaluation
            mask = segment(embed_image(bb, sl.img, st), pos, neg, st, sl.pixel_um).mask
            f = f"ex_{name}_{kind}.jpg"
            overlay(sl.img, mask, sl.gt, pos, neg).save(OUT / f, quality=84)
            cases.append({"sample": name, "kind": kind, "z": int(row.z), "dice": round(metrics.dice(mask, sl.gt), 3),
                          "dice_recorded": round(float(row.dice), 3), "bar_pred": round(100 * float(mask.mean()), 1),
                          "bar_gt": round(100 * float(sl.gt.mean()), 1), "file": f})
            print(cases[-1], flush=True)
    (OUT / "cases.json").write_text(json.dumps(cases, indent=1))


if __name__ == "__main__":
    main()
