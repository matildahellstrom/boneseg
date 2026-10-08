"""How much does the choice of clicks matter? One test slice, many random click draws.

For a median test slice of a sample, boneseg (app defaults) segments the slice with 20 random click draws at three
budgets (3 + 6, 10 + 10, 25 + 25 clicks), careful and careless. Outputs, in paper/figures/clicks/:
  grid_<sample>.jpg       four draws per budget (careful clicks), each with its clicks and Dice
  agreement_<sample>.jpg  per budget, the share of the 20 careful draws that call each pixel bone, with the expert outline
  click_variation.csv     Dice of every draw
Usage: python paper/click_variation.py [--samples E C]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from PIL import Image, ImageDraw

from common import load_sample, simulated_clicks
from boneseg import metrics
from boneseg.backbone import get_backbone
from boneseg.segment import SegmentationSettings, embed_image, segment

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures" / "clicks"
BUDGETS = [(3, 6), (10, 10), (25, 25)]
N_DRAWS = 20
TILE = 360


def tile(img, mask, gt, pos, neg, label):
    g = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([g] * 3, -1).astype(float) * 0.6
    for m, col in ((mask & gt, (40, 210, 205)), (mask & ~gt, (255, 170, 30)), (~mask & gt, (235, 60, 140))):
        rgb[m] = rgb[m] * 0.3 + np.array(col) * 0.7
    im = Image.fromarray(rgb.astype(np.uint8)).resize((TILE, int(TILE * img.shape[0] / img.shape[1])), Image.LANCZOS)
    f = TILE / img.shape[1]
    d = ImageDraw.Draw(im)
    for (y, x), col in [(p, (60, 255, 90)) for p in pos] + [(p, (255, 60, 60)) for p in neg]:
        d.ellipse([x * f - 4, y * f - 4, x * f + 4, y * f + 4], fill=col, outline=(0, 0, 0))
    d.rectangle([0, 0, 118, 20], fill=(0, 0, 0))
    d.text((6, 4), label, fill=(255, 255, 255))
    return im


def agreement_tile(img, freq, gt, label):
    """Share of draws calling each pixel bone, from dark (none) through orange to white (all), over the dimmed image."""
    import matplotlib
    cmap = matplotlib.colormaps["inferno"]
    g = np.clip(img, 0, 1)[..., None] * 0.35
    col = cmap(freq)[..., :3]
    a = np.clip(freq * 1.5, 0, 1)[..., None]
    rgb = (g * (1 - a) + col * a) * 255
    edge = gt ^ ndi.binary_erosion(gt, iterations=3)
    rgb[edge] = (80, 220, 255)
    im = Image.fromarray(rgb.astype(np.uint8)).resize((TILE, int(TILE * img.shape[0] / img.shape[1])), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 150, 20], fill=(0, 0, 0))
    d.text((6, 4), label, fill=(255, 255, 255))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="*", default=["E", "C"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    st = SegmentationSettings()
    bb = get_backbone(st.backbone)
    res = pd.read_csv(HERE / "results" / "slices.csv")
    rows, meta = [], {}
    for name in args.samples:
        s = load_sample(name)
        sel = res[(res.method == "dino_clicks_v2") & (res.budget == "25+25") & (~res.noisy) & (res.seed == 0) & (res["sample"] == name)].sort_values("dice")
        z = int(sel.iloc[len(sel) // 2].z)
        sl = next(x for x in s.test if x.z == z)
        emb = embed_image(bb, sl.img, st)
        grid_rows, agree = [], []
        for budget in BUDGETS:
            masks = []
            for noisy in (False, True):
                for seed in range(N_DRAWS):
                    pos, neg = simulated_clicks(sl.gt, *budget, 500 + seed, noisy)
                    m = segment(emb, pos, neg, st, sl.pixel_um).mask
                    d = metrics.dice(m, sl.gt)
                    rows.append({"sample": name, "z": z, "budget": f"{budget[0]}+{budget[1]}", "noisy": noisy, "seed": seed, "dice": d,
                                 "bone_area_pct": 100 * float(m.mean()), "expert_area_pct": 100 * float(sl.gt.mean())})
                    if not noisy:
                        masks.append(m)
                        if seed < 4:
                            grid_rows.append((budget, seed, tile(sl.img, m, sl.gt, pos, neg, f"{budget[0]}+{budget[1]}  Dice {d:.2f}")))
            agree.append(agreement_tile(sl.img, np.mean(masks, axis=0), sl.gt, f"{budget[0]}+{budget[1]}, {N_DRAWS} draws"))
        th = grid_rows[0][2].height
        grid = Image.new("RGB", (TILE * 4 + 30, th * len(BUDGETS) + 20 * (len(BUDGETS) - 1)), (255, 255, 255))
        for i, (_, seed, im) in enumerate(grid_rows):
            r, c = divmod(i, 4)
            grid.paste(im, (c * (TILE + 10), r * (th + 20)))
        grid.save(OUT / f"grid_{name}.jpg", quality=84)
        ag = Image.new("RGB", (TILE * 3 + 20, agree[0].height), (255, 255, 255))
        for i, im in enumerate(agree):
            ag.paste(im, (i * (TILE + 10), 0))
        ag.save(OUT / f"agreement_{name}.jpg", quality=86)
        meta[name] = {"z": z, "expert_area_pct": round(100 * float(sl.gt.mean()), 1)}
        print(name, "slice", z, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "click_variation.csv", index=False)
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(df.groupby(["sample", "budget", "noisy"])["dice"].agg(["mean", "std", "min", "max"]).round(3).to_string())


if __name__ == "__main__":
    main()
