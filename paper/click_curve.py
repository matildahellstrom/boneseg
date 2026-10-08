"""Dice against the number of clicks, and images for a side-by-side slice viewer.

Curve: every test slice of the Liu samples, clicks from 1 + 2 to 50 + 50 (careful, two seeds), for boneseg (app
defaults), SAM, micro-SAM and the random forest (ilastik-style). Writes paper/results/click_curve.csv.

Viewer: for the median and worst test slice of each sample (by boneseg with 25 + 25 clicks), 600 px images of the slice
and of each method's mask over it (boneseg and SAM with the same 25 + 25 clicks, boneseg's learned model from 5 labelled
slices of the same sample) with the expert outline. Writes paper/figures/viewer/ and viewer.json.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from PIL import Image

import methods as M
from common import available_samples, load_sample, simulated_clicks
from boneseg import metrics
from boneseg.backbone import get_backbone
from boneseg.head import segment_with_head, train_head
from boneseg.segment import SegmentationSettings, embed_image, segment

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
VIEW = HERE / "figures" / "viewer"
BUDGETS = [(1, 2), (3, 6), (5, 10), (10, 10), (25, 25), (50, 50)]
SIDE = 600


def render(img, mask=None, gt=None):
    g = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([g] * 3, -1).astype(float)
    if mask is not None:
        rgb[mask] = rgb[mask] * 0.35 + np.array((40, 210, 205)) * 0.65
    if gt is not None:
        rgb[gt ^ ndi.binary_erosion(gt, iterations=max(2, img.shape[1] // 400))] = (255, 70, 120)
    im = Image.fromarray(rgb.astype(np.uint8))
    return im.resize((SIDE, int(SIDE * img.shape[0] / img.shape[1])), Image.LANCZOS)


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
    st = SegmentationSettings()
    bb = get_backbone(st.backbone)
    VIEW.mkdir(parents=True, exist_ok=True)
    res = pd.read_csv(OUT / "slices.csv")
    rows, viewer = [], []
    for name in available_samples():
        s = load_sample(name)
        head = train_head([(embed_image(bb, sl.img, st), sl.gt) for sl in s.dev[:5]], st, [(0, i) for i in range(5)], pixel_um=s.dev[0].pixel_um)
        sel = res[(res.method == "dino_clicks_v2") & (res.budget == "25+25") & (~res.noisy) & (res.seed == 0) & (res["sample"] == name)].sort_values("dice")
        show = {int(sel.iloc[len(sel) // 2].z): "median", int(sel.iloc[0].z): "worst"}
        for sl in s.test:
            key = (sl.sample, sl.z)
            emb = M.dino_embedding(key, sl.img, st)
            for budget, seed in itertools.product(BUDGETS, range(2)):
                pos, neg = simulated_clicks(sl.gt, *budget, seed + 100)
                preds = {"boneseg": segment(emb, pos, neg, st, sl.pixel_um).mask, "sam": M.sam_points("sam", key, sl.img, pos, neg),
                         "microsam": M.sam_points("microsam", key, sl.img, pos, neg), "rf_clicks": M.rf_clicks(key, sl.img, pos, neg, seed=seed)}
                for m, p in preds.items():
                    rows.append({"sample": name, "z": sl.z, "n_pos": budget[0], "n_neg": budget[1], "seed": seed, "method": m, "dice": metrics.dice(p, sl.gt)})
                if sl.z in show and budget == (25, 25) and seed == 0:
                    learned = segment_with_head(head, emb, st, sl.pixel_um).mask
                    tag = f"{name}_{sl.z}"
                    render(sl.img).save(VIEW / f"{tag}_image.jpg", quality=82)
                    render(sl.img, None, sl.gt).save(VIEW / f"{tag}_expert.jpg", quality=82)
                    entry = {"sample": name, "z": sl.z, "kind": show[sl.z], "tag": tag, "dice": {}}
                    for m, p in (("boneseg", preds["boneseg"]), ("sam", preds["sam"]), ("learned", learned)):
                        render(sl.img, p, sl.gt).save(VIEW / f"{tag}_{m}.jpg", quality=82)
                        entry["dice"][m] = round(metrics.dice(p, sl.gt), 3)
                    viewer.append(entry)
        log(f"{name} done")
        pd.DataFrame(rows).to_csv(OUT / "click_curve.csv", index=False)
        (VIEW / "viewer.json").write_text(json.dumps(viewer, indent=1))
    df = pd.DataFrame(rows)
    print(df.groupby(["method", "n_pos", "n_neg"])["dice"].mean().unstack("method").round(3).to_string())


if __name__ == "__main__":
    main()
