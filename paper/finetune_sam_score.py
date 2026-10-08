"""Scores the predictions of the fine-tuned-SAM Kaggle kernels (paper/kaggle_finetune_sam) like sam_zeroshot.py.

preds.zip holds <dataset>/<method>__<prompt>[_d<draw>|_full]/<image>.png. Writes paper/results/finetune_sam.csv
(one row per dataset, image, method, prompt, prompt draw and few-shot draw; Dice, NSD, area bias).
Usage: kaggle kernels output matildahellstrom/boneseg-sam-finetune-a -p ft_a   (and -b)
       python paper/finetune_sam_score.py ft_a/preds.zip ft_b/preds.zip
"""
from __future__ import annotations

import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from common import load_sample
from boneseg import metrics

OUT = Path(__file__).resolve().parent / "results"


def ground_truth():
    """{(dataset, image): (expert mask, pixel size, NSD tolerance)} for every test image."""
    from noise_common import BATCHES, load_batch
    from segpc_common import load
    from common import available_samples
    gt = {}
    for s in available_samples():
        for sl in load_sample(s).test:
            gt[("liu", f"{s}_{sl.z}")] = (sl.gt, sl.pixel_um, 5.0)
    for b in BATCHES:
        for p in load_batch(b)["test"]:
            gt[("noise", f"{b}_{p.name}")] = (p.gt, (1.0, 1.0), 2.0)
    for it in load("val"):
        if len(it.cells):
            gt[("segpc", it.name)] = (it.cell_mask, (1.0, 1.0), 2.0)
    return gt


def parse(folder: str):
    """'samft_b_ende_adapter__boxes_s1_d0' -> method, prompt, prompt seed, few-shot draw."""
    method, rest = folder.split("__", 1)
    draw = "full" if rest.endswith("_full") else (re.search(r"_d(\d+)$", rest).group(1) if re.search(r"_d\d+$", rest) else "0")
    rest = re.sub(r"_(d\d+|full)$", "", rest)
    m = re.match(r"(.+)_s(\d+)$", rest)
    prompt, seed = (m.group(1), int(m.group(2))) if m else (rest, 0)
    return method, prompt, seed, draw


def main(zips):
    gt = ground_truth()
    names = {"liu": "bone", "noise": "osteoclasts", "segpc": "plasma"}
    rows = []
    for zp in zips:
        z = zipfile.ZipFile(zp)
        for name in z.namelist():
            if not name.endswith(".png"):
                continue
            ds, folder, fname = Path(name).parts[-3:]
            key = (ds, Path(fname).stem)
            if key not in gt:
                continue
            g, px, tol = gt[key]
            pred = np.asarray(Image.open(io.BytesIO(z.read(name))).convert("L")) > 127
            if pred.shape != g.shape:
                pred = np.asarray(Image.fromarray(pred).resize(g.shape[::-1], Image.NEAREST))
            method, prompt, seed, draw = parse(folder)
            rows.append({"dataset": names[ds], "image": key[1], "method": method, "prompt": prompt, "seed": seed, "draw": draw,
                         "dice": metrics.dice(pred, g), "nsd": metrics.nsd(pred, g, tol, px, max_side=1500),
                         "area_bias": 100 * (float(pred.mean()) - float(g.mean()))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "finetune_sam.csv", index=False)
    print(df.groupby(["dataset", "method", "prompt", "draw"])[["dice", "nsd", "area_bias"]].mean().round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
