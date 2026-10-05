"""Scores the NOISe detector's predictions from the Kaggle kernel (paper/kaggle_noise) and adds them to
paper/results/noise_slices.csv as method 'noise_yolo', condition 'trained on other batches'.

Usage:
  kaggle kernels output matildahellstrom/boneseg-noise-detector-baseline -p noise_out
  python paper/noise_yolo_score.py --preds noise_out/preds.zip
  python paper/noise_analyze.py
"""
from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from noise_common import BATCHES, load_batch
from noise_evaluate import score_instances

OUT = Path(__file__).resolve().parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="preds.zip from the kernel")
    args = ap.parse_args()
    z = zipfile.ZipFile(args.preds)
    names = set(z.namelist())
    rows, missing = [], 0
    for b in BATCHES:
        for p in load_batch(b)["test"]:
            f = f"{b}/{p.name}.png"
            if f not in names:
                missing += 1
                continue
            inst = np.asarray(Image.open(io.BytesIO(z.read(f)))).astype(np.int32)
            # Relabel to consecutive ids (overlaps may have hidden some)
            ids = np.unique(inst[inst > 0])
            lut = np.zeros(inst.max() + 1, np.int32)
            lut[ids] = np.arange(1, len(ids) + 1)
            rows.append({"batch": b, "patch": p.name, "method": "noise_yolo", "condition": "trained on other batches", "seed": 0,
                         **score_instances(lut[inst], p)})
    df = pd.read_csv(OUT / "noise_slices.csv")
    df = pd.concat([df[df["method"] != "noise_yolo"], pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(OUT / "noise_slices.csv", index=False)
    print(f"added {len(rows)} NOISe detector rows ({missing} test patches missing); now run paper/noise_analyze.py")


if __name__ == "__main__":
    main()
