"""Scores nnU-Net predictions on the test slices and appends them to paper/results/slices.csv as method 'nnunet'.

Usage: python paper/nnunet_score.py --pred-dir PREDICTIONS [--condition "labels from other samples"]
Predictions are PNG files named like the test cases (for example A_z0042.png), each predicted by the fold
that held out its sample.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from common import bone_measures, load_sample
from evaluate import OUT, score


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--condition", default="other samples", help="How nnU-Net was trained, as a label for the results")
    args = ap.parse_args(argv)
    preds = {p.stem: p for p in Path(args.pred_dir).glob("*.png")}
    rows = []
    for name in sorted({k.split("_z")[0] for k in preds}):
        s = load_sample(name)
        for sl in s.test:
            case = f"{name}_z{sl.z:04d}"
            if case not in preds:
                print(f"missing prediction for {case}")
                continue
            pred = np.asarray(Image.open(preds[case])) > 0
            gtm = bone_measures(sl.gt, sl.pixel_um)
            rows.append({"sample": name, "z": sl.z, "method": "nnunet", "budget": args.condition, "noisy": False, "seed": 0,
                         **{f"gt_{k}": v for k, v in gtm.items()}, **score(pred, sl)})
    df = pd.read_csv(OUT / "slices.csv")
    df = pd.concat([df[df["method"] != "nnunet"], pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(OUT / "slices.csv", index=False)
    print(f"added {len(rows)} nnU-Net rows; rerun paper/analyze.py")


if __name__ == "__main__":
    main()
