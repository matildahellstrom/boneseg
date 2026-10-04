"""Exports the evaluation slices for nnU-Net v2, the supervised reference most reviewers will ask for.

Creates nnUNet_raw/Dataset501_LiuBone with
  imagesTr/  development slices of every sample (labelled), as 8-bit PNG
  labelsTr/  their expert masks (0 background, 1 bone)
  imagesTs/  test slices, never used for training
and splits_final.json with one leave-one-sample-out fold per sample: train on the development slices
of the other samples (5 labelled slices each, as for boneseg and the random forest), validate on their remaining
development slices. The held-out sample is never seen before its test slices are predicted. Copy splits_final.json into
nnUNet_preprocessed/Dataset501_LiuBone/ after planning. Then, on a GPU (for example Kaggle):

  pip install nnunetv2
  export nnUNet_raw=... nnUNet_preprocessed=... nnUNet_results=...
  nnUNetv2_plan_and_preprocess -d 501 -c 2d --verify_dataset_integrity
  cp splits_final.json $nnUNet_preprocessed/Dataset501_LiuBone/
  for f in 0 1 2; do nnUNetv2_train 501 2d $f; done            # one fold per held-out sample
  # predict each held-out sample's test slices with its own fold, see test_cases.json
  nnUNetv2_predict -i imagesTs_A -o pred_A -d 501 -c 2d -f 0   # and so on

and score the predictions with: python paper/nnunet_score.py --pred-dir <folder with all predictions>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from common import available_samples, load_sample


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="nnUNet_raw")
    ap.add_argument("--samples", nargs="*", default=None)
    args = ap.parse_args(argv)
    names = args.samples or available_samples()
    root = Path(args.out) / "Dataset501_LiuBone"
    for d in ("imagesTr", "labelsTr", "imagesTs"):
        (root / d).mkdir(parents=True, exist_ok=True)
    dev_cases, val_cases, test_cases = {}, {}, {}
    for n in names:
        s = load_sample(n)
        # The same 5 labelled development slices per sample that boneseg and the random forest get; the
        # remaining development slices serve as nnU-Net's validation set, so the held-out sample is never seen
        for kind, slices, store in (("dev", s.dev[:5], dev_cases), ("dev", s.dev[5:], val_cases), ("test", s.test, test_cases)):
            for sl in slices:
                case = f"{n}_z{sl.z:04d}"
                img = Image.fromarray((np.clip(sl.img, 0, 1) * 255).astype(np.uint8))
                if kind == "dev":
                    img.save(root / "imagesTr" / f"{case}_0000.png")
                    Image.fromarray(sl.gt.astype(np.uint8)).save(root / "labelsTr" / f"{case}.png")
                else:
                    img.save(root / "imagesTs" / f"{case}_0000.png")
                store.setdefault(n, []).append(case)
        print(f"{n}: {len(dev_cases[n])} training cases, {len(val_cases.get(n, []))} validation cases, {len(test_cases[n])} test cases")
    (root / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "autofluorescence"}, "labels": {"background": 0, "bone": 1},
        "numTraining": sum(len(v) for v in dev_cases.values()) + sum(len(v) for v in val_cases.values()), "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO"}, indent=1))
    splits = [{"train": [c for m in names if m != n for c in dev_cases[m]], "val": [c for m in names if m != n for c in val_cases.get(m, [])]}
              for n in names]
    (root / "splits_final.json").write_text(json.dumps(splits, indent=1))
    (root / "test_cases.json").write_text(json.dumps({"fold_of_sample": {n: i for i, n in enumerate(names)}, "test_cases": test_cases}, indent=1))
    print(f"Wrote {root}")


if __name__ == "__main__":
    main()
