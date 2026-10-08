"""Freezes everything the fine-tuned-SAM Kaggle kernels need into setup.json, so the kernels rebuild exactly the
images, folds, few-shot picks and prompts used locally (paper/sam_zeroshot.py):

  liu     per sample: Imaris file, image and mask channel, development and test slice indices
  noise   per batch: development and test patch names (paper/noise_common.py selection)
  segpc   validation images with outlines, and three draws of 5 + 5 training images (train / early stopping)
  prompts per test image and draw: the paper-style points, background points and noisy boxes (paper/prompts.py)

Few-shot: 5 labelled images of the target (the paper's setting): the first 5 development slices of the held-out Liu
sample (the next 5 for early stopping), the first 5 development patches of the held-out NOISe batch (the next 5 for
early stopping), and for SegPC three random draws from the training images.
Usage: python paper/kaggle_finetune_sam/export_setup.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import prompts as P  # noqa: E402
from common import SAMPLES, available_samples, load_sample  # noqa: E402
from noise_common import BATCHES, DATA as NOISE_DATA, load_batch  # noqa: E402
from segpc_common import load  # noqa: E402


def main():
    setup = {"liu": {}, "noise": {}, "segpc": {}, "prompts": {}}
    for name in available_samples():
        s = load_sample(name)
        fname, ch = SAMPLES[name]
        setup["liu"][name] = {"file": fname, "channel": ch, "dev": [sl.z for sl in s.dev], "test": [sl.z for sl in s.test]}
        for sl in s.test:
            setup["prompts"][f"liu/{name}_{sl.z}"] = [P.make(sl.gt, None, seed) for seed in range(3)]
    sel = json.loads((NOISE_DATA / "selection.json").read_text())
    for b in BATCHES:
        setup["noise"][b] = sel[b]
        for p in load_batch(b)["test"]:
            setup["prompts"][f"noise/{b}_{p.name}"] = [P.make(p.gt, p.instances, seed) for seed in range(3)]
    train = sorted(it.name for it in load("train"))
    val = [it for it in load("val") if len(it.cells)]
    draws = []
    for seed in range(3):
        pick = random.Random(seed).sample(train, 10)
        draws.append({"train": pick[:5], "val": pick[5:]})
    setup["segpc"] = {"val": [it.name for it in val], "draws": draws, "train_all": train}
    for it in val:
        setup["prompts"][f"segpc/{it.name}"] = [P.make(it.cell_mask, it.instances, seed) for seed in range(3)]

    def plain(o):
        if isinstance(o, dict):
            return {k: plain(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [plain(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return o
    (HERE / "setup.json").write_text(json.dumps(plain(setup)))
    print(f"setup.json: {len(setup['liu'])} Liu samples, {len(setup['noise'])} NOISe batches, {len(setup['segpc']['val'])} SegPC images, "
          f"{len(setup['prompts'])} test images with prompts; {(HERE / 'setup.json').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
