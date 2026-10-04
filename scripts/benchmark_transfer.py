"""How well do profiles and learned models carry over between files?

For every file with an expert mask channel, this script
  1. clicks on each test slice of that file (simulated clicks from the expert mask), as a baseline,
  2. builds a click profile on one slice of every file and applies it to the test slices of every file,
  3. trains the learned model on five expert-labelled slices of every file and applies it to every file,
  4. trains it on the expert slices of all other files together and applies it to the held-out file.
Test slices never overlap with the slices used for profiles or labels.

Usage:
  python scripts/benchmark_transfer.py "A=data/liudata/A.ims:3" "E=data/liudata/E.ims:4" "F=data/liudata/F.ims:4"
Each argument is an optional short name, a file and its image channel; the expert mask is the last channel.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_file import clicks  # noqa: E402
from boneseg import io as bio, metrics  # noqa: E402
from boneseg.backbone import get_backbone  # noqa: E402
from boneseg.head import segment_with_head, train_head  # noqa: E402
from boneseg.segment import SegmentationSettings, embed_image, prototypes, segment, segment_with_prototypes  # noqa: E402


def load(spec: str, n_test: int, n_train: int, bb, s):
    label, _, spec = spec.rpartition("=") if "=" in spec.split(":")[0] else ("", "", spec)
    path, ch = spec.rsplit(":", 1)
    vol = bio.load_volume(path)
    ref_ch = vol.n_channels - 1
    lo, hi = int(0.1 * (vol.n_z - 1)), int(0.9 * (vol.n_z - 1))
    zs = np.linspace(lo, hi, n_test + n_train).round().astype(int).tolist()
    train_z, test_z = zs[1::(n_test + n_train) // n_train][:n_train], None
    test_z = [z for z in zs if z not in train_z][:n_test]
    data = {}
    for z in sorted(set(train_z + test_z)):
        gt = vol.get_plane(ref_ch, z) > 0
        if gt.any():
            data[z] = (embed_image(bb, bio.normalize_plane(vol.get_plane(int(ch), z)), s), gt)
    name = label or Path(path).stem
    return {"name": name, "vol": vol, "data": data, "train": [z for z in train_z if z in data], "test": [z for z in test_z if z in data]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--backbone", default="dinov2_s14")
    ap.add_argument("--n-test", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=5)
    args = ap.parse_args(argv)
    s = SegmentationSettings(backbone=args.backbone)
    bb = get_backbone(args.backbone)
    files = [load(f, args.n_test, args.n_train, bb, s) for f in args.files]
    for f in files:
        print(f"{f['name']}: {f['vol'].height}x{f['vol'].width} px, {f['vol'].pixel_um[0]:.3f} um/px, "
              f"train slices {f['train']}, test slices {f['test']}", flush=True)

    rows = []
    for tgt in files:
        d = np.mean([metrics.dice(segment(tgt["data"][z][0], *clicks(tgt["data"][z][1], 25, 25, z), s).mask, tgt["data"][z][1]) for z in tgt["test"]])
        rows.append({"method": "clicks on every slice (25+25)", "train": tgt["name"], "test": tgt["name"], "dice": d})
    for src in files:
        # A click profile from one labelled slice of the source file
        z0 = src["train"][len(src["train"]) // 2]
        emb0, gt0 = src["data"][z0]
        pp, nn = clicks(gt0, 25, 25, 0)
        pos, neg = prototypes(emb0, pp), prototypes(emb0, nn)
        thr = segment_with_prototypes(emb0, pos, neg, s, pos_points=pp, neg_points=nn).raw_threshold
        # A learned model from five expert-labelled slices of the source file
        head = train_head([src["data"][z] for z in src["train"]], s, [(0, z) for z in src["train"]], pixel_um=src["vol"].pixel_um)
        for tgt in files:
            for method, seg in (("click profile from 1 slice", lambda e: segment_with_prototypes(e, pos, neg, s, raw_threshold=thr).mask),
                                (f"learned model from {len(src['train'])} expert slices", lambda e: segment_with_head(head, e, s).mask)):
                d = np.mean([metrics.dice(seg(tgt["data"][z][0]), tgt["data"][z][1]) for z in tgt["test"]])
                rows.append({"method": method, "train": src["name"], "test": tgt["name"], "dice": d})
        print(f"done with source {src['name']} ({head.kind}, context {head.context})", flush=True)
    # Leave one file out: a model trained on the expert slices of all the other files
    if len(files) > 2:
        for tgt in files:
            others = [f for f in files if f is not tgt]
            pooled = [f["data"][z] for f in others for z in f["train"]]
            head = train_head(pooled, s, [(0, z) for f in others for z in f["train"]], pixel_um=tgt["vol"].pixel_um)
            d = np.mean([metrics.dice(segment_with_head(head, tgt["data"][z][0], s).mask, tgt["data"][z][1]) for z in tgt["test"]])
            rows.append({"method": "learned model from all other files", "train": "+".join(f["name"] for f in others), "test": tgt["name"], "dice": d})
    df = pd.DataFrame(rows)
    for method, sub in df.groupby("method", sort=False):
        print(f"\n{method}: rows = trained on, columns = tested on")
        print(sub.pivot(index="train", columns="test", values="dice").round(3).to_string())
    return df


if __name__ == "__main__":
    main()
