"""Does a model trained on a few labelled slices beat clicking on every slice?

Usage: python scripts/benchmark_head.py FILE --channel 3 --reference 4
Trains the learned head on 1, 2 and 4 slices whose labels come from the expert mask channel,
and scores it on held-out slices.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_file import clicks  # noqa: E402
from boneseg import io as bio, metrics  # noqa: E402
from boneseg.backbone import get_backbone  # noqa: E402
from boneseg.head import segment_with_head, train_head  # noqa: E402
from boneseg.segment import SegmentationSettings, embed_image, segment  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--channel", type=int, required=True)
    ap.add_argument("--reference", type=int, required=True)
    ap.add_argument("--backbone", default="dinov2_s14")
    ap.add_argument("--vit-size", type=int, default=980)
    args = ap.parse_args(argv)
    vol = bio.load_volume(args.path)
    s = SegmentationSettings(backbone=args.backbone, vit_size=args.vit_size)
    bb = get_backbone(args.backbone)
    zs = np.linspace(40, vol.n_z - 41, 12).round().astype(int).tolist()
    data = {z: (embed_image(bb, bio.normalize_plane(vol.get_plane(args.channel, z)), s), vol.get_plane(args.reference, z) > 0) for z in zs}
    train_pool, test = zs[::3], [z for i, z in enumerate(zs) if i % 3]
    print(f"{Path(args.path).name}: train pool {train_pool}, test {test}")
    clicks_dice = {n: np.mean([metrics.dice(segment(data[z][0], *clicks(data[z][1], n, n if n > 3 else 6, z), s, vol.pixel_um).mask, data[z][1]) for z in test]) for n in (3, 25)}
    print(f"clicks on every test slice: 3+6 {clicks_dice[3]:.3f}, 25+25 {clicks_dice[25]:.3f}")
    for k in (1, 2, 4):
        train = train_pool[:k]
        head = train_head([data[z] for z in train], s, [(args.channel, z) for z in train], pixel_um=vol.pixel_um)
        d = np.mean([metrics.dice(segment_with_head(head, data[z][0], s, vol.pixel_um).mask, data[z][1]) for z in test])
        cv = head.cv.get(head.cv["chosen"], {}).get("mean_dice")
        print(f"learned from {k} labelled slice(s), {head.kind}: test Dice {d:.3f}" + (f", cross-validated estimate {cv:.3f}" if cv is not None else ""))


if __name__ == "__main__":
    main()
