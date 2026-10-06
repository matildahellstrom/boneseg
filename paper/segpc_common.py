"""The SegPC-2021 plasma cell data (ISBI 2021 challenge; Kaggle sbilab/segpc2021dataset, CC BY-NC-SA 4.0).

Bone marrow aspirate slides of multiple myeloma patients, Jenner-Giemsa stained, brightfield colour, from two
cameras (2040 x 1536 Olympus and 2560 x 1920 Nikon). Experts outlined each plasma cell "of interest" with its
nucleus and cytoplasm; other cells are present but not outlined. Training (298 images) and validation (200 images)
have outlines, the challenge test set does not.

`prepare` stores every training and validation image at 1080 x 1440 px, the resolution at which the official
evaluation scores, as PNG, with the cells as an [K, H, W] stack (0 background, 1 cytoplasm, 2 nucleus).
Cite the three papers listed in the dataset's readme when using it.
Usage: python paper/segpc_common.py prepare [--zip data/segpc/segpc2021dataset.zip]
"""
from __future__ import annotations

import argparse
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from common import ROOT  # noqa: F401  (puts the repository on the path)

DATA = ROOT / "data" / "segpc"
H, W = 1080, 1440


@dataclass
class Item:
    split: str
    name: str
    camera: str
    rgb: np.ndarray     # [1080, 1440, 3] uint8
    cells: np.ndarray   # [K, 1080, 1440] uint8: 0 background, 1 cytoplasm, 2 nucleus

    @property
    def cell_mask(self) -> np.ndarray:
        return (self.cells > 0).any(0)

    @property
    def instances(self) -> np.ndarray:
        """Instance map of whole cells; where outlines overlap, the later cell wins."""
        out = np.zeros((H, W), np.int32)
        for k, c in enumerate(self.cells, start=1):
            out[c > 0] = k
        return out

    @property
    def classes(self) -> np.ndarray:
        """0 background, 1 cytoplasm, 2 nucleus (nucleus wins overlaps)."""
        return self.cells.max(0) if len(self.cells) else np.zeros((H, W), np.uint8)


def prepare(zip_path: Path):
    z = zipfile.ZipFile(zip_path)
    names = z.namelist()
    for split, key in (("train", "/train/train/train/"), ("val", "/validation/validation/")):
        out = DATA / split
        out.mkdir(parents=True, exist_ok=True)
        xs = sorted(n for n in names if key + "x/" in n and n.endswith(".bmp"))
        ys = [n for n in names if key + "y/" in n and n.endswith(".bmp")]
        for n in xs:
            stem = Path(n).stem
            im = Image.open(io.BytesIO(z.read(n))).convert("RGB")
            camera = "nikon" if im.size[0] == 2560 or im.size[1] == 2560 else "olympus"
            im.resize((W, H), Image.BILINEAR).save(out / f"{stem}_{camera}.png")
            cells = []
            for m in sorted(y for y in ys if Path(y).stem.split("_")[0] == stem):
                a = Image.open(io.BytesIO(z.read(m))).convert("L").resize((W, H), Image.NEAREST)
                a = np.asarray(a)
                cells.append(np.where(a == 40, 2, np.where(a == 20, 1, 0)).astype(np.uint8))
            np.savez_compressed(out / f"{stem}.npz", cells=np.stack(cells) if cells else np.zeros((0, H, W), np.uint8))
        print(f"{split}: {len(xs)} images", flush=True)


def load(split: str, limit: int | None = None) -> list[Item]:
    files = sorted((DATA / split).glob("*.png"))[:limit]
    items = []
    for f in files:
        stem, camera = f.stem.rsplit("_", 1)
        items.append(Item(split, stem, camera, np.asarray(Image.open(f).convert("RGB")), np.load(DATA / split / f"{stem}.npz")["cells"]))
    return items


def official_miou(pred_instances: np.ndarray, item: Item) -> list[float]:
    """The challenge's score per expert cell: the best IoU of any predicted cell with the whole expert cell
    (nucleus and cytoplasm), at 1080 x 1440. Predicted cells that match no expert cell are not penalized."""
    n_p = int(pred_instances.max())
    out = []
    for c in item.cells:
        g = c > 0
        if n_p == 0:
            out.append(0.0)
            continue
        ids, inter = np.unique(pred_instances[g], return_counts=True)
        best = 0.0
        for i, n in zip(ids, inter):
            if i == 0:
                continue
            union = g.sum() + (pred_instances == i).sum() - n
            best = max(best, n / union)
        out.append(float(best))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare"])
    ap.add_argument("--zip", default=str(DATA / "segpc2021dataset.zip"))
    a = ap.parse_args()
    prepare(Path(a.zip))
