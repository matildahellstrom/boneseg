"""Loading the augmented Liu test slices written by make_augmented.py (robustness testing only, never training).

Each item behaves like paper/common.py::Slice (sample, z, img, gt, pixel_um) and also carries augmentation, level,
params and seed. img and gt are read from disk when accessed (not kept in memory), because the full set is several
GB. Read them once into a local variable if you need them more than once.

Import from a script in paper/:          from robustness.common import load_augmented
Import from a script in paper/robustness/: put paper/ first on sys.path (see make_augmented.py), then the same line.
(There is no __init__.py; paper/robustness is used as a namespace package. It is not imported as plain `common`,
so it does not clash with paper/common.py.)
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = "data/augmented"


@dataclass
class AugSlice:
    sample: str
    z: int
    augmentation: str
    level: int
    pixel_um: tuple          # (y, x) in um; swapped for 90/270 degree rotations
    img_path: Path
    mask_path: Path
    params: dict = field(default_factory=dict)
    seed: int = 0

    @property
    def img(self) -> np.ndarray:
        """Augmented image, float32 in [0, 1]."""
        return tifffile.imread(self.img_path)

    @property
    def gt(self) -> np.ndarray:
        """Expert mask after the same geometric transform (boolean)."""
        return np.asarray(Image.open(self.mask_path)) > 127


def resolve_root(root=DEFAULT_ROOT) -> Path:
    """Relative roots are taken from the repository root, so it does not matter where a script is run from."""
    root = Path(root)
    return root if root.is_absolute() else ROOT / root


def load_augmented(root=DEFAULT_ROOT, augmentation=None, level=None, samples=None) -> list[AugSlice]:
    """Items from manifest.csv, optionally filtered. augmentation, level and samples each take one value or a list
    (e.g. augmentation=["clean", "blur"], level=[0, 3]). Clean images are augmentation "clean", level 0."""
    root = resolve_root(root)
    as_set = lambda v: None if v is None else set(v) if isinstance(v, (list, tuple, set)) else {v}
    augs, levels, samps = as_set(augmentation), as_set(level), as_set(samples)
    out = []
    with open(root / "manifest.csv", newline="") as f:
        for r in csv.DictReader(f):
            # Skip rows that do not match the filters
            if augs and r["augmentation"] not in augs or levels and int(r["level"]) not in levels \
                    or samps and r["sample"] not in samps:
                continue
            out.append(AugSlice(r["sample"], int(r["z"]), r["augmentation"], int(r["level"]),
                                (float(r["pixel_um_y"]), float(r["pixel_um_x"])),
                                root / r["img_path"], root / r["mask_path"], json.loads(r["params"]), int(r["seed"])))
    return out
