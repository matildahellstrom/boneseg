"""A synthetic bone-like z-stack for trying the app without real data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import tifffile


def make_demo_stack(n_z: int = 12, h: int = 384, w: int = 512, seed: int = 7, depth_degradation: float = 0.0) -> tuple[np.ndarray, dict]:
    """Returns a (Z, C, Y, X) float32 array with three channels:
    0, a bone matrix channel with trabecular texture,
    1, a TRAP-like channel with bright multinucleated cells that drift and change size with depth,
    2, the expert segmentation of the cells in channel 1.
    depth_degradation from 0 to 1 mimics a confocal stack imaged from the top: deeper slices get
    dimmer, blurrier and noisier, which changes how the cells look with depth."""
    rng = np.random.default_rng(seed)
    # Trabecular bone: thresholded smooth noise, shared by all slices with a slow drift
    base = ndi.gaussian_filter(rng.normal(size=(n_z + 8, h, w)), sigma=(3, 14, 14))
    yy, xx = np.mgrid[:h, :w]
    n_cells = 9
    cells = [{"y": rng.uniform(40, h - 40), "x": rng.uniform(40, w - 40), "r": rng.uniform(12, 24),
              "z0": rng.uniform(-2, n_z + 2), "dz": rng.uniform(3, 7), "vy": rng.normal(0, 1.5), "vx": rng.normal(0, 1.5),
              "lobes": int(rng.integers(2, 5))} for _ in range(n_cells)]
    out = np.zeros((n_z, 3, h, w), np.float32)
    for z in range(n_z):
        b = base[z + 4]
        bone = (b > np.percentile(b, 55)).astype(np.float32)
        bone = ndi.gaussian_filter(bone, 1.5)
        matrix = 0.25 + 0.5 * bone + 0.08 * rng.normal(size=(h, w))
        gt = np.zeros((h, w), bool)
        for c in cells:
            # Each cell is brightest near its own depth z0 and fades out above and below
            k = np.exp(-0.5 * ((z - c["z0"]) / c["dz"]) ** 2)
            if k < 0.25:
                continue
            cy, cx = c["y"] + c["vy"] * z, c["x"] + c["vx"] * z
            r = c["r"] * (0.6 + 0.4 * k)
            cell = np.zeros((h, w), bool)
            for li in range(c["lobes"]):
                a = 2 * np.pi * li / c["lobes"]
                ly, lx = cy + 0.45 * r * np.sin(a), cx + 0.45 * r * np.cos(a)
                cell |= (yy - ly) ** 2 + (xx - lx) ** 2 < (0.7 * r) ** 2
            gt |= cell
        trap = 0.1 + 0.15 * bone + 0.55 * ndi.gaussian_filter(gt.astype(np.float32), 1.2)
        # Granular texture inside the cells and speckle everywhere
        trap += 0.12 * gt * rng.random((h, w)) + 0.06 * rng.normal(size=(h, w))
        if depth_degradation > 0:
            d = depth_degradation * z / max(1, n_z - 1)
            trap = ndi.gaussian_filter(trap, 0.3 + 2.5 * d) * (1 - 0.7 * d) + (0.1 + 0.25 * d) * rng.normal(size=(h, w)) * d
            matrix = ndi.gaussian_filter(matrix, 0.3 + 2.5 * d) * (1 - 0.7 * d)
        out[z, 0] = np.clip(matrix, 0, 1)
        out[z, 1] = np.clip(trap, 0, 1)
        out[z, 2] = gt.astype(np.float32)
    meta = {"axes": "ZCYX", "spacing": 2.0, "unit": "um", "Labels": None}
    return out, meta


def write_demo_tiff(path: str | Path, **kw) -> Path:
    arr, _ = make_demo_stack(**kw)
    path = Path(path)
    tifffile.imwrite(path, (arr * 65535).astype(np.uint16), imagej=True, resolution=(1 / 0.65, 1 / 0.65),
                     metadata={"axes": "ZCYX", "spacing": 2.0, "unit": "um"})
    return path


DEMO_CHANNEL_NAMES = ["Bone matrix (demo)", "TRAP osteoclasts (demo)", "Expert segmentation (demo)"]
