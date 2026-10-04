"""Shared pieces of the paper evaluation: samples, slice splits, simulated clicks and bone measures.

Each sample contributes slices from the central 80% of its stack where the expert mask is not empty.
They are split, interleaved, into development slices (used for tuning and as labelled training
slices) and test slices (used only for the final scores when that sample is held out).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boneseg import io as bio  # noqa: E402
from boneseg.histo import histomorphometry  # noqa: E402

DATA = ROOT / "data" / "liudata"
# Short name -> (file, image channel). The expert mask is always the last channel.
SAMPLES = {
    "A": ("10-26-40_6_Blaze_crop2 quantified.ims", 3),
    "C": ("16-22-39_IC 4_Trap_SOST_col1-_Blaze crop1 segmentation quantified.ims", 4),
    "D": ("15-15-06_IC 2_Trap_SOST_col1_Blaze crop2 segmentation quantified.ims", 4),
    "E": ("11-26-23_1TRAP_SOST_NCAM_Blaze crop3 segmentation quantified.ims", 4),
    "F": ("11-30-50_5TRAP_SOST_NCAM_Blazecrop1 segmentation quantified.ims", 4),
}


@dataclass
class Slice:
    sample: str
    z: int
    img: np.ndarray      # Normalized to [0, 1]
    gt: np.ndarray       # Expert mask
    pixel_um: tuple


@dataclass
class Sample:
    name: str
    vol: bio.Volume
    channel: int
    dev: list = field(default_factory=list)   # [Slice]
    test: list = field(default_factory=list)  # [Slice]


def available_samples() -> list[str]:
    return [k for k, (f, _) in SAMPLES.items() if (DATA / f).exists()]


def load_sample(name: str, n_slices: int = 20) -> Sample:
    fname, ch = SAMPLES[name]
    vol = bio.load_volume(DATA / fname)
    ref = vol.n_channels - 1
    lo, hi = int(0.1 * (vol.n_z - 1)), int(0.9 * (vol.n_z - 1))
    # More candidates than needed, so slices with an empty expert mask can be dropped
    cands = np.unique(np.linspace(lo, hi, n_slices * 2).round().astype(int))
    picked = []
    for z in cands:
        gt = vol.get_plane(ref, int(z)) > 0
        if gt.mean() > 0.005:
            picked.append((int(z), gt))
    idx = np.linspace(0, len(picked) - 1, min(n_slices, len(picked))).round().astype(int)
    picked = [picked[i] for i in sorted(set(idx))]
    s = Sample(name, vol, ch)
    for i, (z, gt) in enumerate(picked):
        sl = Slice(name, z, bio.normalize_plane(vol.get_plane(ch, z)), gt, vol.pixel_um)
        (s.test if i % 2 == 0 else s.dev).append(sl)
    return s


def simulated_clicks(gt: np.ndarray, n_pos: int, n_neg: int, seed: int, noisy: bool = False):
    """Object clicks inside the expert mask and background clicks outside it.

    clean: at least 10 px from the boundary, like a careful user.
    noisy: anywhere, including right at the boundary, and 10% of the clicks of each kind land on the
           wrong side, like a hurried user. Returns (pos, neg) as lists of (y, x)."""
    rng = np.random.default_rng(seed)
    margin = 0 if noisy else 10
    inner = ndi.binary_erosion(gt, iterations=margin) if margin else gt
    outer = ~ndi.binary_dilation(gt, iterations=margin) if margin else ~gt
    inner = inner if inner.any() else gt
    outer = outer if outer.any() else ~gt

    def pick(region, k):
        ys, xs = np.nonzero(region)
        i = rng.choice(len(ys), k, replace=len(ys) < k)
        return [(int(ys[j]), int(xs[j])) for j in i]

    pos, neg = pick(inner, n_pos), pick(outer, n_neg)
    if noisy:
        n_wrong_p, n_wrong_n = max(1, round(0.1 * n_pos)) if n_pos >= 5 else 0, max(1, round(0.1 * n_neg)) if n_neg >= 5 else 0
        pos = pos[n_wrong_p:] + pick(~gt, n_wrong_p)
        neg = neg[n_wrong_n:] + pick(gt, n_wrong_n)
    return pos, neg


def bone_measures(mask: np.ndarray, pixel_um) -> dict:
    """2D bone measures (ASBMR names): bone area fraction B.Ar/T.Ar in %, bone surface density
    B.Pm/T.Ar in 1/mm, and the plate-model thickness Tb.Th = 2 B.Ar / B.Pm in um."""
    s, _ = histomorphometry(mask, np.zeros_like(mask), pixel_um)
    b_ar_um2 = s["B.Ar_mm2"] * 1e6
    b_pm_um = s["B.Pm_mm"] * 1e3
    return {"B.Ar/T.Ar_%": s["B.Ar/T.Ar_%"], "B.Pm/T.Ar_per_mm": s["B.Pm_mm"] / s["T.Ar_mm2"] if s["T.Ar_mm2"] else np.nan,
            "Tb.Th_um": 2 * b_ar_um2 / b_pm_um if b_pm_um > 0 else np.nan}
