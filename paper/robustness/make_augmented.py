"""Augmented copies of the Liu test slices, each with its exact expert mask, for robustness testing (never training).

Sources: the 10 test slices of each available sample (common.load_sample(...).test), already normalised to [0, 1].
Every augmentation has three levels (1 mild, 2 moderate, 3 strong) and is applied with a seed derived from
(sample, z, augmentation, level), so a rerun gives the same files.
  geometric    rot90_flip, rotate, elastic: the same transform moves image and mask (mask: bilinear, threshold 0.5)
  photometric  contrast, gamma_lo, gamma_hi, gauss_noise, poisson_noise, blur, illumination, dimming, lowres,
               saturation: the mask is the clean mask, unchanged
The unaugmented slice is written as augmentation "clean", level 0. All images are float32, clipped to [0, 1] and
rounded to multiples of 2^-16 (error below 1e-5) so the zlib-compressed TIFFs are about 40% smaller.
Writes data/augmented/<augmentation>/L<level>/<sample>_<z>_img.tif and _mask.png (0/255), plus
data/augmented/manifest.csv (paths relative to data/augmented). data/ is git-ignored: the Liu licence is unclear,
so derived images stay local. Parameters per level: the LEVELS table below and paper/robustness/README.md.
Usage (from paper/): python robustness/make_augmented.py [--samples A C] [--augs blur rotate]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import tifffile
from PIL import Image
from skimage.transform import resize

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))   # paper/common.py must win over robustness/common.py

from common import available_samples, load_sample  # noqa: E402
from robustness.common import resolve_root  # noqa: E402

MIN_FREE_GB = 8   # Stop before the disk gets full

# Parameters per augmentation and level. Lengths are in pixels unless the name says um.
LEVELS = {
    "rot90_flip":    {1: {"k": 0, "flip": "h"}, 2: {"k": 1, "flip": None}, 3: {"k": 3, "flip": "v"}},  # k x 90 deg anticlockwise, then flip
    "rotate":        {1: {"angle": 10}, 2: {"angle": 25}, 3: {"angle": 45}},                     # Degrees, reflect padding
    "elastic":       {1: {"sigma_px": 60, "amp_px": 3}, 2: {"sigma_px": 45, "amp_px": 6},
                      3: {"sigma_px": 40, "amp_px": 8}},                                        # amp = RMS displacement per axis
    "contrast":      {1: {"c": 0.75, "b": 0.05}, 2: {"c": 0.5, "b": -0.1}, 3: {"c": 0.3, "b": 0.2}},  # c (img - 0.5) + 0.5 + b
    "gamma_lo":      {1: {"gamma": 0.7}, 2: {"gamma": 0.5}, 3: {"gamma": 0.35}},                # img^gamma, brightens dark parts
    "gamma_hi":      {1: {"gamma": 1.4}, 2: {"gamma": 2.0}, 3: {"gamma": 2.8}},                 # img^gamma, darkens
    "gauss_noise":   {1: {"sigma": 0.03}, 2: {"sigma": 0.06}, 3: {"sigma": 0.12}},              # Additive, in [0, 1] units
    "poisson_noise": {1: {"photons": 100}, 2: {"photons": 30}, 3: {"photons": 10}},             # Photons at intensity 1
    "blur":          {1: {"sigma_um": 1.5}, 2: {"sigma_um": 3.0}, 3: {"sigma_um": 6.0}},        # Gaussian PSF
    "illumination":  {1: {"vignette": 0.2, "field": 0.15}, 2: {"vignette": 0.4, "field": 0.3},
                      3: {"vignette": 0.6, "field": 0.45}},                                     # Gain (1 - v r^2)(1 + a f)
    "dimming":       {1: {"scale": 0.7, "offset": 0.03, "floor_sigma": 0.01},
                      2: {"scale": 0.45, "offset": 0.06, "floor_sigma": 0.02},
                      3: {"scale": 0.25, "offset": 0.1, "floor_sigma": 0.03}},                  # scale img + offset + noise
    "lowres":        {1: {"factor": 2}, 2: {"factor": 3}, 3: {"factor": 4}},                    # Down- and upsample
    "saturation":    {1: {"bone_pct": 1}, 2: {"bone_pct": 5}, 3: {"bone_pct": 15}},             # % of bone pixels clipped
}
GEOMETRIC = {"rot90_flip", "rotate", "elastic"}


def make_seed(sample, z, aug, level) -> int:
    """Deterministic 32-bit seed from the item's identity (Python's hash() changes between runs)."""
    return int.from_bytes(hashlib.sha256(f"{sample}_{z}_{aug}_{level}".encode()).digest()[:4], "little")


def smooth_field(shape, sigma_px, rng, n=1) -> np.ndarray:
    """n smooth random fields with unit standard deviation. Built on a coarse grid (where sigma is about 4 cells)
    and upsampled, which looks the same but is much faster than filtering a 3000 x 3000 image."""
    f = max(1, int(sigma_px // 4))
    small = (int(np.ceil(shape[0] / f)) + 1, int(np.ceil(shape[1] / f)) + 1)
    out = np.empty((n, *shape), np.float32)
    for i in range(n):
        g = ndi.gaussian_filter(rng.standard_normal(small), sigma_px / f, mode="reflect")
        g = (g - g.mean()) / g.std()
        out[i] = resize(g, shape, order=3, mode="reflect").astype(np.float32)
    return out


# ---------- Geometric: warp(arr, params, order) moves any array (image or float mask) the same way ----------

def warp(arr: np.ndarray, aug: str, p: dict, order: int = 1) -> np.ndarray:
    if aug == "rot90_flip":
        out = np.rot90(arr, p["k"])
        out = out[:, ::-1] if p["flip"] == "h" else out[::-1] if p["flip"] == "v" else out
        return np.ascontiguousarray(out)
    if aug == "rotate":
        return ndi.rotate(arr, p["angle"], reshape=False, order=order, mode="reflect")
    if aug == "elastic":
        # The displacement field is rebuilt from its own seed, so image and mask get exactly the same one
        dy, dx = smooth_field(arr.shape, p["sigma_px"], np.random.default_rng(p["field_seed"]), n=2) * p["amp_px"]
        yy, xx = np.mgrid[:arr.shape[0], :arr.shape[1]].astype(np.float32)
        return ndi.map_coordinates(arr, [yy + dy, xx + dx], order=order, mode="reflect")
    raise ValueError(aug)


def geometric(img, gt, aug, p, seed, pixel_um):
    """Returns the warped image, warped mask and the (possibly swapped) pixel size."""
    if aug == "elastic":
        p["field_seed"] = seed
    out = warp(img, aug, p, order=1)
    # Mask: warp the 0/1 mask as floats and threshold at one half
    mask = warp(gt.astype(np.float32), aug, p, order=1) >= 0.5
    if aug == "rot90_flip" and p["k"] % 2:
        pixel_um = (pixel_um[1], pixel_um[0])
    return out, mask, pixel_um


# ---------- Photometric: image only, the mask stays the clean mask ----------

def photometric(img, gt, aug, p, rng, pixel_um):
    if aug == "contrast":
        return p["c"] * (img - 0.5) + 0.5 + p["b"]
    if aug in ("gamma_lo", "gamma_hi"):
        return img ** p["gamma"]
    if aug == "gauss_noise":
        return img + rng.normal(0, p["sigma"], img.shape).astype(np.float32)
    if aug == "poisson_noise":
        # Scale to photon counts, draw the counts, scale back
        return rng.poisson(img * p["photons"]).astype(np.float32) / p["photons"]
    if aug == "blur":
        sig = (p["sigma_um"] / pixel_um[0], p["sigma_um"] / pixel_um[1])
        p["sigma_px"] = [round(s, 3) for s in sig]
        return ndi.gaussian_filter(img, sig, mode="reflect")
    if aug == "illumination":
        # Radial vignette (r = 0 at the centre, 1 in the corners) times a smooth random gain field
        h, w = img.shape
        yy, xx = np.ogrid[:h, :w]
        r2 = (((yy - (h - 1) / 2) / (h / 2)) ** 2 + ((xx - (w - 1) / 2) / (w / 2)) ** 2) / 2
        f = smooth_field(img.shape, max(h, w) / 4, rng)[0]
        f /= np.abs(f).max()
        gain = np.clip((1 - p["vignette"] * r2) * (1 + p["field"] * f), 0, None)
        p["gain_min"], p["gain_max"] = round(float(gain.min()), 3), round(float(gain.max()), 3)
        return img * gain
    if aug == "dimming":
        # Weaker signal, a haze offset and a noise floor, like a slice deeper in the tissue
        return p["scale"] * img + p["offset"] + rng.normal(0, p["floor_sigma"], img.shape).astype(np.float32)
    if aug == "lowres":
        small = resize(img, (img.shape[0] // p["factor"], img.shape[1] // p["factor"]), order=1, anti_aliasing=True)
        return resize(small, img.shape, order=1)
    if aug == "saturation":
        # Gain raised until bone_pct % of the bone pixels reach the top of the range
        t = float(np.percentile(img[gt], 100 - p["bone_pct"]))
        p["clip_at"] = round(t, 4)
        return img / max(t, 1e-6)
    raise ValueError(aug)


# ---------- Writing ----------

def save(root, aug, level, sl_name, img, mask) -> tuple[str, str]:
    d = root / aug / f"L{level}"
    d.mkdir(parents=True, exist_ok=True)
    img = (np.round(np.clip(img, 0, 1) * 65536) / 65536).astype(np.float32)
    tifffile.imwrite(d / f"{sl_name}_img.tif", img, compression="zlib")
    Image.fromarray(mask.astype(np.uint8) * 255).save(d / f"{sl_name}_mask.png")
    return f"{aug}/L{level}/{sl_name}_img.tif", f"{aug}/L{level}/{sl_name}_mask.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", default=None)
    ap.add_argument("--augs", nargs="+", default=list(LEVELS), choices=list(LEVELS))
    ap.add_argument("--root", default="data/augmented")
    args = ap.parse_args()
    root = resolve_root(args.root)
    root.mkdir(parents=True, exist_ok=True)
    samples = args.samples or available_samples()
    rows = []
    for name in samples:
        t0 = time.time()
        s = load_sample(name)
        s.dev.clear()   # Only the test slices are used; free the rest
        for sl in s.test:
            # Disk guard: the full set is about 12 GB
            if shutil.disk_usage(root).free / 1e9 < MIN_FREE_GB:
                sys.exit(f"Less than {MIN_FREE_GB} GB free on disk; stopped before {name} z={sl.z}")
            sl_name = f"{sl.sample}_{sl.z}"
            # The clean slice first
            ip, mp = save(root, "clean", 0, sl_name, sl.img, sl.gt)
            rows.append(dict(sample=sl.sample, z=sl.z, augmentation="clean", level=0, params="{}", seed=0,
                             pixel_um_y=sl.pixel_um[0], pixel_um_x=sl.pixel_um[1], img_path=ip, mask_path=mp,
                             mask_area_fraction=float(sl.gt.mean())))
            for aug in args.augs:
                for level, base in LEVELS[aug].items():
                    p, seed = dict(base), make_seed(sl.sample, sl.z, aug, level)
                    rng = np.random.default_rng(seed)
                    if aug in GEOMETRIC:
                        img, mask, pix = geometric(sl.img, sl.gt, aug, p, seed, sl.pixel_um)
                    else:
                        img, mask, pix = photometric(sl.img, sl.gt, aug, p, rng, sl.pixel_um), sl.gt, sl.pixel_um
                    ip, mp = save(root, aug, level, sl_name, img, mask)
                    rows.append(dict(sample=sl.sample, z=sl.z, augmentation=aug, level=level, params=json.dumps(p),
                                     seed=seed, pixel_um_y=pix[0], pixel_um_x=pix[1], img_path=ip, mask_path=mp,
                                     mask_area_fraction=float(mask.mean())))
                    del img, mask
        print(f"{name}: {len(s.test)} slices in {time.time() - t0:.0f} s", flush=True)
        del s
    # Merge with an existing manifest, replacing the rows that were just regenerated
    new = pd.DataFrame(rows)
    man = root / "manifest.csv"
    if man.exists():
        old = pd.read_csv(man, dtype={"params": str})
        key = lambda df: df["sample"].astype(str) + "|" + df["z"].astype(str) + "|" + df["augmentation"] + "|" + df["level"].astype(str)
        new = pd.concat([old[~key(old).isin(set(key(new)))], new], ignore_index=True)
    new.sort_values(["augmentation", "level", "sample", "z"]).to_csv(man, index=False)
    print(f"{len(rows)} items written, manifest has {len(new)} rows: {man}")


if __name__ == "__main__":
    main()
