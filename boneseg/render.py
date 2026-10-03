"""Turning arrays into PNG images for the browser."""
from __future__ import annotations

import base64
import io

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

# Perceptually ordered colormap anchors (dark purple to yellow), close to "magma"
_HEAT_ANCHORS = np.array([
    [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164],
], dtype=np.float32)
_HEAT_LUT = np.stack([np.interp(np.linspace(0, 1, 256), np.linspace(0, 1, len(_HEAT_ANCHORS)), _HEAT_ANCHORS[:, i]) for i in range(3)], 1).astype(np.uint8)


def display_shape(h: int, w: int, max_side: int) -> tuple[int, int]:
    s = min(1.0, max_side / max(h, w))
    return max(1, int(round(h * s))), max(1, int(round(w * s)))


def _resize(img: Image.Image, shape: tuple[int, int], nearest=False) -> Image.Image:
    h, w = shape
    if img.size == (w, h):
        return img
    return img.resize((w, h), Image.NEAREST if nearest else Image.BILINEAR)


def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=3)
    return buf.getvalue()


def data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


def gray_png(plane01: np.ndarray, max_side: int = 1600, gamma: float = 1.0) -> bytes:
    p = np.clip(plane01, 0, 1)
    if gamma != 1.0:
        p = p ** gamma
    img = Image.fromarray((p * 255).astype(np.uint8), "L")
    return to_png_bytes(_resize(img, display_shape(*plane01.shape, max_side)))


def heat_png(heat01: np.ndarray, max_side: int = 1600, alpha: int = 255) -> bytes:
    shape = display_shape(*heat01.shape, max_side)
    small = np.asarray(_resize(Image.fromarray((np.clip(heat01, 0, 1) * 255).astype(np.uint8), "L"), shape))
    rgba = np.concatenate([_HEAT_LUT[small], np.full(small.shape + (1,), alpha, np.uint8)], -1)
    return to_png_bytes(Image.fromarray(rgba, "RGBA"))


def mask_png(mask: np.ndarray, max_side: int = 1600, color=(0, 220, 255), fill_alpha: int = 70, edge_alpha: int = 255) -> bytes:
    """Semi-transparent fill with a solid outline, at display resolution."""
    shape = display_shape(*mask.shape, max_side)
    small = np.asarray(_resize(Image.fromarray(mask.astype(np.uint8) * 255, "L"), shape, nearest=False)) > 127
    edge = small & ~ndi.binary_erosion(small, iterations=1)
    rgba = np.zeros(small.shape + (4,), np.uint8)
    rgba[small] = (*color, fill_alpha)
    rgba[edge] = (*color, edge_alpha)
    return to_png_bytes(Image.fromarray(rgba, "RGBA"))


def uncertainty_png(u: np.ndarray, max_side: int = 1600) -> bytes:
    """Orange where the mask is unstable, transparent where it is stable."""
    shape = display_shape(*u.shape, max_side)
    small = np.asarray(_resize(Image.fromarray((np.clip(u / 0.5, 0, 1) * 255).astype(np.uint8), "L"), shape)).astype(np.float32) / 255
    rgba = np.zeros(small.shape + (4,), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = 255, 140, 0
    rgba[..., 3] = (small * 200).astype(np.uint8)
    return to_png_bytes(Image.fromarray(rgba, "RGBA"))


def mask_full_png(mask: np.ndarray) -> bytes:
    return to_png_bytes(Image.fromarray(mask.astype(np.uint8) * 255, "L"))
