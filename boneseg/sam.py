"""An optional second opinion from Segment Anything (SAM ViT-B) on a click segmentation.

SAM gets the same object and background clicks as point prompts, and the mask keeps only the pixels both agree on.
On the test slices of the five Liu samples, with 25 + 25 clicks, this raised Dice from 0.755 to 0.784 (+0.029, 95% CI
+0.013 to +0.047) and removed boneseg's tendency to overestimate bone area; with 3 + 6 clicks it made no difference
(-0.003) and underestimated bone area by 2.4 points (paper/results/combine_sam_summary.md). SAM needs the
segment-anything package and downloads its ViT-B weights (375 MB) on first use.
"""
from __future__ import annotations

import os
import threading
import urllib.request
from pathlib import Path

import numpy as np
import torch

WEIGHTS_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
_lock = threading.Lock()
_state: dict = {"predictor": None, "image_key": None}


def weights_path() -> Path:
    return Path(os.environ.get("BONESEG_SAM_WEIGHTS", Path(torch.hub.get_dir()) / "checkpoints" / "sam_vit_b_01ec64.pth"))


def available() -> bool:
    try:
        import segment_anything  # noqa: F401
        return True
    except ImportError:
        return False


def weights_cached() -> bool:
    return weights_path().exists()


def _predictor():
    if _state["predictor"] is None:
        if not available():
            raise ValueError("Agreeing with SAM needs the segment-anything package: pip install segment-anything")
        from segment_anything import SamPredictor, sam_model_registry

        path = weights_path()
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".part")
            try:
                urllib.request.urlretrieve(WEIGHTS_URL, tmp)
            except Exception as e:
                raise ValueError(f"Could not download the SAM weights from Meta; check the internet connection ({type(e).__name__}: {str(e)[:160]})") from None
            tmp.rename(path)
        model = sam_model_registry["vit_b"](checkpoint=str(path)).eval()  # On the CPU: SAM's prompt encoder has MPS gaps
        _state["predictor"] = SamPredictor(model)
    return _state["predictor"]


def to_rgb8(img: np.ndarray) -> np.ndarray:
    """A [0, 1] grey or colour image as the uint8 RGB image SAM expects."""
    img = np.clip(img, 0, 1)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, -1)
    return (img * 255).astype(np.uint8)


def sam_mask(key, img: np.ndarray, pos, neg) -> np.ndarray:
    """SAM's mask from object (pos) and background (neg) clicks, (y, x) in full-resolution pixels. The image
    encoding, the slow part, is reused while the key (dataset, channel, slice, clipping) stays the same."""
    with _lock:
        pred = _predictor()
        if _state["image_key"] != key:
            pred.set_image(to_rgb8(img))
            _state["image_key"] = key
        pts = np.array([(x, y) for y, x in list(pos) + list(neg)], np.float32)
        labels = np.r_[np.ones(len(pos)), np.zeros(len(neg))].astype(np.int32)
        masks, _, _ = pred.predict(point_coords=pts, point_labels=labels, multimask_output=False)
    return masks[0].astype(bool)


def agree(mask: np.ndarray, key, img: np.ndarray, pos, neg) -> np.ndarray:
    """The pixels both the click segmentation and SAM call the structure."""
    return mask & sam_mask(key, img, pos, neg)
