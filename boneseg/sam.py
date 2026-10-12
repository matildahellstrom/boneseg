"""An optional second opinion from Segment Anything on a click segmentation, or its mask instead.

Two models take the same object and background clicks as point prompts:
- "sam": the original SAM ViT-B (Meta)
- "microsam": micro-SAM's ViT-B fine-tuned on light microscopy (vit_b_lm; Archit et al., Nature Methods 2025)
"agree" keeps only the pixels both boneseg and the model call the structure; "replace" uses the model's mask alone.
With SAM ViT-B and 25 + 25 clicks on the test slices of the five Liu samples, "agree" raised Dice from 0.755 to 0.784
(+0.029, 95% CI +0.013 to +0.047) and removed boneseg's tendency to overestimate bone area; with 3 + 6 clicks it made
no difference (-0.003) and underestimated bone area by 2.4 points (paper/results/combine_sam_summary.md). Both need the
segment-anything package; the weights (375 MB each) are downloaded on first use.
"""
from __future__ import annotations

import os
import threading
import urllib.request
from pathlib import Path

import numpy as np
import torch

# model -> (download URL, file name in the torch hub cache, environment variable that overrides the path)
MODELS = {
    "sam": ("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", "sam_vit_b_01ec64.pth", "BONESEG_SAM_WEIGHTS"),
    "microsam": ("https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/diplomatic-bug/1.2/files/vit_b.pt",
                 "microsam_vit_b_lm.pt", "BONESEG_MICROSAM_WEIGHTS"),
}
NAMES = {"sam": "SAM", "microsam": "micro-SAM"}
WEIGHTS_URL = MODELS["sam"][0]
_lock = threading.Lock()
_state: dict = {}   # model -> {"predictor", "image_key"}


def weights_path(model: str = "sam") -> Path:
    url, name, env = MODELS[model]
    return Path(os.environ.get(env, Path(torch.hub.get_dir()) / "checkpoints" / name))


def available() -> bool:
    try:
        import segment_anything  # noqa: F401
        return True
    except ImportError:
        return False


def weights_cached(model: str = "sam") -> bool:
    return weights_path(model).exists()


def _predictor(model: str):
    st = _state.setdefault(model, {"predictor": None, "image_key": None})
    if st["predictor"] is None:
        if not available():
            raise ValueError(f"{NAMES[model]} needs the segment-anything package: pip install segment-anything")
        from segment_anything import SamPredictor, sam_model_registry

        path = weights_path(model)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".part")
            try:
                urllib.request.urlretrieve(MODELS[model][0], tmp)
            except Exception as e:
                raise ValueError(f"Could not download the {NAMES[model]} weights; check the internet connection ({type(e).__name__}: {str(e)[:160]})") from None
            tmp.rename(path)
        # micro-SAM's checkpoints are SAM checkpoints with other weights. On the CPU: SAM's prompt encoder has MPS gaps
        net = sam_model_registry["vit_b"](checkpoint=str(path)).eval()
        st["predictor"] = SamPredictor(net)
    return st["predictor"]


def to_rgb8(img: np.ndarray) -> np.ndarray:
    """A [0, 1] grey or colour image as the uint8 RGB image SAM expects."""
    img = np.clip(img, 0, 1)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, -1)
    return (img * 255).astype(np.uint8)


def sam_mask(key, img: np.ndarray, pos, neg, model: str = "sam") -> np.ndarray:
    """The model's mask from object (pos) and background (neg) clicks, (y, x) in full-resolution pixels. The image
    encoding, the slow part, is reused while the key (dataset, channel, slice, clipping) stays the same."""
    with _lock:
        pred = _predictor(model)
        st = _state[model]
        if st["image_key"] != key:
            pred.set_image(to_rgb8(img))
            st["image_key"] = key
        pts = np.array([(x, y) for y, x in list(pos) + list(neg)], np.float32)
        labels = np.r_[np.ones(len(pos)), np.zeros(len(neg))].astype(np.int32)
        masks, _, _ = pred.predict(point_coords=pts, point_labels=labels, multimask_output=False)
    return masks[0].astype(bool)


def agree(mask: np.ndarray, key, img: np.ndarray, pos, neg, model: str = "sam") -> np.ndarray:
    """The pixels both the click segmentation and the model call the structure."""
    return mask & sam_mask(key, img, pos, neg, model)


def apply(mode: str, mask: np.ndarray, key, img: np.ndarray, pos, neg, model: str = "sam") -> np.ndarray:
    """mode "agree": the overlap with the model's mask; "replace": the model's mask alone."""
    return agree(mask, key, img, pos, neg, model) if mode == "agree" else sam_mask(key, img, pos, neg, model)
