"""micro-SAM (Archit et al., Nature Methods 2025): fine-tuning on a few labelled images and automatic segmentation.

This file only imports numpy, torch and micro_sam, so it runs unchanged in boneseg and in the Kaggle notebooks
(they paste its source), and a model trained on Kaggle predicts the same way in the app.

Recipe (micro-SAM's own training code; only the data handling is ours):
- start from vit_b_lm, micro-SAM's light-microscopy generalist, the paper's recommended starting point
- train SAM and the UNETR decoder together (micro-SAM's JointSamTrainer): SAM learns from point and box prompts
  sampled from the labelled objects, with iterative mask refinement; the decoder learns foreground and distance maps
- "full" updates every weight; "lora" freezes the image encoder and adds rank-4 LoRA to its attention (micro-SAM's
  parameter-efficient fine-tuning follow-up, Teuber et al. 2025)
- AdamW, learning rate 1e-5, ReduceLROnPlateau, early stopping on the validation images (micro-SAM's defaults)
Automatic segmentation: the decoder's foreground probability >= 0.5, predicted in 512-pixel tiles like the patches
seen in training. Prompted segmentation: SAM with points or boxes, as in the original SAM.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

MODEL = "vit_b_lm"
RECIPES = ("full", "lora")
TILE, HALO = (512, 512), (64, 64)


def peft_kwargs(recipe: str):
    if recipe == "full":
        return None
    from micro_sam.models.peft_sam import LoRASurgery
    return {"rank": 4, "peft_module": LoRASurgery}


def instances(mask: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
    """micro-SAM trains on objects: given instance labels, or else the connected components of a binary mask."""
    if labels is not None:
        return labels.astype(np.int64)
    from scipy import ndimage
    return ndimage.label(mask > 0)[0].astype(np.int64)


def to_uint8(img: np.ndarray) -> np.ndarray:
    """micro-SAM expects 8-bit values (0-255); images in [0, 1] are scaled."""
    if img.dtype == np.uint8:
        return img
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def _loader(images, labels, n_samples, train):
    from micro_sam.training import default_sam_loader
    side = min(512, *(min(i.shape[:2]) for i in images))       # Patches no larger than the smallest image
    colour = images[0].ndim == 3
    if colour:   # micro-SAM's loader takes colour arrays channels first (3, H, W)
        images = [np.ascontiguousarray(i.transpose(2, 0, 1)) for i in images]
    return default_sam_loader(raw_paths=list(images), raw_key=None, label_paths=list(labels), label_key=None,
                              patch_shape=(side, side), with_segmentation_decoder=True, with_channels=colour,
                              n_samples=n_samples, is_train=train, min_size=25, batch_size=1, shuffle=train)


def _free_gpu():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train(images, labels, val_images, val_labels, out_dir, name="microsam", recipe="full", epochs=40,
          per_epoch=25, patience=8, n_objects=10, device=None, log=print) -> Path | None:
    """Fine-tunes micro-SAM; images are uint8 (H, W) or (H, W, 3), labels instance labels of the same size.
    Returns the best checkpoint (lowest validation loss), or None if training failed."""
    from micro_sam.training import train_sam
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    t = time.time()
    while True:
        _free_gpu()   # Memory left by an earlier run in the same session made full fine-tuning run out on a 16 GB T4
        try:
            train_sam(name, MODEL, _loader(images, labels, per_epoch, True), _loader(val_images, val_labels, 10, False),
                      n_epochs=epochs, early_stopping=patience, n_objects_per_batch=n_objects, device=device,
                      save_root=str(out_dir), peft_kwargs=peft_kwargs(recipe), verify_n_labels_in_loader=None)
            break
        except torch.OutOfMemoryError:
            if n_objects <= 2:
                log(f"{name}: training failed (out of GPU memory even with 2 objects per batch)")
                return None
            n_objects //= 2   # Fewer objects per image and step need less memory; training starts again
            log(f"{name}: out of GPU memory, starting again with {n_objects} objects per batch")
        except Exception as e:
            import traceback
            traceback.print_exc()
            log(f"{name}: training failed ({type(e).__name__}: {e})")
            return None
    _free_gpu()
    ck = Path(out_dir) / "checkpoints" / name / "best.pt"
    log(f"{name}: {'trained' if ck.exists() else 'no checkpoint written'} in {time.time() - t:.0f} s")
    return ck if ck.exists() else None


def export(checkpoint, out_path, recipe: str, **info) -> Path:
    """Keeps only the weights of a training checkpoint (528 MB with optimizer state and pickled classes) in a file
    that loads with torch.load(weights_only=True), so the app never unpickles code from a downloaded file."""
    st = torch.load(checkpoint, map_location="cpu", weights_only=False)
    slim = {"model_state": st["model_state"], "decoder_state": st["decoder_state"],
            "boneseg": {"model": MODEL, "recipe": recipe, "best_epoch": int(st.get("best_epoch", -1)), **info}}
    torch.save(slim, out_path)
    return Path(out_path)


def recipe_of(state: dict) -> str:
    """'lora' if the weights hold LoRA matrices, else 'full'."""
    return "lora" if any("w_a_linear" in k for k in state["model_state"]) else "full"


class Model:
    """A fine-tuned (or, without a checkpoint, the generalist) micro-SAM: SAM predictor plus the UNETR decoder."""

    def __init__(self, checkpoint=None, recipe="full", device=None):
        from micro_sam.instance_segmentation import TiledInstanceSegmentationWithDecoder, get_predictor_and_decoder
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor, decoder = get_predictor_and_decoder(
            MODEL, None if checkpoint is None else str(checkpoint), device=self.device,
            peft_kwargs=None if checkpoint is None else peft_kwargs(recipe))
        self.segmenter = TiledInstanceSegmentationWithDecoder(self.predictor, decoder)

    def foreground(self, img: np.ndarray) -> np.ndarray:
        """Foreground probability (H, W) of the decoder, tiled."""
        from micro_sam import util
        img = to_uint8(img)
        # ndim=2: a colour image (H, W, 3) is one 2D image, not a 3-slice volume
        emb = util.precompute_image_embeddings(self.predictor, img, ndim=2, tile_shape=TILE, halo=HALO, verbose=False)
        self.segmenter.initialize(img, emb, tile_shape=TILE, halo=HALO, verbose=False)
        return np.asarray(self.segmenter._foreground, np.float32)

    def mask(self, img: np.ndarray) -> np.ndarray:
        return self.foreground(img) >= 0.5

    def set_image(self, img: np.ndarray):
        """Whole image for prompts, resized to SAM's 1024 pixels as in the original SAM."""
        img = to_uint8(img)
        self.predictor.set_image(img if img.ndim == 3 else np.repeat(img[..., None], 3, -1))

    def points(self, pos, neg) -> np.ndarray:
        """Mask from object (pos) and background (neg) points, (y, x) pixels; call set_image first."""
        pts = np.array([(x, y) for y, x in list(pos) + list(neg)], np.float32)
        lab = np.r_[np.ones(len(pos)), np.zeros(len(neg))].astype(np.int32)
        return self.predictor.predict(point_coords=pts, point_labels=lab, multimask_output=False)[0][0].astype(bool)

    def boxes(self, boxes) -> np.ndarray:
        """Union of the masks from boxes (x0, y0, x1, y1); call set_image first."""
        m = None
        for b in boxes:
            r = self.predictor.predict(box=np.asarray(b, np.float32), multimask_output=False)[0][0].astype(bool)
            m = r if m is None else m | r
        return m
