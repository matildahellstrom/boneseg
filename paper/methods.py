"""Segmentation methods compared in the paper, all with the same inputs.

Interactive methods get the same simulated clicks. Methods that learn from labels get the same expert-labelled
development slices.

  otsu               Otsu threshold on the image, no input from the user
  rf_clicks          ilastik-style pixel classification: a random forest on a multiscale filter bank
                     (intensity, edges, texture) trained on small disks around the clicks
  sam / microsam     Segment Anything with the clicks as point prompts (original SAM ViT-B, and micro-SAM's
                     ViT-B fine-tuned on light microscopy)
  dino_clicks        boneseg from clicks (frozen DINOv2 prototypes, click-calibrated threshold)
  rf_labels          the random forest trained on labelled slices (dense labels, subsampled)
  dino_labels        boneseg's learned model trained on labelled slices
  dino_clicks_v2     boneseg from clicks with the boundary improvements: 2 x 2 feature passes, a tuned threshold
                     position, optional guided-filter edge snapping
  dino_clicks_v2_refined   the same, followed by the learned refiner trained on other samples
  dino_labels_v2     the learned model on 2 x 2 feature passes
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from skimage.feature import multiscale_basic_features
from skimage.filters import threshold_otsu
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier

from common import ROOT  # noqa: F401  (puts the repository on the path)
from boneseg.backbone import get_backbone, pick_device
from boneseg.head import segment_with_head, train_head
from boneseg.segment import SegmentationSettings, embed_image, segment

WEIGHTS = Path(os.environ.get("BONESEG_PAPER_WEIGHTS", Path.home() / ".cache" / "boneseg-paper"))
RF_SIDE = 768  # Random-forest features are computed on a copy at most this many pixels wide, as ilastik users often do


def otsu(img: np.ndarray) -> np.ndarray:
    return img >= threshold_otsu(img)


# ilastik-style pixel classification ---------------------------------------------------------------
def _rf_features(img: np.ndarray, sigma_max: float) -> tuple[np.ndarray, float]:
    f = min(1.0, RF_SIDE / max(img.shape))
    small = resize(img, (max(1, round(img.shape[0] * f)), max(1, round(img.shape[1] * f))), anti_aliasing=True) if f < 1 else img
    feats = multiscale_basic_features(small, intensity=True, edges=True, texture=True, sigma_min=0.7, sigma_max=sigma_max, channel_axis=None)
    return feats.astype(np.float32), f


@lru_cache(maxsize=64)
def _rf_features_cached(key, sigma_max):
    return _rf_features(_IMAGES[key], sigma_max)


_IMAGES: dict = {}


def _disk_pixels(points, f, shape, r=3):
    out = []
    for y, x in points:
        cy, cx = int(y * f), int(x * f)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy * dy + dx * dx <= r * r and 0 <= cy + dy < shape[0] and 0 <= cx + dx < shape[1]:
                    out.append((cy + dy, cx + dx))
    return out


def _upsample_mask(prob_small: np.ndarray, shape) -> np.ndarray:
    return resize(prob_small, shape, order=1) >= 0.5


def rf_clicks(key, img, pos, neg, sigma_max=16.0, seed=0) -> np.ndarray:
    _IMAGES[key] = img
    feats, f = _rf_features_cached(key, sigma_max)
    p = _disk_pixels(pos, f, feats.shape[:2])
    n = _disk_pixels(neg, f, feats.shape[:2])
    X = np.concatenate([feats[tuple(np.array(p).T)], feats[tuple(np.array(n).T)]])
    y = np.r_[np.ones(len(p)), np.zeros(len(n))]
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed, class_weight="balanced").fit(X, y)
    prob = rf.predict_proba(feats.reshape(-1, feats.shape[-1]))[:, 1].reshape(feats.shape[:2])
    return _upsample_mask(prob, img.shape)


def rf_labels_fit(train: list, sigma_max=16.0, per_slice=20000, seed=0):
    """Random forest on labelled slices: [(key, img, gt)]. Pixels are subsampled and balanced per slice."""
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for key, img, gt in train:
        _IMAGES[key] = img
        feats, f = _rf_features_cached(key, sigma_max)
        g = resize(gt.astype(np.float32), feats.shape[:2], order=0) > 0.5
        for cls in (True, False):
            idx = np.flatnonzero(g.ravel() == cls)
            if len(idx):
                take = rng.choice(idx, min(per_slice // 2, len(idx)), replace=False)
                Xs.append(feats.reshape(-1, feats.shape[-1])[take])
                ys.append(np.full(len(take), float(cls)))
    return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed, class_weight="balanced").fit(np.concatenate(Xs), np.concatenate(ys))


def rf_labels_predict(rf, key, img, sigma_max=16.0) -> np.ndarray:
    _IMAGES[key] = img
    feats, _ = _rf_features_cached(key, sigma_max)
    prob = rf.predict_proba(feats.reshape(-1, feats.shape[-1]))[:, 1].reshape(feats.shape[:2])
    return _upsample_mask(prob, img.shape)


# Segment Anything ------------------------------------------------------------------------------
@lru_cache(maxsize=2)
def _sam_predictor(kind: str):
    from segment_anything import SamPredictor, sam_model_registry

    ckpt = WEIGHTS / ("sam_vit_b.pth" if kind == "sam" else "microsam_vit_b_lm.pt")
    model = sam_model_registry["vit_b"](checkpoint=str(ckpt))
    dev = pick_device()
    model.to(dev if dev.type != "mps" else torch.device("cpu"))  # SAM's prompt encoder has MPS gaps; CPU is reliable
    return SamPredictor(model)


_SAM_IMAGE: dict = {}


def sam_points(kind: str, key, img, pos, neg) -> np.ndarray:
    pred = _sam_predictor(kind)
    if _SAM_IMAGE.get(kind) != key:
        rgb = (np.repeat(np.clip(img, 0, 1)[..., None], 3, -1) * 255).astype(np.uint8)
        pred.set_image(rgb)
        _SAM_IMAGE[kind] = key
    pts = np.array([(x, y) for y, x in list(pos) + list(neg)], np.float32)
    labels = np.r_[np.ones(len(pos)), np.zeros(len(neg))].astype(np.int32)
    masks, scores, _ = pred.predict(point_coords=pts, point_labels=labels, multimask_output=False)
    return masks[0].astype(bool)


# boneseg -------------------------------------------------------------------------------------
_EMB: dict = {}


def dino_embedding(key, img, settings: SegmentationSettings):
    k = (key, settings.backbone, settings.vit_size, settings.layer_from_end, settings.shift_passes)
    if k not in _EMB:
        _EMB[k] = embed_image(get_backbone(settings.backbone), img, settings)
    return _EMB[k]


def dino_clicks(key, img, pos, neg, settings: SegmentationSettings, pixel_um) -> np.ndarray:
    return segment(dino_embedding(key, img, settings), pos, neg, settings, pixel_um).mask


def refiner_fit(slices, settings_for_budget: dict, budgets, seeds, simulated_clicks, steps=1500, log=None):
    """Trains a refiner on labelled slices, several simulated click draws each (every second draw noisy),
    each budget with the settings tuned for it."""
    from boneseg.refine import click_example, train_refiner
    def examples():
        for sl in slices:
            for budget in budgets:
                st = settings_for_budget[budget]
                for seed in seeds:
                    pos, neg = simulated_clicks(sl.gt, *budget, 1000 + seed, seed % 2 == 1)
                    raw, thr = click_example(dino_embedding((sl.sample, sl.z), sl.img, st), pos, neg, st)
                    yield sl.img, raw, thr, sl.gt
    return train_refiner(examples(), steps=steps, log=log)


def dino_labels_fit(train: list, settings: SegmentationSettings, pixel_um):
    samples = [(dino_embedding(key, img, settings), gt) for key, img, gt in train]
    return train_head(samples, settings, [(0, i) for i in range(len(samples))], pixel_um=pixel_um)


def dino_labels_predict(head, key, img, settings: SegmentationSettings, pixel_um) -> np.ndarray:
    return segment_with_head(head, dino_embedding(key, img, settings), settings, pixel_um).mask




