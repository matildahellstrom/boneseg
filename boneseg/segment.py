"""Few-shot segmentation from clicked reference points.

The pipeline follows the notebook: the patches under positive and negative clicks
become prototypes, every patch gets the score mean(pos similarity) - lambda * mean(neg
similarity), and the upsampled score map is thresholded into a mask.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F
from skimage import measure, morphology  # Imported eagerly: lazy imports deadlock across server threads
from skimage.filters import threshold_otsu

from .backbone import Backbone


@dataclass
class SegmentationSettings:
    backbone: str = "dinov2_s14"
    vit_size: int = 980              # Longest side of the backbone input, a multiple of 14
    layer_from_end: int = 1          # Transformer block counted from the end
    neg_weight: float = 0.8          # Lambda in pos - lambda * neg
    threshold_mode: str = "otsu"     # "otsu", "top_percent" or "manual"
    top_percent: float = 10.0
    manual_threshold: float = 0.5    # On the heatmap rescaled to [0, 1]
    min_object_um2: float = 0.0      # Objects smaller than this are removed
    fill_holes_um2: float = 0.0      # Holes smaller than this are filled
    smooth_px: int = 0               # Radius of a morphological opening and closing
    clip_low: float = 1.0
    clip_high: float = 99.5

    @classmethod
    def from_dict(cls, d: dict | None) -> "SegmentationSettings":
        d = d or {}
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Embedding:
    grid: torch.Tensor   # [H_grid, W_grid, D], L2-normalized
    height: int          # Original image height
    width: int


def vit_input_size(h: int, w: int, vit_size: int, patch: int) -> tuple[int, int]:
    """Backbone input size that keeps the aspect ratio, rounded to whole patches."""
    scale = vit_size / max(h, w)
    return max(patch, int(round(h * scale / patch)) * patch), max(patch, int(round(w * scale / patch)) * patch)


def embed_image(backbone: Backbone, img: np.ndarray, settings: SegmentationSettings) -> Embedding:
    """Embeds a normalized [0, 1] image."""
    h, w = img.shape
    in_h, in_w = vit_input_size(h, w, settings.vit_size, backbone.patch_size)
    grid = backbone.embed(torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)), in_h, in_w, settings.layer_from_end)
    return Embedding(grid=grid, height=h, width=w)


def points_to_cells(points, h, w, h_grid, w_grid) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    ys = np.clip((pts[:, 0] / h * h_grid).astype(int), 0, h_grid - 1)
    xs = np.clip((pts[:, 1] / w * w_grid).astype(int), 0, w_grid - 1)
    return ys, xs


def prototypes(emb: Embedding, points) -> torch.Tensor:
    """Embeddings of the patches under the given (y, x) points, as an [N, D] tensor."""
    if len(points) == 0:
        return emb.grid.new_zeros((0, emb.grid.shape[-1]))
    hg, wg = emb.grid.shape[:2]
    ys, xs = points_to_cells(points, emb.height, emb.width, hg, wg)
    return emb.grid[torch.as_tensor(ys, device=emb.grid.device), torch.as_tensor(xs, device=emb.grid.device)]


@torch.no_grad()
def score_grid(grid: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor, neg_weight: float) -> torch.Tensor:
    """Patch-level score, mean positive similarity minus lambda times mean negative similarity."""
    if pos.shape[0] == 0:
        raise ValueError("At least one positive point or prototype is needed")
    H, W, D = grid.shape
    feats = grid.reshape(-1, D)
    pos = F.normalize(pos.to(grid.device, grid.dtype), dim=-1)
    score = (feats @ pos.T).mean(dim=1)
    if neg.shape[0] > 0:
        neg = F.normalize(neg.to(grid.device, grid.dtype), dim=-1)
        score = score - neg_weight * (feats @ neg.T).mean(dim=1)
    return score.reshape(H, W)


def upsample(score: torch.Tensor, shape: tuple[int, int]) -> np.ndarray:
    out = F.interpolate(score[None, None].float(), size=tuple(shape), mode="bilinear", align_corners=False)[0, 0]
    return out.cpu().numpy()


def rescale01(a: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(a)), float(np.max(a))
    return np.zeros_like(a, dtype=np.float32) if hi - lo < 1e-12 else ((a - lo) / (hi - lo)).astype(np.float32)


def threshold_heatmap(heat: np.ndarray, settings: SegmentationSettings) -> tuple[np.ndarray, float]:
    """Binary mask from a heatmap already rescaled to [0, 1]. Returns the mask and the threshold used."""
    mode = settings.threshold_mode
    if mode == "otsu":
        thr = float(threshold_otsu(heat)) if np.ptp(heat) > 0 else 1.0
    elif mode == "top_percent":
        thr = float(np.percentile(heat, 100 - settings.top_percent))
    elif mode == "manual":
        thr = float(settings.manual_threshold)
    else:
        raise ValueError(f"Unknown threshold mode {mode}")
    return heat >= thr, thr


def remove_small(mask: np.ndarray, min_px: int) -> np.ndarray:
    """Removes connected objects with fewer than min_px pixels."""
    labels, n = ndi.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[labels]


def postprocess(mask: np.ndarray, settings: SegmentationSettings, pixel_um: tuple[float, float]) -> np.ndarray:
    """Optional smoothing, hole filling and removal of small objects, with sizes in square micrometres."""

    px_area = float(pixel_um[0] * pixel_um[1])
    mask = mask.astype(bool)
    if settings.smooth_px > 0:
        fp = morphology.disk(settings.smooth_px)
        mask = ndi.binary_closing(ndi.binary_opening(mask, fp), fp)
    if settings.fill_holes_um2 > 0:
        # Holes are background objects that do not touch the border
        holes = ~mask & ~remove_small(~mask, max(1, int(settings.fill_holes_um2 / px_area)))
        border = np.zeros_like(mask)
        border[[0, -1], :] = border[:, [0, -1]] = True
        labels, _ = ndi.label(~mask)
        touching = np.unique(labels[border & ~mask])
        holes &= ~np.isin(labels, touching[touching > 0])
        mask = mask | holes
    if settings.min_object_um2 > 0:
        mask = remove_small(mask, max(1, int(settings.min_object_um2 / px_area)))
    return mask.astype(bool)


@dataclass
class SegmentationResult:
    heat: np.ndarray        # Heatmap rescaled to [0, 1], full resolution
    mask: np.ndarray        # Final binary mask after post-processing
    threshold: float
    raw_score_range: tuple[float, float] = (0.0, 0.0)
    uncertainty: np.ndarray | None = None
    suggestion: tuple[int, int] | None = None
    extra: dict = field(default_factory=dict)


def segment_with_prototypes(emb: Embedding, pos: torch.Tensor, neg: torch.Tensor, settings: SegmentationSettings,
                            pixel_um=(1.0, 1.0)) -> SegmentationResult:
    score = score_grid(emb.grid, pos, neg, settings.neg_weight)
    raw = upsample(score, (emb.height, emb.width))
    heat = rescale01(raw)
    mask, thr = threshold_heatmap(heat, settings)
    mask = postprocess(mask, settings, pixel_um)
    return SegmentationResult(heat=heat, mask=mask, threshold=thr, raw_score_range=(float(raw.min()), float(raw.max())))


def segment(emb: Embedding, pos_points, neg_points, settings: SegmentationSettings, pixel_um=(1.0, 1.0)) -> SegmentationResult:
    return segment_with_prototypes(emb, prototypes(emb, pos_points), prototypes(emb, neg_points), settings, pixel_um)


@torch.no_grad()
def uncertainty_map(emb: Embedding, pos: torch.Tensor, neg: torch.Tensor, settings: SegmentationSettings,
                    n_boot: int = 16, seed: int = 0) -> np.ndarray:
    """Where the mask depends on individual clicks.

    With few clicks, each one is left out in turn (a jackknife). With many clicks, random 80% subsets
    are drawn instead. Every draw is scored on the same scale and with the same threshold as the full
    result, so only the change in clicks matters. The output is the fraction of draws that disagree
    with the full mask, from 0 (stable) to 1. Works at patch resolution, so it is cheap."""
    full = score_grid(emb.grid, pos, neg, settings.neg_weight)
    lo, hi = float(full.min()), float(full.max())
    norm = lambda s: ((s - lo) / max(hi - lo, 1e-12)).clamp(0, 1)
    heat_full = upsample(full, (emb.height, emb.width))
    _, thr = threshold_heatmap(rescale01(heat_full), settings)
    full_mask = norm(full) >= thr
    n_pos, n_neg = pos.shape[0], neg.shape[0]
    draws = []
    if n_pos + n_neg <= 24:
        for i in range(n_pos):
            if n_pos > 1:
                keep = torch.arange(n_pos, device=pos.device) != i
                draws.append((pos[keep], neg))
        for i in range(n_neg):
            keep = torch.arange(n_neg, device=neg.device) != i
            draws.append((pos, neg[keep]))
    else:
        g = torch.Generator().manual_seed(seed)
        for _ in range(n_boot):
            pi = torch.randperm(n_pos, generator=g)[: max(1, int(0.8 * n_pos))].to(pos.device)
            ni = torch.randperm(n_neg, generator=g)[: int(0.8 * n_neg)].to(neg.device)
            draws.append((pos[pi], neg[ni]))
    if not draws:
        return np.zeros((emb.height, emb.width), np.float32)
    disagree = torch.zeros_like(full)
    for p, n in draws:
        disagree += (norm(score_grid(emb.grid, p, n, settings.neg_weight)) >= thr) != full_mask
    u = (disagree / len(draws)).float()
    return upsample(u, (emb.height, emb.width)).clip(0, 1)


def suggest_click(uncertainty: np.ndarray, existing_points, min_dist_frac: float = 0.05) -> tuple[int, int] | None:
    """The most uncertain location that is not close to an existing click."""
    if uncertainty.max() <= 1e-6:
        return None
    h, w = uncertainty.shape
    u = ndi.gaussian_filter(uncertainty, sigma=max(1.0, min(h, w) / 100))
    if len(existing_points):
        yy, xx = np.ogrid[:h, :w]
        r = min_dist_frac * max(h, w)
        for y, x in existing_points:
            u[(yy - y) ** 2 + (xx - x) ** 2 < r ** 2] = 0
    y, x = np.unravel_index(int(np.argmax(u)), u.shape)
    return (int(y), int(x)) if u[y, x] > 1e-6 else None


@dataclass
class Profile:
    """Saved positive and negative prototypes that can segment new images without new clicks.
    The transfer experiments in the notebook showed that prototypes carry over between slices
    of the same sample, and partly between samples."""

    name: str
    backbone: str
    layer_from_end: int
    pos: np.ndarray
    neg: np.ndarray
    settings: dict
    description: str = ""
    source: str = ""

    def save(self, path) -> None:
        np.savez(path, pos=self.pos, neg=self.neg, meta=np.array(
            [repr({"name": self.name, "backbone": self.backbone, "layer_from_end": self.layer_from_end,
                   "settings": self.settings, "description": self.description, "source": self.source})]))

    @classmethod
    def load(cls, path) -> "Profile":
        import ast

        d = np.load(path, allow_pickle=False)
        meta = ast.literal_eval(str(d["meta"][0]))
        return cls(pos=d["pos"], neg=d["neg"], **meta)

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.pos.astype(np.float32)), torch.from_numpy(self.neg.astype(np.float32))
