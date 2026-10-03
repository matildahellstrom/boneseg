"""A small model trained on corrected masks, on top of the frozen backbone features.

This is the app's version of the notebook's supervised U-Net step, kept light enough to train in
seconds: a linear or small two-layer classifier over the patch embeddings, fitted to the share of
each patch that the user's corrected mask covers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import metrics
from .segment import Embedding, SegmentationResult, SegmentationSettings, postprocess, upsample


def patch_targets(mask: np.ndarray, emb: Embedding) -> torch.Tensor:
    """Share of every patch covered by the mask, flattened to match the embedding grid."""
    hg, wg = emb.grid.shape[:2]
    m = torch.from_numpy(mask.astype(np.float32))[None, None]
    return F.adaptive_avg_pool2d(m, (hg, wg))[0, 0].reshape(-1)


@dataclass
class Head:
    backbone: str
    layer_from_end: int
    vit_size: int
    kind: str                       # "linear" or "mlp"
    state: dict                     # Model weights
    dim: int
    trained_on: list = field(default_factory=list)  # [(channel, z)]
    cv: dict = field(default_factory=dict)

    def module(self) -> torch.nn.Module:
        m = _make_module(self.kind, self.dim)
        m.load_state_dict(self.state)
        return m.eval()

    @torch.no_grad()
    def logits(self, emb: Embedding) -> torch.Tensor:
        g = emb.grid.float().cpu()
        return self.module()(g.reshape(-1, g.shape[-1])).reshape(g.shape[:2])

    def compatible(self, s: SegmentationSettings) -> bool:
        return (self.backbone, self.layer_from_end, self.vit_size) == (s.backbone, s.layer_from_end, s.vit_size)

    def save(self, path: str | Path):
        torch.save(self.__dict__, path)

    @classmethod
    def load(cls, path: str | Path) -> "Head":
        return cls(**torch.load(path, map_location="cpu", weights_only=False))

    def info(self) -> dict:
        return {"backbone": self.backbone, "layer_from_end": self.layer_from_end, "vit_size": self.vit_size,
                "kind": self.kind, "trained_on": self.trained_on, "cv": self.cv}


def _make_module(kind: str, dim: int) -> torch.nn.Module:
    if kind == "linear":
        return torch.nn.Linear(dim, 1)
    if kind == "mlp":
        return torch.nn.Sequential(torch.nn.Linear(dim, 64), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(64, 1))
    raise ValueError(f"Unknown head kind {kind}")


def fit(samples: list[tuple[Embedding, np.ndarray]], kind: str = "linear", l2: float = 1e-3, epochs: int = 150, seed: int = 0):
    """Fits a head on (embedding, mask) pairs. Patches are weighted so both classes count equally."""
    torch.manual_seed(seed)
    X = torch.cat([e.grid.float().cpu().reshape(-1, e.grid.shape[-1]) for e, _ in samples])
    y = torch.cat([patch_targets(m, e) for e, m in samples])
    pos_share = float(y.mean().clamp(1e-3, 1 - 1e-3))
    w = torch.where(y > 0.5, 0.5 / pos_share, 0.5 / (1 - pos_share))
    model = _make_module(kind, X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2 if kind == "linear" else 3e-3, weight_decay=l2)
    for _ in range(epochs):
        opt.zero_grad()
        loss = (F.binary_cross_entropy_with_logits(model(X).squeeze(1), y, reduction="none") * w).mean()
        loss.backward()
        opt.step()
    return model.eval(), X.shape[1]


def segment_with_head(head: Head, emb: Embedding, settings: SegmentationSettings, pixel_um=(1.0, 1.0)) -> SegmentationResult:
    """Probability map from the head, thresholded at 0.5 unless the threshold mode says otherwise."""
    logits = head.logits(emb)
    prob = upsample(torch.sigmoid(logits), (emb.height, emb.width)).clip(0, 1)
    if settings.threshold_mode == "manual":
        thr = settings.manual_threshold
    elif settings.threshold_mode == "top_percent":
        thr = float(np.percentile(prob, 100 - settings.top_percent))
    else:
        thr = 0.5
    mask = postprocess(prob >= thr, settings, pixel_um)
    return SegmentationResult(heat=prob.astype(np.float32), mask=mask, threshold=float(thr), raw_threshold=float(thr),
                              raw_score_range=(0.0, 1.0), threshold_source="learned (probability 0.5)" if thr == 0.5 else settings.threshold_mode)


def train_head(samples: list[tuple[Embedding, np.ndarray]], settings: SegmentationSettings, keys: list,
               kind: str = "auto", pixel_um=(1.0, 1.0)) -> Head:
    """Trains on all samples. With two or more labelled slices, leave-one-slice-out cross-validation
    estimates the Dice on unseen slices, and "auto" picks the better of a linear and an MLP head."""
    kinds = ["linear", "mlp"] if kind == "auto" else [kind]
    cv = {}
    if len(samples) >= 2:
        for k in kinds:
            dices = []
            for i in range(len(samples)):
                model, dim = fit([s for j, s in enumerate(samples) if j != i], k)
                h = Head(settings.backbone, settings.layer_from_end, settings.vit_size, k, model.state_dict(), dim)
                e, m = samples[i]
                dices.append(metrics.dice(segment_with_head(h, e, settings, pixel_um).mask, m))
            cv[k] = {"mean_dice": float(np.mean(dices)), "per_slice": [float(d) for d in dices]}
        best = max(kinds, key=lambda k: cv[k]["mean_dice"])
    else:
        best = kinds[0]
    model, dim = fit(samples, best)
    return Head(settings.backbone, settings.layer_from_end, settings.vit_size, best, model.state_dict(), dim,
                trained_on=[list(k) for k in keys], cv={"chosen": best, **cv})
