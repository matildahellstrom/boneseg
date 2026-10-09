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


CONTEXT = 5  # Neighbourhood, in patches, whose mean features are added to each patch's own


def features(emb: Embedding, context: int) -> torch.Tensor:
    """Patch features for the head, flattened to [N, D]. With context > 1, each patch also gets the mean
    features of the context x context patches around it. On Liu file A this raised Dice on unseen
    slices from 0.67 to 0.69 with four labelled slices, and from 0.64 to 0.67 with one."""
    g = emb.grid.float().cpu()
    if context > 1:
        # The neighbourhood is measured in patches, so a model keeps its spatial context on a finer grid
        k = context * getattr(emb, "passes", 1)
        k += 1 - k % 2
        t = g.permute(2, 0, 1)[None]
        ctx = F.avg_pool2d(t, k, stride=1, padding=k // 2, count_include_pad=False)[0].permute(1, 2, 0)
        g = torch.cat([g, ctx], -1)
    return g.reshape(-1, g.shape[-1])


def with_z_context(emb: Embedding, below: Embedding, above: Embedding) -> Embedding:
    """The slice's features with the mean features of a slice below and one above appended, so a learned model
    sees where the structure continues in z. Both neighbours must come from the same stack at the same settings."""
    from dataclasses import replace

    nb = F.normalize((below.grid.float() + above.grid.float()) / 2, dim=-1)
    return replace(emb, grid=torch.cat([emb.grid.float(), nb.to(emb.grid.device)], -1))


Z_CONTEXT_UM = 4.0   # Neighbours this far below and above; on the five Liu stacks (2 um slices, so 2 slices) this
                     # raised 3D Dice of the learned model on every sample (mean 0.804 -> 0.810) and brought
                     # Tb.N and Tb.Sp within 1% of the experts' (paper/results/stack3d_summary.md)


def z_context_slices(z_um: float, n_z: int) -> int:
    """How many slices away the z-context neighbours sit for a stack with this slice spacing; 0 for single images."""
    d = max(1, min(3, int(round(Z_CONTEXT_UM / z_um)))) if z_um > 0 else 1
    return d if n_z >= 2 * d + 1 else 0


def head_input(head: "Head", embed_at, z: int, n_z: int) -> Embedding:
    """The embedding a head expects for slice z: its own, or with the neighbours' features for a z-context model.
    embed_at(z) returns a slice's embedding; neighbours beyond the stack ends are clamped to the end slices."""
    emb = embed_at(z)
    d = getattr(head, "z_context", 0)
    if not d:
        return emb
    return with_z_context(emb, embed_at(max(z - d, 0)), embed_at(min(z + d, n_z - 1)))


def smooth_z(probs: list[np.ndarray], sigma: float) -> list[np.ndarray]:
    """Gaussian smoothing of a stack of probability maps along z (sigma in slices), before thresholding. The stack
    ends are mirrored."""
    from scipy.ndimage import gaussian_filter1d

    if sigma <= 0 or len(probs) < 2:
        return probs
    a = gaussian_filter1d(np.stack(probs).astype(np.float32), sigma, axis=0, mode="mirror")
    return list(a)


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
    context: int = 1                # Models saved before context was added use their patch features only
    names: list = field(default_factory=list)  # Structure names for a model of several structures; empty for one
    threshold: float = 0.5          # Probability threshold, calibrated on held-out labelled slices (0.5 for older models)
    z_context: int = 0              # Slices between a slice and the neighbours whose features it also sees; 0 for none

    @property
    def n_out(self) -> int:
        return len(self.names) + 1 if self.names else 1

    def module(self) -> torch.nn.Module:
        m = _make_module(self.kind, self.dim, self.n_out)
        m.load_state_dict(self.state)
        return m.eval()

    @torch.no_grad()
    def logits(self, emb: Embedding) -> torch.Tensor:
        out = self.module()(features(emb, self.context))
        hg, wg = emb.grid.shape[:2]
        return out.reshape(hg, wg) if self.n_out == 1 else out.reshape(hg, wg, self.n_out)

    def compatible(self, s: SegmentationSettings) -> bool:
        return (self.backbone, self.layer_from_end, self.vit_size) == (s.backbone, s.layer_from_end, s.vit_size)

    def save(self, path: str | Path):
        torch.save(self.__dict__, path)

    @classmethod
    def load(cls, path: str | Path) -> "Head":
        d = torch.load(path, map_location="cpu", weights_only=False)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def info(self) -> dict:
        return {"backbone": self.backbone, "layer_from_end": self.layer_from_end, "vit_size": self.vit_size,
                "kind": self.kind, "trained_on": self.trained_on, "cv": self.cv, "context": self.context, "names": self.names,
                "threshold": self.threshold, "z_context": self.z_context}


def _make_module(kind: str, dim: int, n_out: int = 1) -> torch.nn.Module:
    if kind == "linear":
        return torch.nn.Linear(dim, n_out)
    if kind == "mlp":
        return torch.nn.Sequential(torch.nn.Linear(dim, 64), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(64, n_out))
    raise ValueError(f"Unknown head kind {kind}")


def fit(samples: list[tuple[Embedding, np.ndarray]], kind: str = "linear", l2: float = 1e-3, epochs: int = 150, seed: int = 0,
        context: int = 1):
    """Fits a head on (embedding, mask) pairs. Patches are weighted so both classes count equally."""
    torch.manual_seed(seed)
    X = torch.cat([features(e, context) for e, _ in samples])
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
    """Probability map from the head, thresholded at the head's calibrated threshold unless the threshold mode says otherwise."""
    return mask_from_prob(head_prob(head, emb), head, settings, pixel_um)


def head_prob(head: Head, emb: Embedding) -> np.ndarray:
    """The head's probability map at the image's full resolution."""
    return upsample(torch.sigmoid(head.logits(emb)), (emb.height, emb.width)).clip(0, 1)


def mask_from_prob(prob: np.ndarray, head: Head, settings: SegmentationSettings, pixel_um=(1.0, 1.0)) -> SegmentationResult:
    """Thresholds a probability map of the head, for example one smoothed through the stack."""
    if settings.threshold_mode == "manual":
        thr, source = settings.manual_threshold, "manual"
    elif settings.threshold_mode == "top_percent":
        thr, source = float(np.percentile(prob, 100 - settings.top_percent)), "top_percent"
    else:
        thr = head.threshold
        source = f"learned (probability {thr:.2f}{', calibrated on held-out labels' if thr != 0.5 else ''})"
    mask = postprocess(prob >= thr, settings, pixel_um)
    return SegmentationResult(heat=prob.astype(np.float32), mask=mask, threshold=float(thr), raw_threshold=float(thr),
                              raw_score_range=(0.0, 1.0), threshold_source=source)


THRESHOLDS = np.round(np.arange(0.2, 0.91, 0.05), 2)


def calibrate_threshold(held_out: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, dict]:
    """The probability threshold with the best mean Dice over held-out (probability, mask) pairs. Training weights
    both classes equally, which pushes probabilities of a rare structure up, so 0.5 overestimates its area
    (bone +3.4 points of the image on the Liu test slices)."""
    scores = {float(t): float(np.mean([metrics.dice(p >= t, m) for p, m in held_out])) for t in THRESHOLDS}
    best = max(scores, key=lambda t: (round(scores[t], 4), -abs(t - 0.5)))   # Ties go to the threshold nearest 0.5
    return best, scores


def _small(a: np.ndarray, max_side: int = 512) -> np.ndarray:
    """Every k-th pixel, so the held-out maps kept for calibration stay small."""
    k = max(1, int(np.ceil(max(a.shape[:2]) / max_side)))
    return a[::k, ::k]


def cv_folds(n: int, max_folds: int = 5) -> list[list[int]]:
    """Leave-one-slice-out for up to six labelled slices, otherwise five interleaved folds, so that model selection
    stays fast when many slices are labelled (20 slices took over half an hour with leave-one-out)."""
    if n <= 6:
        return [[i] for i in range(n)]
    return [list(range(f, n, max_folds)) for f in range(max_folds)]


def train_head(samples: list[tuple[Embedding, np.ndarray]], settings: SegmentationSettings, keys: list,
               kind: str = "auto", pixel_um=(1.0, 1.0), context: int | None = None, calibrate: bool = True) -> Head:
    """Trains on all samples. With two or more labelled slices, cross-validation (leave-one-slice-out, or five
    folds beyond six slices) estimates the Dice on unseen slices and picks the best of a linear or MLP head, with or without
    neighbourhood features (unless kind or context is given). Neighbourhood features helped on Liu
    file A and cost a little on the small cells of the demo stack, so the data decides. With calibrate, the chosen
    model's probability threshold is then set on the same held-out predictions (see calibrate_threshold)."""
    kinds = ["linear", "mlp"] if kind == "auto" else [kind]
    contexts = [1, CONTEXT] if context is None else [context]
    candidates = [(k, c) for k in kinds for c in contexts]
    cv, held = {}, {}
    if len(samples) >= 2:
        for k, c in candidates:
            dices, key = [], f"{k}" if len(contexts) == 1 else f"{k}, context {c}"
            held[key] = []
            for fold in cv_folds(len(samples)):
                model, dim = fit([s for j, s in enumerate(samples) if j not in fold], k, context=c)
                h = Head(settings.backbone, settings.layer_from_end, settings.vit_size, k, model.state_dict(), dim, context=c)
                for i in fold:
                    e, m = samples[i]
                    prob = head_prob(h, e)
                    dices.append(metrics.dice(mask_from_prob(prob, h, settings, pixel_um).mask, m))
                    held[key].append((_small(prob), _small(m)))
            cv[key] = {"mean_dice": float(np.mean(dices)), "per_slice": [float(d) for d in dices], "kind": k, "context": c}
        best_key = max(cv, key=lambda key: cv[key]["mean_dice"])
        best, best_ctx = cv[best_key]["kind"], cv[best_key]["context"]
    else:
        best_key, best, best_ctx = None, kinds[0], contexts[-1]
    threshold = 0.5
    if calibrate and best_key is not None:
        threshold, scores = calibrate_threshold(held[best_key])
        cv["threshold"] = {"chosen": threshold, "mean_dice_at_0.5": scores[0.5], "mean_dice_at_chosen": scores[threshold]}
    model, dim = fit(samples, best, context=best_ctx)
    return Head(settings.backbone, settings.layer_from_end, settings.vit_size, best, model.state_dict(), dim,
                trained_on=[list(k) for k in keys], cv={"chosen": best_key, **cv}, context=best_ctx, threshold=threshold)


def head_to_profile_dict(head: Head) -> dict:
    return {"kind": head.kind, "dim": head.dim, "vit_size": head.vit_size, "context": head.context, "names": head.names, "threshold": head.threshold,
            "z_context": head.z_context,
            "state": {k: v.detach().cpu().numpy() for k, v in head.state.items()}, "cv": head.cv,
            "n_trained_on": len(head.trained_on)}


def head_from_profile(profile) -> Head:
    h = profile.head
    state = {k: torch.from_numpy(np.asarray(v)) for k, v in h["state"].items()}
    return Head(profile.backbone, profile.layer_from_end, h["vit_size"], h["kind"], state, h["dim"], cv=h.get("cv", {}),
                context=int(h.get("context", 1)), names=list(h.get("names", [])), threshold=float(h.get("threshold", 0.5)),
                z_context=int(h.get("z_context", 0)))


# Several structures ----------------------------------------------------------------------------------
def class_targets(labels: np.ndarray, emb: Embedding, n_classes: int) -> torch.Tensor:
    """Share of every patch covered by each class (0 is background), as an [N, n_classes] tensor."""
    hg, wg = emb.grid.shape[:2]
    one_hot = torch.stack([torch.from_numpy((labels == k).astype(np.float32)) for k in range(n_classes)])[None]
    return F.adaptive_avg_pool2d(one_hot, (hg, wg))[0].reshape(n_classes, -1).T


def fit_multi(samples, n_classes: int, kind: str = "linear", context: int = 1, epochs: int = 200, l2: float = 1e-3, seed: int = 0):
    """Soft cross-entropy over background and each structure, with classes weighted by inverse frequency."""
    torch.manual_seed(seed)
    X = torch.cat([features(e, context) for e, _ in samples])
    Y = torch.cat([class_targets(m, e, n_classes) for e, m in samples])
    freq = Y.mean(0).clamp_min(1e-3)
    w = (1 / freq) / (1 / freq).sum() * n_classes
    model = _make_module(kind, X.shape[1], n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2 if kind == "linear" else 3e-3, weight_decay=l2)
    for _ in range(epochs):
        opt.zero_grad()
        loss = -(Y * F.log_softmax(model(X), dim=1) * w).sum(1).mean()
        loss.backward()
        opt.step()
    return model.eval(), X.shape[1]


def labels_with_head(head: Head, emb: Embedding, settings: SegmentationSettings, pixel_um=(1.0, 1.0)) -> np.ndarray:
    """Label map from a model of several structures: each pixel takes the most probable class."""
    probs = torch.softmax(head.logits(emb), dim=-1).permute(2, 0, 1)
    up = F.interpolate(probs[None], size=(emb.height, emb.width), mode="bilinear", align_corners=False)[0]
    labels = up.argmax(0).numpy().astype(np.uint8)
    if settings.min_object_um2 or settings.fill_holes_um2 or settings.smooth_px:
        out = np.zeros_like(labels)
        for k in range(1, head.n_out):
            m = postprocess(labels == k, settings, pixel_um)
            out[m & (out == 0)] = k
        labels = out
    return labels


def train_head_multi(samples, names: list[str], settings: SegmentationSettings, keys: list, pixel_um=(1.0, 1.0)) -> Head:
    """Like train_head, for labels with several structures. Cross-validation scores the mean Dice over structures."""
    n = len(names) + 1
    candidates = [(k, c) for k in ("linear", "mlp") for c in (1, CONTEXT)]
    cv = {}
    if len(samples) >= 2:
        for k, c in candidates:
            scores = []
            for fold in cv_folds(len(samples)):
                model, dim = fit_multi([s for j, s in enumerate(samples) if j not in fold], n, k, c)
                h = Head(settings.backbone, settings.layer_from_end, settings.vit_size, k, model.state_dict(), dim, context=c, names=list(names))
                for i in fold:
                    e, m = samples[i]
                    pred = labels_with_head(h, e, settings, pixel_um)
                    present = [cls for cls in range(1, n) if (m == cls).any()]
                    if present:
                        scores.append(float(np.mean([metrics.dice(pred == cls, m == cls) for cls in present])))
            cv[f"{k}, context {c}"] = {"mean_dice": float(np.mean(scores)) if scores else 0.0, "per_slice": scores, "kind": k, "context": c}
        best_key = max(cv, key=lambda key: cv[key]["mean_dice"])
        best, best_ctx = cv[best_key]["kind"], cv[best_key]["context"]
    else:
        best_key, best, best_ctx = None, "linear", CONTEXT
    model, dim = fit_multi(samples, n, best, best_ctx)
    return Head(settings.backbone, settings.layer_from_end, settings.vit_size, best, model.state_dict(), dim,
                trained_on=[list(k) for k in keys], cv={"chosen": best_key, **cv}, context=best_ctx, names=list(names))
