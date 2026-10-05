"""Few-shot segmentation from clicked reference points.

The pipeline follows the notebook: the patches under positive and negative clicks
become prototypes, every patch gets the score mean(pos similarity) - lambda * mean(neg
similarity), and the upsampled score map is thresholded into a mask.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F
from skimage import morphology  # Imported eagerly: lazy imports deadlock across server threads
from skimage.filters import threshold_otsu

from .backbone import Backbone


@dataclass
class SegmentationSettings:
    backbone: str = "dinov2_s14"
    vit_size: int = 980              # Longest side of the backbone input, a multiple of 14
    layer_from_end: int = 1          # Transformer block counted from the end
    neg_weight: float = 0.8          # Lambda in pos - lambda * neg
    threshold_mode: str = "clicks"   # "clicks", "otsu", "top_percent" or "manual"
    top_percent: float = 10.0
    manual_threshold: float = 0.5    # On the heatmap rescaled to [0, 1]
    min_object_um2: float = 0.0      # Objects smaller than this are removed
    fill_holes_um2: float = 0.0      # Holes smaller than this are filled
    smooth_px: int = 0               # Radius of a morphological opening and closing
    score_norm: str = "robust"       # "robust" standardizes scores per image by median and MAD, or "none"
    # Defaults below follow the four-sample nested evaluation (paper/README.md); method-v1 was 0.5, "none" and 1
    threshold_position: float = -1.0 # "clicks" mode: 0.5 is halfway between background and object clicks, higher is
                                     # stricter; -1 is automatic: 0.7 with five or fewer clicks of a kind, else 0.9
    edge_refine: str = "guided"      # "guided" snaps the score map to intensity edges of the image before thresholding
    guided_eps: float = 0.01         # Guided-filter edge sensitivity: smaller follows weaker edges
    shift_passes: int = 2            # Feature extraction at n x n sub-patch shifts, giving an n times finer feature grid
    refiner: str = ""                # Learned full-resolution refiner after the click threshold: "", "bundled" or a file path
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
    image: np.ndarray | None = None   # The normalized image, kept for edge refinement
    passes: int = 1                   # Grid cells per patch along each axis (shift_passes)


def vit_input_size(h: int, w: int, vit_size: int, patch: int) -> tuple[int, int]:
    """Backbone input size that keeps the aspect ratio, rounded to whole patches."""
    scale = vit_size / max(h, w)
    return max(patch, int(round(h * scale / patch)) * patch), max(patch, int(round(w * scale / patch)) * patch)


def _shifted(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """The image moved by (-dy, -dx) pixels, so that pixel (y + dy, x + dx) lands at (y, x)."""
    if dy == 0 and dx == 0:
        return img
    return ndi.shift(img, (-dy, -dx), order=1, mode="reflect")


def embed_grid(backbone: Backbone, img: np.ndarray, in_h: int, in_w: int, layer_from_end: int, passes: int = 1) -> torch.Tensor:
    """Feature grid of an image. With passes = n > 1, the image is embedded n x n times, shifted by fractions of a
    patch, and the grids are interleaved into one n times finer grid. Shift (a, b) samples patch centres at
    offsets ((a + 0.5) / n - 0.5) patches, so cell (n i + a, n j + b) of the fine grid sits exactly where a uniform
    grid of that size expects it, and the rest of the pipeline needs no change."""
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))  # noqa: E731
    if passes <= 1:
        return backbone.embed(t(img), in_h, in_w, layer_from_end)
    h, w = img.shape
    step_y, step_x = backbone.patch_size * h / in_h, backbone.patch_size * w / in_w   # One patch in image pixels
    fine = None
    for a in range(passes):
        for b in range(passes):
            off = lambda k: (k + 0.5) / passes - 0.5  # noqa: E731
            g = backbone.embed(t(_shifted(img, off(a) * step_y, off(b) * step_x)), in_h, in_w, layer_from_end)
            if fine is None:
                fine = g.new_zeros((g.shape[0] * passes, g.shape[1] * passes, g.shape[2]))
            fine[a::passes, b::passes] = g
    return fine


def embed_image(backbone: Backbone, img: np.ndarray, settings: SegmentationSettings) -> Embedding:
    """Embeds a normalized [0, 1] image."""
    h, w = img.shape
    in_h, in_w = vit_input_size(h, w, settings.vit_size, backbone.patch_size)
    try:
        grid = embed_grid(backbone, img, in_h, in_w, settings.layer_from_end, max(1, int(settings.shift_passes)))
    except (torch.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" not in str(e).lower() and not isinstance(e, torch.OutOfMemoryError):
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise ValueError("The GPU ran out of memory. Pick a smaller backbone, such as DINOv2 Small, or a lower "
                         "'Detail' setting under 'Clean-up and advanced'") from None
    return Embedding(grid=grid, height=h, width=w, image=img, passes=max(1, int(settings.shift_passes)))


_REFINERS: dict = {}
BUNDLED_REFINER = Path(__file__).resolve().parent / "models" / "refiner_liu.pt"


def get_refiner(spec: str):
    """Loads a refiner once: "bundled" is the one shipped with boneseg, anything else a file path."""
    from .refine import Refiner
    path = BUNDLED_REFINER if spec == "bundled" else Path(spec).expanduser()
    if str(path) not in _REFINERS:
        if not path.exists():
            raise ValueError(f"Refiner file not found: {path}")
        _REFINERS[str(path)] = Refiner.load(path)
    return _REFINERS[str(path)]


def box_mean(a: np.ndarray, r: int) -> np.ndarray:
    return ndi.uniform_filter(a, size=2 * r + 1, mode="reflect")


def guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """He et al.'s guided filter: smooths src where the guide is flat and keeps the guide's edges.
    Used to move the coarse, patch-level score boundary onto real intensity edges."""
    I, p = guide.astype(np.float32), src.astype(np.float32)
    mI, mp = box_mean(I, r), box_mean(p, r)
    a = (box_mean(I * p, r) - mI * mp) / (box_mean(I * I, r) - mI * mI + eps)
    b = mp - a * mI
    return box_mean(a, r) * I + box_mean(b, r)


def refine_scores(raw: np.ndarray, emb: Embedding, settings: SegmentationSettings) -> np.ndarray:
    """Applies the chosen edge refinement to a full-resolution score map."""
    if settings.edge_refine == "none" or emb.image is None:
        return raw
    if settings.edge_refine == "guided":
        # Radius of one feature cell, the scale at which the score map is blurred
        cell = max(emb.height / emb.grid.shape[0], emb.width / emb.grid.shape[1])
        return guided_filter(emb.image, raw, max(1, int(round(cell))), settings.guided_eps)
    raise ValueError(f"Unknown edge refinement {settings.edge_refine}")


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


def normalize_scores(score: torch.Tensor, mode: str) -> torch.Tensor:
    """Optionally standardizes patch scores by the image's own median and spread.
    Keeps a carried-over threshold meaningful when contrast changes between slices."""
    if mode == "none":
        return score
    if mode == "robust":
        med = score.median()
        mad = (score - med).abs().median() * 1.4826
        return (score - med) / (mad + 1e-6)
    raise ValueError(f"Unknown score normalization {mode}")


def upsample(score: torch.Tensor, shape: tuple[int, int]) -> np.ndarray:
    out = F.interpolate(score[None, None].float(), size=tuple(shape), mode="bilinear", align_corners=False)[0, 0]
    return out.cpu().numpy()


def rescale01(a: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(a)), float(np.max(a))
    return np.zeros_like(a, dtype=np.float32) if hi - lo < 1e-12 else ((a - lo) / (hi - lo)).astype(np.float32)


def threshold_heatmap(heat: np.ndarray, settings: SegmentationSettings) -> tuple[np.ndarray, float]:
    """Binary mask from a heatmap already rescaled to [0, 1], for the modes that need no clicks.
    Returns the mask and the threshold used. The "clicks" mode falls back to Otsu here."""
    mode = settings.threshold_mode
    if mode in ("otsu", "clicks"):
        thr = float(threshold_otsu(heat)) if np.ptp(heat) > 0 else 1.0
    elif mode == "top_percent":
        thr = float(np.percentile(heat, 100 - settings.top_percent))
    elif mode == "manual":
        thr = float(settings.manual_threshold)
    else:
        raise ValueError(f"Unknown threshold mode {mode}")
    return heat >= thr, thr


def auto_position(position: float, n_pos: int, n_neg: int) -> float:
    """The threshold position to use. The automatic choice (-1) matches what nested tuning picked in every fold:
    0.7 when the extreme clicks set the threshold (five or fewer of a kind), 0.9 when quantiles do."""
    if position >= 0:
        return float(position)
    return 0.9 if n_pos > 5 and n_neg > 5 else 0.7


def calibrate_threshold(pos_scores, neg_scores, position: float = 0.5) -> float | None:
    """Raw-score threshold between the object clicks and the background clicks, halfway by default.
    A position above 0.5 moves it towards the object clicks, which makes the mask stricter.

    Uses the weakest object click and the strongest background click, or the 10th and 90th percentiles
    once there are more than five clicks of a kind, so one stray click does not decide the threshold.
    On the demo stack this beat Otsu by a wide margin (Dice 0.78 against 0.31), since Otsu tends to split
    tissue from empty space instead of the target from everything else."""
    pos_scores, neg_scores = np.asarray(pos_scores, float), np.asarray(neg_scores, float)
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return None
    lo = np.quantile(pos_scores, 0.1) if len(pos_scores) > 5 else pos_scores.min()
    hi = np.quantile(neg_scores, 0.9) if len(neg_scores) > 5 else neg_scores.max()
    return float(hi + position * (lo - hi))


def sample_points(a: np.ndarray, points) -> np.ndarray:
    pts = np.asarray(points, float).reshape(-1, 2)
    ys = np.clip(np.round(pts[:, 0]).astype(int), 0, a.shape[0] - 1)
    xs = np.clip(np.round(pts[:, 1]).astype(int), 0, a.shape[1] - 1)
    return a[ys, xs]


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
    threshold: float        # On the rescaled heatmap
    raw_threshold: float = 0.0  # On the raw score, comparable across slices for the same prototypes
    raw_score_range: tuple[float, float] = (0.0, 0.0)
    threshold_source: str = ""  # Which rule set the threshold
    uncertainty: np.ndarray | None = None
    suggestion: tuple[int, int] | None = None
    extra: dict = field(default_factory=dict)


def segment_with_prototypes(emb: Embedding, pos: torch.Tensor, neg: torch.Tensor, settings: SegmentationSettings,
                            pixel_um=(1.0, 1.0), raw_threshold: float | None = None,
                            pos_points=None, neg_points=None) -> SegmentationResult:
    """Segments with the given prototypes.

    In "clicks" mode the threshold comes from, in order: clicks on this image, a raw threshold handed in
    (from the annotated slice of a stack or from a profile), or Otsu as the last resort."""
    score = normalize_scores(score_grid(emb.grid, pos, neg, settings.neg_weight), settings.score_norm)
    raw = refine_scores(upsample(score, (emb.height, emb.width)), emb, settings)
    lo, hi = float(raw.min()), float(raw.max())
    span = max(hi - lo, 1e-12)
    heat = rescale01(raw)
    source = settings.threshold_mode
    if settings.threshold_mode == "clicks":
        calibrated = None
        if pos_points is not None and neg_points is not None and len(pos_points) and len(neg_points):
            calibrated = calibrate_threshold(sample_points(raw, pos_points), sample_points(raw, neg_points),
                                             auto_position(settings.threshold_position, len(pos_points), len(neg_points)))
        if calibrated is not None:
            raw_threshold, source = calibrated, "clicks"
        elif raw_threshold is not None:
            source = "carried over"
        if raw_threshold is not None:
            thr = (raw_threshold - lo) / span
            mask = raw >= raw_threshold
        else:
            mask, thr = threshold_heatmap(heat, settings)
            raw_threshold, source = lo + thr * span, "otsu (no background clicks)"
    else:
        mask, thr = threshold_heatmap(heat, settings)
        raw_threshold = lo + thr * span
    if settings.refiner and emb.image is not None:
        mask = get_refiner(settings.refiner).predict(emb.image, raw, float(raw_threshold)) >= 0.5
        source += ", refined"
    mask = postprocess(mask, settings, pixel_um)
    return SegmentationResult(heat=heat, mask=mask, threshold=float(thr), raw_threshold=float(raw_threshold),
                              raw_score_range=(lo, hi), threshold_source=source)


def segment(emb: Embedding, pos_points, neg_points, settings: SegmentationSettings, pixel_um=(1.0, 1.0)) -> SegmentationResult:
    return segment_with_prototypes(emb, prototypes(emb, pos_points), prototypes(emb, neg_points), settings, pixel_um,
                                   pos_points=pos_points, neg_points=neg_points)


@torch.no_grad()
def uncertainty_map(emb: Embedding, pos: torch.Tensor, neg: torch.Tensor, settings: SegmentationSettings,
                    n_boot: int = 16, seed: int = 0, threshold: float | None = None) -> np.ndarray:
    """Where the mask depends on individual clicks.

    With few clicks, each one is left out in turn (a jackknife). With many clicks, random 80% subsets
    are drawn instead. Every draw is scored on the same scale and with the same threshold as the full
    result, so only the change in clicks matters. The output is the fraction of draws that disagree
    with the full mask, from 0 (stable) to 1. Works at patch resolution, so it is cheap."""
    full = score_grid(emb.grid, pos, neg, settings.neg_weight)
    lo, hi = float(full.min()), float(full.max())
    norm = lambda s: ((s - lo) / max(hi - lo, 1e-12)).clamp(0, 1)
    if threshold is None:
        _, threshold = threshold_heatmap(rescale01(upsample(full, (emb.height, emb.width))), settings)
    thr = threshold
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


def suggest_click(uncertainty: np.ndarray, existing_points, min_dist_frac: float = 0.05, work_side: int = 400) -> tuple[int, int] | None:
    """The most uncertain location that is not close to an existing click.
    Works on a copy at most work_side pixels wide, since only a rough location is needed."""
    if uncertainty.max() <= 1e-6:
        return None
    h, w = uncertainty.shape
    f = max(1, int(np.ceil(max(h, w) / work_side)))
    small = uncertainty[: h - h % f or h, : w - w % f or w]
    small = small.reshape(small.shape[0] // f, f, small.shape[1] // f, f).mean(axis=(1, 3)) if f > 1 else small.copy()
    sh, sw = small.shape
    u = ndi.gaussian_filter(small, sigma=max(1.0, min(sh, sw) / 100))
    if len(existing_points):
        yy, xx = np.ogrid[:sh, :sw]
        r = min_dist_frac * max(sh, sw)
        for y, x in existing_points:
            u[(yy - y / f) ** 2 + (xx - x / f) ** 2 < r ** 2] = 0
    y, x = np.unravel_index(int(np.argmax(u)), u.shape)
    if u[y, x] <= 1e-6:
        return None
    return (int(min(h - 1, y * f + f // 2)), int(min(w - 1, x * f + f // 2)))


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
    raw_threshold: float | None = None  # Calibrated on the source image, reused when no clicks are given
    head: dict | None = None            # A learned model: {"kind", "dim", "vit_size", "state": {name: array}}
    structures: list | None = None      # Several structures: [{"name", "color", "pos", "neg", "threshold"}]

    @property
    def kind(self) -> str:
        return "learned" if self.head else "structures" if self.structures else "prototypes"

    def multi_model(self) -> "MultiModel":
        return MultiModel(names=[st["name"] for st in self.structures],
                          pos=[torch.from_numpy(np.asarray(st["pos"], np.float32)) for st in self.structures],
                          neg=[torch.from_numpy(np.asarray(st["neg"], np.float32)) for st in self.structures],
                          thresholds=[float(st["threshold"]) for st in self.structures])

    @classmethod
    def from_multi(cls, model: "MultiModel", colors: list[str], **kw) -> "Profile":
        structures = [{"name": n, "color": c, "pos": p.cpu().numpy(), "neg": q.cpu().numpy(), "threshold": t}
                      for n, c, p, q, t in zip(model.names, colors, model.pos, model.neg, model.thresholds)]
        d = model.pos[0].shape[1]
        return cls(pos=np.zeros((0, d), np.float32), neg=np.zeros((0, d), np.float32), structures=structures, **kw)

    def save(self, path) -> None:
        arrays = {"pos": self.pos, "neg": self.neg}
        meta = {"name": self.name, "backbone": self.backbone, "layer_from_end": self.layer_from_end,
                "settings": self.settings, "description": self.description, "source": self.source,
                "raw_threshold": self.raw_threshold}
        if self.head:
            meta["head"] = {k: v for k, v in self.head.items() if k != "state"}
            meta["head"]["state_keys"] = list(self.head["state"])
            for k, v in self.head["state"].items():
                arrays[f"head__{k}"] = np.asarray(v)
        if self.structures:
            meta["structures"] = [{"name": st["name"], "color": st.get("color", ""), "threshold": float(st["threshold"])} for st in self.structures]
            for k, st in enumerate(self.structures):
                arrays[f"s{k}__pos"], arrays[f"s{k}__neg"] = np.asarray(st["pos"]), np.asarray(st["neg"])
        np.savez(path, meta=np.array([repr(meta)]), **arrays)

    @classmethod
    def load(cls, path) -> "Profile":
        import ast

        d = np.load(path, allow_pickle=False)
        meta = ast.literal_eval(str(d["meta"][0]))
        head = meta.pop("head", None)
        if head:
            head["state"] = {k: d[f"head__{k}"] for k in head.pop("state_keys")}
        structures = meta.pop("structures", None)
        if structures:
            for k, st in enumerate(structures):
                st["pos"], st["neg"] = d[f"s{k}__pos"], d[f"s{k}__neg"]
        return cls(pos=d["pos"], neg=d["neg"], head=head, structures=structures, **meta)

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.pos.astype(np.float32)), torch.from_numpy(self.neg.astype(np.float32))


@dataclass
class MultiResult:
    labels: np.ndarray          # 0 for background, k + 1 for structure k
    thresholds: list[float]     # Raw-score threshold per structure
    names: list[str]


@dataclass
class MultiModel:
    """Prototypes and raw-score thresholds for several structures, ready to apply to any slice."""
    names: list[str]
    pos: list[torch.Tensor]
    neg: list[torch.Tensor]
    thresholds: list[float]


def calibrate_multi(emb: Embedding, classes: list[dict], neg_points, settings: SegmentationSettings) -> MultiModel:
    """Builds per-structure prototypes from the clicks on one slice and calibrates each threshold there.

    Negatives for each structure are the background clicks plus every other structure's clicks, so each
    structure learns what separates it from the others, not only from the background."""
    classes = [c for c in classes if len(c.get("pos", []))]
    if not classes:
        raise ValueError("Add clicks for at least one structure")
    shape = (emb.height, emb.width)
    model = MultiModel(names=[], pos=[], neg=[], thresholds=[])
    for k, c in enumerate(classes):
        others = [p for j, o in enumerate(classes) if j != k for p in o["pos"]]
        neg_k = list(neg_points) + others
        pos_t, neg_t = prototypes(emb, c["pos"]), prototypes(emb, neg_k)
        raw = upsample(normalize_scores(score_grid(emb.grid, pos_t, neg_t, settings.neg_weight), settings.score_norm), shape)
        thr = calibrate_threshold(sample_points(raw, c["pos"]), sample_points(raw, neg_k)) if len(neg_k) else None
        if thr is None:
            thr = float(threshold_otsu(raw)) if float(raw.max()) > float(raw.min()) else float(raw.max())
        model.names.append(c.get("name", f"Structure {k + 1}"))
        model.pos.append(pos_t)
        model.neg.append(neg_t)
        model.thresholds.append(float(thr))
    return model


def apply_multi(emb: Embedding, model: MultiModel, settings: SegmentationSettings, pixel_um=(1.0, 1.0)) -> np.ndarray:
    """Label map for one slice: a pixel goes to the structure it resembles most among those whose
    threshold it clears. Scores are standardized per slice, so thresholds carry over through a stack."""
    shape = (emb.height, emb.width)
    passes, sims = [], []
    for pos_t, neg_t, thr in zip(model.pos, model.neg, model.thresholds):
        raw = upsample(normalize_scores(score_grid(emb.grid, pos_t, neg_t, settings.neg_weight), settings.score_norm), shape)
        passes.append(raw >= thr)
        # Plain similarity to the structure's own clicks decides between structures that both pass
        sims.append(upsample(score_grid(emb.grid, pos_t, pos_t[:0], 0.0), shape))
    sim = np.where(np.stack(passes), np.stack(sims), -np.inf)
    best = np.argmax(sim, axis=0)
    labels = np.where(np.isfinite(np.max(sim, axis=0)), best + 1, 0).astype(np.uint8)
    if settings.min_object_um2 or settings.fill_holes_um2 or settings.smooth_px:
        out = np.zeros_like(labels)
        for k in range(len(model.names)):
            m = postprocess(labels == k + 1, settings, pixel_um)
            out[m & (out == 0)] = k + 1
        labels = out
    return labels


def segment_multi(emb: Embedding, classes: list[dict], neg_points, settings: SegmentationSettings, pixel_um=(1.0, 1.0)) -> MultiResult:
    """Several structures at once on one slice. classes: [{"name", "pos": [(y, x), ...]}]."""
    model = calibrate_multi(emb, classes, neg_points, settings)
    return MultiResult(labels=apply_multi(emb, model, settings, pixel_um), thresholds=model.thresholds, names=model.names)
