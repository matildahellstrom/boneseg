"""Feature backbones that turn an image into a grid of patch embeddings.

The DINOv2 backbones are the ones evaluated in the notebook. The "classic"
backbone uses hand-made intensity and texture features. It needs no download
and no GPU, which makes it useful for tests, offline use and quick previews.
"""
from __future__ import annotations

import gc
import os
import threading

import numpy as np
import torch
import torch.nn.functional as F

# DINOv2 code pinned to a fixed commit so torch.hub always loads the same model definition
DINOV2_REPO = "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINO_HUB_NAMES = {
    "dinov2_s14": "dinov2_vits14",
    "dinov2_b14": "dinov2_vitb14",
    "dinov2_l14": "dinov2_vitl14",
    "dinov2_g14": "dinov2_vitg14",
}
BACKBONE_LABELS = {
    "classic": "Classic features (fast, no download)",
    "dinov2_s14": "DINOv2 Small",
    "dinov2_b14": "DINOv2 Base",
    "dinov2_l14": "DINOv2 Large",
    "dinov2_g14": "DINOv2 Giant",
}

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def pick_device() -> torch.device:
    forced = os.environ.get("BONESEG_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Backbone:
    """Interface: embed a [0, 1] image resized to (in_h, in_w) into an [H, W, D] grid."""

    name: str = "base"
    patch_size: int = 14

    def embed(self, img: torch.Tensor, in_h: int, in_w: int, layer_from_end: int = 1) -> torch.Tensor:
        raise NotImplementedError


class DinoBackbone(Backbone):
    def __init__(self, name: str, device: torch.device):
        self.name = name
        self.device = device
        model = torch.hub.load(DINOV2_REPO, DINO_HUB_NAMES[name], trust_repo=True, skip_validation=True)
        self.model = model.to(device).eval()
        self.patch_size = int(self.model.patch_size)

    @torch.no_grad()
    def embed(self, img, in_h, in_w, layer_from_end=1):
        x = img[None, None].to(self.device).repeat(1, 3, 1, 1)
        x = F.interpolate(x, (in_h, in_w), mode="bilinear", align_corners=False)
        x = (x - _IMAGENET_MEAN.to(self.device)) / _IMAGENET_STD.to(self.device)
        # n=k returns the last k blocks, so index 0 is the k-th block from the end
        tokens = self.model.get_intermediate_layers(x, n=layer_from_end, reshape=False, return_class_token=False, norm=True)[0]
        grid = tokens[0].reshape(in_h // self.patch_size, in_w // self.patch_size, -1)
        return F.normalize(grid.float(), dim=-1)


class ClassicBackbone(Backbone):
    """Per-patch statistics of intensity, gradients and local contrast at a few scales,
    lifted with a fixed random projection so that cosine similarity behaves sensibly."""

    name = "classic"
    patch_size = 14

    def __init__(self, device: torch.device | None = None, dim: int = 64):
        self.device = torch.device("cpu") if device is None else device
        g = torch.Generator().manual_seed(0)
        self.n_raw = 14
        self.proj = torch.randn(self.n_raw, dim, generator=g) / np.sqrt(self.n_raw)

    @torch.no_grad()
    def embed(self, img, in_h, in_w, layer_from_end=1):
        p = self.patch_size
        x = F.interpolate(img[None, None].float(), (in_h, in_w), mode="bilinear", align_corners=False)
        gy = x[..., 1:, :] - x[..., :-1, :]
        gx = x[..., :, 1:] - x[..., :, :-1]
        gy = F.pad(gy, (0, 0, 0, 1))
        gx = F.pad(gx, (0, 1, 0, 0))
        grad = torch.sqrt(gx ** 2 + gy ** 2)
        feats = []
        for src in (x, grad):
            feats += [F.avg_pool2d(src, p), F.avg_pool2d(src ** 2, p).sub(F.avg_pool2d(src, p) ** 2).clamp_min(0).sqrt(),
                      F.max_pool2d(src, p), -F.max_pool2d(-src, p)]
        # Context at two larger scales, sampled back on the patch grid
        for k in (3, 7):
            ctx = F.avg_pool2d(F.avg_pool2d(x, p), k, stride=1, padding=k // 2, count_include_pad=False)
            feats.append(ctx)
            feats.append(F.avg_pool2d(x, p) - ctx)
        feats.append(torch.ones_like(feats[0]) * 0.5)
        feats.append(F.avg_pool2d((x > x.mean()).float(), p))
        f = torch.cat(feats, 1)[0]  # [n_raw, H, W]
        f = (f - f.mean(dim=(1, 2), keepdim=True)) / (f.std(dim=(1, 2), keepdim=True) + 1e-6)
        grid = f.permute(1, 2, 0) @ self.proj
        return F.normalize(grid, dim=-1)


_cache: dict[str, Backbone] = {}
_lock = threading.Lock()


# Folders model files (fine-tuned backbones, refiners) may be loaded from; None allows any path. The app restricts
# this to its own models folder when it is reachable over the network without path access.
MODEL_DIRS: list | None = None


def check_model_path(path: str) -> str:
    from pathlib import Path
    p = Path(os.path.expanduser(path)).resolve()
    if MODEL_DIRS is not None and not any(p.is_relative_to(Path(d).resolve()) for d in MODEL_DIRS):
        raise ValueError("Model files can only be loaded from the app's models folder")
    return str(p)


def backbone_label(name: str) -> str:
    base, _, path = name.partition("@")
    label = BACKBONE_LABELS.get(base, base)
    return f"{label}, fine-tuned ({os.path.basename(path)})" if path else label


def get_backbone(name: str) -> Backbone:
    """Loads a backbone, keeping at most one DINOv2 model in memory at a time.
    'dinov2_s14@/path/file.pt' is DINOv2 Small with the fine-tuned weights saved by boneseg.finetune."""
    base, _, ft_path = name.partition("@")
    if base not in BACKBONE_LABELS or (ft_path and base == "classic"):
        raise ValueError(f"Unknown backbone {name}. Choose one of {', '.join(BACKBONE_LABELS)}")
    if ft_path:
        ft_path = check_model_path(ft_path)
        if not os.path.exists(ft_path):
            raise ValueError(f"Fine-tuned model not found: {ft_path}")
    with _lock:
        if name not in _cache:
            if name == "classic":
                _cache[name] = ClassicBackbone()
            else:
                for key in [k for k in _cache if k != "classic"]:
                    del _cache[key]
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                try:
                    bb = DinoBackbone(base, pick_device())
                    if ft_path:
                        from .finetune import FineTuned
                        ft = FineTuned.load(os.path.expanduser(ft_path))
                        if ft.base != base:
                            raise ValueError(f"{ft_path} was fine-tuned from {ft.base}, not {base}")
                        bb.model.load_state_dict({k: v.to(bb.device) for k, v in ft.backbone_state.items()}, strict=False)
                        bb.name = name
                    _cache[name] = bb
                except Exception as e:  # Usually no internet on first use, or a blocked download
                    raise ValueError(
                        f"Could not load {backbone_label(name)}. The first use downloads its code and weights from GitHub "
                        f"and Meta, so check the internet connection, or pick 'Classic features', which needs no download "
                        f"({type(e).__name__}: {str(e)[:160]})") from None
        return _cache[name]


def dino_weights_cached(name: str) -> bool:
    """Whether the DINOv2 weights are already downloaded, so loading will not hit the network."""
    name = name.partition("@")[0]
    if name == "classic":
        return True
    ckpt = os.path.join(torch.hub.get_dir(), "checkpoints", f"{DINO_HUB_NAMES[name]}_pretrain.pth")
    return os.path.exists(ckpt)
