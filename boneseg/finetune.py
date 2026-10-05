"""Fine-tuning DINOv2 on expert-labelled slices.

The frozen backbone was pretrained on natural photographs. Fine-tuning adapts its last transformer blocks to
the microscopy images: the blocks are trained, together with a 1 x 1 output layer on the patch tokens, to predict
the expert mask. With only a handful of labelled slices, everything else stays frozen, the learning rate of the
backbone is small, and training stops at the step with the best Dice on separate validation slices.

The result is saved as a small file holding only the changed weights. It is used as a backbone named
'dinov2_s14@/path/to/file.pt', so clicks, learned models and the refiner all run on the adapted features, and its
output layer can segment on its own. train_blocks=0 trains the output layer alone on frozen features, which is the
fair baseline: the same data, code and budget, with DINO left unchanged.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import _IMAGENET_MEAN, _IMAGENET_STD, DINO_HUB_NAMES, DINOV2_REPO, pick_device
from .segment import vit_input_size


class SegNet(nn.Module):
    """DINOv2 with its last train_blocks blocks trainable and a 1 x 1 convolution on the final patch tokens."""

    def __init__(self, dino: nn.Module, train_blocks: int):
        super().__init__()
        self.dino = dino
        self.patch = int(dino.patch_size)
        for p in self.dino.parameters():
            p.requires_grad = False
        self.train_blocks = train_blocks
        if train_blocks > 0:
            for blk in self.dino.blocks[-train_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True
            for p in self.dino.norm.parameters():
                p.requires_grad = True
        self.head = nn.Conv2d(dino.embed_dim, 1, 1)

    def tokens(self, x):
        """[B, D, h, w] final-layer patch tokens, layer-normed as the app uses them."""
        return self.dino.get_intermediate_layers(x, n=1, reshape=True, norm=True)[0]

    def forward(self, x):
        t = self.tokens(x)
        return self.head(F.normalize(t, dim=1) * 10.0)   # Normalized like the app's features, scaled for a sane logit range


def to_input(img: np.ndarray, device) -> torch.Tensor:
    """A [0, 1] grey image (already at backbone resolution) as a normalized 3-channel batch of one."""
    x = torch.from_numpy(np.ascontiguousarray(img, np.float32))[None, None].repeat(1, 3, 1, 1)
    return ((x - _IMAGENET_MEAN) / _IMAGENET_STD).to(device)


@dataclass
class Prepared:
    """A slice resized to the backbone input size, with its mask at the same size and at full size."""
    img: np.ndarray       # [in_h, in_w]
    gt_small: np.ndarray  # [in_h, in_w] float in [0, 1]
    gt: np.ndarray        # Full-resolution boolean mask
    shape: tuple


def prepare(img: np.ndarray, gt: np.ndarray, vit_size: int, patch: int = 14) -> Prepared:
    h, w = img.shape
    in_h, in_w = vit_input_size(h, w, vit_size, patch)
    t = torch.from_numpy(np.ascontiguousarray(img, np.float32))[None, None]
    small = F.interpolate(t, (in_h, in_w), mode="bilinear", align_corners=False)[0, 0].numpy()
    g = torch.from_numpy(gt.astype(np.float32))[None, None]
    gs = F.interpolate(g, (in_h, in_w), mode="area")[0, 0].numpy()
    return Prepared(small, gs, gt.astype(bool), (h, w))


def _batch(items: list[Prepared], size: int, n: int, rng):
    xs, ys = [], []
    for _ in range(n):
        it = items[rng.integers(len(items))]
        h, w = it.img.shape
        s_h, s_w = min(size, h // 14 * 14), min(size, w // 14 * 14)
        # Half the crops are centred on bone, which is a small share of most slices
        if rng.random() < 0.5 and it.gt_small.max() > 0.5:
            ys_, xs_ = np.nonzero(it.gt_small > 0.5)
            k = rng.integers(len(ys_))
            r0 = int(np.clip(ys_[k] - s_h // 2, 0, h - s_h))
            c0 = int(np.clip(xs_[k] - s_w // 2, 0, w - s_w))
        else:
            r0, c0 = int(rng.integers(0, h - s_h + 1)), int(rng.integers(0, w - s_w + 1))
        x = it.img[r0:r0 + s_h, c0:c0 + s_w]
        y = it.gt_small[r0:r0 + s_h, c0:c0 + s_w]
        k = int(rng.integers(4))
        x, y = np.rot90(x, k), np.rot90(y, k)
        if rng.random() < 0.5:
            x, y = x[:, ::-1], y[:, ::-1]
        x = np.clip(np.clip(x, 0, 1) ** rng.uniform(0.7, 1.4) * rng.uniform(0.8, 1.2), 0, 1)   # Gamma and gain
        xs.append(np.ascontiguousarray(x))
        ys.append(np.ascontiguousarray(y))
    # Crops of one batch share a size only if every slice was large enough; pad the rest
    H, W = max(a.shape[0] for a in xs), max(a.shape[1] for a in xs)
    X = np.zeros((n, H, W), np.float32)
    Y = np.zeros((n, H, W), np.float32)
    M = np.zeros((n, H, W), np.float32)
    for i, (a, b) in enumerate(zip(xs, ys)):
        X[i, :a.shape[0], :a.shape[1]], Y[i, :b.shape[0], :b.shape[1]], M[i, :a.shape[0], :a.shape[1]] = a, b, 1
    return X, Y, M


@torch.no_grad()
def predict(net: SegNet, p: Prepared, device) -> np.ndarray:
    """Probability map at full resolution."""
    net.eval()
    logits = net(to_input(p.img, device))
    up = F.interpolate(logits, size=p.shape, mode="bilinear", align_corners=False)
    return torch.sigmoid(up)[0, 0].float().cpu().numpy()


def _dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2 * (a & b).sum() / s)


@dataclass
class FineTuned:
    base: str
    train_blocks: int
    vit_size: int
    backbone_state: dict          # Only the trainable blocks and the final norm
    head_state: dict
    info: dict = field(default_factory=dict)

    def save(self, path):
        torch.save({"base": self.base, "train_blocks": self.train_blocks, "vit_size": self.vit_size,
                    "backbone_state": self.backbone_state, "head_state": self.head_state, "info": self.info}, path)

    @classmethod
    def load(cls, path) -> "FineTuned":
        d = torch.load(path, map_location="cpu", weights_only=True)
        return cls(d["base"], int(d["train_blocks"]), int(d["vit_size"]), d["backbone_state"], d["head_state"], d.get("info", {}))


def load_dino(base: str = "dinov2_s14") -> nn.Module:
    return torch.hub.load(DINOV2_REPO, DINO_HUB_NAMES[base], trust_repo=True, skip_validation=True)


def build(ft: FineTuned, dino: nn.Module | None = None, device=None) -> SegNet:
    """The fine-tuned network, from a fresh or given pretrained backbone."""
    device = device or pick_device()
    net = SegNet(dino if dino is not None else load_dino(ft.base), ft.train_blocks)
    net.dino.load_state_dict(ft.backbone_state, strict=False)
    net.head.load_state_dict(ft.head_state)
    return net.to(device).eval()


def finetune(train: list[tuple[np.ndarray, np.ndarray]], val: list[tuple[np.ndarray, np.ndarray]], base: str = "dinov2_s14",
             train_blocks: int = 4, vit_size: int = 980, steps: int = 1000, batch: int = 4, crop: int = 448,
             lr_backbone: float = 2e-5, lr_head: float = 1e-3, eval_every: int = 100, seed: int = 0,
             pretrained: nn.Module | None = None, device=None, log=None) -> FineTuned:
    """Fine-tunes on (image, mask) pairs of [0, 1] images and boolean masks; keeps the step with the best mean
    Dice on the validation pairs. Pass a pretrained backbone to reuse one download across runs (it is copied)."""
    device = device or pick_device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dino = copy.deepcopy(pretrained) if pretrained is not None else load_dino(base)
    net = SegNet(dino, train_blocks).to(device)
    patch = net.patch
    tr = [prepare(i, g, vit_size, patch) for i, g in train]
    va = [prepare(i, g, vit_size, patch) for i, g in val]
    groups = [{"params": net.head.parameters(), "lr": lr_head}]
    bb_params = [p for p in net.dino.parameters() if p.requires_grad]
    if bb_params:
        groups.append({"params": bb_params, "lr": lr_backbone})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=steps, pct_start=0.1)
    best, best_dice, history = None, -1.0, []
    t0 = time.time()

    def snapshot():
        return ({k: v.detach().cpu().clone() for k, v in net.dino.state_dict().items()
                 if train_blocks > 0 and (k.startswith("norm.") or any(k.startswith(f"blocks.{i}.") for i in range(len(net.dino.blocks) - train_blocks, len(net.dino.blocks))))},
                {k: v.detach().cpu().clone() for k, v in net.head.state_dict().items()})

    for step in range(1, steps + 1):
        net.train()
        net.dino.eval() if train_blocks == 0 else None   # Frozen backbone: no dropout or other train-time behaviour
        X, Y, M = _batch(tr, crop, batch, rng)
        x = torch.from_numpy(X)[:, None].repeat(1, 3, 1, 1)
        x = ((x - _IMAGENET_MEAN) / _IMAGENET_STD).to(device)
        y, m = torch.from_numpy(Y)[:, None].to(device), torch.from_numpy(M)[:, None].to(device)
        logits = F.interpolate(net(x), size=x.shape[-2:], mode="bilinear", align_corners=False)
        p = torch.sigmoid(logits) * m
        bce = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * m).sum() / m.sum()
        dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + (y * m).sum() + 1)
        loss = bce + dice
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % eval_every == 0 or step == steps:
            vd = float(np.mean([_dice(predict(net, v, device) >= 0.5, v.gt) for v in va])) if va else -float(loss.item())
            history.append({"step": step, "loss": float(loss.item()), "val_dice": vd})
            if vd > best_dice:
                best_dice, best = vd, snapshot()
            if log:
                log(f"  step {step}/{steps}: loss {loss.item():.3f}, validation Dice {vd:.3f}, {time.time() - t0:.0f}s")
    bstate, hstate = best
    return FineTuned(base, train_blocks, vit_size, bstate, hstate,
                     info={"n_train": len(train), "n_val": len(val), "steps": steps, "best_val_dice": best_dice,
                           "best_step": max(history, key=lambda r: r["val_dice"])["step"], "history": history,
                           "lr_backbone": lr_backbone, "lr_head": lr_head, "crop": crop, "batch": batch})
