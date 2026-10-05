"""A small learned refiner that sharpens click masks at full image resolution.

The DINO score map is computed on 14-pixel patches, so its boundary is coarse and tends to spill over thin
structures. The refiner is a light U-Net that sees the image and the score map relative to the calibrated
threshold, and predicts the mask at pixel resolution. Its output starts out equal to the thresholded score
(the score channel is added straight to its logits), so it only learns corrections. It is trained once on
labelled slices with simulated clicks, and then runs after the user's clicks without any further training.
This is the light version of the notebook's U-Net on the raw image plus the DINO heatmap, the best of its variants.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCORE_SCALE = 2.0   # Score units (robust standard deviations) per logit unit of the input channel


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.SiLU())


class RefinerNet(nn.Module):
    """Three-level U-Net, about 120k parameters."""

    def __init__(self, width: int = 16):
        super().__init__()
        w = width
        self.e1, self.e2, self.e3 = _block(2, w), _block(w, 2 * w), _block(2 * w, 4 * w)
        self.d2, self.d1 = _block(6 * w, 2 * w), _block(3 * w, w)
        self.out = nn.Conv2d(w, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.gain = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        d2 = self.d2(torch.cat([F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], 1))
        # Starts as the thresholded score: logit = gain * relative score, plus learned corrections
        return self.gain * x[:, 1:2] * 4.0 + self.out(d1)


def refiner_input(img: np.ndarray, raw: np.ndarray, threshold: float) -> np.ndarray:
    """[2, H, W]: the image, and the score relative to the threshold squashed to [-1, 1]."""
    rel = np.tanh((raw - threshold) / (SCORE_SCALE * 2))
    return np.stack([img.astype(np.float32), rel.astype(np.float32)])


@dataclass
class Refiner:
    state: dict
    width: int = 16
    info: dict = field(default_factory=dict)   # How it was trained, for reports
    _net: RefinerNet | None = field(default=None, repr=False)

    def net(self, device) -> RefinerNet:
        if self._net is None:
            self._net = RefinerNet(self.width)
            self._net.load_state_dict({k: torch.as_tensor(v) for k, v in self.state.items()})
            self._net.eval()
        return self._net.to(device)

    @torch.no_grad()
    def predict(self, img: np.ndarray, raw: np.ndarray, threshold: float, device=None) -> np.ndarray:
        """Probability map at full resolution."""
        device = device or _device()
        x = torch.from_numpy(refiner_input(img, raw, threshold))[None].to(device)
        h, w = x.shape[-2:]
        ph, pw = (-h) % 4, (-w) % 4
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        return torch.sigmoid(self.net(device)(x))[0, 0, :h, :w].float().cpu().numpy()

    def save(self, path) -> None:
        torch.save({"state": {k: v.cpu() for k, v in self.state.items()}, "width": self.width, "info": self.info}, path)

    @classmethod
    def load(cls, path) -> "Refiner":
        d = torch.load(path, map_location="cpu", weights_only=True)
        return cls(state=d["state"], width=d["width"], info=d.get("info", {}))


def click_example(emb, pos, neg, settings):
    """(raw score map, calibrated raw threshold) for clicks, exactly as segmentation computes them before refining."""
    from dataclasses import replace
    from .segment import segment
    r = segment(emb, pos, neg, replace(settings, refiner=""))
    lo, hi = r.raw_score_range
    return lo + r.heat.astype(np.float64) * (hi - lo), float(r.raw_threshold)


def _device():
    from .backbone import pick_device
    return pick_device()


def _crops(items, size, rng):
    """Random crops, flips and 90-degree rotations, with mild intensity jitter on the image channel."""
    xs, ys = [], []
    for x, y in items:
        _, h, w = x.shape
        s = min(size, h, w)
        r0, c0 = rng.integers(0, h - s + 1), rng.integers(0, w - s + 1)
        xc, yc = x[:, r0:r0 + s, c0:c0 + s].copy(), y[r0:r0 + s, c0:c0 + s].copy()
        k = rng.integers(4)
        xc, yc = np.rot90(xc, k, axes=(1, 2)), np.rot90(yc, k)
        if rng.random() < 0.5:
            xc, yc = xc[:, :, ::-1], yc[:, ::-1]
        xc = xc.astype(np.float32)
        xc[0] = np.clip(xc[0] * rng.uniform(0.8, 1.2) + rng.uniform(-0.1, 0.1), 0, 1)
        xs.append(xc)
        ys.append(yc.copy())
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)[:, None].astype(np.float32))


def crop_pool(img, raw, thr, gt, k: int, size: int, rng) -> list:
    """k crops of one example, half centred on the expert boundary where the refiner has work to do, stored
    compactly so that many click draws of large slices fit in memory."""
    x = refiner_input(img, raw, thr)
    gt = gt.astype(bool)
    h, w = gt.shape
    s = min(size, h, w)
    edge = np.argwhere(gt ^ np.roll(gt, 1, 0) | gt ^ np.roll(gt, 1, 1))
    out = []
    for i in range(k):
        if i % 2 == 0 and len(edge):
            cy, cx = edge[rng.integers(len(edge))]
            r0, c0 = int(np.clip(cy - s // 2, 0, h - s)), int(np.clip(cx - s // 2, 0, w - s))
        else:
            r0, c0 = int(rng.integers(0, h - s + 1)), int(rng.integers(0, w - s + 1))
        out.append((x[:, r0:r0 + s, c0:c0 + s].astype(np.float16), gt[r0:r0 + s, c0:c0 + s].copy()))
    return out


def train_refiner(examples, steps: int = 1500, batch: int = 8, crop: int = 256, lr: float = 2e-3, width: int = 16,
                  seed: int = 0, crops_per_example: int = 8, device=None, log=None) -> Refiner:
    """Trains on (image, raw score map, threshold, expert mask) examples, normally several click draws per
    labelled slice; examples may be a generator, since each is reduced to a few crops at once.
    Loss: binary cross-entropy plus soft Dice."""
    device = device or _device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    items, n_examples = [], 0
    for img, raw, thr, gt in examples:
        items += crop_pool(img, raw, thr, gt, crops_per_example, crop, rng)
        n_examples += 1
    if not items:
        raise ValueError("No training examples for the refiner")
    net = RefinerNet(width).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(4, int(steps))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=min(0.5, max(0.1, 2.5 / steps)))
    t0 = time.time()
    net.train()
    for step in range(steps):
        pick = [items[i] for i in rng.integers(0, len(items), batch)]
        x, y = _crops(pick, crop, rng)
        x, y = x.to(device), y.to(device)
        logits = net(x)
        p = torch.sigmoid(logits)
        dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
        loss = F.binary_cross_entropy_with_logits(logits, y) + dice
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if log and (step + 1) % 250 == 0:
            log(f"refiner step {step + 1}/{steps}, loss {loss.item():.3f}, {time.time() - t0:.0f}s")
    net.eval()
    return Refiner(state={k: v.detach().cpu() for k, v in net.state_dict().items()}, width=width,
                   info={"examples": n_examples, "crops": len(items), "steps": steps, "crop": crop})


def simulate_clicks(gt: np.ndarray, n_pos: int, n_neg: int, rng, noisy: bool = False):
    """Object clicks inside a mask and background clicks outside it, at least 10 px from the boundary
    (noisy: anywhere, with 10% on the wrong side), as (y, x) lists."""
    import scipy.ndimage as ndi
    inner = gt if noisy else ndi.binary_erosion(gt, iterations=10)
    outer = ~gt if noisy else ~ndi.binary_dilation(gt, iterations=10)
    inner, outer = (inner if inner.any() else gt), (outer if outer.any() else ~gt)

    def pick(region, k):
        ys, xs = np.nonzero(region)
        i = rng.choice(len(ys), k, replace=len(ys) < k)
        return [(int(ys[j]), int(xs[j])) for j in i]

    pos, neg = pick(inner, n_pos), pick(outer, n_neg)
    if noisy:
        wp, wn = (max(1, round(0.1 * n_pos)) if n_pos >= 5 else 0), (max(1, round(0.1 * n_neg)) if n_neg >= 5 else 0)
        pos, neg = pos[wp:] + pick(~gt, wp), neg[wn:] + pick(gt, wn)
    return pos, neg


def train_refiner_from_files(specs: list[str], settings, n_slices: int = 10, draws: int = 6, steps: int = 1500,
                             seed: int = 0, log=print) -> Refiner:
    """Trains a refiner from microscopy files that contain an expert mask channel.

    Each spec is 'file:image_channel:mask_channel' (the mask channel defaults to the last one). n_slices slices
    with a non-empty mask are taken evenly from each file, and each gets several simulated click draws
    (3+6, 10+10 and 25+25 clicks, alternately clean and noisy), segmented with the given settings."""
    from . import io as bio
    from .backbone import get_backbone
    from .segment import embed_image

    rng = np.random.default_rng(seed)
    bb = get_backbone(settings.backbone)
    budgets = [(3, 6), (10, 10), (25, 25)]

    def examples():
        for spec in specs:
            parts = spec.rsplit(":", 2)
            path, ch = parts[0], int(parts[1])
            vol = bio.load_volume(path)
            mask_ch = int(parts[2]) if len(parts) == 3 else vol.n_channels - 1
            zs = [z for z in np.linspace(0, vol.n_z - 1, n_slices * 3).round().astype(int)]
            picked = [z for z in dict.fromkeys(zs) if (vol.get_plane(mask_ch, int(z)) > 0).mean() > 0.005]
            picked = [picked[i] for i in sorted(set(np.linspace(0, len(picked) - 1, min(n_slices, len(picked))).round().astype(int)))] if picked else []
            log(f"{Path(path).name}: {len(picked)} slices with a mask")
            for z in picked:
                img = bio.normalize_plane(vol.get_plane(ch, int(z)), settings.clip_low, settings.clip_high)
                gt = vol.get_plane(mask_ch, int(z)) > 0
                emb = embed_image(bb, img, settings)
                for d in range(draws):
                    pos, neg = simulate_clicks(gt, *budgets[d % len(budgets)], rng, noisy=d % 2 == 1)
                    raw, thr = click_example(emb, pos, neg, settings)
                    yield img, raw, thr, gt

    ref = train_refiner(examples(), steps=steps, seed=seed, log=log)
    ref.info.update({"files": [Path(s.rsplit(":", 2)[0]).name for s in specs], "settings": {k: v for k, v in settings.to_dict().items() if k != "refiner"}})
    return ref
