"""Segmenting whole z-stacks from one annotated slice or a saved profile."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import tifffile
import torch

from . import metrics, quantify
from .segment import Embedding, SegmentationSettings, segment_with_prototypes


@dataclass
class StackRequest:
    z_list: list[int]
    ref_z: int | None = None      # Slice the prototypes come from, processing starts there
    adaptive: bool = False        # Refresh prototypes from confident regions of each finished slice
    adaptive_k: int = 24          # Confident patches added per class and slice
    adaptive_keep: int = 96       # Maximum number of adaptive prototypes kept per class


def _confident_prototypes(emb: Embedding, heat: np.ndarray, mask: np.ndarray, k: int):
    """Patch embeddings from the most confident foreground and background of a finished slice."""
    hg, wg, d = emb.grid.shape
    small = torch.nn.functional.adaptive_avg_pool2d(torch.from_numpy(heat)[None, None], (hg, wg))[0, 0].numpy()
    msmall = torch.nn.functional.adaptive_avg_pool2d(torch.from_numpy(mask.astype(np.float32))[None, None], (hg, wg))[0, 0].numpy()
    flat = emb.grid.reshape(-1, d)
    fg = np.flatnonzero((msmall.ravel() > 0.95))
    bg = np.flatnonzero((msmall.ravel() < 0.05))
    fg = fg[np.argsort(-small.ravel()[fg])][:k]
    bg = bg[np.argsort(small.ravel()[bg])][:k]
    to_t = lambda idx: flat[torch.as_tensor(idx, dtype=torch.long, device=flat.device)].cpu() if len(idx) else flat.new_zeros((0, d)).cpu()
    return to_t(fg), to_t(bg)


def processing_order(z_list: list[int], ref_z: int | None) -> list[int]:
    """Starts at the reference slice and walks outward, so adaptive prototypes drift smoothly."""
    zs = sorted(set(z_list))
    if ref_z is None or not zs:
        return zs
    start = min(range(len(zs)), key=lambda i: abs(zs[i] - ref_z))
    order = [zs[start]]
    lo, hi = start - 1, start + 1
    while lo >= 0 or hi < len(zs):
        if hi < len(zs):
            order.append(zs[hi]); hi += 1
        if lo >= 0:
            order.append(zs[lo]); lo -= 1
    return order


def run_stack(req: StackRequest, pos: torch.Tensor, neg: torch.Tensor, settings: SegmentationSettings,
              get_embedding: Callable[[int], Embedding], get_image: Callable[[int], np.ndarray],
              get_reference: Callable[[int], np.ndarray | None], voxel_um, out_dir: Path,
              progress: Callable[[float, str], None] = lambda p, m: None, cancelled: Callable[[], bool] = lambda: False) -> dict:
    """Segments every slice in req.z_list. Writes a mask stack, per-slice stats and a summary to out_dir."""
    out_dir = Path(out_dir)
    pixel_um = (voxel_um[1], voxel_um[2])
    order = processing_order(req.z_list, req.ref_z)
    masks: dict[int, np.ndarray] = {}
    rows = []
    # Adaptive prototypes are tracked separately for each direction away from the reference slice
    empty = pos.new_zeros((0, pos.shape[1])).cpu()
    adaptive = {"up": (empty, empty), "down": (empty, empty)}
    for i, z in enumerate(order):
        if cancelled():
            break
        progress(i / max(1, len(order)), f"Slice {z} ({i + 1}/{len(order)})")
        emb = get_embedding(z)
        direction = "up" if req.ref_z is None or z >= req.ref_z else "down"
        ap, an = adaptive[direction]
        p = torch.cat([pos.cpu(), ap]) if req.adaptive and len(ap) else pos
        n = torch.cat([neg.cpu(), an]) if req.adaptive and len(an) else neg
        res = segment_with_prototypes(emb, p, n, settings, pixel_um)
        masks[z] = res.mask
        img = get_image(z)
        row = {"z": z, "z_um": z * voxel_um[0], "threshold": res.threshold, **quantify.summarize_mask(res.mask, img, pixel_um)}
        ref = get_reference(z)
        if ref is not None:
            row.update(metrics.compare(res.mask, ref, pixel_um))
        rows.append(row)
        if req.adaptive:
            fp, fn = _confident_prototypes(emb, res.heat, res.mask, req.adaptive_k)
            adaptive[direction] = (torch.cat([ap, fp])[-req.adaptive_keep:], torch.cat([an, fn])[-req.adaptive_keep:])
    df = pd.DataFrame(rows).sort_values("z").reset_index(drop=True) if rows else pd.DataFrame()
    zs = sorted(masks)
    if zs:
        stack = np.stack([masks[z] for z in zs]).astype(np.uint8) * 255
        tifffile.imwrite(out_dir / "masks.tif", stack, imagej=True, compression="zlib",
                         resolution=(1 / pixel_um[1], 1 / pixel_um[0]),
                         metadata={"axes": "ZYX", "spacing": float(voxel_um[0] * _step(zs)), "unit": "um"})
    df.to_csv(out_dir / "slices.csv", index=False)
    summary = quantify.summarize_stack(df, voxel_um, _step(zs)) if len(df) else {"n_slices": 0}
    if len(df) and "dice" in df:
        summary["mean_dice_vs_reference"] = float(df["dice"].mean())
    summary["z_processed"] = zs
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return {"summary": summary, "slices": json.loads(df.to_json(orient="records")) if len(df) else []}


def _step(zs: list[int]) -> int:
    return int(np.median(np.diff(zs))) if len(zs) > 1 else 1
