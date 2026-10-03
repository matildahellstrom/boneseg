"""Segmenting whole z-stacks from one annotated slice or a saved profile."""
from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import tifffile
import torch

from . import metrics, quantify
from .head import segment_with_head
from .segment import SegmentationSettings, segment_with_prototypes


@dataclass
class StackRequest:
    z_list: list[int]
    ref_z: int | None = None      # Slice the prototypes come from, processing starts there


def processing_order(z_list: list[int], ref_z: int | None) -> list[int]:
    """Starts at the reference slice and walks outward, so the slices nearest the clicks finish first."""
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
              raw_threshold: float | None = None, *,
              get_embedding: Callable[[int], Embedding], get_image: Callable[[int], np.ndarray],
              get_reference: Callable[[int], np.ndarray | None], voxel_um, out_dir: Path,
              progress: Callable[[float, str], None] = lambda p, m: None, cancelled: Callable[[], bool] = lambda: False,
              lock=None, head=None) -> dict:
    """Segments every slice in req.z_list. Writes a mask stack, per-slice stats and a summary to out_dir."""
    out_dir = Path(out_dir)
    pixel_um = (voxel_um[1], voxel_um[2])
    order = processing_order(req.z_list, req.ref_z)
    masks: dict[int, np.ndarray] = {}
    rows = []
    # Adaptive prototypes are tracked separately for each direction away from the reference slice
    for i, z in enumerate(order):
        if cancelled():
            break
        progress(i / max(1, len(order)), f"Slice {z} ({i + 1}/{len(order)})")
        with (lock or contextlib.nullcontext()):
            emb = get_embedding(z)
            # Scores are standardized per slice (score_norm), so the threshold calibrated on the
            # annotated slice stays meaningful when contrast fades with depth
            if head is not None:
                res = segment_with_head(head, emb, settings, pixel_um)
            else:
                res = segment_with_prototypes(emb, pos, neg, settings, pixel_um, raw_threshold=raw_threshold)
            masks[z] = res.mask
        img = get_image(z)
        row = {"z": z, "z_um": z * voxel_um[0], "threshold": res.threshold, **quantify.summarize_mask(res.mask, img, pixel_um)}
        ref = get_reference(z)
        if ref is not None:
            row.update(metrics.compare(res.mask, ref, pixel_um))
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("z").reset_index(drop=True) if rows else pd.DataFrame()
    zs = sorted(masks)
    if zs:
        stack = np.stack([masks[z] for z in zs]).astype(np.uint8) * 255
        tifffile.imwrite(out_dir / "masks.tif", stack, imagej=True, compression="zlib",
                         resolution=(1 / pixel_um[1], 1 / pixel_um[0]),
                         metadata={"axes": "ZYX", "spacing": float(voxel_um[0] * _step(zs)), "unit": "um"})
    df.to_csv(out_dir / "slices.csv", index=False)
    summary = quantify.summarize_stack(df, voxel_um, _step(zs)) if len(df) else {"n_slices": 0}
    if zs:
        # 3D objects: a cell that spans several slices is counted once
        obj = quantify.objects_3d(np.stack([masks[z] for z in zs]), (voxel_um[0] * _step(zs), voxel_um[1], voxel_um[2]), zs)
        obj.to_csv(out_dir / "objects_3d.csv", index=False)
        summary["n_objects_3d"] = int(len(obj))
        summary["n_objects_3d_inside"] = int((~obj["touches_stack_edge"]).sum()) if len(obj) else 0
        summary["median_object_volume_um3"] = float(obj["volume_um3"].median()) if len(obj) else 0.0
    if len(df) and "dice" in df:
        summary["mean_dice_vs_reference"] = float(df["dice"].mean())
    summary["z_processed"] = zs
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return {"summary": summary, "slices": json.loads(df.to_json(orient="records")) if len(df) else []}


def _step(zs: list[int]) -> int:
    return int(np.median(np.diff(zs))) if len(zs) > 1 else 1
