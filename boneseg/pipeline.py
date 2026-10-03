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
from .segment import Embedding, SegmentationSettings, segment_with_prototypes


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
              lock=None, head=None, roi: np.ndarray | None = None) -> dict:
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
            masks[z] = res.mask & roi if roi is not None else res.mask
        img = get_image(z)
        row = {"z": z, "z_um": z * voxel_um[0], "threshold": res.threshold, **quantify.summarize_mask(masks[z], img, pixel_um, roi)}
        ref = get_reference(z)
        if ref is not None:
            # HD95 on masks downsampled to at most 2048 px: exact on smaller images, and it halves the
            # time per slice on 3000 px images, where the distance transforms dominate
            row.update(metrics.compare(masks[z], ref, pixel_um, roi, hd95_max_side=2048))
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
        mstack = np.stack([masks[z] for z in zs])
        labels = quantify.label_3d(mstack)
        obj = quantify.objects_3d(mstack, (voxel_um[0] * _step(zs), voxel_um[1], voxel_um[2]), zs, labels=labels)
        obj.to_csv(out_dir / "objects_3d.csv", index=False)
        # Object IDs match the label column of objects_3d.csv, for Fiji, Imaris or napari
        tifffile.imwrite(out_dir / "labels_3d.tif", labels.astype(np.uint16 if labels.max() < 65535 else np.uint32), imagej=True,
                         compression="zlib", resolution=(1 / pixel_um[1], 1 / pixel_um[0]),
                         metadata={"axes": "ZYX", "spacing": float(voxel_um[0] * _step(zs)), "unit": "um"})
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


def run_stack_multi(z_list: list[int], ref_z: int | None, model, settings: SegmentationSettings,
                    get_embedding: Callable[[int], Embedding], get_image: Callable[[int], np.ndarray], voxel_um, out_dir: Path,
                    progress: Callable[[float, str], None] = lambda p, m: None, cancelled: Callable[[], bool] = lambda: False,
                    lock=None, roi: np.ndarray | None = None, labeler: Callable | None = None) -> dict:
    """Several structures through a stack. model needs .names; labeler(embedding) -> label map replaces
    the prototype rule, for a learned model of several structures. Writes a label stack (0 background, k for structure k),
    a union mask stack for the side view, per-slice measurements per structure and 3D objects per structure."""
    from .segment import apply_multi

    import re

    from . import histo

    out_dir = Path(out_dir)
    pixel_um = (voxel_um[1], voxel_um[2])
    order = processing_order(z_list, ref_z)
    labels: dict[int, np.ndarray] = {}
    rows = []
    histo_rows = []
    # With a structure named like bone, histomorphometry runs on every slice against the first other structure
    bone_k = next((k for k, n in enumerate(model.names) if re.search(r"bone|matrix", n, re.I)), None)
    cell_k = next((k for k in range(len(model.names)) if k != bone_k), None) if bone_k is not None else None
    for i, z in enumerate(order):
        if cancelled():
            break
        progress(i / max(1, len(order)), f"Slice {z} ({i + 1}/{len(order)})")
        with (lock or contextlib.nullcontext()):
            emb = get_embedding(z)
            lab = labeler(emb) if labeler is not None else apply_multi(emb, model, settings, pixel_um)
        if roi is not None:
            lab = np.where(roi, lab, 0).astype(np.uint8)
        labels[z] = lab
        img = get_image(z)
        for k, name in enumerate(model.names):
            rows.append({"z": z, "z_um": z * voxel_um[0], "structure": name, **quantify.summarize_mask(lab == k + 1, img, pixel_um, roi)})
        if bone_k is not None and cell_k is not None:
            hm, _ = histo.histomorphometry(lab == bone_k + 1, lab == cell_k + 1, pixel_um, roi=roi)
            histo_rows.append({"z": z, **hm})
    df = pd.DataFrame(rows).sort_values(["structure", "z"]).reset_index(drop=True) if rows else pd.DataFrame()
    zs = sorted(labels)
    summary: dict = {"n_slices": len(zs), "z_processed": zs, "structures": {}}
    if zs:
        step = _step(zs)
        lstack = np.stack([labels[z] for z in zs])
        meta = {"axes": "ZYX", "spacing": float(voxel_um[0] * step), "unit": "um"}
        res = (1 / pixel_um[1], 1 / pixel_um[0])
        tifffile.imwrite(out_dir / "labels.tif", lstack, imagej=True, compression="zlib", resolution=res, metadata=meta)
        tifffile.imwrite(out_dir / "masks.tif", (lstack > 0).astype(np.uint8) * 255, imagej=True, compression="zlib", resolution=res, metadata=meta)
        objs = []
        for k, name in enumerate(model.names):
            sub = df[df["structure"] == name]
            st = quantify.summarize_stack(sub, voxel_um, step)
            obj = quantify.objects_3d(lstack == k + 1, (voxel_um[0] * step, voxel_um[1], voxel_um[2]), zs)
            obj.insert(0, "structure", name)
            objs.append(obj)
            st.update(n_objects_3d=int(len(obj)), median_object_volume_um3=float(obj["volume_um3"].median()) if len(obj) else 0.0)
            summary["structures"][name] = st
        pd.concat(objs, ignore_index=True).to_csv(out_dir / "objects_3d.csv", index=False)
        if histo_rows:
            hdf = pd.DataFrame(histo_rows).sort_values("z")
            hdf.to_csv(out_dir / "histomorphometry.csv", index=False)
            # Stack-level values: perimeters and counts summed over slices, so thin slices do not dominate
            b_pm = hdf["B.Pm_mm"].sum()
            summary["histomorphometry"] = {
                "bone": model.names[bone_k], "cells": model.names[cell_k],
                "B.Ar/T.Ar_%": float(100 * hdf["B.Ar_mm2"].sum() / hdf["T.Ar_mm2"].sum()) if hdf["T.Ar_mm2"].sum() else None,
                "Oc.Pm/B.Pm_%": float(100 * hdf["Oc.Pm_mm"].sum() / b_pm) if b_pm else None,
                "N.Oc/B.Pm_per_mm": float(hdf["N.Oc"].sum() / b_pm) if b_pm else None,
                "B.Pm_mm_total": float(b_pm),
            }
    df.to_csv(out_dir / "slices.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return {"summary": summary, "slices": json.loads(df.to_json(orient="records")) if len(df) else []}
