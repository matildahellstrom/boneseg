"""Segmenting whole z-stacks from one annotated slice or a saved profile."""
from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import tifffile
import torch

from . import metrics, quantify
from .head import head_input, segment_with_head
from .segment import Embedding, SegmentationSettings, segment_with_prototypes


# 3D object labelling holds one int32 per voxel; above this many voxels the stack is labelled at a reduced in-plane
# resolution (every f-th pixel), so a full-resolution run of a large stack does not run out of memory
VOXEL_BUDGET = int(float(os.environ.get("BONESEG_3D_VOXELS", 2e8)))


def downsample_factor(n_z: int, h: int, w: int, budget: int | None = None) -> int:
    """In-plane step that keeps n_z x (h / f) x (w / f) within the voxel budget (1 when it already fits)."""
    budget = VOXEL_BUDGET if budget is None else budget   # Read at call time, so the setting can be changed at runtime
    voxels = n_z * h * w
    return 1 if voxels <= budget else int(np.ceil(np.sqrt(voxels / budget)))


def write_stack(path: Path, pages, shape: tuple, dtype, pixel_um, spacing_um: float, factor: int = 1):
    """Writes a (Z, Y, X) TIFF stack from an iterator of slices, so the whole stack is never in memory at once."""
    tifffile.imwrite(path, data=pages, shape=shape, dtype=dtype, imagej=True, compression="zlib",
                     resolution=(1 / (pixel_um[1] * factor), 1 / (pixel_um[0] * factor)),
                     metadata={"axes": "ZYX", "spacing": float(spacing_um), "unit": "um"})


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
              lock=None, head=None, roi: np.ndarray | None = None, read_ahead: Callable[[int], object] | None = None,
              n_z: int | None = None, segment_fn: Callable[[int], object] | None = None) -> dict:
    """Segments every slice in req.z_list. Writes a mask stack, per-slice stats and a summary to out_dir. A learned
    model with z context also gets the features of its neighbouring slices (n_z is the stack's depth). segment_fn(z),
    if given, replaces boneseg's own segmentation (micro-SAM)."""
    out_dir = Path(out_dir)
    pixel_um = (voxel_um[1], voxel_um[2])
    order = processing_order(req.z_list, req.ref_z)
    masks: dict[int, np.ndarray] = {}   # Bit-packed along x (np.packbits): 8 pixels per byte while the run goes
    shape = None
    rows = []
    ahead = _ReadAhead(read_ahead)
    n_z = n_z or max(req.z_list, default=0) + 1
    embed_at = _recent(get_embedding, 2 * getattr(head, "z_context", 0) + 3)
    for i, z in enumerate(order):
        if cancelled():
            break
        progress(i / max(1, len(order)), f"Slice {z} ({i + 1}/{len(order)})")
        ahead.next(order[i + 1] if i + 1 < len(order) else None)
        with (lock or contextlib.nullcontext()):
            # Scores are standardized per slice (score_norm), so the threshold calibrated on the
            # annotated slice stays meaningful when contrast fades with depth
            if segment_fn is not None:
                res = segment_fn(z)
            elif head is not None:
                res = segment_with_head(head, head_input(head, embed_at, z, n_z, voxel_um[0]), settings, pixel_um)
            else:
                res = segment_with_prototypes(embed_at(z), pos, neg, settings, pixel_um, raw_threshold=raw_threshold)
            m = res.mask & roi if roi is not None else res.mask
        shape = m.shape
        masks[z] = np.packbits(m, axis=-1)
        img = get_image(z)
        row = {"z": z, "z_um": z * voxel_um[0], "threshold": res.threshold, **quantify.summarize_mask(m, img, pixel_um, roi)}
        ref = get_reference(z)
        if ref is not None:
            # HD95 on masks downsampled to at most 2048 px: exact on smaller images, and it halves the
            # time per slice on 3000 px images, where the distance transforms dominate
            row.update(metrics.compare(m, ref, pixel_um, roi, hd95_max_side=2048))
        rows.append(row)
    ahead.close()
    df = pd.DataFrame(rows).sort_values("z").reset_index(drop=True) if rows else pd.DataFrame()
    zs = sorted(masks)
    unpack = lambda z: np.unpackbits(masks[z], axis=-1, count=shape[1]).astype(bool)  # noqa: E731
    if zs:
        write_stack(out_dir / "masks.tif", (unpack(z).astype(np.uint8) * 255 for z in zs), (len(zs), *shape), np.uint8,
                    pixel_um, voxel_um[0] * _step(zs))
    df.to_csv(out_dir / "slices.csv", index=False)
    summary = quantify.summarize_stack(df, voxel_um, _step(zs)) if len(df) else {"n_slices": 0}
    if zs:
        # 3D objects: a cell that spans several slices is counted once. Large stacks are labelled at a reduced in-plane
        # resolution (factor f), which keeps memory bounded; volumes are scaled accordingly
        f = downsample_factor(len(zs), *shape)
        mstack = np.stack([unpack(z)[::f, ::f] for z in zs])
        labels = quantify.label_3d(mstack)
        obj = quantify.objects_3d(mstack, (voxel_um[0] * _step(zs), voxel_um[1] * f, voxel_um[2] * f), zs, labels=labels)
        obj.to_csv(out_dir / "objects_3d.csv", index=False)
        # Object IDs match the label column of objects_3d.csv, for Fiji, Imaris or napari
        lab_dtype = np.uint16 if labels.max() < 65535 else np.uint32
        write_stack(out_dir / "labels_3d.tif", (lab.astype(lab_dtype) for lab in labels), labels.shape, lab_dtype, pixel_um,
                    voxel_um[0] * _step(zs), factor=f)
        if f > 1:
            summary["objects_3d_downsampled"] = f   # 3D objects and labels_3d.tif at 1/f of the in-plane resolution
        summary["n_objects_3d"] = int(len(obj))
        summary["n_objects_3d_inside"] = int((~obj["touches_stack_edge"]).sum()) if len(obj) else 0
        summary["median_object_volume_um3"] = float(obj["volume_um3"].median()) if len(obj) else 0.0
        summary["median_nearest_neighbour_um"] = float(obj["nearest_neighbour_um"].median()) if len(obj) > 1 else None
    if len(df) and "dice" in df:
        summary["mean_dice_vs_reference"] = float(df["dice"].mean())
    summary["z_processed"] = zs
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return {"summary": summary, "slices": json.loads(df.to_json(orient="records")) if len(df) else []}


def _recent(fn: Callable[[int], Embedding], size: int) -> Callable[[int], Embedding]:
    """fn with its last few results kept, so a z-context model does not recompute its neighbours' features."""
    from collections import OrderedDict

    cache: OrderedDict = OrderedDict()

    def get(z):
        if z in cache:
            cache.move_to_end(z)
        else:
            cache[z] = fn(z)
            if len(cache) > size:
                cache.popitem(last=False)
        return cache[z]
    return get


class _ReadAhead:
    """Loads the next slice in a background thread while the current one is computed, so file reading
    (about 0.25 s per plane for Imaris) overlaps with the model instead of adding to it."""

    def __init__(self, load: Callable[[int], object] | None):
        from concurrent.futures import ThreadPoolExecutor

        self.load = load
        self.pool = ThreadPoolExecutor(max_workers=1) if load else None

    def next(self, z):
        if self.pool is not None and z is not None:
            self.pool.submit(self.load, z)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)


def _step(zs: list[int]) -> int:
    return int(np.median(np.diff(zs))) if len(zs) > 1 else 1


def run_stack_multi(z_list: list[int], ref_z: int | None, model, settings: SegmentationSettings,
                    get_embedding: Callable[[int], Embedding], get_image: Callable[[int], np.ndarray], voxel_um, out_dir: Path,
                    progress: Callable[[float, str], None] = lambda p, m: None, cancelled: Callable[[], bool] = lambda: False,
                    lock=None, roi: np.ndarray | None = None, labeler: Callable | None = None,
                    read_ahead: Callable[[int], object] | None = None) -> dict:
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
    ahead = _ReadAhead(read_ahead)
    for i, z in enumerate(order):
        if cancelled():
            break
        progress(i / max(1, len(order)), f"Slice {z} ({i + 1}/{len(order)})")
        ahead.next(order[i + 1] if i + 1 < len(order) else None)
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
    ahead.close()
    df = pd.DataFrame(rows).sort_values(["structure", "z"]).reset_index(drop=True) if rows else pd.DataFrame()
    zs = sorted(labels)
    summary: dict = {"n_slices": len(zs), "z_processed": zs, "structures": {}}
    if zs:
        step = _step(zs)
        shape = labels[zs[0]].shape
        write_stack(out_dir / "labels.tif", (labels[z] for z in zs), (len(zs), *shape), np.uint8, pixel_um, voxel_um[0] * step)
        write_stack(out_dir / "masks.tif", ((labels[z] > 0).astype(np.uint8) * 255 for z in zs), (len(zs), *shape), np.uint8,
                    pixel_um, voxel_um[0] * step)
        f = downsample_factor(len(zs), *shape)   # 3D objects of large stacks at a reduced in-plane resolution
        lstack = np.stack([labels[z][::f, ::f] for z in zs])
        if f > 1:
            summary["objects_3d_downsampled"] = f
        objs = []
        for k, name in enumerate(model.names):
            sub = df[df["structure"] == name]
            st = quantify.summarize_stack(sub, voxel_um, step)
            obj = quantify.objects_3d(lstack == k + 1, (voxel_um[0] * step, voxel_um[1] * f, voxel_um[2] * f), zs)
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
