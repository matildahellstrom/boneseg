"""Overlap and boundary metrics, as used in the notebook."""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice score. Two empty masks count as a perfect match."""
    tp = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return 1.0 if denom == 0 else float(2 * tp / denom)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else float(inter / union)


def hd95(pred: np.ndarray, gt: np.ndarray, spacing=(1.0, 1.0), max_side: int | None = None) -> float:
    """95th percentile Hausdorff distance between mask boundaries, in the units of spacing.
    NaN when one of the masks is empty. With max_side, large masks are first downsampled by an integer
    factor (spacing scaled to match), which keeps interactive use fast at a small loss of precision."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() or not gt.any():
        return float("nan")
    if max_side and max(pred.shape) > max_side:
        f = int(np.ceil(max(pred.shape) / max_side))
        pred, gt = pred[::f, ::f], gt[::f, ::f]
        spacing = (spacing[0] * f, spacing[1] * f)
        if not pred.any() or not gt.any():
            return float("nan")
    d_to_gt = ndi.distance_transform_edt(~gt, sampling=spacing)
    d_to_pred = ndi.distance_transform_edt(~pred, sampling=spacing)
    bp = pred & ~ndi.binary_erosion(pred)
    bg = gt & ~ndi.binary_erosion(gt)
    return float(np.percentile(np.concatenate([d_to_gt[bp], d_to_pred[bg]]), 95))


def nsd(pred: np.ndarray, gt: np.ndarray, tol: float, spacing=(1.0, 1.0), max_side: int | None = None) -> float:
    """Normalized surface distance (surface Dice): the share of both boundaries that lies within tol (in the units
    of spacing) of the other boundary, as in Gu et al. (2025) and Nikolov et al. (2021). Two empty masks score 1,
    one empty mask 0. With max_side, large masks are downsampled by an integer factor first, as in hd95."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() and not gt.any():
        return 1.0
    if not pred.any() or not gt.any():
        return 0.0
    if max_side and max(pred.shape) > max_side:
        f = int(np.ceil(max(pred.shape) / max_side))
        pred, gt = pred[::f, ::f], gt[::f, ::f]
        spacing = (spacing[0] * f, spacing[1] * f)
    bp = pred & ~ndi.binary_erosion(pred)
    bg = gt & ~ndi.binary_erosion(gt)
    if not bp.any() or not bg.any():
        return 0.0
    d_to_bg = ndi.distance_transform_edt(~bg, sampling=spacing)
    d_to_bp = ndi.distance_transform_edt(~bp, sampling=spacing)
    close = (d_to_bg[bp] <= tol).sum() + (d_to_bp[bg] <= tol).sum()
    return float(close / (bp.sum() + bg.sum()))


def compare(pred: np.ndarray, gt: np.ndarray, spacing=(1.0, 1.0), roi: np.ndarray | None = None, hd95_max_side: int | None = None) -> dict:
    if roi is not None:
        pred, gt = pred & roi, gt & roi
    return {"dice": dice(pred, gt), "iou": iou(pred, gt), "hd95_um": hd95(pred, gt, spacing, hd95_max_side)}
