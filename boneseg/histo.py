"""2D bone histomorphometry from a bone mask and an osteoclast (or other cell) mask.

Names follow the ASBMR nomenclature for 2D sections (Dempster et al., J Bone Miner Res 2013):
T.Ar tissue area, B.Ar bone area, B.Pm bone perimeter, Oc.Pm osteoclast perimeter (the part of
the bone perimeter covered by osteoclasts), N.Oc number of osteoclasts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from skimage import measure


def bone_boundary(bone: np.ndarray) -> np.ndarray:
    """Bone pixels that touch non-bone, excluding the image border, which is not a real surface."""
    b = bone.astype(bool)
    edge = b & ~ndi.binary_erosion(b, structure=np.ones((3, 3)), border_value=1)
    return edge


def histomorphometry(bone: np.ndarray, cells: np.ndarray, pixel_um=(1.0, 1.0), contact_um: float = 3.0) -> tuple[dict, pd.DataFrame]:
    """Standard 2D measurements and a per-cell table.

    A bone surface pixel counts as covered when an osteoclast pixel lies within contact_um of it,
    and an osteoclast counts as on bone when any of its pixels lies within contact_um of bone."""
    bone, cells = bone.astype(bool), cells.astype(bool)
    py, px = pixel_um
    iso = float(np.sqrt(py * px))
    t_ar = bone.size * py * px
    b_ar = float(bone.sum() * py * px)
    boundary = bone_boundary(bone)
    # Perimeter length from scikit-image, which handles diagonals, in micrometres
    # The cut through the image border is not a bone surface, so its length is subtracted
    border_px = int(bone[0, :].sum() + bone[-1, :].sum() + bone[1:-1, 0].sum() + bone[1:-1, -1].sum())
    b_pm = float(max(0.0, measure.perimeter(bone) - border_px) * iso) if bone.any() else 0.0
    dist_to_cells = ndi.distance_transform_edt(~cells, sampling=pixel_um) if cells.any() else np.full(bone.shape, np.inf)
    covered = boundary & (dist_to_cells <= contact_um)
    covered_share = float(covered.sum() / boundary.sum()) if boundary.any() else 0.0

    dist_to_bone = ndi.distance_transform_edt(~bone, sampling=pixel_um) if bone.any() else np.full(bone.shape, np.inf)
    labels = measure.label(cells, connectivity=2)
    rows = []
    for r in measure.regionprops(labels):
        coords = r.coords
        d = float(dist_to_bone[coords[:, 0], coords[:, 1]].min())
        cy, cx = r.centroid
        rows.append({"label": r.label, "area_um2": r.area * py * px, "centroid_y_um": cy * py, "centroid_x_um": cx * px,
                     "distance_to_bone_um": d, "on_bone": d <= contact_um,
                     "overlap_with_bone": float(bone[coords[:, 0], coords[:, 1]].mean())})
    table = pd.DataFrame(rows, columns=["label", "area_um2", "centroid_y_um", "centroid_x_um", "distance_to_bone_um", "on_bone", "overlap_with_bone"])
    n_on = int(table["on_bone"].sum()) if len(table) else 0
    b_pm_mm = b_pm / 1000
    summary = {
        "T.Ar_mm2": t_ar / 1e6,
        "B.Ar_mm2": b_ar / 1e6,
        "B.Ar/T.Ar_%": 100 * b_ar / t_ar if t_ar else 0.0,
        "B.Pm_mm": b_pm_mm,
        "Oc.Pm/B.Pm_%": 100 * covered_share,
        "Oc.Pm_mm": covered_share * b_pm_mm,
        "N.Oc": n_on,
        "N.Oc/B.Pm_per_mm": n_on / b_pm_mm if b_pm_mm > 0 else 0.0,
        "N.Oc/T.Ar_per_mm2": n_on / (t_ar / 1e6) if t_ar else 0.0,
        "cells_total": int(len(table)),
        "cells_on_bone_%": 100 * n_on / len(table) if len(table) else 0.0,
        "median_distance_to_bone_um": float(table["distance_to_bone_um"].median()) if len(table) else float("nan"),
        "contact_um": contact_um,
    }
    return summary, table


def overlay(bone: np.ndarray, cells: np.ndarray, pixel_um=(1.0, 1.0), contact_um: float = 3.0) -> np.ndarray:
    """RGBA overlay: bone surface in white, covered surface in magenta, cells in cyan."""
    boundary = bone_boundary(bone)
    covered = boundary & (ndi.distance_transform_edt(~cells, sampling=pixel_um) <= contact_um) if cells.any() else np.zeros_like(boundary)
    cell_edge = cells & ~ndi.binary_erosion(cells)
    rgba = np.zeros(bone.shape + (4,), np.uint8)
    rgba[bone.astype(bool)] = (255, 255, 255, 28)
    rgba[boundary] = (235, 235, 235, 230)
    rgba[cells.astype(bool)] = (0, 220, 255, 60)
    rgba[cell_edge] = (0, 220, 255, 255)
    thick = ndi.binary_dilation(covered, iterations=1)
    rgba[thick] = (255, 40, 200, 255)
    return rgba
