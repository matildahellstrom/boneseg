"""Measurements on segmentation masks, in physical units."""
from __future__ import annotations

import numpy as np
import pandas as pd
from skimage import measure


def object_table(mask: np.ndarray, image: np.ndarray | None, pixel_um=(1.0, 1.0)) -> pd.DataFrame:
    """One row per connected object with its size, shape and intensity."""
    labels = measure.label(mask, connectivity=2)
    props = ["label", "area", "centroid", "perimeter", "eccentricity", "solidity", "major_axis_length", "minor_axis_length"]
    if image is not None:
        props.append("intensity_mean")
    if labels.max() == 0:
        return pd.DataFrame(columns=["label", "area_um2", "centroid_y_um", "centroid_x_um", "perimeter_um",
                                     "eccentricity", "solidity", "major_axis_um", "minor_axis_um", "equivalent_diameter_um"])
    t = pd.DataFrame(measure.regionprops_table(labels, intensity_image=image, properties=props))
    py, px = pixel_um
    iso = float(np.sqrt(py * px))  # Lengths use the geometric mean pixel size
    out = pd.DataFrame({
        "label": t["label"],
        "area_um2": t["area"] * py * px,
        "centroid_y_um": t["centroid-0"] * py,
        "centroid_x_um": t["centroid-1"] * px,
        "perimeter_um": t["perimeter"] * iso,
        "eccentricity": t["eccentricity"],
        "solidity": t["solidity"],
        "major_axis_um": t["major_axis_length"] * iso,
        "minor_axis_um": t["minor_axis_length"] * iso,
    })
    out["equivalent_diameter_um"] = np.sqrt(4 * out["area_um2"] / np.pi)
    if image is not None:
        out["mean_intensity"] = t["intensity_mean"]
    return out.sort_values("area_um2", ascending=False).reset_index(drop=True)


def summarize_mask(mask: np.ndarray, image: np.ndarray | None = None, pixel_um=(1.0, 1.0)) -> dict:
    """Headline numbers for one plane."""
    mask = mask.astype(bool)
    py, px = pixel_um
    total_um2 = mask.size * py * px
    area_um2 = float(mask.sum() * py * px)
    objects = object_table(mask, image, pixel_um)
    out = {
        "area_um2": area_um2,
        "area_fraction": float(mask.mean()),
        "image_area_um2": float(total_um2),
        "n_objects": int(len(objects)),
        "objects_per_mm2": float(len(objects) / (total_um2 / 1e6)) if total_um2 > 0 else 0.0,
        "mean_object_area_um2": float(objects["area_um2"].mean()) if len(objects) else 0.0,
        "median_object_area_um2": float(objects["area_um2"].median()) if len(objects) else 0.0,
        "largest_object_area_um2": float(objects["area_um2"].max()) if len(objects) else 0.0,
    }
    if image is not None:
        img = np.asarray(image, np.float32)
        inside = img[mask]
        outside = img[~mask]
        out["mean_intensity_inside"] = float(inside.mean()) if inside.size else float("nan")
        out["mean_intensity_outside"] = float(outside.mean()) if outside.size else float("nan")
        # Signal-to-background ratio, a quick sanity check of the segmentation
        out["contrast_ratio"] = float(out["mean_intensity_inside"] / max(out["mean_intensity_outside"], 1e-6)) if inside.size and outside.size else float("nan")
    return out


def summarize_stack(per_slice: pd.DataFrame, voxel_um=(1.0, 1.0, 1.0), slice_step: int = 1) -> dict:
    """Volume estimate from per-slice areas. Each processed slice stands for slice_step slices."""
    if per_slice.empty:
        return {"volume_um3": 0.0, "n_slices": 0}
    dz = voxel_um[0] * slice_step
    return {
        "n_slices": int(len(per_slice)),
        "volume_um3": float(per_slice["area_um2"].sum() * dz),
        "mean_area_fraction": float(per_slice["area_fraction"].mean()),
        "total_objects_counted": int(per_slice["n_objects"].sum()),
        "slice_spacing_um": float(dz),
    }


def objects_3d(stack: np.ndarray, voxel_um=(1.0, 1.0, 1.0), z_values=None, min_voxels: int = 1) -> pd.DataFrame:
    """Connected objects in a (Z, Y, X) mask stack, so a cell spanning several slices counts once.

    voxel_um should already include any slice step (z spacing times step)."""
    labels = measure.label(stack.astype(bool), connectivity=1)
    cols = ["label", "volume_um3", "n_slices", "z_first", "z_last", "centroid_z_um", "centroid_y_um", "centroid_x_um",
            "max_area_um2", "touches_stack_edge"]
    if labels.max() == 0:
        return pd.DataFrame(columns=cols)
    vz, vy, vx = voxel_um
    z_values = np.arange(stack.shape[0]) if z_values is None else np.asarray(z_values)
    rows = []
    for r in measure.regionprops(labels):
        if r.area < min_voxels:
            continue
        z0, _, _, z1, _, _ = r.bbox
        areas = [(labels[z] == r.label).sum() * vy * vx for z in range(z0, z1)]
        cz, cy, cx = r.centroid
        rows.append({"label": r.label, "volume_um3": r.area * vz * vy * vx, "n_slices": z1 - z0,
                     "z_first": int(z_values[z0]), "z_last": int(z_values[z1 - 1]),
                     "centroid_z_um": cz * vz, "centroid_y_um": cy * vy, "centroid_x_um": cx * vx,
                     "max_area_um2": float(max(areas)), "touches_stack_edge": bool(z0 == 0 or z1 == stack.shape[0])})
    return pd.DataFrame(rows, columns=cols).sort_values("volume_um3", ascending=False).reset_index(drop=True)
