"""Comparing samples between groups, such as control and treated."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

# Stack measurements that can be compared, with labels and units for display
METRICS = {
    "mean_area_fraction": ("Mean area fraction", "%", 100.0),
    "volume_um3": ("Segmented volume", "µm³", 1.0),
    "n_objects_3d": ("Objects in 3D", "", 1.0),
    "n_objects_3d_inside": ("Objects in 3D, not cut by the stack ends", "", 1.0),
    "median_object_volume_um3": ("Median object volume", "µm³", 1.0),
    "objects_per_mm3": ("Objects per mm³", "/mm³", 1.0),
    "mean_dice_vs_reference": ("Mean Dice against the reference", "", 1.0),
    "B.Ar/T.Ar_%": ("Bone area fraction, B.Ar/T.Ar", "%", 1.0),
    "Oc.Pm/B.Pm_%": ("Osteoclast surface, Oc.Pm/B.Pm", "%", 1.0),
    "N.Oc/B.Pm_per_mm": ("Osteoclast number, N.Oc/B.Pm", "/mm", 1.0),
}


def derived(summary: dict, voxel_um, n_pixels: int) -> dict:
    """Adds measurements computed from a stack summary, such as object density."""
    out = dict(summary)
    out.update(summary.get("histomorphometry") or {})  # Stack-level histomorphometry from multi-structure runs
    n = summary.get("n_slices", 0)
    if n and summary.get("n_objects_3d") is not None:
        span_um = n * summary.get("slice_spacing_um", voxel_um[0])
        volume_mm3 = n_pixels * voxel_um[1] * voxel_um[2] * span_um / 1e9
        out["objects_per_mm3"] = summary["n_objects_3d"] / volume_mm3 if volume_mm3 else None
    return out


def compare_groups(values: dict[str, list[float]]) -> dict:
    """Rank-based test between groups: Mann-Whitney U for two groups, Kruskal-Wallis for more.
    Rank tests make no normality assumption, which suits the handful of samples a study usually has."""
    groups = {g: [float(v) for v in vs if v is not None and np.isfinite(v)] for g, vs in values.items()}
    groups = {g: vs for g, vs in groups.items() if vs}
    summary = {g: {"n": len(vs), "median": float(np.median(vs)), "mean": float(np.mean(vs)),
                   "sd": float(np.std(vs, ddof=1)) if len(vs) > 1 else None} for g, vs in groups.items()}
    out = {"groups": summary, "test": None, "p_value": None, "note": ""}
    if len(groups) < 2:
        out["note"] = "Put samples in at least two groups to compare them."
        return out
    if min(len(v) for v in groups.values()) < 2:
        out["note"] = "Each group needs at least two samples for a test."
        return out
    if len(groups) == 2:
        (ga, a), (gb, b) = groups.items()
        res = stats.mannwhitneyu(a, b, alternative="two-sided")
        out.update(test="Mann-Whitney U", statistic=float(res.statistic), p_value=float(res.pvalue),
                   effect=f"median {gb} − {ga} = {np.median(b) - np.median(a):.4g}")
    else:
        res = stats.kruskal(*groups.values())
        out.update(test="Kruskal-Wallis H", statistic=float(res.statistic), p_value=float(res.pvalue))
    if len(groups) == 2:
        n1, n2 = (len(v) for v in groups.values())
        # The most extreme arrangement of ranks gives the smallest possible two-sided p-value
        p_floor = 2 / math.comb(n1 + n2, n1)
        out["min_possible_p"] = p_floor
        if p_floor >= 0.05:
            out["note"] = (f"With {n1} and {n2} samples, the smallest p-value this test can give is {p_floor:.2f}, "
                           "so it cannot show a difference at p < 0.05. Add samples before drawing conclusions.")
    elif min(len(v) for v in groups.values()) < 4:
        out["note"] = "Groups this small give the test little power. Treat the p-value with care."
    return out
