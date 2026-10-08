"""Prompts as in Gu et al. (2025), "How to build the best medical image segmentation algorithm using foundation
models" (MELBA; code github.com/mazurowski-lab/finetune-SAM, utils/funcs.py), shared by boneseg and SAM.

  points  for the largest k objects, a random pixel among each object's 10% most interior pixels (largest distance to
          its edge), as get_first_prompt with region_type 'largest_k'; given to SAM together in one prompt
  boxes   for the largest k objects, the tight box widened on each side by a random share of up to 10% of its size,
          as MaskToBoxSimple / get_top_boxes; one SAM prompt per box, masks joined
The paper's prompts have no background points. boneseg needs some, so both methods get the same background points
(away from every object); SAM is also scored with object points only, the paper's exact setting.
Objects are the expert instances where they exist (NOISe, SegPC), otherwise the 8-connected pieces of the mask (Liu).
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi

K = 5          # Largest objects prompted, as region_type 'largest_5' in the paper's code
INTERIOR = 0.1  # dist_thre_ratio: top share of interior pixels points are drawn from
BOX_NOISE = 0.1


def objects(gt: np.ndarray, instances: np.ndarray | None = None, k: int | None = K) -> list[np.ndarray]:
    """Object masks, largest first (at most k)."""
    if instances is None:
        lab, n = ndi.label(gt, structure=np.ones((3, 3)))
    else:
        lab, n = instances, int(instances.max())
    sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    order = [i + 1 for i in np.argsort(-sizes) if sizes[i] > 0]
    return [lab == i for i in (order[:k] if k else order)]


def centre_point(obj: np.ndarray, rng) -> tuple[int, int]:
    d = ndi.distance_transform_edt(np.pad(obj, 1))[1:-1, 1:-1]
    vals = np.sort(d[d > 0])[::-1]
    thr = max(vals[int(INTERIOR * len(vals))] if len(vals) else 1, 1)
    ys, xs = np.nonzero(d >= thr)
    if not len(ys):
        ys, xs = np.nonzero(obj)
    i = rng.integers(len(ys))
    return (int(ys[i]), int(xs[i]))


def noisy_box(obj: np.ndarray, rng) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1), each side pushed out by up to 10% of the box size."""
    ys, xs = np.nonzero(obj)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = obj.shape
    ty, tx = (y1 - y0) * BOX_NOISE, (x1 - x0) * BOX_NOISE
    return (max(0.0, x0 - tx * rng.random()), max(0.0, y0 - ty * rng.random()),
            min(float(w), x1 + tx * rng.random()), min(float(h), y1 + ty * rng.random()))


def background_points(gt: np.ndarray, n: int, rng, margin: int = 10) -> list[tuple[int, int]]:
    far = ~ndi.binary_dilation(gt, iterations=margin)
    far = far if far.any() else ~gt
    ys, xs = np.nonzero(far)
    idx = rng.choice(len(ys), n, replace=len(ys) < n)
    return [(int(ys[i]), int(xs[i])) for i in idx]


def background_outside_boxes(shape, boxes, n: int, rng) -> list[tuple[int, int]]:
    inside = np.zeros(shape, bool)
    for x0, y0, x1, y1 in boxes:
        inside[int(y0):int(np.ceil(y1)) + 1, int(x0):int(np.ceil(x1)) + 1] = True
    ys, xs = np.nonzero(~inside)
    if not len(ys):
        return []
    idx = rng.choice(len(ys), n, replace=len(ys) < n)
    return [(int(ys[i]), int(xs[i])) for i in idx]


def make(gt: np.ndarray, instances: np.ndarray | None, seed: int, n_background: int = 10) -> dict:
    """One draw of point and box prompts for an image."""
    rng = np.random.default_rng(seed)
    objs = objects(gt, instances)
    pts = [centre_point(o, rng) for o in objs]
    boxes = [noisy_box(o, rng) for o in objs]
    return {"points": pts, "background": background_points(gt, n_background, rng), "boxes": boxes,
            "box_background": background_outside_boxes(gt.shape, boxes, n_background, rng)}
