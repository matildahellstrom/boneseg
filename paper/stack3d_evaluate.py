"""Whole-stack 3D evaluation on the Liu samples: boneseg's bone volumes against the experts' 3D masks.

For each sample, the central 80% of the stack is segmented slice by slice, three ways:
  clicks          25 + 25 simulated clicks on the middle slice only; the prototypes and the click-calibrated raw
                  threshold carry over to every other slice (the app's stack mode), app default settings
  labels_same     the learned model from 5 labelled development slices of the same sample (about 1% of the
                  evaluated slices, so the volume is not fully independent of its training)
  labels_other    the learned model from 5 labelled development slices of each other sample
The predicted and expert stacks are reduced the same way (block-averaged 4 x 4 in-plane and 2 x in z, then
thresholded at one half; voxels 4 x 6.5 x 6.5 um) and compared in 3D: Dice, bone volume fraction BV/TV (TV is the
whole evaluated block), bone surface density BS/BV from a marching-cubes surface, and the plate-model measures
Tb.Th = 2 BV/BS, Tb.N = (BV/TV) / Tb.Th and Tb.Sp = 1 / Tb.N - Tb.Th (Parfitt). Per slice, the bone area fraction
is recorded for both, to see how well the method follows the expert through the stack.
Writes paper/results/stack3d.csv (one row per sample and method) and stack3d_slices.csv.
Usage: python paper/stack3d_evaluate.py [--samples A C E F]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import SAMPLES, DATA, available_samples, load_sample, simulated_clicks
from boneseg import io as bio
from boneseg.backbone import get_backbone
from boneseg.head import segment_with_head, train_head
from boneseg.segment import SegmentationSettings, embed_image, prototypes, segment, segment_with_prototypes

OUT = Path(__file__).resolve().parent / "results"
XY, ZF = 4, 2


def reduce2d(m: np.ndarray) -> np.ndarray:
    h, w = (m.shape[0] // XY) * XY, (m.shape[1] // XY) * XY
    return m[:h, :w].reshape(h // XY, XY, w // XY, XY).mean((1, 3)).astype(np.float16)


def measures(vol: np.ndarray, spacing) -> dict:
    """3D measures of a binary volume with voxel spacing (z, y, x) in um."""
    from skimage.measure import marching_cubes, mesh_surface_area
    voxel = float(np.prod(spacing))
    bv = vol.sum() * voxel
    tv = vol.size * voxel
    if vol.any():
        padded = np.pad(vol, 1, mode="edge").astype(np.float32)   # Cut faces at the block edge are not bone surface
        verts, faces, _, _ = marching_cubes(padded, 0.5, spacing=spacing)
        bs = mesh_surface_area(verts, faces)
    else:
        bs = 0.0
    out = {"BV_mm3": bv / 1e9, "BV/TV_%": 100 * bv / tv, "BS/BV_per_mm": 1e3 * bs / bv if bv else np.nan}
    th = 2 * bv / bs if bs else np.nan
    out["Tb.Th_um"] = th
    tbn = (bv / tv) / th if th and th == th else np.nan
    out["Tb.N_per_mm"] = 1e3 * tbn if tbn == tbn else np.nan
    out["Tb.Sp_um"] = 1 / tbn - th if tbn and tbn == tbn else np.nan
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="*", default=None)
    args = ap.parse_args(argv)
    names = args.samples or available_samples()
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
    st = SegmentationSettings()
    bb = get_backbone(st.backbone)
    samples = {n: load_sample(n) for n in names}
    rows, slice_rows = [], []
    for name in names:
        s = samples[name]
        vol, ch = s.vol, s.channel
        mask_ch = vol.n_channels - 1
        lo, hi = int(0.1 * (vol.n_z - 1)), int(0.9 * (vol.n_z - 1))
        zs = list(range(lo, hi + 1))
        ref_z = zs[len(zs) // 2]
        img_ref = bio.normalize_plane(vol.get_plane(ch, ref_z))
        gt_ref = vol.get_plane(mask_ch, ref_z) > 0
        emb_ref = embed_image(bb, img_ref, st)
        pos, neg = simulated_clicks(gt_ref, 25, 25, 0)
        P, N = prototypes(emb_ref, pos), prototypes(emb_ref, neg)
        thr = segment(emb_ref, pos, neg, st, vol.pixel_um).raw_threshold
        same = [(embed_image(bb, sl.img, st), sl.gt) for sl in s.dev[:5]]
        other = [(embed_image(bb, sl.img, st), sl.gt) for n2 in names if n2 != name for sl in samples[n2].dev[:5]]
        heads = {"labels_same": train_head(same, st, [(0, i) for i in range(len(same))], pixel_um=vol.pixel_um),
                 "labels_other": train_head(other, st, [(0, i) for i in range(len(other))], pixel_um=vol.pixel_um)}
        log(f"{name}: {len(zs)} slices ({lo}-{hi}), reference slice {ref_z}; learned models ready")
        red = {k: [] for k in ("expert", "clicks", "labels_same", "labels_other")}
        for i, z in enumerate(zs):
            img = bio.normalize_plane(vol.get_plane(ch, z))
            gt = vol.get_plane(mask_ch, z) > 0
            emb = embed_image(bb, img, st)
            preds = {"expert": gt, "clicks": segment_with_prototypes(emb, P, N, st, vol.pixel_um, raw_threshold=thr).mask}
            for k, h in heads.items():
                preds[k] = segment_with_head(h, emb, st, vol.pixel_um).mask
            for k, m in preds.items():
                red[k].append(reduce2d(m))
            slice_rows.append({"sample": name, "z": z, **{f"area_{k}": 100 * float(m.mean()) for k, m in preds.items()},
                               **{f"dice_{k}": float(2 * (m & gt).sum() / max(1, m.sum() + gt.sum())) for k, m in preds.items() if k != "expert"}})
            if (i + 1) % 100 == 0:
                log(f"  {name}: {i + 1} of {len(zs)} slices")
        spacing = (vol.voxel_um[0] * ZF, vol.voxel_um[1] * XY, vol.voxel_um[2] * XY)
        vols = {}
        for k, lst in red.items():
            a = np.stack(lst)
            nz = (a.shape[0] // ZF) * ZF
            vols[k] = a[:nz].reshape(nz // ZF, ZF, *a.shape[1:]).astype(np.float32).mean(1) >= 0.5
        e = vols["expert"]
        for k, v in vols.items():
            row = {"sample": name, "method": k, "n_slices": len(zs), **measures(v, spacing)}
            if k != "expert":
                row["dice_3d"] = float(2 * (v & e).sum() / max(1, v.sum() + e.sum()))
            rows.append(row)
        log(f"{name}: done; 3D Dice " + ", ".join(f"{r['method']} {r['dice_3d']:.3f}" for r in rows if r["sample"] == name and "dice_3d" in r))
        pd.DataFrame(rows).to_csv(OUT / "stack3d.csv", index=False)
        pd.DataFrame(slice_rows).to_csv(OUT / "stack3d_slices.csv", index=False)
    log("done")


if __name__ == "__main__":
    main()
