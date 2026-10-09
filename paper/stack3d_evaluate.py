"""Whole-stack 3D evaluation on the Liu samples: boneseg's bone volumes against the experts' 3D masks.

For each sample, the central 80% of the stack is segmented slice by slice, three ways:
  clicks          25 + 25 simulated clicks on the middle slice only; the prototypes and the click-calibrated raw
                  threshold carry over to every other slice (the app's stack mode), app default settings
  labels_same     the learned model from 5 labelled development slices of the same sample (about 1% of the
                  evaluated slices, so the volume is not fully independent of its training)
  labels_other    the learned model from 5 labelled development slices of each other sample
Variants of the learned model (all from the same training slices):
  ..._cal         its probability threshold calibrated on held-out labelled slices (boneseg.head.calibrate_threshold)
                  instead of 0.5
  ..._zs          probabilities smoothed along z (Gaussian, sigma ZS_SIGMA slices) before the calibrated threshold
  labels_same_zf  a model that also sees the mean features of the slices ZF_DIST below and above (with_z_context);
                  calibrated, and with _zs also smoothed
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
import torch

from common import available_samples, load_sample, simulated_clicks
from boneseg import io as bio
from boneseg.backbone import get_backbone
from boneseg.head import smooth_z, train_head, with_z_context
from boneseg.segment import SegmentationSettings, embed_image, prototypes, segment, segment_with_prototypes, upsample

OUT = Path(__file__).resolve().parent / "results"
XY, ZF = 4, 2
ZS_SIGMA = 1.0   # Slices; Liu slices are 2 um apart
ZF_DIST = 2      # Slices between a slice and the neighbours whose features the z-context model sees


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
    ap.add_argument("--d-block", default="224-287", help="Sample D, read remotely: a contiguous block of slices (two 32-slice storage blocks)")
    ap.add_argument("--append", action="store_true", help="Add to the existing results instead of replacing them")
    ap.add_argument("--limit", type=int, default=0, help="Only the first N slices of each stack, for a quick check")
    ap.add_argument("--out", default="stack3d", help="Name of the result files")
    args = ap.parse_args(argv)
    names = args.samples or available_samples()
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
    st = SegmentationSettings()
    bb = get_backbone(st.backbone)
    samples = {n: load_sample(n) for n in names}
    rows, slice_rows = [], []
    if args.append and (OUT / f"{args.out}.csv").exists():
        rows = [r for r in pd.read_csv(OUT / f"{args.out}.csv").to_dict("records") if r["sample"] not in names]
        slice_rows = [r for r in pd.read_csv(OUT / f"{args.out}_slices.csv").to_dict("records") if r["sample"] not in names]
    for name in names:
        s = samples[name]
        vol, ch = s.vol, s.channel
        if vol is None:   # Sample D, read from Kaggle over range requests
            from remote_d import remote_volume
            vol = remote_volume()
        mask_ch = vol.n_channels - 1
        lo, hi = int(0.1 * (vol.n_z - 1)), int(0.9 * (vol.n_z - 1))
        if s.vol is None:
            lo, hi = (int(v) for v in args.d_block.split("-"))
        zs = list(range(lo, hi + 1))[:args.limit or None]
        plane = lambda z: bio.normalize_plane(vol.get_plane(ch, int(np.clip(z, 0, vol.n_z - 1))))  # noqa: E731
        ref_z = zs[len(zs) // 2]
        gt_ref = vol.get_plane(mask_ch, ref_z) > 0
        emb_ref = embed_image(bb, plane(ref_z), st)
        pos, neg = simulated_clicks(gt_ref, 25, 25, 0)
        P, N = prototypes(emb_ref, pos), prototypes(emb_ref, neg)
        thr = segment(emb_ref, pos, neg, st, vol.pixel_um).raw_threshold
        same = [(embed_image(bb, sl.img, st), sl.gt) for sl in s.dev[:5]]
        same_z = [(with_z_context(e, embed_image(bb, plane(sl.z - ZF_DIST), st), embed_image(bb, plane(sl.z + ZF_DIST), st)), g)
                  for (e, g), sl in zip(same, s.dev[:5])]
        others = [n2 for n2 in available_samples() if n2 != name]
        for n2 in others:
            samples.setdefault(n2, load_sample(n2))
        other = [(embed_image(bb, sl.img, st), sl.gt) for n2 in others for sl in samples[n2].dev[:5]]
        keys = lambda lst: [(0, i) for i in range(len(lst))]  # noqa: E731
        heads = {"same": train_head(same, st, keys(same), pixel_um=vol.pixel_um),
                 "other": train_head(other, st, keys(other), pixel_um=vol.pixel_um),
                 "zf": train_head(same_z, st, keys(same_z), pixel_um=vol.pixel_um)}
        log(f"{name}: {len(zs)} slices ({lo}-{hi}), reference slice {ref_z}; calibrated thresholds "
            + ", ".join(f"{k} {h.threshold:.2f}" for k, h in heads.items()))
        # Pass 1: embed every slice once; clicks give masks directly, the learned models probabilities on the patch
        # grid (small), so they can be smoothed along z afterwards. The z-context model needs the slices ZF_DIST
        # below and above, so it runs ZF_DIST slices behind.
        red = {"expert": [], "clicks": []}
        click_rows = []
        probs = {"same": [], "other": [], "zf": [None] * len(zs)}
        grids, size = {}, None

        def zf_prob(j):
            e = with_z_context(grids[j], grids[max(j - ZF_DIST, 0)], grids[min(j + ZF_DIST, len(zs) - 1)])
            probs["zf"][j] = torch.sigmoid(heads["zf"].logits(e)).numpy().astype(np.float16)

        for i, z in enumerate(zs):
            gt = vol.get_plane(mask_ch, z) > 0
            emb = embed_image(bb, plane(z), st)
            size = (emb.height, emb.width)
            m = segment_with_prototypes(emb, P, N, st, vol.pixel_um, raw_threshold=thr).mask
            red["expert"].append(reduce2d(gt))
            red["clicks"].append(reduce2d(m))
            click_rows.append((100 * float(gt.mean()), 100 * float(m.mean()), float(2 * (m & gt).sum() / max(1, m.sum() + gt.sum()))))
            for k in ("same", "other"):
                probs[k].append(torch.sigmoid(heads[k].logits(emb)).numpy().astype(np.float16))
            grids[i] = emb
            if i - ZF_DIST >= 0:
                zf_prob(i - ZF_DIST)
            grids.pop(i - 2 * ZF_DIST, None)
            if (i + 1) % 100 == 0:
                log(f"  {name}: {i + 1} of {len(zs)} slices")
        for j in range(max(0, len(zs) - ZF_DIST), len(zs)):
            zf_prob(j)
        grids.clear()
        # Pass 2: the learned-model variants as masks, from the stored probabilities
        variants = {"labels_same": ("same", 0.5, False), "labels_same_cal": ("same", heads["same"].threshold, False),
                    "labels_same_cal_zs": ("same", heads["same"].threshold, True),
                    "labels_same_zf": ("zf", heads["zf"].threshold, False), "labels_same_zf_zs": ("zf", heads["zf"].threshold, True),
                    "labels_other": ("other", 0.5, False), "labels_other_cal": ("other", heads["other"].threshold, False)}
        smoothed = {k: smooth_z([p.astype(np.float32) for p in v], ZS_SIGMA) for k, v in probs.items()}
        for k in variants:
            red[k] = []
        for i, z in enumerate(zs):
            gt = vol.get_plane(mask_ch, z) > 0
            row = {"sample": name, "z": z, "area_expert": click_rows[i][0], "area_clicks": click_rows[i][1], "dice_clicks": click_rows[i][2]}
            for k, (src, t, zsm) in variants.items():
                p = (smoothed if zsm else probs)[src][i].astype(np.float32)
                m = upsample(torch.from_numpy(p), size).clip(0, 1) >= t
                red[k].append(reduce2d(m))
                row[f"area_{k}"] = 100 * float(m.mean())
                row[f"dice_{k}"] = float(2 * (m & gt).sum() / max(1, m.sum() + gt.sum()))
            slice_rows.append(row)
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
            if k.startswith("labels"):
                row["threshold"] = variants[k][1]
            rows.append(row)
        log(f"{name}: done; 3D Dice " + ", ".join(f"{r['method']} {r['dice_3d']:.3f}" for r in rows if r["sample"] == name and "dice_3d" in r))
        pd.DataFrame(rows).to_csv(OUT / f"{args.out}.csv", index=False)
        pd.DataFrame(slice_rows).to_csv(OUT / f"{args.out}_slices.csv", index=False)
    log("done")


if __name__ == "__main__":
    main()
