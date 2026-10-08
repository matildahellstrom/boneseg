"""boneseg against zero-shot SAM with the prompts of Gu et al. (2025), on all three datasets.

Images: the Liu bone test slices (5 samples), the NOISe osteoclast test patches (5 batches) and the SegPC-2021
validation images with outlines. For three prompt draws per image (paper/prompts.py):
  points   boneseg (object points + shared background points); SAM ViT-B, micro-SAM ViT-B LM and MobileSAM ViT-T
           with the same points and background points ("_bg"), and with object points only (the paper's setting)
  boxes    the same SAMs with one prompt per box (masks joined); boneseg in box mode: background points outside all
           boxes, in each box the place least like that background becomes the object point, and the mask is kept
           inside the boxes
Metrics: Dice, NSD (bone 5 um, cells 2 px), and the area bias (method minus expert, percentage points of the image).
ViT-H runs in the Kaggle kernel (paper/kaggle_finetune_sam), not here. Writes paper/results/sam_zeroshot.csv.
Usage: python paper/sam_zeroshot.py [--datasets bone osteoclasts plasma] [--limit N]
"""
from __future__ import annotations

import argparse
import os
import time
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import prompts as P
from common import available_samples, load_sample
from boneseg import metrics
from boneseg.backbone import get_backbone
from boneseg.segment import SegmentationSettings, embed_image, prototypes, segment

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "results"
WEIGHTS = Path(os.environ.get("BONESEG_PAPER_WEIGHTS", Path.home() / ".cache" / "boneseg-paper"))
SAMS = {"sam": ("segment_anything", "vit_b", "sam_vit_b.pth"), "microsam": ("segment_anything", "vit_b", "microsam_vit_b_lm.pt"),
        "mobilesam": ("mobile_sam", "vit_t", "mobile_sam.pt")}


@lru_cache(maxsize=4)
def predictor(kind):
    pkg, arch, ckpt = SAMS[kind]
    mod = __import__(pkg)
    model = mod.sam_model_registry[arch](checkpoint=str(WEIGHTS / ckpt)).to("cpu").eval()
    return mod.SamPredictor(model)


def images(datasets, limit=None):
    """(dataset, id, image for the model, image for SAM as uint8 RGB, expert mask, instances or None, pixel size, nsd tolerance)."""
    if "bone" in datasets:
        for name in available_samples():
            for sl in load_sample(name).test[:limit]:
                rgb = (np.repeat(np.clip(sl.img, 0, 1)[..., None], 3, -1) * 255).astype(np.uint8)
                yield "bone", f"{name}_{sl.z}", sl.img, rgb, sl.gt, None, sl.pixel_um, 5.0
    if "osteoclasts" in datasets:
        from noise_common import BATCHES, load_batch
        for b in BATCHES:
            for p in load_batch(b)["test"][:limit]:
                yield "osteoclasts", f"{b}_{p.name}", p.rgb.astype(np.float32) / 255, p.rgb, p.gt, p.instances, (1.0, 1.0), 2.0
    if "plasma" in datasets:
        from segpc_common import load
        for it in [x for x in load("val") if len(x.cells)][:limit]:
            yield "plasma", it.name, it.rgb.astype(np.float32) / 255, it.rgb, it.cell_mask, it.instances, (1.0, 1.0), 2.0


def sam_predict_points(pred, pos, neg):
    pts = np.array([(x, y) for y, x in list(pos) + list(neg)], np.float32)
    lbl = np.r_[np.ones(len(pos)), np.zeros(len(neg))].astype(np.int32)
    m, _, _ = pred.predict(point_coords=pts, point_labels=lbl, multimask_output=False)
    return m[0].astype(bool)


def sam_predict_boxes(pred, boxes, shape):
    out = np.zeros(shape, bool)
    for b in boxes:
        m, _, _ = pred.predict(box=np.array(b, np.float32), multimask_output=False)
        out |= m[0].astype(bool)
    return out


def boneseg_boxes(emb, boxes, background, st, pixel_um, share: float = 0.3, cap: int = 15):
    """Box mode: background points outside all boxes; in each box, the share of feature cells least like that
    background (at most cap) become object clicks; threshold halfway (these clicks are noisy); mask kept inside the
    boxes. Share, cap and threshold were chosen on development images of the three datasets."""
    from dataclasses import replace
    N = prototypes(emb, background)
    g = torch.nn.functional.normalize(emb.grid.float(), dim=-1)
    sim_bg = (g @ torch.nn.functional.normalize(N.float().to(g.device), dim=-1).T).mean(-1).cpu().numpy()
    hg, wg = sim_bg.shape
    pos, inside = [], np.zeros((emb.height, emb.width), bool)
    for x0, y0, x1, y1 in boxes:
        inside[int(y0):int(np.ceil(y1)) + 1, int(x0):int(np.ceil(x1)) + 1] = True
        cy0 = int(y0 / emb.height * hg)
        cy1 = max(cy0 + 1, int(np.ceil(y1 / emb.height * hg)))
        cx0 = int(x0 / emb.width * wg)
        cx1 = max(cx0 + 1, int(np.ceil(x1 / emb.width * wg)))
        sub = sim_bg[cy0:cy1, cx0:cx1]
        for f in np.argsort(sub, axis=None)[:max(1, min(cap, int(share * sub.size)))]:
            iy, ix = np.unravel_index(f, sub.shape)
            pos.append((int((cy0 + iy + 0.5) / hg * emb.height), int((cx0 + ix + 0.5) / wg * emb.width)))
    return segment(emb, pos, background, replace(st, threshold_position=0.5), pixel_um).mask & inside


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["bone", "osteoclasts", "plasma"])
    ap.add_argument("--limit", type=int, default=None, help="Images per sample or batch, for a quick run")
    ap.add_argument("--out", default=str(OUT / "sam_zeroshot.csv"))
    args = ap.parse_args()
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
    st = SegmentationSettings()
    bb = get_backbone(st.backbone)
    rows, n = [], 0
    for ds, iid, img, rgb, gt, inst, px, tol in images(args.datasets, args.limit):
        emb = embed_image(bb, img, st)
        for kind in SAMS:
            predictor(kind).set_image(rgb)
            for seed in range(3):
                p = P.make(gt, inst, seed)
                preds = {f"{kind}_points": sam_predict_points(predictor(kind), p["points"], []),
                         f"{kind}_points_bg": sam_predict_points(predictor(kind), p["points"], p["background"]),
                         f"{kind}_boxes": sam_predict_boxes(predictor(kind), p["boxes"], gt.shape)}
                if kind == "sam":   # boneseg once per draw
                    preds["boneseg_points_bg"] = segment(emb, p["points"], p["background"], st, px).mask
                    preds["boneseg_boxes"] = boneseg_boxes(emb, p["boxes"], p["box_background"], st, px)
                for m, pr in preds.items():
                    method, prompt = m.split("_", 1)
                    rows.append({"dataset": ds, "image": iid, "seed": seed, "method": method, "prompt": prompt,
                                 "dice": metrics.dice(pr, gt), "nsd": metrics.nsd(pr, gt, tol, px, max_side=1500),
                                 "area_bias": 100 * (float(pr.mean()) - float(gt.mean())), "n_objects_prompted": len(p["points"])})
        n += 1
        if n % 20 == 0:
            log(f"{n} images")
            pd.DataFrame(rows).to_csv(args.out, index=False)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    df = pd.DataFrame(rows)
    print(df.groupby(["dataset", "prompt", "method"])[["dice", "nsd", "area_bias"]].mean().round(3).to_string())
    log("done")


if __name__ == "__main__":
    main()
