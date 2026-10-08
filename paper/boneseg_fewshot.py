"""boneseg's few-shot arms on exactly the folds, few-shot images and prompts of the fine-tuned-SAM kernels
(paper/kaggle_finetune_sam/setup.json), for the comparison with Gu et al. (2025):

  learned            boneseg's learned model from the 5 labelled images (no prompt)
  ftlayer            DINOv2 fine-tuned on the 5 images (5 more for early stopping), its own output layer (no prompt)
  ftclicks           clicks on the fine-tuned DINOv2 with the paper-style prompts: object + background points,
                     and box mode (as boneseg in sam_zeroshot.py)
Folds: Liu, each held-out sample with its first 5 development slices; NOISe, each held-out batch with its first 5
development patches; SegPC, three draws of 5 training images. Writes paper/results/boneseg_fewshot.csv.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sam_zeroshot as Z
from common import load_sample
from boneseg import metrics
from boneseg.backbone import get_backbone, pick_device
from boneseg.finetune import build, finetune, load_dino, predict, prepare
from boneseg.head import segment_with_head, train_head
from boneseg.segment import SegmentationSettings, embed_image, segment

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
SETUP = json.loads((HERE / "kaggle_finetune_sam" / "setup.json").read_text())


def folds():
    """(dataset, draw, train [(img, gt)], val [(img, gt)], test [(key, img, gt, pixel size, tolerance)])."""
    from noise_common import load_batch
    from segpc_common import load
    for s in SETUP["liu"]:
        smp = load_sample(s)
        yield ("bone", "0", [(sl.img, sl.gt) for sl in smp.dev[:5]], [(sl.img, sl.gt) for sl in smp.dev[5:]],
               [(f"liu/{s}_{sl.z}", sl.img, sl.gt, sl.pixel_um, 5.0) for sl in smp.test])
    for b in SETUP["noise"]:
        d = load_batch(b)
        rgb = lambda p: p.rgb.astype(np.float32) / 255  # noqa: E731
        yield ("osteoclasts", "0", [(rgb(p), p.gt) for p in d["dev"][:5]], [(rgb(p), p.gt) for p in d["dev"][5:10]],
               [(f"noise/{b}_{p.name}", rgb(p), p.gt, (1.0, 1.0), 2.0) for p in d["test"]])
    tr = {it.name: it for it in load("train")}
    val = [it for it in load("val") if len(it.cells)]
    rgb = lambda it: it.rgb.astype(np.float32) / 255  # noqa: E731
    for i, dr in enumerate(SETUP["segpc"]["draws"]):
        yield ("plasma", str(i), [(rgb(tr[n]), tr[n].cell_mask) for n in dr["train"]], [(rgb(tr[n]), tr[n].cell_mask) for n in dr["val"]],
               [(f"segpc/{it.name}", rgb(it), it.cell_mask, (1.0, 1.0), 2.0) for it in val])


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
    st = SegmentationSettings()
    bb = get_backbone(st.backbone)
    device = pick_device()
    dino = load_dino()
    ftdir = OUT / "finetuned"
    ftdir.mkdir(exist_ok=True)
    rows = []

    def add(ds, draw, key, method, prompt, seed, pred, gt, px, tol):
        rows.append({"dataset": ds, "image": key.split("/")[1], "method": method, "prompt": prompt, "seed": seed, "draw": draw,
                     "dice": metrics.dice(pred, gt), "nsd": metrics.nsd(pred, gt, tol, px, max_side=1500),
                     "area_bias": 100 * (float(pred.mean()) - float(gt.mean()))})

    for n_fold, (ds, draw, train, val, test) in enumerate(folds()):
        head = train_head([(embed_image(bb, img, st), g) for img, g in train], st, [(0, i) for i in range(len(train))])
        ft = finetune(train, val, train_blocks=4, steps=1000, pretrained=dino, device=device)
        path = ftdir / f"fewshot_{ds}_{n_fold}.pt"
        ft.save(path)
        net = build(ft, dino=copy.deepcopy(dino), device=device)
        bb_ft = get_backbone(f"dinov2_s14@{path}")
        for key, img, gt, px, tol in test:
            emb = embed_image(bb, img, st)
            add(ds, draw, key, "boneseg_learned", "none", 0, segment_with_head(head, emb, st, px).mask, gt, px, tol)
            add(ds, draw, key, "boneseg_ftlayer", "none", 0, predict(net, prepare(img, gt, 980), device) >= 0.5, gt, px, tol)
            emb_ft = embed_image(bb_ft, img, st)
            for seed, p in enumerate(SETUP["prompts"][key]):
                add(ds, draw, key, "boneseg_ftclicks", "points_bg", seed, segment(emb_ft, p["points"], p["background"], st, px).mask, gt, px, tol)
                add(ds, draw, key, "boneseg_ftclicks", "boxes", seed, Z.boneseg_boxes(emb_ft, p["boxes"], p["box_background"], st, px), gt, px, tol)
        del net
        bb = get_backbone(st.backbone)
        log(f"{ds} fold {n_fold} done (fine-tuning best validation Dice {ft.info['best_val_dice']:.3f})")
        pd.DataFrame(rows).to_csv(OUT / "boneseg_fewshot.csv", index=False)
    df = pd.DataFrame(rows)
    print(df.groupby(["dataset", "method", "prompt"])[["dice", "nsd", "area_bias"]].mean().round(3).to_string())
    log("done")


if __name__ == "__main__":
    main()
