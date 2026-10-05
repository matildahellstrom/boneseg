"""Does fine-tuning DINOv2 make the masks better? Leave-one-sample-out on the test slices.

For each held-out sample and each label source, two models are trained with the same data, code and budget:
  probe_frozen   the 1 x 1 output layer on frozen DINOv2 Small features (train_blocks=0), the baseline
  finetune_bK    the same, with the last K transformer blocks trained as well
Label sources, as in evaluate.py:
  same sample    5 labelled development slices of the held-out sample; its other 5 development slices validate
  other samples  5 labelled development slices of each other sample; their other development slices validate
The held-out sample's test slices are only used for the final scores. With labels from other samples, the
fine-tuned backbone is also used for click segmentation (default settings, the evaluation's click seeds), against
the frozen backbone with the same clicks, to see whether the adapted features help on a sample never trained on.

Writes paper/results/finetune.csv and finetune_runs.json. Usage: python paper/finetune_experiment.py [--blocks 4]
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import available_samples, bone_measures, load_sample, simulated_clicks
from boneseg import metrics
from boneseg.backbone import get_backbone, pick_device
from boneseg.finetune import build, finetune, load_dino, predict, prepare
from boneseg.segment import SegmentationSettings, embed_image, segment

OUT = Path(__file__).resolve().parent / "results"
BUDGETS = [(3, 6), (25, 25)]


def score(pred, sl):
    pm, gm = bone_measures(pred, sl.pixel_um), bone_measures(sl.gt, sl.pixel_um)
    return {"dice": metrics.dice(pred, sl.gt), "hd95_um": metrics.hd95(pred, sl.gt, sl.pixel_um, 2048),
            **{f"pred_{k}": v for k, v in pm.items()}, **{f"gt_{k}": v for k, v in gm.items()}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="*", default=None)
    ap.add_argument("--blocks", type=int, nargs="*", default=[4])
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--no-clicks", action="store_true")
    args = ap.parse_args(argv)
    names = args.samples or available_samples()
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
    samples = {n: load_sample(n) for n in names}
    device = pick_device()
    dino = load_dino("dinov2_s14")
    ft_dir = OUT / "finetuned"
    ft_dir.mkdir(parents=True, exist_ok=True)
    rows, runs = [], []
    st = SegmentationSettings()   # The app's defaults for clicks
    frozen_bb = None
    for held in names:
        tgt = samples[held]
        others = [samples[n] for n in names if n != held]
        sources = {"same sample": ([(sl.img, sl.gt) for sl in tgt.dev[:5]], [(sl.img, sl.gt) for sl in tgt.dev[5:]]),
                   "other samples": ([(sl.img, sl.gt) for s in others for sl in s.dev[:5]], [(sl.img, sl.gt) for s in others for sl in s.dev[5:]])}
        test = [(sl, prepare(sl.img, sl.gt, 980)) for sl in tgt.test]
        for source, (train, val) in sources.items():
            for blocks in [0] + args.blocks:
                method = "probe_frozen" if blocks == 0 else f"finetune_b{blocks}"
                log(f"held-out {held}, labels from {source}: {method}")
                ft = finetune(train, val, train_blocks=blocks, steps=args.steps, pretrained=dino, device=device)
                runs.append({"held_out": held, "source": source, "method": method, **{k: v for k, v in ft.info.items() if k != "history"},
                             "history": ft.info["history"]})
                log(f"  best validation Dice {ft.info['best_val_dice']:.3f} at step {ft.info['best_step']}")
                net = build(ft, dino=__import__("copy").deepcopy(dino), device=device)
                for sl, p in test:
                    rows.append({"sample": held, "z": sl.z, "method": method, "condition": f"labels from {source}",
                                 **score(predict(net, p, device) >= 0.5, sl)})
                del net
                if blocks and source == "other samples" and not args.no_clicks:
                    path = ft_dir / f"finetune_b{blocks}_without_{held}.pt"
                    ft.save(path)
                    bbs = {"clicks_frozen": "dinov2_s14", f"clicks_finetuned_b{blocks}": f"dinov2_s14@{path}"}
                    for cmethod, bname in bbs.items():
                        if cmethod == "clicks_frozen" and any(r["method"] == "clicks_frozen" and r["sample"] == held for r in rows):
                            continue
                        bb = get_backbone(bname)
                        for sl, _ in test:
                            emb = embed_image(bb, sl.img, st)
                            for budget, seed in itertools.product(BUDGETS, range(args.seeds)):
                                pos, neg = simulated_clicks(sl.gt, *budget, seed + 100)
                                rows.append({"sample": held, "z": sl.z, "method": cmethod, "condition": f"{budget[0]}+{budget[1]} clicks, clean",
                                             "seed": seed, **score(segment(emb, pos, neg, st, sl.pixel_um).mask, sl)})
                    log(f"  clicks with frozen and fine-tuned features done")
            pd.DataFrame(rows).to_csv(OUT / "finetune.csv", index=False)
            (OUT / "finetune_runs.json").write_text(json.dumps(runs, indent=1))
    df = pd.DataFrame(rows)
    per = df.groupby(["condition", "method", "sample"])["dice"].mean().unstack("sample")
    per["mean"] = per.mean(axis=1)
    print(per.round(3).to_string())
    log("done")


if __name__ == "__main__":
    main()
