"""micro-SAM against boneseg and fine-tuned SAM, on the same test images, folds and prompts (Gu et al. protocol).

Merges results/sam_zeroshot.csv (zero-shot prompts), boneseg_fewshot.csv (boneseg, 5-shot), finetune_sam.csv (SAM
fine-tuned with finetune-SAM, 5-shot) and microsam.csv (the micro-SAM Kaggle kernels, scored by finetune_sam_score.py)
into results/microsam_summary.md, microsam_summary.csv and figures/microsam_compare.png.
Seeds and few-shot draws are averaged per image; intervals are hierarchical bootstraps over groups and images.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analyze import hier_boot  # noqa: E402

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
NAMES = {"boneseg": "boneseg (clicks)", "microsam": "micro-SAM generalist (vit_b_lm)", "sam": "SAM ViT-B",
         "boneseg_learned": "boneseg learned model", "boneseg_ftlearned": "boneseg, fine-tuned DINOv2 + learned model",
         "boneseg_ftclicks": "boneseg, fine-tuned DINOv2 + prompts", "samft_b_ende_adapter": "SAM ViT-B fine-tuned (enc+dec Adapter)",
         "samft_b_ende_lora": "SAM ViT-B fine-tuned (enc+dec LoRA)", "microsam_zs": "micro-SAM generalist, decoder (no training)",
         "microsam_full": "micro-SAM fine-tuned, full", "microsam_lora": "micro-SAM fine-tuned, LoRA"}
PROMPTS = {"points": "object points", "points_bg": "object + background points", "boxes": "noisy boxes", "none": "no prompt"}
AUTO = ["boneseg_learned", "boneseg_ftlearned", "samft_b_ende_adapter", "samft_b_ende_lora", "microsam_zs", "microsam_full", "microsam_lora"]
PROMPTED = [("boneseg", "zero-shot"), ("sam", "zero-shot"), ("microsam", "zero-shot"), ("boneseg_ftclicks", "5-shot"),
            ("microsam_full", "5-shot"), ("microsam_lora", "5-shot")]
DATASETS = {"bone": "Bone (Liu, 5 held-out samples)", "osteoclasts": "Osteoclasts (NOISe, 5 held-out batches)",
            "plasma": "Plasma cells (SegPC, 3 few-shot draws)"}


def load() -> pd.DataFrame:
    frames = []
    for f in ("sam_zeroshot.csv", "boneseg_fewshot.csv", "finetune_sam.csv", "microsam.csv"):
        if (RES / f).exists():
            d = pd.read_csv(RES / f, dtype={"image": str})
            if "draw" not in d:
                d["draw"] = "0"
            frames.append(d.assign(source=f.split(".")[0]))
    df = pd.concat(frames, ignore_index=True)
    df["draw"] = df["draw"].astype(str)
    df = df[df["draw"] != "full"]
    zero = (df["source"] == "sam_zeroshot") | df["method"].isin(["samh", "microsam_zs"])
    df["setting"] = np.where(zero, "zero-shot", "5-shot")
    df["sample"] = np.where(df["dataset"] == "plasma", "segpc", df["image"].str.split("_").str[0])
    return df.groupby(["dataset", "setting", "method", "prompt", "sample", "image"], as_index=False)[["dice", "nsd", "area_bias"]].mean()


def summarize(df: pd.DataFrame):
    rows, lines = [], ["# micro-SAM against boneseg and fine-tuned SAM\n",
                       "Test images of the three datasets; 5-shot = 5 labelled images of the target (5 more stop the training "
                       "early). micro-SAM (Archit et al. 2025) is fine-tuned from vit_b_lm with its own training code "
                       "(boneseg/microsam_core.py); its automatic mask is the decoder's foreground >= 0.5. Dice, NSD (bone 5 µm, "
                       "cells 2 px) and area bias (percentage points), mean over images with 95% intervals.\n"]

    def row(ds, g, setting, prompt, method):
        e, lo, hi = hier_boot(g, "dice")
        r = {"dataset": ds, "setting": setting, "prompt": prompt, "method": method, "dice": e, "lo": lo, "hi": hi,
             "nsd": g["nsd"].mean(), "area_bias": g["area_bias"].mean(), "n": len(g)}
        rows.append(r)
        return f"| {NAMES.get(method, method)} | {setting} | {PROMPTS.get(prompt, prompt)} | {e:.3f} ({lo:.3f}–{hi:.3f}) | {r['nsd']:.3f} | {r['area_bias']:+.2f} |"

    for ds, title in DATASETS.items():
        d = df[df["dataset"] == ds]
        lines += [f"\n## {title}\n", "**Automatic (no prompt)**\n", "| Method | Setting | Prompt | Dice (95% CI) | NSD | Area bias |", "|---|---|---|---|---|---|"]
        for m in AUTO:
            g = d[(d.method == m) & (d.prompt == "none")]
            if len(g):
                lines.append(row(ds, g, g.setting.iloc[0], "none", m))
        lines += ["\n**With prompts**\n", "| Method | Setting | Prompt | Dice (95% CI) | NSD | Area bias |", "|---|---|---|---|---|---|"]
        for prompt in ("points", "points_bg", "boxes"):
            for m, setting in PROMPTED:
                g = d[(d.method == m) & (d.prompt == prompt) & (d.setting == setting)]
                if len(g):
                    lines.append(row(ds, g, setting, prompt, m))
        lines += ["\n| Paired comparison | Difference in Dice (95% CI) | Images better |", "|---|---|---|"]

        def paired(a, b, prompt_a="none", prompt_b="none"):
            A = d[(d.method == a) & (d.prompt == prompt_a)].set_index(["sample", "image"])["dice"]
            B = d[(d.method == b) & (d.prompt == prompt_b)].set_index(["sample", "image"])["dice"]
            x = (A - B).dropna().reset_index().rename(columns={"dice": "diff"})
            if len(x):
                e, lo, hi = hier_boot(x, "diff")
                pa = "" if prompt_a == "none" else f" ({PROMPTS[prompt_a]})"
                pb = "" if prompt_b == "none" else f" ({PROMPTS[prompt_b]})"
                lines.append(f"| {NAMES.get(a, a)}{pa} vs {NAMES.get(b, b)}{pb} | {e:+.3f} ({lo:+.3f} to {hi:+.3f}) | {(x['diff'] > 0).mean():.0%} |")
        for m in ("microsam_full", "microsam_lora"):
            paired(m, "boneseg_ftlearned")
            paired(m, "boneseg_learned")
            paired(m, "samft_b_ende_adapter")
            paired(m, "microsam_zs")
        paired("microsam_lora", "microsam_full")
        for p in ("points_bg", "boxes"):
            paired("microsam_full", "microsam", p, p)
            paired("microsam_full", "boneseg_ftclicks", p, p)
    return rows, lines


def figure(rows):
    r = pd.DataFrame(rows)
    r = r[(r.prompt == "none")]
    FIG.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    colours = {"boneseg": "#008a87", "samft": "#c27c0e", "microsam": "#6a4fb3"}
    for ax, ds in zip(axes, DATASETS):
        g = r[r.dataset == ds].set_index("method").reindex([m for m in AUTO if m in set(r[r.dataset == ds].method)]).reset_index()
        y = np.arange(len(g))[::-1]
        cols = [next(c for k, c in colours.items() if m.startswith(k)) for m in g.method]
        ax.barh(y, g.dice, color=cols, alpha=0.9)
        ax.errorbar(g.dice, y, xerr=[g.dice - g.lo, g.hi - g.dice], fmt="none", ecolor="k", lw=0.8)
        for yi, v in zip(y, g.dice):
            ax.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=7)
        ax.set_yticks(y)
        ax.set_yticklabels([NAMES[m].replace(" (", "\n(") for m in g.method], fontsize=7)
        ax.set_title(DATASETS[ds].split(" (")[0], fontsize=10)
        ax.set_xlim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_xlabel("Dice, automatic segmentation (no prompt), 5 labelled images")
    fig.tight_layout()
    fig.savefig(FIG / "microsam_compare.png", dpi=170)


def main():
    rows, lines = summarize(load())
    pd.DataFrame(rows).to_csv(RES / "microsam_summary.csv", index=False)
    (RES / "microsam_summary.md").write_text("\n".join(lines) + "\n")
    figure(rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
