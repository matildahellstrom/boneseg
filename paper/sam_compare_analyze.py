"""boneseg against SAM under the protocol of Gu et al. (2025): merges results/sam_zeroshot.csv, boneseg_fewshot.csv
and (once the Kaggle kernels have run) finetune_sam.csv into results/sam_compare_summary.md and a figure.

Seeds (prompt draws) and few-shot draws are averaged per image. Intervals: hierarchical bootstrap over groups (Liu
sample, NOISe batch; SegPC is one group) and images. Paired differences use the same images and prompts.
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
NAMES = {"boneseg": "boneseg", "sam": "SAM ViT-B", "samh": "SAM ViT-H", "microsam": "micro-SAM ViT-B LM", "mobilesam": "MobileSAM ViT-T",
         "boneseg_learned": "boneseg learned model", "boneseg_ftlearned": "boneseg, fine-tuned DINOv2 + learned model", "boneseg_ftlayer": "boneseg, fine-tuned DINOv2 (own output)",
         "boneseg_ftclicks": "boneseg, fine-tuned DINOv2 + prompts", "samft_b_ende_adapter": "SAM ViT-B, enc+dec Adapter (paper's few-shot recipe)",
         "samft_b_dec_adapter": "SAM ViT-B, decoder Adapter", "samft_b_ende_lora": "SAM ViT-B, enc+dec LoRA",
         "samft_t_ende_adapter_box": "MobileSAM ViT-T, enc+dec Adapter, boxes (paper's interactive recipe)"}
PROMPTS = {"points": "object points only (paper)", "points_bg": "object + background points", "boxes": "noisy boxes (paper)", "none": "no prompt"}


def load():
    frames = []
    for f in ("sam_zeroshot.csv", "boneseg_fewshot.csv", "finetune_sam.csv"):
        if (RES / f).exists():
            d = pd.read_csv(RES / f, dtype={"image": str})
            if "draw" not in d:
                d["draw"] = "0"
            frames.append(d.assign(source=f.split(".")[0]))
    df = pd.concat(frames, ignore_index=True)
    df["draw"] = df["draw"].astype(str)
    zero = (df["source"] == "sam_zeroshot") | (df["method"] == "samh")   # ViT-H ran zero-shot in Kaggle kernel B
    df["setting"] = np.where(df["draw"] == "full", "full data", np.where(zero, "zero-shot", "5-shot"))
    df["sample"] = np.where(df["dataset"] == "plasma", "segpc", df["image"].str.split("_").str[0])
    return df.groupby(["dataset", "setting", "method", "prompt", "sample", "image"], as_index=False)[["dice", "nsd", "area_bias"]].mean()


def main():
    df = load()
    lines = ["# boneseg against SAM, following Gu et al. (2025)\n",
             "Paper-style prompts (paper/prompts.py) on the test images of the three datasets; 5-shot = 5 labelled images of the target "
             "(the held-out Liu sample or NOISe batch, or SegPC training images, three draws). Dice, NSD (bone 5 µm, cells 2 px) and the area "
             "bias (percentage points of the image), mean over images with 95% intervals.\n"]
    rows = []
    for ds in ("bone", "osteoclasts", "plasma"):
        lines += [f"\n## {ds.capitalize()}\n", "| Setting | Prompt | Method | Dice (95% CI) | NSD | Area bias |", "|---|---|---|---|---|---|"]
        for (setting, prompt, method), g in df[df["dataset"] == ds].groupby(["setting", "prompt", "method"]):
            e, lo, hi = hier_boot(g, "dice")
            rows.append({"dataset": ds, "setting": setting, "prompt": prompt, "method": method, "dice": e, "lo": lo, "hi": hi,
                         "nsd": g["nsd"].mean(), "area_bias": g["area_bias"].mean(), "n": len(g)})
            lines.append(f"| {setting} | {PROMPTS.get(prompt, prompt)} | {NAMES.get(method, method)} | {e:.3f} ({lo:.3f}–{hi:.3f}) | {g['nsd'].mean():.3f} | {g['area_bias'].mean():+.2f} |")
        lines += ["\n| Paired comparison | Difference in Dice (95% CI) | Images better |", "|---|---|---|"]
        d = df[df["dataset"] == ds]

        def paired(a, b, label):
            A = d[(d.method == a[0]) & (d.prompt == a[1]) & (d.setting == a[2])].set_index(["sample", "image"])["dice"]
            B = d[(d.method == b[0]) & (d.prompt == b[1]) & (d.setting == b[2])].set_index(["sample", "image"])["dice"]
            x = (A - B).dropna().reset_index().rename(columns={"dice": "diff"})
            if len(x):
                e, lo, hi = hier_boot(x, "diff")
                lines.append(f"| {label} | {e:+.3f} ({lo:+.3f} to {hi:+.3f}) | {(x['diff'] > 0).mean():.0%} |")
        for sam in ("sam", "samh", "microsam", "mobilesam"):
            paired(("boneseg", "points_bg", "zero-shot"), (sam, "points_bg", "zero-shot"), f"boneseg vs {NAMES[sam]}, points + background, zero-shot")
            paired(("boneseg", "points_bg", "zero-shot"), (sam, "points", "zero-shot"), f"boneseg (points + background) vs {NAMES[sam]} (paper's points only)")
            paired(("boneseg", "boxes", "zero-shot"), (sam, "boxes", "zero-shot"), f"boneseg vs {NAMES[sam]}, boxes, zero-shot")
        for samft in ("samft_b_ende_adapter", "samft_b_dec_adapter", "samft_b_ende_lora"):
            paired(("boneseg_learned", "none", "5-shot"), (samft, "none", "5-shot"), f"boneseg learned model vs {NAMES[samft]}, 5-shot, no prompt")
            paired(("boneseg_ftlayer", "none", "5-shot"), (samft, "none", "5-shot"), f"boneseg fine-tuned DINOv2 vs {NAMES[samft]}, 5-shot, no prompt")
            paired(("boneseg_ftlearned", "none", "5-shot"), (samft, "none", "5-shot"), f"boneseg fine-tuned DINOv2 + learned model vs {NAMES[samft]}, 5-shot, no prompt")
        paired(("boneseg_ftclicks", "boxes", "5-shot"), ("samft_t_ende_adapter_box", "boxes", "5-shot"), "boneseg fine-tuned + boxes vs MobileSAM fine-tuned + boxes, 5-shot")
    pd.DataFrame(rows).to_csv(RES / "sam_compare_summary.csv", index=False)
    (RES / "sam_compare_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    # Figure: Dice per method, prompt and setting, one panel per dataset
    r = pd.DataFrame(rows)
    FIG.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharex=True)
    for ax, ds in zip(axes, ("bone", "osteoclasts", "plasma")):
        g = r[r.dataset == ds].sort_values(["setting", "prompt", "dice"])
        labels = [f"{NAMES.get(m, m)[:38]} · {p} · {s}" for m, p, s in zip(g.method, g.prompt, g.setting)]
        cols = ["#008a87" if m.startswith("boneseg") else "#c27c0e" for m in g.method]
        y = np.arange(len(g))
        ax.barh(y, g.dice, color=cols, alpha=0.85)
        ax.errorbar(g.dice, y, xerr=[g.dice - g.lo, g.hi - g.dice], fmt="none", ecolor="k", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_title(ds.capitalize(), fontsize=10)
        ax.set_xlim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_xlabel("Dice")
    fig.tight_layout()
    fig.savefig(FIG / "sam_compare.png", dpi=160)


if __name__ == "__main__":
    main()
