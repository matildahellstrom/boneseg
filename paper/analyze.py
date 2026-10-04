"""Tables, statistics and figures from paper/results/slices.csv.

  * Dice per method and condition, per sample and overall, with a hierarchical bootstrap 95% CI
    (resample samples, then test slices within each sample). Click seeds are averaged first.
  * Paired differences against boneseg (dino_clicks) on the same slices and clicks.
  * Agreement of bone measures (B.Ar/T.Ar, B.Pm/T.Ar, Tb.Th) between each method and the expert masks:
    bias, 95% limits of agreement (Bland-Altman), ICC(A,1) and Pearson r.
Writes paper/results/summary.md and figures to paper/figures/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
LABELS = {"otsu": "Otsu threshold", "rf_clicks": "Random forest, clicks (ilastik-style)", "sam": "SAM ViT-B, point prompts",
          "microsam": "micro-SAM ViT-B LM, point prompts", "dino_clicks": "boneseg, clicks",
          "rf_labels": "Random forest, 5 labelled slices", "dino_labels": "boneseg learned model, 5 labelled slices",
          "nnunet": "nnU-Net 2D, labelled slices"}
MEASURES = {"B.Ar/T.Ar_%": "B.Ar/T.Ar (%)", "B.Pm/T.Ar_per_mm": "B.Pm/T.Ar (1/mm)", "Tb.Th_um": "Tb.Th (µm)"}


def seed_mean(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["sample", "z", "method", "budget", "noisy"]
    return df.groupby(keys, as_index=False).mean(numeric_only=True)


def hier_boot(df: pd.DataFrame, col: str, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of per-sample means, with a CI from resampling samples and then slices within them."""
    rng = np.random.default_rng(seed)
    groups = [g[col].dropna().to_numpy() for _, g in df.groupby("sample")]
    groups = [g for g in groups if len(g)]
    est = float(np.mean([g.mean() for g in groups]))
    boots = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        boots.append(np.mean([rng.choice(groups[i], len(groups[i])).mean() for i in pick]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return est, float(lo), float(hi)


def icc_a1(x: np.ndarray, y: np.ndarray) -> float:
    """Two-way, absolute agreement, single measures ICC (McGraw and Wong ICC(A,1)) for two raters."""
    data = np.c_[x, y]
    data = data[np.isfinite(data).all(1)]
    n, k = data.shape
    if n < 3:
        return float("nan")
    gm = data.mean()
    ssr = k * ((data.mean(1) - gm) ** 2).sum()
    ssc = n * ((data.mean(0) - gm) ** 2).sum()
    sse = ((data - gm) ** 2).sum() - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k / n * (msc - mse)))


def condition_label(row) -> str:
    if row["method"] == "otsu":
        return "no input"
    if row["method"] in ("rf_labels", "dino_labels", "nnunet"):
        return f"labels from {row['budget']}"
    return f"{row['budget']} clicks, {'noisy' if row['noisy'] else 'clean'}"


def main():
    FIG.mkdir(exist_ok=True)
    raw = pd.read_csv(RES / "slices.csv")
    run = json.loads((RES / "run.json").read_text()) if (RES / "run.json").exists() else {}
    df = seed_mean(raw)
    df["condition"] = df.apply(condition_label, axis=1)
    samples = sorted(df["sample"].unique())
    lines = [f"# Evaluation results\n", f"Samples: {', '.join(samples)}. Nested leave-one-sample-out; settings tuned on the other samples only. "
             f"Method version: tag `{run.get('method_tag', 'method-v1')}`. Click seeds: {run.get('seeds')}. Run time {run.get('minutes')} min.\n"]
    if len(samples) < 5:
        lines.append(f"**Caveat:** with {len(samples)} samples the confidence intervals mostly reflect variation between these samples; "
                     "they are not a substitute for more samples.\n")

    # 1. Dice table ---------------------------------------------------------------------------------
    lines.append("## Dice per method\n")
    lines.append("| Method | Condition | " + " | ".join(samples) + " | Mean (95% CI) |")
    lines.append("|---|---|" + "---|" * len(samples) + "---|")
    order = ["otsu", "rf_clicks", "sam", "microsam", "dino_clicks", "rf_labels", "dino_labels", "nnunet"]
    dice_rows = []
    for cond in sorted(df["condition"].unique(), key=lambda c: (c != "no input", "labels" in c, c)):
        for m in order:
            sub = df[(df["method"] == m) & (df["condition"] == cond)]
            if sub.empty:
                continue
            per = sub.groupby("sample")["dice"].mean()
            est, lo, hi = hier_boot(sub, "dice")
            dice_rows.append({"method": m, "condition": cond, "mean": est, "lo": lo, "hi": hi, **per.to_dict()})
            lines.append(f"| {LABELS[m]} | {cond} | " + " | ".join(f"{per.get(s, np.nan):.3f}" for s in samples) + f" | {est:.3f} ({lo:.3f}–{hi:.3f}) |")
    pd.DataFrame(dice_rows).to_csv(RES / "dice_summary.csv", index=False)

    # 2. Paired differences against boneseg -------------------------------------------------------------
    lines.append("\n## Paired differences in Dice against boneseg with the same clicks\n")
    lines.append("Positive means boneseg is better. Wins: share of test slices where boneseg scored higher.\n")
    lines.append("| Compared with | Condition | Difference (95% CI) | Wins |")
    lines.append("|---|---|---|---|")
    ref = df[df["method"] == "dino_clicks"].set_index(["sample", "z", "condition"])["dice"]
    for m in ("rf_clicks", "sam", "microsam"):
        for cond in sorted(df.loc[df["method"] == m, "condition"].unique()):
            other = df[(df["method"] == m) & (df["condition"] == cond)].set_index(["sample", "z", "condition"])["dice"]
            diff = (ref - other).dropna().reset_index().rename(columns={"dice": "diff"})
            if diff.empty:
                continue
            est, lo, hi = hier_boot(diff, "diff")
            lines.append(f"| {LABELS[m]} | {cond} | {est:+.3f} ({lo:+.3f} to {hi:+.3f}) | {(diff['diff'] > 0).mean():.0%} |")
    lab = df[df["method"] == "dino_labels"].set_index(["sample", "z", "condition"])["dice"]
    for cond in sorted(df.loc[df["method"] == "rf_labels", "condition"].unique()):
        other = df[(df["method"] == "rf_labels") & (df["condition"] == cond)].set_index(["sample", "z", "condition"])["dice"]
        diff = (lab - other).dropna().reset_index().rename(columns={"dice": "diff"})
        est, lo, hi = hier_boot(diff, "diff")
        lines.append(f"| {LABELS['rf_labels']} (vs boneseg learned model) | {cond} | {est:+.3f} ({lo:+.3f} to {hi:+.3f}) | {(diff['diff'] > 0).mean():.0%} |")

    # 3. Agreement of bone measures ---------------------------------------------------------------------
    lines.append("\n## Agreement of bone measures with the expert masks\n")
    lines.append("Per test slice, pooled over samples. Bias = method minus expert; LoA = 95% limits of agreement.\n")
    lines.append("| Method | Condition | Measure | Expert mean | Bias | LoA | ICC(A,1) | r |")
    lines.append("|---|---|---|---|---|---|---|---|")
    picks = [("otsu", "no input"), ("rf_clicks", "25+25 clicks, clean"), ("sam", "25+25 clicks, clean"), ("microsam", "25+25 clicks, clean"),
             ("dino_clicks", "25+25 clicks, clean"), ("rf_labels", "labels from same sample"), ("dino_labels", "labels from same sample"),
             ("dino_labels", "labels from other samples"), ("nnunet", "labels from other samples")]
    agree = []
    for m, cond in picks:
        sub = df[(df["method"] == m) & (df["condition"] == cond)]
        if sub.empty:
            continue
        for key, label in MEASURES.items():
            x, y = sub[f"pred_{key}"].to_numpy(), sub[f"gt_{key}"].to_numpy()
            ok = np.isfinite(x) & np.isfinite(y)
            d = x[ok] - y[ok]
            bias, sd = d.mean(), d.std(ddof=1)
            r = np.corrcoef(x[ok], y[ok])[0, 1] if ok.sum() > 2 else np.nan
            icc = icc_a1(x, y)
            agree.append({"method": m, "condition": cond, "measure": key, "expert_mean": y[ok].mean(), "bias": bias, "loa_low": bias - 1.96 * sd,
                          "loa_high": bias + 1.96 * sd, "icc_a1": icc, "r": r, "n": int(ok.sum())})
            lines.append(f"| {LABELS[m]} | {cond} | {label} | {y[ok].mean():.3g} | {bias:+.3g} | {bias - 1.96 * sd:+.3g} to {bias + 1.96 * sd:+.3g} | {icc:.2f} | {r:.2f} |")
    pd.DataFrame(agree).to_csv(RES / "agreement.csv", index=False)

    # Figures --------------------------------------------------------------------------------------------
    clean25 = [("otsu", "no input"), ("rf_clicks", "25+25 clicks, clean"), ("sam", "25+25 clicks, clean"), ("microsam", "25+25 clicks, clean"),
               ("dino_clicks", "25+25 clicks, clean"), ("rf_labels", "labels from same sample"), ("dino_labels", "labels from same sample"),
               ("dino_labels", "labels from other samples")]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    width = 0.8 / len(samples)
    for i, s in enumerate(samples):
        vals = [df[(df["method"] == m) & (df["condition"] == c) & (df["sample"] == s)]["dice"].mean() for m, c in clean25]
        ax.bar(np.arange(len(clean25)) + i * width - 0.4 + width / 2, vals, width, label=f"Sample {s}")
    ax.set_xticks(range(len(clean25)))
    ax.set_xticklabels([f"{LABELS[m]}\n({c})".replace(", clicks (ilastik-style)", " clicks").replace(", point prompts", "") for m, c in clean25],
                       rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Dice against the expert mask")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "dice_by_method.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
    for ax, (m, c) in zip(axes, [("otsu", "no input"), ("sam", "25+25 clicks, clean"), ("dino_clicks", "25+25 clicks, clean"), ("dino_labels", "labels from same sample")]):
        sub = df[(df["method"] == m) & (df["condition"] == c)]
        x, y = sub["pred_B.Ar/T.Ar_%"].to_numpy(), sub["gt_B.Ar/T.Ar_%"].to_numpy()
        mean, diff = (x + y) / 2, x - y
        for s in samples:
            sel = (sub["sample"] == s).to_numpy()
            ax.scatter(mean[sel], diff[sel], s=14, label=f"Sample {s}")
        b, sd = np.nanmean(diff), np.nanstd(diff, ddof=1)
        for v, ls in ((b, "-"), (b - 1.96 * sd, "--"), (b + 1.96 * sd, "--")):
            ax.axhline(v, color="k", lw=0.8, ls=ls)
        ax.set_title(f"{LABELS[m]}\n{c}", fontsize=8)
        ax.set_xlabel("Mean of method and expert B.Ar/T.Ar (%)", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Method minus expert (%)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "bland_altman_bone_area.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for m in ("rf_clicks", "sam", "microsam", "dino_clicks"):
        vals = [df[(df["method"] == m) & (df["condition"] == c)]["dice"].mean() for c in
                ("3+6 clicks, clean", "3+6 clicks, noisy", "25+25 clicks, clean", "25+25 clicks, noisy")]
        ax.plot(range(4), vals, marker="o", label=LABELS[m])
    ax.set_xticks(range(4))
    ax.set_xticklabels(["3+6 clean", "3+6 noisy", "25+25 clean", "25+25 noisy"])
    ax.set_ylabel("Mean Dice")
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "clicks_and_noise.png", dpi=180)
    plt.close(fig)

    lines.append("\n## Figures\n\n![Dice by method](../figures/dice_by_method.png)\n\n![Bland-Altman, bone area](../figures/bland_altman_bone_area.png)\n\n"
                 "![Clicks and noise](../figures/clicks_and_noise.png)\n")
    (RES / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()
