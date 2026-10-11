"""Scores and shows the whole-stack masks of three methods trained on the same 5 labelled slices per sample:
  learned      boneseg's learned model on frozen DINOv2 (stack_compare.py)
  ft_learned   boneseg's learned model on DINOv2 fine-tuned on the 5 slices (stack_compare.py)
  sam_ft       SAM ViT-B fine-tuned on the 5 slices (Kaggle notebook kernel_stack; its stack_masks/ folder or zip)

Outputs
- paper/results/stack_compare_summary.md: per sample and method, 3D Dice and the plate-model bone measures against
  the experts' 3D masks (computed as in stack3d_evaluate.py), and the mean per-slice Dice.
- data/stack_compare/viewer.html (git-ignored, it shows Liu images): a page with a slice slider; for every slice the
  image with the expert outline and each method's mask coloured by agreement with the expert.
Usage: python paper/stack_compare_view.py [--sam path/to/stack_masks(.zip)] [--samples A C] [--no-images]
Open the viewer with:  open data/stack_compare/viewer.html
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from common import available_samples, load_sample
from boneseg import io as bio
from stack3d_evaluate import XY, ZF, measures, reduce2d
from stack_compare import OUT, stack_range

HERE = Path(__file__).resolve().parent
METHODS = {"learned": "boneseg learned model", "ft_learned": "boneseg, fine-tuned DINOv2 + learned model", "sam_ft": "SAM ViT-B fine-tuned"}
PANEL = 420   # Width of one panel in the viewer, in pixels
COLOURS = {"both": (40, 210, 205), "extra": (255, 170, 30), "missed": (235, 60, 140)}


def load_npz(src) -> dict:
    """{z: boolean mask} from a bit-packed mask file (the format both scripts write)."""
    d = np.load(src)
    w = int(d["shape"][1])
    return {int(z): np.unpackbits(m, axis=-1)[:, :w].astype(bool) for z, m in zip(d["z"], d["masks"])}


def sam_masks(path, name):
    """The SAM notebook's masks for a sample, from its stack_masks folder or zip; None if not there."""
    if path is None:
        return None
    p = Path(path)
    if p.is_dir() and (p / f"{name}.npz").exists():
        return load_npz(p / f"{name}.npz")
    if p.suffix == ".zip" and p.exists():
        z = zipfile.ZipFile(p)
        hit = [n for n in z.namelist() if n.endswith(f"{name}.npz")]
        return load_npz(io.BytesIO(z.read(hit[0]))) if hit else None
    return None


def overlay(img, pred, gt):
    """The slice in grey with the method's mask: teal where it agrees with the expert, orange extra, pink missed;
    the expert's outline in white."""
    rgb = np.repeat((np.clip(img, 0, 1) * 0.85 * 255)[..., None], 3, -1)
    if pred is not None:
        for m, col in ((pred & gt, COLOURS["both"]), (pred & ~gt, COLOURS["extra"]), (~pred & gt, COLOURS["missed"])):
            rgb[m] = 0.45 * rgb[m] + 0.55 * np.array(col)
    rgb[gt ^ ndi.binary_erosion(gt, iterations=2)] = 255
    return rgb.astype(np.uint8)


def dice(a, b):
    return float(2 * (a & b).sum() / max(1, a.sum() + b.sum()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam", default=None, help="The SAM notebook's stack_masks folder or stack_masks.zip")
    ap.add_argument("--samples", nargs="*", default=None)
    ap.add_argument("--no-images", action="store_true", help="Only the scores, no viewer images")
    args = ap.parse_args(argv)
    view_dir = OUT / "viewer"
    summary = {}
    lines = ["# Whole stacks: boneseg against fine-tuned SAM, from the same 5 labelled slices\n",
             "Every slice of the central 80% of each stack (sample D: slices 224-287), against the experts' 3D masks; measures on "
             f"volumes reduced to {XY} x {XY} pixels in-plane and {ZF} slices in z, as in stack3d_summary.md. Errors are relative to the expert.\n",
             "| Sample | Method | 3D Dice | Mean slice Dice | BV/TV | BS/BV | Tb.Th | Tb.N | Tb.Sp |", "|---|---|---|---|---|---|---|---|---|"]
    rel = {m: [] for m in METHODS}
    for name in args.samples or available_samples():
        masks = {m: (load_npz(OUT / f"{name}_{m}.npz") if (OUT / f"{name}_{m}.npz").exists() else None) for m in ("learned", "ft_learned")}
        masks["sam_ft"] = sam_masks(args.sam, name)
        if all(v is None for v in masks.values()):
            continue
        s = load_sample(name)
        vol = s.vol
        if vol is None:
            from remote_d import remote_volume
            vol = remote_volume()
        ref = vol.n_channels - 1
        zs = stack_range(name, vol)
        red = {k: [] for k in ["expert", *METHODS]}
        per_slice = {m: [] for m in METHODS}
        (view_dir / name).mkdir(parents=True, exist_ok=True)
        for z in zs:
            gt = vol.get_plane(ref, z) > 0
            red["expert"].append(reduce2d(gt))
            for m in METHODS:
                pm = masks[m].get(z) if masks[m] is not None else None
                if pm is not None:
                    red[m].append(reduce2d(pm))
                    per_slice[m].append(dice(pm, gt))
                else:
                    per_slice[m].append(None)
            if not args.no_images:
                # One image per slice: expert outline on the slice, then one panel per method
                img = bio.normalize_plane(vol.get_plane(s.channel, z))
                k = max(1, int(np.ceil(img.shape[1] / PANEL)))
                small = lambda a: a[::k, ::k]  # noqa: E731
                panels = [overlay(small(img), None, small(gt))] + [
                    overlay(small(img), small(masks[m][z]) if masks[m] is not None and z in masks[m] else None, small(gt)) for m in METHODS]
                strip = Image.fromarray(np.concatenate([np.pad(p, ((0, 0), (0, 6), (0, 0)), constant_values=255) for p in panels], 1))
                if any(masks[m] is None for m in METHODS):
                    dr = ImageDraw.Draw(strip)
                    for i, m in enumerate(METHODS, start=1):
                        if masks[m] is None:
                            dr.text((i * (panels[0].shape[1] + 6) + 10, 10), "not run yet", fill=(255, 255, 255))
                strip.save(view_dir / name / f"z{z}.jpg", quality=82)
        summary[name] = {"z": zs, "dice": per_slice, "panels": len(METHODS) + 1}
        spacing = (vol.voxel_um[0] * ZF, vol.voxel_um[1] * XY, vol.voxel_um[2] * XY)
        to3d = lambda lst: (lambda a: a[: (a.shape[0] // ZF) * ZF].reshape(-1, ZF, *a.shape[1:]).astype(np.float32).mean(1) >= 0.5)(np.stack(lst))  # noqa: E731
        e = to3d(red["expert"])
        em = measures(e, spacing)
        lines.append(f"| {name} ({len(zs)} slices) | expert | – | – | {em['BV/TV_%']:.3g}% | {em['BS/BV_per_mm']:.3g} | {em['Tb.Th_um']:.3g} µm | {em['Tb.N_per_mm']:.3g} | {em['Tb.Sp_um']:.3g} µm |")
        for m, label in METHODS.items():
            if len(red[m]) != len(zs):
                continue
            v = to3d(red[m])
            mm = measures(v, spacing)
            d3 = float(2 * (v & e).sum() / max(1, v.sum() + e.sum()))
            err = {k: 100 * (mm[k] - em[k]) / em[k] for k in ("BV/TV_%", "BS/BV_per_mm", "Tb.Th_um", "Tb.N_per_mm", "Tb.Sp_um")}
            rel[m].append({"dice_3d": d3, "slice": float(np.mean(per_slice[m])), **err})
            lines.append(f"| {name} | {label} | {d3:.3f} | {np.mean(per_slice[m]):.3f} | " + " | ".join(f"{err[k]:+.0f}%" for k in err) + " |")
        print(f"{name}: done", flush=True)
    lines += ["\n## Mean over samples\n", "| Method | Samples | 3D Dice | Mean slice Dice | BV/TV | BS/BV | Tb.Th | Tb.N | Tb.Sp |", "|---|---|---|---|---|---|---|---|---|"]
    for m, label in METHODS.items():
        if rel[m]:
            r = rel[m]
            lines.append(f"| {label} | {len(r)} | {np.mean([x['dice_3d'] for x in r]):.3f} | {np.mean([x['slice'] for x in r]):.3f} | "
                         + " | ".join(f"{np.mean([x[k] for x in r]):+.0f}% (|{np.mean([abs(x[k]) for x in r]):.0f}%|)" for k in ("BV/TV_%", "BS/BV_per_mm", "Tb.Th_um", "Tb.N_per_mm", "Tb.Sp_um")) + " |")
    (HERE / "results" / "stack_compare_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if not args.no_images:
        write_viewer(view_dir.parent / "viewer.html", summary)


def write_viewer(path, summary):
    """A self-contained page: sample buttons, slice slider (arrow keys too), the slice's strip and its Dice per method."""
    data = json.dumps(summary)
    labels = json.dumps(["Expert outline"] + list(METHODS.values()))
    path.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Whole-stack comparison</title>
<style>
 body {{ margin: 0; font: 14px system-ui, sans-serif; background: #111; color: #eee; }}
 header {{ padding: 12px 16px; display: flex; flex-wrap: wrap; gap: 12px; align-items: center; border-bottom: 1px solid #333; }}
 button {{ background: #222; color: #eee; border: 1px solid #444; border-radius: 6px; padding: 6px 12px; cursor: pointer; }}
 button[aria-pressed=true] {{ background: #1fa8a4; border-color: #1fa8a4; color: #031; }}
 main {{ padding: 12px 16px; }}
 .labels {{ display: grid; gap: 6px; margin-bottom: 6px; }}
 .labels div {{ font-size: 13px; color: #bbb; }}
 .labels b {{ color: #fff; font-variant-numeric: tabular-nums; }}
 img {{ width: 100%; height: auto; display: block; border-radius: 4px; }}
 input[type=range] {{ flex: 1; min-width: 200px; accent-color: #1fa8a4; }}
 .key span {{ display: inline-block; width: 11px; height: 11px; border-radius: 2px; margin: 0 4px 0 12px; vertical-align: -1px; }}
</style></head><body>
<header><b>Whole-stack comparison</b><span id="samples"></span>
 <input type="range" id="z" min="0" value="0" aria-label="Slice"><span id="zt"></span>
 <span class="key"><span style="background:#28d2cd"></span>agrees with expert<span style="background:#ffaa1e"></span>extra<span style="background:#eb3c8c"></span>missed<span style="background:#fff"></span>expert outline</span></header>
<main><div class="labels" id="labels"></div><img id="strip" alt="The slice with each method's mask"></main>
<script>
const D = {data}, L = {labels};
let cur = Object.keys(D)[0];
const sEl = document.getElementById("samples"), z = document.getElementById("z");
for (const s of Object.keys(D)) {{ const b = document.createElement("button"); b.textContent = "Sample " + s; b.onclick = () => {{ cur = s; z.value = Math.floor(D[s].z.length / 2); show(); }}; sEl.appendChild(b); }}
function show() {{
  const d = D[cur], i = +z.value, zz = d.z[i];
  z.max = d.z.length - 1;
  [...sEl.children].forEach((b) => b.setAttribute("aria-pressed", b.textContent === "Sample " + cur));
  document.getElementById("zt").textContent = `slice ${{zz}} (${{i + 1}} of ${{d.z.length}})`;
  document.getElementById("strip").src = `viewer/${{cur}}/z${{zz}}.jpg`;
  const lab = document.getElementById("labels");
  lab.style.gridTemplateColumns = `repeat(${{d.panels}}, 1fr)`;
  const keys = Object.keys(d.dice);
  lab.innerHTML = L.map((t, k) => `<div>${{t}}${{k ? (d.dice[keys[k - 1]][i] == null ? " · <i>not run yet</i>" : ` · Dice <b>${{d.dice[keys[k - 1]][i].toFixed(3)}}</b>`) : ""}}</div>`).join("");
}}
z.oninput = show;
window.addEventListener("keydown", (e) => {{ if (e.key === "ArrowRight") {{ z.value = +z.value + 1; show(); }} if (e.key === "ArrowLeft") {{ z.value = +z.value - 1; show(); }} }});
z.value = Math.floor(D[cur].z.length / 2); show();
</script></body></html>""")
    print("viewer:", path)


if __name__ == "__main__":
    main()
