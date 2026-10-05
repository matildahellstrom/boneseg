"""A self-contained HTML report for one slice, with the latest stack run, for lab notebooks."""
from __future__ import annotations

import datetime as dt
import html

import numpy as np
import scipy.ndimage as ndi
from PIL import Image, ImageDraw

from . import __version__, render
from .backbone import BACKBONE_LABELS


def composite(plane01: np.ndarray, mask: np.ndarray | None, reference: np.ndarray | None, points: dict | None,
              roi: list | None, max_side: int = 1200) -> bytes:
    """Gray image with the mask filled in cyan, the reference outlined in red and the clicks as dots."""
    h, w = render.display_shape(*plane01.shape, max_side)
    base = Image.fromarray((np.clip(plane01, 0, 1) * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR).convert("RGB")
    rgb = np.asarray(base).astype(np.float32)
    if mask is not None:
        m = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 127
        rgb[m] = rgb[m] * 0.55 + np.array([0, 220, 255]) * 0.45
        edge = m & ~ndi.binary_erosion(m)
        rgb[edge] = [0, 220, 255]
    if reference is not None:
        r = np.asarray(Image.fromarray(reference.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 127
        rgb[r & ~ndi.binary_erosion(r)] = [255, 60, 90]
    img = Image.fromarray(rgb.astype(np.uint8))
    d = ImageDraw.Draw(img)
    sy, sx = h / plane01.shape[0], w / plane01.shape[1]
    if roi:
        pts = [(x * sx, y * sy) for y, x in roi]
        d.line(pts + [pts[0]], fill=(230, 230, 230), width=2)
    for kind, color in (("pos", (34, 210, 122)), ("neg", (255, 90, 110))):
        for y, x in (points or {}).get(kind, []):
            d.ellipse([x * sx - 5, y * sy - 5, x * sx + 5, y * sy + 5], fill=color, outline=(11, 18, 32), width=2)
    return render.to_png_bytes(img)


def composite_labels(plane01: np.ndarray, labels: np.ndarray, colors: list[str], max_side: int = 1200) -> bytes:
    """Grey image with each structure filled and outlined in its colour."""
    h, w = render.display_shape(*plane01.shape, max_side)
    base = Image.fromarray((np.clip(plane01, 0, 1) * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR).convert("RGB")
    rgb = np.asarray(base).astype(np.float32)
    lab = np.asarray(Image.fromarray(labels.astype(np.uint8)).resize((w, h), Image.NEAREST))
    for k, hexc in enumerate(colors, start=1):
        c = hexc.lstrip("#")
        color = np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)] if len(c) == 6 else [0, 200, 240], np.float32)
        m = lab == k
        rgb[m] = rgb[m] * 0.55 + color * 0.45
        rgb[m & ~ndi.binary_erosion(m)] = color
    return render.to_png_bytes(Image.fromarray(rgb.astype(np.uint8)))


def _table(rows: list[tuple[str, str]]) -> str:
    return "<table>" + "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in rows) + "</table>"


def _fmt(v, digits=2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if isinstance(v, float):
        return f"{v:,.{digits}f}"
    return str(v)


def _chart(slices: list[dict]) -> str:
    if not slices:
        return ""
    W, H, pad = 640, 180, 28
    zs = [s["z"] for s in slices]
    z0, z1 = min(zs), max(zs)
    sx = lambda z: pad + (z - z0) / max(1, z1 - z0) * (W - 2 * pad)
    amax = max(s["area_fraction"] for s in slices) or 1
    series = [("area_fraction", "#0a8fb0", amax, "Area fraction")]
    if "dice" in slices[0]:
        series.append(("dice", "#14a05c", 1.0, "Dice against the reference"))
    paths = "".join(
        f'<path d="{"".join(("L" if i else "M") + f"{sx(s["z"]):.1f},{H - pad - s[k] / m * (H - 2 * pad):.1f}" for i, s in enumerate(slices))}" '
        f'fill="none" stroke="{c}" stroke-width="2"/>' for k, c, m, _ in series)
    axis = f'<line x1="{pad}" y1="{H - pad}" x2="{W - pad}" y2="{H - pad}" stroke="#999"/>' \
           f'<text x="{pad}" y="{H - 8}" font-size="11">z {z0}</text><text x="{W - pad}" y="{H - 8}" font-size="11" text-anchor="end">z {z1}</text>'
    legend = " · ".join(f'<span style="color:{c}">━</span> {label}' for _, c, _, label in series) + f" · area fraction peaks at {100 * amax:.1f}%"
    return f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Per-slice results">{axis}{paths}</svg><p class="small">{legend}</p>'


def methods_text(info: dict, settings: dict, source: str, n_pos: int, n_neg: int, profile: str | None, head: dict | None,
                 stack: bool = False) -> str:
    bb = BACKBONE_LABELS.get(settings["backbone"], settings["backbone"])
    if source == "structures":
        how = (f"Several structures were segmented together from {n_pos} clicks on the structures and {n_neg} shared background "
               f"clicks{' with the saved profile ' + repr(profile) if profile else ''}. Each structure was scored against the background "
               f"and the other structures' clicks, with its own threshold halfway between the scores at its clicks and the others, "
               f"and each pixel was assigned to the most similar structure whose threshold it cleared")
    elif source == "learned structures" and head:
        how = (f"A {'linear' if head['kind'] == 'linear' else 'two-layer'} classifier over background and {len(head.get('names', []))} "
               f"structures was trained on the frozen patch features of {len(head['trained_on'])} labelled slices, and each pixel took "
               f"the most probable class")
    elif source == "learned" and head:
        how = (f"A {'linear' if head['kind'] == 'linear' else 'two-layer'} classifier was trained on the frozen patch features "
               f"of {len(head['trained_on'])} manually corrected slices and thresholded at a probability of 0.5")
    else:
        clicks = f"{n_pos} object and {n_neg} background clicks" + (f" combined with the saved profile '{profile}'" if profile else "")
        pos_ = float(settings.get("threshold_position", 0.5))
        pos_ = 0.5 if pos_ == 0.5 else pos_
        thr = {"clicks": "placed halfway between the scores at the object and background clicks" if pos_ == 0.5 else
                         (f"placed {pos_:.0%} of the way from the scores at the background clicks to those at the object clicks" if pos_ >= 0 else
                          "placed 70% (with five or fewer clicks of a kind) or 90% (with more) of the way from the scores at the background "
                          "clicks to those at the object clicks"),
               "otsu": "chosen with Otsu's method", "top_percent": f"set to keep the top {settings['top_percent']}% of pixels",
               "manual": f"set manually to {settings['manual_threshold']}"}.get(settings["threshold_mode"], settings["threshold_mode"])
        how = (f"Prototypes were taken from the patches under {clicks}. Each patch was scored by its mean cosine similarity to the "
               f"object prototypes minus {settings['neg_weight']} times its mean similarity to the background prototypes, and the "
               f"score map was upsampled bilinearly and thresholded, with the threshold {thr}")
        if int(settings.get("shift_passes", 1)) > 1:
            n = int(settings["shift_passes"])
            how = (f"Features were extracted {n * n} times with the image shifted by fractions of a patch, and interleaved into a "
                   f"{n} times finer feature grid. ") + how
        if settings.get("edge_refine") == "guided":
            how += (f"; before thresholding, the score map was edge-aligned with a guided filter (He et al., 2013) using the image as "
                    f"guide (radius one feature cell, eps {settings.get('guided_eps', 0.01):g})")
        if settings.get("refiner"):
            how += ("; the final mask came from a small learned refiner, a U-Net that takes the image and the score relative to "
                    "the threshold and predicts the mask at full resolution")
    clean = []
    if settings.get("min_object_um2"):
        clean.append(f"objects smaller than {settings['min_object_um2']} µm² were removed")
    if settings.get("fill_holes_um2"):
        clean.append(f"holes smaller than {settings['fill_holes_um2']} µm² were filled")
    if settings.get("smooth_px"):
        clean.append(f"edges were smoothed with a {settings['smooth_px']} px opening and closing")
    vox = info["voxel_um"]
    return (f"Images were segmented with boneseg {__version__}. Intensities were clipped at the {settings['clip_low']:g} and "
            f"{settings['clip_high']:g} percentiles and scaled to [0, 1]. Features came from the frozen, self-supervised {bb} backbone "
            f"(block {settings['layer_from_end']} from the end, longest image side resized to {settings['vit_size']} px with the aspect "
            f"ratio kept). {how}." + (f" After thresholding, {', '.join(clean)}." if clean else "") +
            f" Measurements used a pixel size of {vox[2]:.3f} × {vox[1]:.3f} µm" + (f" and a slice spacing of {vox[0]:.2f} µm." if info["n_z"] > 1 else ".")
            + (" For the z-stack, the prototypes and threshold from the annotated slice were applied to every slice, with scores "
               "standardized per slice by their median and median absolute deviation, and objects were counted in 3D with "
               "6-connectivity." if stack and source != "learned" and settings.get("score_norm") == "robust" else ""))


def build_report(info: dict, c: int, z: int, image_png: bytes, stats: dict | None, evaluation: dict | None, settings: dict,
                 threshold: dict | None, histo: dict | None, stack: dict | None, methods: str, structures: list | None = None) -> str:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    ch = info["channel_names"][c] if c < len(info["channel_names"]) else f"Channel {c}"
    parts = [f"<h1>{html.escape(info['name'])}</h1>",
             f"<p class='small'>boneseg {__version__} report · {now} · channel {c} ({html.escape(ch)}) · slice {z}</p>",
             f"<img src='{render.data_url(image_png)}' alt='Slice {z} with the segmentation mask'>",
             "<p class='small'>Cyan: segmentation. Red outline: reference mask. Green and red dots: object and background clicks. White line: region of interest.</p>"]
    file_rows = [("Image size", f"{info['width']} × {info['height']} px, {info['n_z']} slices, {info['n_channels']} channels"),
                 ("Pixel size", f"{info['voxel_um'][2]:.3f} µm" + ("" if info["voxel_size_known"] else " (unknown, measurements are in pixels)")),
                 ("Slice spacing", f"{info['voxel_um'][0]:.2f} µm")]
    if info.get("notes"):
        file_rows.append(("Notes", info["notes"]))
    parts += ["<h2>Image</h2>", _table(file_rows)]
    if stats:
        rows = [("Area", f"{_fmt(stats['area_um2'], 1)} µm²"), ("Area fraction", f"{100 * stats['area_fraction']:.2f}%"),
                ("Objects", str(stats["n_objects"])), ("Objects per mm²", _fmt(stats["objects_per_mm2"], 1)),
                ("Median object area", f"{_fmt(stats['median_object_area_um2'], 1)} µm²"),
                ("Signal contrast", _fmt(stats.get("contrast_ratio")))]
        if threshold:
            rows.append(("Threshold", f"{threshold['value']:.3f} on the rescaled score ({threshold['source']})"))
        parts += ["<h2>Segmentation</h2>", _table(rows)]
    if structures:
        head = "<tr><th>Structure</th><td><b>Area fraction</b></td><td><b>Area</b></td><td><b>Objects</b></td><td><b>Objects per mm²</b></td></tr>"
        body = "".join(f"<tr><th><span style='color:{html.escape(st.get('color', ''))}'>●</span> {html.escape(st['name'])}</th>"
                       f"<td>{100 * st['area_fraction']:.2f}%</td><td>{_fmt(st['area_um2'], 1)} µm²</td><td>{st['n_objects']}</td>"
                       f"<td>{_fmt(st['objects_per_mm2'], 1)}</td></tr>" for st in structures)
        parts += ["<h2>Structures</h2>", f"<table>{head}{body}</table>"]
    if evaluation:
        parts += [f"<h2>Against {html.escape(evaluation.get('against', 'the reference mask'))}</h2>",
                  _table([("Dice", f"{evaluation['dice']:.3f}"), ("IoU", f"{evaluation['iou']:.3f}"), ("HD95", f"{_fmt(evaluation['hd95_um'], 1)} µm")])]
    if histo:
        labels = {"T.Ar_mm2": ("Tissue area, T.Ar", "mm²", 4), "B.Ar_mm2": ("Bone area, B.Ar", "mm²", 4), "B.Ar/T.Ar_%": ("Bone area fraction, B.Ar/T.Ar", "%", 1),
                  "B.Pm_mm": ("Bone perimeter, B.Pm", "mm", 3), "Oc.Pm/B.Pm_%": ("Osteoclast surface, Oc.Pm/B.Pm", "%", 1),
                  "Oc.Pm_mm": ("Osteoclast perimeter, Oc.Pm", "mm", 3), "N.Oc": ("Osteoclasts on bone, N.Oc", "", 0),
                  "N.Oc/B.Pm_per_mm": ("Osteoclast number, N.Oc/B.Pm", "/mm", 1), "N.Oc/T.Ar_per_mm2": ("Osteoclast density, N.Oc/T.Ar", "/mm²", 1),
                  "cells_total": ("Cells in total", "", 0), "cells_on_bone_%": ("Cells on bone", "%", 1),
                  "median_distance_to_bone_um": ("Median distance to bone", "µm", 1), "contact_um": ("Contact distance", "µm", 1)}
        rows = []
        for k, v in histo.items():
            label, unit, digits = labels.get(k, (k, "", 2))
            rows.append((label, f"{_fmt(float(v), digits) if digits else int(v)} {unit}".strip()))
        parts += ["<h2>Bone histomorphometry</h2>", _table(rows)]
    if stack and (stack.get("summary") or {}).get("structures"):
        s = stack["summary"]
        rows = [(name, f"volume {_fmt(st.get('volume_um3'), 0)} µm³ · mean area {100 * (st.get('mean_area_fraction') or 0):.2f}% · "
                       f"{st.get('n_objects_3d', '–')} objects in 3D") for name, st in s["structures"].items()]
        hm = s.get("histomorphometry")
        if hm:
            rows.append(("Histomorphometry", f"B.Ar/T.Ar {_fmt(hm.get('B.Ar/T.Ar_%'))}% · Oc.Pm/B.Pm {_fmt(hm.get('Oc.Pm/B.Pm_%'))}% · "
                                             f"N.Oc/B.Pm {_fmt(hm.get('N.Oc/B.Pm_per_mm'))} /mm"))
        parts += [f"<h2>Latest stack run ({s.get('n_slices')} slices)</h2>", _table(rows)]
    elif stack:
        s = stack["summary"]
        rows = [("Slices", str(s.get("n_slices"))), ("Volume", f"{_fmt(s.get('volume_um3'), 0)} µm³"),
                ("Mean area fraction", f"{100 * s.get('mean_area_fraction', 0):.2f}%"),
                ("Objects in 3D", f"{s.get('n_objects_3d', '–')} ({s.get('n_objects_3d_inside', '–')} not cut by the stack ends)"),
                ("Median object volume", f"{_fmt(s.get('median_object_volume_um3'), 0)} µm³")]
        if s.get("mean_dice_vs_reference") is not None:
            rows.append(("Mean Dice against the reference", f"{s['mean_dice_vs_reference']:.3f}"))
        parts += ["<h2>Latest stack run</h2>", _table(rows), _chart(stack.get("slices", []))]
    parts += ["<h2>Settings</h2>", _table([(k, str(v)) for k, v in settings.items()]),
              "<h2>Methods</h2>", f"<p>{html.escape(methods)}</p>",
              "<p class='small'>Generated automatically. Check the numbers and adapt the text before using it in a publication.</p>",
              "<h2>References</h2><ul class='small'>"
              "<li>Oquab M, et al. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research (2024).</li>"
              "<li>Dempster DW, et al. Standardized nomenclature, symbols, and units for bone histomorphometry: a 2012 update of the report of the "
              "ASBMR Histomorphometry Nomenclature Committee. J Bone Miner Res 28:2–17 (2013).</li></ul>"]
    css = """body{font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:860px;margin:32px auto;padding:0 16px;color:#16202e;background:#fff}
h1{font-size:22px;margin-bottom:0}h2{font-size:15px;margin-top:28px;border-bottom:1px solid #dde3ec;padding-bottom:4px}
img,svg{max-width:100%;border-radius:6px}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:5px 8px;border-bottom:1px solid #eef1f5;vertical-align:top}
th{width:38%;font-weight:500;color:#4a5568}.small{font-size:12px;color:#5a6577}
@media (prefers-color-scheme:dark){body{background:#0b1220;color:#e6edf7}h2{border-color:#22314f}th{color:#8fa0bd}th,td{border-color:#16223a}.small{color:#8fa0bd}}"""
    return f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>boneseg report: {html.escape(info['name'])}</title><style>{css}</style></head><body>{''.join(parts)}</body></html>"
