"""The NOISe mouse osteoclast data (Wan et al., 2024, github.com/michaelwwan/noise), sampled for boneseg's evaluation.

The data is one 18 GB zip on Google Drive with 832 x 832 brightfield patches of TRAP-stained mouse osteoclast
cultures in five batches (m1 to m5), cut with 50% overlap from 10 to 36 well images per batch. Labels are YOLO
polygons; class 0 outlines osteoclasts (TRAP-positive, three or more nuclei), class 1 boxes single small cells and
is not used here. Instead of downloading the zip, `fetch` reads only the chosen patches from it with HTTP range
requests.

Selection, per batch: label files are drawn at random, patches with at least one osteoclast are kept, and they are
split by well into development (even well index) and test (odd) patches, so that overlapping neighbours never end
up on both sides. Usage: python paper/noise_common.py fetch [--per-batch 40]
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from common import ROOT  # noqa: F401  (puts the repository on the path)

DATA = ROOT / "data" / "noise"
URL = "https://drive.usercontent.google.com/download?id=1WgIKqd346BtTpU_N5lBzIFw0zPHOXNOP&export=download&confirm=t"
BATCHES = ["m1", "m2", "m3", "m4", "m5"]
SIZE = 832


@dataclass
class Patch:
    batch: str
    name: str
    well: int
    rgb: np.ndarray        # [832, 832, 3] uint8
    instances: np.ndarray  # [832, 832] int32, 0 background, k for the k-th osteoclast

    @property
    def gt(self) -> np.ndarray:
        return self.instances > 0

    @property
    def n_cells(self) -> int:
        return int(self.instances.max())


def osteoclast_polygons(label_text: str) -> list[np.ndarray]:
    polys = []
    for line in label_text.splitlines():
        parts = line.split()
        if len(parts) >= 7 and parts[0] == "0":
            polys.append(np.asarray(parts[1:], float).reshape(-1, 2))   # Normalized (x, y)
    return polys


def rasterize(polys: list[np.ndarray], h: int = SIZE, w: int = SIZE) -> np.ndarray:
    """Instance map from normalized outlines. Larger outlines are drawn first, so a smaller overlapping one stays
    visible. Patches at a well's edge can be smaller than 832 px."""
    polys = [q * np.array([w, h]) for q in polys]
    im = Image.new("I", (w, h), 0)
    dr = ImageDraw.Draw(im)
    area = lambda p: 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], 1)) - np.dot(p[:, 1], np.roll(p[:, 0], 1)))  # noqa: E731
    for k, p in enumerate(sorted(polys, key=area, reverse=True), start=1):
        dr.polygon([tuple(q) for q in p.tolist()], fill=k)
    return np.asarray(im, np.int32)


def fetch(per_batch: int = 40, candidates_per_batch: int = 80, seed: int = 0):
    """Streams the zip once, keeping the osteoclast outlines of every patch and the images of a random candidate set
    chosen in advance from the zip index (half from even wells, half from odd ones). Then keeps, per batch, the first
    per_batch / 2 candidates of each split that contain an osteoclast. Writes data/noise/<batch>/<name>.png and .txt,
    osteoclast_labels.json (all patches) and selection.json. About 18 GB pass through; little is written to disk."""
    import requests
    from stream_unzip import stream_unzip
    idx = json.loads((DATA / "zip_index.json").read_text())
    rng = random.Random(seed)
    cands = {}
    for b in BATCHES:
        names = sorted(Path(f).stem for f, _, _ in idx if f.startswith(f"Mouse/{b}/labels/") and f.endswith(".txt"))
        even = [n for n in names if int(n.split("_")[1]) % 2 == 0]
        odd = [n for n in names if int(n.split("_")[1]) % 2 == 1]
        cands[b] = {"dev": rng.sample(even, candidates_per_batch // 2), "test": rng.sample(odd, candidates_per_batch // 2)}
    want = {f"Mouse/{b}/images/{n}.png": (b, n) for b in BATCHES for sp in ("dev", "test") for n in cands[b][sp]}
    labels, images = {}, {}
    total = sum(c for f, _, c in idx)
    seen = [0]

    def chunks():
        with requests.get(URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            for c in r.iter_content(1 << 20):
                seen[0] += len(c)
                if seen[0] // (1 << 30) != (seen[0] - len(c)) // (1 << 30):
                    print(f"  {seen[0] / 1e9:.0f} of {total / 1e9:.0f} GB", flush=True)
                yield c

    for name, _, data in stream_unzip(chunks()):
        name = name.decode()
        if name.startswith("Mouse/") and "/labels/" in name and name.endswith(".txt"):
            text = b"".join(data).decode()
            b, stem = name.split("/")[1], Path(name).stem
            labels[f"{b}/{stem}"] = "\n".join(l for l in text.splitlines() if l.startswith("0 "))
        elif name in want:
            b, n = want[name]
            images[(b, n)] = b"".join(data)
        else:
            for _ in data:   # Every entry must be drained to move on
                pass
    (DATA / "osteoclast_labels.json").write_text(json.dumps(labels))
    selection = {}
    for b in BATCHES:
        out = DATA / b
        out.mkdir(parents=True, exist_ok=True)
        picked = {}
        for sp in ("dev", "test"):
            keep = [n for n in cands[b][sp] if labels.get(f"{b}/{n}") and (b, n) in images][:per_batch // 2]
            for n in keep:
                (out / f"{n}.png").write_bytes(images[(b, n)])
                (out / f"{n}.txt").write_text(labels[f"{b}/{n}"])
            picked[sp] = keep
        selection[b] = picked
        n_all = sum(1 for k in labels if k.startswith(b + "/"))
        n_oc = sum(1 for k, v in labels.items() if k.startswith(b + "/") and v)
        print(f"{b}: {len(picked['dev'])} development and {len(picked['test'])} test patches; {n_oc} of {n_all} patches hold an osteoclast", flush=True)
    (DATA / "selection.json").write_text(json.dumps(selection, indent=1))


def load_batch(batch: str) -> dict[str, list[Patch]]:
    sel = json.loads((DATA / "selection.json").read_text())[batch]
    out = {}
    for split, names in sel.items():
        out[split] = []
        for n in names:
            rgb = np.asarray(Image.open(DATA / batch / f"{n}.png").convert("RGB"))
            inst = rasterize(osteoclast_polygons((DATA / batch / f"{n}.txt").read_text()), *rgb.shape[:2])
            out[split].append(Patch(batch, n, int(n.split("_")[1]), rgb, inst))
    return out


# Single-channel versions of the brightfield image for boneseg, which works on one channel --------------------------
def to_channel(rgb: np.ndarray, kind: str) -> np.ndarray:
    """[0, 1] image where stain is bright. 'green': inverted green channel (purple TRAP stain absorbs green);
    'od': mean optical density; 'hema': the haematoxylin-like channel of a colour deconvolution."""
    x = rgb.astype(np.float32) / 255.0
    if kind == "green":
        ch = 1 - x[..., 1]
    elif kind == "od":
        ch = -np.log(np.clip(x, 1 / 255, 1)).mean(-1)
    elif kind == "hema":
        from skimage.color import rgb2hed
        ch = rgb2hed(rgb)[..., 0]
    elif kind == "gray":
        ch = 1 - x.mean(-1)
    else:
        raise ValueError(kind)
    lo, hi = np.percentile(ch, [1, 99.5])
    return np.clip((ch - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def stain_mask(img: np.ndarray, q: float = 0.85) -> np.ndarray:
    """Stained pixels (any cell), used to put some background clicks on cells that are not osteoclasts."""
    return img > np.quantile(img, q)


def osteoclast_clicks(p: Patch, img: np.ndarray, n_pos: int, n_neg: int, seed: int):
    """Simulated clicks like a careful user's: object clicks inside osteoclasts, at least 5 px from their edge and
    spread over the cells; half of the background clicks on stained cells that are not osteoclasts (the hard cases
    a user would correct), the other half anywhere else, at least 10 px from any osteoclast."""
    import scipy.ndimage as ndi
    rng = np.random.default_rng(seed)
    inner = ndi.binary_erosion(p.gt, iterations=5)
    inner = inner if inner.any() else p.gt
    far = ~ndi.binary_dilation(p.gt, iterations=10)
    stained = far & stain_mask(img)

    def pick(region, k):
        ys, xs = np.nonzero(region)
        if not len(ys) or k == 0:
            return []
        i = rng.choice(len(ys), k, replace=len(ys) < k)
        return [(int(ys[j]), int(xs[j])) for j in i]

    # Spread object clicks over the osteoclasts: every cell gets at least one while clicks last
    pos = []
    cells = [c for c in range(1, p.n_cells + 1) if (inner & (p.instances == c)).any()]
    rng.shuffle(cells)
    for c in cells[:n_pos]:
        pos += pick(inner & (p.instances == c), 1)
    pos += pick(inner, n_pos - len(pos))
    n_hard = n_neg // 2 if stained.any() else 0
    neg = pick(stained, n_hard) + pick(far & ~stained if (far & ~stained).any() else far, n_neg - n_hard)
    return pos, neg


def instances_from_mask(mask: np.ndarray, min_px: int = 0) -> np.ndarray:
    import scipy.ndimage as ndi
    lab, n = ndi.label(ndi.binary_fill_holes(mask))
    if min_px and n:
        sizes = np.bincount(lab.ravel())
        keep = sizes >= min_px
        keep[0] = False
        lab, n = ndi.label(keep[lab])
    return lab


def match_instances(pred: np.ndarray, gt: np.ndarray, iou_thr: float = 0.5) -> dict:
    """One-to-one matching of predicted and expert cells at IoU >= iou_thr (greedy by IoU, as in detection metrics)."""
    n_p, n_g = int(pred.max()), int(gt.max())
    if n_p == 0 or n_g == 0:
        return {"tp": 0, "fp": n_p, "fn": n_g}
    # Overlap counts between every predicted and expert label via a joint histogram
    joint = np.bincount(pred.ravel().astype(np.int64) * (n_g + 1) + gt.ravel(), minlength=(n_p + 1) * (n_g + 1)).reshape(n_p + 1, n_g + 1)
    inter = joint[1:, 1:].astype(float)
    a_p, a_g = joint[1:, :].sum(1, keepdims=True), joint[:, 1:].sum(0, keepdims=True)
    iou = inter / (a_p + a_g - inter + 1e-9)
    pairs = sorted(((iou[i, j], i, j) for i, j in zip(*np.nonzero(iou >= iou_thr))), reverse=True)
    used_p, used_g = set(), set()
    for _, i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
    tp = len(used_p)
    return {"tp": tp, "fp": n_p - tp, "fn": n_g - tp}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch"])
    ap.add_argument("--per-batch", type=int, default=40)
    a = ap.parse_args()
    fetch(a.per_batch)
