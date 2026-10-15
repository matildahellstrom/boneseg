"""micro-SAM fine-tuned on this dataset's labels: trained on Kaggle, used in the app.

Training micro-SAM needs a GPU: on a laptop one step takes 30-60 s, on Kaggle's free T4 about a second. So the app
writes a ready-to-push Kaggle package (the labelled slices as a private dataset plus a notebook), the user pushes it
with the kaggle command-line tool, and loads the resulting model file back. Prediction then runs locally: the
decoder's foreground probability, thresholded at 0.5 (boneseg/microsam_core.py does both, here and on Kaggle).
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
import threading
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import microsam_core as MC

MICROSAM_VERSION = "1.8.14"   # Pinned on Kaggle, so the trained file matches the code that loads it here
_lock = threading.Lock()
_cache: dict = {}             # (path, mtime) -> loaded model; one at a time, a ViT-B takes about 1 GB


def available() -> bool:
    try:
        import micro_sam  # noqa: F401
        return True
    except ImportError:
        return False


def require():
    if not available():
        raise ValueError("micro-SAM is not installed: pip install micro-sam (or pip install -e .[microsam])")


# Kaggle package ------------------------------------------------------------------------------------------
def _slug(text: str, n: int = 30) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:n].strip("-") or "dataset"


def _notebook(cells: list[str]) -> dict:
    return {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": c} for c in cells],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


TRAIN_CELL = r'''
# Fine-tunes micro-SAM on the labelled slices exported by boneseg, then saves the weights for the app
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, "/tmp"); import microsam_core as MC
root = next(Path("/kaggle/input").rglob("boneseg_split.json")).parent
split = json.loads((root / "boneseg_split.json").read_text())
def read(n):
    return (np.asarray(Image.open(root / "images" / f"{n}.png")),
            np.asarray(Image.open(root / "instances" / f"{n}.png")).astype(np.int64))
tr, va = [read(n) for n in split["train"]], [read(n) for n in split["val"]]
print(f"{len(tr)} slices to learn from, {len(va)} to stop early; recipe {split['recipe']}")
ck = MC.train([i for i, _ in tr], [l for _, l in tr], [i for i, _ in va], [l for _, l in va], "/tmp/runs",
              name="boneseg", recipe=split["recipe"], epochs=split["epochs"], device="cuda")
assert ck, "Training failed; see the error above"
model = MC.Model(ck, split["recipe"], device="cuda")
dice = []
for img, lab in va:
    m, g = model.mask(img), lab > 0
    dice.append(2 * (m & g).sum() / max(1, m.sum() + g.sum()))
out = MC.export(ck, f"/kaggle/working/{split['output']}", split["recipe"], val_dice=float(np.mean(dice)),
                n_train=len(tr), n_val=len(va), source=split["source"], channel=split["channel"], clip=split["clip"])
print(f"Validation Dice {np.mean(dice):.3f}. Load {out.name} in boneseg: section 5, micro-SAM, Load model file")
'''


def kaggle_package(items, *, username: str, name: str, channel: int, clip: tuple[float, float], recipe: str = "full",
                   epochs: int = 40) -> Path:
    """Writes a zip with a Kaggle dataset (the labelled slices) and a notebook that trains on it.
    items: (z, image in [0, 1], binary label, valid region or None). Every second slice validates, as for DINOv2."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,48}", username or ""):
        raise ValueError("Enter your Kaggle user name (lower-case letters, digits and hyphens)")
    if recipe not in MC.RECIPES:
        raise ValueError(f"Unknown recipe {recipe}; use one of {', '.join(MC.RECIPES)}")
    if len(items) < 2:
        raise ValueError("micro-SAM needs at least 2 labels on this channel (5 or more work best): one in two checks the training")
    if not 5 <= epochs <= 200:
        raise ValueError("Epochs must be 5 to 200")
    slug = _slug(name)
    tag = f"boneseg-microsam-{slug}"
    out_name = f"microsam_{slug}_c{channel}_{recipe}.pt"
    tmp = Path(tempfile.mkdtemp(prefix="boneseg_kaggle_"))
    top = tmp / tag
    data, kern = top / "dataset", top / "notebook"
    for d in (data / "images", data / "instances", kern):
        d.mkdir(parents=True)
    names = []
    for z, img, gt, region in items:
        if region is not None and region.any():
            # Pixels outside a label's region are not labelled: train on the region's bounding box only
            ys, xs = np.nonzero(region)
            sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
            img, gt = img[sl], gt[sl] & region[sl]
        n = f"z{z:05d}"
        Image.fromarray(MC.to_uint8(img)).save(data / "images" / f"{n}.png")
        Image.fromarray(MC.instances(gt).astype(np.uint16)).save(data / "instances" / f"{n}.png")
        names.append(n)
    split = {"train": names[0::2], "val": names[1::2], "recipe": recipe, "epochs": epochs, "output": out_name,
             "source": name, "channel": channel, "clip": list(clip)}
    (data / "boneseg_split.json").write_text(json.dumps(split, indent=1))
    (data / "dataset-metadata.json").write_text(json.dumps(
        {"title": f"boneseg micro-SAM labels {slug}"[:50], "id": f"{username}/{tag}", "licenses": [{"name": "unknown"}]}, indent=2))
    core = (Path(__file__).parent / "microsam_core.py").read_text()
    cells = [f"import subprocess\nsubprocess.run('pip install -q micro-sam=={MICROSAM_VERSION}', shell=True, check=True)",
             f"open('/tmp/microsam_core.py', 'w').write({core!r})", TRAIN_CELL]
    (kern / "notebook.ipynb").write_text(json.dumps(_notebook(cells), indent=1))
    title = f"boneseg micro-SAM {slug}"[:50]
    (kern / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{username}/{_slug(title, 50)}", "title": title, "code_file": "notebook.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": [f"{username}/{tag}"], "competition_sources": [],
        "kernel_sources": [], "model_sources": []}, indent=2))
    (top / "README.txt").write_text(f"""Fine-tune micro-SAM on Kaggle ({len(split['train'])} slices to learn from, {len(split['val'])} to stop early, recipe {recipe})

Needs the kaggle command-line tool (pip install kaggle) and an API token (kaggle.com > Settings > API).
1. Upload the labelled slices as a private dataset:
     kaggle datasets create -p {tag}/dataset
2. Wait until the dataset shows as ready on kaggle.com (a minute or two), then start the notebook (free T4 GPU):
     kaggle kernels push -p {tag}/notebook
3. Check on it; training takes about 10 to 30 minutes:
     kaggle kernels status {username}/{_slug(title, 50)}
4. Download the model file:
     kaggle kernels output {username}/{_slug(title, 50)} -p {tag}/output
5. In boneseg, section 5, micro-SAM: Load model file, and choose {tag}/output/{out_name}
""")
    zpath = tmp / f"{tag}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(top.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(tmp))
    shutil.rmtree(top)
    return zpath


# Model files ---------------------------------------------------------------------------------------------
def read_info(path: Path) -> dict:
    """Checks a model file without running code from it (weights only) and returns its description."""
    try:
        st = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        raise ValueError(f"Not a micro-SAM model file from the boneseg notebook ({type(e).__name__})") from None
    if not isinstance(st, dict) or "model_state" not in st or "decoder_state" not in st:
        raise ValueError("Not a micro-SAM model file: it needs the weights of SAM and of the decoder")
    info = dict(st.get("boneseg") or {})
    info["recipe"] = MC.recipe_of(st)
    return info


def install(src: Path, dest: Path) -> dict:
    """Validates a downloaded model file and moves it into place, with its description beside it."""
    info = read_info(src)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), dest)
    dest.with_suffix(".json").write_text(json.dumps(info, indent=1))
    return info


def model_info(path: Path) -> dict | None:
    if not path.exists():
        return None
    meta = path.with_suffix(".json")
    info = json.loads(meta.read_text()) if meta.exists() else {}
    return {**info, "file": path.name, "size_mb": round(path.stat().st_size / 2**20)}


def _device():
    from .backbone import pick_device
    return pick_device()


def get_model(path: Path):
    """The loaded model for a file, kept while it is used."""
    require()
    key = (str(path), path.stat().st_mtime)
    with _lock:
        if key not in _cache:
            _cache.clear()
            info = model_info(path) or {}
            _cache[key] = MC.Model(path, info.get("recipe", "full"), device=_device())
        return _cache[key]


def foreground(path: Path, img: np.ndarray) -> np.ndarray:
    """Foreground probability (H, W) for an image in [0, 1]."""
    model = get_model(path)
    with _lock:
        return model.foreground(img)


def forget(path: Path):
    with _lock:
        for k in [k for k in _cache if k[0] == str(path)]:
            del _cache[k]


def clip_for(path: Path, settings) -> tuple[float, float]:
    """The intensity clipping the model was trained with (it saw the slices as the app showed them then)."""
    clip = (model_info(path) or {}).get("clip")
    return (float(clip[0]), float(clip[1])) if clip else (settings.clip_low, settings.clip_high)


def segment(path: Path, img: np.ndarray, settings, pixel_um):
    """micro-SAM's automatic mask for one slice: foreground probability >= 0.5, then boneseg's usual clean-up."""
    from .segment import SegmentationResult, postprocess
    fg = foreground(path, img)
    return SegmentationResult(heat=fg, mask=postprocess(fg >= 0.5, settings, pixel_um), threshold=0.5, raw_threshold=0.5,
                              threshold_source="micro-SAM decoder")
