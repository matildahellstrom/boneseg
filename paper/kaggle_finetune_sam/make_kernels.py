"""Writes the two Kaggle kernels that fine-tune SAM with the code of Gu et al. (2025), github.com/mazurowski-lab/
finetune-SAM (pinned commit), on exactly the images, folds and prompts in setup.json (export_setup.py):

  kernel_a  Liu bone (5 held-out samples) and SegPC (3 draws), few-shot, 4 configurations each
  kernel_b  NOISe osteoclasts (5 held-out batches), few-shot, 4 configurations; SegPC with all 298 training images
            (ViT-B encoder+decoder Adapter); zero-shot SAM ViT-H with the paper-style point and box prompts on every
            test image of the three datasets

Configurations: the paper's few-shot recipe (ViT-B, encoder+decoder, Adapter), ViT-B decoder-only Adapter, ViT-B
encoder+decoder LoRA, and its interactive recipe (MobileSAM ViT-T, encoder+decoder, Adapter, trained with boxes).
Every kernel writes preds.zip: one binary PNG per test image, configuration and prompt, at the evaluation resolution.
Usage: python paper/kaggle_finetune_sam/make_kernels.py; then kaggle kernels push -p paper/kaggle_finetune_sam/kernel_a
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMIT = "e41f73287329ea68df3bfb153c867bdec4eb23b2"

COMMON = r'''
import io, json, os, random, shutil, subprocess, sys, time, zipfile
from argparse import Namespace
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
SETUP = json.loads(open("/tmp/setup.json").read())
DATA, OUT, REPO, W = Path("/tmp/fsam/datasets"), Path("/kaggle/working"), Path("/tmp/finetune-SAM"), Path("/tmp/weights")
for d in (DATA, W): d.mkdir(parents=True, exist_ok=True)
T0 = time.time()
def log(m): print(f"[{time.time() - T0:7.0f}s] {m}", flush=True)
def sh(cmd, cwd=None):
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if r.returncode: print(r.stdout[-3000:], r.stderr[-3000:])
    return r.returncode
sh("pip install -q monai einops icecream segment-anything stream-unzip")
if not REPO.exists():
    sh(f"git clone -q https://github.com/mazurowski-lab/finetune-SAM {REPO} && cd {REPO} && git checkout -q COMMIT")
for url, name in (("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", "sam_vit_b_01ec64.pth"),
                  ("https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt", "mobile_sam.pt")):
    if not (W / name).exists(): sh(f"curl -sL -o {W / name} {url}")
def find_input(name):
    hits = [p for p in Path("/kaggle/input").rglob(name)]
    if not hits: raise FileNotFoundError(name)
    return hits[0]
def save_pair(ds, stem, img, mask):
    (DATA / ds / "images").mkdir(parents=True, exist_ok=True); (DATA / ds / "masks").mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(DATA / ds / "images" / f"{stem}.png")
    Image.fromarray((mask > 0).astype(np.uint8) * 255).save(DATA / ds / "masks" / f"{stem}.png")
def build_liu():
    import h5py
    for s, info in SETUP["liu"].items():
        f = h5py.File(find_input(info["file"]), "r")
        a = f["DataSetInfo/Image"].attrs
        X, Y = int(b"".join(a["X"]).decode()), int(b"".join(a["Y"]).decode())
        tp = f["DataSet/ResolutionLevel 0/TimePoint 0"]
        mask_ch = len(tp.keys()) - 1
        for z in info["dev"] + info["test"]:
            plane = np.asarray(tp[f"Channel {info['channel']}"]["Data"][z, :Y, :X], np.float32)
            lo, hi = np.percentile(plane, [1, 99.5])
            img = (np.clip((plane - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
            save_pair("liu", f"{s}_{z}", img, np.asarray(tp[f"Channel {mask_ch}"]["Data"][z, :Y, :X]) > 0)
        log(f"Liu {s}: {len(info['dev']) + len(info['test'])} slices")
def build_segpc(names):
    xs = {p.stem: p for p in Path("/kaggle/input").rglob("*.bmp") if p.parent.name == "x"}
    ys = {}
    for p in Path("/kaggle/input").rglob("*.bmp"):
        if p.parent.name == "y": ys.setdefault(p.stem.split("_")[0], []).append(p)
    for n in names:
        img = Image.open(xs[n]).convert("RGB").resize((1440, 1080), Image.BILINEAR)
        m = np.zeros((1080, 1440), bool)
        for y in ys.get(n, []):
            m |= np.asarray(Image.open(y).convert("L").resize((1440, 1080), Image.NEAREST)) > 0
        save_pair("segpc", n, np.asarray(img), m)
    log(f"SegPC: {len(names)} images")
def drive_stream(file_id):
    """Streams a large Google Drive file. From cloud machines Drive often answers with an HTML page (a virus-scan
    warning or confirmation form) instead of the file; its form fields are then sent back to get the file itself."""
    import requests, re as _re
    s = requests.Session()
    url = "https://drive.usercontent.google.com/download"
    params = {"id": file_id, "export": "download", "confirm": "t"}
    for attempt in range(4):
        r = s.get(url, params=params, stream=True, timeout=120)
        if "text/html" not in r.headers.get("Content-Type", ""):
            return r
        page = r.text
        fields = dict(_re.findall(r'name="([^"]+)" value="([^"]*)"', page))
        action = _re.search(r'action="([^"]+)"', page)
        print("Drive answered with a page; resubmitting its form", sorted(fields), flush=True)
        if action: url = action.group(1).replace("&amp;", "&")
        params = {**params, **fields}
    raise RuntimeError("Google Drive did not return the file: " + page[:300])
def build_noise():
    from stream_unzip import stream_unzip
    want = {f"Mouse/{b}/{k}/{n}.{e}": (b, n, k) for b, d in SETUP["noise"].items() for sp in ("dev", "test") for n in d[sp] for k, e in (("images", "png"), ("labels", "txt"))}
    got = {}
    def chunks():
        with drive_stream("1WgIKqd346BtTpU_N5lBzIFw0zPHOXNOP") as r:
            for c in r.iter_content(1 << 20): yield c
    for name, _, data in stream_unzip(chunks()):
        name = name.decode()
        if name in want: got[want[name]] = b"".join(data)
        else:
            for _ in data: pass
    for (b, n, k), raw in got.items():
        if k != "images": continue
        img = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
        h, w = img.shape[:2]
        m = Image.new("L", (w, h), 0); dr = ImageDraw.Draw(m)
        for line in got.get((b, n, "labels"), b"").decode().splitlines():
            p = line.split()
            if len(p) >= 7 and p[0] == "0":
                xy = (np.asarray(p[1:], float).reshape(-1, 2) * [w, h]).tolist()
                dr.polygon([tuple(q) for q in xy], fill=255)
        save_pair("noise", f"{b}_{n}", img, np.asarray(m))
    log(f"NOISe: {sum(1 for k in got if k[2] == 'images')} patches")
CONFIGS = {
    "b_ende_adapter": ("vit_b", "sam_vit_b_01ec64.pth", "adapter", "noprompt", "-if_update_encoder True -if_encoder_adapter True -if_mask_decoder_adapter True"),
    "b_dec_adapter": ("vit_b", "sam_vit_b_01ec64.pth", "adapter", "noprompt", "-if_mask_decoder_adapter True"),
    "b_ende_lora": ("vit_b", "sam_vit_b_01ec64.pth", "lora", "noprompt", "-if_update_encoder True -if_encoder_lora_layer True -if_decoder_lora_layer True"),
    "t_ende_adapter_box": ("vit_t", "mobile_sam.pt", "adapter", "box", "-if_update_encoder True -if_encoder_adapter True -if_mask_decoder_adapter True"),
}
def write_list(path, ds, stems):
    path.write_text("\n".join(f"{ds}/images/{s}.png,{ds}/masks/{s}.png" for s in stems))
def train(tag, cfg, ds, train_stems, val_stems, epochs=200):
    arch, ckpt, ft, script, flags = CONFIGS[cfg]
    run = Path(f"/tmp/runs/{tag}_{cfg}"); run.parent.mkdir(exist_ok=True)
    tr, va = Path(f"/tmp/lists/{tag}_train.csv"), Path(f"/tmp/lists/{tag}_val.csv"); tr.parent.mkdir(exist_ok=True)
    write_list(tr, ds, train_stems); write_list(va, ds, val_stems)
    py = "SingleGPU_train_finetune_noprompt.py" if script == "noprompt" else "SingleGPU_train_finetune_box.py"
    t = time.time()
    code = sh(f"python {py} -if_warmup True -finetune_type {ft} -arch {arch} -img_folder {DATA} -mask_folder {DATA} -sam_ckpt {W / ckpt} "
              f"-targets combine_all -dataset_name {tag} -dir_checkpoint {run} -train_img_list {tr} -val_img_list {va} -epochs {epochs} -b 2 {flags}", cwd=REPO)
    log(f"  {tag} {cfg}: exit {code}, {time.time() - t:.0f}s")
    return run if (run / "checkpoint_best.pth").exists() else None
def load_model(run, cfg):
    import torch
    sys.path.insert(0, str(REPO))
    from models.sam import sam_model_registry
    from models.sam_LoRa import LoRA_Sam
    args = Namespace(**json.loads((run / "args.json").read_text()))
    arch, ckpt, ft, _, _ = CONFIGS[cfg]
    if ft == "lora":
        sam = sam_model_registry[arch](args, checkpoint=str(W / ckpt), num_classes=args.num_cls)
        m = LoRA_Sam(args, sam, r=4).sam
        m.load_state_dict(torch.load(run / "checkpoint_best.pth"), strict=False)
    else:
        m = sam_model_registry[arch](args, checkpoint=str(run / "checkpoint_best.pth"), num_classes=args.num_cls)
    return m.cuda().eval()
def predict(model, img_path, boxes=None):
    """Foreground mask at the image's own size; boxes (x0, y0, x1, y1) in image pixels, joined over boxes."""
    import torch, torch.nn.functional as F
    from torchvision import transforms
    im = Image.open(img_path).convert("RGB"); w, h = im.size
    x = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])(im)[None].cuda()
    def run(bx):
        with torch.no_grad():
            emb = model.image_encoder(x)
            sp, de = model.prompt_encoder(points=None, boxes=bx, masks=None)
            pred, _ = model.mask_decoder(image_embeddings=emb, image_pe=model.prompt_encoder.get_dense_pe(),
                                         sparse_prompt_embeddings=sp, dense_prompt_embeddings=de, multimask_output=True)
            prob = torch.softmax(pred, 1)[:, 1:2]
            return F.interpolate(prob, size=(h, w), mode="bilinear", align_corners=False)[0, 0].cpu().numpy() >= 0.5
    if boxes is None:
        return run(None)
    sb = torch.tensor([[x0 * 1024 / w, y0 * 1024 / h, x1 * 1024 / w, y1 * 1024 / h] for x0, y0, x1, y1 in boxes], dtype=torch.float32).cuda()
    try:
        return run(sb[None])          # All boxes at once, as in training
    except Exception:
        out = np.zeros((h, w), bool)
        for b in sb:
            out |= run(b[None])
        return out
def tag_new(ds, suffix):
    """Adds suffix to prediction folders that have none yet, so runs on the same images stay apart."""
    import re
    for p in (OUT / "preds" / ds).glob("samft_*"):
        if not re.search(r"_(d\d+|full)$", p.name):
            p.rename(p.with_name(p.name + suffix))
def save_pred(ds, method, prompt, stem, mask):
    d = OUT / "preds" / ds / f"{method}__{prompt}"; d.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(d / f"{stem}.png")
def predict_fold(run, cfg, ds, key_prefix, test_stems):
    model = load_model(run, cfg)
    for stem in test_stems:
        path = DATA / ds / "images" / f"{stem}.png"
        if cfg.endswith("_box"):
            for seed, p in enumerate(SETUP["prompts"][f"{key_prefix}/{stem}"]):
                save_pred(ds, f"samft_{cfg}", f"boxes_s{seed}", stem, predict(model, path, p["boxes"]))
        else:
            save_pred(ds, f"samft_{cfg}", "none", stem, predict(model, path))
    del model
    import torch; torch.cuda.empty_cache()
'''.replace("COMMIT", COMMIT)

KERNEL_A = r'''
build_liu()
build_segpc(SETUP["segpc"]["val"] + sorted({n for d in SETUP["segpc"]["draws"] for k in ("train", "val") for n in d[k]}))
for s, info in SETUP["liu"].items():
    stems = lambda zs: [f"{s}_{z}" for z in zs]
    for cfg in CONFIGS:
        run = train(f"liu_{s}", cfg, "liu", stems(info["dev"][:5]), stems(info["dev"][5:]))
        if run: predict_fold(run, cfg, "liu", "liu", stems(info["test"]))
for i, d in enumerate(SETUP["segpc"]["draws"]):
    for cfg in CONFIGS:
        run = train(f"segpc_d{i}", cfg, "segpc", d["train"], d["val"])
        if run:
            predict_fold(run, cfg, "segpc", "segpc", SETUP["segpc"]["val"])
            tag_new("segpc", f"_d{i}")
shutil.make_archive(str(OUT / "preds"), "zip", OUT / "preds")
log("kernel A done")
'''

KERNEL_B = r'''
build_noise()
build_segpc(SETUP["segpc"]["val"] + SETUP["segpc"]["train_all"])
build_liu()
for b, d in SETUP["noise"].items():
    stems = lambda ns: [f"{b}_{n}" for n in ns]
    for cfg in CONFIGS:
        run = train(f"noise_{b}", cfg, "noise", stems(d["dev"][:5]), stems(d["dev"][5:10]))
        if run: predict_fold(run, cfg, "noise", "noise", stems(d["test"]))
# SegPC with all training images (20 held back for early stopping), the paper's few-shot recipe configuration
tr = SETUP["segpc"]["train_all"]
run = train("segpc_full", "b_ende_adapter", "segpc", tr[20:], tr[:20], epochs=40)
if run:
    predict_fold(run, "b_ende_adapter", "segpc", "segpc", SETUP["segpc"]["val"])
    tag_new("segpc", "_full")
# Zero-shot SAM ViT-H with the paper-style prompts
import torch
from segment_anything import SamPredictor, sam_model_registry
if not (W / "sam_vit_h.pth").exists(): sh(f"curl -sL -o {W / 'sam_vit_h.pth'} https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
pred = SamPredictor(sam_model_registry["vit_h"](checkpoint=str(W / "sam_vit_h.pth")).cuda().eval())
for key, draws in SETUP["prompts"].items():
    ds, stem = key.split("/")
    img = np.asarray(Image.open(DATA / ds / "images" / f"{stem}.png").convert("RGB"))
    pred.set_image(img)
    for seed, p in enumerate(draws):
        def pts(pos, neg):
            c = np.array([(x, y) for y, x in list(pos) + list(neg)], np.float32)
            l = np.r_[np.ones(len(pos)), np.zeros(len(neg))].astype(np.int32)
            return pred.predict(point_coords=c, point_labels=l, multimask_output=False)[0][0]
        save_pred(ds, "samh", f"points_s{seed}", stem, pts(p["points"], []))
        save_pred(ds, "samh", f"points_bg_s{seed}", stem, pts(p["points"], p["background"]))
        m = np.zeros(img.shape[:2], bool)
        for bx in p["boxes"]:
            m |= pred.predict(box=np.array(bx, np.float32), multimask_output=False)[0][0]
        save_pred(ds, "samh", f"boxes_s{seed}", stem, m)
log("ViT-H zero-shot done")
shutil.make_archive(str(OUT / "preds"), "zip", OUT / "preds")
log("kernel B done")
'''


KERNEL_SMOKE = r'''
# Smoke test: one Liu fold, two configurations, 2 epochs; checks building, training, loading and predicting
info = dict(list(SETUP["liu"].items())[:1])
SETUP["liu"] = info
build_liu()
s, info = next(iter(info.items()))
stems = lambda zs: [f"{s}_{z}" for z in zs]
for cfg in ("b_ende_adapter", "t_ende_adapter_box"):
    run = train(f"smoke_{s}", cfg, "liu", stems(info["dev"][:5]), stems(info["dev"][5:]), epochs=2)
    print(cfg, "trained" if run else "FAILED")
    if run: predict_fold(run, cfg, "liu", "liu", stems(info["test"][:2]))
print(sorted(str(p.relative_to(OUT)) for p in (OUT / "preds").rglob("*.png")))
log("smoke done")
'''


def notebook(cells):
    return {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": c} for c in cells],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def main():
    setup = (HERE / "setup.json").read_text()
    write_setup = f"open('/tmp/setup.json', 'w').write(r'''{setup}''')"
    for name, body, sources, title in (("kernel_a", KERNEL_A, ["matildahellstrom/liudata", "sbilab/segpc2021dataset"], "boneseg SAM fine-tuning A (bone, plasma cells)"),
                                       ("kernel_smoke", KERNEL_SMOKE, ["matildahellstrom/liudata"], "boneseg SAM fine-tuning smoke test"),
                                       ("kernel_b", KERNEL_B, ["matildahellstrom/liudata", "sbilab/segpc2021dataset"], "boneseg SAM fine-tuning B (osteoclasts, full SegPC, ViT-H)")):
        d = HERE / name
        d.mkdir(exist_ok=True)
        (d / "notebook.ipynb").write_text(json.dumps(notebook([write_setup, COMMON, body]), indent=1))
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": f"matildahellstrom/boneseg-sam-finetune-{name.split('_')[1]}", "title": title, "code_file": "notebook.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "true",
            "machine_shape": "NvidiaTeslaT4", "dataset_sources": sources, "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2))
        print("wrote", d)


if __name__ == "__main__":
    main()
