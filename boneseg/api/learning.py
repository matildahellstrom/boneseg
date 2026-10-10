"""Saved clicks, corrected labels and the learned model."""
from __future__ import annotations

import io
import time

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from .. import render
from ..head import Head, head_to_profile_dict, train_head, z_context_slices
from ..segment import Profile, SegmentationSettings
from .context import AppContext
from .models import AnnotationRequest, FinetuneRequest, HeadRequest, LabelRequest, ProfileFromHeadRequest, ReferenceLabelRequest


def router(ctx: AppContext) -> APIRouter:
    r = APIRouter()
    store = ctx.store

    def check_cz(ds_id: str, c: int, z: int):
        """Clicks and labels are only stored for channels and slices the image has."""
        vol = store.get(ds_id).volume
        if not 0 <= c < vol.n_channels:
            raise ValueError(f"Channel {c} is out of range, the image has {vol.n_channels}")
        if not 0 <= z < vol.n_z:
            raise ValueError(f"Slice {z} is out of range, the image has {vol.n_z}")

    @r.get("/api/datasets/{ds_id}/annotations")
    def get_annotations(ds_id: str):
        return store.get_annotations(ds_id)

    @r.post("/api/datasets/{ds_id}/annotations")  # For navigator.sendBeacon, which can only POST, when the page closes
    @r.put("/api/datasets/{ds_id}/annotations")
    def put_annotation(ds_id: str, req: AnnotationRequest):
        check_cz(ds_id, req.channel, req.z)
        store.set_annotation(ds_id, req.channel, req.z, req.pos, req.neg, req.extra)
        return {"ok": True}

    @r.get("/api/datasets/{ds_id}/labels")
    def list_labels(ds_id: str):
        return store.list_labels(ds_id)

    @r.post("/api/datasets/{ds_id}/labels")
    def save_label(ds_id: str, req: LabelRequest):
        check_cz(ds_id, req.channel, req.z)
        multi = len(req.structures) > 1
        if req.mask_png:
            import base64

            from PIL import Image

            try:
                raw = base64.b64decode(req.mask_png.split(",", 1)[-1])
                img = np.asarray(Image.open(io.BytesIO(raw)).convert("RGBA"))
            except Exception:
                raise ValueError("The corrected mask could not be read as a PNG image") from None
            painted = img[..., 3] > 127  # Painted pixels are opaque
            if multi:
                labels = np.where(painted, np.clip(img[..., 0], 0, len(req.structures)), 0).astype(np.uint8)
                store.save_label_map(ds_id, req.channel, req.z, labels, req.structures)
            else:
                store.save_label(ds_id, req.channel, req.z, painted)
        elif multi:
            res = store.get(ds_id).results.get(("multi", req.channel, req.z))
            if res is None:
                raise HTTPException(404, "Segment several structures on this slice first")
            store.save_label_map(ds_id, req.channel, req.z, res["labels"], res["names"])
        else:
            store.save_label(ds_id, req.channel, req.z, ctx.last_result(ds_id, req.channel, req.z).mask)
        # With a region of interest, the label only speaks for the inside: training ignores the rest
        store.set_label_region(ds_id, req.channel, req.z, store.get(ds_id).meta.get("roi"))
        return {"ok": True, "labels": store.list_labels(ds_id)}

    @r.post("/api/datasets/{ds_id}/labels/from-reference")
    def labels_from_reference(ds_id: str, req: ReferenceLabelRequest):
        """Saves the expert reference mask of n evenly spaced slices as labels, for training a model that can then
        segment files without expert masks."""
        ds = store.get(ds_id)
        ref = ds.meta.get("reference_channel")
        if ref is None:
            raise ValueError("Set a reference mask channel first")
        if req.channel == ref:
            raise ValueError("Pick the image channel, not the reference channel itself")
        # Evenly spaced over the central 80% of the stack, where slices are usually best; a slice whose
        # reference is empty is replaced by the nearest non-empty one around it. Reads only the slices it needs.
        n_z = ds.volume.n_z
        lo, hi = int(0.1 * (n_z - 1)), int(round(0.9 * (n_z - 1)))
        targets = sorted({int(round(v)) for v in np.linspace(lo, hi, max(1, min(req.n, hi - lo + 1)))})
        saved = []
        for t in targets:
            for z in sorted(range(n_z), key=lambda zz: abs(zz - t))[:max(3, n_z // 20)]:
                if z in saved:
                    continue
                m = store.raw_plane(ds_id, int(ref), z) > 0
                if m.any():
                    store.save_label(ds_id, req.channel, z, m)
                    store.set_label_region(ds_id, req.channel, z, None)   # Expert masks cover the whole slice
                    saved.append(z)
                    break
        if not saved:
            raise ValueError("The reference channel is empty on the slices tried")
        return {"saved": saved, "labels": store.list_labels(ds_id)}

    @r.get("/api/datasets/{ds_id}/labels/png")
    def label_png(ds_id: str, c: int = 0, z: int = 0, max_side: int = 1600):
        m = store.load_label_map(ds_id, c, z)
        if m is None:
            raise HTTPException(404, "No label for this slice")
        return Response(render.mask_png(m > 0, max_side, color=(255, 210, 0), fill_alpha=0), media_type="image/png")

    @r.delete("/api/datasets/{ds_id}/labels")
    def delete_label(ds_id: str, c: int = 0, z: int = 0):
        store.delete_label(ds_id, c, z)
        return {"ok": True, "labels": store.list_labels(ds_id)}

    @r.post("/api/datasets/{ds_id}/head")
    def train_head_endpoint(ds_id: str, req: HeadRequest):
        """Trains the learned model on this channel's labels. With background=true it runs as a job with progress
        (the app uses that: on a real stack, computing the features takes minutes); otherwise it answers when done."""
        settings = SegmentationSettings.from_dict(req.settings)
        labels = [l for l in store.list_labels(ds_id) if l["channel"] == req.channel]
        if not labels:
            raise ValueError("Save at least one corrected mask as a label on this channel first")

        def train(progress, cancelled=lambda: False):
            t0 = time.time()
            ds = store.get(ds_id)
            vol = ds.volume
            names = store.label_structures(ds_id, req.channel)
            multi = bool(names and len(names) > 1)
            d = 0 if multi else z_context_slices(vol.voxel_um[0], vol.n_z)   # Stacks: neighbours about 4 um away
            probe = Head(settings.backbone, settings.layer_from_end, settings.vit_size, "linear", {}, 0, z_context=d)
            samples = []
            for i, l in enumerate(labels):
                if cancelled():
                    raise ValueError("Training was cancelled")
                progress(0.8 * i / len(labels), f"Computing features for labelled slice {l['z']} ({i + 1} of {len(labels)})")
                with store.compute_lock:   # Held per slice, so the viewer can still get a slice in between
                    if multi:
                        samples.append((store.embedding(ds_id, req.channel, l["z"], settings), store.load_label_map(ds_id, req.channel, l["z"])))
                    else:
                        samples.append((ctx.head_embedding(ds_id, req.channel, l["z"], settings, probe), store.load_label(ds_id, req.channel, l["z"]),
                                        store.label_region(ds_id, req.channel, l["z"])))
            progress(0.8, "Choosing the model variant and threshold on held-out slices")
            if multi:
                # Labels with several structures train a model that predicts all of them
                from ..head import train_head_multi

                head = train_head_multi(samples, names, settings, [(req.channel, l["z"]) for l in labels], pixel_um=vol.pixel_um)
            else:
                head = train_head(samples, settings, [(req.channel, l["z"]) for l in labels], kind=req.kind, pixel_um=vol.pixel_um)
                head.z_context = d
            head.save(store.head_path(ds_id, req.channel))
            out = {**head.info(), "seconds": round(time.time() - t0, 2)}
            if ds.meta.get("reference_channel") is not None and not head.names:
                # An independent check: Dice against the expert reference on up to five slices that were not labelled.
                # Cross-validation only measures agreement with the labels, which may themselves be imperfect.
                from ..head import segment_with_head
                from ..metrics import dice

                labelled = {l["z"] for l in labels}
                pool = [z for z in range(vol.n_z) if z not in labelled]
                picks = sorted({pool[int(i)] for i in np.linspace(0, len(pool) - 1, min(5, len(pool)))}) if pool else []
                scores = []
                for k, z in enumerate(picks):
                    progress(0.85 + 0.15 * k / max(1, len(picks)), f"Checking against the reference on slice {z}")
                    ref = store.reference_mask(ds_id, z, req.channel)
                    if ref is None or not ref.any():
                        continue
                    with store.compute_lock:
                        res = segment_with_head(head, ctx.head_embedding(ds_id, req.channel, z, settings, head), settings, vol.pixel_um)
                    scores.append(dice(res.mask, ref))
                if scores:
                    out["reference_check"] = {"mean_dice": float(np.mean(scores)), "n_slices": len(scores)}
            out["seconds"] = round(time.time() - t0, 2)
            return out

        if req.background:
            def work(job):
                return train(lambda p, m: (setattr(job, "progress", p), setattr(job, "message", m)), job.cancel.is_set)
            return store.start_job("train", work, meta={"dataset_id": ds_id, "channel": req.channel, "n_labels": len(labels)}).info()
        return train(lambda p, m: None)

    @r.post("/api/datasets/{ds_id}/head/export")
    def export_head(ds_id: str, req: ProfileFromHeadRequest):
        path = store.head_path(ds_id, req.channel)
        if not path.exists():
            raise ValueError("Train a model on this channel first")
        head = Head.load(path)
        ds = store.get(ds_id)
        s = SegmentationSettings(backbone=head.backbone, layer_from_end=head.layer_from_end, vit_size=head.vit_size)
        prof = Profile(name=req.name.strip() or "Learned model", backbone=head.backbone, layer_from_end=head.layer_from_end,
                       pos=np.zeros((0, head.dim), np.float32), neg=np.zeros((0, head.dim), np.float32), settings=s.to_dict(),
                       description=req.description, source=f"{ds.volume.name}, channel {req.channel}, {len(head.trained_on)} labelled slices",
                       head=head_to_profile_dict(head))
        pid = store.save_profile(prof)
        return {"id": pid, **[p for p in store.list_profiles() if p["id"] == pid][0]}

    @r.post("/api/datasets/{ds_id}/finetune")
    def finetune_endpoint(ds_id: str, req: FinetuneRequest):
        """Fine-tunes DINOv2 Small on this channel's labels in a background job. Every second label validates. The
        result is saved in the models folder and appears as a backbone. For comparison, the learned model on the
        frozen features is trained on the same labels and scored on the same validation slices."""
        import re
        from datetime import datetime

        ds = store.get(ds_id)
        names = store.label_structures(ds_id, req.channel)
        if names and len(names) > 1:
            raise ValueError("Fine-tuning works with labels of one structure; these labels hold several")
        zs = [l["z"] for l in store.list_labels(ds_id) if l["channel"] == req.channel]
        if len(zs) < 2:
            raise ValueError("Fine-tuning needs at least 2 labels on this channel (5 or more work best): one in two checks the training")
        if not 50 <= req.steps <= 5000 or not 0 <= req.blocks <= 12:
            raise ValueError("Steps must be 50 to 5000 and blocks 0 to 12")
        slug = re.sub(r"[^A-Za-z0-9]+", "-", ds.volume.name).strip("-")[:40] or "dataset"
        out_path = ctx.models_dir / f"dinov2_s14_{slug}_c{req.channel}_{datetime.now():%Y%m%d-%H%M%S}.pt"

        def work(job):
            from ..finetune import finetune
            from ..head import segment_with_head
            from ..metrics import dice

            job.message = "Reading the labelled slices"
            pairs = [(z, store.color_plane(ds_id, req.channel, z), store.load_label(ds_id, req.channel, z)) for z in zs]
            regions = {z: store.label_region(ds_id, req.channel, z) for z in zs}   # Pixels outside a label's region are ignored
            train = [(img, gt, regions[z]) for i, (z, img, gt) in enumerate(pairs) if i % 2 == 0]
            val = [(img, gt, regions[z]) for i, (z, img, gt) in enumerate(pairs) if i % 2 == 1]
            job.message = "Loading DINOv2"
            ft = finetune(train, val, train_blocks=req.blocks, steps=req.steps,
                          progress=lambda p, m: (setattr(job, "progress", 0.97 * p), setattr(job, "message", m)), cancelled=job.cancel.is_set)
            ft.save(out_path)
            # The same labels and validation slices with DINOv2 left frozen: the learned model of section 6
            job.message = "Comparing with the frozen features"
            st = SegmentationSettings.from_dict({**req.settings, "backbone": "dinov2_s14"})
            with store.compute_lock:
                tr = [(store.embedding(ds_id, req.channel, z, st), gt, regions[z]) for i, (z, _, gt) in enumerate(pairs) if i % 2 == 0]
                head = train_head(tr, st, [(req.channel, z) for i, (z, _, _) in enumerate(pairs) if i % 2 == 0], pixel_um=ds.volume.pixel_um)
                frozen = []
                for i, (z, _, gt) in enumerate(pairs):
                    if i % 2 == 1:
                        pred, v = segment_with_head(head, store.embedding(ds_id, req.channel, z, st), st, ds.volume.pixel_um).mask, regions[z]
                        frozen.append(dice(pred, gt) if v is None else dice(pred & v, gt & v))   # Scored where the label is valid
            return {"backbone": f"dinov2_s14@{out_path}", "file": out_path.name, "n_train": len(train), "n_val": len(val),
                    "val_dice_finetuned": ft.info["best_val_dice"], "val_dice_frozen": float(np.mean(frozen)),
                    "best_step": ft.info["best_step"], "steps": req.steps}

        job = store.start_job("finetune", work, meta={"dataset_id": ds_id, "channel": req.channel, "n_labels": len(zs)})
        return job.info()

    @r.get("/api/datasets/{ds_id}/head")
    def head_info(ds_id: str, c: int = 0):
        """The learned model for a channel, or null when none has been trained."""
        path = store.head_path(ds_id, c)
        return Head.load(path).info() if path.exists() else None

    return r
