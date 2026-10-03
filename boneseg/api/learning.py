"""Saved clicks, corrected labels and the learned model."""
from __future__ import annotations

import io
import time

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from .. import render
from ..head import Head, head_to_profile_dict, train_head
from ..segment import Profile, SegmentationSettings
from .context import AppContext
from .models import AnnotationRequest, LabelRequest, ProfileFromHeadRequest, HeadRequest


def router(ctx: AppContext) -> APIRouter:
    r = APIRouter()
    store = ctx.store

    @r.get("/api/datasets/{ds_id}/annotations")
    def get_annotations(ds_id: str):
        return store.get_annotations(ds_id)

    @r.post("/api/datasets/{ds_id}/annotations")  # For navigator.sendBeacon, which can only POST, when the page closes
    @r.put("/api/datasets/{ds_id}/annotations")
    def put_annotation(ds_id: str, req: AnnotationRequest):
        store.set_annotation(ds_id, req.channel, req.z, req.pos, req.neg, req.extra)
        return {"ok": True}

    @r.get("/api/datasets/{ds_id}/labels")
    def list_labels(ds_id: str):
        return store.list_labels(ds_id)

    @r.post("/api/datasets/{ds_id}/labels")
    def save_label(ds_id: str, req: LabelRequest):
        multi = len(req.structures) > 1
        if req.mask_png:
            import base64

            from PIL import Image

            raw = base64.b64decode(req.mask_png.split(",", 1)[-1])
            img = np.asarray(Image.open(io.BytesIO(raw)).convert("RGBA"))
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
        return {"ok": True, "labels": store.list_labels(ds_id)}

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
        settings = SegmentationSettings.from_dict(req.settings)
        labels = [l for l in store.list_labels(ds_id) if l["channel"] == req.channel]
        if not labels:
            raise ValueError("Save at least one corrected mask as a label on this channel first")
        t0 = time.time()
        names = store.label_structures(ds_id, req.channel)
        with store.compute_lock:
            if names and len(names) > 1:
                # Labels with several structures train a model that predicts all of them
                from ..head import train_head_multi

                samples = [(store.embedding(ds_id, req.channel, l["z"], settings), store.load_label_map(ds_id, req.channel, l["z"])) for l in labels]
                head = train_head_multi(samples, names, settings, [(req.channel, l["z"]) for l in labels], pixel_um=store.get(ds_id).volume.pixel_um)
            else:
                samples = [(store.embedding(ds_id, req.channel, l["z"], settings), store.load_label(ds_id, req.channel, l["z"])) for l in labels]
                head = train_head(samples, settings, [(req.channel, l["z"]) for l in labels], kind=req.kind, pixel_um=store.get(ds_id).volume.pixel_um)
        head.save(store.head_path(ds_id, req.channel))
        out = {**head.info(), "seconds": round(time.time() - t0, 2)}
        ds = store.get(ds_id)
        if ds.meta.get("reference_channel") is not None and not head.names:
            # An independent check: Dice against the expert reference on up to five slices that were not labelled.
            # Cross-validation only measures agreement with the labels, which may themselves be imperfect.
            from ..head import segment_with_head
            from ..metrics import dice

            labelled = {l["z"] for l in labels}
            pool = [z for z in range(ds.volume.n_z) if z not in labelled]
            picks = sorted({pool[int(i)] for i in np.linspace(0, len(pool) - 1, min(5, len(pool)))}) if pool else []
            scores = []
            with store.compute_lock:
                for z in picks:
                    ref = store.reference_mask(ds_id, z, req.channel)
                    if ref is None or not ref.any():
                        continue
                    res = segment_with_head(head, store.embedding(ds_id, req.channel, z, settings), settings, ds.volume.pixel_um)
                    scores.append(dice(res.mask, ref))
            if scores:
                out["reference_check"] = {"mean_dice": float(np.mean(scores)), "n_slices": len(scores)}
        return out

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

    @r.get("/api/datasets/{ds_id}/head")
    def head_info(ds_id: str, c: int = 0):
        """The learned model for a channel, or null when none has been trained."""
        path = store.head_path(ds_id, c)
        return Head.load(path).info() if path.exists() else None

    return r
