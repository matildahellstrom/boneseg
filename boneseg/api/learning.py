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

    @r.put("/api/datasets/{ds_id}/annotations")
    def put_annotation(ds_id: str, req: AnnotationRequest):
        store.set_annotation(ds_id, req.channel, req.z, req.pos, req.neg)
        return {"ok": True}

    @r.get("/api/datasets/{ds_id}/labels")
    def list_labels(ds_id: str):
        return store.list_labels(ds_id)

    @r.post("/api/datasets/{ds_id}/labels")
    def save_label(ds_id: str, req: LabelRequest):
        if req.mask_png:
            import base64

            from PIL import Image

            raw = base64.b64decode(req.mask_png.split(",", 1)[-1])
            img = np.asarray(Image.open(io.BytesIO(raw)).convert("RGBA"))
            mask = img[..., 3] > 127  # Painted pixels are opaque
        else:
            mask = ctx.last_result(ds_id, req.channel, req.z).mask
        store.save_label(ds_id, req.channel, req.z, mask)
        return {"ok": True, "labels": store.list_labels(ds_id)}

    @r.get("/api/datasets/{ds_id}/labels/png")
    def label_png(ds_id: str, c: int = 0, z: int = 0, max_side: int = 1600):
        m = store.load_label(ds_id, c, z)
        if m is None:
            raise HTTPException(404, "No label for this slice")
        return Response(render.mask_png(m, max_side, color=(255, 210, 0), fill_alpha=0), media_type="image/png")

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
        with store.compute_lock:
            samples = [(store.embedding(ds_id, req.channel, l["z"], settings), store.load_label(ds_id, req.channel, l["z"])) for l in labels]
            head = train_head(samples, settings, [(req.channel, l["z"]) for l in labels], kind=req.kind, pixel_um=store.get(ds_id).volume.pixel_um)
        head.save(store.head_path(ds_id, req.channel))
        return {**head.info(), "seconds": round(time.time() - t0, 2)}

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
