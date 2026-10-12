"""micro-SAM fine-tuned on a dataset's labels: the Kaggle package, loading the trained file, its status."""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from .. import microsam
from .context import AppContext, attachment

MAX_MODEL_BYTES = 2 * 2**30   # A ViT-B model file is about 0.4 GB


class KaggleRequest(BaseModel):
    channel: int = 0
    username: str = ""
    recipe: str = "full"       # "full" (micro-SAM's default) or "lora"
    epochs: int = 40
    settings: dict = Field(default_factory=dict)


def router(ctx: AppContext) -> APIRouter:
    r = APIRouter()
    store = ctx.store

    def model_path(ds_id: str, c: int) -> Path:
        store.get(ds_id)   # 404 for an unknown dataset
        return ctx.microsam_path(ds_id, c)

    @r.get("/api/datasets/{ds_id}/microsam")
    def status(ds_id: str, c: int = 0):
        """Whether micro-SAM can run here, the loaded model for this channel (or null) and the number of labels."""
        return {"available": microsam.available(), "model": microsam.model_info(model_path(ds_id, c)),
                "n_labels": sum(1 for l in store.list_labels(ds_id) if l["channel"] == c)}

    @r.post("/api/datasets/{ds_id}/microsam/kaggle")
    def kaggle(ds_id: str, req: KaggleRequest):
        """A zip with this channel's labelled slices as a Kaggle dataset and a notebook that fine-tunes micro-SAM."""
        ds = store.get(ds_id)
        names = store.label_structures(ds_id, req.channel)
        if names and len(names) > 1:
            raise ValueError("micro-SAM learns one structure; these labels hold several")
        low, high = req.settings.get("clip_low", 1.0), req.settings.get("clip_high", 99.5)
        zs = sorted(l["z"] for l in store.list_labels(ds_id) if l["channel"] == req.channel)
        # The images are normalised as the app shows them, and the model later sees them the same way
        items = [(z, store.color_plane(ds_id, req.channel, z, low, high), store.load_label(ds_id, req.channel, z),
                  store.label_region(ds_id, req.channel, z)) for z in zs]
        path = microsam.kaggle_package(items, username=req.username.strip().lower(), name=Path(ds.volume.name).stem,
                                       channel=req.channel, clip=(low, high), recipe=req.recipe, epochs=req.epochs)
        return FileResponse(path, media_type="application/zip", headers=attachment(path.name),
                            background=BackgroundTask(lambda: shutil.rmtree(path.parent, ignore_errors=True)))

    @r.post("/api/datasets/{ds_id}/microsam/model")
    async def load_model(ds_id: str, request: Request, c: int = 0):
        """Loads the model file the Kaggle notebook wrote (raw file as the request body). It is checked as weights
        only, so a wrong or tampered file is refused without running any code from it."""
        dest = model_path(ds_id, c)
        if int(request.headers.get("content-length") or 0) > MAX_MODEL_BYTES:
            raise HTTPException(413, "That file is too large for a micro-SAM model")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.parent / f".incoming-{uuid.uuid4().hex}.pt"
        try:
            written = 0
            with open(tmp, "wb") as out:
                async for chunk in request.stream():
                    written += len(chunk)
                    if written > MAX_MODEL_BYTES:
                        raise HTTPException(413, "That file is too large for a micro-SAM model")
                    out.write(chunk)
            microsam.forget(dest)
            info = microsam.install(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)
        return {**microsam.model_info(dest), **info}

    @r.delete("/api/datasets/{ds_id}/microsam/model")
    def delete_model(ds_id: str, c: int = 0):
        dest = model_path(ds_id, c)
        microsam.forget(dest)
        dest.unlink(missing_ok=True)
        dest.with_suffix(".json").unlink(missing_ok=True)
        return {"ok": True}

    return r
