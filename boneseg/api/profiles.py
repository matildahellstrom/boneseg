"""Saving, sharing and deleting profiles."""
from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ..backbone import BACKBONE_LABELS
from ..segment import Profile, SegmentationSettings, prototypes, segment_with_prototypes
from .context import AppContext
from .models import ProfileRequest


def router(ctx: AppContext) -> APIRouter:
    r = APIRouter()
    store = ctx.store

    @r.get("/api/profiles")
    def list_profiles():
        return store.list_profiles()

    @r.post("/api/profiles")
    def create_profile(req: ProfileRequest):
        settings = SegmentationSettings.from_dict(req.settings)
        if not req.pos:
            raise ValueError("A profile needs at least one positive click")
        with store.compute_lock:
            emb = store.embedding(req.dataset_id, req.channel, req.z, settings)
            pos_t, neg_t = prototypes(emb, req.pos), prototypes(emb, req.neg)
            cal = segment_with_prototypes(emb, pos_t, neg_t, replace(settings, threshold_mode="clicks"), pos_points=req.pos, neg_points=req.neg)
            pos_np, neg_np = pos_t.cpu().numpy(), neg_t.cpu().numpy()
        ds = store.get(req.dataset_id)
        prof = Profile(name=req.name.strip() or "Profile", backbone=settings.backbone, layer_from_end=settings.layer_from_end,
                       pos=pos_np, neg=neg_np,
                       settings=settings.to_dict(), description=req.description,
                       source=f"{ds.volume.name}, channel {req.channel}, slice {req.z}",
                       raw_threshold=cal.raw_threshold if cal.threshold_source == "clicks" else None)
        pid = store.save_profile(prof)
        return {"id": pid, **[p for p in store.list_profiles() if p["id"] == pid][0]}

    @r.get("/api/profiles/{pid}/download")
    def download_profile(pid: str):
        path = store.profile_path(pid)
        if not path.exists():
            raise HTTPException(404, "Unknown profile")
        return FileResponse(path, filename=f"{pid}.boneseg-profile.npz", media_type="application/octet-stream")

    @r.post("/api/profiles/import")
    def import_profile(file: UploadFile = File(...)):
        """Adds a profile file shared by someone else. It is checked by loading it before it is stored."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
        try:
            prof = Profile.load(tmp.name)
        except Exception as e:
            raise HTTPException(400, f"Not a boneseg profile: {type(e).__name__}") from None
        finally:
            Path(tmp.name).unlink(missing_ok=True)
        if prof.backbone not in BACKBONE_LABELS:
            raise HTTPException(400, f"Unknown backbone {prof.backbone}")
        pid = store.save_profile(prof)
        return {"id": pid, **[p for p in store.list_profiles() if p["id"] == pid][0]}

    @r.delete("/api/profiles/{pid}")
    def delete_profile(pid: str):
        store.delete_profile(pid)
        return {"ok": True}

    return r
