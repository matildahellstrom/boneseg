"""FastAPI server for the boneseg web app."""
from __future__ import annotations

import io
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__, metrics, quantify, render
from .backbone import BACKBONE_LABELS, dino_weights_cached, pick_device
from .pipeline import StackRequest, run_stack
from .segment import Profile, SegmentationSettings, prototypes, segment_with_prototypes, suggest_click, uncertainty_map
from .store import Store

STATIC = Path(__file__).parent / "static"


class SegmentRequest(BaseModel):
    channel: int = 0
    z: int = 0
    pos: list[tuple[float, float]] = Field(default_factory=list)  # (y, x) in full-resolution pixels
    neg: list[tuple[float, float]] = Field(default_factory=list)
    profile_id: str | None = None
    settings: dict = Field(default_factory=dict)
    uncertainty: bool = False
    max_side: int = 1600


class StackJobRequest(BaseModel):
    channel: int = 0
    ref_z: int | None = None
    pos: list[tuple[float, float]] = Field(default_factory=list)
    neg: list[tuple[float, float]] = Field(default_factory=list)
    profile_id: str | None = None
    z_start: int = 0
    z_end: int | None = None
    z_step: int = 1
    settings: dict = Field(default_factory=dict)


class ProfileRequest(BaseModel):
    name: str
    description: str = ""
    dataset_id: str
    channel: int = 0
    z: int = 0
    pos: list[tuple[float, float]]
    neg: list[tuple[float, float]] = Field(default_factory=list)
    settings: dict = Field(default_factory=dict)


class PathRequest(BaseModel):
    path: str


class MetaRequest(BaseModel):
    reference_channel: int | None = None
    notes: str | None = None


def create_app(data_dir: str | Path | None = None) -> FastAPI:
    data_dir = Path(data_dir or os.environ.get("BONESEG_DATA_DIR", "projects"))
    store = Store(data_dir)
    app = FastAPI(title="boneseg", version=__version__)
    app.state.store = store

    @app.exception_handler(KeyError)
    async def key_error(_: Request, exc: KeyError):
        return JSONResponse({"detail": str(exc).strip("'\"")}, status_code=404)

    @app.exception_handler(ValueError)
    async def value_error(_: Request, exc: ValueError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(IndexError)
    async def index_error(_: Request, exc: IndexError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    # Pages and health ---------------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text()

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/api/health")
    def health():
        return {
            "version": __version__,
            "device": str(pick_device()),
            "default_settings": SegmentationSettings().to_dict(),
            "backbones": [{"id": k, "label": v, "ready": dino_weights_cached(k)} for k, v in BACKBONE_LABELS.items()],
        }

    # Datasets -----------------------------------------------------------------------------------
    @app.get("/api/datasets")
    def list_datasets():
        return [ds.info() for ds in sorted(store.datasets.values(), key=lambda d: -d.created)]

    @app.post("/api/datasets")
    def upload(file: UploadFile = File(...)):
        def write(dest: Path):
            with open(dest, "wb") as out:
                shutil.copyfileobj(file.file, out, length=8 * 1024 * 1024)

        return store.add_dataset_file(file.filename or "upload", write).info()

    @app.post("/api/datasets/demo")
    def demo():
        from .demo import DEMO_CHANNEL_NAMES, write_demo_tiff

        ds = store.add_dataset_file("demo_bone_stack.tif", lambda dest: write_demo_tiff(dest))
        ds.volume.channel_names = list(DEMO_CHANNEL_NAMES)
        store.update_meta(ds.id, reference_channel=2, reference_guessed=False, channel_names=DEMO_CHANNEL_NAMES, default_channel=1)
        return ds.info()

    @app.post("/api/datasets/from-path")
    def from_path(req: PathRequest):
        try:
            return store.add_dataset_path(req.path).info()
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))

    @app.get("/api/datasets/{ds_id}")
    def get_dataset(ds_id: str):
        return store.get(ds_id).info()

    @app.patch("/api/datasets/{ds_id}")
    def patch_dataset(ds_id: str, req: MetaRequest):
        ds = store.get(ds_id)
        updates = req.model_dump(exclude_unset=True)
        if updates.get("reference_channel") is not None and not 0 <= updates["reference_channel"] < ds.volume.n_channels:
            raise HTTPException(400, "Reference channel out of range")
        return store.update_meta(ds_id, **updates, reference_guessed=False).info()

    @app.delete("/api/datasets/{ds_id}")
    def delete_dataset(ds_id: str):
        store.delete(ds_id)
        return {"ok": True}

    @app.get("/api/datasets/{ds_id}/plane")
    def plane(ds_id: str, c: int = 0, z: int = 0, low: float = 1.0, high: float = 99.5, max_side: int = 1600, gamma: float = 1.0):
        p = store.plane(ds_id, c, z, low, high)
        return Response(render.gray_png(p, max_side, gamma), media_type="image/png", headers={"Cache-Control": "max-age=3600"})

    @app.get("/api/datasets/{ds_id}/reference")
    def reference(ds_id: str, z: int = 0, max_side: int = 1600):
        ref = store.reference_mask(ds_id, z)
        if ref is None:
            raise HTTPException(404, "No reference channel set")
        return Response(render.mask_png(ref, max_side, color=(255, 60, 90), fill_alpha=0), media_type="image/png")

    # Segmentation -------------------------------------------------------------------------------
    def _prototypes_for(ds_id, channel, z, pos_pts, neg_pts, profile_id, settings):
        """Prototypes from clicks, a profile, or a profile refined by clicks.
        Also returns the profile's calibrated raw threshold, if any."""
        if profile_id:
            prof = store.load_profile(profile_id)
            if prof.backbone != settings.backbone or prof.layer_from_end != settings.layer_from_end:
                raise ValueError(f"Profile '{prof.name}' was made with {BACKBONE_LABELS.get(prof.backbone, prof.backbone)} "
                                 f"(block {prof.layer_from_end} from the end). Switch to that backbone to use it.")
            pos, neg = prof.tensors()
            if pos_pts or neg_pts:  # Clicks refine the profile
                import torch

                emb = store.embedding(ds_id, channel, z, settings)
                pos = torch.cat([pos.to(emb.grid.device), prototypes(emb, pos_pts)]) if pos_pts else pos
                neg = torch.cat([neg.to(emb.grid.device), prototypes(emb, neg_pts)]) if neg_pts else neg
                # The stored threshold belongs to the stored prototypes only
                return pos, neg, None
            return pos, neg, prof.raw_threshold
        if not pos_pts:
            raise ValueError("Add at least one positive click, or pick a profile")
        emb = store.embedding(ds_id, channel, z, settings)
        return prototypes(emb, pos_pts), prototypes(emb, neg_pts), None

    @app.post("/api/datasets/{ds_id}/segment")
    def segment_endpoint(ds_id: str, req: SegmentRequest):
        t0 = time.time()
        ds = store.get(ds_id)
        settings = SegmentationSettings.from_dict(req.settings)
        with store.compute_lock:
            # Prototypes first, so that a profile made with another backbone fails before any model is loaded
            pos, neg, prof_thr = _prototypes_for(ds_id, req.channel, req.z, req.pos, req.neg, req.profile_id, settings)
            emb = store.embedding(ds_id, req.channel, req.z, settings)
            t_embed = time.time() - t0
            res = segment_with_prototypes(emb, pos, neg, settings, ds.volume.pixel_um, raw_threshold=prof_thr,
                                          pos_points=req.pos, neg_points=req.neg)
            u = uncertainty_map(emb, pos, neg, settings, threshold=res.threshold) if req.uncertainty and pos.shape[0] > 0 else None
        ds.results[(req.channel, req.z)] = res
        img = store.plane(ds_id, req.channel, req.z, settings.clip_low, settings.clip_high)
        out = {
            "heat_png": render.data_url(render.heat_png(res.heat, req.max_side)),
            "mask_png": render.data_url(render.mask_png(res.mask, req.max_side)),
            "threshold": res.threshold,
            "raw_threshold": res.raw_threshold,
            "threshold_source": res.threshold_source,
            "stats": quantify.summarize_mask(res.mask, img, ds.volume.pixel_um),
            "timing": {"embed_s": round(t_embed, 3), "total_s": 0.0},
        }
        ref = store.reference_mask(ds_id, req.z)
        if ref is not None:
            out["evaluation"] = metrics.compare(res.mask, ref, ds.volume.pixel_um)
        if u is not None:
            res.uncertainty = u
            out["uncertainty_png"] = render.data_url(render.uncertainty_png(u, req.max_side))
            out["uncertain_fraction"] = float((u > 0.05).mean())
            sug = suggest_click(u, list(req.pos) + list(req.neg))
            out["suggestion"] = list(sug) if sug else None
        out["timing"]["total_s"] = round(time.time() - t0, 3)
        return out

    def _last_result(ds_id, c, z):
        res = store.get(ds_id).results.get((c, z))
        if res is None:
            raise HTTPException(404, "Segment this slice first")
        return res

    @app.get("/api/datasets/{ds_id}/export/mask")
    def export_mask(ds_id: str, c: int = 0, z: int = 0, fmt: str = "png"):
        ds = store.get(ds_id)
        res = _last_result(ds_id, c, z)
        stem = f"{Path(ds.volume.name).stem}_c{c}_z{z}_mask"
        if fmt == "tif":
            buf = io.BytesIO()
            py, px = ds.volume.pixel_um
            tifffile.imwrite(buf, res.mask.astype(np.uint8) * 255, imagej=True, resolution=(1 / px, 1 / py), metadata={"unit": "um"})
            return Response(buf.getvalue(), media_type="image/tiff", headers={"Content-Disposition": f'attachment; filename="{stem}.tif"'})
        return Response(render.mask_full_png(res.mask), media_type="image/png", headers={"Content-Disposition": f'attachment; filename="{stem}.png"'})

    @app.get("/api/datasets/{ds_id}/export/objects.csv")
    def export_objects(ds_id: str, c: int = 0, z: int = 0):
        ds = store.get(ds_id)
        res = _last_result(ds_id, c, z)
        table = quantify.object_table(res.mask, store.plane(ds_id, c, z), ds.volume.pixel_um)
        stem = f"{Path(ds.volume.name).stem}_c{c}_z{z}_objects"
        return Response(table.to_csv(index=False), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{stem}.csv"'})

    # Stack jobs ---------------------------------------------------------------------------------
    @app.post("/api/datasets/{ds_id}/stack")
    def stack_job(ds_id: str, req: StackJobRequest):
        ds = store.get(ds_id)
        settings = SegmentationSettings.from_dict(req.settings)
        z_end = ds.volume.n_z - 1 if req.z_end is None else min(req.z_end, ds.volume.n_z - 1)
        z_list = list(range(max(0, req.z_start), z_end + 1, max(1, req.z_step)))
        if not z_list:
            raise ValueError("The slice range is empty")
        ref_z = req.ref_z if req.ref_z is not None else z_list[0]
        with store.compute_lock:
            pos, neg, raw_thr = _prototypes_for(ds_id, req.channel, ref_z, req.pos, req.neg, req.profile_id, settings)
            if settings.threshold_mode == "clicks" and req.pos and req.neg:
                # Calibrate on the annotated slice, then carry the raw threshold through the stack
                ref = segment_with_prototypes(store.embedding(ds_id, req.channel, ref_z, settings), pos, neg, settings,
                                              ds.volume.pixel_um, pos_points=req.pos, neg_points=req.neg)
                raw_thr = ref.raw_threshold
        c = req.channel

        def work(job):
            return run_stack(
                StackRequest(z_list=z_list, ref_z=ref_z), pos, neg, settings,
                raw_threshold=raw_thr,
                get_embedding=lambda z: store.embedding(ds_id, c, z, settings),
                lock=store.compute_lock,
                get_image=lambda z: store.plane(ds_id, c, z, settings.clip_low, settings.clip_high),
                get_reference=lambda z: store.reference_mask(ds_id, z),
                voxel_um=ds.volume.voxel_um, out_dir=job.out_dir,
                progress=lambda p, m: (setattr(job, "progress", p), setattr(job, "message", m)),
                cancelled=job.cancel.is_set,
            )

        job = store.start_job("stack", work)
        job.result = {"dataset_id": ds_id, "n_slices": len(z_list)}
        return job.info()

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        return store.get_job(job_id).info()

    @app.post("/api/jobs/{job_id}/cancel")
    def job_cancel(job_id: str):
        job = store.get_job(job_id)
        job.cancel.set()
        return job.info()

    @app.get("/api/jobs/{job_id}/files/{name}")
    def job_file(job_id: str, name: str):
        job = store.get_job(job_id)
        if name not in ("masks.tif", "slices.csv", "summary.json", "objects_3d.csv"):
            raise HTTPException(404, "Unknown file")
        path = job.out_dir / name
        if not path.exists():
            raise HTTPException(404, "Not ready yet")
        return FileResponse(path, filename=f"{job_id}_{name}")

    # Profiles -----------------------------------------------------------------------------------
    @app.get("/api/profiles")
    def list_profiles():
        return store.list_profiles()

    @app.post("/api/profiles")
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

    @app.delete("/api/profiles/{pid}")
    def delete_profile(pid: str):
        store.delete_profile(pid)
        return {"ok": True}

    return app


