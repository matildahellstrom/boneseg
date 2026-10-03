"""Segmenting slices and stacks, exports and jobs."""
from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import tifffile
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from .. import metrics, quantify, render
from ..head import segment_with_head
from ..pipeline import StackRequest, run_stack
from ..segment import SegmentationSettings, segment_with_prototypes, suggest_click, uncertainty_map
from .context import AppContext
from .models import SegmentRequest, StackJobRequest


def router(ctx: AppContext) -> APIRouter:
    r = APIRouter()
    store = ctx.store

    @r.post("/api/datasets/{ds_id}/segment")
    def segment_endpoint(ds_id: str, req: SegmentRequest):
        t0 = time.time()
        ds = store.get(ds_id)
        settings = SegmentationSettings.from_dict(req.settings)
        with store.compute_lock:
            profile_head = ctx.learned_profile_head(req.profile_id, settings)
            if req.method == "learned" or profile_head is not None:
                head = profile_head or ctx.load_head(ds_id, req.channel, settings)
                emb = store.embedding(ds_id, req.channel, req.z, settings)
                t_embed = time.time() - t0
                res = segment_with_head(head, emb, settings, ds.volume.pixel_um)
                # Uncertain where the probability is near one half
                u = np.clip(1 - 4 * np.abs(res.heat - 0.5), 0, 1).astype(np.float32) if req.uncertainty else None
            else:
                # Prototypes first, so that a profile made with another backbone fails before any model is loaded
                pos, neg, prof_thr = ctx.prototypes_for(ds_id, req.channel, req.z, req.pos, req.neg, req.profile_id, settings)
                emb = store.embedding(ds_id, req.channel, req.z, settings)
                t_embed = time.time() - t0
                res = segment_with_prototypes(emb, pos, neg, settings, ds.volume.pixel_um, raw_threshold=prof_thr,
                                              pos_points=req.pos, neg_points=req.neg)
                u = uncertainty_map(emb, pos, neg, settings, threshold=res.threshold) if req.uncertainty and pos.shape[0] > 0 else None
        roi = store.roi_mask(ds_id)
        if roi is not None:
            # Keep the unclipped mask for histomorphometry, where the region's edge is not a bone surface
            res.extra["unclipped"] = res.mask
            res.mask = res.mask & roi
        ds.results[(req.channel, req.z)] = res
        img = store.plane(ds_id, req.channel, req.z, settings.clip_low, settings.clip_high)
        out = {
            "heat_png": render.data_url(render.heat_png(res.heat, req.max_side)),
            "mask_png": render.data_url(render.mask_png(res.mask, req.max_side)),
            "threshold": res.threshold,
            "raw_threshold": res.raw_threshold,
            "threshold_source": res.threshold_source,
            "stats": quantify.summarize_mask(res.mask, img, ds.volume.pixel_um, roi),
            "timing": {"embed_s": round(t_embed, 3), "total_s": 0.0},
        }
        ref = store.reference_mask(ds_id, req.z, req.channel)
        if ref is not None:
            out["evaluation"] = metrics.compare(res.mask, ref, ds.volume.pixel_um, roi, hd95_max_side=1500)
            out["evaluation"]["against"] = "reference channel" if ds.meta.get("reference_channel") is not None else "your saved label"
            refc = ref & roi if roi is not None else ref
            out["error_png"] = render.data_url(render.error_png(res.mask, refc, req.max_side))
            out["evaluation"]["false_positive_um2"] = float((res.mask & ~refc).sum() * np.prod(ds.volume.pixel_um))
            out["evaluation"]["false_negative_um2"] = float((~res.mask & refc).sum() * np.prod(ds.volume.pixel_um))
        if u is not None:
            res.uncertainty = u
            out["uncertainty_png"] = render.data_url(render.uncertainty_png(u, req.max_side))
            out["uncertain_fraction"] = float((u > 0.05).mean())
            sug = suggest_click(u, list(req.pos) + list(req.neg))
            out["suggestion"] = list(sug) if sug else None
        out["timing"]["total_s"] = round(time.time() - t0, 3)
        # Neighbouring slices next, so stepping through the stack stays fast
        store.prefetch(ds_id, req.channel, [req.z + 1, req.z - 1, req.z + 2, req.z - 2], settings)
        # Remembered for the report
        res.extra.update({"settings": settings.to_dict(), "method": req.method, "profile_id": req.profile_id, "n_pos": len(req.pos),
                          "n_neg": len(req.neg), "stats": out["stats"], "evaluation": out.get("evaluation"),
                          "points": {"pos": [list(p) for p in req.pos], "neg": [list(p) for p in req.neg]}})
        return out

    @r.get("/api/datasets/{ds_id}/export/mask")
    def export_mask(ds_id: str, c: int = 0, z: int = 0, fmt: str = "png"):
        ds = store.get(ds_id)
        res = ctx.last_result(ds_id, c, z)
        stem = f"{Path(ds.volume.name).stem}_c{c}_z{z}_mask"
        if fmt == "tif":
            buf = io.BytesIO()
            py, px = ds.volume.pixel_um
            tifffile.imwrite(buf, res.mask.astype(np.uint8) * 255, imagej=True, resolution=(1 / px, 1 / py), metadata={"unit": "um"})
            return Response(buf.getvalue(), media_type="image/tiff", headers={"Content-Disposition": f'attachment; filename="{stem}.tif"'})
        return Response(render.mask_full_png(res.mask), media_type="image/png", headers={"Content-Disposition": f'attachment; filename="{stem}.png"'})

    @r.get("/api/datasets/{ds_id}/export/objects.csv")
    def export_objects(ds_id: str, c: int = 0, z: int = 0, all_channels: bool = True):
        """One row per object with size and shape, plus its raw intensity in every channel."""
        import re

        ds = store.get(ds_id)
        res = ctx.last_result(ds_id, c, z)
        table = quantify.object_table(res.mask, store.plane(ds_id, c, z), ds.volume.pixel_um)
        if all_channels and len(table):
            ref = ds.meta.get("reference_channel")
            names = ds.info().get("channel_names", ds.volume.channel_names)
            chans = {}
            for i in range(ds.volume.n_channels):
                if i == ref:
                    continue
                safe = re.sub(r"[^A-Za-z0-9]+", "_", f"ch{i}_{names[i]}").strip("_")
                chans[safe] = store.raw_plane(ds_id, i, z)
            table = table.merge(quantify.channel_intensities(res.mask, chans), on="label", how="left")
        stem = f"{Path(ds.volume.name).stem}_c{c}_z{z}_objects"
        return Response(table.to_csv(index=False), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{stem}.csv"'})

    @r.post("/api/datasets/{ds_id}/stack")
    def stack_job(ds_id: str, req: StackJobRequest):
        ds = store.get(ds_id)
        settings = SegmentationSettings.from_dict(req.settings)
        z_end = ds.volume.n_z - 1 if req.z_end is None else min(req.z_end, ds.volume.n_z - 1)
        z_list = list(range(max(0, req.z_start), z_end + 1, max(1, req.z_step)))
        if not z_list:
            raise ValueError("The slice range is empty")
        ref_z = req.ref_z if req.ref_z is not None else z_list[0]
        head = ctx.learned_profile_head(req.profile_id, settings)
        if head is None and req.method == "learned":
            head = ctx.load_head(ds_id, req.channel, settings)
        with store.compute_lock:
            if head is not None:
                import torch

                pos, neg, raw_thr = torch.zeros((1, 1)), torch.zeros((0, 1)), None
            else:
                pos, neg, raw_thr = ctx.prototypes_for(ds_id, req.channel, ref_z, req.pos, req.neg, req.profile_id, settings)
            if head is None and settings.threshold_mode == "clicks" and req.pos and req.neg:
                # Calibrate on the annotated slice, then carry the raw threshold through the stack
                ref = segment_with_prototypes(store.embedding(ds_id, req.channel, ref_z, settings), pos, neg, settings,
                                              ds.volume.pixel_um, pos_points=req.pos, neg_points=req.neg)
                raw_thr = ref.raw_threshold
        c = req.channel

        def work(job):
            return run_stack(
                StackRequest(z_list=z_list, ref_z=ref_z), pos, neg, settings,
                raw_threshold=raw_thr, head=head,
                get_embedding=lambda z: store.embedding(ds_id, c, z, settings),
                lock=store.compute_lock, roi=store.roi_mask(ds_id),
                get_image=lambda z: store.plane(ds_id, c, z, settings.clip_low, settings.clip_high),
                get_reference=lambda z: store.reference_mask(ds_id, z, c),
                voxel_um=ds.volume.voxel_um, out_dir=job.out_dir,
                progress=lambda p, m: (setattr(job, "progress", p), setattr(job, "message", m)),
                cancelled=job.cancel.is_set,
            )

        job = store.start_job("stack", work, meta={"dataset_id": ds_id, "n_slices": len(z_list), "channel": c, "method": req.method})
        return job.info()

    @r.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        return store.get_job(job_id).info()

    @r.post("/api/jobs/{job_id}/cancel")
    def job_cancel(job_id: str):
        job = store.get_job(job_id)
        job.cancel.set()
        return job.info()

    @r.get("/api/jobs/{job_id}/files/{name}")
    def job_file(job_id: str, name: str):
        job = store.get_job(job_id)
        if name not in ("masks.tif", "labels_3d.tif", "slices.csv", "summary.json", "objects_3d.csv"):
            raise HTTPException(404, "Unknown file")
        path = job.out_dir / name
        if not path.exists():
            raise HTTPException(404, "Not ready yet")
        return FileResponse(path, filename=f"{job_id}_{name}")

    return r
