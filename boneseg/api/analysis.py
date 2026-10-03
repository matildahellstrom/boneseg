"""Reports and bone histomorphometry."""
from __future__ import annotations

import io
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, Response

from .. import render
from ..head import Head
from ..segment import SegmentationSettings
from .context import AppContext, attachment
from .models import HistoRequest


def router(ctx: AppContext) -> APIRouter:
    r = APIRouter()
    store = ctx.store

    @r.get("/api/datasets/{ds_id}/report", response_class=HTMLResponse)
    def report(ds_id: str, c: int = 0, z: int = 0, download: bool = False):
        from .. import report as rep_mod

        ds = store.get(ds_id)
        res = ctx.last_result(ds_id, c, z)
        x = res.extra
        settings = x.get("settings", SegmentationSettings().to_dict())
        plane = store.plane(ds_id, c, z, settings["clip_low"], settings["clip_high"])
        ref = store.reference_mask(ds_id, z, c)
        png = rep_mod.composite(plane, res.mask, ref, x.get("points"), ds.meta.get("roi"))
        profile = None
        if x.get("profile_id"):
            try:
                profile = store.load_profile(x["profile_id"]).name
            except KeyError:
                profile = x["profile_id"]
        head = Head.load(store.head_path(ds_id, c)).info() if x.get("method") == "learned" and store.head_path(ds_id, c).exists() else None
        job = store.latest_job(ds_id)
        methods = rep_mod.methods_text(ds.info(), settings, x.get("method", "clicks"), x.get("n_pos", 0), x.get("n_neg", 0), profile, head,
                                       stack=job is not None)
        page = rep_mod.build_report(ds.info(), c, z, png, x.get("stats"), x.get("evaluation"), settings,
                                    {"value": res.threshold, "source": res.threshold_source}, ds.results.get(("histo_summary", z)),
                                    job.result if job else None, methods)
        headers = attachment(f"{Path(ds.volume.name).stem}_c{c}_z{z}_report.html") if download else {}
        return HTMLResponse(page, headers=headers)

    @r.post("/api/datasets/{ds_id}/histomorphometry")
    def histomorphometry_endpoint(ds_id: str, req: HistoRequest):
        from .. import histo

        ds = store.get(ds_id)
        settings = SegmentationSettings.from_dict(req.settings)
        bone = ctx.mask_from_spec(ds_id, req.z, req.bone, settings)
        cells = ctx.mask_from_spec(ds_id, req.z, req.cells, settings)
        roi = store.roi_mask(ds_id)
        summary, table = histo.histomorphometry(bone, cells, ds.volume.pixel_um, req.contact_um, roi)
        ds.results[("histo", req.z)] = table
        ds.results[("histo_summary", req.z)] = summary
        from PIL import Image

        rgba = histo.overlay(bone, cells, ds.volume.pixel_um, req.contact_um, roi)
        h, w = render.display_shape(*bone.shape, req.max_side)
        img = Image.fromarray(rgba, "RGBA").resize((w, h), Image.NEAREST)
        return {"summary": summary, "overlay_png": render.data_url(render.to_png_bytes(img)),
                "cells": json.loads(table.head(500).to_json(orient="records"))}

    @r.get("/api/datasets/{ds_id}/histomorphometry/cells.csv")
    def histo_cells(ds_id: str, z: int = 0):
        ds = store.get(ds_id)
        table = ds.results.get(("histo", z))
        if table is None:
            raise HTTPException(404, "Run histomorphometry on this slice first")
        stem = f"{Path(ds.volume.name).stem}_z{z}_histomorphometry_cells"
        return Response(table.to_csv(index=False), media_type="text/csv", headers=attachment(f"{stem}.csv"))

    @r.get("/api/datasets/{ds_id}/export/project.zip")
    def export_project(ds_id: str, include_masks: bool = False):
        """Everything done on a dataset except the image itself: settings, clicks, labels, learned models,
        profiles and stack results, with a README. For archiving an analysis or handing it to someone."""
        import datetime as dt
        import json as _json
        import zipfile

        from .. import __version__

        ds = store.get(ds_id)
        d = store.ds_dir(ds_id)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("dataset.json", _json.dumps({"file": ds.volume.name, "path": str(ds.path), **ds.info()}, indent=1, default=str))
            for name in ("annotations.json",):
                if (d / name).exists():
                    z.write(d / name, name)
            for sub in ("labels", "heads"):
                for f in sorted((d / sub).glob("*")) if (d / sub).exists() else []:
                    z.write(f, f"{sub}/{f.name}")
            for f in sorted((store.root / "profiles").glob("*.npz")):
                z.write(f, f"profiles/{f.name}")
            jobs = [j for j in store.jobs.values() if j.meta.get("dataset_id") == ds_id and j.status == "done"]
            for j in jobs:
                for f in sorted(j.out_dir.glob("*")):
                    if f.suffix == ".tif" and not include_masks:
                        continue
                    z.write(f, f"stack_runs/{j.id}/{f.name}")
            z.writestr("README.txt", (
                f"boneseg {__version__} project export, {dt.datetime.now():%Y-%m-%d %H:%M}\n"
                f"Image: {ds.volume.name} ({ds.volume.width} x {ds.volume.height} px, {ds.volume.n_z} slices, "
                f"{ds.volume.n_channels} channels, {ds.volume.voxel_um[2]:.3f} um/px). The image itself is not included.\n\n"
                "dataset.json      settings for this image: reference channel, region of interest, pixel size, groups\n"
                "annotations.json  clicks per slice, keyed 'channel:slice', as (y, x) pixel positions\n"
                "labels/           corrected masks saved as labels, cC_zZ.png\n"
                "heads/            learned models per channel (PyTorch)\n"
                "profiles/         all saved profiles; import them in boneseg with 'Import a shared profile'\n"
                f"stack_runs/       {len(jobs)} finished stack run(s): per-slice measurements, 3D objects and summaries"
                + (" with mask stacks" if include_masks else " (mask stacks left out; add ?include_masks=true)") + "\n"))
        return Response(buf.getvalue(), media_type="application/zip", headers=attachment(f"{Path(ds.volume.name).stem}_boneseg_project.zip"))

    return r

