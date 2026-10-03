"""Reports and bone histomorphometry."""
from __future__ import annotations

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

    return r
