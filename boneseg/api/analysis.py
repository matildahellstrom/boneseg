"""Reports and the project export."""
from __future__ import annotations

import io
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response

from .. import render
from ..head import Head
from ..segment import SegmentationSettings
from .context import AppContext, attachment


def router(ctx: AppContext) -> APIRouter:
    r = APIRouter()
    store = ctx.store

    @r.get("/api/datasets/{ds_id}/report", response_class=HTMLResponse)
    def report(ds_id: str, c: int = 0, z: int = 0, download: bool = False):
        from .. import report as rep_mod

        ds = store.get(ds_id)
        multi = ds.results.get(("multi", c, z))
        single = ds.results.get((c, z))
        if multi is not None and (single is None or multi.get("time", 0) >= single.extra.get("time", 0)):
            return _multi_report(ds, c, z, multi, download)
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
                                    {"value": res.threshold, "source": res.threshold_source}, job.result if job else None, methods)
        headers = attachment(f"{Path(ds.volume.name).stem}_c{c}_z{z}_report.html") if download else {}
        return HTMLResponse(page, headers=headers)

    def _multi_report(ds, c, z, m, download):
        from .. import report as rep_mod

        settings = m.get("settings", SegmentationSettings().to_dict())
        plane = store.plane(ds.id, c, z, settings["clip_low"], settings["clip_high"])
        png = rep_mod.composite_labels(plane, m["labels"], m.get("colors", []))
        head = Head.load(store.head_path(ds.id, c)).info() if m.get("method") == "learned" and store.head_path(ds.id, c).exists() else None
        n_pos = sum(len(v) for v in (m.get("pos") or {}).values())
        job = store.latest_job(ds.id)
        source = "learned structures" if m.get("method") == "learned" else "structures"
        methods = rep_mod.methods_text(ds.info(), settings, source, n_pos, len(m.get("neg", [])), None, head, stack=job is not None)
        page = rep_mod.build_report(ds.info(), c, z, png, None, None, settings, None, job.result if job else None, methods,
                                    structures=m.get("stats"))
        headers = attachment(f"{Path(ds.volume.name).stem}_c{c}_z{z}_report.html") if download else {}
        return HTMLResponse(page, headers=headers)

    @r.get("/api/datasets/{ds_id}/export/project.zip")
    def export_project(ds_id: str, include_masks: bool = False):
        # Written to a temporary file and streamed from there: with mask stacks the archive can be gigabytes
        """Everything done on a dataset except the image itself: settings, clicks, labels, learned models,
        profiles and stack results, with a README. For archiving an analysis or handing it to someone."""
        import datetime as dt
        import json as _json
        import zipfile

        from .. import __version__

        ds = store.get(ds_id)
        d = store.ds_dir(ds_id)
        import tempfile

        from starlette.background import BackgroundTask

        tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        tmp.close()
        with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as z:
            # The file name only: the server's folders are nobody else's business
            z.writestr("dataset.json", _json.dumps({"file": ds.volume.name, **ds.info()}, indent=1, default=str))
            for name in ("annotations.json",):
                if (d / name).exists():
                    z.write(d / name, name)
            for sub in ("labels", "heads"):
                for f in sorted((d / sub).glob("*")) if (d / sub).exists() else []:
                    z.write(f, f"{sub}/{f.name}")
            # Only this image's profiles; on a shared server the others belong to other people
            own = [p["id"] for p in store.list_profiles() if p.get("dataset_id") == ds_id]
            for pid in own:
                z.write(store.profile_path(pid), f"profiles/{pid}.npz")
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
                f"profiles/         {len(own)} profile(s) made from this image; import them in boneseg with 'Import a shared profile'\n"
                f"stack_runs/       {len(jobs)} finished stack run(s): per-slice measurements, 3D objects and summaries"
                + (" with mask stacks" if include_masks else " (mask stacks left out; add ?include_masks=true)") + "\n"))
        return FileResponse(tmp.name, media_type="application/zip", headers=attachment(f"{Path(ds.volume.name).stem}_boneseg_project.zip"),
                            background=BackgroundTask(lambda: Path(tmp.name).unlink(missing_ok=True)))

    return r

