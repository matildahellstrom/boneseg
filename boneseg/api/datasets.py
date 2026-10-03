"""Datasets: upload, open, browse, region of interest and pixel size."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import tifffile
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import Response

from .. import render
from ..io import normalize_plane as bio_normalize
from .context import AppContext
from .models import PathRequest, RoiRequest, MetaRequest


def router(ctx: AppContext) -> APIRouter:
    r = APIRouter()
    store = ctx.store
    allow_paths = ctx.allow_paths

    @r.get("/api/datasets")
    def list_datasets():
        return [ds.info() for ds in sorted(store.datasets.values(), key=lambda d: -d.created)]

    @r.post("/api/datasets")
    def upload(file: UploadFile = File(...)):
        def write(dest: Path):
            with open(dest, "wb") as out:
                shutil.copyfileobj(file.file, out, length=8 * 1024 * 1024)

        return store.add_dataset_file(file.filename or "upload", write).info()

    @r.post("/api/datasets/stream")
    async def upload_stream(request: Request, filename: str):
        """Upload with the raw file as the request body. Written straight to its final place, so a
        multi-gigabyte Imaris file needs its own size in free disk, not twice that."""
        import asyncio
        import uuid as _uuid

        tmp = store.root / "datasets" / f".incoming-{_uuid.uuid4().hex}"
        try:
            with open(tmp, "wb") as out:
                async for chunk in request.stream():
                    out.write(chunk)

            def write(dest: Path):
                tmp.replace(dest)  # Same filesystem, so this is a rename, not a copy

            return (await asyncio.to_thread(store.add_dataset_file, filename, write)).info()
        finally:
            tmp.unlink(missing_ok=True)

    @r.post("/api/datasets/demo")
    def demo():
        from ..demo import DEMO_CHANNEL_NAMES, write_demo_tiff

        ds = store.add_dataset_file("demo_bone_stack.tif", lambda dest: write_demo_tiff(dest))
        ds.volume.channel_names = list(DEMO_CHANNEL_NAMES)
        store.update_meta(ds.id, reference_channel=2, reference_guessed=False, channel_names=DEMO_CHANNEL_NAMES, default_channel=1)
        return ds.info()

    @r.post("/api/datasets/from-path")
    def from_path(req: PathRequest):
        if not allow_paths:
            raise HTTPException(403, "Opening files by path is turned off on this server. Upload the file, or restart with --allow-paths")
        try:
            return store.add_dataset_path(req.path).info()
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from None

    @r.get("/api/datasets/{ds_id}")
    def get_dataset(ds_id: str):
        return store.get(ds_id).info()

    @r.patch("/api/datasets/{ds_id}")
    def patch_dataset(ds_id: str, req: MetaRequest):
        ds = store.get(ds_id)
        updates = req.model_dump(exclude_unset=True)
        if updates.get("reference_channel") is not None and not 0 <= updates["reference_channel"] < ds.volume.n_channels:
            raise HTTPException(400, "Reference channel out of range")
        if updates.get("voxel_um_override") is not None and min(updates["voxel_um_override"]) <= 0:
            raise HTTPException(400, "Pixel and slice sizes must be positive")
        if "reference_channel" in updates:
            updates["reference_guessed"] = False
        return store.update_meta(ds_id, **updates).info()

    @r.put("/api/datasets/{ds_id}/roi")
    def set_roi(ds_id: str, req: RoiRequest):
        """Sets or clears the region of interest. It applies to every slice of the dataset."""
        ds = store.get(ds_id)
        if req.polygon is not None and len(req.polygon) < 3:
            raise HTTPException(400, "A region needs at least three corners")
        poly = [[float(y), float(x)] for y, x in req.polygon] if req.polygon else None
        store.update_meta(ds_id, roi=poly)
        roi = store.roi_mask(ds_id)
        # Re-clip cached results to the new region, so masks on other channels stay usable
        for key, res in list(ds.results.items()):
            if not hasattr(res, "mask"):
                continue
            full = res.extra.get("unclipped", res.mask)
            res.mask = full & roi if roi is not None else full
            if roi is not None:
                res.extra["unclipped"] = full
            else:
                res.extra.pop("unclipped", None)
        return {"roi": poly, "area_um2": float(roi.sum() * ds.volume.pixel_um[0] * ds.volume.pixel_um[1]) if roi is not None else None}

    @r.delete("/api/datasets/{ds_id}")
    def delete_dataset(ds_id: str):
        store.delete(ds_id)
        return {"ok": True}

    @r.get("/api/datasets/{ds_id}/plane")
    def plane(ds_id: str, c: int = 0, z: int = 0, low: float = 1.0, high: float = 99.5, max_side: int = 1600, gamma: float = 1.0,
              overlay: int | None = None, color: str = "ff00ff"):
        """A slice for display. With overlay, a second channel is added on top in colour."""
        p = store.plane(ds_id, c, z, low, high)
        if overlay is not None and overlay != c:
            hexc = color.lstrip("#")
            rgb = tuple(int(hexc[i:i + 2], 16) for i in (0, 2, 4)) if len(hexc) == 6 else (255, 0, 255)
            png = render.composite_png(p ** gamma if gamma != 1 else p, store.plane(ds_id, overlay, z, low, high), rgb, max_side)
        else:
            png = render.gray_png(p, max_side, gamma)
        return Response(png, media_type="image/png", headers={"Cache-Control": "max-age=3600"})

    @r.get("/api/datasets/{ds_id}/xz")
    def side_view(ds_id: str, c: int = 0, y: int = 0, job_id: str | None = None, low: float = 1.0, high: float = 99.5, max_width: int = 1600,
                  colors: str = ""):
        """A side view through every slice at row y, stretched so z spacing and pixel size match,
        with the mask from a finished stack run on top."""
        from PIL import Image

        ds = store.get(ds_id)
        vol = ds.volume
        xz = bio_normalize(vol.get_xz(c, y), low, high)
        stretch = max(1.0, vol.voxel_um[0] / max(vol.voxel_um[2], 1e-9))
        w = min(vol.width, max_width)
        h = max(1, int(round(vol.n_z * stretch * w / vol.width)))
        rgb = np.repeat((xz * 255).astype(np.uint8)[..., None], 3, -1)
        if job_id:
            job = store.get_job(job_id)
            # A multi-structure run has a label stack; colour each structure with the colours the interface uses
            lpath = job.out_dir / "labels.tif"
            mpath = lpath if lpath.exists() else job.out_dir / "masks.tif"
            zs = job.result.get("summary", {}).get("z_processed", [])
            if mpath.exists() and zs:
                with tifffile.TiffFile(mpath) as tf:
                    rows = np.stack([tf.pages[i].asarray()[y] for i in range(len(zs))])
                if mpath != lpath:
                    rows = (rows > 0).astype(np.uint8)
                full = np.zeros((vol.n_z, vol.width), np.uint8)
                zs_arr = np.asarray(zs)
                step = int(np.median(np.diff(zs_arr))) if len(zs_arr) > 1 else 1
                for z in range(vol.n_z):
                    # Slices between processed ones show the nearest processed slice, within half a step
                    i = int(np.argmin(np.abs(zs_arr - z)))
                    if abs(int(zs_arr[i]) - z) <= step / 2:
                        full[z] = rows[i]
                palette = [tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) for h in (x.strip().lstrip("#") for x in colors.split(",")) if len(h) == 6]
                for k in range(1, int(full.max()) + 1):
                    color = np.array(palette[k - 1] if k - 1 < len(palette) else (0, 220, 255))
                    sel = full == k
                    rgb[sel] = (rgb[sel] * 0.5 + color * 0.5).astype(np.uint8)
        img = Image.fromarray(rgb).resize((w, h), Image.NEAREST if stretch > 1 else Image.BILINEAR)
        return Response(render.to_png_bytes(img), media_type="image/png", headers={"X-Stretch": f"{stretch:.3f}"})

    @r.get("/api/datasets/{ds_id}/reference")
    def reference(ds_id: str, z: int = 0, max_side: int = 1600):
        ref = store.reference_mask(ds_id, z)
        if ref is None:
            raise HTTPException(404, "No reference channel set")
        return Response(render.mask_png(ref, max_side, color=(255, 60, 90), fill_alpha=0), media_type="image/png")

    return r
