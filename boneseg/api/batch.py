"""Running a profile on every sample from the app, so the results feed straight into Compare samples."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..head import head_from_profile
from ..pipeline import StackRequest, run_stack, run_stack_multi
from ..segment import SegmentationSettings
from .context import AppContext


class BatchRequest(BaseModel):
    profile_id: str
    channel: int | None = 0  # None uses each sample's own channel, the one last chosen for it in the app
    dataset_ids: list[str] = Field(default_factory=list)  # Empty means every dataset
    z_start: int = 0
    z_end: int | None = None
    z_step: int = 1
    settings: dict = Field(default_factory=dict)


def router(ctx: AppContext) -> APIRouter:
    r = APIRouter()
    store = ctx.store

    @r.post("/api/batch")
    def batch(req: BatchRequest):
        prof = store.load_profile(req.profile_id)
        base = SegmentationSettings.from_dict(req.settings).to_dict()
        # The profile decides everything that must match its features
        base.update(backbone=prof.backbone, layer_from_end=prof.layer_from_end)
        if prof.head:
            base["vit_size"] = prof.head["vit_size"]
        settings = SegmentationSettings.from_dict(base)
        ids = req.dataset_ids or sorted(store.datasets, key=lambda i: store.datasets[i].volume.name)
        everything = [store.get(i) for i in ids]

        def channel_of(ds):
            c = req.channel if req.channel is not None else ds.meta.get("default_channel")
            if c is None and ds.volume.rgb_channel is not None:
                c = ds.volume.rgb_channel   # Colour images: their colour channel, as the app opens them
            if c is None and ds.volume.n_channels == 1:
                c = 0                       # A single channel needs no choice
            return c if c is not None and 0 <= c < ds.volume.n_channels else None

        # Samples without the channel (or without a chosen channel) are skipped and reported, not fatal
        targets = [ds for ds in everything if channel_of(ds) is not None]
        skipped = [ds for ds in everything if channel_of(ds) is None]
        if not targets:
            raise ValueError("No sample has that channel" if req.channel is not None else "No sample has a chosen channel yet; open each sample and pick its channel")
        head = head_from_profile(prof) if prof.head else None
        multi = prof.multi_model() if prof.structures else None
        pos, neg = prof.tensors()

        def work(job):
            rows = []
            for i, ds in enumerate(targets):
                if job.cancel.is_set():
                    break
                job.message = f"{ds.volume.name} ({i + 1}/{len(targets)})"
                last = ds.volume.n_z - 1 if req.z_end is None else min(req.z_end, ds.volume.n_z - 1)
                zs = list(range(max(0, req.z_start), last + 1, max(1, req.z_step)))
                c, did = channel_of(ds), ds.id
                emb = lambda z, did=did, c=c: store.embedding(did, c, z, settings)  # noqa: E731
                img = lambda z, did=did, c=c: store.plane(did, c, z, settings.clip_low, settings.clip_high)  # noqa: E731
                frac0, span = i / len(targets), 1 / len(targets)
                prog = lambda p, m, f0=frac0: setattr(job, "progress", f0 + p * span)  # noqa: E731

                def run(out_dir, ds=ds, zs=zs, emb=emb, img=img, prog=prog, did=did, c=c):
                    if head is not None and head.names:
                        from types import SimpleNamespace

                        from ..head import labels_with_head

                        return run_stack_multi(zs, None, SimpleNamespace(names=list(head.names)), settings, get_embedding=emb, get_image=img,
                                               voxel_um=ds.volume.voxel_um, out_dir=out_dir, progress=prog, cancelled=job.cancel.is_set,
                                               lock=store.compute_lock, roi=store.roi_mask(did),
                                               labeler=lambda e, ds=ds: labels_with_head(head, e, settings, ds.volume.pixel_um))
                    if multi is not None:
                        return run_stack_multi(zs, None, multi, settings, get_embedding=emb, get_image=img, voxel_um=ds.volume.voxel_um,
                                               out_dir=out_dir, progress=prog, cancelled=job.cancel.is_set, lock=store.compute_lock,
                                               roi=store.roi_mask(did))
                    return run_stack(StackRequest(z_list=zs, ref_z=None), pos, neg, settings, prof.raw_threshold, get_embedding=emb,
                                     get_image=img, get_reference=lambda z: store.reference_mask(did, z, c), voxel_um=ds.volume.voxel_um,
                                     out_dir=out_dir, progress=prog, cancelled=job.cancel.is_set, lock=store.compute_lock, head=head,
                                     roi=store.roi_mask(did), n_z=ds.volume.n_z)

                child = store.record_job("stack", {"dataset_id": did, "channel": c, "method": "profile", "profile": prof.name,
                                                   "batch": job.id, "n_slices": len(zs)}, run, cancelled=job.cancel.is_set)
                rows.append({"dataset_id": did, "name": ds.volume.name, "job_id": child.id, "status": child.status, "error": child.error})
            rows += [{"dataset_id": ds.id, "name": ds.volume.name, "job_id": None, "status": "skipped",
                      "error": f"no channel {req.channel}" if req.channel is not None else "no channel chosen"} for ds in skipped]
            return {"samples": rows, "profile": prof.name}

        job = store.start_job("batch", work, meta={"profile": prof.name, "n_datasets": len(targets), "channel": req.channel})
        job.meta["channels"] = {ds.id: channel_of(ds) for ds in targets}
        return job.info()

    return r
