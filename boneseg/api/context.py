"""Shared state and helpers for the route modules."""
from __future__ import annotations

import numpy as np
from fastapi import HTTPException

from ..backbone import BACKBONE_LABELS
from ..head import Head, head_from_profile, segment_with_head
from ..segment import SegmentationSettings, prototypes, segment_with_prototypes
from ..store import Store
from .models import MaskSpec


def attachment(filename: str) -> dict:
    """A Content-Disposition header with a file name reduced to safe characters, so names from files
    cannot break the header with quotes, line breaks or non-ASCII text."""
    import re

    stem, dot, ext = filename.rpartition(".")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem or filename).strip("._") or "download"
    suffix = "." + re.sub(r"[^A-Za-z0-9]+", "", ext) if stem else ""
    return {"Content-Disposition": f'attachment; filename="{safe[:150]}{suffix}"'}


class AppContext:
    def __init__(self, store: Store, allow_paths: bool):
        self.store = store
        self.allow_paths = allow_paths
        self.models_dir = store.root / "models"   # Fine-tuned backbones; listed as backbones by /api/health

    def prototypes_for(self, ds_id, channel, z, pos_pts, neg_pts, profile_id, settings):
        """Prototypes from clicks, a profile, or a profile refined by clicks.
        Also returns the profile's calibrated raw threshold, if any."""
        if profile_id:
            prof = self.store.load_profile(profile_id)
            if prof.head:
                raise ValueError("This profile holds a learned model; it is used without clicks")
            if prof.structures:
                raise ValueError("This profile holds several structures; use it with several structures")
            if prof.backbone != settings.backbone or prof.layer_from_end != settings.layer_from_end:
                raise ValueError(f"Profile '{prof.name}' was made with {BACKBONE_LABELS.get(prof.backbone, prof.backbone)} "
                                 f"(block {prof.layer_from_end} from the end). Switch to that backbone to use it.")
            pos, neg = prof.tensors()
            if pos_pts or neg_pts:  # Clicks refine the profile
                import torch

                emb = self.store.embedding(ds_id, channel, z, settings)
                pos = torch.cat([pos.to(emb.grid.device), prototypes(emb, pos_pts)]) if pos_pts else pos
                neg = torch.cat([neg.to(emb.grid.device), prototypes(emb, neg_pts)]) if neg_pts else neg
                # The stored threshold belongs to the stored prototypes only
                return pos, neg, None
            return pos, neg, prof.raw_threshold
        if not pos_pts:
            raise ValueError("Add at least one positive click, or pick a profile")
        emb = self.store.embedding(ds_id, channel, z, settings)
        return prototypes(emb, pos_pts), prototypes(emb, neg_pts), None

    def learned_profile_head(self, profile_id, settings) -> Head | None:
        """The learned model stored in a profile, checked against the current settings, or None."""
        if not profile_id:
            return None
        prof = self.store.load_profile(profile_id)
        if not prof.head:
            return None
        head = head_from_profile(prof)
        if not head.compatible(settings):
            raise ValueError(f"Profile '{prof.name}' holds a model trained with {BACKBONE_LABELS.get(head.backbone, head.backbone)} at "
                             f"{head.vit_size} px, block {head.layer_from_end} from the end. Switch to those settings to use it")
        return head

    def multi_profile_model(self, profile_id, settings):
        """The structures stored in a multi-structure profile, checked against the current settings, or None."""
        if not profile_id:
            return None, None
        prof = self.store.load_profile(profile_id)
        if not prof.structures:
            return None, None
        if prof.backbone != settings.backbone or prof.layer_from_end != settings.layer_from_end:
            raise ValueError(f"Profile '{prof.name}' was made with {BACKBONE_LABELS.get(prof.backbone, prof.backbone)}. Switch to that backbone to use it.")
        return prof.multi_model(), prof

    def load_head(self, ds_id, channel, settings) -> Head:
        path = self.store.head_path(ds_id, channel)
        if not path.exists():
            raise ValueError("No learned model for this channel yet. Save corrected masks as labels and train one first")
        head = Head.load(path)
        if not head.compatible(settings):
            raise ValueError(f"The learned model was trained with {BACKBONE_LABELS.get(head.backbone, head.backbone)} at "
                             f"{head.vit_size} px, block {head.layer_from_end} from the end. Switch back or retrain it")
        return head

    def last_result(self, ds_id, c, z):
        res = self.store.get(ds_id).results.get((c, z))
        if res is None:
            raise HTTPException(404, "Segment this slice first")
        return res

    def mask_from_spec(self, ds_id: str, z: int, spec: MaskSpec, settings: SegmentationSettings) -> np.ndarray:
        ds = self.store.get(ds_id)
        if spec.source == "current":
            res = ds.results.get((spec.channel, z))
            if res is None:
                raise ValueError(f"Segment channel {spec.channel} on slice {z} first, or pick another source")
            return res.extra.get("unclipped", res.mask)
        if spec.source.startswith("structure:"):
            # One structure from the last multi-structure result on this channel and slice
            res = ds.results.get(("multi", spec.channel, z))
            if res is None:
                raise ValueError(f"Segment several structures on channel {spec.channel}, slice {z} first")
            k = int(spec.source.split(":", 1)[1])
            if not 0 <= k < len(res["names"]):
                raise ValueError("That structure is not in the last result")
            return res["labels"] == k + 1
        if spec.source == "label":
            m = self.store.load_label(ds_id, spec.channel, z)
            if m is None:
                raise ValueError(f"No saved label for channel {spec.channel} on slice {z}")
            return m
        if spec.source == "reference":
            ref = ds.meta.get("reference_channel")
            if ref is None:
                raise ValueError("No reference channel is set")
            return self.store.raw_plane(ds_id, int(ref), z) > 0
        with self.store.compute_lock:
            emb = self.store.embedding(ds_id, spec.channel, z, settings)
            if spec.source == "learned":
                return segment_with_head(self.load_head(ds_id, spec.channel, settings), emb, settings, ds.volume.pixel_um).mask
            if spec.source == "profile":
                if not spec.profile_id:
                    raise ValueError("Pick a profile")
                ph = self.learned_profile_head(spec.profile_id, settings)
                if ph is not None:
                    return segment_with_head(ph, emb, settings, ds.volume.pixel_um).mask
                pos, neg, thr = self.prototypes_for(ds_id, spec.channel, z, [], [], spec.profile_id, settings)
                return segment_with_prototypes(emb, pos, neg, settings, ds.volume.pixel_um, raw_threshold=thr).mask
        raise ValueError(f"Unknown mask source {spec.source}")
