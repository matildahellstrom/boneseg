"""Shared state and helpers for the route modules."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import HTTPException

from ..backbone import BACKBONE_LABELS
from ..head import Head, head_from_profile, head_input, segment_with_head
from ..segment import SegmentationSettings, prototypes, segment_with_prototypes
from ..store import Store


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

    def microsam_path(self, ds_id: str, c: int) -> Path:
        """Where micro-SAM fine-tuned on this dataset's labels for channel c is kept (boneseg/microsam.py)."""
        return self.models_dir / f"microsam_{ds_id}_c{c}.pt"

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

    def head_embedding(self, ds_id: str, c: int, z: int, settings: SegmentationSettings, head: Head):
        """The embedding the learned model expects for slice z, with its neighbours' features for a z-context model.
        Call with compute_lock held."""
        vol = self.store.get(ds_id).volume
        return head_input(head, lambda zz: self.store.embedding(ds_id, c, zz, settings), z, vol.n_z, vol.voxel_um[0])

    def last_result(self, ds_id, c, z):
        res = self.store.get(ds_id).results.get((c, z))
        if res is None:
            raise HTTPException(404, "Segment this slice first")
        return res

