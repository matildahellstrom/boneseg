"""Batch processing: apply a saved profile to many files without the browser."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from . import io as bio
from .backbone import get_backbone
from .pipeline import StackRequest, run_stack
from .segment import Profile, SegmentationSettings, embed_image


def resolve_profile(spec: str, data_dir: str | Path) -> Profile:
    """A profile given as a path to its .npz file, or as its id in the app's data folder."""
    p = Path(spec)
    if p.suffix == ".npz" and p.exists():
        return Profile.load(p)
    candidate = Path(data_dir) / "profiles" / f"{spec}.npz"
    if candidate.exists():
        return Profile.load(candidate)
    matches = sorted((Path(data_dir) / "profiles").glob(f"{spec}*.npz"))
    if len(matches) == 1:
        return Profile.load(matches[0])
    raise FileNotFoundError(f"No profile '{spec}'. Give a path to a .npz file or an id from the app")


def run_batch(files: list[str | Path], profile: Profile, channel: int, out_dir: str | Path, z_start: int = 0,
              z_end: int | None = None, z_step: int = 1, reference: int | None = None, log=print) -> pd.DataFrame:
    """Segments every file with the profile. Writes one folder per file and a combined summary.csv."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = SegmentationSettings.from_dict({**profile.settings, "backbone": profile.backbone, "layer_from_end": profile.layer_from_end})
    head = None
    if profile.head:
        from .head import head_from_profile

        head = head_from_profile(profile)
        settings = SegmentationSettings.from_dict({**settings.to_dict(), "vit_size": head.vit_size})
    backbone = get_backbone(settings.backbone)
    pos, neg = profile.tensors()
    multi = profile.multi_model() if profile.structures else None
    rows = []
    for f in files:
        f = Path(f)
        t0 = time.time()
        try:
            vol = bio.load_volume(f)
            if channel >= vol.n_channels:
                raise ValueError(f"channel {channel} does not exist, the file has {vol.n_channels}")
            last = vol.n_z - 1 if z_end is None else min(z_end, vol.n_z - 1)
            zs = list(range(max(0, z_start), last + 1, max(1, z_step)))
            target = out_dir / f.stem
            target.mkdir(exist_ok=True)
            planes = {}

            def image(z):
                if z not in planes:
                    planes.clear()  # Keep memory flat on large stacks
                    planes[z] = bio.normalize_plane(vol.get_plane(channel, z), settings.clip_low, settings.clip_high)
                return planes[z]

            if multi is not None:
                from .pipeline import run_stack_multi

                out = run_stack_multi(zs, None, multi, settings, get_embedding=lambda z: embed_image(backbone, image(z), settings),
                                      get_image=image, voxel_um=vol.voxel_um, out_dir=target)
                s = out["summary"]
                row = {"file": f.name, "status": "ok", "seconds": round(time.time() - t0, 1), "n_slices": s["n_slices"]}
                for name, st in s["structures"].items():
                    row.update({f"{name}_volume_um3": st.get("volume_um3"), f"{name}_n_objects_3d": st.get("n_objects_3d"),
                                f"{name}_mean_area_fraction": st.get("mean_area_fraction")})
                row.update({k: v for k, v in (s.get("histomorphometry") or {}).items() if k not in ("bone", "cells")})
                log(f"{f.name}: {s['n_slices']} slices, structures {', '.join(s['structures'])} ({row['seconds']} s)")
                rows.append(row)
                continue
            out = run_stack(StackRequest(z_list=zs, ref_z=None), pos, neg, settings, profile.raw_threshold,
                            get_embedding=lambda z: embed_image(backbone, image(z), settings), get_image=image,
                            get_reference=(lambda z: vol.get_plane(reference, z) > 0) if reference is not None else (lambda z: None),
                            voxel_um=vol.voxel_um, out_dir=target,
                            progress=lambda p, m: None, head=head)
            summary = out["summary"]
            row = {"file": f.name, "status": "ok", "seconds": round(time.time() - t0, 1), **{k: v for k, v in summary.items() if k != "z_processed"}}
            log(f"{f.name}: {summary.get('n_slices', 0)} slices, volume {summary.get('volume_um3', 0):.0f} um3, "
                f"{summary.get('n_objects_3d', 0)} objects in 3D ({row['seconds']} s)")
        except Exception as e:  # One bad file should not stop the batch
            row = {"file": f.name, "status": f"failed: {type(e).__name__}: {e}"}
            log(f"{f.name}: FAILED {e}")
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "profile.json").write_text(json.dumps({"name": profile.name, "backbone": profile.backbone, "source": profile.source,
                                                       "raw_threshold": profile.raw_threshold, "settings": settings.to_dict()}, indent=1))
    return df
