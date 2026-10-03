"""Server-side state: datasets on disk, cached planes and embeddings, profiles and background jobs."""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import io as bio
from .backbone import get_backbone
from .segment import Embedding, Profile, SegmentationSettings, embed_image


class LRU:
    """Least-recently-used cache, bounded by item count and optionally by total size in bytes."""

    def __init__(self, max_items: int, max_bytes: int | None = None, size_of: Callable | None = None):
        self.max_items = max_items
        self.max_bytes = max_bytes
        self.size_of = size_of or (lambda v: 0)
        self.data: OrderedDict = OrderedDict()
        self.bytes = 0
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.data:
                self.data.move_to_end(key)
                return self.data[key]
        return None

    def __contains__(self, key):
        with self.lock:
            return key in self.data

    def put(self, key, value):
        with self.lock:
            if key in self.data:
                self.bytes -= self.size_of(self.data.pop(key))
            self.data[key] = value
            self.bytes += self.size_of(value)
            while len(self.data) > 1 and (len(self.data) > self.max_items or (self.max_bytes and self.bytes > self.max_bytes)):
                _, old = self.data.popitem(last=False)
                self.bytes -= self.size_of(old)

    def drop(self, pred: Callable):
        with self.lock:
            for k in [k for k in self.data if pred(k)]:
                self.bytes -= self.size_of(self.data.pop(k))


def _embedding_bytes(emb) -> int:
    return int(emb.grid.numel() * emb.grid.element_size())


@dataclass
class Dataset:
    id: str
    path: Path
    volume: bio.Volume
    created: float
    meta: dict = field(default_factory=dict)  # User choices, such as the reference mask channel
    results: dict = field(default_factory=dict)  # (channel, z) -> last SegmentationResult

    def info(self) -> dict:
        return {"id": self.id, "created": self.created, **self.volume.info(), **self.meta}

    def apply_meta(self):
        """Applies user overrides, such as a pixel size typed in for files without one."""
        vox = self.meta.get("voxel_um_override")
        if vox:
            self.volume.voxel_um = tuple(float(v) for v in vox)
            self.volume.voxel_size_known = True


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"  # queued, running, done, failed, cancelled
    progress: float = 0.0
    message: str = ""
    result: dict = field(default_factory=dict)
    error: str = ""
    created: float = field(default_factory=time.time)
    cancel: threading.Event = field(default_factory=threading.Event)
    out_dir: Path | None = None
    meta: dict = field(default_factory=dict)  # Set when the job starts, never overwritten by the result

    def info(self) -> dict:
        return {"id": self.id, "kind": self.kind, "status": self.status, "progress": round(self.progress, 4),
                "message": self.message, "result": self.result, "error": self.error, "created": self.created, "meta": self.meta}


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "datasets").mkdir(parents=True, exist_ok=True)
        (self.root / "profiles").mkdir(parents=True, exist_ok=True)
        (self.root / "jobs").mkdir(parents=True, exist_ok=True)
        self.datasets: dict[str, Dataset] = {}
        self.planes = LRU(64)
        # Features are the expensive part; the budget is in bytes because Giant features are four times Small ones
        self.embeddings = LRU(256, max_bytes=int(float(os.environ.get("BONESEG_CACHE_GB", "2")) * 1e9), size_of=_embedding_bytes)
        self._prefetching: set = set()
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        # One model computation at a time: GPU backends such as Apple MPS crash when used from several threads at once
        self.compute_lock = threading.RLock()
        self._load_existing()

    # Datasets ---------------------------------------------------------------------------------
    def _load_existing(self):
        for d in sorted((self.root / "datasets").iterdir()):
            meta_path = d / "dataset.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text())
                path = Path(meta["path"])
                if not path.is_absolute():
                    path = d / path
                vol = bio.load_volume(path)
                ds = Dataset(id=d.name, path=path, volume=vol, created=meta.get("created", 0), meta=meta.get("user", {}))
                ds.apply_meta()
                self.datasets[d.name] = ds
            except Exception:
                traceback.print_exc()

    def _save_meta(self, ds: Dataset):
        d = self.root / "datasets" / ds.id
        path = ds.path.name if ds.path.parent == d else str(ds.path)
        (d / "dataset.json").write_text(json.dumps({"path": path, "created": ds.created, "user": ds.meta}, indent=1))

    def add_dataset_file(self, src_name: str, write: Callable[[Path], None]) -> Dataset:
        """Creates a dataset whose file is written by write(dest_path)."""
        ext = Path(src_name).suffix.lower()
        if ext not in bio.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type {ext or '(none)'}. Supported: {', '.join(sorted(bio.SUPPORTED_EXTENSIONS))}")
        ds_id = uuid.uuid4().hex[:10]
        d = self.root / "datasets" / ds_id
        d.mkdir(parents=True)
        dest = d / Path(src_name).name
        try:
            write(dest)
            vol = bio.load_volume(dest)
        except Exception:
            shutil.rmtree(d, ignore_errors=True)
            raise
        ds = Dataset(id=ds_id, path=dest, volume=vol, created=time.time(), meta={})
        ds.meta.update(self._guess_reference(vol))
        self.datasets[ds_id] = ds
        self._save_meta(ds)
        return ds

    def add_dataset_path(self, path: str | Path) -> Dataset:
        """Registers a file in place, which avoids copying multi-gigabyte Imaris files."""
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"No such file: {path}")
        vol = bio.load_volume(path)
        ds_id = uuid.uuid4().hex[:10]
        (self.root / "datasets" / ds_id).mkdir(parents=True)
        ds = Dataset(id=ds_id, path=path, volume=vol, created=time.time(), meta=self._guess_reference(vol))
        self.datasets[ds_id] = ds
        self._save_meta(ds)
        return ds

    @staticmethod
    def _guess_reference(vol: bio.Volume) -> dict:
        """Liu-style Imaris exports store an expert segmentation as the last channel, and the names often say so."""
        for i, n in enumerate(vol.channel_names):
            if any(k in n.lower() for k in ("segment", "mask", "label", "surface")):
                return {"reference_channel": i, "reference_guessed": True}
        return {"reference_channel": None}

    def update_meta(self, ds_id: str, **kw):
        ds = self.get(ds_id)
        ds.meta.update(kw)
        ds.apply_meta()
        self._save_meta(ds)
        return ds

    def get(self, ds_id: str) -> Dataset:
        if ds_id not in self.datasets:
            raise KeyError(f"Unknown dataset {ds_id}")
        return self.datasets[ds_id]

    def delete(self, ds_id: str):
        ds = self.get(ds_id)
        d = self.root / "datasets" / ds_id
        shutil.rmtree(d, ignore_errors=True)
        del self.datasets[ds_id]
        self.planes.drop(lambda k: k[0] == ds_id)
        self.embeddings.drop(lambda k: k[0] == ds_id)

    # Planes and embeddings ----------------------------------------------------------------------
    def raw_plane(self, ds_id: str, c: int, z: int) -> np.ndarray:
        key = (ds_id, "raw", c, z)
        p = self.planes.get(key)
        if p is None:
            p = np.asarray(self.get(ds_id).volume.get_plane(c, z))
            self.planes.put(key, p)
        return p

    def plane(self, ds_id: str, c: int, z: int, low: float = 1.0, high: float = 99.5) -> np.ndarray:
        key = (ds_id, "norm", c, z, low, high)
        p = self.planes.get(key)
        if p is None:
            p = bio.normalize_plane(self.raw_plane(ds_id, c, z), low, high)
            self.planes.put(key, p)
        return p

    def embedding(self, ds_id: str, c: int, z: int, s: SegmentationSettings) -> Embedding:
        key = self.embedding_key(ds_id, c, z, s)
        emb = self.embeddings.get(key)
        if emb is not None:
            return emb
        # One lock for all model work. Callers may already hold it (it is re-entrant); a second,
        # per-slice lock here once deadlocked against the background prefetch.
        with self.compute_lock:
            emb = self.embeddings.get(key)
            if emb is None:
                emb = embed_image(get_backbone(s.backbone), self.plane(ds_id, c, z, s.clip_low, s.clip_high), s)
                self.embeddings.put(key, emb)
        return emb

    def roi_mask(self, ds_id: str) -> np.ndarray | None:
        """The region of interest as a full-resolution mask, or None for the whole image."""
        ds = self.get(ds_id)
        poly = ds.meta.get("roi")
        if not poly or len(poly) < 3:
            return None
        key = (ds_id, "roi", tuple(map(tuple, poly)))
        m = self.planes.get(key)
        if m is None:
            from skimage.draw import polygon

            m = np.zeros((ds.volume.height, ds.volume.width), bool)
            pts = np.asarray(poly, float)
            rr, cc = polygon(pts[:, 0], pts[:, 1], m.shape)
            m[rr, cc] = True
            self.planes.put(key, m)
        return m

    def embedding_key(self, ds_id, c, z, s: SegmentationSettings):
        return (ds_id, c, z, s.backbone, s.vit_size, s.layer_from_end, s.clip_low, s.clip_high)

    def prefetch(self, ds_id: str, c: int, zs: list[int], s: SegmentationSettings):
        """Computes features for the given slices in the background, so moving to them is instant.
        Takes the model lock per slice, so interactive requests can slip in between."""
        n_z = self.get(ds_id).volume.n_z
        todo = [z for z in zs if 0 <= z < n_z and self.embedding_key(ds_id, c, z, s) not in self.embeddings
                and (ds_id, c, z) not in self._prefetching]
        if not todo:
            return

        def run():
            for z in todo:
                self._prefetching.add((ds_id, c, z))
                try:
                    if ds_id in self.datasets:
                        self.embedding(ds_id, c, z, s)
                except Exception:
                    traceback.print_exc()
                finally:
                    self._prefetching.discard((ds_id, c, z))

        threading.Thread(target=run, daemon=True).start()

    def reference_mask(self, ds_id: str, z: int, c: int | None = None) -> np.ndarray | None:
        """The expert mask for slice z: the reference channel if one is set, otherwise a label the user
        saved for this slice and channel."""
        ref = self.get(ds_id).meta.get("reference_channel")
        if ref is not None:
            return self.raw_plane(ds_id, int(ref), z) > 0
        if c is not None:
            return self.load_label(ds_id, c, z)
        return None

    # Annotations: clicks, corrected label masks and trained heads -------------------------------
    def ds_dir(self, ds_id: str) -> Path:
        self.get(ds_id)
        return self.root / "datasets" / ds_id

    def get_annotations(self, ds_id: str) -> dict:
        p = self.ds_dir(ds_id) / "annotations.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def set_annotation(self, ds_id: str, c: int, z: int, pos: list, neg: list) -> dict:
        ann = self.get_annotations(ds_id)
        key = f"{int(c)}:{int(z)}"
        if pos or neg:
            ann[key] = {"pos": [[int(round(a)), int(round(b))] for a, b in pos], "neg": [[int(round(a)), int(round(b))] for a, b in neg]}
        else:
            ann.pop(key, None)
        (self.ds_dir(ds_id) / "annotations.json").write_text(json.dumps(ann))
        return ann

    def label_path(self, ds_id: str, c: int, z: int) -> Path:
        d = self.ds_dir(ds_id) / "labels"
        d.mkdir(exist_ok=True)
        return d / f"c{int(c)}_z{int(z)}.png"

    def save_label(self, ds_id: str, c: int, z: int, mask: np.ndarray):
        from PIL import Image

        vol = self.get(ds_id).volume
        if mask.shape != (vol.height, vol.width):
            mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((vol.width, vol.height), Image.NEAREST)) > 127
        Image.fromarray(mask.astype(np.uint8) * 255).save(self.label_path(ds_id, c, z))

    def load_label(self, ds_id: str, c: int, z: int) -> np.ndarray | None:
        from PIL import Image

        p = self.label_path(ds_id, c, z)
        return np.asarray(Image.open(p)) > 127 if p.exists() else None

    def list_labels(self, ds_id: str) -> list[dict]:
        out = []
        for p in sorted((self.ds_dir(ds_id) / "labels").glob("c*_z*.png")) if (self.ds_dir(ds_id) / "labels").exists() else []:
            c, z = p.stem[1:].split("_z")
            out.append({"channel": int(c), "z": int(z)})
        return sorted(out, key=lambda d: (d["channel"], d["z"]))

    def delete_label(self, ds_id: str, c: int, z: int):
        self.label_path(ds_id, c, z).unlink(missing_ok=True)

    def head_path(self, ds_id: str, c: int) -> Path:
        d = self.ds_dir(ds_id) / "heads"
        d.mkdir(exist_ok=True)
        return d / f"c{int(c)}.pt"

    # Profiles ---------------------------------------------------------------------------------
    def profile_path(self, pid: str) -> Path:
        if not pid.replace("-", "").replace("_", "").isalnum():
            raise KeyError(f"Invalid profile id {pid}")
        return self.root / "profiles" / f"{pid}.npz"

    def save_profile(self, profile: Profile) -> str:
        base = "".join(ch if ch.isalnum() else "-" for ch in profile.name.lower()).strip("-")[:40] or "profile"
        pid = f"{base}-{uuid.uuid4().hex[:6]}"
        profile.save(self.profile_path(pid))
        return pid

    def list_profiles(self) -> list[dict]:
        out = []
        for p in sorted((self.root / "profiles").glob("*.npz")):
            try:
                prof = Profile.load(p)
                out.append({"id": p.stem, "name": prof.name, "backbone": prof.backbone, "description": prof.description, "kind": prof.kind,
                            "vit_size": prof.head["vit_size"] if prof.head else None,
                            "source": prof.source, "n_pos": int(len(prof.pos)), "n_neg": int(len(prof.neg)),
                            "settings": prof.settings})
            except Exception:
                traceback.print_exc()
        return out

    def load_profile(self, pid: str) -> Profile:
        p = self.profile_path(pid)
        if not p.exists():
            raise KeyError(f"Unknown profile {pid}")
        return Profile.load(p)

    def delete_profile(self, pid: str):
        self.profile_path(pid).unlink(missing_ok=True)

    # Jobs -------------------------------------------------------------------------------------
    def start_job(self, kind: str, fn: Callable[[Job], dict], meta: dict | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex[:10], kind=kind, meta=meta or {})
        job.out_dir = self.root / "jobs" / job.id
        job.out_dir.mkdir(parents=True, exist_ok=True)
        self.jobs[job.id] = job

        def run():
            job.status = "running"
            try:
                job.result = fn(job) or {}
                job.status = "cancelled" if job.cancel.is_set() else "done"
                job.progress = 1.0 if job.status == "done" else job.progress
            except Exception as e:
                traceback.print_exc()
                job.status, job.error = "failed", f"{type(e).__name__}: {e}"

        threading.Thread(target=run, daemon=True).start()
        return job

    def latest_job(self, ds_id: str, kind: str = "stack") -> Job | None:
        done = [j for j in self.jobs.values() if j.kind == kind and j.status == "done" and j.meta.get("dataset_id") == ds_id]
        return max(done, key=lambda j: j.created) if done else None

    def get_job(self, job_id: str) -> Job:
        if job_id not in self.jobs:
            raise KeyError(f"Unknown job {job_id}")
        return self.jobs[job_id]
