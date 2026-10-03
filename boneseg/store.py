"""Server-side state: datasets on disk, cached planes and embeddings, profiles and background jobs."""
from __future__ import annotations

import json
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
    def __init__(self, max_items: int):
        self.max_items = max_items
        self.data: OrderedDict = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.data:
                self.data.move_to_end(key)
                return self.data[key]
        return None

    def put(self, key, value):
        with self.lock:
            self.data[key] = value
            self.data.move_to_end(key)
            while len(self.data) > self.max_items:
                self.data.popitem(last=False)

    def drop(self, pred: Callable):
        with self.lock:
            for k in [k for k in self.data if pred(k)]:
                del self.data[k]


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

    def info(self) -> dict:
        return {"id": self.id, "kind": self.kind, "status": self.status, "progress": round(self.progress, 4),
                "message": self.message, "result": self.result, "error": self.error, "created": self.created}


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "datasets").mkdir(parents=True, exist_ok=True)
        (self.root / "profiles").mkdir(parents=True, exist_ok=True)
        (self.root / "jobs").mkdir(parents=True, exist_ok=True)
        self.datasets: dict[str, Dataset] = {}
        self.planes = LRU(64)
        self.embeddings = LRU(48)
        self.jobs: dict[str, Job] = {}
        self._embed_locks: dict = {}
        self._lock = threading.Lock()
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
                self.datasets[d.name] = Dataset(id=d.name, path=path, volume=vol, created=meta.get("created", 0),
                                                meta=meta.get("user", {}))
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
        key = (ds_id, c, z, s.backbone, s.vit_size, s.layer_from_end, s.clip_low, s.clip_high)
        emb = self.embeddings.get(key)
        if emb is not None:
            return emb
        with self._lock:
            lock = self._embed_locks.setdefault(key, threading.Lock())
        with lock:
            emb = self.embeddings.get(key)
            if emb is None:
                emb = embed_image(get_backbone(s.backbone), self.plane(ds_id, c, z, s.clip_low, s.clip_high), s)
                self.embeddings.put(key, emb)
        return emb

    def reference_mask(self, ds_id: str, z: int) -> np.ndarray | None:
        ref = self.get(ds_id).meta.get("reference_channel")
        if ref is None:
            return None
        return self.raw_plane(ds_id, int(ref), z) > 0

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
                out.append({"id": p.stem, "name": prof.name, "backbone": prof.backbone, "description": prof.description,
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
    def start_job(self, kind: str, fn: Callable[[Job], dict]) -> Job:
        job = Job(id=uuid.uuid4().hex[:10], kind=kind)
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

    def get_job(self, job_id: str) -> Job:
        if job_id not in self.jobs:
            raise KeyError(f"Unknown job {job_id}")
        return self.jobs[job_id]
