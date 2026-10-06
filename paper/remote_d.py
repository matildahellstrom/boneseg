"""Liu sample D (24 GB) read from Kaggle without downloading it.

The Imaris file stores each channel in gzip-compressed chunks of 32 x 128 x 128 voxels, so one slice costs its whole
32-slice block. `remote_volume()` gives a boneseg Volume over HTTP range requests with a large chunk cache, so
consecutive slices (a 3D block) are cheap. `extract_slices()` picks the 2D evaluation slices two per block, from ten
blocks spread over the central 80% of the stack, and stores them locally (data/liudata/D_slices.npz), so the
evaluation reads about 3 GB instead of 24 GB.
"""
from __future__ import annotations

import time

import numpy as np

from common import ROOT  # noqa: F401  (puts the repository on the path)
from remote_file import RangeFile, kaggle_file_url
from boneseg import io as bio

DATASET = "matildahellstrom/liudata"
FILE = "15-15-06_IC 2_Trap_SOST_col1_Blaze crop2 segmentation quantified.ims"
OUT = ROOT / "data" / "liudata" / "D_slices.npz"
IMAGE_CH = 4
CHUNK_Z = 32


def _attr(attrs, key):
    v = attrs.get(key)
    return b"".join(v).decode() if v is not None else None


class RemoteIms:
    """Reads whole 32-slice blocks of sample D: the chunk index comes from HDF5 once per channel, then a block's
    chunks are downloaded in parallel and inflated here (the file's only filter is deflate), since HDF5 itself reads
    one chunk at a time and each request takes about a second."""

    def __init__(self, threads: int = 32):
        import h5py
        self.rf = RangeFile(lambda: kaggle_file_url(DATASET, FILE), max_blocks=1024)
        self.h = h5py.File(self.rf, "r")
        info = self.h["DataSetInfo/Image"].attrs
        self.nx, self.ny, self.nz = int(_attr(info, "X")), int(_attr(info, "Y")), int(_attr(info, "Z"))
        ext = [float(_attr(info, f"ExtMax{i}")) - float(_attr(info, f"ExtMin{i}")) for i in range(3)]
        self.voxel_um = (ext[2] / self.nz, ext[1] / self.ny, ext[0] / self.nx)
        tp = self.h["DataSet/ResolutionLevel 0/TimePoint 0"]
        self.n_channels = len(tp.keys())
        self.ds = [tp[f"Channel {c}"]["Data"] for c in range(self.n_channels)]
        self.shape, self.chunks, self.dtype = self.ds[0].shape, self.ds[0].chunks, self.ds[0].dtype
        self.index = {}
        self.threads = threads
        self.layers = {}

    def _index(self, c):
        """Chunk offsets of channel c, cached in data/liudata/D_chunk_index_<c>.pkl (building it takes minutes)."""
        if c not in self.index:
            import pickle
            cache = OUT.parent / f"D_chunk_index_{c}.pkl"
            if cache.exists():
                self.index[c] = pickle.loads(cache.read_bytes())
            else:
                idx = {}
                self.ds[c].id.chunk_iter(lambda info: idx.__setitem__(tuple(info.chunk_offset), (info.byte_offset, info.size)))
                cache.write_bytes(pickle.dumps(idx))
                self.index[c] = idx
        return self.index[c]

    def layer(self, c: int, cz: int) -> np.ndarray:
        """The 32 slices starting at cz (a multiple of 32) of channel c, cropped to the image size."""
        key = (c, cz)
        if key in self.layers:
            return self.layers[key]
        import zlib
        from concurrent.futures import ThreadPoolExecutor
        import requests
        idx = self._index(c)
        zc, yc, xc = self.chunks
        coords = [(cz, y, x) for y in range(0, self.shape[1], yc) for x in range(0, self.shape[2], xc)]
        out = np.zeros((min(zc, self.shape[0] - cz), self.shape[1], self.shape[2]), self.dtype)
        url = [self.rf.url]

        def fetch(coord):
            if coord not in idx:
                return coord, None
            off, size = idx[coord]
            for attempt in range(8):
                try:
                    r = requests.get(url[0], headers={"Range": f"bytes={off}-{off + size - 1}"}, timeout=120)
                    if r.status_code == 206:
                        return coord, r.content
                    url[0] = self.rf.get_url()
                except requests.exceptions.RequestException:
                    time.sleep(min(60, 3 * 2 ** attempt))
            raise IOError(f"chunk {coord} failed")

        with ThreadPoolExecutor(self.threads) as ex:
            for (z0, y0, x0), raw in ex.map(fetch, coords):
                if raw is None:
                    continue
                a = np.frombuffer(zlib.decompress(raw), self.dtype).reshape(zc, yc, xc)[:out.shape[0]]
                out[:, y0:y0 + yc, x0:x0 + xc] = a[:, :min(yc, self.shape[1] - y0), :min(xc, self.shape[2] - x0)]
        out = out[:, :self.ny, :self.nx]
        self.layers = {k: v for k, v in self.layers.items() if k[0] != c}   # Keep one block per channel in memory
        self.layers[key] = out
        return out

    def plane(self, c: int, z: int) -> np.ndarray:
        cz = (z // self.chunks[0]) * self.chunks[0]
        return self.layer(c, cz)[z - cz]


def remote_volume() -> bio.Volume:
    r = RemoteIms()
    vol = bio.Volume(name=FILE, n_channels=r.n_channels, n_z=r.nz, height=r.ny, width=r.nx, voxel_um=r.voxel_um,
                     channel_names=[f"Channel {c}" for c in range(r.n_channels)], dtype=str(r.dtype), voxel_size_known=True,
                     _reader=r.plane)
    vol._remote = r
    return vol


def extract_slices(n_blocks: int = 10, log=print):
    vol = remote_volume()
    mask_ch = vol.n_channels - 1
    lo, hi = int(0.1 * (vol.n_z - 1)), int(0.9 * (vol.n_z - 1))
    blocks = sorted({z // CHUNK_Z for z in range(lo, hi + 1) if (z // CHUNK_Z) * CHUNK_Z >= lo - CHUNK_Z // 2 and (z // CHUNK_Z) * CHUNK_Z + CHUNK_Z - 1 <= hi + CHUNK_Z // 2})
    pick = [blocks[i] for i in np.linspace(0, len(blocks) - 1, n_blocks).round().astype(int)]
    zs, imgs, gts = [], [], []
    part = OUT.with_suffix(".part.npz")
    if part.exists():   # Resume after an interruption
        d = np.load(part)
        zs, imgs, gts = list(d["zs"]), list(d["img"]), list(np.unpackbits(d["gt"], axis=-1)[..., :vol.width].astype(bool))
        log(f"resuming with {len(zs)} slices")
    t0 = time.time()
    for b in pick:
        if sum(1 for z in zs if z // CHUNK_Z == b) >= 2:
            continue
        for off in (8, 24, 16, 4, 28):   # Two slices per block, avoiding empty masks
            if sum(1 for z in zs if z // CHUNK_Z == b) == 2:
                break
            z = b * CHUNK_Z + off
            if not lo <= z <= min(hi, vol.n_z - 1):
                continue
            gt = vol.get_plane(mask_ch, z) > 0
            if gt.mean() <= 0.005:
                continue
            zs.append(z)
            imgs.append(vol.get_plane(IMAGE_CH, z))
            gts.append(gt)
        log(f"block {b}: slices {[z for z in zs if z // CHUNK_Z == b]}, {time.time() - t0:.0f}s")
        np.savez(part, zs=np.asarray(zs), img=np.stack(imgs), gt=np.packbits(np.stack(gts), axis=-1))
    order = np.argsort(zs)
    np.savez_compressed(OUT, zs=np.asarray(zs)[order], img=np.stack(imgs)[order], gt=np.packbits(np.stack(gts)[order], axis=-1),
                        shape=np.asarray(gts[0].shape), voxel_um=np.asarray(vol.voxel_um), n_z=vol.n_z)
    part.unlink(missing_ok=True)
    log(f"saved {len(zs)} slices to {OUT.name}")
    vol._remote.h.close()


if __name__ == "__main__":
    extract_slices()
