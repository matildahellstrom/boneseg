"""Loading microscopy files as lazily read volumes.

Every format is exposed through the same small interface, a Volume with the axes
(channel, z, y, x). Imaris files are read one plane at a time, since a single file
can hold tens of gigabytes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SUPPORTED_EXTENSIONS = {".ims", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".npy"}


@dataclass
class Volume:
    """A multichannel z-stack. Planes are read on demand through get_plane."""

    name: str
    n_channels: int
    n_z: int
    height: int
    width: int
    voxel_um: tuple[float, float, float] = (1.0, 1.0, 1.0)  # (z, y, x)
    channel_names: list[str] = field(default_factory=list)
    dtype: str = "float32"
    voxel_size_known: bool = False
    _reader: object = None

    def get_plane(self, channel: int, z: int) -> np.ndarray:
        # Plain ints: the Imaris reader treats NumPy integers as slices and fails
        channel, z = int(channel), int(z)
        if not 0 <= channel < self.n_channels:
            raise IndexError(f"Channel {channel} is out of range, the file has {self.n_channels}")
        if not 0 <= z < self.n_z:
            raise IndexError(f"Slice {z} is out of range, the file has {self.n_z}")
        return np.asarray(self._reader(channel, z))

    @property
    def pixel_um(self) -> tuple[float, float]:
        return (self.voxel_um[1], self.voxel_um[2])

    def info(self) -> dict:
        return {
            "name": self.name,
            "n_channels": self.n_channels,
            "n_z": self.n_z,
            "height": self.height,
            "width": self.width,
            "voxel_um": list(self.voxel_um),
            "voxel_size_known": self.voxel_size_known,
            "channel_names": self.channel_names,
            "dtype": self.dtype,
        }


def normalize_plane(plane: np.ndarray, low: float = 1.0, high: float = 99.5) -> np.ndarray:
    """Clips intensity outliers at the given percentiles and scales to [0, 1]."""
    plane = np.asarray(plane, np.float32)
    lo, hi = np.percentile(plane, [low, high])
    if hi - lo < 1e-8:
        return np.zeros_like(plane, dtype=np.float32)
    return np.clip((plane - lo) / (hi - lo), 0, 1).astype(np.float32)


def _default_names(n: int) -> list[str]:
    return [f"Channel {i}" for i in range(n)]


def _from_array(name: str, arr: np.ndarray, voxel_um=(1.0, 1.0, 1.0), names=None, known=False) -> Volume:
    """Wraps an in-memory (C, Z, Y, X) array."""
    assert arr.ndim == 4, arr.shape
    c, z, y, x = arr.shape
    return Volume(
        name=name, n_channels=c, n_z=z, height=y, width=x, voxel_um=tuple(float(v) for v in voxel_um),
        channel_names=list(names) if names else _default_names(c), dtype=str(arr.dtype),
        voxel_size_known=known, _reader=lambda ch, zz: arr[ch, zz],
    )


def _ims_channel_names(path: Path, n: int) -> list[str]:
    try:
        import h5py

        names = []
        with h5py.File(path, "r") as f:
            for i in range(n):
                attrs = f[f"DataSetInfo/Channel {i}"].attrs
                raw = attrs.get("Name")
                text = b"".join(raw).decode(errors="ignore") if raw is not None else ""
                names.append(text.strip() or f"Channel {i}")
        return names
    except Exception:
        return _default_names(n)


def load_ims(path: Path) -> Volume:
    from imaris_ims_file_reader.ims import ims

    f = ims(str(path), squeeze_output=True)
    _, c, z, y, x = f.shape
    voxel = tuple(float(v) for v in f.resolution)
    known = all(v > 0 for v in voxel)
    return Volume(
        name=path.name, n_channels=c, n_z=z, height=y, width=x,
        voxel_um=voxel if known else (1.0, 1.0, 1.0), voxel_size_known=known,
        channel_names=_ims_channel_names(path, c), dtype=str(f.dtype),
        _reader=lambda ch, zz: f[0, ch, zz],
    )


def _tiff_voxel(tif) -> tuple[tuple[float, float, float], bool]:
    """Best-effort voxel size from ImageJ metadata and resolution tags, in micrometres."""
    try:
        page = tif.pages[0]
        tags = page.tags
        xr = tags.get("XResolution")
        yr = tags.get("YResolution")
        unit = tags.get("ResolutionUnit")
        ij = tif.imagej_metadata or {}
        z = float(ij.get("spacing", 1.0))
        if xr is None or yr is None:
            return (z, 1.0, 1.0), False
        xnum, xden = xr.value
        ynum, yden = yr.value
        px_x, px_y = xden / xnum, yden / ynum
        unit_name = str(ij.get("unit", "")).lower()
        if unit is not None and int(unit.value) == 3:  # centimetre
            px_x, px_y = px_x * 1e4, px_y * 1e4
        elif unit is not None and int(unit.value) == 2 and unit_name not in ("micron", "um", "µm", "\\u00b5m"):
            px_x, px_y = px_x * 25400, px_y * 25400  # inch
        known = unit_name in ("micron", "um", "µm", "\\u00b5m") or (unit is not None and int(unit.value) == 3)
        return (z, px_y, px_x), known
    except Exception:
        return (1.0, 1.0, 1.0), False


def _to_czyx(arr: np.ndarray, axes: str) -> tuple[np.ndarray, str]:
    """Reorders an array with tifffile axis letters into (C, Z, Y, X).
    Generic sequence axes (I, Q) count as z, samples (S) count as channels,
    and any other axis, such as time, is reduced to its first index."""
    axes = axes.upper()
    for generic in ("I", "Q"):
        if generic in axes and "Z" not in axes:
            axes = axes.replace(generic, "Z")
    if "S" in axes and "C" not in axes:
        axes = axes.replace("S", "C")
    for i in reversed(range(len(axes))):
        if axes[i] not in "CZYX" or axes.count(axes[i]) > 1 and axes.index(axes[i]) != i:
            arr = np.take(arr, 0, axis=i)
            axes = axes[:i] + axes[i + 1:]
    for ax in "CZ":
        if ax not in axes:
            arr = arr[np.newaxis]
            axes = ax + axes
    return np.transpose(arr, [axes.index(a) for a in "CZYX"]), "CZYX"


def load_tiff(path: Path) -> Volume:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes = series.axes.upper()
        arr = series.asarray()
        voxel, known = _tiff_voxel(tif)
    arr, axes = _to_czyx(arr, axes)
    return _from_array(path.name, arr, voxel, known=known)


def load_image(path: Path) -> Volume:
    from PIL import Image

    img = Image.open(path)
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = arr[None, None]
        names = ["Gray"]
    else:
        arr = np.moveaxis(arr[..., :3], -1, 0)[:, None]
        names = ["Red", "Green", "Blue"]
    return _from_array(path.name, arr, names=names)


def load_npy(path: Path) -> Volume:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim == 2:
        arr = arr[None, None]
    elif arr.ndim == 3:
        arr = arr[None]
    elif arr.ndim != 4:
        raise ValueError(f"Expected a 2D, 3D or 4D array, got shape {arr.shape}")
    return _from_array(path.name, arr)


def load_volume(path: str | Path) -> Volume:
    """Opens any supported microscopy file."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".ims":
        return load_ims(path)
    if ext in (".tif", ".tiff"):
        return load_tiff(path)
    if ext in (".png", ".jpg", ".jpeg"):
        return load_image(path)
    if ext == ".npy":
        return load_npy(path)
    raise ValueError(f"Unsupported file type {ext}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
