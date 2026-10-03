"""Writes small Imaris (.ims) files with the same HDF5 layout as real exports, for tests."""
from __future__ import annotations

import h5py
import numpy as np


def _text(v) -> np.ndarray:
    # Imaris stores attribute values as arrays of single characters
    return np.array([bytes([b]) for b in str(v).encode()], dtype="|S1")


def write_ims(path, arr: np.ndarray, voxel_um=(2.0, 0.65, 0.65), names=None, pad: int = 16):
    """arr is (C, Z, Y, X) uint16. Data blocks are padded beyond the image size, as Imaris does."""
    c, nz, ny, nx = arr.shape
    names = names or [f"Channel {i}" for i in range(c)]
    with h5py.File(path, "w") as f:
        img = f.create_group("DataSetInfo/Image")
        ext = {"ExtMin0": 0.0, "ExtMin1": 0.0, "ExtMin2": 0.0,
               "ExtMax0": nx * voxel_um[2], "ExtMax1": ny * voxel_um[1], "ExtMax2": nz * voxel_um[0]}
        for k, v in {**ext, "X": nx, "Y": ny, "Z": nz, "Unit": "um", "Name": "test"}.items():
            img.attrs[k] = _text(v)
        f.create_group("DataSetInfo/Imaris").attrs["Version"] = _text("7.0")
        for ci in range(c):
            f.create_group(f"DataSetInfo/Channel {ci}").attrs["Name"] = _text(names[ci])
            g = f.create_group(f"DataSet/ResolutionLevel 0/TimePoint 0/Channel {ci}")
            padded = np.zeros((nz + (-nz) % pad, ny + (-ny) % pad, nx + (-nx) % pad), np.uint16)
            padded[:nz, :ny, :nx] = arr[ci]
            g.create_dataset("Data", data=padded, chunks=(min(pad, padded.shape[0]), 32, 32))
            hist = np.histogram(arr[ci], bins=256)[0].astype(np.uint64)
            g.create_dataset("Histogram", data=hist)
            for k, v in {"ImageSizeX": nx, "ImageSizeY": ny, "ImageSizeZ": nz, "HistogramMin": float(arr[ci].min()),
                         "HistogramMax": float(arr[ci].max())}.items():
                g.attrs[k] = _text(v)
        f.create_group("DataSetTimes")
        f.create_group("DataSetInfo/TimeInfo").attrs["FileTimePoints"] = _text(1)
