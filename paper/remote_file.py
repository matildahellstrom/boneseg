"""A read-only, seekable file over HTTP range requests, for opening large HDF5/Imaris files (such as Liu sample D on
Kaggle) without downloading them. Reads are cached in 4 MB blocks; an expired signed URL is fetched again."""
from __future__ import annotations

import io
import urllib.parse
from collections import OrderedDict
from pathlib import Path

import requests

BLOCK = 4 << 20


def kaggle_file_url(dataset: str, filename: str) -> str:
    """A signed download URL for one file of a Kaggle dataset, using the token in ~/.kaggle/access_token."""
    tok = (Path.home() / ".kaggle" / "access_token").read_text().strip()
    url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset}/" + urllib.parse.quote(filename)
    r = requests.get(url, headers={"Authorization": f"Bearer {tok}"}, allow_redirects=False, timeout=60)
    r.raise_for_status()
    return r.headers["Location"]


class RangeFile(io.RawIOBase):
    def __init__(self, get_url, max_blocks: int = 512):
        self.get_url = get_url
        self.url = get_url()
        self.pos = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self.max_blocks = max_blocks
        self.fetched = 0
        r = requests.get(self.url, headers={"Range": "bytes=0-0"}, timeout=60)
        self.size = int(r.headers["Content-Range"].split("/")[-1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _block(self, i: int) -> bytes:
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        lo, hi = i * BLOCK, min(self.size, (i + 1) * BLOCK) - 1
        import time
        for attempt in range(8):
            try:
                r = requests.get(self.url, headers={"Range": f"bytes={lo}-{hi}"}, timeout=120)
            except requests.exceptions.RequestException:   # Dropped connections and read timeouts
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            if r.status_code == 206:
                break
            self.url = self.get_url()   # The signed URL has expired
        else:
            raise IOError(f"range request for bytes {lo}-{hi} failed after retries")
        self.fetched += len(r.content)
        self.cache[i] = r.content
        if len(self.cache) > self.max_blocks:
            self.cache.popitem(last=False)
        return r.content

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.size:
            i, off = divmod(self.pos, BLOCK)
            b = self._block(i)[off:off + n]
            out += b
            self.pos += len(b)
            n -= len(b)
        return bytes(out)

    def readinto(self, buf):
        data = self.read(len(buf))
        buf[:len(data)] = data
        return len(data)
