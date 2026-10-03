import numpy as np
import pytest


def make_blobs(h=200, w=260, n=6, seed=0, radius=(10, 22)):
    """Synthetic fluorescence image: bright round cells on a noisy, textured background."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w]
    gt = np.zeros((h, w), bool)
    centers = []
    for _ in range(n):
        r = rng.uniform(*radius)
        cy, cx = rng.uniform(r + 5, h - r - 5), rng.uniform(r + 5, w - r - 5)
        gt |= (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
        centers.append((int(cy), int(cx)))
    img = 0.15 + 0.6 * gt + rng.normal(0, 0.08, (h, w)) + 0.05 * np.sin(xx / 7.0)
    return np.clip(img, 0, 1).astype(np.float32), gt, centers


@pytest.fixture
def blobs():
    return make_blobs()
