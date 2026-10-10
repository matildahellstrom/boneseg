# Augmented Liu test slices (robustness testing)

Augmented copies of the 10 **test** slices of each Liu sample (A, C, D, E, F; `common.load_sample(name).test`), each
with an exact expert mask. They are meant for testing how robust the segmentation is to imaging changes, **never for
training or tuning**. Development slices are not used.

The generated data lives in `data/augmented/`, which is git-ignored. The licence of the Liu data is unclear, so the
derived images and any figures showing them stay local. Do not copy them under `paper/`.

## Layout

```
data/augmented/
  manifest.csv                                   one row per item
  <augmentation>/L<level>/<sample>_<z>_img.tif   float32 in [0, 1], zlib-compressed
  <augmentation>/L<level>/<sample>_<z>_mask.png  uint8, 0 / 255
  checks.csv, stats.csv                          written by check_augmented.py
  figures/<augmentation>.png                     visual check: clean and levels 1-3 with the mask outline
```

`manifest.csv` columns: `sample, z, augmentation, level, params` (a JSON string with every parameter, including
derived ones such as the blur sigma in pixels or the saturation clip value), `seed, pixel_um_y, pixel_um_x,
img_path, mask_path` (relative to `data/augmented/`), `mask_area_fraction`.

The unaugmented slice is augmentation `clean`, level 0. Every other augmentation has levels 1 (mild), 2 (moderate)
and 3 (strong). The seed is derived from `sha256("<sample>_<z>_<augmentation>_<level>")`, so regenerating gives
identical files. Images are rounded to multiples of 2^-16 before saving (error below 1e-5), which makes the TIFFs
about 40% smaller.

## Augmentations

Geometric augmentations move the image and the mask with the same transform. The image uses bilinear interpolation.
The mask is warped as 0/1 floats, also bilinearly, and thresholded at 0.5. Borders use reflect padding (mirrored
image and mask). Photometric augmentations leave the mask bit-for-bit the clean mask. Every image is clipped to
[0, 1] after augmentation.

| augmentation | what it does | L1 | L2 | L3 |
|---|---|---|---|---|
| rot90_flip | rotate k x 90 deg anticlockwise, then flip (exact; levels are different combinations, not harder) | horizontal flip | 90 deg | 270 deg + vertical flip |
| rotate | arbitrary rotation, reflect padding, same image size | 10 deg | 25 deg | 45 deg |
| elastic | smooth random displacement field (Gaussian-smoothed noise, sigma; RMS displacement per axis amp) | sigma 60 px, amp 3 px | sigma 45 px, amp 6 px | sigma 40 px, amp 8 px |
| contrast | `c (img - 0.5) + 0.5 + b` (reduced contrast plus a brightness shift) | c 0.75, b +0.05 | c 0.5, b -0.10 | c 0.3, b +0.20 |
| gamma_lo | `img ** gamma`, brightens the dark parts | 0.7 | 0.5 | 0.35 |
| gamma_hi | `img ** gamma`, darkens | 1.4 | 2.0 | 2.8 |
| gauss_noise | additive Gaussian noise, sigma in [0, 1] units | 0.03 | 0.06 | 0.12 |
| poisson_noise | shot noise: `Poisson(img N) / N`, N photons at intensity 1 | N 100 | N 30 | N 10 |
| blur | Gaussian PSF, sigma in um (converted per axis with pixel_um) | 1.5 um | 3 um | 6 um |
| illumination | gain `(1 - v r^2)(1 + a f)`: radial vignette (r = 1 in the corners) times a smooth random field f in [-1, 1] (scale a quarter of the image) | v 0.2, a 0.15 | v 0.4, a 0.3 | v 0.6, a 0.45 |
| dimming | deeper-slice look: `s img + offset + N(0, floor)` | s 0.7, +0.03, floor 0.01 | s 0.45, +0.06, floor 0.02 | s 0.25, +0.10, floor 0.03 |
| lowres | downsample by a factor (anti-aliased) and upsample back (bilinear) | x2 | x3 | x4 |
| saturation | gain raised so that p% of the bone pixels reach 1 (`img / t`, t = (100 - p)th percentile of the bone pixels; t is logged) | p 1% | p 5% | p 15% |

Notes:
- Gamma is split into two augmentations, one per direction, instead of mixing directions across levels.
- Elastic L3 was first sigma 30 px with amp 10 px. Its local area change reached 0.1x to 2.5x and the soft tissue
  looked swirled, so it was softened to sigma 40 px with amp 8 px (no folding; local area change about 0.45x to 1.6x).
- rot90_flip at 90/270 deg swaps the image axes; `pixel_um_y`/`pixel_um_x` are swapped with them.
- The clean slices already contain the microscope's own noise; the noise levels come on top of it.
- boneseg's `embed_image` uses the [0, 1] image as given, so linear changes (contrast, dimming) reach the model.
  A pipeline that re-runs `normalize_plane` first would undo most of them.

## Intensity statistics

Averaged over all items of an augmentation and level (whole image, and inside / outside the expert mask), from
`check_augmented.py` (`data/augmented/stats.csv`):

| augmentation | level | mean | std | bone mean | background mean |
|---|---|---|---|---|---|
| blur | 1 | 0.198 | 0.196 | 0.321 | 0.183 |
| blur | 2 | 0.198 | 0.194 | 0.321 | 0.183 |
| blur | 3 | 0.198 | 0.191 | 0.320 | 0.183 |
| clean | 0 | 0.198 | 0.199 | 0.321 | 0.183 |
| contrast | 1 | 0.323 | 0.149 | 0.416 | 0.313 |
| contrast | 2 | 0.249 | 0.099 | 0.311 | 0.242 |
| contrast | 3 | 0.609 | 0.060 | 0.646 | 0.605 |
| dimming | 1 | 0.168 | 0.139 | 0.255 | 0.158 |
| dimming | 2 | 0.149 | 0.092 | 0.205 | 0.143 |
| dimming | 3 | 0.149 | 0.058 | 0.180 | 0.146 |
| elastic | 1 | 0.198 | 0.198 | 0.321 | 0.183 |
| elastic | 2 | 0.198 | 0.198 | 0.321 | 0.184 |
| elastic | 3 | 0.198 | 0.197 | 0.321 | 0.183 |
| gamma_hi | 1 | 0.134 | 0.173 | 0.217 | 0.124 |
| gamma_hi | 2 | 0.083 | 0.146 | 0.128 | 0.078 |
| gamma_hi | 3 | 0.050 | 0.124 | 0.071 | 0.048 |
| gamma_lo | 1 | 0.282 | 0.218 | 0.441 | 0.263 |
| gamma_lo | 2 | 0.374 | 0.225 | 0.550 | 0.353 |
| gamma_lo | 3 | 0.478 | 0.216 | 0.654 | 0.457 |
| gauss_noise | 1 | 0.200 | 0.198 | 0.321 | 0.186 |
| gauss_noise | 2 | 0.204 | 0.200 | 0.321 | 0.191 |
| gauss_noise | 3 | 0.215 | 0.210 | 0.323 | 0.203 |
| illumination | 1 | 0.188 | 0.191 | 0.309 | 0.174 |
| illumination | 2 | 0.177 | 0.181 | 0.293 | 0.163 |
| illumination | 3 | 0.168 | 0.177 | 0.279 | 0.155 |
| lowres | 1 | 0.198 | 0.196 | 0.321 | 0.183 |
| lowres | 2 | 0.198 | 0.195 | 0.321 | 0.183 |
| lowres | 3 | 0.198 | 0.194 | 0.321 | 0.183 |
| poisson_noise | 1 | 0.197 | 0.202 | 0.321 | 0.183 |
| poisson_noise | 2 | 0.197 | 0.211 | 0.321 | 0.183 |
| poisson_noise | 3 | 0.195 | 0.233 | 0.319 | 0.181 |
| rot90_flip | 1 | 0.198 | 0.199 | 0.321 | 0.183 |
| rot90_flip | 2 | 0.198 | 0.199 | 0.321 | 0.183 |
| rot90_flip | 3 | 0.198 | 0.199 | 0.321 | 0.183 |
| rotate | 1 | 0.199 | 0.198 | 0.321 | 0.184 |
| rotate | 2 | 0.205 | 0.201 | 0.321 | 0.191 |
| rotate | 3 | 0.210 | 0.203 | 0.321 | 0.196 |
| saturation | 1 | 0.294 | 0.266 | 0.461 | 0.273 |
| saturation | 2 | 0.351 | 0.306 | 0.568 | 0.324 |
| saturation | 3 | 0.407 | 0.339 | 0.683 | 0.375 |

## Regenerate, check, load

From `paper/`, with `PYTHONPATH` set to the repository root:

```
python robustness/make_augmented.py                 # all samples and augmentations (~12 GB, ~15 min)
python robustness/make_augmented.py --samples A --augs blur rotate   # a subset; the manifest is merged
python robustness/check_augmented.py                # checks, stats.csv, figures/
```

The generator stops if less than 8 GB of disk would remain free.

Loading, from a script in `paper/` (paper/robustness has no `__init__.py`; it is imported as `robustness.common`, so
it does not clash with `paper/common.py`):

```python
from robustness.common import load_augmented
items = load_augmented(augmentation=["clean", "blur"], level=[0, 3])   # clean + strong blur; also samples=..., root=...
for it in items:
    img, gt = it.img, it.gt   # read from disk on access; same fields as common.Slice plus augmentation, level, params, seed
```

`img` and `gt` are not cached, so read each once per item.
