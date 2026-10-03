# boneseg

Segment structures in bone microscopy images with a few clicks. boneseg turns the DINO few-shot segmentation from the notebook in `notebooks/` into a web app. You upload an Imaris, TIFF or PNG file, click a few cells and a few background spots, and get a mask with measurements in micrometres. You can then run the whole z-stack, export the results, or save the clicks as a profile for the next image.

![The boneseg interface segmenting the demo stack](docs/screenshot.png)

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m boneseg serve --open
```

The app opens at http://127.0.0.1:8000. Click **Try a synthetic demo image** to explore it without your own data. The first time you pick a DINOv2 backbone, its weights download from Meta, about 85 MB for Small.

The app runs on an NVIDIA GPU, an Apple Silicon GPU or the CPU, whichever it finds. Small and Base are comfortable on a laptop. Giant needs a large GPU.

### With Docker

```bash
docker build -t boneseg .
docker run -p 8000:8000 -v "$PWD/projects:/data" boneseg
```

The image runs on the CPU. For an NVIDIA GPU, change the PyTorch wheel index in the Dockerfile to a CUDA one and add `--gpus all` to `docker run`. The Dockerfile has not been built yet, because the development machine was short on disk space.

## How to use it

1. **Load an image.** Drop a file on the left panel. For multi-gigabyte Imaris files, use "Open a large file from disk instead" and paste the path, which opens the file in place without copying it.
2. **Pick the channel** that shows the structure. Optionally draw a region of interest, for example trabecular bone without the cortex. Masks, measurements, scores and exports then stay inside it. If the file has an expert segmentation channel, choose it as the reference mask. Every result is then scored with Dice, IoU and HD95 against it. Channels named "segmentation", "mask" or "surface" are picked up automatically.
3. **Click.** Click a few examples of the structure, then shift-click a few background spots. The mask updates after every click.
   To segment several structures at once, for example cells and bone matrix, click "+ Structure" and give each structure its own clicks. Background clicks are shared, each pixel goes to the structure it resembles most, and "Labels TIFF" exports one value per structure. Whole-stack runs then cover every structure, with a label stack and 3D volumes and object counts per structure. When one structure is named like bone or matrix, the run also computes histomorphometry on every slice against the first other structure, and the stack-level Oc.Pm/B.Pm, N.Oc/B.Pm and B.Ar/T.Ar can be compared between groups in "Compare samples".
4. **Refine.** The orange overlay shows where the mask depends on single clicks, and a dashed ring suggests the most useful next click.
5. **Correct and teach.** Press E to fix the mask with a brush, then save it as a label. After one or more labels, "Train model" fits a small classifier on the backbone features in a few seconds and reports its leave-one-slice-out Dice. Switch the method to "Learned model" to segment new slices without clicks. "Save model as profile" makes it reusable on other images and in batch runs.
6. **Measure bone.** "Bone histomorphometry" combines a bone mask and an osteoclast mask on the slice. It reports B.Ar/T.Ar, B.Pm, Oc.Pm/B.Pm, N.Oc/B.Pm and each cell's distance to bone, following the ASBMR nomenclature for 2D sections. Either mask can come from the current result, a profile, the learned model, a saved label or the reference channel.
7. **Export or scale up.**
   - Download the mask as PNG or TIFF, or the per-object measurements as CSV. The CSV includes each object's mean and integrated raw intensity in every channel, for example SOST inside cells segmented on the TRAP channel.
   - Run the whole stack, which writes a mask stack, a 3D label image with one ID per object, per-slice measurements and a 3D object table. Turn on "Side view" to see a cut through all slices with the stack mask on top, which shows at a glance whether the segmentation stays consistent with depth.
   - Save the clicks as a profile to segment the next image of the same stain without clicking.
   - Click "Compare samples" at the top to put samples into groups and compare their latest stack runs, with a dot plot and a Mann-Whitney U or Kruskal-Wallis test. The app says when the groups are too small for the test to show anything.
   - "Project zip" bundles everything done on the image except the image itself: settings, clicks, labels, learned models, profiles and stack results, with a README.
   - Open the report, a self-contained HTML page with the image, measurements, settings, the latest stack run and a draft methods paragraph.

### Keyboard shortcuts

| Key | Action |
|---|---|
| Click / Shift-click | Add an object / background point |
| Right-click | Remove the nearest point |
| 1, 2 | Switch between object and background clicks |
| Ctrl+Z | Undo the last point |
| , and . | Previous and next slice |
| F | Fit the image to the window |
| M, H, U | Toggle the mask, heatmap and uncertainty overlays |
| E, Esc | Start and cancel correcting the mask with a brush |
| [ and ], Ctrl+Z | While correcting: smaller or larger brush, undo the last stroke |
| R, Enter | Draw a region of interest and finish it |
| Alt-click | Move the side-view cut to this row |
| Scroll, drag | Zoom, pan |

## Batch processing

Once a profile works on one image, apply it to many files from the command line:

```bash
python -m boneseg profiles                       # List saved profiles and their ids
python -m boneseg batch data/*.ims --profile trap-cells-1a2b3c --channel 3 --z-step 5 --out results
```

Each file gets its own folder with a mask stack, per-slice measurements and a 3D object table. A combined `summary.csv` covers all files, and a file that fails is reported without stopping the rest. Add `--reference 4` to score every slice against an expert mask channel. On Liu file A, a profile made from one slice segmented 15 slices in 11 seconds on a MacBook, with a mean Dice of 0.61 against the expert mask.

## How it works

The image goes through a frozen, self-supervised DINOv2 backbone, which gives one feature vector per 14×14 pixel patch. The patches under your clicks become prototypes. Every patch is then scored by its mean cosine similarity to the object prototypes, minus λ times its similarity to the background prototypes. The score map is upsampled and thresholded into a mask.

Three choices differ from the notebook. Each was tested on the demo stack with the benchmark script.

- **The threshold comes from your clicks.** It sits halfway between the scores at the object clicks and those at the background clicks. On the demo stack this raised Dice from 0.36 with Otsu to 0.89. Otsu tended to separate tissue from empty space instead of the target from everything else.
- **Scores are standardized per slice.** The median and spread of each slice's scores are used to standardize it. This never changes the slice you clicked. It keeps the calibrated threshold meaningful on deeper, dimmer slices, raising stack Dice from 0.69 to 0.81 on a strongly degraded stack.
- **The image keeps its aspect ratio** when resized for the backbone, and measurements use the voxel size from the file.

The learned model is the app's take on the notebook's supervised U-Net step. It is a linear or small two-layer classifier over the frozen patch features, fitted to the share of each patch covered by your corrected masks, so it trains in seconds. Each patch also sees the mean features of the 5 × 5 patches around it. On Liu file A, a model trained on four labelled slices scored 0.69 Dice on eight unseen slices, against 0.63 for 25 + 25 clicks on every slice (`scripts/benchmark_head.py`). One labelled slice already scored 0.67. Without the neighbourhood features the same models scored 0.67 and 0.64. The same neighbourhood features did not suit the click-based method: they gained about 0.015 Dice on Liu file A but lost 0.07 on the demo stack, where the cells are small, so clicks keep using each patch on its own.

Two more ideas were tested on Liu file A and left out. Feeding DINOv2 three different channels as a colour image instead of one grey channel scored the same or worse (0.65 against 0.65 and 0.63 Dice with 25 + 25 clicks). Tuning the input size and λ on one labelled slice picked settings that did worse on the other slices (0.61 against 0.64), so the defaults stay fixed. The masks on that file mostly spill over the expert boundary, but neither doubling the feature resolution with half-patch-shifted passes (0.61 against 0.65) nor a stricter threshold (better with 25 clicks, worse with 3) fixed it reliably.

An adaptive mode that refreshed the prototypes slice by slice was tried and removed, because it lowered Dice in every setting tested.

| Mean Dice on the demo stack, DINOv2 Small | Clicked slice | Whole stack |
|---|---|---|
| Otsu threshold | 0.58 | 0.36 |
| Fixed 6% of the image | 0.69 | 0.58 |
| Threshold from clicks (default) | 0.88 | 0.89 |

On real data the gap is smaller. Results for Liu file A, channel 3, scored against its masked channel 4 on 12 slices, using DINOv2 Small (`scripts/benchmark_file.py`):

| Mean Dice on Liu file A | From clicks | Otsu | Fixed 10% |
|---|---|---|---|
| Clicks on each slice, 3 object + 6 background | 0.58 | 0.51 | 0.56 |
| Clicks on each slice, 25 + 25 | 0.63 | 0.54 | 0.62 |
| Clicks on the middle slice, run over the stack, 25 + 25 | 0.57 | 0.50 | 0.56 |

On that file a smaller backbone input of 644 px did slightly better than the default 980 px (0.66 against 0.65 with 25 + 25 clicks), and Base was close to Small. Try the "Detail" setting under "Clean-up and advanced" if the mask looks too fragmented.

Run the benchmarks yourself:

```bash
python scripts/benchmark_demo.py --ref top --depth-degradation 1.0
python scripts/benchmark_file.py "data/liudata/10-26-40_6_Blaze_crop2 quantified.ims" --channel 3 --reference 4
```

## Project layout

| Path | Contents |
|---|---|
| `boneseg/io.py` | Lazy loading of Imaris, TIFF, PNG/JPG and NumPy files as (channel, z, y, x) volumes |
| `boneseg/backbone.py` | DINOv2 backbones pinned to a fixed commit, plus a fast "classic" backbone for tests and offline use |
| `boneseg/segment.py` | Prototypes, scores, thresholds, clean-up, uncertainty and profiles |
| `boneseg/pipeline.py` | Whole-stack runs |
| `boneseg/histo.py` | 2D bone histomorphometry |
| `boneseg/head.py` | The small model trained on corrected masks |
| `boneseg/quantify.py`, `metrics.py` | Measurements in micrometres, 3D objects, Dice, IoU and HD95 |
| `boneseg/api.py`, `store.py` | The web server and its storage |
| `boneseg/static/` | The browser interface: plain HTML, CSS and JavaScript files in `js/`, loaded in order without a build step |
| `notebooks/` | The research notebook, with a Kaggle batch-run setup in `kernel-metadata.json`, and `boneseg_quickstart.ipynb`, which uses the package from Python |
| `scripts/` | Benchmarks |
| `tests/` | Tests on synthetic images, run with `pytest` |

## API

Everything the interface does goes through a JSON API, documented interactively at http://127.0.0.1:8000/docs while the app runs.

## Development

```bash
pip install -r requirements-dev.txt
pytest
```

The browser tests in `tests/e2e` drive the real interface on the demo stack. They need Playwright and are skipped without it:

```bash
pip install playwright && python -m playwright install chromium
pytest tests/e2e
```

Uploaded files, profiles and job outputs go to `projects/`, or to the folder set with `--data-dir` or the `BONESEG_DATA_DIR` environment variable. Microscopy files and outputs are kept out of git.
