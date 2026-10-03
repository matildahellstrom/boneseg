# Notebooks

| Notebook | What it is |
|---|---|
| `dino-segmentation-notebook-final.ipynb` | The research notebook: DINOv2 backbones, transfer, sensitivity, the U-Net refinement and plots, on the Liu data |
| `boneseg_quickstart.ipynb` | How to use the boneseg package from Python, run on the demo stack |

## Running the research notebook on Kaggle

`../kernel-metadata.json` pushes it to Kaggle with a GPU and the liudata dataset:

```bash
kaggle kernels push -p ..
kaggle kernels status matildahellstrom/dino-segmentation-unet
kaggle kernels output matildahellstrom/dino-segmentation-unet -p results
```

## Changes since the run started on 3 October 2026

The Kaggle run started that morning uses the reviewed notebook from commit `8dd4b1b`. Later commits added three things, which only take effect after pushing again:

- **A threshold calibrated from the reference points** ("clicks"), compared with top_percent and otsu in the tuning sweep. The best of the three is kept, so later results may use it.
- **Per-slice score standardization** (`Config.score_norm = "robust"`), which keeps carried-over thresholds meaningful in the transfer experiment and leaves same-slice results unchanged.
- **A linear probe on the DINO features** (section 4.2b), compared with the U-Nets on the same held-out slices.

Each change was checked by running every cell with a stand-in model on synthetic data, not on the real data.
