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

## The run of 3 October 2026

It completed tuning, the four backbones and the transfer experiments, then ran out of GPU memory at the first U-Net
training ("CUDA out of memory" on a 15 GB T4). Kaggle keeps no notebook from a failed run, so the tables were lost; the
log shows that tuning chose block 4 from the end, lambda 1.2 and the fixed 10% threshold for DINOv2 Giant.

Fixed since: the Giant backbone is freed before the U-Nets, the U-Nets train in mixed precision on the GPU, the
allocator reuses fragmented memory, and every experiment writes its tables to `results/` and prints them as soon as it
finishes, so a later failure no longer loses them.

## Changes since the run started on 3 October 2026

The Kaggle run started that morning uses the reviewed notebook from commit `8dd4b1b`. Later commits added three things, which only take effect after pushing again:

- **A threshold calibrated from the reference points** ("clicks"), compared with top_percent and otsu in the tuning sweep. The best of the three is kept, so later results may use it.
- **Per-slice score standardization** (`Config.score_norm = "robust"`), which keeps carried-over thresholds meaningful in the transfer experiment and leaves same-slice results unchanged.
- **A linear probe on the DINO features** (section 4.2b), compared with the U-Nets on the same held-out slices.

Each change was checked by running every cell with a stand-in model on synthetic data, not on the real data.
