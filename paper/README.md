# Paper evaluation

Code for a reproducible comparison of boneseg with standard tools, on the Liu samples that have expert masks. Everything here evaluates the method frozen at git tag `method-v1`, and nothing here changes the method.

## Protocol

**Samples.** Every Liu file present in `data/liudata/` that has an expert mask. The code knows A, C, D, E and F; this run used A, C, E and F, since D (24 GB) was not downloaded. The image channel is the autofluorescence channel (3 in A, 4 in the others). The expert mask is the file's last channel, an Imaris segmentation of that channel.

**Slices.** Twenty slices per sample, evenly spaced over the central 80% of the stack, keeping only slices where the expert mask covers at least 0.5% of the image. They alternate between development slices, used for tuning and as labelled training slices, and test slices, used only for the final scores.

**Nested leave-one-sample-out.** Each sample is held out in turn. Every setting is chosen on the development slices of the other samples only:
- boneseg from clicks: background weight λ in {0.4, 0.8, 1.2} and backbone input size in {644, 980} px
- random forest: largest filter scale in {8, 16} px
- boneseg v2 from clicks: background weight λ in {0.4, 0.8, 1.2}, edge refinement {none, guided filter} and threshold position in {0.5, ..., 0.9}, with 2 x 2 feature passes at 980 px
- boneseg v2's refiner: trained on all development slices of the other samples, with simulated clicks at 3 + 6, 10 + 10 and 25 + 25
- boneseg's learned model: its classifier type and neighbourhood features, by its own internal cross-validation on the training slices

**Simulated clicks.** Object clicks inside the expert mask and background clicks outside it, at two budgets: 3 + 6 and 25 + 25.
- *Clean* clicks keep at least 10 px from the boundary.
- *Noisy* clicks may sit at the boundary, and 10% of each kind land on the wrong side (only with 5 or more clicks of a kind).

Three click seeds per test slice, different from the seeds used for tuning. Every interactive method gets exactly the same clicks.

**Methods.**

| Name | What it is |
|---|---|
| `otsu` | Otsu threshold on the normalized image, no user input |
| `rf_clicks` | ilastik-style pixel classification: a random forest (100 trees) on scikit-image's multiscale filter bank (intensity, edges, texture), trained on 3-px disks around the clicks, computed at up to 768 px |
| `sam` | Segment Anything ViT-B with the clicks as point prompts |
| `microsam` | micro-SAM's ViT-B fine-tuned on light microscopy (`vit_b_lm`), same prompts |
| `dino_clicks` | boneseg from clicks: frozen DINOv2 Small prototypes and the click-calibrated threshold |
| `rf_labels` | the random forest trained on labelled slices |
| `dino_labels` | boneseg's learned model trained on labelled slices |
| `nnunet` | nnU-Net v2 2D on the same labelled slices; run separately on a GPU (see below) |

Label-based methods are trained on 5 labelled development slices either of the held-out sample ("same sample") or of each other sample ("other samples").

**Metrics.**
- Dice, IoU and HD95 in µm against the expert mask.
- 2D bone measures from the predicted and the expert mask: B.Ar/T.Ar, B.Pm/T.Ar and Tb.Th = 2 B.Ar/B.Pm (plate model).
- Agreement: bias and 95% limits of agreement (Bland-Altman), ICC(A,1) and Pearson r.

**Statistics.** Click seeds are averaged per slice. Dice is reported per sample and as the mean of per-sample means, with a hierarchical bootstrap 95% CI: samples are resampled, then slices within them. Paired differences against boneseg use the same slices and clicks. With few samples, these intervals describe these samples, not the population of samples.

## Results so far

From `results/summary.md`: five samples (A, C, D, E, F), 10 test slices each, nested leave-one-sample-out. Sample D (24 GB) is read from Kaggle without downloading it (`remote_d.py`): its 20 slices come two per 32-slice storage block from ten blocks over the central 80% of the stack, and its fold was added after the others, tuned on A, C, E and F. Two method versions are scored on the same slices and clicks: `method-v1` (git tag) and method-v2, which adds the boundary improvements below.

| Mean Dice (95% CI over samples and slices) | 3 + 6 clicks | 25 + 25 clicks |
|---|---|---|
| boneseg v1, clicks, clean | 0.64 (0.59–0.70) | 0.70 (0.64–0.75) |
| boneseg v2, clicks, clean | 0.66 (0.62–0.70) | 0.76 (0.70–0.80) |
| boneseg v2 + refiner, clicks, clean | 0.67 (0.62–0.72) | 0.71 (0.65–0.78) |
| SAM ViT-B, clean | 0.62 (0.54–0.69) | 0.76 (0.68–0.82) |
| micro-SAM ViT-B LM, clean | 0.48 (0.38–0.57) | 0.61 (0.49–0.69) |
| Random forest (ilastik-style), clean | 0.38 (0.27–0.51) | 0.48 (0.36–0.61) |
| boneseg v2, noisy | 0.59 (0.53–0.64) | 0.67 (0.62–0.74) |
| SAM ViT-B, noisy | 0.57 (0.49–0.65) | 0.71 (0.64–0.78) |
| Otsu, no input | 0.27 (0.16–0.40) | |

| Mean Dice, 5 labelled slices | Same sample | Other samples |
|---|---|---|
| boneseg learned model | 0.77 (0.73–0.80) | 0.72 (0.68–0.77) |
| boneseg v2 learned model (2 x 2 passes) | 0.76 (0.72–0.80) | 0.74 (0.68–0.78) |
| Random forest | 0.58 (0.50–0.67) | 0.44 (0.33–0.56) |

| B.Ar/T.Ar bias against the expert (expert mean 10.6%) | 3 + 6 clicks | 25 + 25 clicks |
|---|---|---|
| boneseg v1 | +4.4 points | +5.7 points |
| boneseg v2 | +1.4 points | +2.0 points |
| boneseg v2 + refiner | +4.0 points | +4.7 points |
| SAM ViT-B | | +4.0 points |

**What method-v2 changes.** Features from 2 x 2 sub-patch shifts (a twice finer feature grid), a guided filter that moves the score boundary onto image edges, and a stricter click threshold. All three were tuned on the other samples in every fold. Every fold chose the guided filter and a threshold position of 0.7 with 3 + 6 clicks or 0.9 with 25 + 25, and a background weight of 0.8, except sample D's fold with 3 + 6 clicks (1.2). The app now uses these as defaults ("Auto" strictness picks 0.7 or 0.9 from the number of clicks).

What this supports, and what not yet:
- **v2 against v1.** With 25 clean clicks, v2 gains +0.060 Dice (CI +0.014 to +0.110, better on 78% of slices) and cuts the bone-area bias from +5.7 to +2.0 points (ICC 0.65 to 0.83). With 3 clean clicks the gain is +0.020 (CI −0.020 to +0.069); with noisy clicks v2 is level with v1. On sample D alone, v2 did not improve on v1.
- **v2 against SAM.** With 25 clean clicks they are level (−0.001, CI −0.042 to +0.040). With 3 clean clicks boneseg is ahead (+0.042) but, with sample D added, the interval includes zero (−0.033 to +0.116); SAM does especially well on D (0.82 with 25 clicks). boneseg's bone-area bias is half of SAM's (+2.0 against +4.0 points).
- **The learned refiner** helps with few or noisy clicks (+0.008 and +0.033) and costs with many (−0.046, CI −0.101 to +0.010); on sample D it cost 0.12. It stays optional.
- **The learned model from 5 labelled slices** is the most accurate option: 0.77 with labels of the same sample, 0.73 with labels of other samples, against 0.44 to 0.59 for the random forest.
- **Not yet shown.** Agreement with a second human, and nnU-Net as the supervised reference.

Development-only experiments that led to v2, never touching test slices: `refine_experiment.py` (`results/refine_dev.csv`) and `refiner_experiment.py` (`results/refiner_dev.csv`).

## Fine-tuning DINOv2

`finetune_experiment.py` (`results/finetune_summary.md`) asks whether training DINOv2 Small's last 4 blocks on expert slices improves the masks. Same protocol: leave-one-sample-out, test slices only, validation slices for early stopping. The baseline is the same code with DINOv2 frozen, so only the adaptation of DINOv2 differs.

| Fine-tuned minus frozen, test Dice | Difference (95% CI) | Slices improved |
|---|---|---|
| 3 + 6 clicks on a new sample, backbone fine-tuned on the other samples | +0.071 (+0.048 to +0.097) | 82% |
| 25 + 25 clicks on a new sample | −0.001 (−0.021 to +0.015) | 50% |
| Output layer trained on 5 slices of the same sample | +0.034 (−0.009 to +0.063) | 80% |
| Output layer trained on 5 slices of each other sample | −0.001 (−0.045 to +0.031) | 55% |

Fine-tuned features make few clicks almost as good as many (3 + 6 clicks: 0.674 to 0.745, on all four samples). They do not help once there are enough clicks or labels. Fine-tuning on the same sample helped on A, E and F and hurt on C.

```bash
python paper/finetune_experiment.py   # about 75 min for four samples on an M-series Mac
python paper/finetune_analyze.py
```

## Osteoclasts: the NOISe mouse data

`noise_common.py`, `noise_evaluate.py`, `noise_analyze.py` and `noise_finetune.py` test osteoclast segmentation and counting on the public NOISe data (Wan et al., 2024): brightfield images of TRAP-stained mouse osteoclast cultures in five batches, with expert outlines of every osteoclast (TRAP-positive, three or more nuclei). The cells grow on plastic, so there is no bone surface, and Oc.Pm/B.Pm cannot be tested; the human bone-chip images are not public.

**Protocol.** 20 development and 20 test patches (832 × 832 px) per batch, split by well; leave-one-batch-out, with every setting tuned on the development patches of the other four batches. Simulated clicks put object clicks inside osteoclasts and half of the background clicks on stained cells that are not osteoclasts. A predicted cell counts as found when it overlaps an expert outline with IoU ≥ 0.5. A pilot on development patches chose colour input (better than any single stain channel) with 2 × 2 feature passes. Before the run, the targets for counting were set at ICC ≥ 0.8 and bias within 10%.

| Test patches, 5 batches | Dice (95% CI) | Cell F1 | Count bias per patch | Count ICC |
|---|---|---|---|---|
| boneseg, 10 + 20 clicks | 0.735 (0.64–0.82) | 0.66 | +0.81 | 0.79 |
| SAM ViT-B, 10 + 20 clicks | 0.534 (0.39–0.65) | 0.42 | −0.85 | 0.46 |
| micro-SAM ViT-B LM, 10 + 20 clicks | 0.326 (0.23–0.43) | 0.16 | −1.31 | 0.23 |
| Random forest, 10 + 20 clicks | 0.626 (0.54–0.72) | 0.61 | −1.17 | 0.64 |
| boneseg, 3 + 6 clicks | 0.683 (0.58–0.77) | 0.59 | +0.62 | 0.68 |
| boneseg learned model, 5 labelled patches of the same batch | 0.668 (0.60–0.75) | 0.60 | +1.80 | 0.51 |
| Otsu, no input | 0.434 (0.32–0.56) | 0.47 | −1.50 | 0.38 |

- **boneseg leads every baseline** on new batches: +0.20 Dice over SAM (CI +0.14 to +0.27) with 10 + 20 clicks and +0.18 with 3 + 6. SAM is built to outline one object from a set of clicks, while a patch holds several osteoclasts.
- **Counting.** There are about four osteoclasts per patch. Tuned for cell F1, boneseg over-counts by 0.8 (ICC 0.79), just short of the targets. With the app's default settings, which were fixed on the bone data before this evaluation, 10 + 20 clicks give +0.27 per patch (+7%, ICC 0.81) at the same Dice, just inside both targets.
- **What goes wrong.** On development patches, only 10 of 456 predicted cells were fragments or merges of real osteoclasts. Most errors are look-alikes: clusters of small stained cells and pale large cells that the experts do not count (137 predicted cells), and faint or thin osteoclasts that are missed (84 of 400). A per-cell filter on size, stain and shape raised F1 on development patches from 0.68 to 0.72, but turned the over-count into an under-count, so it is not used.
- **Fine-tuning DINOv2** on 5 patches of each other batch (colour, last 4 blocks): with 3 + 6 clicks Dice rises from 0.683 to 0.741 (+0.058, CI +0.033 to +0.086) and the count bias falls to +8% (ICC 0.79); with 10 + 20 clicks it makes no difference (+0.006). The same pattern as on bone.
- **Batch m5 is hard for every method** (boneseg 0.56, SAM 0.27), with paler stain and out-of-focus areas.
- **Not yet compared:** NOISe's own detector, trained per fold, which needs a GPU (the published checkpoint saw all five batches). `kaggle_noise/` is a ready Kaggle kernel: it streams the NOISe archive, retrains YOLOv8-L segmentation once per held-out batch on 1500 patches of each other batch (1.3 GPU-hours per fold, about 7 hours in all), and predicts the same test patches:

```bash
kaggle kernels push -p paper/kaggle_noise
kaggle kernels output matildahellstrom/boneseg-noise-detector-baseline -p noise_out
python paper/noise_yolo_score.py --preds noise_out/preds.zip
python paper/noise_analyze.py
```

## Plasma cells: SegPC-2021

`segpc_common.py`, `segpc_evaluate.py`, `segpc_analyze.py` and `segpc_counts.py` test boneseg on the SegPC-2021 challenge data (Kaggle `sbilab/segpc2021dataset`, CC BY-NC-SA 4.0; cite the three papers in its readme): bone marrow aspirate slides of multiple myeloma patients, Jenner-Giemsa stain, brightfield colour, two cameras. Experts outlined each plasma cell of interest with its nucleus and cytoplasm; other cells are present but not outlined. This is bone marrow, not bone tissue: a test of how far the method carries.

**Protocol.** Settings tuned on 60 training images, scored on the 199 validation images with outlines (image 610 has none), at the official 1080 × 1440 resolution. Object clicks inside outlined plasma cells; background clicks at least 10 px from them (not aimed at unoutlined cells, which may be plasma cells too). The official score is the mean over expert cells of the best IoU of any predicted whole cell; it does not penalize extra cells.

| 199 validation images | Official score (95% CI), 10 + 20 clicks | 3 + 6 clicks |
|---|---|---|
| boneseg, clicks | 0.550 (0.52–0.58) | 0.439 (0.42–0.46) |
| boneseg, nucleus + cytoplasm as two structures | 0.433 (0.41–0.46) | 0.356 |
| SAM ViT-B | 0.475 (0.45–0.50) | 0.389 |
| micro-SAM ViT-B LM | 0.394 | 0.359 |
| Random forest | 0.366 | 0.327 |
| Otsu, no input | 0.417 | |
| boneseg learned model, 5 labelled training images | 0.418 | |

- **boneseg leads every baseline:** +0.075 over SAM (CI +0.052 to +0.099) with 10 + 20 clicks and +0.050 with 3 + 6.
- **Nucleus and cytoplasm.** As two structures, boneseg reaches nucleus Dice 0.84 and cytoplasm Dice 0.69 near the outlined cells with 10 + 20 clicks; the pale cytoplasm edge is the hard part.
- **The main failure is touching cells.** Plasma cells sit in dense clusters, and boneseg's mask, though it covers them well (Dice 0.86 near the outlined cells), joins neighbours into one region, which the per-cell score punishes: in the worst image all eight plasma cells are covered and the score is 0.04. Separating touching cells, for example from nucleus seeds, is the obvious next improvement.
- **Counting** (`segpc_counts.py`, minimum cell size chosen by cell F1 on the training images): with 10 + 20 clicks boneseg finds cells with F1 0.46 and counts with almost no bias (+0.2 per image) but moderate agreement (ICC 0.63); SAM reaches F1 0.40 and under-counts (−1.4 per image, ICC 0.58).
- Published challenge entries are trained on the full training set and reach far higher scores; boneseg here uses clicks on one image and no training.
- **Fine-tuning DINOv2** (`segpc_finetune.py`, last 4 blocks, on 20 or 100 training images not used for tuning): clicks on the fine-tuned features raise the official score with 3 + 6 clicks from 0.439 to 0.534 (+0.095, CI +0.068 to +0.123) and with 10 + 20 clicks from 0.550 to 0.579 (+0.029, CI +0.005 to +0.056). 100 images did no better than 20. The fine-tuned network fills cells far better (validation Dice 0.80 against 0.64 frozen), yet its own output scores lower per cell than an output layer on frozen features, and every fine-tuned variant under-counts by 1.3 to 1.8 cells per image: fuller masks join touching cells even more often. Fine-tuning and separating touching cells are complementary.

## Whole stacks in 3D

`stack3d_evaluate.py` and `stack3d_analyze.py` (`results/stack3d_summary.md`, `figures/stack3d_profiles.png`) segment the central 80% of each Liu stack (A 202, C 505, E 390 and F 233 slices; voxels 2 × 1.625 × 1.625 µm) and compare the bone volumes with the experts' 3D masks. Both volumes are reduced the same way to 4 × 6.5 × 6.5 µm voxels; Tb.Th, Tb.N and Tb.Sp follow the plate model (Tb.Th = 2 BV/BS), checked on a synthetic slab (63 µm measured for 65 µm).

| Mean over the four stacks | 3D Dice | Bone area per slice vs expert (r) | BV/TV | BS/BV | Tb.Th | Tb.N | Tb.Sp |
|---|---|---|---|---|---|---|---|
| Learned model, 5 labelled slices of the sample | 0.778 | 0.90 | +36% | +4% | −1% | +40% | −30% |
| Learned model, labelled slices of the other samples | 0.734 | 0.83 | +37% | +8% | −7% | +48% | −35% |
| 25 + 25 clicks on the middle slice, carried through the stack | 0.605 | −0.27 | −23% | +49% | −27% | +4% | +6% |

- **For whole stacks, label a few slices.** The learned model follows the expert through every stack (r 0.85 to 0.97 on C, E and F, 0.83 on A) and gets bone surface and trabecular thickness close: Tb.Th within 1% on average (within 16% per sample) with labels of the same sample, within 7% with labels of other samples.
- **Bone volume is overestimated by about a third** (BV/TV +36%), the 3D form of the boundary spill-over seen in 2D, and Tb.N and Tb.Sp, which are derived from BV/TV, inherit it.
- **Clicks on one slice do not carry through hundreds of slices.** Prototypes and threshold from the middle slice drift as the image changes with depth (3D Dice 0.61; on sample E the mask falls to almost no bone at the far end). The app's stack mode from one annotated slice is fine for short stacks but should be used with clicks on several slices, or with a learned model, for long ones.
- One percent of the slices in the same-sample condition were the labelled training slices, so that volume is not fully independent of training.

## Running it

```bash
pip install scikit-learn segment-anything
mkdir -p ~/.cache/boneseg-paper && cd ~/.cache/boneseg-paper
curl -LO https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth && mv sam_vit_b_01ec64.pth sam_vit_b.pth
curl -L -o microsam_vit_b_lm.pt https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/diplomatic-bug/1.2/files/vit_b.pt
cd -
python paper/evaluate.py          # about 2 hours for four samples on an M-series Mac; --quick for a smoke test
python paper/analyze.py           # results/summary.md, results/*.csv, figures/*.png
```

### nnU-Net

nnU-Net needs a CUDA GPU and several hours per fold, so it runs separately. `paper/kaggle_nnunet/` is a ready Kaggle kernel that trains the three folds with nnU-Net's 250-epoch trainer and predicts the test slices:

```bash
python paper/nnunet_export.py --out nnUNet_raw
# upload nnUNet_raw/Dataset501_LiuBone as a private Kaggle dataset called liubone-nnunet, then
kaggle kernels push -p paper/kaggle_nnunet
kaggle kernels output matildahellstrom/boneseg-nnunet-baseline -p nnunet_out   # predictions.zip
```

The same steps by hand:

```bash
python paper/nnunet_export.py --out nnUNet_raw        # writes the same slices and leave-one-sample-out folds
# train and predict on a GPU as described at the top of nnunet_export.py, then:
python paper/nnunet_score.py --pred-dir predictions
python paper/analyze.py
```

## What is still missing for a paper

- **More samples.** D, and ideally samples from another lab or stain.
- **A second annotator** on a subset of test slices, to know how well two people agree.
- **A small user study** with real clicks and timing, since simulated clicks are only a proxy.
- **Osteoclast-level agreement** (Oc.Pm/B.Pm, N.Oc/B.Pm), which needs expert osteoclast masks; the current expert masks are bone only.
