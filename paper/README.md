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

From `results/summary.md`: four samples (A, C, E, F), 10 test slices each, nested leave-one-sample-out. Two method versions are scored on the same slices and clicks: `method-v1` (git tag) and method-v2, which adds the boundary improvements below. With four samples, these results are still preliminary.

| Mean Dice (95% CI over samples and slices) | 3 + 6 clicks | 25 + 25 clicks |
|---|---|---|
| boneseg v1, clicks, clean | 0.64 (0.58–0.72) | 0.68 (0.63–0.75) |
| boneseg v2, clicks, clean | 0.67 (0.63–0.72) | 0.76 (0.69–0.80) |
| boneseg v2 + refiner, clicks, clean | 0.69 (0.65–0.74) | 0.73 (0.67–0.80) |
| SAM ViT-B, clean | 0.60 (0.52–0.69) | 0.74 (0.66–0.82) |
| micro-SAM ViT-B LM, clean | 0.49 (0.37–0.60) | 0.61 (0.46–0.71) |
| Random forest (ilastik-style), clean | 0.41 (0.29–0.56) | 0.51 (0.38–0.65) |
| boneseg v2 + refiner, noisy | 0.64 (0.60–0.68) | 0.67 (0.60–0.78) |
| SAM ViT-B, noisy | 0.55 (0.46–0.65) | 0.69 (0.61–0.77) |
| Otsu, no input | 0.31 (0.19–0.45) | |

| Mean Dice, 5 labelled slices | Same sample | Other samples |
|---|---|---|
| boneseg learned model | 0.78 (0.73–0.81) | 0.73 (0.67–0.77) |
| boneseg v2 learned model (2 x 2 passes) | 0.77 (0.72–0.81) | 0.74 (0.68–0.79) |
| Random forest | 0.60 (0.51–0.70) | 0.46 (0.33–0.60) |

| B.Ar/T.Ar bias against the expert (expert mean 11.2%) | 3 + 6 clicks | 25 + 25 clicks |
|---|---|---|
| boneseg v1 | +4.5 points | +6.6 points |
| boneseg v2 | +0.9 points | +2.0 points |
| boneseg v2 + refiner | +3.3 points | +4.1 points |
| SAM ViT-B | | +4.6 points |

**What method-v2 changes.** Features from 2 x 2 sub-patch shifts (a twice finer feature grid), a guided filter that moves the score boundary onto image edges, and a stricter click threshold. All three were tuned on the other samples in every fold, and every fold chose the same: guided filter, background weight 0.8, and threshold position 0.7 with 3 + 6 clicks or 0.9 with 25 + 25. The app now uses these as defaults ("Auto" strictness picks 0.7 or 0.9 from the number of clicks).

What this supports, and what not yet:
- **v2 against v1.** With 25 clean clicks, v2 gains +0.077 Dice (CI +0.028 to +0.128, better on 92% of slices) and cuts the bone-area bias from +6.6 to +2.0 points (ICC 0.64 to 0.85). With 3 clean clicks the gain is +0.031 (CI −0.016 to +0.085). With noisy clicks v2 is level with v1.
- **v2 against SAM.** With few clicks, boneseg (v2 + refiner) beats SAM by +0.087 (CI +0.029 to +0.140) and by +0.089 with noisy clicks. With 25 clean clicks they are level (−0.011, CI −0.053 to +0.031), and boneseg's bone-area bias is smaller (+2.0 against +4.6 points for v2).
- **The learned refiner** (a small U-Net on the image and the score map, trained on the other samples) helps with few or noisy clicks (+0.014 and +0.043, CIs include zero) and costs a little with many clicks (−0.027). It raises the bone-area bias again, so it stays optional.
- **Finer features for the learned model** make no difference (+0.014 and −0.006, CIs include zero).
- **Not yet shown.** Generalization beyond these four samples, agreement with a second human, and nnU-Net as the supervised reference.

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
