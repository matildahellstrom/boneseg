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
