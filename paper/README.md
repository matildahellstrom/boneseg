# Paper evaluation

Code for a reproducible comparison of boneseg with standard tools, on the Liu samples that have expert masks. Everything here evaluates the method frozen at git tag `method-v1`, and nothing here changes the method.

## Protocol

**Samples.** Every Liu file present in `data/liudata/` that has an expert mask. The code knows A, C, D, E and F; this run used A, E and F, since C and D (16 and 24 GB) were not downloaded. The image channel is the autofluorescence channel (3 in A, 4 in the others). The expert mask is the file's last channel, an Imaris segmentation of that channel.

**Slices.** Twenty slices per sample, evenly spaced over the central 80% of the stack, keeping only slices where the expert mask covers at least 0.5% of the image. They alternate between development slices, used for tuning and as labelled training slices, and test slices, used only for the final scores.

**Nested leave-one-sample-out.** Each sample is held out in turn. Every setting is chosen on the development slices of the other samples only:
- boneseg from clicks: background weight λ in {0.4, 0.8, 1.2} and backbone input size in {644, 980} px
- random forest: largest filter scale in {8, 16} px
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

From `results/summary.md`: three samples (A, E, F), 10 test slices each, nested leave-one-sample-out, method `method-v1`. With three samples, these are preliminary.

| Mean Dice (95% CI over samples and slices) | 3 + 6 clicks | 25 + 25 clicks |
|---|---|---|
| boneseg, clicks, clean | 0.66 (0.60–0.74) | 0.70 (0.64–0.79) |
| SAM ViT-B, clean | 0.58 (0.50–0.70) | 0.72 (0.64–0.81) |
| micro-SAM ViT-B LM, clean | 0.48 (0.32–0.62) | 0.58 (0.40–0.72) |
| Random forest (ilastik-style), clean | 0.45 (0.33–0.59) | 0.57 (0.46–0.70) |
| boneseg, clicks, noisy | 0.60 (0.55–0.64) | 0.71 (0.64–0.79) |
| SAM ViT-B, noisy | 0.53 (0.44–0.65) | 0.67 (0.59–0.78) |
| Otsu, no input | 0.35 (0.18–0.49) | |

| Mean Dice, 5 labelled slices | Same sample | Other samples |
|---|---|---|
| boneseg learned model | 0.77 (0.72–0.82) | 0.70 (0.62–0.77) |
| Random forest | 0.65 (0.59–0.72) | 0.48 (0.30–0.61) |

What this supports, and what not yet:
- **Few or imperfect clicks.** boneseg beats SAM with few clicks (+0.08, CI +0.03 to +0.13) and with noisy clicks (+0.04, +0.01 to +0.10), and it barely degrades with noise. With 25 clean clicks, SAM and boneseg are level (SAM +0.01, CI −0.01 to +0.03).
- **Other baselines.** It clearly beats the ilastik-style random forest and micro-SAM in every condition, and the learned model beats the random forest trained on the same labels by 0.12 (same sample) and 0.22 (other samples).
- **Bias in bone measures.** boneseg overestimates bone area: B.Ar/T.Ar is +6.4 percentage points from clicks (expert mean 13.1%), and +3.7 from the learned model. Tb.Th is +20 to +26 µm too thick. The masks spill over the expert boundary. Because the bias is consistent (narrow limits of agreement, ICC 0.64 to 0.81, r 0.93 to 0.95), group comparisons are less affected than absolute values, but the paper should say so, or correct it.
- **Not yet shown.** Generalization beyond these three samples, agreement with a second human, and nnU-Net as the supervised reference.

## Running it

```bash
pip install scikit-learn segment-anything
mkdir -p ~/.cache/boneseg-paper && cd ~/.cache/boneseg-paper
curl -LO https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth && mv sam_vit_b_01ec64.pth sam_vit_b.pth
curl -L -o microsam_vit_b_lm.pt https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/diplomatic-bug/1.2/files/vit_b.pt
cd -
python paper/evaluate.py          # about 40 min for three samples on an M-series Mac; --quick for a smoke test
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

- **More samples.** C and D, and ideally samples from another lab or stain.
- **A second annotator** on a subset of test slices, to know how well two people agree.
- **A small user study** with real clicks and timing, since simulated clicks are only a proxy.
- **Osteoclast-level agreement** (Oc.Pm/B.Pm, N.Oc/B.Pm), which needs expert osteoclast masks; the current expert masks are bone only.
