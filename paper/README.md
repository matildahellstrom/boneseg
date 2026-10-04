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
