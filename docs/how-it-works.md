# How boneseg segments bone, and what each part adds

This page explains the three building blocks behind boneseg's masks: the DINOv2 model, the U-Net additions, and fine-tuning. It says what each does to the masks, how much it helps, and when to use it. The numbers come from four expert-annotated Liu samples (A, C, E and F), always scored on slices and samples that were not used for training or tuning. Details are in `paper/README.md`.

Dice measures how well a mask overlaps the expert's mask: 1 is a perfect match and 0 is no overlap. B.Ar/T.Ar is the bone area fraction, and its bias says how far the masks over- or underestimate it.

## The short version

| What you do | Mean Dice | B.Ar/T.Ar bias | Use it when |
|---|---|---|---|
| 3 bone + 6 background clicks | 0.67 | +0.9 points | You want a quick mask with no labelling |
| 25 + 25 clicks | 0.76 | +2.0 points | You want a good mask without labelling |
| 3 + 6 clicks on a fine-tuned DINOv2 | 0.75 | +1.7 points | You have a fine-tuned model for your kind of images |
| A learned model from 5 labelled slices of the same sample | 0.78 | +2.9 points | You will segment many slices of one sample |

For comparison, Segment Anything (SAM), a widely used general-purpose tool, scored 0.60 with 3 + 6 clicks and 0.74 with 25 + 25 clicks.

## 1. DINOv2: the part that understands the image

DINOv2 is a neural network that Meta trained on 142 million photographs, without any labels. It learned to describe what is in an image. boneseg cuts each slice into small squares of 14 × 14 pixels, called patches, and DINOv2 turns every patch into a list of 384 numbers that describe its texture and context. Patches that look alike get similar descriptions, even when their brightness differs.

When you click, boneseg uses the clicked patches as examples. It compares every patch in the slice with your bone examples and your background examples, and gives each patch a score. Patches more similar to the bone clicks than to the background clicks score high. A threshold turns the scores into a mask, and boneseg places that threshold between the scores of your bone clicks and your background clicks.

**Why this works with few clicks.** DINOv2 already knows what textures and structures look like, so a handful of examples is enough to say which ones you mean. Nothing is trained when you click, so the result appears within a second or two.

**Its weakness is the boundary.** DINOv2 sees the image in 14-pixel patches, so it cannot say exactly where inside a patch the bone ends. Before the improvements below, masks spilled over the bone edge: bone area was overestimated by 6.6 percentage points with 25 + 25 clicks, against an expert mean of 11.2%. Thin trabeculae came out too thick.

### Sharper boundaries, now on by default

Three settings under "Clean-up and advanced" fix most of this. All three were chosen on other samples and then tested on new ones.

- **Feature passes, 2 × 2.** DINOv2 looks at the slice four times, each time shifted by half a patch. The four results interleave into a description every 7 pixels instead of every 14. This takes about 1.5 times as long.
- **Snap edges to image.** A guided filter moves the mask boundary onto real intensity edges in the image.
- **Strictness, Auto.** The threshold moves closer to your bone clicks. That is 70% of the way with five or fewer clicks of a kind, and 90% with more.

Together, these raised Dice with 25 + 25 clicks from 0.68 to 0.76. They cut the bone-area bias from +6.6 to +2.0 points. With 3 + 6 clicks, Dice went from 0.64 to 0.67, and the bias from +4.5 to +0.9 points. With carelessly placed clicks, the gain disappears, so careful clicks pay off.

## 2. The U-Net additions: a second network that works pixel by pixel

A U-Net is a neural network that looks at an image at full resolution and decides for every pixel whether it belongs to the object. It sees fine detail, but it has to be trained on labelled examples, and with few examples it learns little on its own.

The research notebook tried U-Nets in four ways, with DINOv2 Small as the starting point. These results are from the notebook's Kaggle run on five samples, with labels from other samples:

| U-Net input | Dice |
|---|---|
| The image alone | 0.54 |
| DINOv2's score map alone | 0.65 |
| The image plus DINOv2's score map | 0.70 |
| A trained layer on DINOv2's features, no U-Net | 0.71 |

The lesson: a U-Net on the raw image alone does poorly with so few labels. It does well when DINOv2 tells it roughly where the bone is, and the U-Net only has to sharpen the edges. Note that this run evaluated DINOv2 Small with settings tuned for the largest DINOv2 model, which understates it; a corrected run is pending.

boneseg's **learned refiner** is that idea in light form. It is a small U-Net with about 120,000 parameters. It takes the image and the click score map and outputs a sharper mask. It was trained once on expert slices with simulated clicks, so it needs no training when you click. On new samples:

- With few clicks it helps a little: +0.014 Dice with careful clicks and +0.043 with careless ones.
- With many clicks it costs a little: −0.027.
- It brings back part of the bone-area overestimate: +3.3 points instead of +0.9 with 3 + 6 clicks.

So the refiner is off by default. Try it ("Learned refiner" in the advanced settings) when you can only afford a few clicks, or your clicks are rough. Check its effect on a slice where you know the answer. The bundled refiner was trained on Liu autofluorescence images. For other images, train your own:

```bash
python -m boneseg train-refiner your_stack.ims:3 --out my_refiner.pt
```

The file needs a channel with an expert mask, which is the last channel unless you name it as `file:image_channel:mask_channel`.

## 3. Fine-tuning: teaching DINOv2 about bone

DINOv2 learned from photographs, not from microscopy. Fine-tuning continues its training on your expert-labelled slices, so that its descriptions separate bone from everything else more clearly. boneseg trains only DINOv2's last 4 of 12 layers, slowly, and keeps the version that scores best on separate validation slices. With so few labelled slices, retraining everything would make it memorise those slices.

### How we checked that it helps

Fine-tuning can make results worse, so it was tested like this:

1. **A fair baseline.** The same code was trained with DINOv2 left unchanged. The only difference was whether DINOv2 itself was adapted.
2. **Validation slices** chose when to stop training. Test slices were never used for that.
3. **New samples.** Each sample was held out in turn, and the model was trained on the others or on separate slices of the same sample.
4. **Paired comparison.** Both versions were scored on the same test slices with the same clicks, with confidence intervals.

### What it does

| Situation | Frozen DINOv2 | Fine-tuned DINOv2 | Difference (95% CI) |
|---|---|---|---|
| 3 + 6 clicks on a new sample (tuned on other samples) | 0.67 | 0.75 | +0.071 (+0.048 to +0.097) |
| 25 + 25 clicks on a new sample | 0.76 | 0.76 | −0.001 (−0.021 to +0.015) |
| Trained on 5 slices of the same sample | 0.75 | 0.78 | +0.034 (−0.009 to +0.063) |
| Trained on slices of other samples | 0.72 | 0.72 | −0.001 (−0.045 to +0.031) |

- **The clear gain is with few clicks.** After fine-tuning on other bone samples, 3 + 6 clicks on a new sample were almost as good as 25 + 25 clicks on the original DINOv2. This held on all four samples.
- **With many clicks it makes no difference.** Enough clicks already pin down what you mean.
- **Training on your own sample** helped on three of four samples but hurt on sample C, so the average gain is uncertain.
- **It costs** about 5 minutes on an M-series Mac, and the file is 28 MB.

### How to use it

```bash
python -m boneseg finetune stack1.ims:3 stack2.ims:4 --out dinov2_s14_mybone.pt
```

Every second labelled slice is used for validation. Put the file in the app's `models` folder, inside your projects folder. It then appears in the backbone menu as "DINOv2 Small, fine-tuned". A version fine-tuned on the four Liu samples is already there as `dinov2_s14_liu_bone`.

Learned models and profiles belong to the backbone they were made with. After switching to a fine-tuned backbone, train or save them again.

## Checking on your own images

These numbers come from four samples of one kind of bone imaging. Before you rely on any of the options for a study:

1. Set a channel with an expert mask as the reference, or label two or three slices carefully yourself.
2. Segment those slices with and without the option, and compare the Dice and the bone measures that boneseg reports against the reference.
3. Keep at least one checked slice out of any training, so that the check is honest.
