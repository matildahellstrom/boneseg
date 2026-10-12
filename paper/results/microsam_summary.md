# micro-SAM against boneseg and fine-tuned SAM

Test images of the three datasets; 5-shot = 5 labelled images of the target (5 more stop the training early). micro-SAM (Archit et al. 2025) is fine-tuned from vit_b_lm with its own training code (boneseg/microsam_core.py); its automatic mask is the decoder's foreground >= 0.5. Dice, NSD (bone 5 µm, cells 2 px) and area bias (percentage points), mean over images with 95% intervals.


## Bone (Liu, 5 held-out samples)

**Automatic (no prompt)**

| Method | Setting | Prompt | Dice (95% CI) | NSD | Area bias |
|---|---|---|---|---|---|
| boneseg learned model | 5-shot | no prompt | 0.789 (0.757–0.819) | 0.218 | +0.32 |
| boneseg, fine-tuned DINOv2 + learned model | 5-shot | no prompt | 0.815 (0.773–0.848) | 0.266 | +0.08 |
| SAM ViT-B fine-tuned (enc+dec Adapter) | 5-shot | no prompt | 0.841 (0.798–0.872) | 0.309 | +0.60 |
| SAM ViT-B fine-tuned (enc+dec LoRA) | 5-shot | no prompt | 0.833 (0.784–0.868) | 0.302 | +0.68 |

**With prompts**

| Method | Setting | Prompt | Dice (95% CI) | NSD | Area bias |
|---|---|---|---|---|---|
| SAM ViT-B | zero-shot | object points | 0.548 (0.376–0.720) | 0.139 | +24.89 |
| micro-SAM generalist (vit_b_lm) | zero-shot | object points | 0.493 (0.355–0.590) | 0.080 | +18.22 |
| boneseg (clicks) | zero-shot | object + background points | 0.687 (0.625–0.743) | 0.167 | +2.52 |
| SAM ViT-B | zero-shot | object + background points | 0.653 (0.536–0.762) | 0.171 | +9.85 |
| micro-SAM generalist (vit_b_lm) | zero-shot | object + background points | 0.516 (0.378–0.614) | 0.110 | +12.79 |
| boneseg, fine-tuned DINOv2 + prompts | 5-shot | object + background points | 0.777 (0.704–0.829) | 0.219 | +3.24 |
| boneseg (clicks) | zero-shot | noisy boxes | 0.461 (0.390–0.534) | 0.063 | +11.61 |
| SAM ViT-B | zero-shot | noisy boxes | 0.676 (0.597–0.757) | 0.207 | +5.75 |
| micro-SAM generalist (vit_b_lm) | zero-shot | noisy boxes | 0.714 (0.639–0.764) | 0.186 | +2.50 |
| boneseg, fine-tuned DINOv2 + prompts | 5-shot | noisy boxes | 0.781 (0.737–0.821) | 0.186 | +3.08 |

| Paired comparison | Difference in Dice (95% CI) | Images better |
|---|---|---|

## Osteoclasts (NOISe, 5 held-out batches)

**Automatic (no prompt)**

| Method | Setting | Prompt | Dice (95% CI) | NSD | Area bias |
|---|---|---|---|---|---|
| boneseg learned model | 5-shot | no prompt | 0.662 (0.584–0.738) | 0.229 | -0.07 |
| boneseg, fine-tuned DINOv2 + learned model | 5-shot | no prompt | 0.721 (0.648–0.792) | 0.334 | -0.03 |
| SAM ViT-B fine-tuned (enc+dec Adapter) | 5-shot | no prompt | 0.704 (0.633–0.773) | 0.340 | -0.04 |
| SAM ViT-B fine-tuned (enc+dec LoRA) | 5-shot | no prompt | 0.674 (0.568–0.770) | 0.351 | -0.29 |

**With prompts**

| Method | Setting | Prompt | Dice (95% CI) | NSD | Area bias |
|---|---|---|---|---|---|
| SAM ViT-B | zero-shot | object points | 0.551 (0.416–0.663) | 0.300 | +9.85 |
| micro-SAM generalist (vit_b_lm) | zero-shot | object points | 0.317 (0.237–0.397) | 0.144 | +31.12 |
| boneseg (clicks) | zero-shot | object + background points | 0.570 (0.472–0.668) | 0.190 | +1.66 |
| SAM ViT-B | zero-shot | object + background points | 0.325 (0.236–0.424) | 0.121 | +20.85 |
| micro-SAM generalist (vit_b_lm) | zero-shot | object + background points | 0.132 (0.097–0.175) | 0.028 | +52.99 |
| boneseg, fine-tuned DINOv2 + prompts | 5-shot | object + background points | 0.687 (0.583–0.776) | 0.283 | +1.43 |
| boneseg (clicks) | zero-shot | noisy boxes | 0.752 (0.693–0.801) | 0.268 | +0.73 |
| SAM ViT-B | zero-shot | noisy boxes | 0.884 (0.812–0.933) | 0.655 | -0.27 |
| micro-SAM generalist (vit_b_lm) | zero-shot | noisy boxes | 0.884 (0.840–0.919) | 0.594 | +0.08 |
| boneseg, fine-tuned DINOv2 + prompts | 5-shot | noisy boxes | 0.825 (0.758–0.875) | 0.389 | +0.16 |

| Paired comparison | Difference in Dice (95% CI) | Images better |
|---|---|---|

## Plasma cells (SegPC, 3 few-shot draws)

**Automatic (no prompt)**

| Method | Setting | Prompt | Dice (95% CI) | NSD | Area bias |
|---|---|---|---|---|---|
| boneseg learned model | 5-shot | no prompt | 0.584 (0.561–0.607) | 0.092 | +6.26 |
| boneseg, fine-tuned DINOv2 + learned model | 5-shot | no prompt | 0.716 (0.692–0.738) | 0.176 | +1.19 |
| SAM ViT-B fine-tuned (enc+dec Adapter) | 5-shot | no prompt | 0.613 (0.591–0.634) | 0.126 | +3.92 |
| SAM ViT-B fine-tuned (enc+dec LoRA) | 5-shot | no prompt | 0.610 (0.588–0.632) | 0.156 | +4.27 |

**With prompts**

| Method | Setting | Prompt | Dice (95% CI) | NSD | Area bias |
|---|---|---|---|---|---|
| SAM ViT-B | zero-shot | object points | 0.644 (0.621–0.667) | 0.167 | +10.33 |
| micro-SAM generalist (vit_b_lm) | zero-shot | object points | 0.764 (0.740–0.786) | 0.157 | +6.97 |
| boneseg (clicks) | zero-shot | object + background points | 0.466 (0.451–0.482) | 0.046 | +0.05 |
| SAM ViT-B | zero-shot | object + background points | 0.583 (0.567–0.599) | 0.090 | +0.17 |
| micro-SAM generalist (vit_b_lm) | zero-shot | object + background points | 0.703 (0.680–0.723) | 0.107 | +8.63 |
| boneseg, fine-tuned DINOv2 + prompts | 5-shot | object + background points | 0.769 (0.752–0.785) | 0.151 | +2.15 |
| boneseg (clicks) | zero-shot | noisy boxes | 0.589 (0.561–0.615) | 0.089 | -5.38 |
| SAM ViT-B | zero-shot | noisy boxes | 0.782 (0.765–0.798) | 0.332 | -4.88 |
| micro-SAM generalist (vit_b_lm) | zero-shot | noisy boxes | 0.904 (0.891–0.916) | 0.305 | -1.70 |
| boneseg, fine-tuned DINOv2 + prompts | 5-shot | noisy boxes | 0.827 (0.807–0.846) | 0.217 | -3.01 |

| Paired comparison | Difference in Dice (95% CI) | Images better |
|---|---|---|
