# boneseg against SAM, following Gu et al. (2025)

Paper-style prompts (paper/prompts.py) on the test images of the three datasets; 5-shot = 5 labelled images of the target (the held-out Liu sample or NOISe batch, or SegPC training images, three draws). Dice, NSD (bone 5 µm, cells 2 px) and the area bias (percentage points of the image), mean over images with 95% intervals.


## Bone

| Setting | Prompt | Method | Dice (95% CI) | NSD | Area bias |
|---|---|---|---|---|---|
| 5-shot | noisy boxes (paper) | boneseg, fine-tuned DINOv2 + prompts | 0.781 (0.737–0.821) | 0.186 | +3.08 |
| 5-shot | noisy boxes (paper) | MobileSAM ViT-T, enc+dec Adapter, boxes (paper's interactive recipe) | 0.740 (0.686–0.792) | 0.231 | -1.43 |
| 5-shot | no prompt | boneseg, fine-tuned DINOv2 (own output) | 0.781 (0.730–0.825) | 0.234 | -0.73 |
| 5-shot | no prompt | boneseg, fine-tuned DINOv2 + learned model | 0.815 (0.773–0.848) | 0.266 | +0.08 |
| 5-shot | no prompt | boneseg learned model | 0.789 (0.757–0.819) | 0.218 | +0.32 |
| 5-shot | no prompt | SAM ViT-B, decoder Adapter | 0.660 (0.434–0.818) | 0.209 | +1.78 |
| 5-shot | no prompt | SAM ViT-B, enc+dec Adapter (paper's few-shot recipe) | 0.841 (0.798–0.872) | 0.309 | +0.60 |
| 5-shot | no prompt | SAM ViT-B, enc+dec LoRA | 0.833 (0.784–0.868) | 0.302 | +0.68 |
| 5-shot | object + background points | boneseg, fine-tuned DINOv2 + prompts | 0.777 (0.704–0.829) | 0.219 | +3.24 |
| zero-shot | noisy boxes (paper) | boneseg | 0.461 (0.390–0.534) | 0.063 | +11.61 |
| zero-shot | noisy boxes (paper) | micro-SAM ViT-B LM | 0.714 (0.639–0.764) | 0.186 | +2.50 |
| zero-shot | noisy boxes (paper) | MobileSAM ViT-T | 0.637 (0.577–0.696) | 0.155 | +10.51 |
| zero-shot | noisy boxes (paper) | SAM ViT-B | 0.676 (0.597–0.757) | 0.207 | +5.75 |
| zero-shot | noisy boxes (paper) | SAM ViT-H | 0.707 (0.636–0.778) | 0.200 | +6.84 |
| zero-shot | object points only (paper) | micro-SAM ViT-B LM | 0.493 (0.355–0.590) | 0.080 | +18.22 |
| zero-shot | object points only (paper) | MobileSAM ViT-T | 0.285 (0.205–0.393) | 0.059 | +53.41 |
| zero-shot | object points only (paper) | SAM ViT-B | 0.548 (0.376–0.720) | 0.139 | +24.89 |
| zero-shot | object points only (paper) | SAM ViT-H | 0.325 (0.246–0.449) | 0.059 | +46.01 |
| zero-shot | object + background points | boneseg | 0.687 (0.625–0.743) | 0.167 | +2.52 |
| zero-shot | object + background points | micro-SAM ViT-B LM | 0.516 (0.378–0.614) | 0.110 | +12.79 |
| zero-shot | object + background points | MobileSAM ViT-T | 0.496 (0.445–0.549) | 0.105 | +16.89 |
| zero-shot | object + background points | SAM ViT-B | 0.653 (0.536–0.762) | 0.171 | +9.85 |
| zero-shot | object + background points | SAM ViT-H | 0.625 (0.541–0.704) | 0.146 | +9.52 |

| Paired comparison | Difference in Dice (95% CI) | Images better |
|---|---|---|
| boneseg vs SAM ViT-B, points + background, zero-shot | +0.033 (-0.059 to +0.126) | 60% |
| boneseg (points + background) vs SAM ViT-B (paper's points only) | +0.138 (-0.017 to +0.289) | 70% |
| boneseg vs SAM ViT-B, boxes, zero-shot | -0.215 (-0.291 to -0.125) | 12% |
| boneseg vs SAM ViT-H, points + background, zero-shot | +0.062 (-0.025 to +0.161) | 64% |
| boneseg (points + background) vs SAM ViT-H (paper's points only) | +0.362 (+0.216 to +0.479) | 94% |
| boneseg vs SAM ViT-H, boxes, zero-shot | -0.246 (-0.298 to -0.192) | 2% |
| boneseg vs micro-SAM ViT-B LM, points + background, zero-shot | +0.171 (+0.085 to +0.267) | 92% |
| boneseg (points + background) vs micro-SAM ViT-B LM (paper's points only) | +0.194 (+0.107 to +0.285) | 96% |
| boneseg vs micro-SAM ViT-B LM, boxes, zero-shot | -0.253 (-0.344 to -0.140) | 6% |
| boneseg vs MobileSAM ViT-T, points + background, zero-shot | +0.190 (+0.118 to +0.260) | 88% |
| boneseg (points + background) vs MobileSAM ViT-T (paper's points only) | +0.402 (+0.270 to +0.521) | 94% |
| boneseg vs MobileSAM ViT-T, boxes, zero-shot | -0.176 (-0.218 to -0.135) | 8% |
| boneseg learned model vs SAM ViT-B, enc+dec Adapter (paper's few-shot recipe), 5-shot, no prompt | -0.052 (-0.077 to -0.026) | 12% |
| boneseg fine-tuned DINOv2 vs SAM ViT-B, enc+dec Adapter (paper's few-shot recipe), 5-shot, no prompt | -0.060 (-0.083 to -0.038) | 20% |
| boneseg fine-tuned DINOv2 + learned model vs SAM ViT-B, enc+dec Adapter (paper's few-shot recipe), 5-shot, no prompt | -0.026 (-0.043 to -0.011) | 34% |
| boneseg learned model vs SAM ViT-B, decoder Adapter, 5-shot, no prompt | +0.129 (-0.025 to +0.371) | 64% |
| boneseg fine-tuned DINOv2 vs SAM ViT-B, decoder Adapter, 5-shot, no prompt | +0.120 (-0.037 to +0.373) | 58% |
| boneseg fine-tuned DINOv2 + learned model vs SAM ViT-B, decoder Adapter, 5-shot, no prompt | +0.154 (+0.003 to +0.404) | 72% |
| boneseg learned model vs SAM ViT-B, enc+dec LoRA, 5-shot, no prompt | -0.043 (-0.067 to -0.017) | 16% |
| boneseg fine-tuned DINOv2 vs SAM ViT-B, enc+dec LoRA, 5-shot, no prompt | -0.052 (-0.078 to -0.026) | 20% |
| boneseg fine-tuned DINOv2 + learned model vs SAM ViT-B, enc+dec LoRA, 5-shot, no prompt | -0.018 (-0.034 to -0.002) | 34% |
| boneseg fine-tuned + boxes vs MobileSAM fine-tuned + boxes, 5-shot | +0.041 (-0.002 to +0.092) | 70% |

## Osteoclasts

| Setting | Prompt | Method | Dice (95% CI) | NSD | Area bias |
|---|---|---|---|---|---|
| 5-shot | noisy boxes (paper) | boneseg, fine-tuned DINOv2 + prompts | 0.825 (0.758–0.875) | 0.389 | +0.16 |
| 5-shot | noisy boxes (paper) | MobileSAM ViT-T, enc+dec Adapter, boxes (paper's interactive recipe) | 0.191 (0.031–0.461) | 0.097 | +24.74 |
| 5-shot | no prompt | boneseg, fine-tuned DINOv2 (own output) | 0.652 (0.585–0.725) | 0.260 | -0.61 |
| 5-shot | no prompt | boneseg, fine-tuned DINOv2 + learned model | 0.721 (0.648–0.792) | 0.334 | -0.03 |
| 5-shot | no prompt | boneseg learned model | 0.662 (0.584–0.738) | 0.229 | -0.07 |
| 5-shot | no prompt | SAM ViT-B, decoder Adapter | 0.264 (0.000–0.559) | 0.121 | -1.75 |
| 5-shot | no prompt | SAM ViT-B, enc+dec Adapter (paper's few-shot recipe) | 0.704 (0.633–0.773) | 0.340 | -0.04 |
| 5-shot | no prompt | SAM ViT-B, enc+dec LoRA | 0.674 (0.568–0.770) | 0.351 | -0.29 |
| 5-shot | object + background points | boneseg, fine-tuned DINOv2 + prompts | 0.687 (0.583–0.776) | 0.283 | +1.43 |
| zero-shot | noisy boxes (paper) | boneseg | 0.752 (0.693–0.801) | 0.268 | +0.73 |
| zero-shot | noisy boxes (paper) | micro-SAM ViT-B LM | 0.884 (0.840–0.919) | 0.594 | +0.08 |
| zero-shot | noisy boxes (paper) | MobileSAM ViT-T | 0.879 (0.806–0.928) | 0.630 | -0.26 |
| zero-shot | noisy boxes (paper) | SAM ViT-B | 0.884 (0.812–0.933) | 0.655 | -0.27 |
| zero-shot | noisy boxes (paper) | SAM ViT-H | 0.879 (0.797–0.932) | 0.645 | -0.27 |
| zero-shot | object points only (paper) | micro-SAM ViT-B LM | 0.317 (0.237–0.397) | 0.144 | +31.12 |
| zero-shot | object points only (paper) | MobileSAM ViT-T | 0.305 (0.196–0.423) | 0.184 | +52.10 |
| zero-shot | object points only (paper) | SAM ViT-B | 0.551 (0.416–0.663) | 0.300 | +9.85 |
| zero-shot | object points only (paper) | SAM ViT-H | 0.363 (0.241–0.487) | 0.224 | +40.79 |
| zero-shot | object + background points | boneseg | 0.570 (0.472–0.668) | 0.190 | +1.66 |
| zero-shot | object + background points | micro-SAM ViT-B LM | 0.132 (0.097–0.175) | 0.028 | +52.99 |
| zero-shot | object + background points | MobileSAM ViT-T | 0.338 (0.231–0.447) | 0.132 | +19.13 |
| zero-shot | object + background points | SAM ViT-B | 0.325 (0.236–0.424) | 0.121 | +20.85 |
| zero-shot | object + background points | SAM ViT-H | 0.440 (0.314–0.553) | 0.185 | +11.75 |

| Paired comparison | Difference in Dice (95% CI) | Images better |
|---|---|---|
| boneseg vs SAM ViT-B, points + background, zero-shot | +0.245 (+0.197 to +0.289) | 88% |
| boneseg (points + background) vs SAM ViT-B (paper's points only) | +0.018 (-0.054 to +0.090) | 53% |
| boneseg vs SAM ViT-B, boxes, zero-shot | -0.132 (-0.177 to -0.098) | 2% |
| boneseg vs SAM ViT-H, points + background, zero-shot | +0.130 (+0.058 to +0.195) | 74% |
| boneseg (points + background) vs SAM ViT-H (paper's points only) | +0.207 (+0.091 to +0.322) | 67% |
| boneseg vs SAM ViT-H, boxes, zero-shot | -0.127 (-0.176 to -0.082) | 2% |
| boneseg vs micro-SAM ViT-B LM, points + background, zero-shot | +0.437 (+0.357 to +0.512) | 98% |
| boneseg (points + background) vs micro-SAM ViT-B LM (paper's points only) | +0.252 (+0.166 to +0.345) | 78% |
| boneseg vs micro-SAM ViT-B LM, boxes, zero-shot | -0.132 (-0.170 to -0.103) | 2% |
| boneseg vs MobileSAM ViT-T, points + background, zero-shot | +0.231 (+0.144 to +0.310) | 84% |
| boneseg (points + background) vs MobileSAM ViT-T (paper's points only) | +0.264 (+0.145 to +0.391) | 70% |
| boneseg vs MobileSAM ViT-T, boxes, zero-shot | -0.127 (-0.172 to -0.092) | 3% |
| boneseg learned model vs SAM ViT-B, enc+dec Adapter (paper's few-shot recipe), 5-shot, no prompt | -0.041 (-0.066 to -0.015) | 22% |
| boneseg fine-tuned DINOv2 vs SAM ViT-B, enc+dec Adapter (paper's few-shot recipe), 5-shot, no prompt | -0.052 (-0.113 to +0.008) | 39% |
| boneseg fine-tuned DINOv2 + learned model vs SAM ViT-B, enc+dec Adapter (paper's few-shot recipe), 5-shot, no prompt | +0.018 (-0.018 to +0.054) | 47% |
| boneseg learned model vs SAM ViT-B, decoder Adapter, 5-shot, no prompt | +0.399 (+0.112 to +0.678) | 74% |
| boneseg fine-tuned DINOv2 vs SAM ViT-B, decoder Adapter, 5-shot, no prompt | +0.389 (+0.088 to +0.679) | 70% |
| boneseg fine-tuned DINOv2 + learned model vs SAM ViT-B, decoder Adapter, 5-shot, no prompt | +0.458 (+0.158 to +0.750) | 78% |
| boneseg learned model vs SAM ViT-B, enc+dec LoRA, 5-shot, no prompt | -0.012 (-0.058 to +0.035) | 27% |
| boneseg fine-tuned DINOv2 vs SAM ViT-B, enc+dec LoRA, 5-shot, no prompt | -0.022 (-0.094 to +0.048) | 38% |
| boneseg fine-tuned DINOv2 + learned model vs SAM ViT-B, enc+dec LoRA, 5-shot, no prompt | +0.047 (+0.008 to +0.094) | 51% |
| boneseg fine-tuned + boxes vs MobileSAM fine-tuned + boxes, 5-shot | +0.634 (+0.366 to +0.839) | 93% |

## Plasma

| Setting | Prompt | Method | Dice (95% CI) | NSD | Area bias |
|---|---|---|---|---|---|
| 5-shot | noisy boxes (paper) | boneseg, fine-tuned DINOv2 + prompts | 0.827 (0.807–0.846) | 0.217 | -3.01 |
| 5-shot | noisy boxes (paper) | MobileSAM ViT-T, enc+dec Adapter, boxes (paper's interactive recipe) | 0.448 (0.427–0.468) | 0.070 | -2.63 |
| 5-shot | no prompt | boneseg, fine-tuned DINOv2 (own output) | 0.730 (0.708–0.751) | 0.208 | +1.83 |
| 5-shot | no prompt | boneseg, fine-tuned DINOv2 + learned model | 0.716 (0.692–0.738) | 0.176 | +1.19 |
| 5-shot | no prompt | boneseg learned model | 0.584 (0.561–0.607) | 0.092 | +6.26 |
| 5-shot | no prompt | SAM ViT-B, decoder Adapter | 0.504 (0.483–0.526) | 0.073 | +3.47 |
| 5-shot | no prompt | SAM ViT-B, enc+dec Adapter (paper's few-shot recipe) | 0.613 (0.591–0.634) | 0.126 | +3.92 |
| 5-shot | no prompt | SAM ViT-B, enc+dec LoRA | 0.610 (0.588–0.632) | 0.156 | +4.27 |
| 5-shot | object + background points | boneseg, fine-tuned DINOv2 + prompts | 0.769 (0.752–0.785) | 0.151 | +2.15 |
| full data | no prompt | SAM ViT-B, enc+dec Adapter (paper's few-shot recipe) | 0.806 (0.786–0.825) | 0.389 | +2.86 |
| zero-shot | noisy boxes (paper) | boneseg | 0.589 (0.561–0.615) | 0.089 | -5.38 |
| zero-shot | noisy boxes (paper) | micro-SAM ViT-B LM | 0.904 (0.891–0.916) | 0.305 | -1.70 |
| zero-shot | noisy boxes (paper) | MobileSAM ViT-T | 0.853 (0.836–0.869) | 0.371 | -3.58 |
| zero-shot | noisy boxes (paper) | SAM ViT-B | 0.782 (0.765–0.798) | 0.332 | -4.88 |
| zero-shot | noisy boxes (paper) | SAM ViT-H | 0.791 (0.773–0.809) | 0.344 | -4.80 |
| zero-shot | object points only (paper) | micro-SAM ViT-B LM | 0.764 (0.740–0.786) | 0.157 | +6.97 |
| zero-shot | object points only (paper) | MobileSAM ViT-T | 0.434 (0.404–0.468) | 0.076 | +45.88 |
| zero-shot | object points only (paper) | SAM ViT-B | 0.644 (0.621–0.667) | 0.167 | +10.33 |
| zero-shot | object points only (paper) | SAM ViT-H | 0.571 (0.544–0.597) | 0.142 | +18.64 |
| zero-shot | object + background points | boneseg | 0.466 (0.451–0.482) | 0.046 | +0.05 |
| zero-shot | object + background points | micro-SAM ViT-B LM | 0.703 (0.680–0.723) | 0.107 | +8.63 |
| zero-shot | object + background points | MobileSAM ViT-T | 0.575 (0.556–0.593) | 0.130 | +10.18 |
| zero-shot | object + background points | SAM ViT-B | 0.583 (0.567–0.599) | 0.090 | +0.17 |
| zero-shot | object + background points | SAM ViT-H | 0.599 (0.580–0.618) | 0.122 | +0.67 |

| Paired comparison | Difference in Dice (95% CI) | Images better |
|---|---|---|
| boneseg vs SAM ViT-B, points + background, zero-shot | -0.117 (-0.134 to -0.099) | 18% |
| boneseg (points + background) vs SAM ViT-B (paper's points only) | -0.178 (-0.199 to -0.155) | 15% |
| boneseg vs SAM ViT-B, boxes, zero-shot | -0.193 (-0.215 to -0.170) | 11% |
| boneseg vs SAM ViT-H, points + background, zero-shot | -0.133 (-0.152 to -0.113) | 17% |
| boneseg (points + background) vs SAM ViT-H (paper's points only) | -0.104 (-0.129 to -0.079) | 32% |
| boneseg vs SAM ViT-H, boxes, zero-shot | -0.202 (-0.225 to -0.181) | 8% |
| boneseg vs micro-SAM ViT-B LM, points + background, zero-shot | -0.236 (-0.260 to -0.212) | 7% |
| boneseg (points + background) vs micro-SAM ViT-B LM (paper's points only) | -0.297 (-0.324 to -0.272) | 3% |
| boneseg vs micro-SAM ViT-B LM, boxes, zero-shot | -0.315 (-0.337 to -0.293) | 0% |
| boneseg vs MobileSAM ViT-T, points + background, zero-shot | -0.109 (-0.126 to -0.089) | 21% |
| boneseg (points + background) vs MobileSAM ViT-T (paper's points only) | +0.032 (-0.002 to +0.064) | 69% |
| boneseg vs MobileSAM ViT-T, boxes, zero-shot | -0.264 (-0.286 to -0.244) | 1% |
| boneseg learned model vs SAM ViT-B, enc+dec Adapter (paper's few-shot recipe), 5-shot, no prompt | -0.029 (-0.039 to -0.017) | 34% |
| boneseg fine-tuned DINOv2 vs SAM ViT-B, enc+dec Adapter (paper's few-shot recipe), 5-shot, no prompt | +0.118 (+0.102 to +0.132) | 89% |
| boneseg fine-tuned DINOv2 + learned model vs SAM ViT-B, enc+dec Adapter (paper's few-shot recipe), 5-shot, no prompt | +0.103 (+0.087 to +0.118) | 86% |
| boneseg learned model vs SAM ViT-B, decoder Adapter, 5-shot, no prompt | +0.080 (+0.067 to +0.092) | 85% |
| boneseg fine-tuned DINOv2 vs SAM ViT-B, decoder Adapter, 5-shot, no prompt | +0.226 (+0.208 to +0.243) | 95% |
| boneseg fine-tuned DINOv2 + learned model vs SAM ViT-B, decoder Adapter, 5-shot, no prompt | +0.212 (+0.193 to +0.229) | 93% |
| boneseg learned model vs SAM ViT-B, enc+dec LoRA, 5-shot, no prompt | -0.026 (-0.040 to -0.012) | 33% |
| boneseg fine-tuned DINOv2 vs SAM ViT-B, enc+dec LoRA, 5-shot, no prompt | +0.120 (+0.104 to +0.136) | 89% |
| boneseg fine-tuned DINOv2 + learned model vs SAM ViT-B, enc+dec LoRA, 5-shot, no prompt | +0.106 (+0.090 to +0.122) | 83% |
| boneseg fine-tuned + boxes vs MobileSAM fine-tuned + boxes, 5-shot | +0.378 (+0.355 to +0.402) | 97% |
