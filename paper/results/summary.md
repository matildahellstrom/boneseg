# Evaluation results

Samples: A, C, D, E, F. Nested leave-one-sample-out; settings tuned on the other samples only. Method version: tag `method-v1 and method-v2`. Click seeds: [0, 1, 2]. Run time 110.9 min.

## Dice per method

| Method | Condition | A | C | D | E | F | Mean (95% CI) |
|---|---|---|---|---|---|---|---|
| Otsu threshold | no input | 0.177 | 0.208 | 0.127 | 0.464 | 0.396 | 0.274 (0.161–0.399) |
| Random forest, clicks (ilastik-style) | 25+25 clicks, clean | 0.445 | 0.324 | 0.367 | 0.708 | 0.548 | 0.478 (0.363–0.608) |
| SAM ViT-B, point prompts | 25+25 clicks, clean | 0.633 | 0.813 | 0.819 | 0.811 | 0.707 | 0.757 (0.684–0.817) |
| micro-SAM ViT-B LM, point prompts | 25+25 clicks, clean | 0.390 | 0.667 | 0.642 | 0.719 | 0.643 | 0.612 (0.494–0.695) |
| boneseg, clicks | 25+25 clicks, clean | 0.621 | 0.644 | 0.756 | 0.779 | 0.675 | 0.695 (0.640–0.752) |
| boneseg v2, clicks | 25+25 clicks, clean | 0.653 | 0.790 | 0.750 | 0.806 | 0.779 | 0.755 (0.698–0.796) |
| boneseg v2 + refiner, clicks | 25+25 clicks, clean | 0.671 | 0.758 | 0.630 | 0.826 | 0.664 | 0.710 (0.651–0.778) |
| Random forest, clicks (ilastik-style) | 25+25 clicks, noisy | 0.391 | 0.268 | 0.312 | 0.642 | 0.478 | 0.418 (0.306–0.542) |
| SAM ViT-B, point prompts | 25+25 clicks, noisy | 0.614 | 0.754 | 0.805 | 0.782 | 0.610 | 0.713 (0.637–0.785) |
| micro-SAM ViT-B LM, point prompts | 25+25 clicks, noisy | 0.338 | 0.628 | 0.585 | 0.713 | 0.628 | 0.578 (0.448–0.675) |
| boneseg, clicks | 25+25 clicks, noisy | 0.629 | 0.638 | 0.647 | 0.766 | 0.690 | 0.674 (0.632–0.725) |
| boneseg v2, clicks | 25+25 clicks, noisy | 0.609 | 0.655 | 0.611 | 0.779 | 0.714 | 0.674 (0.616–0.736) |
| boneseg v2 + refiner, clicks | 25+25 clicks, noisy | 0.645 | 0.627 | 0.543 | 0.825 | 0.598 | 0.648 (0.572–0.743) |
| Random forest, clicks (ilastik-style) | 3+6 clicks, clean | 0.325 | 0.264 | 0.248 | 0.605 | 0.451 | 0.379 (0.269–0.512) |
| SAM ViT-B, point prompts | 3+6 clicks, clean | 0.543 | 0.651 | 0.691 | 0.702 | 0.497 | 0.617 (0.540–0.692) |
| micro-SAM ViT-B LM, point prompts | 3+6 clicks, clean | 0.316 | 0.531 | 0.447 | 0.626 | 0.500 | 0.484 (0.381–0.574) |
| boneseg, clicks | 3+6 clicks, clean | 0.602 | 0.583 | 0.634 | 0.745 | 0.627 | 0.638 (0.591–0.697) |
| boneseg v2, clicks | 3+6 clicks, clean | 0.609 | 0.685 | 0.609 | 0.717 | 0.671 | 0.658 (0.616–0.701) |
| boneseg v2 + refiner, clicks | 3+6 clicks, clean | 0.659 | 0.671 | 0.590 | 0.760 | 0.650 | 0.666 (0.619–0.719) |
| Random forest, clicks (ilastik-style) | 3+6 clicks, noisy | 0.264 | 0.234 | 0.231 | 0.510 | 0.386 | 0.325 (0.240–0.428) |
| SAM ViT-B, point prompts | 3+6 clicks, noisy | 0.454 | 0.608 | 0.654 | 0.661 | 0.476 | 0.571 (0.487–0.651) |
| micro-SAM ViT-B LM, point prompts | 3+6 clicks, noisy | 0.295 | 0.470 | 0.444 | 0.583 | 0.462 | 0.451 (0.359–0.533) |
| boneseg, clicks | 3+6 clicks, noisy | 0.543 | 0.607 | 0.606 | 0.624 | 0.618 | 0.600 (0.565–0.631) |
| boneseg v2, clicks | 3+6 clicks, noisy | 0.498 | 0.649 | 0.555 | 0.592 | 0.645 | 0.588 (0.528–0.640) |
| boneseg v2 + refiner, clicks | 3+6 clicks, noisy | 0.594 | 0.661 | 0.549 | 0.670 | 0.630 | 0.621 (0.573–0.663) |
| Random forest, 5 labelled slices | labels from other samples | 0.292 | 0.375 | 0.364 | 0.584 | 0.600 | 0.443 (0.327–0.556) |
| boneseg learned model, 5 labelled slices | labels from other samples | 0.648 | 0.724 | 0.722 | 0.776 | 0.755 | 0.725 (0.679–0.766) |
| boneseg v2 learned model, 5 labelled slices | labels from other samples | 0.650 | 0.768 | 0.722 | 0.801 | 0.739 | 0.736 (0.684–0.782) |
| Random forest, 5 labelled slices | labels from same sample | 0.583 | 0.463 | 0.505 | 0.699 | 0.672 | 0.585 (0.498–0.666) |
| boneseg learned model, 5 labelled slices | labels from same sample | 0.713 | 0.797 | 0.734 | 0.812 | 0.780 | 0.767 (0.729–0.802) |
| boneseg v2 learned model, 5 labelled slices | labels from same sample | 0.698 | 0.787 | 0.734 | 0.807 | 0.785 | 0.762 (0.721–0.801) |

## Paired differences in Dice against boneseg with the same clicks

Positive means boneseg is better. Wins: share of test slices where boneseg scored higher.

| Compared with | Condition | Difference (95% CI) | Wins |
|---|---|---|---|
| Random forest, clicks (ilastik-style) | 25+25 clicks, clean | +0.217 (+0.118 to +0.324) | 96% |
| Random forest, clicks (ilastik-style) | 25+25 clicks, noisy | +0.256 (+0.176 to +0.337) | 100% |
| Random forest, clicks (ilastik-style) | 3+6 clicks, clean | +0.260 (+0.177 to +0.343) | 100% |
| Random forest, clicks (ilastik-style) | 3+6 clicks, noisy | +0.274 (+0.184 to +0.360) | 94% |
| SAM ViT-B, point prompts | 25+25 clicks, clean | -0.061 (-0.119 to -0.021) | 12% |
| SAM ViT-B, point prompts | 25+25 clicks, noisy | -0.039 (-0.116 to +0.037) | 36% |
| SAM ViT-B, point prompts | 3+6 clicks, clean | +0.022 (-0.051 to +0.089) | 58% |
| SAM ViT-B, point prompts | 3+6 clicks, noisy | +0.029 (-0.037 to +0.097) | 64% |
| micro-SAM ViT-B LM, point prompts | 25+25 clicks, clean | +0.083 (+0.008 to +0.170) | 72% |
| micro-SAM ViT-B LM, point prompts | 25+25 clicks, noisy | +0.096 (+0.024 to +0.201) | 72% |
| micro-SAM ViT-B LM, point prompts | 3+6 clicks, clean | +0.155 (+0.083 to +0.235) | 86% |
| micro-SAM ViT-B LM, point prompts | 3+6 clicks, noisy | +0.149 (+0.082 to +0.212) | 92% |
| Random forest, 5 labelled slices (vs boneseg learned model) | labels from other samples | +0.282 (+0.201 to +0.361) | 100% |
| Random forest, 5 labelled slices (vs boneseg learned model) | labels from same sample | +0.183 (+0.115 to +0.275) | 100% |

## Paired differences in Dice for method-v2

Positive means the first method is better. Wins: share of test slices where it scored higher.

| Method | Compared with | Condition | Difference (95% CI) | Wins |
|---|---|---|---|---|
| boneseg v2, clicks | boneseg, clicks | 25+25 clicks, clean | +0.060 (+0.014 to +0.110) | 78% |
| boneseg v2, clicks | boneseg, clicks | 25+25 clicks, noisy | -0.000 (-0.024 to +0.022) | 48% |
| boneseg v2, clicks | boneseg, clicks | 3+6 clicks, clean | +0.020 (-0.020 to +0.069) | 52% |
| boneseg v2, clicks | boneseg, clicks | 3+6 clicks, noisy | -0.012 (-0.050 to +0.027) | 46% |
| boneseg v2 + refiner, clicks | boneseg v2, clicks | 25+25 clicks, clean | -0.046 (-0.101 to +0.010) | 40% |
| boneseg v2 + refiner, clicks | boneseg v2, clicks | 25+25 clicks, noisy | -0.026 (-0.081 to +0.030) | 50% |
| boneseg v2 + refiner, clicks | boneseg v2, clicks | 3+6 clicks, clean | +0.008 (-0.023 to +0.038) | 66% |
| boneseg v2 + refiner, clicks | boneseg v2, clicks | 3+6 clicks, noisy | +0.033 (-0.010 to +0.079) | 72% |
| boneseg v2 + refiner, clicks | SAM ViT-B, point prompts | 25+25 clicks, clean | -0.047 (-0.120 to +0.017) | 38% |
| boneseg v2 + refiner, clicks | SAM ViT-B, point prompts | 25+25 clicks, noisy | -0.065 (-0.175 to +0.028) | 38% |
| boneseg v2 + refiner, clicks | SAM ViT-B, point prompts | 3+6 clicks, clean | +0.049 (-0.037 to +0.122) | 72% |
| boneseg v2 + refiner, clicks | SAM ViT-B, point prompts | 3+6 clicks, noisy | +0.050 (-0.039 to +0.132) | 64% |
| boneseg v2 + refiner, clicks | micro-SAM ViT-B LM, point prompts | 25+25 clicks, clean | +0.098 (+0.018 to +0.202) | 74% |
| boneseg v2 + refiner, clicks | micro-SAM ViT-B LM, point prompts | 25+25 clicks, noisy | +0.069 (-0.033 to +0.198) | 60% |
| boneseg v2 + refiner, clicks | micro-SAM ViT-B LM, point prompts | 3+6 clicks, clean | +0.182 (+0.124 to +0.269) | 96% |
| boneseg v2 + refiner, clicks | micro-SAM ViT-B LM, point prompts | 3+6 clicks, noisy | +0.170 (+0.103 to +0.248) | 92% |
| boneseg v2 + refiner, clicks | Random forest, clicks (ilastik-style) | 25+25 clicks, clean | +0.232 (+0.138 to +0.351) | 96% |
| boneseg v2 + refiner, clicks | Random forest, clicks (ilastik-style) | 25+25 clicks, noisy | +0.229 (+0.161 to +0.311) | 98% |
| boneseg v2 + refiner, clicks | Random forest, clicks (ilastik-style) | 3+6 clicks, clean | +0.287 (+0.196 to +0.372) | 100% |
| boneseg v2 + refiner, clicks | Random forest, clicks (ilastik-style) | 3+6 clicks, noisy | +0.296 (+0.213 to +0.381) | 96% |
| boneseg v2 learned model, 5 labelled slices | boneseg learned model, 5 labelled slices | labels from other samples | +0.011 (-0.007 to +0.032) | 50% |
| boneseg v2 learned model, 5 labelled slices | boneseg learned model, 5 labelled slices | labels from same sample | -0.005 (-0.014 to +0.003) | 36% |

## Agreement of bone measures with the expert masks

Per test slice, pooled over samples. Bias = method minus expert; LoA = 95% limits of agreement.

| Method | Condition | Measure | Expert mean | Bias | LoA | ICC(A,1) | r |
|---|---|---|---|---|---|---|---|
| Otsu threshold | no input | B.Ar/T.Ar (%) | 10.6 | +17.6 | -0.778 to +36 | 0.02 | 0.08 |
| Otsu threshold | no input | B.Pm/T.Ar (1/mm) | 2.26 | +27.9 | +4.89 to +50.9 | 0.00 | 0.05 |
| Otsu threshold | no input | Tb.Th (µm) | 93.6 | -72.7 | -136 to -9.93 | 0.03 | 0.41 |
| Random forest, clicks (ilastik-style) | 25+25 clicks, clean | B.Ar/T.Ar (%) | 10.6 | +14.2 | +2.57 to +25.7 | 0.16 | 0.57 |
| Random forest, clicks (ilastik-style) | 25+25 clicks, clean | B.Pm/T.Ar (1/mm) | 2.26 | +14.4 | -3.08 to +32 | 0.03 | 0.59 |
| Random forest, clicks (ilastik-style) | 25+25 clicks, clean | Tb.Th (µm) | 93.6 | -54.4 | -84.1 to -24.7 | 0.33 | 0.92 |
| SAM ViT-B, point prompts | 25+25 clicks, clean | B.Ar/T.Ar (%) | 10.6 | +3.97 | -3.22 to +11.2 | 0.73 | 0.88 |
| SAM ViT-B, point prompts | 25+25 clicks, clean | B.Pm/T.Ar (1/mm) | 2.26 | +2.06 | -1.72 to +5.83 | 0.31 | 0.77 |
| SAM ViT-B, point prompts | 25+25 clicks, clean | Tb.Th (µm) | 93.6 | -20.6 | -74.2 to +32.9 | 0.39 | 0.62 |
| micro-SAM ViT-B LM, point prompts | 25+25 clicks, clean | B.Ar/T.Ar (%) | 10.6 | +8.42 | -19.8 to +36.7 | 0.30 | 0.59 |
| micro-SAM ViT-B LM, point prompts | 25+25 clicks, clean | B.Pm/T.Ar (1/mm) | 2.26 | +2.66 | -6.34 to +11.7 | 0.19 | 0.71 |
| micro-SAM ViT-B LM, point prompts | 25+25 clicks, clean | Tb.Th (µm) | 93.6 | -2.44 | -39.2 to +34.3 | 0.85 | 0.85 |
| boneseg, clicks | 25+25 clicks, clean | B.Ar/T.Ar (%) | 10.6 | +5.71 | -0.787 to +12.2 | 0.65 | 0.90 |
| boneseg, clicks | 25+25 clicks, clean | B.Pm/T.Ar (1/mm) | 2.26 | +0.92 | -0.577 to +2.42 | 0.65 | 0.94 |
| boneseg, clicks | 25+25 clicks, clean | Tb.Th (µm) | 93.6 | +13.8 | -25.5 to +53.1 | 0.76 | 0.82 |
| boneseg v2, clicks | 25+25 clicks, clean | B.Ar/T.Ar (%) | 10.6 | +2.02 | -4.76 to +8.8 | 0.83 | 0.89 |
| boneseg v2, clicks | 25+25 clicks, clean | B.Pm/T.Ar (1/mm) | 2.26 | +0.551 | -0.361 to +1.46 | 0.80 | 0.95 |
| boneseg v2, clicks | 25+25 clicks, clean | Tb.Th (µm) | 93.6 | -7.36 | -43.4 to +28.6 | 0.79 | 0.86 |
| boneseg v2 + refiner, clicks | 25+25 clicks, clean | B.Ar/T.Ar (%) | 10.6 | +4.72 | -3.08 to +12.5 | 0.62 | 0.80 |
| boneseg v2 + refiner, clicks | 25+25 clicks, clean | B.Pm/T.Ar (1/mm) | 2.26 | +1.08 | -1.36 to +3.52 | 0.29 | 0.45 |
| boneseg v2 + refiner, clicks | 25+25 clicks, clean | Tb.Th (µm) | 93.6 | +2.13 | -27.8 to +32 | 0.89 | 0.90 |
| boneseg, clicks | 3+6 clicks, clean | B.Ar/T.Ar (%) | 10.6 | +4.42 | -3.88 to +12.7 | 0.61 | 0.77 |
| boneseg, clicks | 3+6 clicks, clean | B.Pm/T.Ar (1/mm) | 2.26 | +0.593 | -0.331 to +1.52 | 0.76 | 0.90 |
| boneseg, clicks | 3+6 clicks, clean | Tb.Th (µm) | 93.6 | +14.7 | -41.4 to +70.7 | 0.61 | 0.66 |
| boneseg v2, clicks | 3+6 clicks, clean | B.Ar/T.Ar (%) | 10.6 | +1.43 | -5.85 to +8.7 | 0.78 | 0.80 |
| boneseg v2, clicks | 3+6 clicks, clean | B.Pm/T.Ar (1/mm) | 2.26 | +0.403 | -0.758 to +1.56 | 0.74 | 0.80 |
| boneseg v2, clicks | 3+6 clicks, clean | Tb.Th (µm) | 93.6 | -6.27 | -48.8 to +36.3 | 0.68 | 0.82 |
| boneseg v2 + refiner, clicks | 3+6 clicks, clean | B.Ar/T.Ar (%) | 10.6 | +4.01 | -3.46 to +11.5 | 0.64 | 0.79 |
| boneseg v2 + refiner, clicks | 3+6 clicks, clean | B.Pm/T.Ar (1/mm) | 2.26 | +0.831 | -1.07 to +2.73 | 0.32 | 0.45 |
| boneseg v2 + refiner, clicks | 3+6 clicks, clean | Tb.Th (µm) | 93.6 | +3.66 | -22.4 to +29.8 | 0.91 | 0.92 |
| Random forest, 5 labelled slices | labels from same sample | B.Ar/T.Ar (%) | 10.6 | +9.68 | -0.179 to +19.5 | 0.36 | 0.74 |
| Random forest, 5 labelled slices | labels from same sample | B.Pm/T.Ar (1/mm) | 2.26 | +14.3 | -0.282 to +28.9 | 0.03 | 0.53 |
| Random forest, 5 labelled slices | labels from same sample | Tb.Th (µm) | 93.6 | -64.1 | -100 to -28 | 0.21 | 0.93 |
| boneseg learned model, 5 labelled slices | labels from same sample | B.Ar/T.Ar (%) | 10.6 | +2.92 | -2.07 to +7.91 | 0.85 | 0.95 |
| boneseg learned model, 5 labelled slices | labels from same sample | B.Pm/T.Ar (1/mm) | 2.26 | +0.173 | -0.8 to +1.15 | 0.83 | 0.84 |
| boneseg learned model, 5 labelled slices | labels from same sample | Tb.Th (µm) | 93.6 | +15.5 | -22.7 to +53.7 | 0.80 | 0.88 |
| boneseg v2 learned model, 5 labelled slices | labels from same sample | B.Ar/T.Ar (%) | 10.6 | +3.43 | -1.15 to +8.01 | 0.82 | 0.94 |
| boneseg v2 learned model, 5 labelled slices | labels from same sample | B.Pm/T.Ar (1/mm) | 2.26 | +0.407 | -0.452 to +1.27 | 0.81 | 0.89 |
| boneseg v2 learned model, 5 labelled slices | labels from same sample | Tb.Th (µm) | 93.6 | +11.3 | -19 to +41.6 | 0.85 | 0.90 |
| boneseg learned model, 5 labelled slices | labels from other samples | B.Ar/T.Ar (%) | 10.6 | +2.43 | -4.55 to +9.41 | 0.81 | 0.87 |
| boneseg learned model, 5 labelled slices | labels from other samples | B.Pm/T.Ar (1/mm) | 2.26 | +0.487 | -0.997 to +1.97 | 0.76 | 0.91 |
| boneseg learned model, 5 labelled slices | labels from other samples | Tb.Th (µm) | 93.6 | +3.31 | -39.8 to +46.4 | 0.71 | 0.78 |
| boneseg v2 learned model, 5 labelled slices | labels from other samples | B.Ar/T.Ar (%) | 10.6 | +3.02 | -2.89 to +8.94 | 0.82 | 0.92 |
| boneseg v2 learned model, 5 labelled slices | labels from other samples | B.Pm/T.Ar (1/mm) | 2.26 | +0.453 | -0.802 to +1.71 | 0.79 | 0.92 |
| boneseg v2 learned model, 5 labelled slices | labels from other samples | Tb.Th (µm) | 93.6 | +8.2 | -16.8 to +33.2 | 0.89 | 0.93 |

## Figures

![Dice by method](../figures/dice_by_method.png)

![Bland-Altman, bone area](../figures/bland_altman_bone_area.png)

![Clicks and noise](../figures/clicks_and_noise.png)

