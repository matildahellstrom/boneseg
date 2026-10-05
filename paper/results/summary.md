# Evaluation results

Samples: A, E, F. Nested leave-one-sample-out; settings tuned on the other samples only. Method version: tag `method-v1`. Click seeds: [0, 1, 2]. Run time 31.4 min.

**Caveat:** with 3 samples the confidence intervals mostly reflect variation between these samples; they are not a substitute for more samples.

## Dice per method

| Method | Condition | A | E | F | Mean (95% CI) |
|---|---|---|---|---|---|
| Otsu threshold | no input | 0.177 | 0.464 | 0.396 | 0.346 (0.179–0.487) |
| Random forest, clicks (ilastik-style) | 25+25 clicks, clean | 0.458 | 0.708 | 0.548 | 0.571 (0.463–0.699) |
| SAM ViT-B, point prompts | 25+25 clicks, clean | 0.633 | 0.811 | 0.707 | 0.717 (0.636–0.807) |
| micro-SAM ViT-B LM, point prompts | 25+25 clicks, clean | 0.390 | 0.719 | 0.643 | 0.584 (0.403–0.718) |
| boneseg, clicks | 25+25 clicks, clean | 0.640 | 0.798 | 0.675 | 0.704 (0.642–0.794) |
| Random forest, clicks (ilastik-style) | 25+25 clicks, noisy | 0.406 | 0.642 | 0.478 | 0.509 (0.409–0.634) |
| SAM ViT-B, point prompts | 25+25 clicks, noisy | 0.614 | 0.782 | 0.610 | 0.669 (0.590–0.778) |
| micro-SAM ViT-B LM, point prompts | 25+25 clicks, noisy | 0.338 | 0.713 | 0.628 | 0.560 (0.346–0.710) |
| boneseg, clicks | 25+25 clicks, noisy | 0.631 | 0.798 | 0.690 | 0.706 (0.635–0.794) |
| Random forest, clicks (ilastik-style) | 3+6 clicks, clean | 0.325 | 0.588 | 0.441 | 0.451 (0.332–0.586) |
| SAM ViT-B, point prompts | 3+6 clicks, clean | 0.543 | 0.702 | 0.497 | 0.581 (0.497–0.695) |
| micro-SAM ViT-B LM, point prompts | 3+6 clicks, clean | 0.316 | 0.626 | 0.500 | 0.481 (0.323–0.620) |
| boneseg, clicks | 3+6 clicks, clean | 0.602 | 0.745 | 0.627 | 0.658 (0.597–0.741) |
| Random forest, clicks (ilastik-style) | 3+6 clicks, noisy | 0.264 | 0.507 | 0.379 | 0.383 (0.267–0.507) |
| SAM ViT-B, point prompts | 3+6 clicks, noisy | 0.454 | 0.661 | 0.476 | 0.530 (0.443–0.654) |
| micro-SAM ViT-B LM, point prompts | 3+6 clicks, noisy | 0.295 | 0.583 | 0.462 | 0.447 (0.302–0.578) |
| boneseg, clicks | 3+6 clicks, noisy | 0.543 | 0.624 | 0.618 | 0.595 (0.548–0.638) |
| Random forest, 5 labelled slices | labels from other samples | 0.299 | 0.610 | 0.543 | 0.484 (0.301–0.613) |
| boneseg learned model, 5 labelled slices | labels from other samples | 0.615 | 0.772 | 0.711 | 0.699 (0.621–0.774) |
| Random forest, 5 labelled slices | labels from same sample | 0.583 | 0.699 | 0.672 | 0.651 (0.586–0.720) |
| boneseg learned model, 5 labelled slices | labels from same sample | 0.713 | 0.812 | 0.780 | 0.768 (0.718–0.818) |

## Paired differences in Dice against boneseg with the same clicks

Positive means boneseg is better. Wins: share of test slices where boneseg scored higher.

| Compared with | Condition | Difference (95% CI) | Wins |
|---|---|---|---|
| Random forest, clicks (ilastik-style) | 25+25 clicks, clean | +0.133 (+0.087 to +0.182) | 97% |
| Random forest, clicks (ilastik-style) | 25+25 clicks, noisy | +0.198 (+0.154 to +0.236) | 100% |
| Random forest, clicks (ilastik-style) | 3+6 clicks, clean | +0.207 (+0.145 to +0.277) | 100% |
| Random forest, clicks (ilastik-style) | 3+6 clicks, noisy | +0.211 (+0.116 to +0.287) | 90% |
| SAM ViT-B, point prompts | 25+25 clicks, clean | -0.013 (-0.034 to +0.007) | 30% |
| SAM ViT-B, point prompts | 25+25 clicks, noisy | +0.038 (+0.008 to +0.095) | 70% |
| SAM ViT-B, point prompts | 3+6 clicks, clean | +0.078 (+0.030 to +0.131) | 73% |
| SAM ViT-B, point prompts | 3+6 clicks, noisy | +0.065 (-0.034 to +0.145) | 77% |
| micro-SAM ViT-B LM, point prompts | 25+25 clicks, clean | +0.120 (+0.031 to +0.237) | 87% |
| micro-SAM ViT-B LM, point prompts | 25+25 clicks, noisy | +0.147 (+0.054 to +0.285) | 90% |
| micro-SAM ViT-B LM, point prompts | 3+6 clicks, clean | +0.178 (+0.093 to +0.282) | 90% |
| micro-SAM ViT-B LM, point prompts | 3+6 clicks, noisy | +0.148 (+0.045 to +0.245) | 90% |
| Random forest, 5 labelled slices (vs boneseg learned model) | labels from other samples | +0.215 (+0.146 to +0.310) | 100% |
| Random forest, 5 labelled slices (vs boneseg learned model) | labels from same sample | +0.117 (+0.094 to +0.138) | 100% |

## Agreement of bone measures with the expert masks

Per test slice, pooled over samples. Bias = method minus expert; LoA = 95% limits of agreement.

| Method | Condition | Measure | Expert mean | Bias | LoA | ICC(A,1) | r |
|---|---|---|---|---|---|---|---|
| Otsu threshold | no input | B.Ar/T.Ar (%) | 13.1 | +14.5 | -2.9 to +31.9 | 0.11 | 0.32 |
| Otsu threshold | no input | B.Pm/T.Ar (1/mm) | 2.64 | +20.9 | +10.5 to +31.4 | 0.01 | 0.63 |
| Otsu threshold | no input | Tb.Th (µm) | 101 | -77.5 | -157 to +1.93 | 0.03 | 0.36 |
| Random forest, clicks (ilastik-style) | 25+25 clicks, clean | B.Ar/T.Ar (%) | 13.1 | +11.4 | +1.09 to +21.6 | 0.29 | 0.70 |
| Random forest, clicks (ilastik-style) | 25+25 clicks, clean | B.Pm/T.Ar (1/mm) | 2.64 | +12.5 | -4.46 to +29.4 | 0.05 | 0.70 |
| Random forest, clicks (ilastik-style) | 25+25 clicks, clean | Tb.Th (µm) | 101 | -56.9 | -89.4 to -24.3 | 0.41 | 0.95 |
| SAM ViT-B, point prompts | 25+25 clicks, clean | B.Ar/T.Ar (%) | 13.1 | +5.74 | -1.63 to +13.1 | 0.60 | 0.83 |
| SAM ViT-B, point prompts | 25+25 clicks, clean | B.Pm/T.Ar (1/mm) | 2.64 | +3.1 | -0.505 to +6.71 | 0.17 | 0.67 |
| SAM ViT-B, point prompts | 25+25 clicks, clean | Tb.Th (µm) | 101 | -30.1 | -90.6 to +30.4 | 0.41 | 0.73 |
| micro-SAM ViT-B LM, point prompts | 25+25 clicks, clean | B.Ar/T.Ar (%) | 13.1 | +13 | -20.4 to +46.3 | 0.19 | 0.44 |
| micro-SAM ViT-B LM, point prompts | 25+25 clicks, clean | B.Pm/T.Ar (1/mm) | 2.64 | +4.18 | -6.44 to +14.8 | 0.14 | 0.70 |
| micro-SAM ViT-B LM, point prompts | 25+25 clicks, clean | Tb.Th (µm) | 101 | -5.42 | -48.1 to +37.2 | 0.87 | 0.87 |
| boneseg, clicks | 25+25 clicks, clean | B.Ar/T.Ar (%) | 13.1 | +6.43 | +2.44 to +10.4 | 0.64 | 0.95 |
| boneseg, clicks | 25+25 clicks, clean | B.Pm/T.Ar (1/mm) | 2.64 | +0.699 | -0.342 to +1.74 | 0.71 | 0.92 |
| boneseg, clicks | 25+25 clicks, clean | Tb.Th (µm) | 101 | +22.3 | -7.09 to +51.7 | 0.83 | 0.94 |
| Random forest, 5 labelled slices | labels from same sample | B.Ar/T.Ar (%) | 13.1 | +8.58 | +0.126 to +17 | 0.45 | 0.81 |
| Random forest, 5 labelled slices | labels from same sample | B.Pm/T.Ar (1/mm) | 2.64 | +11.8 | -1.73 to +25.4 | 0.06 | 0.87 |
| Random forest, 5 labelled slices | labels from same sample | Tb.Th (µm) | 101 | -64.5 | -110 to -19.1 | 0.27 | 0.95 |
| boneseg learned model, 5 labelled slices | labels from same sample | B.Ar/T.Ar (%) | 13.1 | +3.71 | -1.44 to +8.85 | 0.81 | 0.93 |
| boneseg learned model, 5 labelled slices | labels from same sample | B.Pm/T.Ar (1/mm) | 2.64 | -0.0242 | -0.815 to +0.767 | 0.87 | 0.89 |
| boneseg learned model, 5 labelled slices | labels from same sample | Tb.Th (µm) | 101 | +26.4 | -3.57 to +56.4 | 0.79 | 0.94 |
| boneseg learned model, 5 labelled slices | labels from other samples | B.Ar/T.Ar (%) | 13.1 | +4.86 | -5.53 to +15.2 | 0.63 | 0.79 |
| boneseg learned model, 5 labelled slices | labels from other samples | B.Pm/T.Ar (1/mm) | 2.64 | +0.627 | -1.73 to +2.98 | 0.62 | 0.88 |
| boneseg learned model, 5 labelled slices | labels from other samples | Tb.Th (µm) | 101 | +17.9 | -10.4 to +46.3 | 0.87 | 0.94 |

## Figures

![Dice by method](../figures/dice_by_method.png)

![Bland-Altman, bone area](../figures/bland_altman_bone_area.png)

![Clicks and noise](../figures/clicks_and_noise.png)

