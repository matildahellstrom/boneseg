# Osteoclasts on the NOISe mouse data

Batches m1, m2, m3, m4, m5, 20 test patches each (832 x 832 px brightfield, TRAP stain), leave-one-batch-out: every setting tuned on the other four batches. Cells count as found when a predicted cell overlaps an expert outline with IoU >= 0.5, one-to-one. Counts and areas are per patch.

| Method | Condition | Dice (95% CI) | Precision | Recall | F1 (range over batches) | Count bias | Count ICC | Area bias (points) | Area ICC |
|---|---|---|---|---|---|---|---|---|---|
| Otsu threshold | no input | 0.434 (0.322–0.559) | 0.585 | 0.407 | 0.472 (0.21–0.68) | -1.50 | 0.38 | +3.09 | 0.05 |
| boneseg, clicks | 10+20 clicks | 0.735 (0.637–0.818) | 0.608 | 0.735 | 0.664 (0.47–0.76) | +0.81 | 0.79 | +0.48 | 0.64 |
| SAM ViT-B, point prompts | 10+20 clicks | 0.534 (0.389–0.654) | 0.449 | 0.397 | 0.418 (0.16–0.55) | -0.85 | 0.46 | +10.02 | 0.08 |
| micro-SAM ViT-B LM, point prompts | 10+20 clicks | 0.326 (0.234–0.427) | 0.191 | 0.149 | 0.160 (0.06–0.29) | -1.31 | 0.23 | +36.53 | 0.03 |
| Random forest, clicks (ilastik-style) | 10+20 clicks | 0.626 (0.537–0.717) | 0.713 | 0.527 | 0.605 (0.44–0.74) | -1.17 | 0.64 | +0.07 | 0.93 |
| boneseg, clicks | 3+6 clicks | 0.683 (0.577–0.771) | 0.536 | 0.656 | 0.587 (0.38–0.70) | +0.62 | 0.68 | +0.33 | 0.74 |
| SAM ViT-B, point prompts | 3+6 clicks | 0.499 (0.360–0.612) | 0.370 | 0.355 | 0.360 (0.13–0.47) | -0.59 | 0.26 | +12.63 | 0.04 |
| micro-SAM ViT-B LM, point prompts | 3+6 clicks | 0.268 (0.192–0.346) | 0.143 | 0.091 | 0.110 (0.06–0.20) | -1.48 | 0.37 | +38.03 | 0.02 |
| Random forest, clicks (ilastik-style) | 3+6 clicks | 0.512 (0.436–0.595) | 0.572 | 0.424 | 0.485 (0.34–0.63) | -1.14 | 0.63 | +0.21 | 0.74 |
| boneseg learned model, 5 labelled patches | labels from other batches | 0.641 (0.573–0.712) | 0.466 | 0.764 | 0.558 (0.50–0.62) | +2.28 | 0.37 | +1.18 | 0.73 |
| Random forest, 5 labelled patches | labels from other batches | 0.564 (0.497–0.639) | 0.411 | 0.749 | 0.512 (0.42–0.60) | +3.29 | 0.23 | +2.59 | 0.45 |
| boneseg learned model, 5 labelled patches | labels from same batch | 0.668 (0.595–0.750) | 0.509 | 0.768 | 0.601 (0.52–0.72) | +1.80 | 0.51 | +0.83 | 0.86 |
| Random forest, 5 labelled patches | labels from same batch | 0.572 (0.488–0.661) | 0.427 | 0.767 | 0.540 (0.42–0.62) | +2.77 | 0.34 | +3.48 | 0.13 |

## Paired differences in Dice against boneseg, same patches and clicks

| Compared with | Condition | Difference (95% CI) | Wins |
|---|---|---|---|
| SAM ViT-B, point prompts | 10+20 clicks | +0.202 (+0.136 to +0.271) | 73% |
| SAM ViT-B, point prompts | 3+6 clicks | +0.183 (+0.116 to +0.248) | 71% |
| micro-SAM ViT-B LM, point prompts | 10+20 clicks | +0.409 (+0.320 to +0.497) | 87% |
| micro-SAM ViT-B LM, point prompts | 3+6 clicks | +0.415 (+0.334 to +0.493) | 91% |
| Random forest, clicks (ilastik-style) | 10+20 clicks | +0.110 (+0.069 to +0.152) | 78% |
| Random forest, clicks (ilastik-style) | 3+6 clicks | +0.171 (+0.108 to +0.233) | 79% |

## Dice per batch

| Method | Condition | m1 | m2 | m3 | m4 | m5 |
|---|---|---|---|---|---|---|
| Otsu threshold | no input | 0.343 | 0.465 | 0.443 | 0.627 | 0.294 |
| boneseg, clicks | 10+20 clicks | 0.744 | 0.745 | 0.781 | 0.844 | 0.562 |
| SAM ViT-B, point prompts | 10+20 clicks | 0.559 | 0.537 | 0.647 | 0.656 | 0.268 |
| micro-SAM ViT-B LM, point prompts | 10+20 clicks | 0.321 | 0.270 | 0.475 | 0.350 | 0.214 |
| Random forest, clicks (ilastik-style) | 10+20 clicks | 0.607 | 0.594 | 0.662 | 0.772 | 0.493 |
| boneseg, clicks | 3+6 clicks | 0.701 | 0.691 | 0.733 | 0.799 | 0.490 |
| SAM ViT-B, point prompts | 3+6 clicks | 0.527 | 0.536 | 0.620 | 0.570 | 0.244 |
| micro-SAM ViT-B LM, point prompts | 3+6 clicks | 0.289 | 0.234 | 0.357 | 0.296 | 0.164 |
| Random forest, clicks (ilastik-style) | 3+6 clicks | 0.460 | 0.527 | 0.526 | 0.635 | 0.413 |
| boneseg learned model, 5 labelled patches | labels from other batches | 0.586 | 0.655 | 0.688 | 0.718 | 0.556 |
| Random forest, 5 labelled patches | labels from other batches | 0.499 | 0.531 | 0.661 | 0.580 | 0.547 |
| boneseg learned model, 5 labelled patches | labels from same batch | 0.605 | 0.679 | 0.682 | 0.796 | 0.580 |
| Random forest, 5 labelled patches | labels from same batch | 0.507 | 0.687 | 0.537 | 0.643 | 0.487 |
