# Fine-tuning DINOv2 on SegPC-2021

199 validation images. Official score without a size filter; F1 and counts with cells of at least 16000 px.

| Method | Condition | Official score (95% CI) | Cell F1 | Count bias per image | Count ICC |
|---|---|---|---|---|---|
| Clicks, DINOv2 fine-tuned on 100 images | 10+20 clicks | 0.554 (0.523–0.587) | 0.465 | -1.45 | 0.42 |
| Clicks, DINOv2 fine-tuned on 100 images | 3+6 clicks | 0.503 (0.474–0.533) | 0.404 | -1.42 | 0.35 |
| Clicks, DINOv2 fine-tuned on 20 images | 10+20 clicks | 0.579 (0.548–0.612) | 0.490 | -1.37 | 0.41 |
| Clicks, DINOv2 fine-tuned on 20 images | 3+6 clicks | 0.534 (0.503–0.567) | 0.441 | -1.33 | 0.35 |
| Clicks, frozen DINOv2 | 10+20 clicks | 0.550 (0.523–0.577) | 0.458 | +0.20 | 0.63 |
| Clicks, frozen DINOv2 | 3+6 clicks | 0.439 (0.416–0.465) | 0.303 | -0.25 | 0.41 |
| Fine-tuned DINOv2, 100 images | 100 labelled images | 0.496 (0.466–0.527) | 0.431 | -1.77 | 0.32 |
| Fine-tuned DINOv2, 20 images | 20 labelled images | 0.498 (0.469–0.530) | 0.476 | -1.73 | 0.39 |
| Output layer on frozen DINOv2, 100 images | 100 labelled images | 0.542 (0.518–0.565) | 0.460 | +0.19 | 0.59 |
| Output layer on frozen DINOv2, 20 images | 20 labelled images | 0.520 (0.496–0.545) | 0.445 | +0.05 | 0.50 |

| Comparison | Condition | Difference in official score (95% CI) | Images improved |
|---|---|---|---|
| Clicks, DINOv2 fine-tuned on 20 images vs Clicks, frozen DINOv2 | 10+20 clicks | +0.029 (+0.005 to +0.056) | 58% |
| Clicks, DINOv2 fine-tuned on 20 images vs Clicks, frozen DINOv2 | 3+6 clicks | +0.095 (+0.068 to +0.123) | 67% |
| Clicks, DINOv2 fine-tuned on 100 images vs Clicks, frozen DINOv2 | 10+20 clicks | +0.004 (-0.022 to +0.032) | 55% |
| Clicks, DINOv2 fine-tuned on 100 images vs Clicks, frozen DINOv2 | 3+6 clicks | +0.064 (+0.040 to +0.090) | 65% |
| Fine-tuned DINOv2, 20 images vs Output layer on frozen DINOv2, 20 images | 20 labelled images | -0.022 (-0.052 to +0.008) | 48% |
| Fine-tuned DINOv2, 100 images vs Output layer on frozen DINOv2, 100 images | 100 labelled images | -0.045 (-0.072 to -0.018) | 44% |
| Fine-tuned DINOv2, 100 images vs Fine-tuned DINOv2, 20 images | 100 labelled images | -0.002 (-0.023 to +0.021) | 39% |
