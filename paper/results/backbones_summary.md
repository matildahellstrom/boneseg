# Larger and newer backbones

Liu samples A, C, D, E, F, test slices; app default settings for every backbone. Dice with 95% intervals (hierarchical bootstrap), bone-area bias in percentage points, and the time to compute one slice's features on this Mac.


## 3 + 6 clicks

| Backbone | A | C | D | E | F | Mean (95% CI) | Bias | Seconds per slice | vs Small |
|---|---|---|---|---|---|---|---|---|---|
| DINOv2 Base | 0.552 | 0.679 | 0.588 | 0.691 | 0.617 | 0.625 (0.574–0.676) | -0.6 | 2.0 | -0.038 (-0.061 to -0.013) |
| DINOv2 Large | 0.601 | 0.682 | 0.613 | 0.709 | 0.664 | 0.654 (0.614–0.693) | -1.6 | 6.4 | -0.010 (-0.025 to +0.004) |
| DINOv2 Small | 0.609 | 0.685 | 0.624 | 0.717 | 0.684 | 0.664 (0.623–0.703) | +0.5 | 1.0 | – |

## 25 + 25 clicks

| Backbone | A | C | D | E | F | Mean (95% CI) | Bias | Seconds per slice | vs Small |
|---|---|---|---|---|---|---|---|---|---|
| DINOv2 Base | 0.673 | 0.773 | 0.742 | 0.789 | 0.748 | 0.745 (0.703–0.778) | +2.3 | 2.0 | -0.011 (-0.027 to +0.008) |
| DINOv2 Large | 0.690 | 0.764 | 0.779 | 0.829 | 0.789 | 0.770 (0.725–0.809) | +1.1 | 6.4 | +0.015 (-0.009 to +0.034) |
| DINOv2 Small | 0.653 | 0.790 | 0.750 | 0.806 | 0.779 | 0.755 (0.698–0.796) | +2.0 | 1.0 | – |

## Learned model, 5 labelled slices

| Backbone | A | C | D | E | F | Mean (95% CI) | Bias | Seconds per slice | vs Small |
|---|---|---|---|---|---|---|---|---|---|
| DINOv2 Base | 0.740 | 0.810 | 0.790 | 0.771 | 0.840 | 0.790 (0.751–0.824) | +0.7 | 2.0 | +0.001 (-0.021 to +0.017) |
| DINOv2 Large | 0.761 | 0.810 | 0.799 | 0.839 | 0.840 | 0.810 (0.776–0.839) | +0.3 | 6.4 | +0.020 (+0.006 to +0.034) |
| DINOv2 Small | 0.744 | 0.810 | 0.769 | 0.804 | 0.819 | 0.789 (0.757–0.818) | +0.3 | 1.0 | – |
