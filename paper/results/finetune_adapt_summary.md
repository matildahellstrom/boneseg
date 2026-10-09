# Fine-tune on several samples, then adapt

Leave-one-sample-out over A, C, D, E, F; test slices of the held-out sample. 5 labelled slices of it (plus 5 for validation) are the only target labels. Dice with 95% intervals (hierarchical bootstrap over samples and slices), and the bias in bone area (B.Ar/T.Ar, percentage points).


## Scored by the own output layer

| Backbone | A | C | D | E | F | Mean (95% CI) | B.Ar/T.Ar bias |
|---|---|---|---|---|---|---|---|
| Frozen DINOv2 | 0.649 | 0.785 | 0.731 | 0.789 | 0.764 | 0.744 (0.690–0.787) | +0.1 |
| Fine-tuned on the sample's 5 slices | 0.702 | 0.759 | 0.781 | 0.835 | 0.827 | 0.781 (0.732–0.827) | -0.7 |
| Fine-tuned on the other samples | 0.639 | 0.805 | 0.645 | 0.715 | 0.722 | 0.705 (0.649–0.764) | +0.3 |
| Fine-tuned on the other samples, then the sample's 5 slices | 0.701 | 0.748 | 0.793 | 0.859 | 0.841 | 0.788 (0.730–0.842) | -0.8 |

Fine-tuned on the other samples, then the sample's 5 slices minus fine-tuned on the sample's 5 slices: +0.008 (-0.007 to +0.021), better on 72% of slices

Fine-tuned on the other samples minus fine-tuned on the sample's 5 slices: -0.075 (-0.129 to -0.006), better on 18% of slices

Frozen DINOv2 minus fine-tuned on the sample's 5 slices: -0.037 (-0.059 to +0.001), better on 16% of slices

## Scored by the learned model (calibrated)

| Backbone | A | C | D | E | F | Mean (95% CI) | B.Ar/T.Ar bias |
|---|---|---|---|---|---|---|---|
| Frozen DINOv2 | 0.744 | 0.810 | 0.769 | 0.804 | 0.819 | 0.789 (0.757–0.818) | +0.3 |
| Fine-tuned on the sample's 5 slices | 0.746 | 0.825 | 0.806 | 0.837 | 0.859 | 0.815 (0.773–0.850) | +0.1 |
| Fine-tuned on the other samples | 0.755 | 0.826 | 0.794 | 0.867 | 0.816 | 0.812 (0.775–0.847) | +0.7 |
| Fine-tuned on the other samples, then the sample's 5 slices | 0.752 | 0.820 | 0.817 | 0.859 | 0.865 | 0.823 (0.779–0.860) | +0.2 |

Fine-tuned on the other samples, then the sample's 5 slices minus fine-tuned on the sample's 5 slices: +0.008 (-0.003 to +0.020), better on 68% of slices

Fine-tuned on the other samples minus fine-tuned on the sample's 5 slices: -0.003 (-0.026 to +0.021), better on 42% of slices

Frozen DINOv2 minus fine-tuned on the sample's 5 slices: -0.025 (-0.039 to -0.010), better on 16% of slices

## Scored by the 3 + 6 clicks

| Backbone | A | C | D | E | F | Mean (95% CI) | B.Ar/T.Ar bias |
|---|---|---|---|---|---|---|---|
| Frozen DINOv2 | 0.609 | 0.685 | 0.624 | 0.717 | 0.684 | 0.664 (0.623–0.703) | +0.5 |
| Fine-tuned on the sample's 5 slices | 0.699 | 0.819 | 0.772 | 0.825 | 0.833 | 0.790 (0.736–0.832) | +1.1 |
| Fine-tuned on the other samples | 0.652 | 0.807 | 0.746 | 0.815 | 0.754 | 0.755 (0.697–0.803) | +1.8 |
| Fine-tuned on the other samples, then the sample's 5 slices | 0.725 | 0.792 | 0.791 | 0.854 | 0.831 | 0.799 (0.754–0.838) | +0.1 |

Fine-tuned on the other samples, then the sample's 5 slices minus fine-tuned on the sample's 5 slices: +0.009 (-0.012 to +0.029), better on 56% of slices

Fine-tuned on the other samples minus fine-tuned on the sample's 5 slices: -0.035 (-0.060 to -0.010), better on 22% of slices

Frozen DINOv2 minus fine-tuned on the sample's 5 slices: -0.126 (-0.150 to -0.099), better on 2% of slices
