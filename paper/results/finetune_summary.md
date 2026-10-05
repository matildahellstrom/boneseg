# Fine-tuning DINOv2: results

Samples A, C, E, F, leave-one-sample-out, test slices only. Click conditions use the backbone fine-tuned on the other samples, so the held-out sample was never seen in training.

| Condition | Method | A | C | E | F | Mean (95% CI) | B.Ar/T.Ar bias (points) |
|---|---|---|---|---|---|---|---|
| labels from same sample | Frozen DINOv2 + trained output layer | 0.649 | 0.785 | 0.789 | 0.764 | 0.747 (0.680–0.796) | +0.7 |
| labels from same sample | Fine-tuned DINOv2 (last 4 blocks) + output layer | 0.702 | 0.759 | 0.835 | 0.827 | 0.781 (0.723–0.834) | -0.6 |
| labels from other samples | Frozen DINOv2 + trained output layer | 0.640 | 0.784 | 0.724 | 0.721 | 0.717 (0.663–0.768) | +1.2 |
| labels from other samples | Fine-tuned DINOv2 (last 4 blocks) + output layer | 0.658 | 0.720 | 0.746 | 0.742 | 0.717 (0.677–0.750) | -0.3 |
| 3+6 clicks, clean | Clicks on frozen DINOv2 features | 0.609 | 0.685 | 0.717 | 0.684 | 0.674 (0.630–0.716) | +0.2 |
| 3+6 clicks, clean | Clicks on fine-tuned DINOv2 features | 0.660 | 0.753 | 0.812 | 0.756 | 0.745 (0.687–0.796) | +1.7 |
| 25+25 clicks, clean | Clicks on frozen DINOv2 features | 0.653 | 0.790 | 0.806 | 0.779 | 0.757 (0.688–0.804) | +2.0 |
| 25+25 clicks, clean | Clicks on fine-tuned DINOv2 features | 0.666 | 0.771 | 0.816 | 0.770 | 0.756 (0.696–0.805) | +2.0 |

## Fine-tuned minus frozen, paired on the same slices

| Condition | Difference in Dice (95% CI) | Slices improved |
|---|---|---|
| labels from same sample | +0.034 (-0.009 to +0.063) | 80% |
| labels from other samples | -0.001 (-0.045 to +0.031) | 55% |
| 3+6 clicks, clean | +0.071 (+0.048 to +0.097) | 82% |
| 25+25 clicks, clean | -0.001 (-0.021 to +0.015) | 50% |

## Training

| Held out | Labels from | Method | Best validation Dice | At step |
|---|---|---|---|---|
| A | same sample | probe_frozen | 0.614 | 300 |
| A | same sample | finetune_b4 | 0.632 | 100 |
| A | other samples | probe_frozen | 0.754 | 600 |
| A | other samples | finetune_b4 | 0.784 | 300 |
| C | same sample | probe_frozen | 0.813 | 700 |
| C | same sample | finetune_b4 | 0.796 | 500 |
| C | other samples | probe_frozen | 0.694 | 900 |
| C | other samples | finetune_b4 | 0.726 | 500 |
| E | same sample | probe_frozen | 0.722 | 400 |
| E | same sample | finetune_b4 | 0.799 | 200 |
| E | other samples | probe_frozen | 0.708 | 1000 |
| E | other samples | finetune_b4 | 0.722 | 300 |
| F | same sample | probe_frozen | 0.732 | 900 |
| F | same sample | finetune_b4 | 0.767 | 400 |
| F | other samples | probe_frozen | 0.722 | 700 |
| F | other samples | finetune_b4 | 0.737 | 100 |
