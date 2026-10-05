# Fine-tuning DINOv2 on the NOISe osteoclasts

Leave-one-batch-out, test patches only, colour input.

| Condition | Method | Dice (95% CI) | F1 at IoU 0.5 | Count bias |
|---|---|---|---|---|
| labels from other batches | probe_frozen | 0.618 (0.476–0.736) | 0.574 | -0.79 |
| labels from other batches | finetune_b4 | 0.653 (0.466–0.794) | 0.615 | -1.69 |
| 3+6 clicks | clicks_frozen | 0.683 (0.577–0.771) | 0.587 | +0.62 |
| 3+6 clicks | clicks_finetuned_b4 | 0.741 (0.641–0.828) | 0.652 | +0.32 |
| 10+20 clicks | clicks_frozen | 0.731 (0.637–0.808) | 0.673 | +0.27 |
| 10+20 clicks | clicks_finetuned_b4 | 0.737 (0.642–0.816) | 0.676 | +0.30 |

| Condition | Fine-tuned minus frozen, Dice (95% CI) | Patches improved |
|---|---|---|
| labels from other batches | +0.035 (-0.034 to +0.107) | 63% |
| 3+6 clicks | +0.058 (+0.033 to +0.086) | 78% |
| 10+20 clicks | +0.006 (-0.023 to +0.032) | 61% |
