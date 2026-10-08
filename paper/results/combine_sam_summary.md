# boneseg and SAM together

Test slices of the Liu samples, the evaluation's clicks (three seeds), app default settings.

| Method | Budget | Dice (95% CI) | B.Ar/T.Ar bias (points) |
|---|---|---|---|
| boneseg | 3+6 | 0.664 (0.623–0.703) | +0.48 |
| sam | 3+6 | 0.617 (0.540–0.692) | +6.06 |
| sam_boxes | 3+6 | 0.685 (0.624–0.740) | +2.18 |
| sam_maskprompt | 3+6 | 0.678 (0.636–0.717) | +0.65 |
| intersection | 3+6 | 0.661 (0.625–0.694) | -2.41 |
| union | 3+6 | 0.620 (0.551–0.692) | +8.95 |
| boneseg | 25+25 | 0.755 (0.698–0.796) | +2.02 |
| sam | 25+25 | 0.757 (0.684–0.817) | +3.97 |
| sam_boxes | 25+25 | 0.770 (0.713–0.814) | +2.71 |
| sam_maskprompt | 25+25 | 0.768 (0.707–0.811) | +2.30 |
| intersection | 25+25 | 0.784 (0.731–0.818) | -0.29 |
| union | 25+25 | 0.733 (0.664–0.791) | +6.28 |

| Comparison | Budget | Difference in Dice (95% CI) | Slices better |
|---|---|---|---|
| sam_boxes vs boneseg | 3+6 | +0.022 (-0.024 to +0.069) | 64% |
| sam_maskprompt vs boneseg | 3+6 | +0.015 (+0.008 to +0.021) | 96% |
| intersection vs boneseg | 3+6 | -0.003 (-0.023 to +0.021) | 40% |
| union vs boneseg | 3+6 | -0.044 (-0.107 to +0.013) | 44% |
| sam_boxes vs boneseg | 25+25 | +0.015 (-0.013 to +0.042) | 64% |
| sam_maskprompt vs boneseg | 25+25 | +0.013 (+0.007 to +0.019) | 100% |
| intersection vs boneseg | 25+25 | +0.029 (+0.013 to +0.047) | 84% |
| union vs boneseg | 25+25 | -0.022 (-0.053 to +0.005) | 42% |
