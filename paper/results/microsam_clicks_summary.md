# micro-SAM on the clicks

Test slices of the Liu samples, the evaluation's clicks (three seeds), app default settings.

| Method | Budget | Dice (95% CI) | B.Ar/T.Ar bias (points) |
|---|---|---|---|
| boneseg | 3+6 | 0.664 (0.623–0.703) | +0.48 |
| microsam | 3+6 | 0.484 (0.381–0.574) | +3.57 |
| microsam_agree | 3+6 | 0.534 (0.479–0.591) | -4.52 |
| sam | 3+6 | 0.617 (0.540–0.692) | +6.06 |
| sam_agree | 3+6 | 0.661 (0.625–0.694) | -2.41 |
| boneseg | 25+25 | 0.755 (0.698–0.796) | +2.02 |
| microsam | 25+25 | 0.612 (0.494–0.695) | +8.42 |
| microsam_agree | 25+25 | 0.724 (0.678–0.765) | -1.57 |
| sam | 25+25 | 0.757 (0.684–0.817) | +3.97 |
| sam_agree | 25+25 | 0.784 (0.731–0.818) | -0.29 |

| Comparison | Budget | Difference in Dice (95% CI) | Slices better |
|---|---|---|---|
| microsam vs boneseg | 3+6 | -0.180 (-0.250 to -0.122) | 4% |
| microsam_agree vs boneseg | 3+6 | -0.130 (-0.159 to -0.101) | 2% |
| microsam vs sam | 3+6 | -0.133 (-0.216 to -0.048) | 14% |
| microsam_agree vs sam_agree | 3+6 | -0.128 (-0.162 to -0.094) | 0% |
| microsam vs boneseg | 25+25 | -0.143 (-0.211 to -0.092) | 2% |
| microsam_agree vs boneseg | 25+25 | -0.032 (-0.050 to -0.013) | 24% |
| microsam vs sam | 25+25 | -0.145 (-0.209 to -0.089) | 8% |
| microsam_agree vs sam_agree | 25+25 | -0.061 (-0.088 to -0.036) | 6% |
