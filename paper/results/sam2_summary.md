# SAM 2 next to SAM

Liu test slices, the same clean clicks as combine_sam.py (three seeds). boneseg recomputed with the current defaults; SAM ViT-B rows from combine_sam.csv. Dice with 95% intervals, bone-area bias in points, and the paired difference to boneseg alone.


## 3+6 clicks

| Method | Dice (95% CI) | B.Ar/T.Ar bias | Minus boneseg (95% CI) | Slices better |
|---|---|---|---|---|
| boneseg | 0.664 (0.623–0.703) | +0.5 | – | – |
| SAM ViT-B | 0.617 (0.540–0.692) | +6.1 | -0.047 (-0.122 to +0.024) | 36% |
| boneseg ∩ SAM ViT-B (the app's Agree with SAM) | 0.661 (0.625–0.694) | -2.4 | -0.003 (-0.023 to +0.021) | 40% |
| SAM 2.1 Base+ | 0.663 (0.596–0.731) | +6.3 | -0.000 (-0.069 to +0.066) | 54% |
| SAM 2.1 Large | 0.583 (0.521–0.652) | +0.8 | -0.081 (-0.139 to -0.017) | 36% |
| boneseg ∩ SAM 2.1 Base+ | 0.687 (0.644–0.722) | -2.2 | +0.023 (-0.000 to +0.049) | 60% |
| boneseg ∩ SAM 2.1 Large | 0.580 (0.517–0.637) | -3.9 | -0.084 (-0.131 to -0.039) | 32% |

## 25+25 clicks

| Method | Dice (95% CI) | B.Ar/T.Ar bias | Minus boneseg (95% CI) | Slices better |
|---|---|---|---|---|
| boneseg | 0.755 (0.698–0.796) | +2.0 | – | – |
| SAM ViT-B | 0.757 (0.684–0.817) | +4.0 | +0.001 (-0.040 to +0.042) | 58% |
| boneseg ∩ SAM ViT-B (the app's Agree with SAM) | 0.784 (0.731–0.818) | -0.3 | +0.029 (+0.013 to +0.047) | 84% |
| SAM 2.1 Base+ | 0.480 (0.377–0.583) | +11.0 | -0.275 (-0.349 to -0.206) | 0% |
| SAM 2.1 Large | 0.714 (0.664–0.760) | +0.5 | -0.041 (-0.101 to +0.012) | 46% |
| boneseg ∩ SAM 2.1 Base+ | 0.633 (0.533–0.719) | -2.8 | -0.123 (-0.211 to -0.055) | 14% |
| boneseg ∩ SAM 2.1 Large | 0.719 (0.675–0.754) | -2.3 | -0.037 (-0.090 to +0.006) | 40% |
