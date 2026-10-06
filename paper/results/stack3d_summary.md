# Whole stacks in 3D

Central 80% of each stack; measures on volumes reduced to 4 x 6.5 x 6.5 um voxels, the same for method and expert. Tb.Th, Tb.N and Tb.Sp use the plate model (Tb.Th = 2 BV/BS).

| Sample | Method | 3D Dice | BV/TV_% | BS/BV_per_mm | Tb.Th_um | Tb.N_per_mm | Tb.Sp_um | Slice r | Mean slice Dice |
|---|---|---|---|---|---|---|---|---|---|
| A (202 slices) | expert | – | 13.7 | 35.9 | 55.7 | 2.45 | 352 | – | – |
| A | boneseg, 25 + 25 clicks on one slice | 0.631 | 19.2 (+40%) | 31.9 (-11%) | 62.6 (+13%) | 3.06 (+25%) | 264 (-25%) | -0.67 | 0.629 |
| A | boneseg learned model, 5 slices of the sample | 0.707 | 20.5 (+50%) | 27.8 (-23%) | 71.9 (+29%) | 2.85 (+16%) | 279 (-21%) | 0.83 | 0.707 |
| A | boneseg learned model, slices of other samples | 0.648 | 21.4 (+57%) | 35.4 (-1%) | 56.4 (+1%) | 3.8 (+55%) | 207 (-41%) | 0.69 | 0.647 |
| C (505 slices) | expert | – | 5.06 | 34.1 | 58.6 | 0.863 | 1.1e+03 | – | – |
| C | boneseg, 25 + 25 clicks on one slice | 0.565 | 2.31 (-54%) | 47.4 (+39%) | 42.2 (-28%) | 0.547 (-37%) | 1.79e+03 (+62%) | 0.04 | 0.544 |
| C | boneseg learned model, 5 slices of the sample | 0.786 | 6.94 (+37%) | 36.8 (+8%) | 54.3 (-7%) | 1.28 (+48%) | 729 (-34%) | 0.85 | 0.782 |
| C | boneseg learned model, slices of other samples | 0.749 | 6.26 (+24%) | 34.4 (+1%) | 58.2 (-1%) | 1.08 (+25%) | 871 (-21%) | 0.78 | 0.739 |
| E (390 slices) | expert | – | 17.7 | 22.6 | 88.6 | 2 | 413 | – | – |
| E | boneseg, 25 + 25 clicks on one slice | 0.600 | 12.3 (-31%) | 42.6 (+89%) | 46.9 (-47%) | 2.61 (+31%) | 336 (-19%) | -0.88 | 0.599 |
| E | boneseg learned model, 5 slices of the sample | 0.842 | 19.8 (+12%) | 26.1 (+15%) | 76.8 (-13%) | 2.58 (+29%) | 311 (-25%) | 0.97 | 0.811 |
| E | boneseg learned model, slices of other samples | 0.816 | 21.6 (+22%) | 26.5 (+17%) | 75.5 (-15%) | 2.86 (+44%) | 274 (-34%) | 0.93 | 0.796 |
| F (233 slices) | expert | – | 7.65 | 27.7 | 72.1 | 1.06 | 871 | – | – |
| F | boneseg, 25 + 25 clicks on one slice | 0.624 | 4.16 (-46%) | 49.7 (+79%) | 40.2 (-44%) | 1.03 (-2%) | 927 (+6%) | 0.43 | 0.594 |
| F | boneseg learned model, 5 slices of the sample | 0.778 | 11 (+43%) | 32.1 (+16%) | 62.3 (-14%) | 1.76 (+66%) | 506 (-42%) | 0.96 | 0.780 |
| F | boneseg learned model, slices of other samples | 0.721 | 11.2 (+46%) | 31.8 (+15%) | 62.9 (-13%) | 1.78 (+67%) | 500 (-43%) | 0.91 | 0.719 |

## Mean over samples

| Method | 3D Dice | BV/TV_% difference | BS/BV_per_mm difference | Tb.Th_um difference | Tb.N_per_mm difference | Tb.Sp_um difference | Slice r |
|---|---|---|---|---|---|---|---|
| boneseg, 25 + 25 clicks on one slice | 0.605 | -23% (|43%|) | +49% (|55%|) | -27% (|33%|) | +4% (|24%|) | +6% (|28%|) | -0.27 |
| boneseg learned model, 5 slices of the sample | 0.778 | +36% (|36%|) | +4% (|15%|) | -1% (|16%|) | +40% (|40%|) | -30% (|30%|) | 0.90 |
| boneseg learned model, slices of other samples | 0.734 | +37% (|37%|) | +8% (|9%|) | -7% (|7%|) | +48% (|48%|) | -35% (|35%|) | 0.83 |
