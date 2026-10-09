# Whole stacks in 3D

Central 80% of each stack; measures on volumes reduced to 4 x 6.5 x 6.5 um voxels, the same for method and expert. Tb.Th, Tb.N and Tb.Sp use the plate model (Tb.Th = 2 BV/BS).

| Sample | Method | 3D Dice | BV/TV_% | BS/BV_per_mm | Tb.Th_um | Tb.N_per_mm | Tb.Sp_um | Slice r | Mean slice Dice |
|---|---|---|---|---|---|---|---|---|---|
| A (202 slices) | expert | – | 13.7 | 35.9 | 55.7 | 2.45 | 352 | – | – |
| A | boneseg, 25 + 25 clicks on one slice | 0.631 | 19.2 (+40%) | 31.9 (-11%) | 62.6 (+13%) | 3.06 (+25%) | 264 (-25%) | -0.67 | 0.629 |
| A | boneseg learned model, 5 slices of the sample | 0.707 | 20.5 (+50%) | 27.8 (-23%) | 71.9 (+29%) | 2.85 (+16%) | 279 (-21%) | 0.83 | 0.707 |
| A |   + calibrated threshold | 0.744 | 14.5 (+6%) | 30.3 (-16%) | 66 (+19%) | 2.19 (-11%) | 390 (+11%) | 0.74 | 0.745 |
| A |   + calibrated, smoothed along z | 0.745 | 14.4 (+5%) | 28.1 (-22%) | 71.2 (+28%) | 2.02 (-18%) | 425 (+21%) | 0.75 | 0.746 |
| A |   + calibrated, features of neighbouring slices | 0.751 | 13.6 (-0%) | 28.1 (-22%) | 71.1 (+28%) | 1.92 (-22%) | 450 (+28%) | 0.81 | 0.752 |
| A |   + calibrated, neighbouring slices, smoothed | 0.751 | 13.6 (-0%) | 27.8 (-23%) | 72 (+29%) | 1.89 (-23%) | 457 (+30%) | 0.81 | 0.753 |
| A | boneseg learned model, slices of other samples | 0.633 | 23.1 (+69%) | 34.8 (-3%) | 57.4 (+3%) | 4.02 (+64%) | 191 (-46%) | 0.72 | 0.632 |
| A |   + calibrated threshold | 0.660 | 17 (+24%) | 41.6 (+16%) | 48.1 (-14%) | 3.53 (+44%) | 235 (-33%) | 0.44 | 0.661 |
| C (505 slices) | expert | – | 5.06 | 34.1 | 58.6 | 0.863 | 1.1e+03 | – | – |
| C | boneseg, 25 + 25 clicks on one slice | 0.565 | 2.31 (-54%) | 47.4 (+39%) | 42.2 (-28%) | 0.547 (-37%) | 1.79e+03 (+62%) | 0.04 | 0.544 |
| C | boneseg learned model, 5 slices of the sample | 0.786 | 6.94 (+37%) | 36.8 (+8%) | 54.3 (-7%) | 1.28 (+48%) | 729 (-34%) | 0.85 | 0.782 |
| C |   + calibrated threshold | 0.823 | 5.11 (+1%) | 40.3 (+18%) | 49.6 (-15%) | 1.03 (+19%) | 921 (-16%) | 0.83 | 0.822 |
| C |   + calibrated, smoothed along z | 0.824 | 5.06 (+0%) | 37.3 (+9%) | 53.6 (-9%) | 0.945 (+10%) | 1e+03 (-9%) | 0.83 | 0.824 |
| C |   + calibrated, features of neighbouring slices | 0.826 | 5.12 (+1%) | 36.1 (+6%) | 55.4 (-6%) | 0.925 (+7%) | 1.03e+03 (-7%) | 0.83 | 0.826 |
| C |   + calibrated, neighbouring slices, smoothed | 0.827 | 5.1 (+1%) | 35.7 (+5%) | 56 (-4%) | 0.91 (+6%) | 1.04e+03 (-5%) | 0.83 | 0.826 |
| C | boneseg learned model, slices of other samples | 0.760 | 6.93 (+37%) | 34.6 (+1%) | 57.8 (-1%) | 1.2 (+39%) | 776 (-30%) | 0.73 | 0.756 |
| C |   + calibrated threshold | 0.765 | 4.55 (-10%) | 39.4 (+16%) | 50.7 (-14%) | 0.897 (+4%) | 1.06e+03 (-3%) | 0.72 | 0.755 |
| D (64 slices) | expert | – | 9.03 | 40.7 | 49.2 | 1.84 | 495 | – | – |
| D | boneseg, 25 + 25 clicks on one slice | 0.758 | 8.97 (-1%) | 35.1 (-14%) | 57 (+16%) | 1.57 (-14%) | 578 (+17%) | 0.77 | 0.754 |
| D | boneseg learned model, 5 slices of the sample | 0.723 | 13.9 (+54%) | 39.7 (-2%) | 50.3 (+2%) | 2.76 (+50%) | 313 (-37%) | -0.67 | 0.724 |
| D |   + calibrated threshold | 0.778 | 10.8 (+19%) | 42.4 (+4%) | 47.1 (-4%) | 2.28 (+24%) | 391 (-21%) | 0.04 | 0.775 |
| D |   + calibrated, smoothed along z | 0.780 | 10.6 (+17%) | 37.6 (-7%) | 53.1 (+8%) | 1.99 (+8%) | 449 (-9%) | 0.07 | 0.778 |
| D |   + calibrated, features of neighbouring slices | 0.782 | 10.7 (+18%) | 36.2 (-11%) | 55.2 (+12%) | 1.94 (+5%) | 462 (-7%) | 0.14 | 0.780 |
| D |   + calibrated, neighbouring slices, smoothed | 0.783 | 10.6 (+18%) | 35.4 (-13%) | 56.5 (+15%) | 1.88 (+2%) | 476 (-4%) | 0.16 | 0.780 |
| D | boneseg learned model, slices of other samples | 0.733 | 10.9 (+20%) | 38.4 (-6%) | 52.1 (+6%) | 2.09 (+14%) | 427 (-14%) | -0.77 | 0.733 |
| D |   + calibrated threshold | 0.712 | 6.9 (-24%) | 46.3 (+14%) | 43.2 (-12%) | 1.6 (-13%) | 582 (+17%) | -0.26 | 0.706 |
| E (390 slices) | expert | – | 17.7 | 22.6 | 88.6 | 2 | 413 | – | – |
| E | boneseg, 25 + 25 clicks on one slice | 0.600 | 12.3 (-31%) | 42.6 (+89%) | 46.9 (-47%) | 2.61 (+31%) | 336 (-19%) | -0.88 | 0.599 |
| E | boneseg learned model, 5 slices of the sample | 0.842 | 19.8 (+12%) | 26.1 (+15%) | 76.8 (-13%) | 2.58 (+29%) | 311 (-25%) | 0.97 | 0.811 |
| E |   + calibrated threshold | 0.848 | 17.7 (-0%) | 26.6 (+18%) | 75.1 (-15%) | 2.35 (+18%) | 350 (-15%) | 0.98 | 0.809 |
| E |   + calibrated, smoothed along z | 0.849 | 17.5 (-1%) | 22.8 (+1%) | 87.5 (-1%) | 2 (+0%) | 413 (+0%) | 0.98 | 0.810 |
| E |   + calibrated, features of neighbouring slices | 0.853 | 17.9 (+1%) | 22.4 (-1%) | 89.3 (+1%) | 2 (+0%) | 411 (-0%) | 0.98 | 0.817 |
| E |   + calibrated, neighbouring slices, smoothed | 0.854 | 17.8 (+1%) | 21.8 (-3%) | 91.7 (+4%) | 1.94 (-3%) | 424 (+3%) | 0.98 | 0.818 |
| E | boneseg learned model, slices of other samples | 0.811 | 21.4 (+21%) | 27.5 (+22%) | 72.6 (-18%) | 2.95 (+48%) | 267 (-35%) | 0.89 | 0.786 |
| E |   + calibrated threshold | 0.789 | 15.9 (-10%) | 33.2 (+47%) | 60.3 (-32%) | 2.63 (+32%) | 320 (-22%) | 0.79 | 0.753 |
| F (233 slices) | expert | – | 7.65 | 27.7 | 72.1 | 1.06 | 871 | – | – |
| F | boneseg, 25 + 25 clicks on one slice | 0.624 | 4.16 (-46%) | 49.7 (+79%) | 40.2 (-44%) | 1.03 (-2%) | 927 (+6%) | 0.43 | 0.594 |
| F | boneseg learned model, 5 slices of the sample | 0.778 | 11 (+43%) | 32.1 (+16%) | 62.3 (-14%) | 1.76 (+66%) | 506 (-42%) | 0.96 | 0.780 |
| F |   + calibrated threshold | 0.827 | 8.12 (+6%) | 36.4 (+31%) | 55 (-24%) | 1.48 (+39%) | 622 (-29%) | 0.96 | 0.823 |
| F |   + calibrated, smoothed along z | 0.830 | 8 (+5%) | 31.5 (+14%) | 63.5 (-12%) | 1.26 (+19%) | 730 (-16%) | 0.97 | 0.826 |
| F |   + calibrated, features of neighbouring slices | 0.837 | 8.13 (+6%) | 30.2 (+9%) | 66.2 (-8%) | 1.23 (+16%) | 748 (-14%) | 0.97 | 0.834 |
| F |   + calibrated, neighbouring slices, smoothed | 0.839 | 8.09 (+6%) | 29.6 (+7%) | 67.6 (-6%) | 1.2 (+13%) | 768 (-12%) | 0.97 | 0.835 |
| F | boneseg learned model, slices of other samples | 0.671 | 14 (+84%) | 28.9 (+4%) | 69.2 (-4%) | 2.03 (+91%) | 424 (-51%) | 0.83 | 0.669 |
| F |   + calibrated threshold | 0.745 | 9.89 (+29%) | 35 (+26%) | 57.1 (-21%) | 1.73 (+63%) | 520 (-40%) | 0.84 | 0.743 |

## Mean over samples

| Method | 3D Dice | BV/TV_% difference | BS/BV_per_mm difference | Tb.Th_um difference | Tb.N_per_mm difference | Tb.Sp_um difference | Slice r |
|---|---|---|---|---|---|---|---|
| boneseg, 25 + 25 clicks on one slice | 0.636 | -18% (|34%|) | +36% (|46%|) | -18% (|30%|) | +0% (|22%|) | +8% (|26%|) | -0.06 |
| boneseg learned model, 5 slices of the sample | 0.767 | +39% (|39%|) | +3% (|13%|) | -1% (|13%|) | +42% (|42%|) | -32% (|32%|) | 0.59 |
|   + calibrated threshold | 0.804 | +6% (|6%|) | +11% (|17%|) | -8% (|15%|) | +18% (|22%|) | -14% (|18%|) | 0.71 |
|   + calibrated, smoothed along z | 0.806 | +5% (|6%|) | -1% (|11%|) | +3% (|12%|) | +4% (|11%|) | -3% (|11%|) | 0.72 |
|   + calibrated, features of neighbouring slices | 0.810 | +5% (|5%|) | -4% (|10%|) | +5% (|11%|) | +1% (|10%|) | -0% (|11%|) | 0.75 |
|   + calibrated, neighbouring slices, smoothed | 0.811 | +5% (|5%|) | -6% (|10%|) | +7% (|12%|) | -1% (|9%|) | +2% (|11%|) | 0.75 |
| boneseg learned model, slices of other samples | 0.722 | +46% (|46%|) | +4% (|7%|) | -3% (|7%|) | +51% (|51%|) | -35% (|35%|) | 0.48 |
|   + calibrated threshold | 0.734 | +2% (|20%|) | +24% (|24%|) | -18% (|18%|) | +26% (|31%|) | -16% (|23%|) | 0.50 |
