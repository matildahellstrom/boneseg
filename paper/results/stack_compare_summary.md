# Whole stacks: boneseg against fine-tuned SAM, from the same 5 labelled slices

Every slice of the central 80% of each stack (sample D: slices 224-287), against the experts' 3D masks; measures on volumes reduced to 4 x 4 pixels in-plane and 2 slices in z, as in stack3d_summary.md. Errors are relative to the expert.

| Sample | Method | 3D Dice | Mean slice Dice | BV/TV | BS/BV | Tb.Th | Tb.N | Tb.Sp |
|---|---|---|---|---|---|---|---|---|
| A (202 slices) | expert | – | – | 13.7% | 35.9 | 55.7 µm | 2.45 | 352 µm |
| A | boneseg learned model | 0.751 | 0.752 | -0% | -22% | +28% | -22% | +28% |
| A | boneseg, fine-tuned DINOv2 + learned model | 0.751 | 0.753 | +3% | -23% | +29% | -20% | +25% |
| A | SAM ViT-B fine-tuned | 0.742 | 0.746 | +2% | +4% | -4% | +6% | -6% |
| C (505 slices) | expert | – | – | 5.06% | 34.1 | 58.6 µm | 0.863 | 1.1e+03 µm |
| C | boneseg learned model | 0.826 | 0.826 | +1% | +6% | -6% | +7% | -7% |
| C | boneseg, fine-tuned DINOv2 + learned model | 0.847 | 0.848 | -6% | +1% | -1% | -5% | +6% |
| C | SAM ViT-B fine-tuned | 0.851 | 0.851 | -4% | +7% | -6% | +3% | -2% |
| D (64 slices) | expert | – | – | 9.03% | 40.7 | 49.2 µm | 1.84 | 495 µm |
| D | boneseg learned model | 0.782 | 0.780 | +18% | -11% | +12% | +6% | -7% |
| D | boneseg, fine-tuned DINOv2 + learned model | 0.796 | 0.793 | +3% | -12% | +14% | -9% | +10% |
| D | SAM ViT-B fine-tuned | 0.853 | 0.851 | +6% | -17% | +20% | -11% | +12% |
| E (390 slices) | expert | – | – | 17.7% | 22.6 | 88.6 µm | 2 | 413 µm |
| E | boneseg learned model | 0.853 | 0.817 | +1% | -1% | +1% | +0% | -0% |
| E | boneseg, fine-tuned DINOv2 + learned model | 0.875 | 0.846 | -0% | +3% | -3% | +2% | -2% |
| E | SAM ViT-B fine-tuned | 0.882 | 0.867 | -0% | +19% | -16% | +18% | -15% |
| F (233 slices) | expert | – | – | 7.65% | 27.7 | 72.1 µm | 1.06 | 871 µm |
| F | boneseg learned model | 0.837 | 0.834 | +6% | +9% | -8% | +16% | -14% |
| F | boneseg, fine-tuned DINOv2 + learned model | 0.871 | 0.864 | +1% | +4% | -4% | +5% | -5% |
| F | SAM ViT-B fine-tuned | 0.889 | 0.886 | -2% | +13% | -11% | +10% | -9% |

## Mean over samples

| Method | Samples | 3D Dice | Mean slice Dice | BV/TV | BS/BV | Tb.Th | Tb.N | Tb.Sp |
|---|---|---|---|---|---|---|---|---|
| boneseg learned model | 5 | 0.810 | 0.802 | +5% (|5%|) | -4% (|10%|) | +5% (|11%|) | +1% (|10%|) | -0% (|11%|) |
| boneseg, fine-tuned DINOv2 + learned model | 5 | 0.828 | 0.821 | +0% (|3%|) | -5% (|8%|) | +7% (|10%|) | -5% (|9%|) | +7% (|10%|) |
| SAM ViT-B fine-tuned | 5 | 0.843 | 0.840 | +0% (|3%|) | +5% (|12%|) | -4% (|11%|) | +5% (|10%|) | -4% (|9%|) |
