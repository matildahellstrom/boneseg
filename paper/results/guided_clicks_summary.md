# Click suggestions against extra random clicks

Test slices of the Liu samples, starting from 3 + 6 careful clicks; mean Dice after each extra click.

| Strategy | 0 extra | 1 extra | 2 extra | 3 extra | Fallback clicks |
|---|---|---|---|---|---|
| missed | 0.664 | 0.654 | 0.649 | 0.652 | 8% |
| uncertainty | 0.664 | 0.685 | 0.694 | 0.691 | 0% |
| random | 0.664 | 0.669 | 0.680 | 0.686 | 0% |

| After 3 extra clicks | Difference in Dice (95% CI) | Slices better |
|---|---|---|
| missed vs random | -0.034 (-0.057 to -0.012) | 32% |
| uncertainty vs random | +0.005 (-0.019 to +0.028) | 54% |
| missed vs uncertainty | -0.039 (-0.065 to -0.012) | 32% |
