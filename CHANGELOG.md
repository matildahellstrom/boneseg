# Changelog

How boneseg grew from the research notebook, grouped by theme. Hashes point to the commits, so `git show <hash>` gives the details and the reasoning in each message. Numbers are Dice scores unless stated otherwise.

## 3 October 2026

### The notebook
- `317de70` The original notebook, kept for comparison. `8dd4b1b` The reviewed version: separate tuning and evaluation slices, a held-out test sample for the U-Nets, confidence intervals across samples, HD95 in micrometres, pinned versions, and Kaggle batch-run metadata.

### The app's foundation
- `e0bcaa9` The `boneseg` package: lazy loading of Imaris, TIFF, PNG and NumPy files; DINOv2 backbones plus a fast "classic" backbone; prototype segmentation, uncertainty and profiles; metrics and measurements in micrometres.
- `c0b9faa` The web API, stack runs and storage. `91ddb63` The browser interface and a synthetic demo stack.
- `171c322` README and a real-data benchmark. `7b2101a` A GitHub Actions workflow, ready to enable once the GitHub login has the `workflow` permission (see `ci/README.md`).

### Better segmentation, measured
- `613d9f1` The threshold comes from the clicks (stack Dice on the demo 0.89, against 0.36 with Otsu), and scores are standardized per slice (0.81 against 0.69 on a strongly degraded stack). The adaptive stack mode was removed after it lowered Dice everywhere.
- `2041711`, `a8e6223` The learned model gets neighbourhood features, chosen by cross-validation (unseen-slice Dice on Liu file A 0.69 against 0.67 with four labels).
- Tested and left out, with numbers in the README: colour input, per-dataset tuning, shifted passes, a stricter threshold, flip averaging, two-layer features, neighbourhood features for clicks (`c0eb945`, `c10ecd2`, `0a0f291`).

### Learning from corrections
- `dde6ad5` Brush correction, labels and a small learned model with cross-validated Dice. `14f377e` Learned models saved as profiles.
- `f072941` Training also reports Dice against the expert reference on unlabelled slices, since cross-validation only measures agreement with the labels.
- `5e0e81e` "Label 5 slices from the reference" turns expert masks into labels (0.65 to 0.74 on unlabelled slices of Liu file A).

### Several structures
- `2724057` Several structures on one slice, each with its own clicks. `9d57a1d` Whole stacks with several structures. `bc7d3e6` Multi-structure profiles.
- `ffdf18f`, `287510b` Labels and a learned model for several structures.

### Bone analysis
- `7a1b05f` 2D histomorphometry following the ASBMR nomenclature. `d23cda3` Regions of interest. `d801963` Structures as masks for histomorphometry.
- `956f89f` Stack-level histomorphometry in multi-structure runs. `a89a270` Whole-stack histomorphometry with bone and cells on different channels.
- `9da51e9` Objects counted in 3D. `757d363` 3D label images. `e6f3b7c` Each object's intensity in every channel. `cdaae44` Nearest-neighbour distances.

### Scaling up and comparing
- `f30a53e` Command-line batch processing. `e2eb4b3` Batch runs from the app.
- `077a0b1` "Compare samples" with group statistics and an honest small-sample warning.
- `dfa72b4` HTML reports with a draft methods paragraph. `8dbc3a5` Project zips. `9746530` Sharing profiles.

### Speed
- `ea95657` Features for neighbouring slices prefetched (a neighbouring slice in 0.53 s instead of 2.3 s). `d47e2b5` Clicks on 3000 px images five times faster. `f354b29`, `09daa73` Stack runs faster on large images and through read-ahead (0.65 to 0.44 s per slice on Liu file A).

### Safety and robustness
- `610f4af` Opening files by path is off on the network. `f0fe261` An optional access token.
- `e77fc2b` Names from files and shared profiles are escaped before display. `e614f80` Download names are sanitized.
- Bugs found by tests and fixed: concurrent GPU use crashing the server (`91ddb63`), a deadlock in prefetching (`ea95657`), NaN breaking JSON responses (`077a0b1`), regions discarding cached results (`f8ba922`), unsaved clicks lost on closing the page (`92734e8`).
- `2bcc1a2` Every common file type tested end to end. `1c7c678` The Imaris path tested on synthetic `.ims` files.

### Interface
- `3e3f22f` A side view through the stack. `c16cbd6` An error overlay against the reference. `a9ffff2` A second channel in colour.
- `b502135` Next-step hints and a light theme. `a77a357` Collapsible panel sections. `ff48c55` Remembered settings and copying clicks to the next slice. `b1ca41c` Notes per dataset.

### Back into the notebook
- `83fbebd`, `d283310` The click-calibrated threshold and per-slice standardization. `d71ea5d` A linear probe on DINO features as a baseline next to the U-Nets.

## 4 October 2026

- `8aae1fa` Reports cover multi-structure results.
- `3754f0f` The notebook's Kaggle run of 3 October ran out of GPU memory at the first U-Net and lost its tables. Fixed: Giant is freed first, U-Nets train in mixed precision, and every experiment saves its tables as soon as it finishes.
- `10dc675` The Docker image was built and tested: token, disabled path opening, persistent `/data` volume, both backbones on the CPU.
- `c512ed7` A cross-file benchmark on Liu files A, E and F. Five expert slices of a file beat clicking every slice (0.72, 0.83, 0.78 against 0.63, 0.77, 0.62); a model from another file roughly matches clicking without clicks.
- `63ef068` Measured and left out for carrying models between files: per-image feature centering, brightness and blur copies, DINOv2 Base and Large.
- `2ae92b0`, `b8d701c` Each dataset remembers its channel, and batch runs (in the app and on the command line) can use each file's own channel, because the autofluorescence is channel 3 in A but 4 in E and F. On the real files, a model trained on five expert slices of A scored 0.67 on A, 0.75 on E and 0.57 on F over whole stacks.


## 5 October 2026

### Sharper boundaries, measured on four samples
- `7749f47` The research notebook tunes every DINOv2 backbone separately instead of reusing Giant's settings.
- Sample C was added to the paper evaluation, which now covers A, C, E and F.
- `4569641` Four boundary options: a stricter click threshold, guided-filter edge snapping, 2 x 2 shifted feature passes, and a small learned refiner. `f057733` `python -m boneseg train-refiner` trains a refiner on your own labelled files.
- The nested evaluation on four samples made the first three the defaults: Dice with 25 clicks 0.68 to 0.76, bone-area bias +6.6 to +2.0 points. The refiner helps only with few or noisy clicks and stays optional. Learned models measure their neighbourhood in patches, so models trained before keep working on the finer grid.
