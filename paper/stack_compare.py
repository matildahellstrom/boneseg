"""boneseg's masks for every slice of the stacks that the whole-stack SAM notebook segments
(kaggle_finetune_sam/kernel_stack), from the same 5 labelled slices per sample, for a side-by-side comparison:
  learned      the learned model on frozen DINOv2 Small (calibrated threshold, features of the slices 4 um below
               and above), as the app trains it on a stack
  ft_learned   DINOv2 Small fine-tuned on the 5 slices (the other 5 development slices stop it early), and the
               learned model on its features
Slices: the central 80% of each stack, and slices 224-287 of sample D (read remotely), as in stack3d_evaluate.py.
Masks are written bit-packed to data/stack_compare/<sample>_<method>.npz (git-ignored: derived from the Liu images),
in the same format as the SAM notebook's stack_masks/<sample>.npz. stack_compare_view.py scores and shows them.
Usage: python paper/stack_compare.py [--samples A C D E F]
"""
from __future__ import annotations

import argparse
import copy
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from common import ROOT, available_samples, load_sample
from boneseg import io as bio
from boneseg.backbone import get_backbone, pick_device
from boneseg.finetune import finetune, load_dino
from boneseg.head import Head, head_input, segment_with_head, train_head, z_context_slices
from boneseg.segment import SegmentationSettings, embed_image

OUT = ROOT / "data" / "stack_compare"
D_BLOCK = (224, 287)


def stack_range(name, vol):
    """The slices segmented for a sample: the central 80% of the stack, or D's block."""
    if name == "D":
        return list(range(D_BLOCK[0], D_BLOCK[1] + 1))
    return list(range(int(0.1 * (vol.n_z - 1)), int(0.9 * (vol.n_z - 1)) + 1))


def save_masks(path, zs, packed, shape):
    """Masks bit-packed along x (np.packbits), as the SAM notebook writes them; unpack with np.unpackbits(..., axis=-1)."""
    np.savez_compressed(path, z=np.array(zs), shape=np.array(shape), masks=np.stack(packed))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="*", default=None)
    args = ap.parse_args(argv)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
    OUT.mkdir(parents=True, exist_ok=True)
    st = SegmentationSettings()
    device = pick_device()
    dino = load_dino("dinov2_s14")
    for name in args.samples or available_samples():
        s = load_sample(name)
        vol = s.vol
        if vol is None:   # Sample D, read from Kaggle over range requests
            from remote_d import remote_volume
            vol = remote_volume()
        zs = stack_range(name, vol)
        plane = lambda z: bio.normalize_plane(vol.get_plane(s.channel, int(np.clip(z, 0, vol.n_z - 1))))  # noqa: E731
        d = z_context_slices(vol.voxel_um[0], vol.n_z)

        # The fine-tuned backbone: the same 5 labelled slices, the other 5 development slices for early stopping
        ft = finetune([(sl.img, sl.gt) for sl in s.dev[:5]], [(sl.img, sl.gt) for sl in s.dev[5:]], train_blocks=4, steps=1000,
                      pretrained=copy.deepcopy(dino), device=device)
        ft_path = OUT / f"dinov2_s14_{name}.pt"
        ft.save(ft_path)
        backbones = {"learned": get_backbone("dinov2_s14"), "ft_learned": get_backbone(f"dinov2_s14@{ft_path}")}
        log(f"{name}: {len(zs)} slices ({zs[0]}-{zs[-1]}), fine-tuning best validation Dice {ft.info['best_val_dice']:.3f}")

        for method, bb in backbones.items():
            # Embeddings of recent slices are kept, so each slice is computed about once although it is also a neighbour
            cache: OrderedDict = OrderedDict()

            def embed_at(z, bb=bb, cache=cache):
                if z not in cache:
                    cache[z] = embed_image(bb, plane(z), st)
                    if len(cache) > 2 * d + 3:
                        cache.popitem(last=False)
                return cache[z]
            probe = Head(st.backbone, st.layer_from_end, st.vit_size, "linear", {}, 0, z_context=d)
            train = [(head_input(probe, lambda zz: embed_image(bb, plane(zz), st), sl.z, vol.n_z), sl.gt) for sl in s.dev[:5]]
            head = train_head(train, st, [(0, sl.z) for sl in s.dev[:5]], pixel_um=vol.pixel_um)
            head.z_context = d
            masks = []
            for i, z in enumerate(zs):
                m = segment_with_head(head, head_input(head, embed_at, z, vol.n_z), st, vol.pixel_um).mask
                masks.append(np.packbits(m, axis=-1))   # Packed at once: full masks of a 500-slice stack would not fit in memory
                if (i + 1) % 100 == 0:
                    log(f"  {name} {method}: {i + 1} of {len(zs)}")
            save_masks(OUT / f"{name}_{method}.npz", zs, masks, (vol.height, vol.width))
            log(f"{name} {method}: done (threshold {head.threshold:.2f}, z context {d} slices)")
    log("done")


if __name__ == "__main__":
    main()
