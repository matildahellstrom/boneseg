"""Points every Kaggle notebook in the repository at another Kaggle account.

Kaggle notebooks belong to an account: their kernel-metadata.json names it in "id", and private datasets they read
are named "<owner>/<dataset>". This rewrites both for every kernel-metadata.json, from the old owner to the new one,
and regenerates the SAM / micro-SAM notebooks with KAGGLE_USERNAME set. The datasets themselves (Liu files, nnU-Net
export) must then exist under the new account; see "Data" in paper/README.md.
Usage: python scripts/set_kaggle_user.py NEW_USER [--old matildahellstrom]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("user")
    ap.add_argument("--old", default="matildahellstrom")
    args = ap.parse_args(argv)
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,48}", args.user):
        sys.exit("Kaggle user names are lower-case letters, digits and hyphens")
    swap = lambda s: f"{args.user}/{s.split('/', 1)[1]}" if s.startswith(f"{args.old}/") else s  # noqa: E731
    for path in sorted(ROOT.rglob("kernel-metadata.json")):
        if ".venv" in path.parts:
            continue
        meta = json.loads(path.read_text())
        new = {**meta, "id": swap(meta["id"]), "dataset_sources": [swap(d) for d in meta.get("dataset_sources", [])]}
        if new != meta:
            path.write_text(json.dumps(new, indent=2))
            print("updated", path.relative_to(ROOT))
    env = {**os.environ, "KAGGLE_USERNAME": args.user}
    subprocess.run([sys.executable, str(ROOT / "paper" / "kaggle_finetune_sam" / "make_kernels.py")], env=env, check=True,
                   stdout=subprocess.DEVNULL)
    print("regenerated paper/kaggle_finetune_sam/kernel_* for", args.user)
    print(f"Also set BONESEG_LIU_DATASET=<owner>/<dataset> for paper/remote_d.py if your Liu dataset is not {args.user}/liudata.")


if __name__ == "__main__":
    main()
