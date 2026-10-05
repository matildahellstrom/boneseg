"""Command-line entry point: python -m boneseg serve"""
from __future__ import annotations

import argparse
import os
import webbrowser


def main(argv=None):
    parser = argparse.ArgumentParser(prog="boneseg", description="Few-shot segmentation of bone microscopy images")
    sub = parser.add_subparsers(dest="cmd", required=True)
    serve = sub.add_parser("serve", help="Start the web app")
    serve.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 to allow other computers on the network")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--data-dir", default=os.environ.get("BONESEG_DATA_DIR", "projects"))
    serve.add_argument("--open", action="store_true", help="Open the app in the browser")
    serve.add_argument("--token", default=os.environ.get("BONESEG_TOKEN"),
                       help="Require this access token for the API; share http://HOST:PORT/?token=... with users")
    serve.add_argument("--allow-paths", action="store_true",
                       help="Allow opening files by path even when listening on the network (on by default for 127.0.0.1)")
    batch = sub.add_parser("batch", help="Segment many files with a saved profile")
    batch.add_argument("files", nargs="+", help="Microscopy files; 'file.ims:3' gives a file its own channel")
    batch.add_argument("--profile", required=True, help="Profile id from the app, or a path to its .npz file")
    batch.add_argument("--channel", type=int, default=None, help="Channel for files without their own ':channel'")
    batch.add_argument("--out", default="results", help="Output folder")
    batch.add_argument("--z-start", type=int, default=0)
    batch.add_argument("--z-end", type=int, default=None)
    batch.add_argument("--z-step", type=int, default=1)
    batch.add_argument("--reference", type=int, default=None, help="Channel with an expert mask, to report Dice")
    batch.add_argument("--data-dir", default=os.environ.get("BONESEG_DATA_DIR", "projects"))
    profiles = sub.add_parser("profiles", help="List saved profiles")
    profiles.add_argument("--data-dir", default=os.environ.get("BONESEG_DATA_DIR", "projects"))
    tr = sub.add_parser("train-refiner", help="Train a learned refiner from files with an expert mask channel")
    tr.add_argument("files", nargs="+", help="'file:image_channel' or 'file:image_channel:mask_channel' (mask defaults to the last channel)")
    tr.add_argument("--out", default="refiner.pt")
    tr.add_argument("--slices", type=int, default=10, help="Labelled slices per file")
    tr.add_argument("--steps", type=int, default=1500)
    tr.add_argument("--settings", default="{}", help='Segmentation settings as JSON, e.g. \'{"shift_passes": 2}\'')
    fine = sub.add_parser("finetune", help="Fine-tune DINOv2 Small on files with an expert mask channel")
    fine.add_argument("files", nargs="+", help="'file:image_channel' or 'file:image_channel:mask_channel' (mask defaults to the last channel)")
    fine.add_argument("--out", default="dinov2_s14_finetuned.pt")
    fine.add_argument("--slices", type=int, default=10, help="Labelled slices per file; every second one validates")
    fine.add_argument("--blocks", type=int, default=4, help="Transformer blocks to train, counted from the end (0 trains the output layer only)")
    fine.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args(argv)

    if args.cmd == "finetune":
        from .finetune import finetune
        from .io import labelled_slices

        train, val = [], []
        for spec in args.files:
            sl = labelled_slices(spec, args.slices)
            print(f"{spec}: {len(sl)} labelled slices")
            train += [(img, gt) for i, (_, img, gt) in enumerate(sl) if i % 2 == 0]
            val += [(img, gt) for i, (_, img, gt) in enumerate(sl) if i % 2 == 1]
        if not train:
            raise SystemExit("No slices with a mask were found")
        ft = finetune(train, val, train_blocks=args.blocks, steps=args.steps, log=print)
        ft.save(args.out)
        print(f"Wrote {args.out} (best validation Dice {ft.info['best_val_dice']:.3f} at step {ft.info['best_step']}). "
              f"Use it as the backbone 'dinov2_s14@{args.out}', or place it in the app's models folder.")
        return

    if args.cmd == "train-refiner":
        import json

        from .refine import train_refiner_from_files
        from .segment import SegmentationSettings

        st = SegmentationSettings.from_dict({**json.loads(args.settings), "refiner": ""})
        ref = train_refiner_from_files(args.files, st, n_slices=args.slices, steps=args.steps)
        ref.save(args.out)
        print(f"Wrote {args.out}. Use it with the 'refiner' setting set to this path, with the same other settings.")
        return

    if args.cmd == "batch":
        from .batch import resolve_profile, run_batch

        prof = resolve_profile(args.profile, args.data_dir)
        print(f"Profile '{prof.name}' ({prof.backbone}), {len(args.files)} file(s)")
        df = run_batch(args.files, prof, args.channel, args.out, args.z_start, args.z_end, args.z_step, args.reference)
        print(f"Wrote {args.out}/summary.csv ({(df['status'] == 'ok').sum()} of {len(df)} files ok)")
        return
    if args.cmd == "profiles":
        from .store import Store

        for p in Store(args.data_dir).list_profiles():
            print(f"{p['id']:40s} {p['name']} · {p['backbone']} · {p['source']}")
        return

    if args.cmd == "serve":
        import uvicorn

        from .api import create_app

        local = args.host in ("127.0.0.1", "localhost", "::1")
        app = create_app(args.data_dir, allow_paths=local or args.allow_paths, token=args.token)
        if args.token:
            print(f"Access link: http://{args.host if local else '<this-computer>'}:{args.port}/?token={args.token}", flush=True)
        elif not local:
            print("Listening on the network without a token: anyone who can reach this port can use the app. Consider --token.", flush=True)
        if not local and not args.allow_paths:
            print("Listening on the network: opening files by path is off. Users can still upload files.", flush=True)
        if args.open:
            webbrowser.open(f"http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
