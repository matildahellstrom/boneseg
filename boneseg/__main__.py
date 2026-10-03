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
    args = parser.parse_args(argv)

    if args.cmd == "serve":
        import uvicorn

        from .api import create_app

        app = create_app(args.data_dir)
        if args.open:
            webbrowser.open(f"http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
