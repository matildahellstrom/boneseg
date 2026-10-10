"""FastAPI server for the boneseg web app. Routes live in the modules next to this file."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from .. import backbone as backbone_mod
from .. import sam
from ..backbone import BACKBONE_LABELS, dino_weights_cached, pick_device
from ..segment import SegmentationSettings
from ..store import Store
from . import analysis, batch, datasets, learning, profiles, segmentation, study
from .context import AppContext

STATIC = Path(__file__).resolve().parent.parent / "static"


def _finite(obj):
    """Replaces NaN and infinity, which JSON cannot represent, with null."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


class SafeJSONResponse(JSONResponse):
    """JSON responses where undefined numbers, such as HD95 of an empty mask, become null instead of an error."""

    def render(self, content) -> bytes:
        return json.dumps(_finite(content), ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def create_app(data_dir: str | Path | None = None, allow_paths: bool = True, token: str | None = None) -> FastAPI:
    """allow_paths lets clients open files by path on the server. Keep it off when the app is reachable
    from other computers, since it would let anyone read files the server can read.
    With a token, every API request must carry it, as a 'boneseg_token' cookie (set by opening the app
    once with ?token=...) or as an 'Authorization: Bearer' header."""
    data_dir = Path(data_dir or os.environ.get("BONESEG_DATA_DIR", "projects"))
    store = Store(data_dir)
    app = FastAPI(title="boneseg", version=__version__, default_response_class=SafeJSONResponse)
    app.state.store = store
    ctx = AppContext(store, allow_paths)
    models_dir = store.root / "models"
    models_dir.mkdir(exist_ok=True)
    backbone_mod.MODEL_DIRS = None if allow_paths else [models_dir]

    if token:
        import hmac

        @app.middleware("http")
        async def require_token(request: Request, call_next):
            if request.url.path.startswith("/api/") or request.url.path in ("/docs", "/openapi.json"):
                given = request.cookies.get("boneseg_token") or request.headers.get("authorization", "").removeprefix("Bearer ").strip()
                if not given or not hmac.compare_digest(given, token):
                    return JSONResponse({"detail": "This boneseg server needs an access link with a token. Ask whoever runs it."}, status_code=401)
            return await call_next(request)

    @app.exception_handler(KeyError)
    async def key_error(_: Request, exc: KeyError):
        return JSONResponse({"detail": str(exc).strip("'\"")}, status_code=404)

    @app.exception_handler(ValueError)
    async def value_error(_: Request, exc: ValueError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(IndexError)
    async def index_error(_: Request, exc: IndexError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text()

    @app.get("/simple", response_class=HTMLResponse)
    def simple():
        """Simple mode: open, click, run the stack, download. Same API as the full app."""
        return (STATIC / "simple.html").read_text()

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/api/health")
    def health():
        return {
            "version": __version__,
            "device": str(pick_device()),
            "allow_paths": allow_paths,
            "default_settings": SegmentationSettings().to_dict(),
            "backbones": [{"id": k, "label": v, "ready": dino_weights_cached(k)} for k, v in BACKBONE_LABELS.items()]
                         + [{"id": f"dinov2_s14@{p}", "label": f"DINOv2 Small, fine-tuned · {p.stem}", "ready": dino_weights_cached("dinov2_s14")}
                            for p in sorted(models_dir.glob("*.pt")) if p.stem.startswith("dinov2_s14")],
            "models_dir": str(models_dir),
            "sam": {"available": sam.available(), "weights_cached": sam.weights_cached()},
        }

    for module in (datasets, segmentation, learning, analysis, profiles, study, batch):
        app.include_router(module.router(ctx))
    return app
