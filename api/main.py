"""FastAPI application entry point for TraceChain.

Implements SPEC section 5:
- Loads all artifacts once at startup via lifespan context manager.
- Configures CORS for http://localhost:5173 (and 127.0.0.1:5173).
- Routes:
  - GET /api/health
  - GET /api/wallets/sample
  - GET /api/score/{address}
  - GET /api/trace/{address}
  - GET /api/metrics
- Returns 404 for unknown addresses and 503 when artifacts are missing/unloaded.
"""

from __future__ import annotations

import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Ensure src/ is importable when running `uvicorn api.main:app` from repo root
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from tracechain.service import (
    AddressNotFoundError,
    ArtifactsNotLoadedError,
    TraceChainService,
)


def create_app(
    artifacts_dir: Path | str | None = None,
    reports_dir: Path | str | None = None,
    raw_dir: Path | str | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    resolved_artifacts = Path(
        artifacts_dir
        if artifacts_dir is not None
        else os.environ.get("TRACECHAIN_ARTIFACTS_DIR", _REPO_ROOT / "artifacts")
    )
    resolved_reports = Path(
        reports_dir
        if reports_dir is not None
        else os.environ.get("TRACECHAIN_REPORTS_DIR", _REPO_ROOT / "reports")
    )
    resolved_raw = Path(
        raw_dir
        if raw_dir is not None
        else os.environ.get("TRACECHAIN_RAW_DIR", _REPO_ROOT / "data" / "raw")
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        service = TraceChainService(
            artifacts_dir=resolved_artifacts,
            reports_dir=resolved_reports,
            raw_dir=resolved_raw,
        )
        service.load_all()
        app.state.service = service
        yield

    app = FastAPI(
        title="TraceChain API",
        version="2.0.0",
        description="Bitcoin wallet illicit-risk scoring and nearest VASP-like hub attribution.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _get_service(request: Request) -> TraceChainService:
        service: TraceChainService | None = getattr(
            request.app.state, "service", None
        )
        if service is None:
            service = TraceChainService(
                artifacts_dir=resolved_artifacts,
                reports_dir=resolved_reports,
                raw_dir=resolved_raw,
            )
            service.load_all()
            request.app.state.service = service
        return service

    @app.get("/api/health")
    def health(request: Request) -> dict[str, Any]:
        """Return API health, model loaded status, and graph node count."""
        service = _get_service(request)
        return service.get_health()

    @app.get("/api/wallets/sample")
    def sample_wallets(
        request: Request,
        label: str = Query(default="illicit", pattern="^(illicit|licit)$"),
        n: int = Query(default=5, ge=1, le=50),
    ) -> dict[str, Any]:
        """Return sample test-period wallets with known labels for demos."""
        service = _get_service(request)
        try:
            return service.get_sample_wallets(label=label, n=n)
        except ArtifactsNotLoadedError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/score/{address}")
    def score_wallet(address: str, request: Request) -> dict[str, Any]:
        """Return risk score, explanation, and nearest VASP attribution for a wallet."""
        service = _get_service(request)
        try:
            return service.score_address(address)
        except ArtifactsNotLoadedError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except AddressNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/trace/{address}")
    def trace_wallet(address: str, request: Request) -> dict[str, Any]:
        """Return fund-flow shortest path to nearest VASP seed and neighbor subgraph."""
        service = _get_service(request)
        try:
            return service.trace_address(address)
        except ArtifactsNotLoadedError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except AddressNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/metrics")
    def metrics(request: Request) -> dict[str, Any]:
        """Return contents of reports/metrics.json plus attribution_stats.json."""
        service = _get_service(request)
        try:
            return service.get_metrics()
        except ArtifactsNotLoadedError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


app = create_app()
