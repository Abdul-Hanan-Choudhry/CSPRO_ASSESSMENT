"""FastAPI application factory: middleware, error mapping, lifespan wiring."""

import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__
from app.api.routes import router
from app.config import Settings, get_settings
from app.container import Services, build_services
from app.errors import AppError
from app.observability import configure_logging, request_id_var

log = logging.getLogger("app")
STATIC_DIR = Path(__file__).resolve().parent / "static"
_CLIENT_REQUEST_ID = re.compile(r"^[A-Za-z0-9-]{1,64}$")


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(settings.log_level)
        app.state.services = services or build_services(settings)
        yield
        if services is None:  # only close what this app created
            app.state.services.store.close()

    app = FastAPI(title="Support RAG", version=__version__, lifespan=lifespan)
    app.middleware("http")(_request_context)
    _register_error_handlers(app)
    app.include_router(router)
    # Test console: static HTML that calls this same API, so it needs no CORS or extra server.
    app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
    return app


async def _request_context(request: Request, call_next):
    supplied = request.headers.get("X-Request-ID", "")
    request_id = supplied if _CLIENT_REQUEST_ID.match(supplied) else uuid.uuid4().hex
    request.state.request_id = request_id
    token = request_id_var.set(request_id)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = request_id
    log.info(
        "request",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "ms": round((time.perf_counter() - started) * 1000),
        },
    )
    return response


def _error(request: Request, status: int, code: str, message: str) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "-")
    body = {"error": {"code": code, "message": message, "request_id": request_id}}
    return JSONResponse(status_code=status, content=body, headers={"X-Request-ID": request_id})


def _register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error(request: Request, exc: AppError) -> JSONResponse:
        log.warning("request failed", extra={"code": exc.code, "detail": exc.message})
        return _error(request, exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Field locations and messages only: never echo the submitted input back.
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:5]
        )
        return _error(request, 422, "validation_error", problems)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error(request, exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        return _error(request, 500, "internal_error", "internal server error")


app = create_app()
