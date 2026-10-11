"""FastAPI inference service for the plant disease classifier.

    uvicorn serving.app:app --port 8000

Endpoints
    GET  /health   liveness: the process is up
    GET  /ready    readiness: 200 only when the champion model is loaded
    GET  /metrics  request, prediction, latency and confidence metrics (Prometheus text format)
    GET  /model    which model version is being served, with its test metrics and input contract
    POST /predict  multipart upload (field "file") -> prediction, confidence, top-k, model version, latency
    POST /reload   re-resolve the champion alias and load it (only enabled when ADMIN_TOKEN is set)

Every error is JSON: {"error": {"code", "message", "request_id"}}. Internal details are logged, never returned.
"""
import hmac
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Header, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from serving.errors import ApiError
from serving.model_store import ModelStore
from serving.monitoring import KNOWN_ENDPOINTS, Metrics, log_request, setup_logging
from serving.service import predict_bytes
from serving.settings import Settings

logger = logging.getLogger("plant_api")
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_UPLOAD_OVERHEAD = 64 * 1024  # multipart framing around the file


class ModelRef(BaseModel):
    name: str
    alias: str
    version: str


class TopK(BaseModel):
    label: str
    probability: float


class Latency(BaseModel):
    preprocess_ms: float
    inference_ms: float
    total_ms: float


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    top_k: list[TopK]
    model: ModelRef
    latency_ms: Latency
    request_id: str


def _error(request: Request, status: int, code: str, message: str) -> JSONResponse:
    request.state.error_code = code   # read by the request middleware for logs and metrics
    request_id = getattr(request.state, "request_id", None)
    headers = {"X-Request-ID": request_id} if request_id else None
    return JSONResponse(status_code=status, headers=headers,
                        content={"error": {"code": code, "message": message, "request_id": request_id}})


def create_app(store: ModelStore | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or (store.settings if store else Settings.from_env())
    store = store or ModelStore(settings)
    metrics = Metrics()
    setup_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not store.load():  # the service still starts; /ready reports 503 and /predict refuses
            logger.error("champion model could not be loaded: %s", store.last_error)
        yield

    app = FastAPI(title="Plant Disease Classifier", version="1.0", lifespan=lifespan,
                  description="Serves the champion model from the MLflow model registry.")
    app.state.store = store
    app.state.metrics = metrics

    def record(request: Request, status: int, started: float) -> None:
        """Count the request and write its structured log line."""
        endpoint = request.url.path if request.url.path in KNOWN_ENDPOINTS else "unmatched"
        error_code = getattr(request.state, "error_code", None) or ("internal_error" if status >= 500 else None)
        prediction = getattr(request.state, "prediction", None)
        metrics.observe_request(endpoint, status, error_code)
        if prediction:
            metrics.observe_prediction(prediction["label"], prediction["confidence"],
                                       prediction["total_s"], prediction["inference_s"])
        log_request(request_id=request.state.request_id, method=request.method, endpoint=endpoint, status=status,
                    latency_ms=round((time.perf_counter() - started) * 1000, 2),
                    model_version=prediction["version"] if prediction else None,
                    prediction=prediction["label"] if prediction else None,
                    confidence=prediction["confidence"] if prediction else None,
                    inference_ms=round(prediction["inference_s"] * 1000, 2) if prediction else None,
                    error_category=error_code if status >= 400 else None)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        started = time.perf_counter()
        supplied = request.headers.get("x-request-id", "")
        request.state.request_id = supplied if _REQUEST_ID.match(supplied) else uuid.uuid4().hex
        length = request.headers.get("content-length", "")
        try:
            if request.method == "POST" and length.isdigit() \
                    and int(length) > settings.max_upload_bytes + _UPLOAD_OVERHEAD:
                response = _error(request, 413, "file_too_large",
                                  f"The upload exceeds the {settings.max_upload_bytes} byte limit.")
            else:
                response = await call_next(request)
        except Exception:
            record(request, 500, started)   # an unhandled error still counts; the error handler answers the client
            raise
        response.headers["X-Request-ID"] = request.state.request_id
        record(request, response.status_code, started)
        return response

    @app.exception_handler(ApiError)
    async def handle_api_error(request: Request, exc: ApiError):
        return _error(request, exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        return _error(request, 422, "malformed_request",
                      "Send multipart/form-data with the image in a field named 'file'.")

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(request: Request, exc: StarletteHTTPException):
        codes = {404: "not_found", 405: "method_not_allowed"}
        return _error(request, exc.status_code, codes.get(exc.status_code, "http_error"), str(exc.detail))

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception):
        logger.exception("unhandled error request_id=%s", getattr(request.state, "request_id", None))
        return _error(request, 500, "internal_error", "Internal server error.")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        if not store.ready:
            return JSONResponse(status_code=503, content={"status": "unavailable", "reason": "model not loaded"})
        return {"status": "ready", "model": store.require().ref()}

    @app.get("/model")
    def model_info():
        return store.require().info()

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: Request, file: Annotated[UploadFile, File()]):
        data = file.file.read(settings.max_upload_bytes + 1)
        result = predict_bytes(store, settings, data, request.state.request_id)
        request.state.prediction = {   # picked up by the request middleware for logs and metrics
            "label": result["prediction"], "confidence": result["confidence"], "version": result["model"]["version"],
            "inference_s": result["latency_ms"]["inference_ms"] / 1000, "total_s": result["latency_ms"]["total_ms"] / 1000}
        return result

    @app.get("/metrics", include_in_schema=False)
    def metrics_endpoint():
        model = store.require().ref() if store.ready else None
        return Response(content=metrics.render(model), media_type="text/plain; version=0.0.4; charset=utf-8")

    if settings.admin_token:
        @app.post("/reload")
        def reload(x_admin_token: Annotated[str | None, Header()] = None):
            if not hmac.compare_digest((x_admin_token or "").encode(), settings.admin_token.encode()):
                raise ApiError(401, "unauthorized", "A valid X-Admin-Token header is required.")
            reloaded = store.load()
            if not store.ready:
                raise ApiError(503, "model_unavailable", "The model is not available right now.")
            return {"reloaded": reloaded, "model": store.require().info()}

    return app


app = create_app()
