"""FastAPI inference service for the plant disease classifier.

    uvicorn serving.app:app --port 8000

Endpoints
    GET  /health   liveness: the process is up
    GET  /ready    readiness: 200 only when the champion model is loaded
    GET  /model    which model version is being served, with its test metrics and input contract
    POST /predict  multipart upload (field "file") -> prediction, confidence, top-k, model version, latency
    POST /reload   re-resolve the champion alias and load it (only enabled when ADMIN_TOKEN is set)

Every error is JSON: {"error": {"code", "message", "request_id"}}. Internal details are logged, never returned.
"""
import hmac
import logging
import re
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, Header, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from serving.errors import ApiError
from serving.model_store import ModelStore
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
    top_k: List[TopK]
    model: ModelRef
    latency_ms: Latency
    request_id: str


def _error(request: Request, status: int, code: str, message: str) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    headers = {"X-Request-ID": request_id} if request_id else None
    return JSONResponse(status_code=status, headers=headers,
                        content={"error": {"code": code, "message": message, "request_id": request_id}})


def create_app(store: Optional[ModelStore] = None, settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or (store.settings if store else Settings.from_env())
    store = store or ModelStore(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not store.load():  # the service still starts; /ready reports 503 and /predict refuses
            logger.error("champion model could not be loaded: %s", store.last_error)
        yield

    app = FastAPI(title="Plant Disease Classifier", version="1.0", lifespan=lifespan,
                  description="Serves the champion model from the MLflow model registry.")
    app.state.store = store

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        supplied = request.headers.get("x-request-id", "")
        request.state.request_id = supplied if _REQUEST_ID.match(supplied) else uuid.uuid4().hex
        length = request.headers.get("content-length", "")
        if request.method == "POST" and length.isdigit() \
                and int(length) > settings.max_upload_bytes + _UPLOAD_OVERHEAD:
            response = _error(request, 413, "file_too_large",
                              f"The upload exceeds the {settings.max_upload_bytes} byte limit.")
        else:
            response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
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
    def predict(request: Request, file: UploadFile = File(...)):
        data = file.file.read(settings.max_upload_bytes + 1)
        return predict_bytes(store, settings, data, request.state.request_id)

    if settings.admin_token:
        @app.post("/reload")
        def reload(x_admin_token: Optional[str] = Header(default=None)):
            if not hmac.compare_digest((x_admin_token or "").encode(), settings.admin_token.encode()):
                raise ApiError(401, "unauthorized", "A valid X-Admin-Token header is required.")
            reloaded = store.load()
            if not store.ready:
                raise ApiError(503, "model_unavailable", "The model is not available right now.")
            return {"reloaded": reloaded, "model": store.require().info()}

    return app


app = create_app()
