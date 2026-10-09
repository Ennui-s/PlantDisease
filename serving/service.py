"""The /predict logic as a plain function (no web framework), so it can be tested directly."""
from __future__ import annotations

import logging
import time

from serving.errors import ApiError
from serving.images import load_image, preprocess
from serving.model_store import ModelStore, build_prediction
from serving.settings import Settings

logger = logging.getLogger("plant_api")


def predict_bytes(store: ModelStore, settings: Settings, data: bytes, request_id: str) -> dict:
    """Validate the upload, preprocess it, run the champion model and return the response body.

    `data` is read with a limit of max_upload_bytes + 1, so a longer value means the upload was too big.
    """
    started = time.perf_counter()
    loaded = store.require()  # a snapshot: the version reported below is the version that predicts

    if len(data) > settings.max_upload_bytes:
        raise ApiError(413, "file_too_large", f"The upload exceeds the {settings.max_upload_bytes} byte limit.")
    image = load_image(data, settings)

    t0 = time.perf_counter()
    batch = preprocess(image, loaded.input_size)
    t1 = time.perf_counter()
    try:
        with store.infer_lock:
            probabilities = loaded.predict(batch)
        result = build_prediction(probabilities[0], loaded.classes, settings.top_k)
    except Exception:
        logger.exception("prediction failed request_id=%s model_version=%s", request_id, loaded.version)
        raise ApiError(500, "inference_failed", "The prediction could not be completed.") from None
    t2 = time.perf_counter()

    return {**result, "model": loaded.ref(), "request_id": request_id,
            "latency_ms": {"preprocess_ms": round((t1 - t0) * 1000, 2),
                           "inference_ms": round((t2 - t1) * 1000, 2),
                           "total_ms": round((t2 - started) * 1000, 2)}}
