"""Request logging and metrics for the inference service. No extra dependencies.

Logging: one JSON line per request, built only from an allow-list of fields (LOG_FIELDS), so uploaded
images, file names and headers can never end up in the logs.

Metrics: counters and histograms rendered in the Prometheus text exposition format (GET /metrics).

Definitions used by the metrics
    rejection : a /predict request refused because of the client's input (HTTP 4xx)
    error     : a failure on our side (HTTP 5xx), by category
"""
from __future__ import annotations

import json
import logging
import sys
import threading
from bisect import bisect_left
from datetime import UTC, datetime

ACCESS_LOGGER = "plant_api.access"
LOG_FIELDS = ("timestamp", "event", "request_id", "method", "endpoint", "status", "latency_ms",
              "model_version", "prediction", "confidence", "inference_ms", "error_category")
KNOWN_ENDPOINTS = ("/health", "/ready", "/model", "/predict", "/reload", "/metrics")
QUIET_ENDPOINTS = ("/health", "/ready", "/metrics")   # successful calls (probes, scrapes) are logged at DEBUG

CONFIDENCE_BUCKETS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)
LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.5, 1.0, 2.5, 5.0)   # seconds


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(ACCESS_LOGGER)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


def log_request(**fields) -> None:
    """Write one structured log line. Fields outside LOG_FIELDS are dropped."""
    record = {"timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"), "event": "request"}
    record.update({k: v for k, v in fields.items() if k in LOG_FIELDS and v is not None})
    status = int(fields.get("status") or 0)
    quiet = fields.get("endpoint") in QUIET_ENDPOINTS and status < 400
    level = logging.DEBUG if quiet else logging.ERROR if status >= 500 else logging.WARNING if status >= 400 else logging.INFO
    logging.getLogger(ACCESS_LOGGER).log(level, json.dumps(record, separators=(",", ":")))


class Histogram:
    def __init__(self, buckets):
        self.buckets = tuple(sorted(buckets))
        self.counts = [0] * (len(self.buckets) + 1)   # per bucket, the last one is +Inf
        self.sum = 0.0
        self.count = 0

    def observe(self, value: float) -> None:
        self.counts[bisect_left(self.buckets, value)] += 1
        self.sum += value
        self.count += 1

    def lines(self, name: str):
        running = 0
        for bound, n in zip(self.buckets, self.counts):
            running += n
            yield f'{name}_bucket{{le="{bound:g}"}} {running}'
        yield f'{name}_bucket{{le="+Inf"}} {self.count}'
        yield f"{name}_sum {self.sum:.6f}"
        yield f"{name}_count {self.count}"


def _escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _labels(**labels) -> str:
    return "{" + ",".join(f'{k}="{_escape(v)}"' for k, v in labels.items()) + "}"


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.requests: dict[tuple[str, str], int] = {}
        self.predictions: dict[str, int] = {}
        self.rejected: dict[str, int] = {}
        self.errors: dict[str, int] = {}
        self.confidence = Histogram(CONFIDENCE_BUCKETS)
        self.predict_latency = Histogram(LATENCY_BUCKETS)
        self.inference_latency = Histogram(LATENCY_BUCKETS)

    def observe_request(self, endpoint: str, status: int, error_code: str | None = None) -> None:
        with self._lock:
            key = (endpoint, str(status))
            self.requests[key] = self.requests.get(key, 0) + 1
            if status >= 500:
                category = error_code or "internal_error"
                self.errors[category] = self.errors.get(category, 0) + 1
            elif status >= 400 and endpoint == "/predict":
                reason = error_code or "rejected"
                self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def observe_prediction(self, label: str, confidence: float, total_s: float, inference_s: float) -> None:
        with self._lock:
            self.predictions[label] = self.predictions.get(label, 0) + 1
            self.confidence.observe(confidence)
            self.predict_latency.observe(total_s)
            self.inference_latency.observe(inference_s)

    def render(self, model: dict | None = None) -> str:
        """Prometheus text format. `model` is {"name", "alias", "version"} of the served model, or None."""
        out = []

        def block(name, help_text, kind, lines):
            out.extend([f"# HELP {name} {help_text}", f"# TYPE {name} {kind}", *lines])

        with self._lock:
            block("plant_requests_total", "HTTP requests by endpoint and status code.", "counter",
                  [f"plant_requests_total{_labels(endpoint=e, status=s)} {n}"
                   for (e, s), n in sorted(self.requests.items())])
            block("plant_predictions_total", "Successful predictions by predicted class.", "counter",
                  [f"plant_predictions_total{_labels(label=k)} {n}" for k, n in sorted(self.predictions.items())])
            block("plant_rejected_requests_total", "Prediction requests refused because of the input, by reason.",
                  "counter", [f"plant_rejected_requests_total{_labels(reason=k)} {n}"
                              for k, n in sorted(self.rejected.items())])
            block("plant_errors_total", "Server-side failures (HTTP 5xx), by category.", "counter",
                  [f"plant_errors_total{_labels(category=k)} {n}" for k, n in sorted(self.errors.items())])
            block("plant_prediction_confidence", "Top-1 softmax probability of successful predictions.", "histogram",
                  list(self.confidence.lines("plant_prediction_confidence")))
            block("plant_predict_duration_seconds", "Total time spent handling successful /predict requests.",
                  "histogram", list(self.predict_latency.lines("plant_predict_duration_seconds")))
            block("plant_inference_duration_seconds", "Model inference time of successful predictions.", "histogram",
                  list(self.inference_latency.lines("plant_inference_duration_seconds")))
        block("plant_model_loaded", "1 if a model is loaded and served, else 0.", "gauge",
              [f"plant_model_loaded {1 if model else 0}"])
        block("plant_model_info", "The model version currently served.", "gauge",
              [f"plant_model_info{_labels(name=model['name'], alias=model['alias'], version=model['version'])} 1"]
              if model else [])
        return "\n".join(out) + "\n"
