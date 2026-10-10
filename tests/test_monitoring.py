"""Unit tests for structured logging and metrics (no web framework needed)."""
import json
import logging

import pytest

from serving import monitoring
from serving.metrics_summary import parse, quantile, summarize
from serving.monitoring import Histogram, Metrics, log_request


class Capture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def captured():
    logger = logging.getLogger(monitoring.ACCESS_LOGGER)
    handler = Capture()
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    yield handler
    logger.removeHandler(handler)
    logger.setLevel(old_level)


def test_histogram_is_cumulative_and_consistent():
    h = Histogram((0.5, 1.0))
    for v in (0.2, 0.5, 0.7, 5.0):
        h.observe(v)
    lines = list(h.lines("x"))
    assert lines == ['x_bucket{le="0.5"} 2', 'x_bucket{le="1"} 3', 'x_bucket{le="+Inf"} 4', "x_sum 6.400000", "x_count 4"]


def test_counting_rules():
    m = Metrics()
    m.observe_request("/predict", 200)
    m.observe_request("/predict", 415, "unsupported_media_type")
    m.observe_request("/predict", 422, "image_corrupted")
    m.observe_request("/predict", 500, "inference_failed")
    m.observe_request("/model", 503, "model_unavailable")
    m.observe_request("unmatched", 404, "not_found")          # a 404 elsewhere is not a prediction rejection
    assert m.rejected == {"unsupported_media_type": 1, "image_corrupted": 1}
    assert m.errors == {"inference_failed": 1, "model_unavailable": 1}
    assert m.requests[("/predict", "200")] == 1 and m.requests[("unmatched", "404")] == 1


def test_render_is_valid_prometheus_text_and_round_trips():
    m = Metrics()
    m.observe_request("/predict", 200)
    m.observe_prediction("Tomato_healthy", 0.97, 0.15, 0.12)
    m.observe_prediction('weird"label\\x', 0.4, 0.2, 0.1)
    text = m.render({"name": "plant-disease-classifier", "alias": "champion", "version": "1"})
    assert text.endswith("\n") and "# TYPE plant_requests_total counter" in text
    samples = parse(text)
    labels = {l["label"] for n, l, v in samples if n == "plant_predictions_total"}
    assert labels == {"Tomato_healthy", 'weird"label\\x'}                      # escaping survives a round trip
    for name in ("plant_prediction_confidence", "plant_predict_duration_seconds"):
        buckets = [v for n, l, v in samples if n == f"{name}_bucket"]
        assert buckets == sorted(buckets)                                      # cumulative, never decreasing
        assert buckets[-1] == next(v for n, _, v in samples if n == f"{name}_count") == 2
    assert next(v for n, _, v in samples if n == "plant_model_loaded") == 1
    assert ("plant_model_info", {"name": "plant-disease-classifier", "alias": "champion", "version": "1"}, 1.0) in samples


def test_render_without_a_model():
    samples = parse(Metrics().render(None))
    assert next(v for n, _, v in samples if n == "plant_model_loaded") == 0
    assert not [s for s in samples if s[0] == "plant_model_info"]


def test_summary_numbers():
    m = Metrics()
    for conf, total in ((0.99, 0.12), (0.95, 0.14), (0.4, 0.3)):
        m.observe_request("/predict", 200)
        m.observe_prediction("Tomato_healthy" if conf > 0.9 else "Potato___healthy", conf, total, total - 0.02)
    m.observe_request("/predict", 415, "unsupported_media_type")
    m.observe_request("/predict", 500, "inference_failed")
    text = summarize(m.render({"name": "plant-disease-classifier", "alias": "champion", "version": "2"}))
    assert "version 2" in text
    assert "/predict requests: 5   successful: 3   rejected: 1   server errors: 1" in text
    assert "Rejection rate: 20.0%" in text and "unsupported_media_type 1" in text and "inference_failed 1" in text
    assert "mean 0.780" in text and "33.3% of predictions below 0.7" in text
    assert "Tomato_healthy 2, Potato___healthy 1" in text


def test_quantile_estimate():
    buckets = [(0.1, 50.0), (0.5, 90.0), (1.0, 100.0)]
    assert quantile(buckets, 100.0, 0.5) == pytest.approx(0.1)
    assert quantile(buckets, 100.0, 0.7) == pytest.approx(0.3)
    assert quantile([], 0.0, 0.5) is None


def test_log_lines_are_json_with_only_allowed_fields(captured):
    log_request(request_id="abc", method="POST", endpoint="/predict", status=200, latency_ms=12.3,
                model_version="1", prediction="Tomato_healthy", confidence=0.99, inference_ms=10.0,
                filename="SECRET-photo.jpg", image=b"\xff\xd8binary", authorization="Bearer token", error_category=None)
    line = captured.records[0].getMessage()
    record = json.loads(line)
    assert set(record) <= set(monitoring.LOG_FIELDS)
    assert record["request_id"] == "abc" and record["prediction"] == "Tomato_healthy" and "timestamp" in record
    assert "SECRET" not in line and "binary" not in line and "Bearer" not in line
    assert "error_category" not in record                                       # empty values are omitted


def test_log_levels(captured):
    log_request(request_id="1", method="GET", endpoint="/ready", status=200)
    log_request(request_id="2", method="GET", endpoint="/ready", status=503)
    log_request(request_id="3", method="POST", endpoint="/predict", status=415, error_category="unsupported_media_type")
    log_request(request_id="4", method="POST", endpoint="/predict", status=500, error_category="inference_failed")
    log_request(request_id="5", method="POST", endpoint="/predict", status=200)
    # successful probe -> DEBUG; 503 and 500 are server-side -> ERROR; 415 is the client's fault -> WARNING
    assert [r.levelno for r in captured.records] == [logging.DEBUG, logging.ERROR, logging.WARNING,
                                                     logging.ERROR, logging.INFO]
