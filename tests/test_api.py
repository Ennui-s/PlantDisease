"""API tests: the real FastAPI app with a stand-in model (no MLflow or torch needed)."""
import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from serving.app import create_app  # noqa: E402
from serving.model_store import LoadedModel, ModelStore  # noqa: E402
from serving.settings import Settings  # noqa: E402
from tests.helpers import image_bytes  # noqa: E402

CLASSES = ["a", "b", "c"]


def loader(fail=False, broken=False):
    def load(settings):
        if fail:
            raise RuntimeError("cannot load /secret/path")

        def predict(batch):
            if broken:
                raise RuntimeError("boom inside the model")
            return np.array([[0.1, 0.8, 0.1]], dtype=np.float32)
        return LoadedModel(predict=predict, classes=CLASSES, name="plant-disease-classifier", alias="champion",
                           version="3", input_size=224, tags={"test_accuracy": "0.95", "quality_gate": "passed"})
    return load


def client_for(settings=None, **loader_kwargs):
    settings = settings or Settings()
    store = ModelStore(settings, loader(**loader_kwargs))
    return TestClient(create_app(store=store, settings=settings), raise_server_exceptions=False)


def upload(client, data, name="leaf.jpg", headers=None):
    return client.post("/predict", files={"file": (name, data, "image/jpeg")}, headers=headers or {})


def assert_error(response, status, code):
    assert response.status_code == status
    body = response.json()["error"]
    assert body["code"] == code and body["request_id"] == response.headers["x-request-id"]
    assert "Traceback" not in response.text and "/secret" not in response.text


def test_health_and_ready():
    with client_for() as client:
        assert client.get("/health").json() == {"status": "ok"}
        ready = client.get("/ready")
        assert ready.status_code == 200 and ready.json()["model"]["version"] == "3"


def test_not_ready_when_the_model_cannot_load():
    with client_for(fail=True) as client:
        assert client.get("/health").status_code == 200          # the process is still alive
        assert client.get("/ready").status_code == 503
        assert_error(client.get("/model"), 503, "model_unavailable")
        assert_error(upload(client, image_bytes()), 503, "model_unavailable")


def test_model_endpoint():
    with client_for() as client:
        body = client.get("/model").json()
        assert body["version"] == "3" and body["alias"] == "champion" and body["num_classes"] == 3
        assert body["tags"]["quality_gate"] == "passed"


def test_predict_valid_image():
    with client_for() as client:
        response = upload(client, image_bytes(), headers={"X-Request-ID": "my-request-1"})
        assert response.status_code == 200
        body = response.json()
        assert body["prediction"] == "b" and body["confidence"] == 0.8
        assert body["model"]["version"] == "3" and body["request_id"] == "my-request-1"
        assert response.headers["x-request-id"] == "my-request-1"
        assert body["latency_ms"]["total_ms"] >= body["latency_ms"]["inference_ms"]


def test_predict_accepts_png():
    with client_for() as client:
        assert upload(client, image_bytes("PNG", mode="RGBA"), name="leaf.png").status_code == 200


def test_invalid_uploads():
    with client_for() as client:
        good = image_bytes()
        assert_error(upload(client, b"just some text"), 415, "unsupported_media_type")
        assert_error(upload(client, image_bytes("GIF")), 415, "unsupported_media_type")
        assert_error(upload(client, good[: len(good) // 2]), 422, "image_corrupted")
        assert_error(upload(client, b""), 400, "empty_file")
        assert_error(upload(client, image_bytes(size=(8, 8))), 422, "image_too_small")


def test_malformed_requests():
    with client_for() as client:
        assert_error(client.post("/predict"), 422, "malformed_request")                        # no body
        assert_error(client.post("/predict", json={"file": "x"}), 422, "malformed_request")    # not multipart
        assert_error(client.post("/predict", files={"other": ("x.jpg", image_bytes(), "image/jpeg")}),
                     422, "malformed_request")                                                 # wrong field name


def test_oversized_upload_is_rejected():
    with client_for(Settings(max_upload_bytes=1000)) as client:
        assert_error(upload(client, b"x" * 5000), 413, "file_too_large")


def test_unknown_route_and_wrong_method_use_the_error_format():
    with client_for() as client:
        assert_error(client.get("/nope"), 404, "not_found")
        assert_error(client.get("/predict"), 405, "method_not_allowed")


def test_model_failure_gives_generic_500_without_internals():
    with client_for(broken=True) as client:
        response = upload(client, image_bytes())
        assert_error(response, 500, "inference_failed")
        assert "boom" not in response.text


def test_unexpected_exception_gives_generic_500(monkeypatch):
    with client_for() as client:
        def explode():
            raise RuntimeError("secret internals")
        monkeypatch.setattr(client.app.state.store, "require", explode)
        response = client.get("/model")
        assert_error(response, 500, "internal_error")
        assert "secret internals" not in response.text


def test_reload_is_absent_without_a_token_and_protected_with_one():
    with client_for() as client:
        assert client.post("/reload").status_code == 404
    with client_for(Settings(admin_token="s3cret")) as client:
        assert_error(client.post("/reload"), 401, "unauthorized")
        assert_error(client.post("/reload", headers={"X-Admin-Token": "wrong"}), 401, "unauthorized")
        ok = client.post("/reload", headers={"X-Admin-Token": "s3cret"})
        assert ok.status_code == 200 and ok.json()["reloaded"] is True and ok.json()["model"]["version"] == "3"


def test_metrics_and_logs_reflect_traffic():
    import json
    import logging

    from serving.metrics_summary import parse

    class Capture(logging.Handler):
        def __init__(self):
            super().__init__(level=logging.DEBUG)
            self.lines = []

        def emit(self, record):
            self.lines.append(record.getMessage())

    capture = Capture()
    logger = logging.getLogger("plant_api.access")
    with client_for() as client:
        logger.addHandler(capture)
        logger.setLevel(logging.DEBUG)
        try:
            for _ in range(2):
                assert upload(client, image_bytes(), name="SECRET-leaf-name.jpg").status_code == 200
            assert upload(client, b"just text").status_code == 415
            assert client.get("/nope").status_code == 404
            metrics = client.get("/metrics")
        finally:
            logger.removeHandler(capture)
        assert metrics.status_code == 200 and metrics.headers["content-type"].startswith("text/plain")
        samples = parse(metrics.text)

        def value(name, **labels):
            return next(v for n, l, v in samples if n == name and all(l.get(k) == x for k, x in labels.items()))

        assert value("plant_requests_total", endpoint="/predict", status="200") == 2
        assert value("plant_requests_total", endpoint="/predict", status="415") == 1
        assert value("plant_requests_total", endpoint="unmatched", status="404") == 1
        assert value("plant_predictions_total", label="b") == 2
        assert value("plant_rejected_requests_total", reason="unsupported_media_type") == 1
        assert value("plant_prediction_confidence_count") == 2 and value("plant_model_loaded") == 1
        assert value("plant_model_info", version="3") == 1

    records = [json.loads(line) for line in capture.lines]
    predictions = [r for r in records if r["endpoint"] == "/predict" and r["status"] == 200]
    assert len(predictions) == 2 and predictions[0]["prediction"] == "b" and predictions[0]["model_version"] == "3"
    assert {"timestamp", "request_id", "endpoint", "status", "latency_ms", "confidence"} <= set(predictions[0])
    assert [r for r in records if r["status"] == 415][0]["error_category"] == "unsupported_media_type"
    assert not any("SECRET" in line for line in capture.lines)         # file names are never logged


def test_server_errors_are_counted():
    from serving.metrics_summary import parse

    with client_for(broken=True) as client:
        assert upload(client, image_bytes()).status_code == 500
        samples = parse(client.get("/metrics").text)
        assert any(n == "plant_errors_total" and l["category"] == "inference_failed" and v == 1 for n, l, v in samples)
