"""Unit tests for upload validation, preprocessing, model store and the prediction function."""
import numpy as np
import pytest
from PIL import Image

from serving.errors import ApiError
from serving.images import load_image, preprocess
from serving.model_store import LoadedModel, ModelStore, build_prediction
from serving.service import predict_bytes
from serving.settings import Settings
from tests.helpers import image_bytes

S = Settings()
CLASSES = ["a", "b", "c", "d"]


def refusal(data, settings=S):
    with pytest.raises(ApiError) as exc:
        load_image(data, settings)
    return exc.value.status, exc.value.code


@pytest.mark.parametrize("fmt, mode", [("JPEG", "RGB"), ("PNG", "RGB"), ("PNG", "RGBA"), ("PNG", "L"), ("PNG", "P")])
def test_valid_images_are_accepted_and_converted_to_rgb(fmt, mode):
    assert load_image(image_bytes(fmt, mode=mode), S).mode == "RGB"


def test_invalid_uploads_get_specific_errors():
    good = image_bytes()
    assert refusal(b"") == (400, "empty_file")
    assert refusal(b"plain text, not an image") == (415, "unsupported_media_type")
    assert refusal(image_bytes("GIF")) == (415, "unsupported_media_type")
    assert refusal(image_bytes("BMP")) == (415, "unsupported_media_type")
    assert refusal(good[: len(good) // 2]) == (422, "image_corrupted")
    assert refusal(image_bytes(size=(8, 8))) == (422, "image_too_small")
    assert refusal(image_bytes(size=(400, 300)), Settings(max_pixels=100_000)) == (413, "image_too_large")


def test_preprocess_shape_dtype_range_and_determinism():
    image = load_image(image_bytes(), S)
    x = preprocess(image, 224)
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0 and x.flags["C_CONTIGUOUS"]
    assert np.array_equal(x, preprocess(image, 224))
    solid = preprocess(Image.new("RGB", (256, 256), (255, 0, 51)), 224)
    assert np.allclose(solid[0, :, 0, 0] * 255, [255, 0, 51])     # channel order is RGB


def test_build_prediction_orders_and_validates():
    result = build_prediction([0.1, 0.6, 0.25, 0.05], CLASSES, 3)
    assert result["prediction"] == "b" and result["confidence"] == 0.6
    assert [t["label"] for t in result["top_k"]] == ["b", "c", "a"]
    with pytest.raises(ValueError):
        build_prediction([0.5, 0.5], CLASSES, 3)
    with pytest.raises(ValueError):
        build_prediction([0.1, float("nan"), 0.3, 0.6], CLASSES, 3)


def fake_loader(version="1", fail=False, broken=False):
    def loader(settings):
        if fail:
            raise RuntimeError("registry unreachable at /secret/path")

        def predict(batch):
            assert batch.shape == (1, 3, 224, 224) and batch.dtype == np.float32
            if broken:
                raise RuntimeError("boom inside the model")
            return np.array([[0.05, 0.9, 0.03, 0.02]], dtype=np.float32)
        return LoadedModel(predict=predict, classes=CLASSES, name="m", alias="champion", version=version,
                           input_size=224, tags={"test_accuracy": "0.95", "secret": "x"}, loaded_at="t")
    return loader


def test_prediction_requires_a_loaded_model():
    store = ModelStore(S, fake_loader())
    with pytest.raises(ApiError) as exc:
        predict_bytes(store, S, image_bytes(), "r1")
    assert exc.value.status == 503


def test_prediction_response():
    store = ModelStore(S, fake_loader("7"))
    assert store.load()
    out = predict_bytes(store, S, image_bytes(), "req-1")
    assert out["prediction"] == "b" and out["confidence"] == 0.9
    assert out["model"] == {"name": "m", "alias": "champion", "version": "7"} and out["request_id"] == "req-1"
    assert set(out["latency_ms"]) == {"preprocess_ms", "inference_ms", "total_ms"}
    assert "secret" not in store.require().info()["tags"]          # only selected tags are exposed


def test_oversized_and_empty_uploads_are_refused():
    store = ModelStore(S, fake_loader())
    store.load()
    with pytest.raises(ApiError) as big:
        predict_bytes(store, S, b"x" * (S.max_upload_bytes + 1), "r")
    assert big.value.status == 413
    with pytest.raises(ApiError) as empty:
        predict_bytes(store, S, b"", "r")
    assert empty.value.status == 400


def test_failed_reload_keeps_the_previous_model_and_hides_details():
    store = ModelStore(S, fake_loader("1"))
    assert store.load()
    store._loader = fake_loader("2", fail=True)
    assert store.load() is False and store.ready and store.require().version == "1"
    assert "secret" in store.last_error                     # kept server-side for logs only
    store._loader = fake_loader("2")
    assert store.load() and store.require().version == "2"
    assert ModelStore(S, fake_loader(fail=True)).load() is False


def test_model_crash_becomes_a_generic_500():
    store = ModelStore(S, fake_loader(broken=True))
    store.load()
    with pytest.raises(ApiError) as exc:
        predict_bytes(store, S, image_bytes(), "r")
    assert exc.value.status == 500 and "boom" not in exc.value.message


def test_settings_from_environment(monkeypatch):
    monkeypatch.setenv("MODEL_ALIAS", "challenger")
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "1000")
    monkeypatch.setenv("MODEL_DIR", "/x")
    s = Settings.from_env()
    assert (s.model_alias, s.max_upload_bytes, s.model_dir, s.admin_token) == ("challenger", 1000, "/x", None)
    assert ModelStore(Settings())._loader.__name__ == "mlflow_loader"
    assert ModelStore(Settings(model_dir="/x"))._loader.__name__ == "local_loader"
