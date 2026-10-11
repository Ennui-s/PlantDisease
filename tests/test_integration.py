"""Integration tests: the real exported champion model, preprocessing parity, and the real MLflow registry.

They are skipped automatically when torch/MLflow or model_export/ are not available.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from tests.helpers import image_bytes

MODEL_DIR = Path("model_export")


def need_model():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("mlflow")
    if not (MODEL_DIR / "model_info.json").is_file():
        pytest.skip("model_export/ not found (create it with: python -m registry export)")


def test_exported_model_loads_and_predicts():
    need_model()
    from serving.model_store import local_loader
    from serving.settings import Settings

    loaded = local_loader(Settings(model_dir=str(MODEL_DIR)))
    assert len(loaded.classes) == 15 and loaded.input_size == 224
    batch = np.random.default_rng(0).random((2, 3, 224, 224), dtype=np.float32)
    probs = loaded.predict(batch)
    assert probs.shape == (2, 15) and np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)
    assert np.array_equal(probs, loaded.predict(batch))            # deterministic


def test_serving_preprocessing_matches_the_training_transform():
    need_model()
    from PIL import Image
    from torchvision import transforms

    from config import CONFIG
    from serving.images import load_image, preprocess

    size = CONFIG["image_size"]
    training = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    from serving.settings import Settings
    for fmt in ("JPEG", "PNG"):
        data = image_bytes(fmt)
        import io
        expected = training(Image.open(io.BytesIO(data)).convert("RGB")).numpy()[None, ...]
        actual = preprocess(load_image(data, Settings()), size)
        assert expected.shape == actual.shape
        assert np.abs(expected - actual).max() <= 1e-6


def test_real_model_through_the_api():
    need_model()
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from serving.app import create_app
    from serving.settings import Settings

    info = json.loads((MODEL_DIR / "model_info.json").read_text())
    with TestClient(create_app(settings=Settings(model_dir=str(MODEL_DIR)))) as client:
        assert client.get("/ready").status_code == 200
        assert client.get("/model").json()["version"] == info["version"]
        data = image_bytes()
        first = client.post("/predict", files={"file": ("leaf.jpg", data, "image/jpeg")})
        assert first.status_code == 200
        body = first.json()
        assert body["prediction"] in client.get("/model").json()["classes"] and 0 < body["confidence"] <= 1
        second = client.post("/predict", files={"file": ("leaf.jpg", data, "image/jpeg")}).json()
        assert (second["prediction"], second["confidence"]) == (body["prediction"], body["confidence"])
        assert client.post("/predict", files={"file": ("x.jpg", b"not an image", "image/jpeg")}).status_code == 415


def test_registry_operations_against_a_real_mlflow_store(tmp_path):
    """Promotion, listing and export against a real (temporary, SQLite) MLflow registry with a dummy model."""
    pytest.importorskip("mlflow")
    import mlflow
    import mlflow.pyfunc

    from registry import ops
    from training.run_config import load_run_config

    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    run_cfg = load_run_config(None)
    run_cfg["tracking"] = {"tracking_uri": uri, "experiment_name": "registry-test"}

    class Dummy(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            return model_input

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("registry-test")
    versions = []
    for _ in range(2):
        with mlflow.start_run() as run:
            model_dir = tmp_path / f"m{len(versions)}"
            mlflow.pyfunc.save_model(path=str(model_dir), python_model=Dummy())
            mlflow.log_artifacts(str(model_dir), "model")
        versions.append(mlflow.register_model(f"runs:/{run.info.run_id}/model", "plant-disease-classifier"))

    client = ops.MlflowClient(tracking_uri=uri, registry_uri=uri)
    client.set_model_version_tag("plant-disease-classifier", "1", "quality_gate", "passed")
    client.set_model_version_tag("plant-disease-classifier", "2", "quality_gate", "failed")
    client.set_registered_model_alias("plant-disease-classifier", "champion", "1")

    with pytest.raises(ops.RegistryError):                                     # gate not passed -> refused
        ops.promote("2", run_cfg)
    with pytest.raises(ops.RegistryError):                                     # unknown version
        ops.promote("9", run_cfg)
    client.set_model_version_tag("plant-disease-classifier", "2", "quality_gate", "passed")
    result = ops.promote("2", run_cfg)
    assert result["version"] == "2" and result["previous"] == "1"
    assert ops.champion_version(run_cfg) == "2"
    empty_cfg = load_run_config(None)
    empty_cfg["tracking"] = {"tracking_uri": f"sqlite:///{(tmp_path / 'empty.db').as_posix()}"}
    assert ops.champion_version(empty_cfg) is None          # a registry with no champion (or no model at all)

    listing = ops.list_versions(run_cfg)
    assert [v["version"] for v in listing["versions"]] == ["1", "2"]
    assert [v["aliases"] for v in listing["versions"]] == [[], ["champion"]]

    out = tmp_path / "export"
    info = ops.export_champion(run_cfg, str(out))
    assert info["version"] == "2" and (out / "model" / "MLmodel").is_file()
    assert json.loads((out / "model_info.json").read_text())["alias"] == "champion"
