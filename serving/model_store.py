"""Holds the champion model for the service and turns its output into an API response.

The model is addressed through the registry alias (models:/<name>@<alias>), or, in a container, from an
exported copy of the champion (MODEL_DIR). It is loaded by the concrete version the alias resolves to, so the version
reported by the API is always the version that is actually loaded.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from serving.errors import ApiError
from serving.settings import Settings

REPORTED_TAGS = ("test_accuracy", "test_macro_f1", "quality_gate", "dataset_fingerprint",
                 "git_commit", "training_run_id", "model_sha256")


@dataclass
class LoadedModel:
    predict: Callable[[np.ndarray], np.ndarray]
    classes: list[str]
    name: str
    alias: str
    version: str
    input_size: int
    run_id: str | None = None
    tags: dict = field(default_factory=dict)
    loaded_at: str = ""

    def ref(self) -> dict:
        return {"name": self.name, "alias": self.alias, "version": self.version}

    def info(self) -> dict:
        return {**self.ref(), "num_classes": len(self.classes), "classes": self.classes,
                "input": {"shape": [1, 3, self.input_size, self.input_size], "dtype": "float32",
                          "range": [0.0, 1.0]},
                "tags": {k: self.tags[k] for k in REPORTED_TAGS if k in self.tags},
                "loaded_at": self.loaded_at}


def _wrap(model, settings: Settings, name: str, alias: str, version: str, run_id, tags) -> LoadedModel:
    classes = list(model.unwrap_python_model().classes)
    try:  # input size from the model signature: (-1, 3, H, W)
        input_size = int(model.metadata.signature.inputs.inputs[0].shape[-1])
    except (AttributeError, IndexError, TypeError, ValueError):   # no signature, or not the expected shape
        input_size = settings.default_input_size
    return LoadedModel(
        predict=model.predict, classes=classes, name=name, alias=alias, version=str(version),
        input_size=input_size, run_id=run_id, tags=dict(tags or {}),
        loaded_at=datetime.now(UTC).isoformat(timespec="seconds"))


def mlflow_loader(settings: Settings) -> LoadedModel:
    """Resolve the alias to a version in the registry and load that version."""
    import mlflow
    import mlflow.pyfunc
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(settings.tracking_uri)
    client = MlflowClient(tracking_uri=settings.tracking_uri, registry_uri=settings.tracking_uri)
    mv = client.get_model_version_by_alias(settings.model_name, settings.model_alias)
    model = mlflow.pyfunc.load_model(f"models:/{settings.model_name}/{mv.version}")
    return _wrap(model, settings, settings.model_name, settings.model_alias, mv.version,
                 getattr(mv, "run_id", None), mv.tags)


def local_loader(settings: Settings) -> LoadedModel:
    """Load an exported model folder (python -m registry export): <dir>/model and <dir>/model_info.json.

    Used by the container image, which carries its own copy of the champion and needs no registry.
    """
    import json
    from pathlib import Path

    import mlflow.pyfunc

    folder = Path(settings.model_dir)
    info = json.loads((folder / "model_info.json").read_text())
    model = mlflow.pyfunc.load_model(str(folder / "model"))
    return _wrap(model, settings, info["name"], info["alias"], info["version"], info.get("run_id"), info.get("tags"))


class ModelStore:
    def __init__(self, settings: Settings, loader: Callable[[Settings], LoadedModel] | None = None):
        self.settings = settings
        self._loader = loader or (local_loader if settings.model_dir else mlflow_loader)
        self._loaded: LoadedModel | None = None
        self._load_lock = threading.Lock()
        self.infer_lock = threading.Lock()  # one prediction at a time; also keeps a reload from interleaving
        self.last_error: str | None = None  # kept server-side; never returned to API users

    def load(self) -> bool:
        """(Re)load the champion. On failure the previously loaded model, if any, keeps serving."""
        with self._load_lock:
            try:
                self._loaded = self._loader(self.settings)
                self.last_error = None
                return True
            except Exception as exc:  # noqa: BLE001  any loader failure must leave the previous model serving
                self.last_error = f"{type(exc).__name__}: {exc}"
                return False

    @property
    def ready(self) -> bool:
        return self._loaded is not None

    def require(self) -> LoadedModel:
        loaded = self._loaded
        if loaded is None:
            raise ApiError(503, "model_unavailable", "The model is not available right now.")
        return loaded


def build_prediction(probabilities, classes: list[str], top_k: int) -> dict:
    """Top-k classes with probabilities from one row of model output."""
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if probs.shape[0] != len(classes):
        raise ValueError(f"model returned {probs.shape[0]} scores for {len(classes)} classes")
    if not np.all(np.isfinite(probs)):
        raise ValueError("model returned non-finite scores")
    order = np.argsort(-probs)[:max(1, top_k)]
    top = [{"label": classes[i], "probability": round(float(probs[i]), 6)} for i in order]
    return {"prediction": top[0]["label"], "confidence": top[0]["probability"], "top_k": top}
