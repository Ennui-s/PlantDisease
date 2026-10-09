"""Holds the champion model for the service and turns its output into an API response.

The model is addressed only through the registry alias (models:/<name>@<alias>); no file path is
configured anywhere. It is loaded by the concrete version the alias resolves to, so the version
reported by the API is always the version that is actually loaded.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, List, Optional

import numpy as np

from serving.errors import ApiError
from serving.settings import Settings

REPORTED_TAGS = ("test_accuracy", "test_macro_f1", "quality_gate", "dataset_fingerprint",
                 "git_commit", "training_run_id", "model_sha256")


@dataclass
class LoadedModel:
    predict: Callable[[np.ndarray], np.ndarray]
    classes: List[str]
    name: str
    alias: str
    version: str
    input_size: int
    run_id: Optional[str] = None
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


def mlflow_loader(settings: Settings) -> LoadedModel:
    """Resolve the alias to a version in the registry and load that version."""
    import mlflow
    import mlflow.pyfunc
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(settings.tracking_uri)
    client = MlflowClient(tracking_uri=settings.tracking_uri, registry_uri=settings.tracking_uri)
    mv = client.get_model_version_by_alias(settings.model_name, settings.model_alias)
    model = mlflow.pyfunc.load_model(f"models:/{settings.model_name}/{mv.version}")

    classes = list(model.unwrap_python_model().classes)
    try:  # input size from the model signature: (-1, 3, H, W)
        input_size = int(model.metadata.signature.inputs.inputs[0].shape[-1])
    except Exception:
        input_size = settings.default_input_size
    return LoadedModel(
        predict=model.predict, classes=classes, name=settings.model_name, alias=settings.model_alias,
        version=str(mv.version), input_size=input_size, run_id=getattr(mv, "run_id", None),
        tags=dict(mv.tags or {}), loaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))


class ModelStore:
    def __init__(self, settings: Settings, loader: Optional[Callable[[Settings], LoadedModel]] = None):
        self.settings = settings
        self._loader = loader or mlflow_loader
        self._loaded: Optional[LoadedModel] = None
        self._load_lock = threading.Lock()
        self.infer_lock = threading.Lock()  # one prediction at a time; also keeps a reload from interleaving
        self.last_error: Optional[str] = None  # kept server-side; never returned to API users

    def load(self) -> bool:
        """(Re)load the champion. On failure the previously loaded model, if any, keeps serving."""
        with self._load_lock:
            try:
                self._loaded = self._loader(self.settings)
                self.last_error = None
                return True
            except Exception as exc:
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


def build_prediction(probabilities, classes: List[str], top_k: int) -> dict:
    """Top-k classes with probabilities from one row of model output."""
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if probs.shape[0] != len(classes):
        raise ValueError(f"model returned {probs.shape[0]} scores for {len(classes)} classes")
    if not np.all(np.isfinite(probs)):
        raise ValueError("model returned non-finite scores")
    order = np.argsort(-probs)[:max(1, top_k)]
    top = [{"label": classes[i], "probability": round(float(probs[i]), 6)} for i in order]
    return {"prediction": top[0]["label"], "confidence": top[0]["probability"], "top_k": top}
