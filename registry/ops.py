"""Registry operations on top of MLflow: package + register a checkpoint, promote a version, inspect, load."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import mlflow
from mlflow import MlflowClient

from config import CONFIG
from registry.policy import (check_registration_preconditions, find_version_by_sha, promotion_problems,
                             registry_cfg, version_tags)
from tracking.tracker import RunTracker, tracking_cfg, training_run_id


class RegistryError(RuntimeError):
    """A registry operation was refused (not a crash): the message explains why."""


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _setup(run_cfg: dict):
    track = tracking_cfg(run_cfg)
    reg = registry_cfg(run_cfg)
    mlflow.set_tracking_uri(track["tracking_uri"])
    client = MlflowClient(tracking_uri=track["tracking_uri"], registry_uri=track["tracking_uri"])
    return track, reg, client


def _package(checkpoint_path: str, classes: list, out_dir: Path) -> Path:
    """Write the MLflow model (weights, class names, code, signature) to out_dir/model."""
    import numpy as np
    import torch
    from mlflow.models import ModelSignature
    from mlflow.types import Schema, TensorSpec
    from registry.pyfunc_model import PlantDiseaseClassifier

    work = Path(tempfile.mkdtemp(prefix="pkg_"))
    weights = work / "weights.pth"
    # keep only the model weights (the training checkpoint also holds optimizer state)
    torch.save({"model": torch.load(checkpoint_path, map_location="cpu")["model"]}, weights)
    classes_file = work / "classes.json"
    classes_file.write_text(json.dumps(classes))

    size = CONFIG["image_size"]
    signature = ModelSignature(
        inputs=Schema([TensorSpec(np.dtype("float32"), (-1, 3, size, size))]),
        outputs=Schema([TensorSpec(np.dtype("float32"), (-1, len(classes)))]))
    base = lambda v: v.split("+")[0]  # strip local build tags such as +cu121
    import torchvision
    path = out_dir / "model"
    mlflow.pyfunc.save_model(
        path=str(path),
        python_model=PlantDiseaseClassifier(),
        artifacts={"checkpoint": str(weights), "classes": str(classes_file)},
        code_paths=["registry", "models", "config.py"],
        signature=signature,
        pip_requirements=[f"torch=={base(torch.__version__)}",
                          f"torchvision=={base(torchvision.__version__)}", "numpy"])
    return path


def register(model_path: str, run_cfg: dict, run_id: Optional[str] = None,
             eval_dir: Optional[str] = None) -> dict:
    """Register a gate-passed checkpoint as a new model version. The first version becomes the champion."""
    track, reg, client = _setup(run_cfg)
    stem = Path(model_path).stem
    eval_dir = eval_dir or str(Path("reports") / "evaluation" / stem)

    check = check_registration_preconditions(model_path, eval_dir, _sha256)
    if not check["ok"]:
        raise RegistryError("Not registered:\n  - " + "\n  - ".join(check["problems"]))
    metrics, gate = check["metrics"], check["gate"]

    existing = find_version_by_sha(client.search_model_versions(f"name='{reg['model_name']}'"),
                                   metrics["model_sha256"])
    if existing is not None:
        return {"status": "already_registered", "version": str(existing.version), "name": reg["model_name"]}

    run_id = run_id or training_run_id(model_path)
    cfg = {**track, "enabled": True}  # registering needs a run to hold the model artifacts
    with RunTracker(cfg, run_name=None if run_id else f"register-{stem}", resume_run_id=run_id) as tracker:
        run_id = tracker.run_id
        with tempfile.TemporaryDirectory() as tmp:
            tracker.log_artifacts(_package(model_path, metrics["classes"], Path(tmp)), "model")
        run_tags = client.get_run(run_id).data.tags

    mv = mlflow.register_model(f"runs:/{run_id}/model", reg["model_name"])
    tags = version_tags(metrics, gate, run_tags, model_path, run_id)
    for key, value in tags.items():
        client.set_model_version_tag(reg["model_name"], mv.version, key, value)
    client.update_model_version(
        reg["model_name"], mv.version,
        description=f"Plant disease classifier; test accuracy {metrics['accuracy']:.4f}, "
                    f"macro F1 {metrics['macro_f1']:.4f}; quality gate passed.")

    aliases = client.get_registered_model(reg["model_name"]).aliases or {}
    became_champion = reg["champion_alias"] not in aliases
    if became_champion:  # bootstrap: the first registered version has nothing to be compared with
        client.set_registered_model_alias(reg["model_name"], reg["champion_alias"], mv.version)
    return {"status": "registered", "version": str(mv.version), "name": reg["model_name"],
            "became_champion": became_champion, "run_id": run_id}


def promote(version: str, run_cfg: dict) -> dict:
    """Point the champion alias at a registered version (it must carry quality_gate=passed)."""
    _, reg, client = _setup(run_cfg)
    version = str(version)
    try:
        mv = client.get_model_version(reg["model_name"], version)
    except Exception as exc:
        raise RegistryError(f"{reg['model_name']} version {version} not found: {exc}") from exc
    problems = promotion_problems(mv.tags)
    if problems:
        raise RegistryError("Not promoted:\n  - " + "\n  - ".join(problems))
    aliases = client.get_registered_model(reg["model_name"]).aliases or {}
    previous = aliases.get(reg["champion_alias"])
    previous = None if previous is None else str(previous)  # MLflow versions differ: int or str
    client.set_registered_model_alias(reg["model_name"], reg["champion_alias"], version)
    client.set_model_version_tag(reg["model_name"], version, "replaced_champion", str(previous))
    return {"name": reg["model_name"], "alias": reg["champion_alias"], "version": version, "previous": previous}


def list_versions(run_cfg: dict) -> dict:
    _, reg, client = _setup(run_cfg)
    versions = client.search_model_versions(f"name='{reg['model_name']}'")
    if not versions:
        return {"name": reg["model_name"], "aliases": {}, "versions": []}
    aliases = client.get_registered_model(reg["model_name"]).aliases or {}
    rows = []
    for v in sorted(versions, key=lambda x: int(x.version)):
        t = v.tags or {}
        rows.append({"version": str(v.version),
                     "aliases": sorted(a for a, ver in aliases.items() if str(ver) == str(v.version)),
                     "test_accuracy": t.get("test_accuracy"), "test_macro_f1": t.get("test_macro_f1"),
                     "quality_gate": t.get("quality_gate"), "source_checkpoint": t.get("source_checkpoint")})
    return {"name": reg["model_name"], "aliases": aliases, "versions": rows}


def load_champion(run_cfg: dict):
    """Load whatever version the champion alias currently points at (no model file path anywhere)."""
    track, reg, _ = _setup(run_cfg)
    return mlflow.pyfunc.load_model(f"models:/{reg['model_name']}@{reg['champion_alias']}")


def export_champion(run_cfg: dict, out_dir: str) -> dict:
    """Copy the champion's packaged model to out_dir (model/ + model_info.json), e.g. for a container image."""
    import mlflow.artifacts

    _, reg, client = _setup(run_cfg)
    try:
        mv = client.get_model_version_by_alias(reg["model_name"], reg["champion_alias"])
    except Exception as exc:
        raise RegistryError(f"No '{reg['champion_alias']}' version of {reg['model_name']} found: {exc}") from exc
    out = Path(out_dir)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as tmp:
        mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{reg['model_name']}/{mv.version}", dst_path=tmp)
        found = list(Path(tmp).rglob("MLmodel"))
        if not found:
            raise RegistryError("The downloaded artifacts do not contain an MLflow model.")
        shutil.copytree(found[0].parent, out / "model")
    info = {"name": reg["model_name"], "alias": reg["champion_alias"], "version": str(mv.version),
            "run_id": getattr(mv, "run_id", None), "tags": dict(mv.tags or {}),
            "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (out / "model_info.json").write_text(json.dumps(info, indent=2))
    return info


def champion_version(run_cfg: dict) -> Optional[str]:
    """Version number the champion alias points to, or None if there is no champion."""
    _, reg, client = _setup(run_cfg)
    try:
        return str(client.get_model_version_by_alias(reg["model_name"], reg["champion_alias"]).version)
    except Exception:
        return None


def tag_version(run_cfg: dict, version: str, tags: dict) -> None:
    _, reg, client = _setup(run_cfg)
    for key, value in tags.items():
        client.set_model_version_tag(reg["model_name"], str(version), key, str(value))
