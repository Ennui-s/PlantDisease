"""Thin wrapper around MLflow experiment tracking.

The training code talks to RunTracker only. When tracking is disabled every method is a no-op,
so the training loop is identical either way and does not import MLflow.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Self

TRACKING_DEFAULTS = {
    "enabled": True,
    # A local SQLite database (created on first use). Artifacts go under ./mlruns by default.
    "tracking_uri": "sqlite:///mlflow.db",
    "experiment_name": "plant-disease-classifier",
    "run_name": None,
}


def tracking_cfg(run_cfg: dict) -> dict:
    extra = set(run_cfg["tracking"]) - set(TRACKING_DEFAULTS)
    if extra:
        raise ValueError(f"Unknown tracking setting(s): {sorted(extra)}")
    return {**TRACKING_DEFAULTS, **run_cfg["tracking"]}


def git_info():
    """(commit hash, has uncommitted changes to tracked files) or (None, None) outside a git repo."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                timeout=5, check=True).stdout.strip()
        changes = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                 capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        return commit, bool(changes)
    except Exception:
        return None, None


def training_run_id(model_path, report_dir="reports") -> str | None:
    """MLflow run id of the training run that wrote this checkpoint, if reports/training_run.json says so."""
    p = Path(report_dir) / "training_run.json"
    if not p.is_file():
        return None
    summary = json.loads(p.read_text())
    same = Path(summary.get("model_path", "")).resolve() == Path(model_path).resolve()
    return (summary.get("tracking") or {}).get("run_id") if same else None


class RunTracker:
    """Context manager: opens an MLflow run on entry, closes it (FINISHED/FAILED/KILLED) on exit."""

    def __init__(self, cfg: dict, run_name: str | None = None, resume_run_id: str | None = None):
        """resume_run_id: attach to an existing run (e.g. add evaluation results to a training run)."""
        self.cfg = cfg
        self.enabled = bool(cfg["enabled"])
        self.run_name = run_name or cfg["run_name"]
        self.resume_run_id = resume_run_id
        self._mlflow = None
        self.run_id: str | None = None

    def __enter__(self) -> Self:
        if not self.enabled:
            return self
        try:
            import mlflow
        except ImportError as exc:
            raise ImportError(
                "Experiment tracking is enabled but mlflow is not installed. Run "
                "`pip install -r requirements.txt`, or disable it with --no-tracking / "
                "`tracking: {enabled: false}`.") from exc
        mlflow.set_tracking_uri(self.cfg["tracking_uri"])
        if self.resume_run_id:
            run = mlflow.start_run(run_id=self.resume_run_id)
        else:
            mlflow.set_experiment(self.cfg["experiment_name"])
            run = mlflow.start_run(run_name=self.run_name)
        self._mlflow, self.run_id = mlflow, run.info.run_id
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._mlflow is not None:
            status = "FINISHED" if exc_type is None else (
                "KILLED" if issubclass(exc_type, KeyboardInterrupt) else "FAILED")
            self._mlflow.end_run(status=status)
        return False  # never swallow exceptions

    def log_params(self, params: dict) -> None:
        if self._mlflow is not None:
            self._mlflow.log_params(params)

    def set_tags(self, tags: dict) -> None:
        if self._mlflow is not None:
            self._mlflow.set_tags({k: v for k, v in tags.items() if v is not None})

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        if self._mlflow is not None:
            self._mlflow.log_metrics(metrics, step=step)

    def log_artifacts(self, directory, artifact_path: str | None = None) -> None:
        if self._mlflow is not None:
            self._mlflow.log_artifacts(str(directory), artifact_path)

    def log_artifact(self, path, artifact_path: str | None = None) -> None:
        if self._mlflow is not None and path is not None and Path(path).is_file():
            self._mlflow.log_artifact(str(path), artifact_path)
