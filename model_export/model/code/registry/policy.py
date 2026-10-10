"""Registry rules that do not need MLflow or torch: what may be registered or promoted, and what
metadata a registered version carries. Kept pure so they can be tested on their own."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

REGISTRY_DEFAULTS = {
    "model_name": "plant-disease-classifier",
    "champion_alias": "champion",
}


def registry_cfg(run_cfg: dict) -> dict:
    extra = set(run_cfg["registry"]) - set(REGISTRY_DEFAULTS)
    if extra:
        raise ValueError(f"Unknown registry setting(s): {sorted(extra)}")
    return {**REGISTRY_DEFAULTS, **run_cfg["registry"]}


def check_registration_preconditions(model_path: str, eval_dir: str, sha256_fn: Callable[[str], str]) -> dict:
    """A checkpoint may be registered only if it was evaluated on the test split and passed the quality gate.

    The evaluation must be of exactly this file (matching hash), otherwise a different or retrained
    checkpoint could be registered on the strength of an old evaluation.
    """
    problems, metrics, gate = [], None, None
    eval_dir = Path(eval_dir)
    if not Path(model_path).is_file():
        problems.append(f"Checkpoint not found: {model_path}")
    for name in ("metrics.json", "gate.json"):
        if not (eval_dir / name).is_file():
            problems.append(f"{eval_dir / name} not found. Evaluate the checkpoint first: "
                            f"python -m evaluation.evaluate --model {model_path}")
    if not problems:
        metrics = json.loads((eval_dir / "metrics.json").read_text())
        gate = json.loads((eval_dir / "gate.json").read_text())
        if metrics.get("split") != "test":
            problems.append(f"The evaluation was not on the test split (split={metrics.get('split')!r}).")
        if metrics.get("model_sha256") != sha256_fn(model_path):
            problems.append("The evaluation in " + str(eval_dir) + " is of a different file than "
                            f"{model_path} (hash mismatch). Re-run the evaluation.")
        if not gate.get("passed"):
            problems.append("The quality gate did not pass; failed candidates are not registered.")
    return {"ok": not problems, "problems": problems, "metrics": metrics, "gate": gate}


def find_version_by_sha(versions, sha256: str):
    """Existing registered version built from the same checkpoint file, if any."""
    for v in versions:
        if (v.tags or {}).get("model_sha256") == sha256:
            return v
    return None


def version_tags(metrics: dict, gate: dict, run_tags: Optional[dict], checkpoint_path: str,
                 run_id: Optional[str]) -> dict:
    """Metadata stored on a registered model version (lineage, results and gate outcome)."""
    run_tags = run_tags or {}
    tags = {
        "model_sha256": metrics["model_sha256"],
        "source_checkpoint": str(checkpoint_path),
        "training_run_id": run_id,
        "quality_gate": "passed" if gate.get("passed") else "failed",
        "reference_model_sha256": gate.get("reference_model_sha256"),
        "test_accuracy": f"{metrics['accuracy']:.4f}",
        "test_macro_f1": f"{metrics['macro_f1']:.4f}",
        "test_n_images": str(metrics["n_images"]),
        "num_classes": str(len(metrics["classes"])),
        "dataset_fingerprint": run_tags.get("dataset_fingerprint"),
        "git_commit": run_tags.get("git_commit"),
    }
    return {k: str(v) for k, v in tags.items() if v is not None}


def promotion_problems(version_tags_: dict) -> list:
    problems = []
    if (version_tags_ or {}).get("quality_gate") != "passed":
        problems.append("This version is not tagged quality_gate=passed.")
    return problems
