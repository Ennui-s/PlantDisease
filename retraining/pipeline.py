"""The retraining workflow as a sequence of steps. The steps themselves (training, evaluation, registry)
are provided by an `ops` object, so this logic can be tested without torch or MLflow.

    new data (optional) -> validate -> train -> evaluate -> quality gate -> register -> compare with the
    champion -> promote or reject
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from retraining.compare import decide


class PipelineError(RuntimeError):
    """The workflow cannot continue (not a model-quality outcome)."""


def run_pipeline(name: str, ops, policy: dict, new_data: Optional[str] = None,
                 say: Callable[[str], None] = print) -> dict:
    out_dir = Path("reports") / "retraining" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    champion = ops.champion_version()
    if champion is None:
        raise PipelineError("There is no champion yet. Register and promote a first model before retraining "
                            "(python -m registry register --model best_model.pth).")
    say(f"[1/6] Current champion: version {champion}")

    overrides = {"model_path": f"checkpoints/{name}.pth", "log_dir": f"logs/{name}"}
    ingest_report = None
    if new_data:
        ingest_report = ops.ingest(new_data, name)
        (out_dir / "new_data_report.json").write_text(json.dumps(ingest_report, indent=2))
        overrides["data_dir"] = ingest_report["derived_dir"]
        say(f"      New data: {ingest_report['n_accepted']} of {ingest_report['n_found']} images accepted "
            f"into the training split ({sum(v['count'] for v in ingest_report['excluded'].values())} excluded)")
    else:
        say("      No new data given: retraining the same recipe on the existing training split.")

    say("[2/6] Validating the data and training the candidate ...")
    summary = ops.train(name, overrides)
    run_id = (summary.get("tracking") or {}).get("run_id")
    model_path = summary["model_path"]

    say("[3/6] Evaluating the candidate on the test split and applying the quality gate ...")
    metrics, gate, y_true, y_pred = ops.evaluate(model_path, run_id)

    result = {"name": name, "champion_version": champion, "model_path": model_path, "run_id": run_id,
              "new_data": ingest_report, "candidate_test_accuracy": metrics["accuracy"],
              "quality_gate_passed": gate["passed"]}
    if not gate["passed"]:
        result.update({"outcome": "rejected_by_quality_gate", "version": None,
                       "reasons": ["the candidate failed the quality gate against the reference model"]})
        say("      Quality gate FAILED: the candidate is not registered.")
        return _finish(result, out_dir, ops, run_id, None)

    say("[4/6] Registering the candidate ...")
    registered = ops.register(model_path, run_id)
    version = registered["version"]
    result["version"] = version
    say(f"      Registered as version {version}.")

    say(f"[5/6] Comparing with the champion (version {champion}) on the same test images ...")
    champ_true, champ_pred, champ_metrics = ops.champion_predictions()
    if list(champ_true) != list(y_true):
        raise PipelineError("The champion and the candidate were not evaluated on the same test images.")
    decision = decide(champ_metrics, metrics, y_true, champ_pred, y_pred, policy)
    result.update({"decision": decision, "reasons": decision["reasons"]})

    if decision["promote"]:
        say(f"[6/6] Promoting version {version} to champion ...")
        ops.promote(version)
        result["outcome"] = "promoted"
    else:
        say(f"[6/6] Not promoted; version {champion} stays champion.")
        result["outcome"] = "rejected"
    return _finish(result, out_dir, ops, run_id, version)


def _finish(result: dict, out_dir: Path, ops, run_id, version) -> dict:
    path = out_dir / "decision.json"
    path.write_text(json.dumps(result, indent=2))
    ops.record(result, str(path), run_id, version)
    return result
