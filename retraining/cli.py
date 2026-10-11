"""Retraining command.

    python -m retraining                                   # retrain the same recipe, compare with the champion
    python -m retraining --new-data path/to/new_images     # add new labelled images to the training split
    python -m retraining --config configs/experiments/lr_0005.yaml --name retrain-lr
"""
import argparse
from datetime import datetime, timezone

from retraining.compare import load_policy
from retraining.new_data import NewDataError
from retraining.pipeline import PipelineError, run_pipeline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m retraining", description="Retrain, evaluate, gate, register, compare and promote or reject.")
    parser.add_argument("--name", default=datetime.now(timezone.utc).strftime("retrain-%Y%m%d-%H%M%S"),
                        help="name of the candidate (checkpoint, MLflow run, reports); default: retrain-<UTC time>")
    parser.add_argument("--config", default=None, help="run config YAML for the training recipe (default: configs/training.yaml)")
    parser.add_argument("--new-data", default=None, help="folder of new labelled images, one sub-folder per class")
    parser.add_argument("--promotion-config", default="configs/promotion.yaml")
    args = parser.parse_args(argv)

    from retraining.real_ops import RealOps   # imports torch and MLflow, so only when actually running
    from validation.validate_data import DataValidationError
    try:
        result = run_pipeline(args.name, RealOps(args.config), load_policy(args.promotion_config), args.new_data)
    except (PipelineError, NewDataError, DataValidationError) as exc:
        print(f"\nStopped: {exc}")
        return 2
    print(f"\nOutcome: {result['outcome'].upper()}  (candidate version {result.get('version')}, "
          f"champion was version {result['champion_version']})")
    for reason in result["reasons"]:
        print(f"  - {reason}")
    print(f"Details: reports/retraining/{args.name}/decision.json")
    if result["outcome"] == "promoted":
        print("To ship it in the Docker image: python -m registry export, then rebuild the image.")
    return 0
