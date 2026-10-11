"""Training entry point.

    python -m training.train                              # config.py defaults (+ configs/training.yaml)
    python -m training.train --config configs/experiments/lr_0005.yaml
    python -m training.train --run-name my-run --no-tracking --skip-validation

The training logic below is the original classifier training loop, unchanged. It has been moved
into run_training() so the platform can call it, and wrapped with (1) a data validation gate that
runs before any training, (2) MLflow experiment tracking and (3) a JSON record of what was run.
"""
import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from config import CONFIG
from data.dataloader import get_loaders
from models.model import PlantDiseaseModel
from tracking.tracker import RunTracker, git_info, tracking_cfg
from training.prepare import prepare_run
from utils.checkpoint import save_checkpoint
from utils.early_stopping import EarlyStopping
from validation.validate_data import DataValidationError


def run_training(config_path=None, skip_validation=False, report_dir="reports",
                 run_name=None, no_tracking=False, overrides=None) -> dict:
    """Validate the data, train the classifier, track the run and return a summary."""
    started_at = datetime.now(UTC)
    prepared = prepare_run(config_path, skip_validation, report_dir, overrides)
    run_cfg = prepared["run_cfg"]
    track_cfg = tracking_cfg(run_cfg)
    if no_tracking:
        track_cfg["enabled"] = False

    if run_cfg["seed"] is not None:
        torch.manual_seed(run_cfg["seed"])
        torch.cuda.manual_seed_all(run_cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # torch.save does not create folders, so make sure the checkpoint folder exists
    Path(CONFIG["model_path"]).parent.mkdir(parents=True, exist_ok=True)

    # Load data
    train_loader, val_loader, test_loader, classes = get_loaders()

    # Model initialization
    model = PlantDiseaseModel(num_classes=len(classes)).to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["learning_rate"]
    )

    # Tensorboard writer
    writer = SummaryWriter(CONFIG["log_dir"])

    # Early Stopping
    early_stopper = EarlyStopping(CONFIG["patience"])
    best_acc = 0
    best_epoch = None
    epochs_run = 0
    last_val_acc = last_val_loss = None

    commit, dirty = git_info()
    validation = prepared["validation"]

    with RunTracker(track_cfg, run_name) as tracker:
        # What this run is: configuration, data identity and code version
        tracker.log_params({
            **CONFIG,
            "seed": run_cfg["seed"],
            "num_classes": len(classes),
            "n_train_images": len(train_loader.dataset),
            "n_val_images": len(val_loader.dataset),
            "n_test_images": len(test_loader.dataset),
        })
        tracker.set_tags({
            "dataset_fingerprint": validation.get("dataset_fingerprint"),
            "data_validation_passed": str(validation["passed"]) if validation["ran"] else "skipped",
            "git_commit": commit,
            "git_uncommitted_changes": str(dirty) if dirty is not None else None,
            "device": str(device),
            "torch_version": torch.__version__,
            "config_file": run_cfg["path"],
        })

        # Training loop
        for epoch in range(CONFIG["epochs"]):

            model.train()
            train_loss = 0

            # training step
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # validation step
            model.eval()
            val_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

            val_acc = correct / total
            epochs_run = epoch + 1
            last_val_acc, last_val_loss = val_acc, val_loss

            # Tensorboard logging
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)

            # MLflow logging (losses are summed over batches, as in the TensorBoard curves)
            tracker.log_metrics({"train_loss_sum": train_loss,
                                 "val_loss_sum": val_loss,
                                 "val_accuracy": val_acc}, step=epoch)

            print(f"Epoch {epoch}: Val Acc {val_acc:.4f}")

            # checkpointing
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                save_checkpoint(model, optimizer, epoch, CONFIG["model_path"])

            # Early Stopping
            early_stopper.step(val_loss)
            if early_stopper.stop:
                print("Early stopping triggered")
                break

        writer.close()

        summary = {
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "config_file": run_cfg["path"],
            "effective_config": dict(CONFIG),
            "seed": run_cfg["seed"],
            "device": str(device),
            "classes": classes,
            "epochs_run": epochs_run,
            "stopped_early": bool(early_stopper.stop),
            "best_val_acc": best_acc,
            "best_epoch": best_epoch,
            "last_val_acc": last_val_acc,
            "last_val_loss": last_val_loss,
            "model_path": CONFIG["model_path"],
            "data_validation": validation,
            "tracking": {
                "enabled": tracker.enabled,
                "run_id": tracker.run_id,
                "experiment": track_cfg["experiment_name"] if tracker.enabled else None,
                "tracking_uri": track_cfg["tracking_uri"] if tracker.enabled else None,
            },
        }
        out = Path(report_dir) / "training_run.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        print(f"Run summary written to {out}")

        # Final results and artifacts
        tracker.log_metrics({"best_val_accuracy": best_acc,
                             "best_epoch": best_epoch if best_epoch is not None else -1,
                             "epochs_run": epochs_run,
                             "stopped_early": float(early_stopper.stop)})
        tracker.log_artifact(CONFIG["model_path"], "checkpoint")
        tracker.log_artifact(validation.get("report_path"), "reports")
        tracker.log_artifact(out, "reports")
        tracker.log_artifact(run_cfg["path"], "config")
        if tracker.enabled:
            print(f"MLflow run {tracker.run_id} logged to experiment '{track_cfg['experiment_name']}'")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train the plant disease classifier.")
    parser.add_argument("--config", default=None, help="run config YAML (default: configs/training.yaml if present)")
    parser.add_argument("--run-name", default=None, help="name for the MLflow run")
    parser.add_argument("--no-tracking", action="store_true", help="do not log to MLflow")
    parser.add_argument("--skip-validation", action="store_true", help="do not run the data validation gate")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args(argv)
    try:
        run_training(args.config, args.skip_validation, args.report_dir, args.run_name, args.no_tracking)
    except DataValidationError as exc:
        print(f"\n{exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
