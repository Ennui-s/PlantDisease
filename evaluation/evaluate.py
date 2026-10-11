"""Evaluate a trained checkpoint on the test split and (optionally) apply the quality gate.

    python -m evaluation.evaluate --model best_model.pth --write-reference   # once, for the original model
    python -m evaluation.evaluate --model checkpoints/lr_0005.pth            # evaluate + quality gate

Produces, under reports/evaluation/<model name>/: metrics.json, confusion_matrix.png, gradcam_<n>.png
and gate.json. Results are logged to MLflow: to the training run that produced the checkpoint if it
can be identified (or --run-id), otherwise to a new run. Exit code 1 means the quality gate failed.

training/evaluate.py (the original script) is left as it was; this module reuses its building blocks
(utils/visualization.py) and wraps them so the platform can call evaluation and get metrics back.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from data.dataloader import get_loaders
from models.model import PlantDiseaseModel
from utils.visualization import generate_confusion_matrix, save_gradcam_image
from evaluation.gate import evaluate_gate, format_gate, load_gate_config
from evaluation.metrics import compute_metrics
from tracking.tracker import RunTracker, tracking_cfg, training_run_id
from training.run_config import apply_overrides, load_run_config


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_model(model_path, num_classes, device):
    model = PlantDiseaseModel(num_classes).to(device)
    # map_location lets a checkpoint saved on a GPU be loaded on a CPU-only machine
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def _collect(model, loader, device, n_samples):
    """Run the model over the loader. Returns true labels, predictions and a few images for Grad-CAM."""
    y_true, y_pred, samples = [], [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().tolist())
            y_true.extend(labels.tolist())
            if len(samples) < n_samples:  # same selection as the original script: first image of the first batches
                samples.append(images[0].cpu())
    return y_true, y_pred, samples


def run_evaluation(model_path, output_dir, n_gradcam=3) -> dict:
    """Evaluate the checkpoint on the test split, write artifacts to output_dir and return the metrics."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, classes = get_loaders()
    model = _load_model(model_path, len(classes), device)
    y_true, y_pred, samples = _collect(model, test_loader, device, n_gradcam)

    metrics = compute_metrics(y_true, y_pred, classes)
    metrics.update({
        "split": "test",
        "model_path": str(model_path),
        "model_sha256": _sha256(model_path),
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": str(device),
    })
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "predictions.json").write_text(json.dumps({"y_true": y_true, "y_pred": y_pred}))  # for paired comparisons

    generate_confusion_matrix(model, test_loader, classes, device, str(out / "confusion_matrix.png"))
    for i, img_tensor in enumerate(samples):
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
        save_gradcam_image(model, img_tensor, img_np, device, classes, str(out / f"gradcam_{i}.png"))
    return metrics


def _log_to_mlflow(tracker, metrics, gate, out_dir):
    logged = {f"test_{k}": metrics[k] for k in
              ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")}
    for cls, m in metrics["per_class"].items():
        logged[f"test_recall/{cls}"] = m["recall"]
        logged[f"test_f1/{cls}"] = m["f1"]
    tracker.log_metrics(logged)
    tracker.set_tags({"model_sha256": metrics["model_sha256"],
                      "quality_gate": None if gate is None else ("passed" if gate["passed"] else "failed")})
    for f in sorted(Path(out_dir).glob("*")):
        tracker.log_artifact(f, "evaluation")


log_evaluation = _log_to_mlflow   # used by the retraining workflow


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint and apply the quality gate.")
    parser.add_argument("--model", required=True, help="checkpoint (.pth) to evaluate")
    parser.add_argument("--config", default=None, help="run config YAML (default: configs/training.yaml)")
    parser.add_argument("--gate-config", default="configs/quality_gate.yaml")
    parser.add_argument("--output-dir", default=None, help="default: reports/evaluation/<model name>")
    parser.add_argument("--gradcam", type=int, default=3, help="number of Grad-CAM images (default 3, as before)")
    parser.add_argument("--write-reference", action="store_true",
                        help="store these metrics as the reference for the quality gate (original model)")
    parser.add_argument("--no-gate", action="store_true")
    parser.add_argument("--run-id", default=None, help="MLflow run to attach results to")
    parser.add_argument("--no-tracking", action="store_true")
    args = parser.parse_args(argv)

    run_cfg = load_run_config(args.config)
    apply_overrides(run_cfg["config_overrides"])
    gate_cfg = load_gate_config(args.gate_config)
    out_dir = args.output_dir or str(Path("reports") / "evaluation" / Path(args.model).stem)

    metrics = run_evaluation(args.model, out_dir, args.gradcam)
    print(f"Test accuracy: {metrics['accuracy']:.4f}  macro F1: {metrics['macro_f1']:.4f}  "
          f"({metrics['n_images']} images)")
    weakest = sorted(metrics["per_class"].items(), key=lambda kv: kv[1]["recall"])[:3]
    print("Lowest recall: " + ", ".join(f"{c} {m['recall']:.3f}" for c, m in weakest))
    print(f"Artifacts written to {out_dir}")

    gate = None
    if args.write_reference:
        ref_path = Path(gate_cfg["reference_metrics"])
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        ref_path.write_text(json.dumps(metrics, indent=2))
        print(f"Reference metrics written to {ref_path}")
    elif not args.no_gate:
        ref_path = Path(gate_cfg["reference_metrics"])
        if not ref_path.is_file():
            print(f"No reference metrics at {ref_path}. Create them from the original model with "
                  "--write-reference, or skip the gate with --no-gate.")
            return 2
        reference = json.loads(ref_path.read_text())
        gate = evaluate_gate(metrics, reference, gate_cfg)
        gate["reference_model_sha256"] = reference.get("model_sha256")
        (Path(out_dir) / "gate.json").write_text(json.dumps(gate, indent=2))
        print(format_gate(gate))

    track_cfg = tracking_cfg(run_cfg)
    if args.no_tracking:
        track_cfg["enabled"] = False
    run_id = args.run_id or training_run_id(args.model)
    name = None if run_id else f"evaluate-{Path(args.model).stem}"
    with RunTracker(track_cfg, run_name=name, resume_run_id=run_id) as tracker:
        _log_to_mlflow(tracker, metrics, gate, out_dir)
        if tracker.enabled:
            print(f"Logged to MLflow run {tracker.run_id}")
    return 0 if gate is None or gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
