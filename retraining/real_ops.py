"""The real steps of the workflow: training, evaluation, registry (needs torch and MLflow)."""
import json
from pathlib import Path

import numpy as np

from data.dataloader import get_loaders
from evaluation.evaluate import log_evaluation, run_evaluation
from evaluation.gate import evaluate_gate, load_gate_config
from evaluation.metrics import compute_metrics
from registry import ops as registry_ops
from retraining.new_data import ingest
from tracking.tracker import RunTracker, tracking_cfg
from training.run_config import apply_overrides, load_run_config
from training.train import run_training


class RealOps:
    def __init__(self, config_path=None, gate_config="configs/quality_gate.yaml"):
        self.config_path = config_path
        self.run_cfg = load_run_config(config_path)
        apply_overrides(self.run_cfg["config_overrides"])
        self.gate_cfg = load_gate_config(gate_config)
        self.track_cfg = {**tracking_cfg(self.run_cfg), "enabled": True}

    def champion_version(self):
        return registry_ops.champion_version(self.run_cfg)

    def ingest(self, new_data, name):
        from config import CONFIG
        return ingest(new_data, CONFIG["data_dir"], str(Path("datasets") / name), self.run_cfg["validation"])

    def train(self, name, overrides):
        return run_training(self.config_path, report_dir=str(Path("reports") / "retraining" / name),
                            run_name=name, overrides=overrides)

    def evaluate(self, model_path, run_id):
        out_dir = Path("reports") / "evaluation" / Path(model_path).stem
        metrics = run_evaluation(model_path, str(out_dir))
        reference = json.loads(Path(self.gate_cfg["reference_metrics"]).read_text())
        gate = evaluate_gate(metrics, reference, self.gate_cfg)
        gate["reference_model_sha256"] = reference.get("model_sha256")
        (out_dir / "gate.json").write_text(json.dumps(gate, indent=2))
        with RunTracker(self.track_cfg, resume_run_id=run_id) as tracker:
            log_evaluation(tracker, metrics, gate, out_dir)
        predictions = json.loads((out_dir / "predictions.json").read_text())
        return metrics, gate, predictions["y_true"], predictions["y_pred"]

    def register(self, model_path, run_id):
        return registry_ops.register(model_path, self.run_cfg, run_id)

    def champion_predictions(self):
        """Run the current champion over the test split (through the registry alias)."""
        champion = registry_ops.load_champion(self.run_cfg)
        _, _, test_loader, classes = get_loaders()
        if list(champion.unwrap_python_model().classes) != list(classes):
            raise RuntimeError("The champion's classes differ from the dataset's classes.")
        y_true, y_pred = [], []
        for images, labels in test_loader:
            y_pred.extend(np.argmax(champion.predict(images.numpy()), axis=1).tolist())
            y_true.extend(labels.tolist())
        return y_true, y_pred, compute_metrics(y_true, y_pred, classes)

    def promote(self, version):
        return registry_ops.promote(version, self.run_cfg)

    def record(self, result, decision_path, run_id, version):
        """Write the outcome onto the registered version and the MLflow run."""
        tags = {"promotion_decision": result["outcome"], "promotion_reason": "; ".join(result["reasons"]),
                "compared_to_champion": str(result["champion_version"])}
        decision = result.get("decision")
        if decision:
            tags["mcnemar_p"] = f"{decision['mcnemar_p']:.4f}"
        if version is not None:
            registry_ops.tag_version(self.run_cfg, version, tags)
        if run_id:
            with RunTracker(self.track_cfg, resume_run_id=run_id) as tracker:
                tracker.set_tags(tags)
                if decision:
                    tracker.log_metrics({"champion_accuracy": decision["champion_accuracy"],
                                         "accuracy_gain_vs_champion": decision["accuracy_gain"],
                                         "mcnemar_p": decision["mcnemar_p"]})
                tracker.log_artifact(decision_path, "retraining")
