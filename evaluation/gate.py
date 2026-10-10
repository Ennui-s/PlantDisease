"""Quality gate: decides whether a trained candidate is allowed to move on to registration.

The candidate's evaluation metrics are compared with the reference metrics of the existing
classifier (the model the project already had). A candidate passes only if it is not worse than
the reference by more than the configured margins. Margins are justified in configs/quality_gate.yaml.
This module is pure Python so it can be tested and reused without torch.
"""
from __future__ import annotations

import math
from pathlib import Path

import yaml

GATE_KEYS = {"reference_metrics", "max_accuracy_drop", "max_macro_f1_drop", "max_class_recall_drop"}
EPS = 1e-12


def load_gate_config(path: str) -> dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Quality gate config not found: {p}")
    cfg = yaml.safe_load(p.read_text()) or {}
    missing, extra = GATE_KEYS - set(cfg), set(cfg) - GATE_KEYS
    if missing or extra:
        raise ValueError(f"{p}: missing keys {sorted(missing)}, unknown keys {sorted(extra)}")
    return cfg


def accuracy_noise_halfwidth(accuracy: float, n: int) -> float:
    """95% half-width of the sampling noise on an accuracy measured on n images."""
    return 1.96 * math.sqrt(accuracy * (1 - accuracy) / n)


def evaluate_gate(candidate: dict, reference: dict, cfg: dict) -> dict:
    """Return {"passed": bool, "checks": [...], "context": {...}}."""
    checks = []

    same_classes = candidate["classes"] == reference["classes"]
    checks.append({"name": "class_set", "passed": same_classes,
                   "detail": "candidate and reference predict the same classes" if same_classes
                   else "candidate and reference have different classes; metrics are not comparable"})

    for name, key, limit in (("accuracy", "accuracy", cfg["max_accuracy_drop"]),
                             ("macro_f1", "macro_f1", cfg["max_macro_f1_drop"])):
        drop = reference[key] - candidate[key]
        checks.append({"name": name, "passed": drop <= limit + EPS,
                       "candidate": candidate[key], "reference": reference[key],
                       "drop": drop, "allowed_drop": limit})

    failing = {}
    if same_classes:
        for cls, ref in reference["per_class"].items():
            drop = ref["recall"] - candidate["per_class"][cls]["recall"]
            if drop > cfg["max_class_recall_drop"] + EPS:
                failing[cls] = {"reference_recall": ref["recall"],
                                "candidate_recall": candidate["per_class"][cls]["recall"],
                                "drop": drop, "support": ref["support"]}
    checks.append({"name": "per_class_recall", "passed": same_classes and not failing,
                   "allowed_drop": cfg["max_class_recall_drop"], "failing_classes": failing})

    return {
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "context": {
            "n_images": candidate["n_images"],
            "reference_accuracy_noise_halfwidth": accuracy_noise_halfwidth(
                reference["accuracy"], reference["n_images"]),
        },
    }


def format_gate(result: dict) -> str:
    lines = [f"Quality gate: {'PASSED' if result['passed'] else 'FAILED'}"]
    for c in result["checks"]:
        mark = "ok  " if c["passed"] else "FAIL"
        if c["name"] in ("accuracy", "macro_f1"):
            lines.append(f"  [{mark}] {c['name']}: candidate {c['candidate']:.4f} vs reference "
                         f"{c['reference']:.4f} (drop {c['drop']:+.4f}, allowed {c['allowed_drop']:.4f})")
        elif c["name"] == "per_class_recall":
            lines.append(f"  [{mark}] per-class recall: allowed drop {c['allowed_drop']:.2f} per class")
            for cls, f in c["failing_classes"].items():
                lines.append(f"           {cls}: {f['reference_recall']:.3f} -> {f['candidate_recall']:.3f} "
                             f"(support {f['support']})")
        else:
            lines.append(f"  [{mark}] {c['name']}: {c['detail']}")
    return "\n".join(lines)
