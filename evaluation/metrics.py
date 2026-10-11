"""Classification metrics computed from predictions. Pure numpy/scikit-learn, no torch."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], classes: list[str]) -> dict:
    """Accuracy, macro/weighted averages, per-class precision/recall/F1/support and the confusion matrix."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be non-empty and the same length")
    labels = list(range(len(classes)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "n_images": len(y_true),
        "classes": list(classes),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)) if support.sum() else 0.0,
        "per_class": {
            c: {"precision": float(precision[i]), "recall": float(recall[i]),
                "f1": float(f1[i]), "support": int(support[i])}
            for i, c in enumerate(classes)
        },
        "confusion_matrix": cm.tolist(),
    }
