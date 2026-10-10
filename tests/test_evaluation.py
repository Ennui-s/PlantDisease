"""Unit tests for metrics and the quality gate."""
import pytest

from evaluation.gate import evaluate_gate, load_gate_config
from evaluation.metrics import compute_metrics

GATE = {"max_accuracy_drop": 0.01, "max_macro_f1_drop": 0.01, "max_class_recall_drop": 0.10}


def test_metrics_match_hand_computation():
    y_true = [0, 0, 0, 0, 1, 1, 2, 2, 2, 2]
    y_pred = [0, 0, 0, 1, 1, 1, 2, 2, 0, 2]
    m = compute_metrics(y_true, y_pred, ["a", "b", "c"])
    assert m["accuracy"] == pytest.approx(0.8)
    assert m["per_class"]["a"]["recall"] == 0.75 and m["per_class"]["a"]["precision"] == pytest.approx(0.75)
    assert m["per_class"]["b"]["recall"] == 1.0 and m["per_class"]["b"]["precision"] == pytest.approx(2 / 3)
    assert m["per_class"]["c"]["support"] == 4
    assert m["confusion_matrix"] == [[3, 1, 0], [0, 2, 0], [1, 0, 3]]
    assert m["macro_f1"] == pytest.approx((0.75 + 0.8 + 6 / 7) / 3)


def test_metrics_reject_bad_input():
    with pytest.raises(ValueError):
        compute_metrics([], [], ["a"])
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0], ["a", "b"])


def metrics(acc, f1, recalls):
    return {"classes": ["x", "y"], "n_images": 2073, "accuracy": acc, "macro_f1": f1,
            "per_class": {c: {"recall": r, "support": s} for c, r, s in recalls}}


REF = metrics(0.952, 0.95, [("x", 0.95, 2000), ("y", 0.95, 16)])


@pytest.mark.parametrize("candidate, expected", [
    (metrics(0.952, 0.95, [("x", 0.95, 2000), ("y", 0.95, 16)]), True),         # identical
    (metrics(0.947, 0.945, [("x", 0.95, 2000), ("y", 0.95, 16)]), True),        # within noise
    (metrics(0.932, 0.95, [("x", 0.95, 2000), ("y", 0.95, 16)]), False),        # accuracy -2 pp
    (metrics(0.952, 0.93, [("x", 0.95, 2000), ("y", 0.95, 16)]), False),        # macro F1 -2 pp
    (metrics(0.952, 0.95, [("x", 0.95, 2000), ("y", 0.8875, 16)]), True),       # one image lost in small class
    (metrics(0.952, 0.95, [("x", 0.95, 2000), ("y", 0.65, 16)]), False),        # small class collapses
    (metrics(0.97, 0.97, [("x", 0.97, 2000), ("y", 1.0, 16)]), True),           # better than the reference
])
def test_gate_decisions(candidate, expected):
    assert evaluate_gate(candidate, REF, GATE)["passed"] is expected


def test_gate_fails_on_different_class_set_and_names_failing_class():
    other = metrics(0.952, 0.95, [("x", 0.95, 2000), ("y", 0.95, 16)])
    other["classes"] = ["x", "z"]
    assert not evaluate_gate(other, REF, GATE)["passed"]
    result = evaluate_gate(metrics(0.952, 0.95, [("x", 0.95, 2000), ("y", 0.5, 16)]), REF, GATE)
    failing = [c for c in result["checks"] if c["name"] == "per_class_recall"][0]["failing_classes"]
    assert list(failing) == ["y"]


def test_gate_config_validation(tmp_path):
    good = tmp_path / "g.yaml"
    good.write_text("reference_metrics: r.json\nmax_accuracy_drop: 0.01\nmax_macro_f1_drop: 0.01\nmax_class_recall_drop: 0.1\n")
    assert load_gate_config(str(good))["max_accuracy_drop"] == 0.01
    bad = tmp_path / "b.yaml"
    bad.write_text("max_accuracy_drop: 0.01\nbogus: 1\n")
    with pytest.raises(ValueError):
        load_gate_config(str(bad))
    with pytest.raises(FileNotFoundError):
        load_gate_config(str(tmp_path / "missing.yaml"))
