"""Unit tests for the retraining workflow: paired comparison, promotion rules, new-data ingestion, pipeline flow."""
import json
import os

import pytest

from evaluation.metrics import compute_metrics
from retraining.compare import decide, load_policy, mcnemar_exact, paired_counts
from retraining.new_data import NewDataError, ingest
from retraining.pipeline import PipelineError, run_pipeline
from tests.helpers import image_bytes, make_dataset

POLICY = {"alpha": 0.05, "min_accuracy_gain": 0.0, "max_class_recall_drop": 0.10}
CLASSES = ["a", "b"]


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(3, 3) == 1.0
    assert mcnemar_exact(0, 10) == pytest.approx(2 * 0.5 ** 10)            # 0.00195
    assert mcnemar_exact(2, 9) == pytest.approx(2 * (1 + 11 + 55) / 2048)   # 0.0654
    assert mcnemar_exact(9, 2) == mcnemar_exact(2, 9)                       # symmetric


def test_paired_counts():
    y = [0, 0, 1, 1, 1]
    champion = [0, 1, 1, 0, 1]
    candidate = [0, 0, 0, 1, 1]
    assert paired_counts(y, champion, candidate) == {"champion_only_correct": 1, "candidate_only_correct": 2}
    with pytest.raises(ValueError):
        paired_counts([0], [0, 1], [0])


def make_case(n_both_right, n_champion_only, n_candidate_only, n_both_wrong):
    """y_true / champion / candidate predictions for two balanced classes with the given agreement pattern."""
    y, champ, cand = [], [], []
    for count, champ_ok, cand_ok in ((n_both_right, True, True), (n_champion_only, True, False),
                                     (n_candidate_only, False, True), (n_both_wrong, False, False)):
        for i in range(count):
            label = i % 2
            y.append(label)
            champ.append(label if champ_ok else 1 - label)
            cand.append(label if cand_ok else 1 - label)
    return y, champ, cand


def decision_for(*pattern, policy=POLICY):
    y, champ, cand = make_case(*pattern)
    return decide(compute_metrics(y, champ, CLASSES), compute_metrics(y, cand, CLASSES), y, champ, cand, policy)


def test_clear_improvement_is_promoted():
    d = decision_for(900, 2, 30, 68)               # fixes 30, breaks 2
    assert d["promote"] and d["mcnemar_p"] < 0.001 and d["accuracy_gain"] > 0
    assert d["reasons"] == ["all promotion rules satisfied"]


def test_small_improvement_is_not_statistically_significant():
    d = decision_for(900, 5, 9, 86)                # +4 net images out of 1000: looks better, but is noise
    assert not d["promote"] and d["accuracy_gain"] > 0
    assert any("not statistically significant" in r for r in d["reasons"])


def test_equal_or_worse_candidates_are_rejected():
    assert not decision_for(900, 10, 10, 80)["promote"]
    d = decision_for(900, 30, 2, 68)
    assert not d["promote"] and any("does not fix more images" in r for r in d["reasons"])


def test_a_class_that_gets_much_worse_blocks_promotion():
    # The candidate is clearly and significantly better overall (fixes 100, breaks 12), but loses 20 points of
    # recall on the small class 'b'. Only the per-class rule can stop it.
    y = [0] * 1500 + [1] * 60
    champ = [1] * 100 + [0] * 1400 + [1] * 60                 # 100 errors on class a, none on b
    cand = [0] * 1500 + [0] * 12 + [1] * 48                   # no errors on a, 12 of 60 wrong on b
    d = decide(compute_metrics(y, champ, CLASSES), compute_metrics(y, cand, CLASSES), y, champ, cand, POLICY)
    assert d["candidate_only_correct"] == 100 and d["champion_only_correct"] == 12 and d["mcnemar_p"] < 0.001
    assert not d["promote"] and list(d["class_recall_drops_beyond_limit"]) == ["b"]
    lenient = decide(compute_metrics(y, champ, CLASSES), compute_metrics(y, cand, CLASSES), y, champ, cand,
                     {**POLICY, "max_class_recall_drop": 0.5})
    assert lenient["promote"]                                  # with a looser class limit the same candidate passes


def test_policy_file_validation(tmp_path):
    assert load_policy("configs/promotion.yaml")["alpha"] == 0.05
    bad = tmp_path / "p.yaml"
    bad.write_text("alpha: 0.05\nbogus: 1\n")
    with pytest.raises(ValueError):
        load_policy(str(bad))
    with pytest.raises(FileNotFoundError):
        load_policy(str(tmp_path / "missing.yaml"))


# ---------------------------------------------------------------- new data ingestion

def base_dataset(tmp_path):
    base = tmp_path / "dataset"
    for split, n in (("train", 12), ("val", 4), ("test", 4)):
        make_dataset(base / split, classes={"class_a": n, "class_b": n})
    return base


def count(folder):
    return sum(1 for p in folder.rglob("*") if p.is_file())


def test_ingest_builds_a_derived_dataset_with_new_images_in_train_only(tmp_path):
    base = base_dataset(tmp_path)
    new = make_dataset(tmp_path / "new", classes={"class_a": 5, "class_b": 3})
    report = ingest(str(new), str(base), str(tmp_path / "derived"), {"workers": 2})
    d = tmp_path / "derived"
    assert report["n_accepted"] == 8 and report["accepted_per_class"] == {"class_a": 5, "class_b": 3}
    assert count(d / "train") == count(base / "train") + 8
    assert count(d / "val") == count(base / "val") and count(d / "test") == count(base / "test")
    assert count(base / "train") == 24                                      # the original dataset is untouched


def test_ingest_excludes_duplicates_conflicts_and_bad_files(tmp_path):
    base = base_dataset(tmp_path)
    new = make_dataset(tmp_path / "new", classes={"class_a": 4, "class_b": 4})
    first, second = sorted((base / "test" / "class_a").iterdir())[:2]
    (new / "class_a" / "dup_of_test.jpg").write_bytes(first.read_bytes())            # leak into the test set
    (new / "class_b" / "mislabelled.jpg").write_bytes(second.read_bytes())           # same image as an existing one, other label
    (new / "class_a" / "copy1.jpg").write_bytes((new / "class_a" / "0.jpg").read_bytes())   # duplicate in batch
    (new / "class_a" / "broken.jpg").write_bytes(b"not an image")
    (new / "class_a" / "tiny.jpg").write_bytes(image_bytes(size=(8, 8)))
    (new / "class_a" / "notes.txt").write_text("x")
    report = ingest(str(new), str(base), str(tmp_path / "derived"), {"workers": 2})
    ex = {k: v["count"] for k, v in report["excluded"].items()}
    assert ex == {"duplicate_of_existing": 1, "label_conflict_with_existing": 1, "duplicate_in_batch": 1,
                  "unreadable": 1, "too_small": 1, "not_an_image": 1}
    assert report["n_accepted"] == 8


def test_ingest_refuses_unusable_batches(tmp_path):
    base = base_dataset(tmp_path)
    unknown = make_dataset(tmp_path / "unknown", classes={"class_a": 3, "new_disease": 3})
    with pytest.raises(NewDataError, match="do not know|does not know|know"):
        ingest(str(unknown), str(base), str(tmp_path / "d1"), {"workers": 2})
    with pytest.raises(NewDataError):
        ingest(str(tmp_path / "nowhere"), str(base), str(tmp_path / "d2"), {"workers": 2})
    only_dups = tmp_path / "dups" / "class_a"
    only_dups.mkdir(parents=True)
    (only_dups / "x.jpg").write_bytes(next((base / "train" / "class_a").iterdir()).read_bytes())
    with pytest.raises(NewDataError, match="No usable new images"):
        ingest(str(tmp_path / "dups"), str(base), str(tmp_path / "d3"), {"workers": 2})
    ok = make_dataset(tmp_path / "ok", classes={"class_a": 2})
    ingest(str(ok), str(base), str(tmp_path / "d4"), {"workers": 2})
    with pytest.raises(NewDataError, match="already exists"):
        ingest(str(ok), str(base), str(tmp_path / "d4"), {"workers": 2})


# ---------------------------------------------------------------- pipeline flow with stand-in steps

class FakeOps:
    def __init__(self, patterns=(900, 2, 30, 68), champion="1", gate_passes=True):
        y, champ, cand = make_case(*patterns)
        self.y, self.champ, self.cand = y, champ, cand
        self.champion, self.gate_passes = champion, gate_passes
        self.calls, self.recorded, self.overrides = [], None, None

    def champion_version(self):
        return self.champion

    def ingest(self, new_data, name):
        self.calls.append("ingest")
        return {"derived_dir": f"datasets/{name}", "n_found": 10, "n_accepted": 9, "excluded": {}}

    def train(self, name, overrides):
        self.calls.append("train")
        self.overrides = overrides
        return {"model_path": overrides["model_path"], "tracking": {"run_id": "run-1"}}

    def evaluate(self, model_path, run_id):
        self.calls.append("evaluate")
        return compute_metrics(self.y, self.cand, CLASSES), {"passed": self.gate_passes}, self.y, self.cand

    def register(self, model_path, run_id):
        self.calls.append("register")
        return {"version": "2"}

    def champion_predictions(self):
        self.calls.append("champion_predictions")
        return self.y, self.champ, compute_metrics(self.y, self.champ, CLASSES)

    def promote(self, version):
        self.calls.append(f"promote:{version}")

    def record(self, result, path, run_id, version):
        self.calls.append("record")
        self.recorded = (result["outcome"], version)


@pytest.fixture
def in_tmp(tmp_path):
    old = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(old)


def quiet(_):
    pass


def test_pipeline_promotes_a_clearly_better_candidate(in_tmp):
    ops = FakeOps()
    result = run_pipeline("cand-1", ops, POLICY, say=quiet)
    assert result["outcome"] == "promoted" and result["version"] == "2"
    assert ops.calls == ["train", "evaluate", "register", "champion_predictions", "promote:2", "record"]
    assert ops.recorded == ("promoted", "2")
    assert ops.overrides == {"model_path": "checkpoints/cand-1.pth", "log_dir": "logs/cand-1"}
    saved = json.loads((in_tmp / "reports" / "retraining" / "cand-1" / "decision.json").read_text())
    assert saved["outcome"] == "promoted" and saved["decision"]["candidate_only_correct"] == 30


def test_pipeline_registers_but_does_not_promote_an_unconvincing_candidate(in_tmp):
    ops = FakeOps(patterns=(900, 5, 9, 86))
    result = run_pipeline("cand-2", ops, POLICY, say=quiet)
    assert result["outcome"] == "rejected" and "promote:2" not in ops.calls and "register" in ops.calls
    assert ops.recorded == ("rejected", "2")


def test_pipeline_stops_at_the_quality_gate(in_tmp):
    ops = FakeOps(gate_passes=False)
    result = run_pipeline("cand-3", ops, POLICY, say=quiet)
    assert result["outcome"] == "rejected_by_quality_gate" and result["version"] is None
    assert "register" not in ops.calls and "champion_predictions" not in ops.calls
    assert ops.recorded == ("rejected_by_quality_gate", None)


def test_pipeline_needs_a_champion_and_the_same_test_set(in_tmp):
    with pytest.raises(PipelineError, match="no champion"):
        run_pipeline("c", FakeOps(champion=None), POLICY, say=quiet)
    ops = FakeOps()
    ops.champion_predictions = lambda: (ops.y[::-1], ops.champ, compute_metrics(ops.y, ops.champ, CLASSES))
    with pytest.raises(PipelineError, match="same test images"):
        run_pipeline("c", ops, POLICY, say=quiet)


def test_new_data_is_passed_to_training_as_a_data_dir_override(in_tmp):
    ops = FakeOps()
    run_pipeline("cand-4", ops, POLICY, new_data="some/new/images", say=quiet)
    assert ops.calls[0] == "ingest" and ops.overrides["data_dir"] == "datasets/cand-4"
    assert (in_tmp / "reports" / "retraining" / "cand-4" / "new_data_report.json").is_file()
