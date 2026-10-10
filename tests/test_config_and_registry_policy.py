"""Unit tests for run-config handling and the registry rules."""
import json
from types import SimpleNamespace

import pytest

from config import CONFIG
from registry.policy import (check_registration_preconditions, find_version_by_sha, promotion_problems,
                             registry_cfg, version_tags)
from training import run_config


@pytest.fixture
def restore_config():
    saved = dict(CONFIG)
    yield
    CONFIG.clear()
    CONFIG.update(saved)


def test_base_inheritance_merges_sections(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("seed: 1\nvalidation:\n  min_side: 40\n  workers: 2\nconfig_overrides:\n  epochs: 5\n")
    child = tmp_path / "child.yaml"
    child.write_text(f"base: {base.as_posix()}\nseed: 2\nvalidation:\n  workers: 8\nconfig_overrides:\n  batch_size: 16\n")
    cfg = run_config.load_run_config(str(child))
    assert cfg["seed"] == 2
    assert cfg["validation"] == {"min_side": 40, "workers": 8}
    assert cfg["config_overrides"] == {"epochs": 5, "batch_size": 16}


def test_unknown_keys_and_circular_base_are_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("oops: 1\n")
    with pytest.raises(ValueError):
        run_config.load_run_config(str(bad))
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text(f"base: {b.as_posix()}\n")
    b.write_text(f"base: {a.as_posix()}\n")
    with pytest.raises(ValueError):
        run_config.load_run_config(str(a))
    with pytest.raises(FileNotFoundError):
        run_config.load_run_config(str(tmp_path / "missing.yaml"))


def test_overrides_apply_to_shared_config_and_unknown_keys_fail(restore_config):
    run_config.apply_overrides({"epochs": 3, "learning_rate": 0.5})
    assert CONFIG["epochs"] == 3 and CONFIG["learning_rate"] == 0.5
    with pytest.raises(ValueError):
        run_config.apply_overrides({"not_a_setting": 1})


def test_repository_configs_load():
    for path in ("configs/training.yaml", "configs/smoke_test.yaml", "configs/experiments/baseline.yaml",
                 "configs/experiments/lr_0005.yaml", "configs/experiments/batch_64.yaml"):
        cfg = run_config.load_run_config(path)
        assert cfg["validation"]["accepted_cross_split_duplicate_groups"] == 5
        assert registry_cfg(cfg)["champion_alias"] == "champion"


def write_eval(tmp_path, sha="s1", split="test", passed=True):
    d = tmp_path / "eval"
    d.mkdir(exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(
        {"model_sha256": sha, "split": split, "accuracy": 0.95, "macro_f1": 0.94, "n_images": 10, "classes": ["a", "b"]}))
    (d / "gate.json").write_text(json.dumps({"passed": passed, "reference_model_sha256": "ref"}))
    ckpt = tmp_path / "m.pth"
    ckpt.write_bytes(b"x")
    return str(ckpt), str(d)


def test_registration_requires_a_passed_evaluation_of_this_exact_file(tmp_path):
    ckpt, d = write_eval(tmp_path)
    assert check_registration_preconditions(ckpt, d, lambda p: "s1")["ok"]
    assert "hash mismatch" in " ".join(check_registration_preconditions(ckpt, d, lambda p: "other")["problems"])
    ckpt, d = write_eval(tmp_path, passed=False)
    assert not check_registration_preconditions(ckpt, d, lambda p: "s1")["ok"]
    ckpt, d = write_eval(tmp_path, split="val")
    assert not check_registration_preconditions(ckpt, d, lambda p: "s1")["ok"]
    missing = check_registration_preconditions(ckpt, str(tmp_path / "nowhere"), lambda p: "s1")
    assert not missing["ok"] and "Evaluate the checkpoint first" in " ".join(missing["problems"])


def test_version_tags_and_promotion_rule():
    metrics = {"model_sha256": "s1", "accuracy": 0.95176, "macro_f1": 0.9534, "n_images": 2073, "classes": ["a"] * 15}
    tags = version_tags(metrics, {"passed": True, "reference_model_sha256": "ref"},
                        {"dataset_fingerprint": "fp"}, "m.pth", "run1")
    assert tags["quality_gate"] == "passed" and tags["test_accuracy"] == "0.9518" and tags["num_classes"] == "15"
    assert tags["dataset_fingerprint"] == "fp" and "git_commit" not in tags      # missing values are omitted
    assert promotion_problems(tags) == []
    assert promotion_problems({"quality_gate": "failed"}) and promotion_problems({})


def test_find_version_by_sha():
    versions = [SimpleNamespace(version="1", tags={"model_sha256": "a"}), SimpleNamespace(version="2", tags={"model_sha256": "b"})]
    assert find_version_by_sha(versions, "b").version == "2"
    assert find_version_by_sha(versions, "zzz") is None
