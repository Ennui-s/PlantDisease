"""Unit tests for the data validation checks."""
import json

from tests.helpers import image_bytes, make_dataset
from validation import checks
from validation.checks import DEFAULTS, ERROR, INFO, WARNING

CFG = {**DEFAULTS, "workers": 2}


def run_checks(root, split=None):
    records, findings = checks.scan_tree(root, split, root)
    checks.inspect_images(records, CFG)
    findings += checks.check_files(records, CFG)
    return records, findings


def severities(findings, check):
    return [f.severity for f in findings if f.check == check]


def test_clean_dataset_has_no_errors_or_warnings(tmp_path):
    make_dataset(tmp_path)
    _, findings = run_checks(tmp_path)
    assert not [f for f in findings if f.severity in (ERROR, WARNING)]


def test_corrupted_truncated_and_empty_files_are_errors(tmp_path):
    make_dataset(tmp_path)
    good = image_bytes()
    (tmp_path / "class_a" / "truncated.jpg").write_bytes(good[: len(good) // 2])
    (tmp_path / "class_a" / "empty.jpg").write_bytes(b"")
    (tmp_path / "class_a" / "garbage.jpg").write_bytes(b"not an image at all")
    _, findings = run_checks(tmp_path)
    broken = [f for f in findings if f.check == "readability"]
    assert broken and broken[0].severity == ERROR
    assert "3 image file(s)" in broken[0].message


def test_disallowed_format_is_error_and_non_image_is_warning(tmp_path):
    make_dataset(tmp_path)
    (tmp_path / "class_a" / "x.bmp").write_bytes(image_bytes("BMP"))
    (tmp_path / "class_a" / "Thumbs.db").write_bytes(b"x")
    _, findings = run_checks(tmp_path)
    assert ERROR in severities(findings, "file_format")
    assert WARNING in severities(findings, "file_format")


def test_tiny_images_are_errors(tmp_path):
    make_dataset(tmp_path)
    (tmp_path / "class_a" / "tiny.jpg").write_bytes(image_bytes(size=(8, 8)))
    _, findings = run_checks(tmp_path)
    assert ERROR in severities(findings, "dimensions")


def test_structure_problems(tmp_path):
    make_dataset(tmp_path)
    (tmp_path / "empty_class").mkdir()
    (tmp_path / "stray.txt").write_text("x")
    _, findings = run_checks(tmp_path)
    assert ERROR in severities(findings, "structure")      # empty class folder
    assert WARNING in severities(findings, "structure")    # stray file
    _, missing = checks.scan_tree(tmp_path / "nope", None, tmp_path)
    assert missing[0].severity == ERROR


def test_duplicates_and_label_conflicts(tmp_path):
    make_dataset(tmp_path)
    data = (tmp_path / "class_a" / "0.jpg").read_bytes()
    (tmp_path / "class_a" / "copy.jpg").write_bytes(data)       # duplicate in the same class
    records, _ = run_checks(tmp_path)
    assert WARNING in severities(checks.check_duplicates(records, CFG), "duplicates")
    (tmp_path / "class_b" / "mislabelled.jpg").write_bytes(data)  # same image, other label
    records, _ = run_checks(tmp_path)
    assert ERROR in severities(checks.check_duplicates(records, CFG), "duplicates")


def _split_dataset(tmp_path):
    for split in ("train", "val", "test"):
        make_dataset(tmp_path / split)
    base = tmp_path
    recs = []
    for split in ("train", "val", "test"):
        r, _ = checks.scan_tree(base / split, split, base)
        recs += r
    checks.inspect_images(recs, CFG)
    return base, recs


def test_cross_split_leakage_respects_accepted_limit(tmp_path):
    base, _ = _split_dataset(tmp_path)
    (base / "test" / "class_a" / "leak.jpg").write_bytes((base / "train" / "class_a" / "0.jpg").read_bytes())
    recs = []
    for split in ("train", "val", "test"):
        r, _ = checks.scan_tree(base / split, split, base)
        recs += r
    checks.inspect_images(recs, CFG)
    assert severities(checks.check_duplicates(recs, {**CFG, "accepted_cross_split_duplicate_groups": 0}), "leakage") == [ERROR]
    assert severities(checks.check_duplicates(recs, {**CFG, "accepted_cross_split_duplicate_groups": 1}), "leakage") == [WARNING]


def test_split_consistency(tmp_path):
    base, recs = _split_dataset(tmp_path)
    by_split = {s: [r for r in recs if r.split == s] for s in ("train", "val", "test")}
    ratios = {"train": 1 / 3, "val": 1 / 3, "test": 1 / 3}
    assert not [f for f in checks.check_split_consistency(by_split, ratios, CFG) if f.severity != INFO]
    by_split["val"] = [r for r in by_split["val"] if r.cls != "class_b"]      # class missing from a split
    assert ERROR in severities(checks.check_split_consistency(by_split, ratios, CFG), "split_consistency")
    skewed = {"train": 0.8, "val": 0.1, "test": 0.1}
    by_split = {s: [r for r in recs if r.split == s] for s in ("train", "val", "test")}
    assert WARNING in severities(checks.check_split_consistency(by_split, skewed, CFG), "split_consistency")


def test_class_distribution_rules(tmp_path):
    make_dataset(tmp_path, classes={"big": 40, "small": 3})
    records, _ = checks.scan_tree(tmp_path, None, tmp_path)
    found = checks.check_class_distribution(records, {**CFG, "max_imbalance_ratio": 5}, "raw")
    assert ERROR in [f.severity for f in found]      # 'small' is below the minimum of 10
    assert WARNING in [f.severity for f in found]    # 40 / 3 exceeds the ratio 5


def test_class_baseline(tmp_path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"classes": ["a", "b"]}))
    cfg = {**CFG, "baseline_path": str(baseline)}
    assert [f.severity for f in checks.check_classes_against_baseline(["a", "b"], cfg)] == [INFO]
    assert ERROR in [f.severity for f in checks.check_classes_against_baseline(["a", "b", "c"], cfg)]  # unexpected
    assert ERROR in [f.severity for f in checks.check_classes_against_baseline(["a"], cfg)]            # missing
    absent = {**CFG, "baseline_path": str(tmp_path / "none.json")}
    assert [f.severity for f in checks.check_classes_against_baseline(["a"], absent)] == [INFO]


def test_fingerprint_is_stable_and_detects_changes(tmp_path):
    make_dataset(tmp_path)
    r1, _ = run_checks(tmp_path)
    r2, _ = run_checks(tmp_path)
    assert checks.fingerprint(r1) == checks.fingerprint(r2)
    (tmp_path / "class_a" / "0.jpg").write_bytes(image_bytes())
    r3, _ = run_checks(tmp_path)
    assert checks.fingerprint(r1) != checks.fingerprint(r3)
