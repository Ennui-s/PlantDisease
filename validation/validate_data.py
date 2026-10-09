"""Data validation gate for the plant disease dataset.

    python -m validation.validate_data --stage raw       # dataset/raw before splitting
    python -m validation.validate_data --stage splits    # dataset/{train,val,test} (used before training)
    python -m validation.validate_data --stage all
    python -m validation.validate_data --write-baseline  # record the current class set

Exit code is 0 when no error-level findings exist, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from config import CONFIG
from training.run_config import apply_overrides, load_run_config
from validation import checks
from validation.checks import DEFAULTS, ERROR, INFO, WARNING, Finding

SPLITS = ("train", "val", "test")


class DataValidationError(RuntimeError):
    """Raised when the dataset has error-level findings and training must not proceed."""


def _validation_cfg(run_cfg: dict) -> dict:
    extra = set(run_cfg["validation"]) - set(DEFAULTS)
    if extra:
        raise ValueError(f"Unknown validation setting(s): {sorted(extra)}")
    return {**DEFAULTS, **run_cfg["validation"]}


def _stage_result(name: str, root: Path, records, findings: List[Finding]) -> dict:
    readable = [r for r in records if r.ok and r.mode]
    return {
        "stage": name,
        "root": str(root),
        "n_files": len(records),
        "n_readable_images": len(readable),
        "fingerprint": checks.fingerprint(records),
        "classes": sorted({r.cls for r in records}),
        "findings": [vars(f) for f in findings],
        "passed": not any(f.severity == ERROR for f in findings),
    }


def validate_raw(cfg: dict) -> dict:
    root = Path(CONFIG["raw_dir"])
    records, findings = checks.scan_tree(root, split=None, base=root)
    if records:
        checks.inspect_images(records, cfg)
        findings += checks.check_files(records, cfg)
        findings += checks.check_class_distribution(records, cfg, "raw")
        findings += checks.check_duplicates(records, cfg)
        findings += checks.check_classes_against_baseline(sorted({r.cls for r in records}), cfg)
    return _stage_result("raw", root, records, findings)


def validate_splits(cfg: dict) -> dict:
    base = Path(CONFIG["data_dir"])
    findings: List[Finding] = []
    by_split: Dict[str, list] = {}
    for split in SPLITS:
        recs, f = checks.scan_tree(base / split, split=split, base=base)
        by_split[split] = recs
        findings += f
    all_records = [r for recs in by_split.values() for r in recs]
    if all_records:
        checks.inspect_images(all_records, cfg)
        findings += checks.check_files(all_records, cfg)
        # The per-class minimum applies to the whole dataset; each split only needs every class present.
        findings += checks.check_class_distribution(all_records, cfg, "all splits")
        for split, recs in by_split.items():
            if recs:
                findings += checks.check_class_distribution(recs, cfg, split, min_images=1)
        findings += checks.check_duplicates(all_records, cfg)
        ratios = {"train": CONFIG["train_ratio"], "val": CONFIG["val_ratio"],
                  "test": CONFIG["test_ratio"]}
        findings += checks.check_split_consistency(
            {s: r for s, r in by_split.items() if r}, ratios, cfg)
        findings += checks.check_classes_against_baseline(
            sorted({r.cls for r in all_records}), cfg)
    return _stage_result("splits", base, all_records, findings)


def run_validation(stage: str, run_cfg: dict, report_path: Optional[Path] = None) -> dict:
    """Validate the dataset and (optionally) write a JSON report. Returns the report dict."""
    cfg = _validation_cfg(run_cfg)
    stages = []
    if stage in ("raw", "all"):
        stages.append(validate_raw(cfg))
    if stage in ("splits", "all"):
        stages.append(validate_splits(cfg))
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "passed": all(s["passed"] for s in stages),
        "stages": stages,
    }
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))
    return report


def print_report(report: dict, verbose: bool = False) -> None:
    for s in report["stages"]:
        print(f"\n== Data validation: {s['stage']} ({s['root']}) ==")
        print(f"files: {s['n_files']}  readable images: {s['n_readable_images']}  "
              f"classes: {len(s['classes'])}  fingerprint: {s['fingerprint'][:16]}")
        for sev in (ERROR, WARNING, INFO):
            for f in s["findings"]:
                if f["severity"] != sev or (sev == INFO and not verbose):
                    continue
                print(f"  [{sev.upper():7}] ({f['check']}) {f['message']}")
                for key, value in f["details"].items():
                    if sev != INFO:
                        print(f"             {key}: {value}")
        print(f"  -> {'PASSED' if s['passed'] else 'FAILED'}")
    print(f"\nOverall: {'PASSED' if report['passed'] else 'FAILED'}")


def write_baseline(run_cfg: dict) -> Path:
    """Record the class set of dataset/raw so later runs can detect unexpected/missing classes."""
    cfg = _validation_cfg(run_cfg)
    stage = validate_raw({**cfg, "baseline_path": "<none>"})
    if not stage["passed"]:
        raise DataValidationError("Raw dataset has validation errors; fix them before writing a baseline.")
    path = Path(cfg["baseline_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "classes": stage["classes"],
        "n_files": stage["n_files"],
        "fingerprint": stage["fingerprint"],
    }, indent=2))
    return path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the plant disease dataset.")
    parser.add_argument("--stage", choices=["raw", "splits", "all"], default="all")
    parser.add_argument("--config", default=None, help="run config YAML (default: configs/training.yaml)")
    parser.add_argument("--report", default="reports/data_validation.json")
    parser.add_argument("--verbose", action="store_true", help="also print info-level findings")
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args(argv)

    run_cfg = load_run_config(args.config)
    apply_overrides(run_cfg["config_overrides"])

    if args.write_baseline:
        print(f"Baseline written to {write_baseline(run_cfg)}")
        return 0

    report = run_validation(args.stage, run_cfg, Path(args.report))
    print_report(report, verbose=args.verbose)
    print(f"Report: {args.report}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
