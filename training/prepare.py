"""Run preparation shared by the training entry point: load config, apply it, gate on data validation.

Kept free of torch imports so it can be tested and reused (e.g. by CI or a retraining job)
without loading the ML stack.
"""
from pathlib import Path
from typing import Optional

from training.run_config import apply_overrides, load_run_config
from validation.validate_data import DataValidationError, print_report, run_validation


def prepare_run(config_path: Optional[str] = None, skip_validation: bool = False,
                report_dir: str = "reports", overrides: Optional[dict] = None) -> dict:
    run_cfg = load_run_config(config_path)
    # overrides given by the caller (e.g. the retraining workflow) win over the config file
    run_cfg["config_overrides"] = {**run_cfg["config_overrides"], **(overrides or {})}
    apply_overrides(run_cfg["config_overrides"])

    validation = {"ran": False}
    if run_cfg["run_validation"] and not skip_validation:
        report_path = Path(report_dir) / "data_validation.json"
        report = run_validation("splits", run_cfg, report_path)
        print_report(report)
        validation = {
            "ran": True,
            "passed": report["passed"],
            "report_path": str(report_path),
            "dataset_fingerprint": report["stages"][0]["fingerprint"],
        }
        if not report["passed"]:
            raise DataValidationError(
                f"Data validation failed; training aborted. See {report_path} for details.")
    else:
        print("Data validation skipped.")
    return {"run_cfg": run_cfg, "validation": validation}
