"""Loads a run configuration (YAML) and applies it on top of the defaults in config.py.

config.py stays the single source of defaults for the classifier. A YAML file can override
individual CONFIG values for one run and holds platform settings (data validation thresholds,
experiment tracking, seed). The classifier code reads CONFIG at call time, so updating the shared
dict in place is enough to change its behaviour without editing the classifier.

A YAML file may name another one with `base:`; the two are merged (nested sections are merged key
by key, anything else in the child replaces the base). This keeps experiment files short.
"""
from pathlib import Path
from typing import Optional

import yaml

from config import CONFIG

DEFAULT_CONFIG_PATH = "configs/training.yaml"
SECTIONS = ("config_overrides", "validation", "tracking", "registry")
RUN_DEFAULTS = {
    "base": None,               # optional path to a YAML file this one extends
    "seed": None,               # optional torch seed; None keeps the original (unseeded) behaviour
    "run_validation": True,     # validate the train/val/test splits before training
    "config_overrides": {},     # keys must already exist in config.py CONFIG
    "validation": {},           # thresholds, see validation/checks.py DEFAULTS
    "tracking": {},             # MLflow settings, see tracking/tracker.py TRACKING_DEFAULTS
    "registry": {},             # model registry settings, see registry/policy.py REGISTRY_DEFAULTS
}


def _read_yaml(path: Path, depth: int = 0) -> dict:
    if depth > 5:
        raise ValueError(f"Config 'base' chain is too deep or circular at {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    unknown = set(data) - set(RUN_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown top-level key(s) in {path}: {sorted(unknown)}")
    base = data.get("base")
    if not base:
        return data
    merged = _read_yaml(Path(base), depth + 1)
    for key, value in data.items():
        if key == "base":
            continue
        if key in SECTIONS:
            merged[key] = {**(merged.get(key) or {}), **(value or {})}
        else:
            merged[key] = value
    return merged


def load_run_config(path: Optional[str] = None) -> dict:
    if path is None:
        path = DEFAULT_CONFIG_PATH if Path(DEFAULT_CONFIG_PATH).is_file() else None
    data = _read_yaml(Path(path)) if path is not None else {}
    data.pop("base", None)
    run = {**RUN_DEFAULTS, **data}
    for key in SECTIONS:
        run[key] = dict(run[key] or {})
    run["path"] = path
    return run


def apply_overrides(overrides: dict) -> None:
    unknown = set(overrides) - set(CONFIG)
    if unknown:
        raise ValueError(f"config_overrides contains keys not in config.py: {sorted(unknown)}")
    CONFIG.update(overrides)
