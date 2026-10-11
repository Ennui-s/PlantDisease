"""Champion vs candidate: paired comparison on the same test images and the promotion decision. Pure Python."""
from __future__ import annotations

from collections.abc import Sequence
from math import comb
from pathlib import Path

import yaml

POLICY_KEYS = {"alpha", "min_accuracy_gain", "max_class_recall_drop"}
EPS = 1e-12


def load_policy(path: str) -> dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Promotion policy not found: {p}")
    cfg = yaml.safe_load(p.read_text()) or {}
    missing, extra = POLICY_KEYS - set(cfg), set(cfg) - POLICY_KEYS
    if missing or extra:
        raise ValueError(f"{p}: missing keys {sorted(missing)}, unknown keys {sorted(extra)}")
    return cfg


def paired_counts(y_true: Sequence[int], champion: Sequence[int], candidate: Sequence[int]) -> dict:
    """b: champion right and candidate wrong; c: champion wrong and candidate right."""
    if not (len(y_true) == len(champion) == len(candidate)):
        raise ValueError("predictions and labels must have the same length")
    b = sum(1 for t, a, c in zip(y_true, champion, candidate) if a == t and c != t)
    c = sum(1 for t, a, c_ in zip(y_true, champion, candidate) if a != t and c_ == t)
    return {"champion_only_correct": b, "candidate_only_correct": c}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value (binomial test of b vs c with p = 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def decide(champion: dict, candidate: dict, y_true, champion_pred, candidate_pred, policy: dict) -> dict:
    """Promotion decision. `champion` and `candidate` are metrics dicts (evaluation.metrics.compute_metrics)."""
    counts = paired_counts(y_true, champion_pred, candidate_pred)
    b, c = counts["champion_only_correct"], counts["candidate_only_correct"]
    p_value = mcnemar_exact(b, c)
    gain = candidate["accuracy"] - champion["accuracy"]

    failures: list[str] = []
    if c <= b:
        failures.append(f"the candidate does not fix more images than it breaks ({c} fixed, {b} broken)")
    if gain < policy["min_accuracy_gain"] - EPS:
        failures.append(f"accuracy gain {gain:+.4f} is below the required {policy['min_accuracy_gain']}")
    if p_value >= policy["alpha"]:
        failures.append(f"the difference is not statistically significant (McNemar p = {p_value:.3f}, alpha = {policy['alpha']})")
    worse = {}
    if candidate["classes"] != champion["classes"]:
        failures.append("candidate and champion predict different classes")
    else:
        for cls, ch in champion["per_class"].items():
            drop = ch["recall"] - candidate["per_class"][cls]["recall"]
            if drop > policy["max_class_recall_drop"] + EPS:
                worse[cls] = {"champion_recall": ch["recall"], "candidate_recall": candidate["per_class"][cls]["recall"],
                              "drop": drop, "support": ch["support"]}
        if worse:
            failures.append("recall dropped too far in: " + ", ".join(worse))
    return {
        "promote": not failures,
        "reasons": failures or ["all promotion rules satisfied"],
        "champion_accuracy": champion["accuracy"], "candidate_accuracy": candidate["accuracy"],
        "accuracy_gain": gain, "champion_macro_f1": champion["macro_f1"], "candidate_macro_f1": candidate["macro_f1"],
        **counts, "mcnemar_p": p_value, "n_images": len(list(y_true)),
        "class_recall_drops_beyond_limit": worse, "policy": dict(policy),
    }
