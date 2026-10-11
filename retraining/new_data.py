"""Ingest a batch of new labelled images for retraining.

New images are added to the TRAINING split only. The validation and test splits stay exactly as they are,
so every model version is evaluated on the same test images and results stay comparable.

The batch is validated (readable, allowed format, minimum size), its class folders must already exist
in the dataset, and an image is excluded if it is
  - byte-identical to an image already in train/val/test (a duplicate, or a leak into the test set),
  - filed under a different class than an identical existing image (label conflict),
  - duplicated inside the batch.
The result is a derived dataset folder (existing splits linked or copied, plus the accepted new images).
"""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from validation import checks

SPLITS = ("train", "val", "test")


class NewDataError(RuntimeError):
    """The new data cannot be used; the message says why."""


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)            # no extra disk space; fails across volumes or on some file systems
    except OSError:
        shutil.copy2(src, dst)


def index_existing(data_dir: Path) -> Dict[str, tuple]:
    """sha256 -> (split, class) for every image already in train/val/test."""
    index = {}
    for split in SPLITS:
        for cls_dir in sorted(p for p in (data_dir / split).iterdir() if p.is_dir()):
            for f in sorted(cls_dir.rglob("*")):
                if f.is_file() and f.suffix.lower() in checks.LOADER_EXTENSIONS:
                    index.setdefault(_sha256(f), (split, cls_dir.name))
    return index


def ingest(new_dir: str, data_dir: str, derived_dir: str, cfg: Optional[dict] = None) -> dict:
    cfg = {**checks.DEFAULTS, **(cfg or {})}
    new, base, derived = Path(new_dir), Path(data_dir), Path(derived_dir)
    records, findings = checks.scan_tree(new, None, new)
    errors = [f.message for f in findings if f.severity == checks.ERROR]
    if errors:
        raise NewDataError("; ".join(errors))

    known = {p.name for p in (base / "train").iterdir() if p.is_dir()}
    unexpected = sorted({r.cls for r in records} - known)
    if unexpected:
        raise NewDataError(f"New data has class folders the model does not know: {unexpected}. "
                           "A model with a fixed set of classes cannot learn new classes by retraining.")
    if derived.exists() and any(derived.iterdir()):
        raise NewDataError(f"{derived} already exists and is not empty; choose another name.")

    checks.inspect_images(records, cfg)
    existing = index_existing(base)
    allowed = {e.lower() for e in cfg["extensions"]}

    excluded: Dict[str, List[str]] = {}

    def exclude(reason, rec):
        excluded.setdefault(reason, []).append(rec.rel)

    candidates = []
    for r in records:
        if r.ext not in checks.LOADER_EXTENSIONS:
            exclude("not_an_image", r)
        elif r.ext not in allowed:
            exclude("unsupported_format", r)
        elif not r.ok:
            exclude("unreadable", r)
        elif min(r.width, r.height) < int(cfg["min_side"]):
            exclude("too_small", r)
        else:
            candidates.append(r)

    by_hash: Dict[str, list] = {}
    for r in candidates:
        by_hash.setdefault(r.sha256, []).append(r)
    accepted = []
    for sha, group in by_hash.items():
        if len({g.cls for g in group}) > 1:
            for g in group:
                exclude("label_conflict_in_batch", g)
        elif sha in existing:
            kind = "duplicate_of_existing" if existing[sha][1] == group[0].cls else "label_conflict_with_existing"
            for g in group:
                exclude(kind, g)
        else:
            accepted.append(group[0])
            for g in group[1:]:
                exclude("duplicate_in_batch", g)
    if not accepted:
        raise NewDataError("No usable new images: " + (", ".join(f"{k} {len(v)}" for k, v in excluded.items()) or "folder empty"))

    for split in SPLITS:
        for cls_dir in sorted(p for p in (base / split).iterdir() if p.is_dir()):
            for f in sorted(cls_dir.rglob("*")):
                if f.is_file():
                    _link_or_copy(f, derived / split / cls_dir.name / f.name)
    for r in accepted:
        _link_or_copy(Path(r.path), derived / "train" / r.cls / f"new_{r.sha256[:12]}{r.ext}")

    per_class: Dict[str, int] = {}
    for r in accepted:
        per_class[r.cls] = per_class.get(r.cls, 0) + 1
    return {
        "new_data_dir": str(new), "derived_dir": str(derived), "n_found": len(records), "n_accepted": len(accepted),
        "accepted_per_class": dict(sorted(per_class.items())),
        "excluded": {k: {"count": len(v), "examples": v[:10]} for k, v in sorted(excluded.items())},
        "new_images_fingerprint": hashlib.sha256("\n".join(sorted(r.sha256 for r in accepted)).encode()).hexdigest(),
    }
