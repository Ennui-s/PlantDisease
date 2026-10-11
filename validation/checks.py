"""Data validation checks for an ImageFolder-style image dataset.

Layout assumed (the same one torchvision's ImageFolder and data/split_dataset.py use):

    <root>/<class_name>/<image files>                  # raw stage
    <root>/{train,val,test}/<class_name>/<image files> # splits stage

Each check returns a list of Finding objects. A Finding has a severity:
    error   - the dataset should not be used for training until this is resolved
    warning - worth knowing about, does not block training
    info    - descriptive statistics
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

ERROR, WARNING, INFO = "error", "warning", "info"

# Extensions torchvision's ImageFolder will load. Files with any other extension are
# silently skipped by the loader, which is why we report them rather than ignore them.
LOADER_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}

# Which file extensions are consistent with the format Pillow detects from the file content.
FORMAT_EXTENSIONS = {
    "JPEG": {".jpg", ".jpeg"},
    "PNG": {".png"},
    "BMP": {".bmp"},
    "WEBP": {".webp"},
    "TIFF": {".tif", ".tiff"},
}

DEFAULTS = {
    # Formats accepted for training. A loadable format outside this list is an error.
    "extensions": [".jpg", ".jpeg", ".png"],
    # Smallest allowed width/height in pixels (images are resized to config image_size).
    "min_side": 32,
    # Fewest images a class may have; a class this small cannot be split meaningfully.
    "min_images_per_class": 10,
    # Warn when largest class / smallest class exceeds this.
    "max_imbalance_ratio": 30.0,
    # Allowed deviation between configured and actual split fractions.
    "split_ratio_tolerance": 0.02,
    # Fully decode every image (catches truncated files) instead of only reading headers.
    "full_decode": True,
    "workers": 4,
    # Identical files inside one split or in raw data: "error" or "warning".
    "duplicates_within_dataset": "warning",
    # Identical files in different splits (train/val/test leakage): "error" or "warning".
    "cross_split_duplicates": "error",
    # Number of leaked groups tolerated as a known, documented issue. Anything above this
    # is reported with the severity above, so new leakage still fails the gate.
    "accepted_cross_split_duplicate_groups": 0,
    # Class-name baseline written by `--write-baseline`. Skipped if the file does not exist.
    "baseline_path": "configs/data_baseline.json",
}

MAX_EXAMPLES = 10


@dataclass
class Finding:
    check: str
    severity: str
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class ImageRecord:
    path: str
    rel: str            # path relative to the dataset root, used for fingerprints/reporting
    cls: str
    split: str | None
    ext: str
    size_bytes: int = 0
    sha256: str = ""
    ok: bool = True
    error: str = ""
    fmt: str = ""
    mode: str = ""
    width: int = 0
    height: int = 0


def _examples(items, n: int = MAX_EXAMPLES) -> list:
    items = list(items)
    return items[:n] + ([f"... and {len(items) - n} more"] if len(items) > n else [])


def _severity(value: str) -> str:
    if value not in (ERROR, WARNING):
        raise ValueError(f"severity must be 'error' or 'warning', got {value!r}")
    return value


# --------------------------------------------------------------------------- scanning

def scan_tree(root: Path, split: str | None, base: Path):
    """Walk <root>/<class>/... and return (records, findings). No image is opened here."""
    findings: list[Finding] = []
    records: list[ImageRecord] = []

    if not root.is_dir():
        findings.append(Finding("structure", ERROR, f"Directory not found: {root}"))
        return records, findings

    class_dirs, stray = [], []
    for entry in sorted(root.iterdir()):
        if entry.name.startswith("."):
            continue
        (class_dirs if entry.is_dir() else stray).append(entry)

    if stray:
        findings.append(Finding(
            "structure", WARNING,
            f"{len(stray)} file(s) sit directly in {root} instead of inside a class folder "
            "(ImageFolder ignores them).",
            {"examples": _examples(p.name for p in stray)}))
    if not class_dirs:
        findings.append(Finding("structure", ERROR, f"No class folders found in {root}"))
        return records, findings

    nested, empty = [], []
    for cd in class_dirs:
        found = 0
        for dirpath, dirnames, filenames in os.walk(cd):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            if Path(dirpath) != cd:
                nested.append(str(Path(dirpath).relative_to(root)))
            for fn in sorted(filenames):
                if fn.startswith("."):
                    continue
                p = Path(dirpath) / fn
                records.append(ImageRecord(
                    path=str(p), rel=p.relative_to(base).as_posix(), cls=cd.name,
                    split=split, ext=p.suffix.lower()))
                found += 1
        if found == 0:
            empty.append(cd.name)

    if empty:
        findings.append(Finding("structure", ERROR, f"{len(empty)} class folder(s) are empty.",
                                {"classes": _examples(empty)}))
    if nested:
        findings.append(Finding(
            "structure", WARNING,
            f"{len(nested)} nested sub-folder(s) inside class folders; their files are loaded "
            "under the parent class by ImageFolder.", {"examples": _examples(nested)}))
    return records, findings


# --------------------------------------------------------------------------- per-file inspection

def _inspect(rec: ImageRecord, full_decode: bool) -> ImageRecord:
    try:
        rec.size_bytes = os.path.getsize(rec.path)
        h = hashlib.sha256()
        with open(rec.path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        rec.sha256 = h.hexdigest()
        if rec.size_bytes == 0:
            raise ValueError("empty file")
        with Image.open(rec.path) as im:
            rec.fmt, rec.mode = im.format or "", im.mode
            rec.width, rec.height = im.size
            im.verify()
        if full_decode:  # verify() only checks structure; load() decodes pixel data
            with Image.open(rec.path) as im:
                im.load()
    except Exception as exc:  # noqa: BLE001  any failure, of any type, means the loader could not use this file
        rec.ok = False
        rec.error = f"{type(exc).__name__}: {exc}"
    return rec


def inspect_images(records: list[ImageRecord], cfg: dict) -> list[ImageRecord]:
    """Hash and open every file. Non-image extensions the loader would skip are hashed only."""
    todo = [r for r in records if r.ext in LOADER_EXTENSIONS]
    with ThreadPoolExecutor(max_workers=int(cfg["workers"])) as pool:
        list(pool.map(lambda r: _inspect(r, bool(cfg["full_decode"])), todo))
    return records


# --------------------------------------------------------------------------- dataset-level checks

def check_files(records: list[ImageRecord], cfg: dict) -> list[Finding]:
    findings: list[Finding] = []
    allowed = {e.lower() for e in cfg["extensions"]}

    ignored = [r for r in records if r.ext not in LOADER_EXTENSIONS]
    disallowed = [r for r in records if r.ext in LOADER_EXTENSIONS and r.ext not in allowed]
    broken = [r for r in records if r.ext in LOADER_EXTENSIONS and not r.ok]
    readable = [r for r in records if r.ext in LOADER_EXTENSIONS and r.ok]

    if ignored:
        findings.append(Finding(
            "file_format", WARNING,
            f"{len(ignored)} non-image file(s) will be skipped by the data loader.",
            {"extensions": dict(Counter(r.ext or "(none)" for r in ignored)),
             "examples": _examples(r.rel for r in ignored)}))
    if disallowed:
        findings.append(Finding(
            "file_format", ERROR,
            f"{len(disallowed)} image(s) use a format outside the allowed set {sorted(allowed)}.",
            {"extensions": dict(Counter(r.ext for r in disallowed)),
             "examples": _examples(r.rel for r in disallowed)}))
    if broken:
        findings.append(Finding(
            "readability", ERROR, f"{len(broken)} image file(s) are corrupted or unreadable.",
            {"examples": _examples(f"{r.rel} ({r.error})" for r in broken)}))

    mismatched = [r for r in readable
                  if r.fmt in FORMAT_EXTENSIONS and r.ext not in FORMAT_EXTENSIONS[r.fmt]]
    if mismatched:
        findings.append(Finding(
            "file_format", WARNING,
            f"{len(mismatched)} image(s) have an extension that does not match their content.",
            {"examples": _examples(f"{r.rel} (is {r.fmt})" for r in mismatched)}))

    if readable:
        modes = Counter(r.mode for r in readable)
        if set(modes) != {"RGB"}:
            findings.append(Finding(
                "channels", WARNING,
                "Not all images are RGB; the loader converts them, but check this is intended.",
                {"modes": dict(modes),
                 "examples": _examples(r.rel for r in readable if r.mode != "RGB")}))
        else:
            findings.append(Finding("channels", INFO, "All readable images are RGB."))

        tiny = [r for r in readable if min(r.width, r.height) < int(cfg["min_side"])]
        if tiny:
            findings.append(Finding(
                "dimensions", ERROR,
                f"{len(tiny)} image(s) have a side shorter than {cfg['min_side']} px.",
                {"examples": _examples(f"{r.rel} ({r.width}x{r.height})" for r in tiny)}))
        sizes = Counter((r.width, r.height) for r in readable)
        findings.append(Finding(
            "dimensions", INFO, f"{len(sizes)} distinct image size(s).",
            {"most_common": [{"size": f"{w}x{h}", "count": c} for (w, h), c in sizes.most_common(5)]}))
    return findings


def check_class_distribution(records: list[ImageRecord], cfg: dict, label: str,
                             min_images: int | None = None) -> list[Finding]:
    """min_images defaults to cfg['min_images_per_class']; pass 1 for individual splits."""
    min_required = int(cfg["min_images_per_class"]) if min_images is None else min_images
    counts = Counter(r.cls for r in records)
    findings = [Finding("class_distribution", INFO, f"{label}: {len(counts)} classes, {len(records)} files.",
                        {"counts": dict(sorted(counts.items()))})]
    small = {c: n for c, n in counts.items() if n < min_required}
    if small:
        findings.append(Finding(
            "class_distribution", ERROR,
            f"{label}: {len(small)} class(es) have fewer than {min_required} image(s).",
            {"classes": small}))
    if counts and min(counts.values()) > 0:
        ratio = max(counts.values()) / min(counts.values())
        if ratio > float(cfg["max_imbalance_ratio"]):
            findings.append(Finding(
                "class_distribution", WARNING,
                f"{label}: class imbalance ratio {ratio:.1f} exceeds {cfg['max_imbalance_ratio']}.",
                {"largest": max(counts, key=counts.get), "smallest": min(counts, key=counts.get)}))
    return findings


def check_duplicates(records: list[ImageRecord], cfg: dict) -> list[Finding]:
    """Exact (byte-identical) duplicates, label conflicts and cross-split leakage."""
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for r in records:
        if r.sha256 and r.ext in LOADER_EXTENSIONS:
            groups[r.sha256].append(r)
    dupes = [g for g in groups.values() if len(g) > 1]
    if not dupes:
        return [Finding("duplicates", INFO, "No byte-identical duplicate images found.")]

    findings: list[Finding] = []
    conflict = [g for g in dupes if len({r.cls for r in g}) > 1]
    leaked = [g for g in dupes if len({r.split for r in g if r.split}) > 1]
    same_split = [g for g in dupes if g not in conflict and g not in leaked]

    def describe(group):
        return " == ".join(r.rel for r in group[:4]) + (" ..." if len(group) > 4 else "")

    if conflict:
        findings.append(Finding(
            "duplicates", ERROR,
            f"{len(conflict)} identical image(s) appear under more than one class label.",
            {"examples": _examples(describe(g) for g in conflict)}))
    if leaked:
        accepted = int(cfg["accepted_cross_split_duplicate_groups"])
        within = len(leaked) <= accepted
        limit = (f"within the accepted limit of {accepted}" if within
                 else f"exceeds the accepted limit of {accepted}")
        findings.append(Finding(
            "leakage", WARNING if within else _severity(cfg["cross_split_duplicates"]),
            f"{len(leaked)} identical image(s) appear in more than one of train/val/test "
            f"(train/evaluation leakage); {limit}.",
            {"examples": _examples(describe(g) for g in leaked)}))
    if same_split:
        extra = sum(len(g) - 1 for g in same_split)
        findings.append(Finding(
            "duplicates", _severity(cfg["duplicates_within_dataset"]),
            f"{len(same_split)} duplicate group(s) ({extra} redundant copies) within the same split/folder.",
            {"examples": _examples(describe(g) for g in same_split)}))
    return findings


def check_classes_against_baseline(classes: list[str], cfg: dict) -> list[Finding]:
    path = Path(cfg["baseline_path"])
    if not path.is_file():
        return [Finding("classes", INFO,
                        f"No class baseline at {path}; class-set check skipped. "
                        "Create one with --write-baseline.")]
    baseline = json.loads(path.read_text())
    expected, actual = set(baseline["classes"]), set(classes)
    findings: list[Finding] = []
    if actual - expected:
        findings.append(Finding("classes", ERROR, "Unexpected class folder(s) not in the baseline.",
                                {"unexpected": sorted(actual - expected)}))
    if expected - actual:
        findings.append(Finding("classes", ERROR, "Class folder(s) from the baseline are missing.",
                                {"missing": sorted(expected - actual)}))
    if not findings:
        findings.append(Finding("classes", INFO, f"Class set matches the baseline ({len(expected)} classes)."))
    return findings


def check_split_consistency(by_split: dict[str, list[ImageRecord]], ratios: dict[str, float],
                            cfg: dict) -> list[Finding]:
    findings: list[Finding] = []
    class_sets = {s: {r.cls for r in recs} for s, recs in by_split.items()}
    union = set().union(*class_sets.values()) if class_sets else set()
    for split, present in class_sets.items():
        if present != union:
            findings.append(Finding(
                "split_consistency", ERROR,
                f"Split '{split}' is missing class(es) that exist in other splits.",
                {"missing": sorted(union - present)}))

    total = sum(len(v) for v in by_split.values())
    if total:
        tol = float(cfg["split_ratio_tolerance"])
        actual = {s: len(v) / total for s, v in by_split.items()}
        off = {s: {"expected": ratios[s], "actual": round(actual[s], 4)}
               for s in by_split if abs(actual[s] - ratios[s]) > tol}
        if off:
            findings.append(Finding(
                "split_consistency", WARNING,
                f"Split sizes deviate from the configured ratios by more than {tol:.0%}.", off))
        else:
            findings.append(Finding(
                "split_consistency", INFO, "Split sizes match the configured ratios.",
                {"counts": {s: len(v) for s, v in by_split.items()},
                 "fractions": {s: round(f, 4) for s, f in actual.items()}}))
    return findings


def fingerprint(records: list[ImageRecord]) -> str:
    """Stable hash of the dataset's contents (relative path + file hash for every file)."""
    lines = sorted(f"{r.rel}\t{r.sha256}" for r in records if r.sha256)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()
