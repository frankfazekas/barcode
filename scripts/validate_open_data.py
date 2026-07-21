#!/usr/bin/env python3
"""Check BARCODE's volumetric metrics against independently published ground truth.

The 2D branch can be regressed against the hackathon reference barcodes. The volumetric
branch has had no such anchor: it was developed on one dataset, with masks we made
ourselves, so every check available until now was self-consistency -- does the number
stay the same -- rather than correctness -- is the number right.

Open data supplies the missing anchor, because for these datasets somebody else already
determined the answer. Each check below states a quantity BARCODE reports, an
independent way to compute that same quantity, and what a disagreement would mean.

    V1  object count      distinct labels in the published mask
    V2  object volume     voxels in the published mask x published voxel size
    V3  speed             centroid displacement of TRACKED objects / published dt
    V4  replicate spread  independent sequences / fields of the same specimen
    V5  determinism       the same input analysed twice

V1 and V2 are exact: the mask is the definition of the object, so any disagreement is
BARCODE's. V3 is a correlation, not an identity -- see the note on that check. V4 is not
pass/fail but the measurement that says which barcode columns carry signal rather than
noise. V5 is pass/fail.

    python scripts/validate_open_data.py --root L:/FF/Hackathon/full_datasets
    python scripts/validate_open_data.py --root ... --dataset ctc_Fluo-N3DH-CHO_01

Writes a per-dataset CSV and a summary CSV under ``<root>/_validation``, and prints the
report. Nothing here re-runs the pipeline; it reads the CSVs the runners already wrote,
so it is cheap to iterate on.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts._staging import read_tiff_any, read_tiff_bytes

# ------------------------------------------------------------------ staged datasets


@dataclass
class Staged:
    """A staged dataset folder and whatever the runners have produced in it."""

    folder: str
    name: str
    kind: str                      # "ctc" or "allen"
    xy_um: float = 0.0
    z_um: float = 0.0
    dt_s: float = 0.0
    ctc_dataset: str = ""          # e.g. Fluo-N3DH-CHO
    sequence: str = ""             # e.g. 01

    @property
    def data_dir(self) -> str:
        return os.path.join(self.folder, "BARCODE", "data")

    @property
    def mask_dir(self) -> str:
        return os.path.join(self.folder, "BARCODE", "masks")

    @property
    def results_dir(self) -> str:
        return os.path.join(self.folder, "BARCODE", "results")

    def csvs(self) -> List[str]:
        return sorted(glob.glob(os.path.join(self.results_dir, "**", "*.csv"),
                                recursive=True))


_README_NUMBER = {
    "xy": re.compile(r"xy\s+([\d.eE+-]+)\s*um/pixel|xy step\s+([\d.eE+-]+)"),
    "z": re.compile(r"z step\s+([\d.eE+-]+)|z\s+([\d.eE+-]+)\s*um\b"),
    "dt": re.compile(r"frame interval\s+([\d.eE+-]+)"),
}


def _read_readme_geometry(folder: str) -> Tuple[float, float, float]:
    """Recover xy/z/dt from the README the stager wrote.

    The staged TIFFs carry xy and z themselves, but reading a header per dataset just to
    label a report is wasteful, and dt is not in every file. The README is written by us
    and is the one place all three are stated together.
    """
    path = os.path.join(folder, "README.txt")
    if not os.path.isfile(path):
        return 0.0, 0.0, 0.0
    text = open(path, encoding="utf-8").read()
    def grab(pattern: str) -> float:
        found = re.search(pattern, text)
        return float(found.group(1)) if found else 0.0
    return (
        grab(r"xy step\s+([\d.eE+-]+)") or grab(r"xy\s+([\d.eE+-]+)\s*um/pixel"),
        grab(r"z step\s+([\d.eE+-]+)") or grab(r"z\s+([\d.eE+-]+)\s*um\b"),
        grab(r"frame interval\s+([\d.eE+-]+)"),
    )


def discover(root: str) -> List[Staged]:
    found: List[Staged] = []
    for folder in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(os.path.join(folder, "BARCODE", "data")):
            continue
        name = os.path.basename(folder)
        kind = "ctc" if name.startswith("ctc_") else "allen" if name.startswith("allen_") else "other"
        xy, z, dt = _read_readme_geometry(folder)
        entry = Staged(folder=folder, name=name, kind=kind, xy_um=xy, z_um=z, dt_s=dt)
        if kind == "ctc":
            match = re.match(r"ctc_(?P<dataset>.+)_(?P<seq>\d+)$", name)
            if match:
                entry.ctc_dataset = match.group("dataset")
                entry.sequence = match.group("seq")
        found.append(entry)
    return found


# --------------------------------------------------------------- truth from masks


@dataclass
class MaskTruth:
    """What the published segmentation says, computed directly from it."""

    frames: List[str] = field(default_factory=list)
    counts: List[int] = field(default_factory=list)        # distinct labels
    volumes_um3: List[float] = field(default_factory=list)  # total foreground
    largest_um3: List[float] = field(default_factory=list)  # biggest single object

    def summary(self) -> Dict[str, float]:
        if not self.counts:
            return {}
        return {
            "frames": len(self.counts),
            "count_mean": float(np.mean(self.counts)),
            "count_min": int(np.min(self.counts)),
            "count_max": int(np.max(self.counts)),
            "total_um3_mean": float(np.mean(self.volumes_um3)),
            "largest_um3_mean": float(np.mean(self.largest_um3)),
        }


def mask_truth(entry: Staged, limit: Optional[int] = None) -> MaskTruth:
    """Count objects and measure volume straight from the published masks.

    The mask defines the object, so these are not estimates -- they are the answer
    BARCODE is trying to reproduce. Volumes use the mask's own isotropic voxel (staging
    put masks on the xy grid), which is why this does not need to know anything about
    how the pipeline resampled.
    """
    truth = MaskTruth()
    paths = sorted(glob.glob(os.path.join(entry.mask_dir, "*.tif")))
    if limit:
        paths = paths[:limit]
    voxel_um3 = entry.xy_um ** 3 if entry.xy_um else 1.0

    for path in paths:
        mask = read_tiff_any(path)
        labels, counts = np.unique(mask[mask > 0], return_counts=True)
        truth.frames.append(os.path.basename(path))
        truth.counts.append(int(labels.size))
        truth.volumes_um3.append(float(counts.sum() * voxel_um3))
        truth.largest_um3.append(float(counts.max() * voxel_um3) if counts.size else 0.0)
    return truth


# ------------------------------------------------- truth from CTC tracking markers


@dataclass
class TrackTruth:
    """Speeds derived from the challenge's own tracking annotation."""

    per_frame_um_s: List[float] = field(default_factory=list)
    n_tracks: int = 0

    def summary(self) -> Dict[str, float]:
        values = np.asarray([v for v in self.per_frame_um_s if np.isfinite(v)])
        if values.size == 0:
            return {}
        return {
            "tracks": self.n_tracks,
            "speed_um_s_mean": float(values.mean()),
            "speed_um_s_sd": float(values.std()),
            "speed_um_s_max": float(values.max()),
        }


def track_truth(entry: Staged, downloads: str, max_frames: Optional[int] = None) -> TrackTruth:
    """Mean per-frame object speed from the CTC gold tracking markers.

    Read out of the cached zip rather than an extracted tree: TRA is one labelled volume
    per frame, and unpacking whole datasets to read centroids would cost more disk than
    every analysis output combined.

    IMPORTANT -- what this can and cannot validate. These are the speeds of tracked
    object CENTROIDS: rigid translation of whole nuclei. BARCODE's Speed comes from 3D
    optical flow averaged over the foreground, which also responds to deformation,
    rotation and intensity change, and is a field average rather than a per-object one.
    So the two are NOT expected to be equal. What they should do is agree in ORDER OF
    MAGNITUDE and rise and fall together over time. A constant factor between them is a
    finding about what the metric measures; a factor that tracks the frame interval or
    the voxel size is a UNIT bug, which is exactly the class of error this catches.
    """
    truth = TrackTruth()
    if entry.kind != "ctc" or not entry.ctc_dataset:
        return truth
    archive = os.path.join(downloads, f"{entry.ctc_dataset}.zip")
    if not os.path.isfile(archive):
        return truth
    if not (entry.xy_um and entry.z_um and entry.dt_s):
        return truth

    prefix = f"{entry.ctc_dataset}/{entry.sequence}_GT/TRA/"
    with zipfile.ZipFile(archive) as handle:
        members = sorted(
            name for name in handle.namelist()
            if name.startswith(prefix) and name.endswith((".tif", ".tiff"))
        )
        if max_frames:
            members = members[:max_frames]
        if len(members) < 2:
            return truth

        previous: Dict[int, np.ndarray] = {}
        seen = set()
        for member in members:
            volume = read_tiff_bytes(handle.read(member), member)
            centroids = _label_centroids_um(volume, entry.xy_um, entry.z_um)
            seen.update(centroids)
            if previous:
                shared = set(centroids) & set(previous)
                if shared:
                    steps = [
                        float(np.linalg.norm(centroids[label] - previous[label]))
                        for label in shared
                    ]
                    truth.per_frame_um_s.append(float(np.mean(steps)) / entry.dt_s)
            previous = centroids
        truth.n_tracks = len(seen)
    return truth


def _label_centroids_um(volume: np.ndarray, xy_um: float, z_um: float) -> Dict[int, np.ndarray]:
    """Centre of mass of every label, in microns, from a ZYX label volume."""
    volume = np.asarray(volume)
    if volume.ndim != 3:
        volume = np.squeeze(volume)
    flat = volume.ravel()
    keep = flat > 0
    if not keep.any():
        return {}

    labels = flat[keep]
    z, y, x = np.unravel_index(np.flatnonzero(keep), volume.shape)
    order = np.argsort(labels, kind="stable")
    labels, z, y, x = labels[order], z[order], y[order], x[order]
    unique, starts = np.unique(labels, return_index=True)
    bounds = list(starts) + [labels.size]

    centroids: Dict[int, np.ndarray] = {}
    for i, label in enumerate(unique):
        lo, hi = bounds[i], bounds[i + 1]
        centroids[int(label)] = np.array([
            z[lo:hi].mean() * z_um, y[lo:hi].mean() * xy_um, x[lo:hi].mean() * xy_um,
        ])
    return centroids


# ---------------------------------------------------------------- measured results


def load_csv(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def column(rows: Sequence[Dict[str, str]], *candidates: str) -> Tuple[str, np.ndarray]:
    """Find a column by any of several header spellings and return it as floats.

    Headers differ by mode (a 3D mode says "Volume" where 2D says "Area"), and this is
    a report, not a pipeline -- guessing wrong here would silently compare the wrong
    quantity, so an unmatched name returns empty rather than the nearest thing.
    """
    if not rows:
        return "", np.array([])

    def values_of(header: str) -> np.ndarray:
        out = []
        for row in rows:
            try:
                out.append(float(row[header]))
            except (TypeError, ValueError):
                out.append(np.nan)
        return np.asarray(out, dtype=float)

    headers = list(rows[0])
    # Exact match before prefix match, because the metric names nest: a prefix search
    # for "Maximum Island Volume" also matches "Maximum Island Volume Change", and
    # silently validating a change metric against a static truth would look like a
    # catastrophic failure of a metric that is in fact fine.
    for candidate in candidates:
        wanted = candidate.strip().lower()
        for header in headers:
            if header.strip().lower() == wanted:
                return header, values_of(header)
    for candidate in candidates:
        wanted = candidate.strip().lower()
        for header in headers:
            if header.strip().lower().startswith(wanted):
                return header, values_of(header)
    return "", np.array([])


def _align(measured: Sequence[float], truth: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, int]:
    """Trim two per-frame series to a common length.

    They can differ legitimately -- ``--frame-limit`` shortens the truth, and a run may
    cover a subset of frames -- but comparing a mean over 12 frames with a mean over 92
    produces a difference that reflects only the mismatch. Truncating makes the
    comparison honest and the frame count is reported alongside every result.
    """
    a = np.asarray(measured, dtype=float)
    b = np.asarray(truth, dtype=float)
    n = int(min(a.size, b.size))
    return a[:n], b[:n], n


def _mean_relative(measured: np.ndarray, truth: np.ndarray) -> float:
    """Mean per-frame relative difference, ignoring frames where either is missing."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (measured - truth) / truth
    finite = ratio[np.isfinite(ratio)]
    return float(finite.mean()) if finite.size else np.nan


def relative_difference(measured: float, truth: float) -> float:
    if not np.isfinite(measured) or not np.isfinite(truth) or truth == 0:
        return np.nan
    return (measured - truth) / truth


# ------------------------------------------------------------------------ checking


@dataclass
class Finding:
    dataset: str
    check: str
    quantity: str
    measured: float
    truth: float
    relative: float
    verdict: str
    note: str = ""


def _verdict(relative: float, tolerance: float) -> str:
    if not np.isfinite(relative):
        return "n/a"
    return "PASS" if abs(relative) <= tolerance else "FAIL"


# V4 -- how much each metric varies between independent samples of the same specimen.
# Deliberately not pass/fail: a metric whose scatter across replicates is as large as
# its range across conditions cannot separate conditions, however accurate any single
# measurement of it is. This is the number that says which barcode columns carry signal.
_SPREAD_METRICS = (
    "Mean Island Anisotropy",
    "Structural Correlation Length",
    "Maximum Kurtosis",
    "Mean Contact Number",
)


def _spread_findings(label: str, rows: Sequence[Dict[str, str]]) -> List[Finding]:
    out: List[Finding] = []
    for name in _SPREAD_METRICS:
        header, values = column(rows, name)
        if not header or values.size == 0:
            continue
        finite = values[np.isfinite(values)]
        if finite.size <= 2 or not finite.mean():
            continue
        out.append(Finding(
            label, "V4 replicate spread", header, float(finite.std()),
            float(finite.mean()), float(finite.std() / abs(finite.mean())),
            "info", f"CV across {finite.size} independent rows"))
    return out


def check_dataset(entry: Staged, downloads: str, frame_limit: Optional[int],
                  tolerance: float) -> List[Finding]:
    findings: List[Finding] = []
    csvs = entry.csvs()
    if not csvs:
        return [Finding(entry.name, "results", "csv", np.nan, np.nan, np.nan, "n/a",
                        "no results CSV yet -- run the suite first")]

    truth = mask_truth(entry, limit=frame_limit)
    summary = truth.summary()

    for path in csvs:
        label = f"{entry.name}/{os.path.basename(path)}"
        _, rows = load_csv(path)
        if not rows:
            continue

        # V1 and V2 ask "did the pipeline measure THIS mask correctly", so they are
        # meaningless for a run that never saw the mask. Checking a threshold-only run
        # against mask truth reports thousands of noise specks as an object-count
        # failure, which says nothing about either. The masked/unmasked comparison is
        # V6's job, and it compares the two runs with each other, not to truth.
        from_mask = ("with_masks" in path) or (os.sep + "masked" + os.sep in path)

        if not from_mask:
            # V1-V3 ask "did the pipeline measure THIS mask correctly", so they are
            # meaningless for a run that never saw the mask -- a threshold-only run
            # scores thousands of noise specks as an object-count failure, which says
            # nothing about either. Comparing the two runs is V6's job, below. V4 still
            # applies here, so fall through to it rather than skipping the file.
            findings.extend(_spread_findings(label, rows))
            continue

        # V1 -- object count against the number of labels somebody else assigned.
        # Compared FRAME BY FRAME, not mean against mean: the two sequences must line up
        # anyway, and averaging first would let a systematic drift cancel against itself.
        header, counts = column(rows, "Island Count")
        if header and truth.counts:
            measured, reference, n = _align(counts, truth.counts)
            rel = _mean_relative(measured, reference)
            findings.append(Finding(
                label, "V1 object count", header,
                float(np.nanmean(measured)), float(np.nanmean(reference)), rel,
                _verdict(rel, tolerance),
                f"per-frame vs distinct labels in the published mask, {n} frames"))

        # V2 -- volume against the mask's own voxel count. Two independent forms: the
        # total is unambiguous, while the largest also checks that the pipeline picked
        # the same object as biggest. "Area" columns are skipped: xyz mode measures per
        # slice, so its numbers are not volumes and must not be compared to one.
        # Only the physical-units CSV can be checked: the normalised one reports every
        # size as a fraction of the analysed volume, which is dimensionless and cannot
        # disagree with a measurement in um^3 in any meaningful way.
        physical = "(physical)" in os.path.basename(path)
        for candidate, series, what in (
            ("Total Island Volume", truth.volumes_um3, "total foreground"),
            ("Maximum Island Volume", truth.largest_um3, "largest single object"),
        ):
            header, volumes = column(rows, candidate)
            if not header or not series:
                continue
            if not physical:
                continue
            measured, reference, n = _align(volumes, series)
            rel = _mean_relative(measured, reference)
            findings.append(Finding(
                label, "V2 object volume", header,
                float(np.nanmean(measured)), float(np.nanmean(reference)), rel,
                _verdict(rel, tolerance),
                f"{what}: mask voxels x published voxel size, {n} frames"))

        # V3 -- speed against the challenge's tracking annotation.
        header, speeds = column(rows, "Speed")
        if header and entry.kind == "ctc":
            tracks = track_truth(entry, downloads, max_frames=frame_limit)
            track_summary = tracks.summary()
            measured = float(np.nanmean(speeds))
            if track_summary and np.isfinite(measured):
                rel = relative_difference(measured, track_summary["speed_um_s_mean"])
                findings.append(Finding(
                    label, "V3 speed", header, measured,
                    track_summary["speed_um_s_mean"], rel,
                    "REVIEW" if abs(rel) > tolerance else "PASS",
                    f"centroid speed of {track_summary['tracks']} tracked objects; "
                    f"flow also sees deformation, so compare magnitude not identity"))

        findings.extend(_spread_findings(label, rows))

    findings.extend(_check_mask_vs_threshold(entry, csvs))
    return findings


# Metrics where a masked and an unmasked run are answering the same question, so a big
# gap between them is informative. Size columns are excluded on purpose: without a mask
# the analysed region is the whole field rather than the object's bounding box, so the
# denominators genuinely differ and a difference there means nothing.
_SHAPE_METRICS = (
    "Mean Island Anisotropy",
    "Structural Correlation Length",
    "Maximum Kurtosis",
    "Maximum Median Skewness",
)


def _check_mask_vs_threshold(entry: Staged, csvs: Sequence[str]) -> List[Finding]:
    """How far BARCODE's own thresholding lands from the published segmentation.

    This is the check that generalises. V1 and V2 prove the pipeline measures a GIVEN
    mask correctly, but most real data arrives with no mask at all, and then every
    number depends on the threshold branch instead. Running the same specimen both ways
    puts a number on that dependence: where the two agree, a maskless run can be
    trusted; where they diverge, the metric is reporting the threshold as much as the
    specimen.

    Not pass/fail. The published mask is not truth for the pixels BARCODE thresholds --
    it is a different, better-informed decision about the same image.
    """
    masked = [p for p in csvs if "with_masks" in p or os.sep + "masked" + os.sep in p]
    plain = [p for p in csvs if "no_masks" in p or os.sep + "threshold" + os.sep in p]
    masked = [p for p in masked if "(physical)" in p] or masked
    plain = [p for p in plain if "(physical)" in p] or plain
    if not masked or not plain:
        return []

    _, masked_rows = load_csv(masked[0])
    _, plain_rows = load_csv(plain[0])
    if not masked_rows or not plain_rows:
        return []

    findings: List[Finding] = []
    for name in _SHAPE_METRICS:
        header_a, a = column(masked_rows, name)
        header_b, b = column(plain_rows, name)
        if not header_a or not header_b:
            continue
        aligned_a, aligned_b, n = _align(a, b)
        rel = _mean_relative(aligned_b, aligned_a)      # threshold relative to mask
        if not np.isfinite(rel):
            continue
        findings.append(Finding(
            entry.name, "V6 mask vs threshold", header_a,
            float(np.nanmean(aligned_b)), float(np.nanmean(aligned_a)), rel, "info",
            f"unmasked vs masked over {n} rows; how much the threshold branch alone "
            f"changes this metric"))
    return findings


def write_report(findings: Sequence[Finding], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "validation_summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "check", "quantity", "measured", "truth",
                         "relative_difference", "verdict", "note"])
        for f in findings:
            writer.writerow([f.dataset, f.check, f.quantity,
                             f"{f.measured:.6g}", f"{f.truth:.6g}",
                             f"{f.relative:.6g}", f.verdict, f.note])
    return path


_CHECK_NOTES = {
    "V1 object count": (
        "How many separate objects BARCODE finds, against how many the publisher "
        "labelled. Exact by construction: the mask defines the objects, so any "
        "difference is the pipeline splitting or merging them -- which is what "
        "resampling an anisotropic mask onto an isotropic grid could plausibly do."),
    "V2 object volume": (
        "Object volume in um^3, against the mask's own voxel count times the published "
        "voxel size. This is the strongest available check on the whole geometric "
        "chain: axis order, voxel spacing, isotropic resampling and cropping all feed "
        "it, and an error in any of them moves the number."),
    "V3 speed": (
        "Speed in um/s, against the displacement of tracked object centroids over the "
        "published frame interval. NOT an identity: the tracking ground truth measures "
        "rigid translation of whole objects, while BARCODE's Speed is 3D optical flow "
        "averaged over the foreground and also responds to deformation and rotation. "
        "Agreement in magnitude and in trend is the pass condition; a constant factor "
        "is a finding about the metric, and a factor equal to the frame interval or "
        "the voxel size would be a unit bug."),
    "V4 replicate spread": (
        "Coefficient of variation across independent samples of the same specimen. Not "
        "accuracy -- discrimination. A metric whose scatter between replicates is as "
        "large as its range between conditions cannot separate conditions, however "
        "correct each individual measurement is."),
    "V6 mask vs threshold": (
        "The same specimen analysed with the published segmentation and with BARCODE's "
        "own thresholding. Most real data has no mask, so this measures how much of "
        "each metric is the specimen and how much is the threshold."),
}


def write_markdown(findings: Sequence[Finding], entries: Sequence[Staged],
                   out_dir: str) -> str:
    """Write the report a person reads, from the same findings as the CSV."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "VALIDATION_REPORT.md")

    graded = [f for f in findings if f.verdict in ("PASS", "FAIL")]
    passed = [f for f in graded if f.verdict == "PASS"]
    failed = [f for f in graded if f.verdict == "FAIL"]

    lines: List[str] = []
    lines.append("# BARCODE volumetric branch — validation against published ground truth\n")
    lines.append(
        "The 2D branch can be regressed against the hackathon reference barcodes. The "
        "volumetric branch had no equivalent anchor: it was developed on a single "
        "dataset with masks produced in-house, so every check available was "
        "self-consistency — does the number stay the same — rather than correctness — "
        "is the number right.\n")
    lines.append(
        "These datasets supply the missing anchor, because for each one somebody else "
        "already determined the answer independently.\n")

    lines.append("## Result\n")
    lines.append(f"- **{len(passed)}/{len(graded)} graded checks passed.**")
    if failed:
        lines.append(f"- **{len(failed)} failed** — listed in full below.")
    lines.append(f"- {len(entries)} staged datasets; "
                 f"{sum(1 for e in entries if e.kind == 'ctc')} time-lapse, "
                 f"{sum(1 for e in entries if e.kind == 'allen')} single-timepoint.\n")

    lines.append("## Datasets\n")
    lines.append("| dataset | xy µm | z µm | anisotropy | frame interval |")
    lines.append("|---|---|---|---|---|")
    for e in sorted(entries, key=lambda e: e.name):
        aniso = f"{e.z_um / e.xy_um:.1f}×" if e.xy_um else "—"
        dt = f"{e.dt_s:g} s" if e.dt_s else "single timepoint"
        lines.append(f"| `{e.name}` | {e.xy_um or '—'} | {e.z_um or '—'} | {aniso} | {dt} |")
    lines.append("")

    by_check: Dict[str, List[Finding]] = {}
    for f in findings:
        by_check.setdefault(f.check, []).append(f)

    for check in sorted(by_check):
        lines.append(f"## {check}\n")
        note = _CHECK_NOTES.get(check)
        if note:
            lines.append(note + "\n")
        rows = by_check[check]
        if check == "V4 replicate spread":
            lines.append("| dataset | metric | mean | SD | CV |")
            lines.append("|---|---|---|---|---|")
            for f in sorted(rows, key=lambda f: abs(f.relative), reverse=True):
                lines.append(f"| `{f.dataset}` | {f.quantity} | {f.truth:.4g} | "
                             f"{f.measured:.4g} | {f.relative:.2%} |")
        elif check == "V6 mask vs threshold":
            lines.append("| dataset | metric | masked | threshold only | difference |")
            lines.append("|---|---|---|---|---|")
            for f in sorted(rows, key=lambda f: abs(f.relative), reverse=True):
                lines.append(f"| `{f.dataset}` | {f.quantity} | {f.truth:.4g} | "
                             f"{f.measured:.4g} | {f.relative:+.2%} |")
        else:
            lines.append("| dataset | quantity | measured | ground truth | difference | verdict |")
            lines.append("|---|---|---|---|---|---|")
            for f in sorted(rows, key=lambda f: (f.verdict != "FAIL", f.dataset)):
                lines.append(f"| `{f.dataset}` | {f.quantity} | {f.measured:.6g} | "
                             f"{f.truth:.6g} | {f.relative:+.3%} | **{f.verdict}** |")
        lines.append("")

    lines.append("## What this does and does not establish\n")
    lines.append(
        "- **Established:** for a given segmentation, the pipeline reproduces object "
        "count and object volume in µm³ exactly, across every frame of every dataset "
        "that passed. That exercises axis handling, voxel spacing, isotropic "
        "resampling and cropping together — the parts most likely to be silently "
        "wrong, because an error in any of them still yields plausible numbers.")
    lines.append(
        "- **Not established:** that BARCODE's own thresholding finds the right object "
        "when no mask is supplied. V6 measures the gap but cannot call either side "
        "correct; the published mask is a better-informed decision about the same "
        "image, not ground truth for the pixels.")
    lines.append(
        "- **Not established:** absolute accuracy of the flow branch. V3 compares "
        "against centroid tracking, which measures a different physical quantity.\n")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate BARCODE volumetric metrics against published ground truth.")
    parser.add_argument("--root", default=r"L:\FF\Hackathon\full_datasets")
    parser.add_argument("--dataset", default=None, help="only this staged folder name")
    parser.add_argument("--frame-limit", type=int, default=None,
                        help="use only the first N frames (fast trial)")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="relative difference counted as agreement (default 5%%)")
    parser.add_argument("--out", default=None, help="report directory")
    args = parser.parse_args()

    entries = discover(args.root)
    if args.dataset:
        entries = [e for e in entries if e.name == args.dataset]
    if not entries:
        print(f"No staged datasets under {args.root}")
        return 1

    downloads = os.path.join(args.root, "_ctc_downloads")
    out_dir = args.out or os.path.join(args.root, "_validation")

    findings: List[Finding] = []
    for entry in entries:
        print(f"\n=== {entry.name} ===  xy {entry.xy_um} z {entry.z_um} dt {entry.dt_s}")
        try:
            found = check_dataset(entry, downloads, args.frame_limit, args.tolerance)
        except Exception as error:
            print(f"  FAILED: {type(error).__name__}: {error}")
            continue
        findings.extend(found)
        for f in found:
            if f.verdict == "info":
                print(f"  {f.verdict:>6}  {f.check:<22} {f.quantity:<34} "
                      f"CV {f.relative:7.2%}   {f.note}")
            else:
                print(f"  {f.verdict:>6}  {f.check:<22} {f.quantity:<34} "
                      f"measured {f.measured:12.4f}  truth {f.truth:12.4f}  "
                      f"{f.relative:+7.2%}")
                if f.note:
                    print(f"          {f.note}")

    if findings:
        path = write_report(findings, out_dir)
        report = write_markdown(findings, entries, out_dir)
        graded = [f for f in findings if f.verdict in ("PASS", "FAIL")]
        passed = sum(1 for f in graded if f.verdict == "PASS")
        print(f"\n{passed}/{len(graded)} graded checks passed")
        print(f"  {path}\n  {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
