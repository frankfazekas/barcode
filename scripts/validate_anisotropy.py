#!/usr/bin/env python3
"""Does a metric survive being measured through a coarser z step?

V1 and V2 in ``validate_open_data.py`` show that BARCODE measures a given mask exactly,
but they check each dataset at its own native sampling. They cannot separate "the
geometry handling is right" from "the geometry handling is consistently wrong in a way
that cancels", because there is only one sampling per dataset to compare against.

This is the controlled version. One specimen, one segmentation, imaged through
progressively coarser z: keep every k-th slice and declare the z step k times larger, so
the PHYSICAL object is identical and only the sampling changes. Every physical metric
should therefore be unchanged. Whatever drifts with k is measuring the microscope's z
step rather than the specimen -- which for a barcode meant to compare datasets acquired
on different instruments is the failure that matters most.

This is not hypothetical for this project: the working Jurkat data is 4.6x anisotropic
and ``make_isotropic`` exists precisely to undo that, but nothing so far has tested it
against a known answer.

    python scripts/validate_anisotropy.py --dataset ctc_Fluo-C3DH-A549_01 --factors 1 2 3 4

Writes the decimated copies under ``<root>/_anisotropy/`` (never into the source
dataset), runs each, and reports each metric's drift relative to k=1.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts._staging import mask_z_to_isotropic, read_tiff_any, write_volume
from scripts.validate_open_data import column, discover, load_csv

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

# Metrics that describe the specimen in physical units and must not depend on how
# finely it was sampled in z. Change and flow columns are excluded: they need a time
# axis, and this test uses a handful of frames.
TRACKED = (
    "Total Island Volume",
    "Maximum Island Volume",
    "Mean Island Anisotropy",
    "Structural Correlation Length",
    "Island Count",
)


def decimate(source_folder: str, out_root: str, factor: int, xy_um: float, z_um: float,
             frames: Optional[int]) -> str:
    """Build a copy of the dataset sampled every ``factor`` slices in z."""
    name = f"{os.path.basename(source_folder)}_z{factor}"
    folder = os.path.join(out_root, name)
    data_out = os.path.join(folder, "BARCODE", "data")
    mask_out = os.path.join(folder, "BARCODE", "masks")
    for path in (data_out, mask_out, os.path.join(folder, "BARCODE", "results")):
        os.makedirs(path, exist_ok=True)

    images = sorted(glob.glob(os.path.join(source_folder, "BARCODE", "data", "*.tif")))
    if frames:
        images = images[:frames]

    new_z = z_um * factor
    for image_path in images:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = os.path.join(source_folder, "BARCODE", "masks", f"{stem}_SegMask.tif")
        if not os.path.isfile(mask_path):
            continue

        volume = read_tiff_any(image_path)[::factor]
        write_volume(os.path.join(data_out, f"{stem}.tif"), volume, xy_um, new_z)

        # The mask has to be degraded the same way, or the test measures interpolation
        # of the image against a mask that still knows the answer. Take the staged
        # isotropic mask down to the coarse ACQUIRED grid, then back up to isotropic --
        # exactly the round trip a mask segmented on coarse data would have gone
        # through. Nearest-neighbour throughout: these are labels.
        mask_iso = read_tiff_any(mask_path)
        # Decimate by taking every factor-th plane, matching how the image above was
        # decimated -- NOT by an endpoint-anchored linspace, whose pitch is
        # (n-1)/(m-1) and silently rescales the object (see _staging).
        coarse = mask_iso[::factor] if mask_iso.shape[0] >= volume.shape[0] else mask_iso
        coarse = coarse[:volume.shape[0]] if coarse.shape[0] > volume.shape[0] else coarse
        write_volume(os.path.join(mask_out, f"{stem}_SegMask.tif"),
                     mask_z_to_isotropic(coarse, new_z, xy_um), xy_um, xy_um)
    return folder


def _mask_volume(folder: str, xy_um: float) -> float:
    """Mean foreground volume of the decimated masks, straight from the files."""
    paths = sorted(glob.glob(os.path.join(folder, "BARCODE", "masks", "*.tif")))
    if not paths:
        return float("nan")
    voxel = xy_um ** 3
    return float(np.mean([float((read_tiff_any(p) > 0).sum()) * voxel for p in paths]))


def run_batch(folder: str) -> Optional[str]:
    csv_path = os.path.join(folder, "BARCODE", "results", "aniso", "Summary.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    log = os.path.join(folder, "run.log")
    with open(log, "w", encoding="utf-8") as handle:
        completed = subprocess.run(
            [PYTHON, os.path.join(HERE, "run_volumetric_batch.py"),
             os.path.join(folder, "BARCODE", "data"),
             "--seg-root", os.path.join(folder, "BARCODE", "masks"),
             "--mode", "xyzt", "--component-stats", "--csv", csv_path],
            stdout=handle, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        print(f"    run failed, see {log}")
        return None
    physical = os.path.splitext(csv_path)[0] + " (physical).csv"
    return physical if os.path.isfile(physical) else csv_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check that physical metrics do not depend on the z sampling.")
    parser.add_argument("--root", default=r"L:\FF\Hackathon\full_datasets\_open_data")
    parser.add_argument("--dataset", required=True, help="staged folder name")
    parser.add_argument("--factors", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--frames", type=int, default=6,
                        help="how many volumes to use (default 6; this is a controlled "
                             "comparison, not a survey)")
    parser.add_argument("--tolerance", type=float, default=0.10)
    parser.add_argument("--keep", action="store_true", help="keep the decimated copies")
    args = parser.parse_args()

    entries = {e.name: e for e in discover(args.root)}
    entry = entries.get(args.dataset)
    if entry is None:
        print(f"{args.dataset} not found under {args.root}")
        return 1
    if not (entry.xy_um and entry.z_um):
        print(f"{args.dataset}: no geometry in README.txt; cannot decimate meaningfully")
        return 1

    out_root = os.path.join(args.root, "_anisotropy")
    os.makedirs(out_root, exist_ok=True)
    print(f"{entry.name}: native xy {entry.xy_um} um, z {entry.z_um} um "
          f"({entry.z_um / entry.xy_um:.1f}x anisotropic), {args.frames} volumes\n")

    measured: Dict[int, Dict[str, float]] = {}
    folders: List[str] = []
    for factor in sorted(args.factors):
        print(f"  z step x{factor}  ->  {entry.z_um * factor:g} um "
              f"({entry.z_um * factor / entry.xy_um:.1f}x anisotropic)")
        folder = decimate(entry.folder, out_root, factor, entry.xy_um, entry.z_um,
                          args.frames)
        folders.append(folder)
        path = run_batch(folder)
        if not path:
            continue
        _, rows = load_csv(path)
        values: Dict[str, float] = {}
        for name in TRACKED:
            header, series = column(rows, name)
            if header and series.size:
                finite = series[np.isfinite(series)]
                if finite.size:
                    values[name] = float(finite.mean())
        # The decimated mask is a different object from the original -- coarser z makes
        # it blockier -- so drift with k has two causes that must not be confused: the
        # specimen's REPRESENTATION genuinely changed, and the pipeline may have
        # measured it wrong. This is the first cause, measured directly from the
        # decimated mask. Subtracting it leaves only the second.
        values["_mask truth volume"] = _mask_volume(folder, xy_um=entry.xy_um)
        measured[factor] = values

    baseline = measured.get(min(args.factors))
    if not baseline:
        print("\nno baseline result; nothing to compare")
        return 1

    print(f"\n{'metric':<32}" + "".join(f"{'x' + str(k):>14}" for k in sorted(measured)))
    print("-" * (32 + 14 * len(measured)))
    verdicts: List[Tuple[str, float, str]] = []
    for name in list(TRACKED) + ["_mask truth volume"]:
        if name not in baseline:
            continue
        row = f"{name:<32}"
        worst = 0.0
        for factor in sorted(measured):
            value = measured[factor].get(name, np.nan)
            row += f"{value:>14.4g}"
            if np.isfinite(value) and baseline[name]:
                worst = max(worst, abs(value - baseline[name]) / abs(baseline[name]))
        print(row)
        verdicts.append((name, worst,
                         "PASS" if worst <= args.tolerance else "DRIFTS"))

    # Drift across k has two causes and only one of them is a defect. Separate them:
    # the pipeline's error is measured volume vs the volume of the SAME decimated mask;
    # the rest is the specimen's representation genuinely changing as z coarsens.
    print(f"\n{'metric':<32}{'drift vs x1':>14}{'pipeline error':>17}   verdict")
    for name, worst, _ in verdicts:
        error = ""
        verdict = "PASS" if worst <= args.tolerance else "sampling-dependent"
        # Only the TOTAL has a matching truth here: "_mask truth volume" sums the whole
        # foreground, so checking the largest single object against it reports a ~90%
        # error on a 29-object field where nothing is wrong.
        if name == "Total Island Volume":
            gaps = [
                abs(measured[k][name] - measured[k]["_mask truth volume"])
                / abs(measured[k]["_mask truth volume"])
                for k in sorted(measured)
                if name in measured[k] and measured[k].get("_mask truth volume")
            ]
            if gaps:
                worst_error = max(gaps)
                error = f"{worst_error:>16.2%}"
                verdict = "EXACT" if worst_error < 1e-6 else (
                    "PASS" if worst_error <= args.tolerance else "FAIL")
        print(f"{name:<32}{worst:>13.2%}{error:>17}   {verdict}")

    print("\nRead this as: 'drift vs x1' is how much the number changes when the same\n"
          "specimen is sampled more coarsely in z -- a property of the acquisition.\n"
          "'pipeline error' is measured volume against the volume of the very mask it\n"
          "was given at that sampling. EXACT there means the drift is information lost\n"
          "in the microscope, not arithmetic lost in BARCODE.")

    report = os.path.join(args.root, "_validation", "anisotropy_invariance.csv")
    os.makedirs(os.path.dirname(report), exist_ok=True)
    with open(report, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "metric"] + [f"z_x{k}" for k in sorted(measured)]
                        + ["worst_drift", "verdict"])
        for name, worst, verdict in verdicts:
            writer.writerow([entry.name, name]
                            + [measured[k].get(name, "") for k in sorted(measured)]
                            + [f"{worst:.6g}", verdict])
    print(f"\nwrote {report}")

    if not args.keep:
        for folder in folders:
            shutil.rmtree(folder, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
