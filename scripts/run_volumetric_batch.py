#!/usr/bin/env python3
"""Run the volumetric pipeline over a folder and write a BARCODE Summary CSV.

Exercises the full output path -- the same ``results_to_csv`` the 2D pipeline uses --
and reports cross-frame consistency, which is the strongest available check on real
data: the same cell imaged at consecutive timepoints must not produce wildly different
structural metrics.

    python scripts/run_volumetric_batch.py <folder> [--seg-root ... --seg-template ...]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tifffile

from analysis.volumetric.run import run_volumetric_analysis
from core import BarcodeConfig
from core.modes import MODES
from scripts._cli import default_output_dir
from utils.writer import results_to_csv


def natural_key(path: str):
    """Sort Cell1_2.tif before Cell1_10.tif."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return [int(p) if p.isdigit() else p for p in __import__("re").split(r"(\d+)", stem)]


def build_config(args) -> BarcodeConfig:
    config = BarcodeConfig()
    config.modules.image_binarization = True
    config.modules.intensity_distribution = True
    config.modules.optical_flow = args.flow
    v = config.volumetric
    v.enabled = True
    # These scripts call the volumetric branch directly, bypassing the dispatch in
    # core.pipeline, so the mode must be stated here too -- it is what decides the
    # metric names and which families the CSV carries.
    v.analysis_mode = getattr(args, "mode", "xyzt")
    # Optional metric families, off unless asked for. Single-timepoint volumes (Allen
    # FOVs, any SizeT=1 export) need these: without them the run reports only the
    # metrics that a lone volume mostly cannot fill, and the per-object families --
    # which are exactly what an instance segmentation is FOR -- never appear.
    v.enable_component_stats = getattr(args, "component_stats", False)
    v.enable_packing_topology = getattr(args, "packing", False)
    v.enable_mask_intensity = getattr(args, "mask_intensity", False)
    v.mesh_enabled = getattr(args, "mesh", False)
    v.flow_reliability_percentile = args.flow_reliability
    v.flow_downsample = args.flow_downsample
    v.frame_interval_s = args.frame_interval
    v.threshold_offset = args.threshold_offset
    v.crop_padding_vox = args.crop_padding
    # Either flag turns segmentation on: the default regex/template pair
    # ({stem} -> {stem}_SegMask.tif) already resolves a flat mask folder.
    if args.seg_template or args.seg_root:
        v.segmentation_enabled = True
        v.segmentation_root = args.seg_root or ""
        if args.seg_regex:
            v.segmentation_regex = args.seg_regex
        if args.seg_template:
            v.segmentation_template = args.seg_template
    return config


def summarise(name, values, unit=""):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return f"  {name:32s} all NaN"
    spread = finite.std() / abs(finite.mean()) if finite.mean() else np.nan
    return (
        f"  {name:32s} mean {finite.mean():10.4f}{unit}  sd {finite.std():9.4f}"
        f"  CV {spread:6.2%}  range [{finite.min():.4f}, {finite.max():.4f}]"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--pattern", default="*.tif")
    p.add_argument("--threshold-offset", type=float, default=0.1)
    p.add_argument("--crop-padding", type=int, default=2)
    p.add_argument("--flow", action="store_true",
                   help="run the 3D optical flow branch (needs 7 contiguous timepoints per file)")
    p.add_argument("--flow-reliability", type=float, default=50.0, metavar="PERCENTILE")
    p.add_argument("--flow-downsample", type=int, default=1)
    p.add_argument("--frame-interval", type=float, default=0.0, metavar="SECONDS",
                   help="seconds between timepoints; 0 falls back to the file's "
                        "metadata, which is often wrong. Speed scales inversely with it")
    p.add_argument("--seg-root", default=None)
    p.add_argument("--seg-regex", default=None, help="default: (?P<stem>.+)")
    p.add_argument("--seg-template", default=None)
    p.add_argument("--csv", default=None, help="output CSV path")
    # Volumetric modes only. This script calls run_volumetric_analysis directly, and that
    # function never consults the mode -- dispatch lives in core.pipeline, which is
    # bypassed here. So a 2D mode would not select a 2D analysis; it would run the full 3D
    # analysis and merely rename the output, writing um^3 volumes under "Maximum Island
    # Area" with area units. Use core.pipeline (the GUI, or scripts/run_barcode.py) for
    # xyt and xyz.
    volumetric_modes = [k for k, m in MODES.items() if m.is_volumetric]
    p.add_argument("--mode", default="xyzt", choices=volumetric_modes,
                   help="; ".join(f"{k}: {MODES[k].label}" for k in volumetric_modes))
    p.add_argument("--component-stats", action="store_true",
                   help="add per-object count, size SD, skewness and median")
    p.add_argument("--packing", action="store_true",
                   help="add contact-number statistics (mean, SD, hexagonal fraction). "
                        "Needs an INSTANCE segmentation: in a confluent field "
                        "connectivity labelling fuses every cell into one component")
    p.add_argument("--mask-intensity", action="store_true",
                   help="add per-object in-mask intensity statistics")
    p.add_argument("--mesh", action="store_true",
                   help="add the mesh + curvature columns (needs a segmentation)")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.folder, args.pattern)), key=natural_key)
    if not files:
        print(f"No files matching {args.pattern} in {args.folder}")
        return 1

    config = build_config(args)
    all_results, rows = [], []

    print(f"{'file':16s} {'shape(Z,Y,X)':>18s} {'islands':>8s} {'largest um^3':>13s} "
          f"{'%vol':>8s} {'aniso':>7s} {'corr um':>8s} {'kurtosis':>9s} {'s':>6s}")
    print("-" * 104)

    for path in files:
        started = time.time()
        try:
            results, detail = run_volumetric_analysis(path, config)
        except Exception as exc:
            print(f"{os.path.basename(path):16s} FAILED: {type(exc).__name__}: {exc}")
            continue
        elapsed = time.time() - started

        b = detail.binarization
        largest_um3 = b.island_voxels[0] * b.voxel_volume_um3
        pct = b.island_voxels[0] / b.voxel_count
        row = {
            "file": os.path.basename(path),
            "mask_path": detail.mask_paths[0] if detail.mask_paths else None,
            "shape": detail.shape_zyx,
            "islands": b.island_counts[0],
            "largest_voxels": b.island_voxels[0],
            "largest_um3": largest_um3,
            "pct_volume": pct,
            "anisotropy": results.binarization.island_anisotropy,
            "corr_um": results.binarization.island_correlation_length,
            "kurtosis": results.intensity.max_kurtosis,
            "speed": results.flow.mean_speed,
            "flow_corr_um": results.flow.velocity_correlation_length,
            "seconds": elapsed,
        }
        rows.append(row)
        all_results.append(results)
        print(f"{row['file']:16s} {str(detail.shape_zyx):>18s} {row['islands']:8d} "
              f"{largest_um3:13.3f} {pct:7.2%} {row['anisotropy']:7.3f} "
              f"{row['corr_um']:8.4f} {row['kurtosis']:9.4f} {elapsed:6.2f}")

    if not rows:
        return 1

    if args.csv:
        csv_path = args.csv
    else:
        out_dir = default_output_dir(args.folder, "results", "per_frame")
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "Volumetric Summary.csv")
    results_to_csv(all_results, csv_path, just_metrics=False, physical_units=False,
                   mode=config.volumetric.mode)
    print(f"\nWrote {csv_path}")

    # One barcode row per input file, and a physical-units CSV beside the normalised
    # one. The normalised CSV is what the barcode renders (each column scaled across
    # rows), but every size in it is a fraction of the analysed volume, so only the
    # physical copy can be checked against an external measurement.
    from core.metrics import selection_mask
    from core.results import OPTIONAL_FAMILIES
    from scripts._cli import write_physical_csv
    from visualization.barcode import generate_combined_barcode

    families = {
        f.switch: any(getattr(r, f.attribute, None) is not None
                      and getattr(r, f.attribute).is_populated() for r in all_results)
        for f in OPTIONAL_FAMILIES
    }
    base = os.path.splitext(csv_path)[0]
    write_physical_csv(all_results, base + " (physical).csv",
                       config.volumetric.mode, families)
    if len(all_results) > 1:
        headers = all_results[0].get_headers(
            just_metrics=True, mode=config.volumetric.mode, **families)
        generate_combined_barcode(all_results, base, mode=config.volumetric.mode,
                                  metrics_to_visualize=selection_mask(headers, []))
        print(f"Wrote {base}*.png   ({len(all_results)} rows, one per file)")
    else:
        # One row has no contrast: the barcode normalises each column ACROSS rows, so a
        # single-row image is uniformly mid-scale and says nothing.
        print("Only one result; skipping the barcode (it normalises across rows)")

    print("\n=== spread across the analysed files ===")
    print(summarise("largest object volume (um^3)", [r["largest_um3"] for r in rows]))
    print(summarise("fraction of analysed volume", [r["pct_volume"] for r in rows]))
    print(summarise("island anisotropy", [r["anisotropy"] for r in rows]))
    print(summarise("structural correlation (um)", [r["corr_um"] for r in rows]))
    print(summarise("max kurtosis", [r["kurtosis"] for r in rows]))
    if args.flow:
        # All NaN here is the expected report for per-file runs: a single volume has no
        # time axis, so flow needs --timelapse-style grouping to have anything to solve.
        print(summarise("mean speed (um/s)", [r["speed"] for r in rows]))
        print(summarise("velocity correlation (um)", [r["flow_corr_um"] for r in rows]))

    # A mask that survived resampling and cropping intact must still contain exactly
    # the voxels it had on disk (the mask is already isotropic, so only the crop
    # applies, and the crop is defined by the mask's own bounding box).
    if rows[0]["mask_path"]:
        print("\n=== mask preservation through resample + crop ===")
        worst = 0.0
        for row in rows:
            disk_mask = tifffile.imread(row["mask_path"])
            # Compare like with like. The pipeline figure is the LARGEST object, so the
            # on-disk figure must be the largest object too -- against a multi-object
            # field's total it reports ~95% "loss" on a mask that survived perfectly,
            # because one cell of forty is indeed 1/40th of the foreground.
            labels, counts = np.unique(disk_mask[disk_mask > 0], return_counts=True)
            on_disk = int(counts.max()) if labels.size > 1 else int(counts.sum())
            delta = abs(row["largest_voxels"] - on_disk) / on_disk if on_disk else np.nan
            worst = max(worst, delta)
            flag = "OK" if delta < 1e-9 else f"DIFF {delta:.3%}"
            print(f"  {row['file']:16s} pipeline {int(row['largest_voxels']):>9,d} "
                  f"  on-disk {on_disk:>9,d}  {flag}")
        print(f"  worst relative difference: {worst:.3e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
