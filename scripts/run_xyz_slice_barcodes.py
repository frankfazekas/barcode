#!/usr/bin/env python3
"""One barcode per timepoint, with one row per z-slice.

2D analysis of each xy slice, read as a function of depth. Each output barcode covers a
single timepoint: rows are z-slices from the bottom of the range to the top, columns are
the 2D metrics. Reading down a column shows how that metric varies with depth.

Z indices refer to **acquired** slices, before any isotropic resampling -- so on data
acquired at 0.3 um, --z-start 12 --z-end 46 means 34 slices covering 10.2 um.

    python scripts/run_xyz_slice_barcodes.py <folder-or-file> --z-start 12 --z-end 46
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")

import numpy as np

from analysis.volumetric.perslice import run_per_slice_analysis
from analysis.volumetric.timelapse import group_timelapse
from core import BarcodeConfig
from core.metrics import selection_mask
from core.modes import get_mode
from core.results import ChannelResults
from utils.writer import generate_combined_barcode, results_to_csv


def natural_key(path):
    import re
    stem = os.path.splitext(os.path.basename(path))[0]
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", stem)]


def build_config(args) -> BarcodeConfig:
    config = BarcodeConfig()
    config.modules.image_binarization = not args.no_binarization
    config.modules.intensity_distribution = not args.no_intensity
    config.modules.optical_flow = False
    v = config.volumetric
    v.analysis_mode = "xyz"
    v.z_start, v.z_end = args.z_start, args.z_end
    v.z_range_units = args.z_units
    config.image_binarization_parameters.bin_factor = args.bin_factor
    config.image_binarization_parameters.threshold_offset = args.threshold_offset
    return config


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="file or folder of z-stacks")
    p.add_argument("--pattern", default="*.tif")
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--z-units", default="acquired",
                   choices=["acquired", "isotropic", "microns"],
                   help="how --z-start/--z-end are read (default: acquired slices)")
    p.add_argument("--z-start", type=float, default=0, help="start of the z range")
    p.add_argument("--z-end", type=float, default=0, help="end; 0 = to the end")
    p.add_argument("--slice-step", type=int, default=1,
                   help="analyse every Nth slice (default every one)")
    p.add_argument("--bin-factor", type=int, default=1)
    p.add_argument("--threshold-offset", type=float, default=0.1)
    p.add_argument("--hide-metric", action="append", default=[], metavar="NAME")
    p.add_argument("--no-binarization", action="store_true")
    p.add_argument("--no-intensity", action="store_true")
    p.add_argument("--out", default=None, help="output folder (default: <parent>/results/xyz_per_slice)")
    args = p.parse_args()

    files = ([args.path] if os.path.isfile(args.path)
             else sorted(glob.glob(os.path.join(args.path, args.pattern)), key=natural_key))
    if not files:
        print(f"No files matching {args.pattern} in {args.path}")
        return 1

    config = build_config(args)
    mode = get_mode("xyz")
    # Default beside the data folder, never inside it -- a single file argument would
    # otherwise drop results into the folder holding the images.
    source_dir = (os.path.dirname(os.path.normpath(args.path))
                  if os.path.isfile(args.path) else os.path.normpath(args.path))
    out_dir = args.out or os.path.join(
        os.path.dirname(source_dir), "results", "xyz_per_slice")
    os.makedirs(out_dir, exist_ok=True)

    headers = ChannelResults.get_headers(just_metrics=True, mode=mode)
    shown = selection_mask(headers, args.hide_metric)
    print(f"{len(files)} file(s); {len(headers)} metrics, {sum(shown)} on each barcode")

    written = 0
    for path in files:
        started = time.time()
        per_timepoint, detail = run_per_slice_analysis(
            path, config, channel=args.channel, slice_step=args.slice_step)

        span = (f"z[{detail.z_range[0]}:{detail.z_range[1]}]" if detail.z_range
                else f"z[0:{detail.n_slices}]")
        depth = detail.n_slices * detail.z_step_um
        for t, rows in enumerate(per_timepoint):
            stem = os.path.splitext(os.path.basename(path))[0]
            suffix = "" if len(per_timepoint) == 1 else f" t{t + 1}"
            base = os.path.join(out_dir, f"{stem}{suffix} per-slice")
            results_to_csv(rows, base + ".csv", just_metrics=False, mode=mode)
            generate_combined_barcode(rows, base, mode=mode, metrics_to_visualize=shown)
            written += 1

        areas = [r.binarization.max_island_size for r in per_timepoint[0]]
        finite = [a for a in areas if a == a]
        print(f"  {os.path.basename(path):16s} {len(per_timepoint[0]):3d} slices "
              f"({span}, {depth:.1f} um)  "
              f"max island area {min(finite):.4f}..{max(finite):.4f}  "
              f"{time.time() - started:.1f}s")

    print(f"\nwrote {written} barcode(s) + CSV(s) to\n  {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
