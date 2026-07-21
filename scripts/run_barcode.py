#!/usr/bin/env python3
"""Run BARCODE in any analysis mode from the command line.

One entry point for all three modes, so the CLI and the GUI expose the same choices
under the same names.

    # what would this mode measure?
    python scripts/run_barcode.py data/ --mode xyz --list-metrics

    # 2D over depth, middle of the stack only, one row per timepoint
    python scripts/run_barcode.py data/ --mode xyz --z-start 15 --z-end 45

    # 3D over time, with masks, meshing and the per-object statistics
    python scripts/run_barcode.py data/ --mode xyzt --timelapse \\
        --seg-root masks/ --mesh --component-stats

    # trim the barcode without touching the CSV
    python scripts/run_barcode.py data/ --mode xyzt --hide-metric Connectivity \\
        --hide-metric Curl
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")

from core import BarcodeConfig, InputConfig
from core.pipeline import run_analysis
from scripts._cli import (
    add_metric_arguments,
    add_mode_arguments,
    apply_common,
    print_metrics,
)


def build_config(args) -> BarcodeConfig:
    config = BarcodeConfig()
    config.modules.image_binarization = not args.no_binarization
    config.modules.intensity_distribution = not args.no_intensity
    config.modules.optical_flow = args.flow

    config.reader.verbose = args.verbose
    config.reader.accept_dim_images = True
    config.reader.accept_dim_channels = True
    config.writer.generate_barcode = not args.no_barcode
    config.writer.save_visualizations = args.save_graphs
    config.writer.save_rds = args.save_rds
    config.channels.selected_channel = args.channel

    apply_common(config, args)

    v = config.volumetric
    v.timelapse_enabled = args.timelapse
    v.mesh_enabled = args.mesh
    if hasattr(v, "mesh_curvature"):
        v.mesh_curvature = args.mesh and not args.no_curvature
    if args.seg_root or args.seg_template:
        v.segmentation_enabled = True
        v.segmentation_root = args.seg_root or ""
        if args.seg_regex:
            v.segmentation_regex = args.seg_regex
        if args.seg_template:
            v.segmentation_template = args.seg_template
    if args.threshold_offset is not None:
        v.threshold_offset = args.threshold_offset
        config.image_binarization_parameters.threshold_offset = args.threshold_offset
    if args.bin_factor is not None:
        config.image_binarization_parameters.bin_factor = args.bin_factor
    return config


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("path", help="file or folder to analyse")
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--verbose", action="store_true")

    add_mode_arguments(p, default="xyt")
    add_metric_arguments(p)

    branches = p.add_argument_group("branches")
    branches.add_argument("--no-binarization", action="store_true")
    branches.add_argument("--no-intensity", action="store_true")
    branches.add_argument("--flow", action="store_true",
                          help="run the flow branch (ignored in xyz, which has no velocity)")
    branches.add_argument("--mesh", action="store_true",
                          help="surface meshing; needs a segmentation, xyzt only")
    branches.add_argument("--no-curvature", action="store_true")

    seg = p.add_argument_group("segmentation")
    seg.add_argument("--seg-root", default=None)
    seg.add_argument("--seg-regex", default=None)
    seg.add_argument("--seg-template", default=None)

    other = p.add_argument_group("other")
    other.add_argument("--timelapse", action="store_true",
                       help="group per-timepoint files into one series (xyzt)")
    other.add_argument("--threshold-offset", type=float, default=None)
    other.add_argument("--bin-factor", type=int, default=None)
    other.add_argument("--no-barcode", action="store_true")
    other.add_argument("--save-graphs", action="store_true")
    other.add_argument("--save-rds", action="store_true")

    args = p.parse_args()
    config = build_config(args)

    if args.list_metrics:
        print_metrics(config)
        return 0

    if not os.path.exists(args.path):
        print(f"No such file or folder: {args.path}")
        return 1

    mode = config.volumetric.mode
    print(f"mode {mode.key} ({mode.label}) on {args.path}")
    if config.volumetric.z_start or config.volumetric.z_end:
        print(f"  z range [{config.volumetric.z_start}:{config.volumetric.z_end or 'end'}]")
    if args.flow and not mode.supports_flow:
        print(f"  note: {mode.key} has no velocity, so the flow branch is not run")
    if args.mesh and not mode.supports_mesh:
        print(f"  note: {mode.key} analyses planes, so there is no surface to mesh")
    if args.component_stats and not mode.supports_component_stats:
        print(f"  note: per-object statistics are volumetric-only; not added for {mode.key}")

    run_analysis(args.path, config, InputConfig())

    # BARCODE writes the Summary CSV, barcode PNG, Settings YAML and Time log into the
    # folder it processed (utils/setup.py::setup_paths). Say so, because it means a data
    # folder stops being images-only after a run.
    target = args.path if os.path.isdir(args.path) else os.path.dirname(args.path)
    print()
    print(f"outputs written into {target}")
    print("  (BARCODE writes beside the input; move them to a results folder if you "
          "want the data folder kept clean)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
