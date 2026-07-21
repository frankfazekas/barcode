"""Shared command-line options for the volumetric scripts.

One definition of each option so the scripts cannot drift apart, and so every setting
reachable in the GUI is reachable from the command line under the same name.
"""
from __future__ import annotations

import argparse
from typing import List

from core import BarcodeConfig
from core.modes import MODES


def add_mode_arguments(parser: argparse.ArgumentParser, default: str = "xyzt") -> None:
    group = parser.add_argument_group("analysis mode")
    group.add_argument(
        "--mode", default=default, choices=list(MODES),
        help="; ".join(f"{k}: {m.label}" for k, m in MODES.items()),
    )
    group.add_argument(
        "--z-units", default="acquired", choices=["acquired", "isotropic", "microns"],
        help="how --z-start/--z-end are read. 'acquired' indexes the stack as acquired; "
             "'isotropic' indexes the finer isotropic grid a mask lives on; 'microns' is "
             "physical depth, which cannot be misread either way (default: acquired)",
    )
    group.add_argument(
        "--t-units", default="index", choices=["index", "seconds"],
        help="how --t-start/--t-end are read: timepoint index, or seconds via the "
             "exposure time (default: index)",
    )
    group.add_argument(
        "--t-start", type=float, default=0, metavar="POS",
        help="first timepoint to analyse, in --t-units (default 0)",
    )
    group.add_argument(
        "--t-end", type=float, default=0, metavar="POS",
        help="end of the timepoint range; 0 means to the end",
    )
    group.add_argument(
        "--z-start", type=float, default=0, metavar="POS",
        help="start of the z range, in --z-units (default 0)",
    )
    group.add_argument(
        "--z-end", type=float, default=0, metavar="POS",
        help="end of the z range, in --z-units; 0 means to the end",
    )

    group = parser.add_argument_group("file layout")
    group.add_argument(
        "--axes", default="", metavar="ORDER",
        help="true axis order (e.g. TZYX), one letter per data dimension, for a file "
             "whose header is wrong. Acquisition software writing a time series into "
             "ImageJ's 'channels' field is the common case. Default: trust the file",
    )
    group.add_argument(
        "--xy-step", type=float, default=0, metavar="UM",
        help="microns per pixel in xy, overriding the file's XResolution tag",
    )
    group.add_argument(
        "--z-step", type=float, default=0, metavar="UM",
        help="microns between z slices, overriding the file's ImageJ 'spacing'",
    )


def add_metric_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("metrics")
    group.add_argument(
        "--component-stats", action="store_true",
        help="add per-object count, size SD, skewness and median (volumetric modes only)",
    )
    group.add_argument(
        "--intensity-in-mask", action="store_true",
        help="build the intensity distribution from voxels INSIDE the segmentation only. "
             "Removes the background peak, which changes the metrics substantially -- "
             "masked and unmasked runs are not comparable with each other.",
    )
    group.add_argument(
        "--hide-metric", action="append", default=[], metavar="NAME",
        help="leave a metric off the barcode image (repeatable). The CSV always keeps "
             "the full set for the mode.",
    )
    group.add_argument(
        "--list-metrics", action="store_true",
        help="print the metrics this mode would produce, then exit",
    )


def apply_common(config: BarcodeConfig, args) -> BarcodeConfig:
    """Copy the shared options onto a config."""
    v = config.volumetric
    v.analysis_mode = getattr(args, "mode", v.analysis_mode)
    v.z_start = getattr(args, "z_start", 0)
    v.z_end = getattr(args, "z_end", 0)
    v.z_range_units = getattr(args, "z_units", "acquired")
    v.t_start = getattr(args, "t_start", 0)
    v.t_end = getattr(args, "t_end", 0)
    v.t_range_units = getattr(args, "t_units", "index")
    v.axes_override = getattr(args, "axes", "") or ""
    v.xy_step_um = getattr(args, "xy_step", 0) or v.xy_step_um
    v.z_step_um = getattr(args, "z_step", 0) or v.z_step_um
    v.enable_component_stats = getattr(args, "component_stats", False)
    v.intensity_use_mask = getattr(args, "intensity_in_mask", False)
    config.writer.hidden_barcode_metrics = list(getattr(args, "hide_metric", []))
    return config


def print_metrics(config: BarcodeConfig) -> None:
    """Show what this configuration will produce, and what the barcode will show."""
    from core.metrics import selection_mask
    from core.results import ChannelResults

    mode = config.volumetric.mode
    headers = ChannelResults.get_headers(
        just_metrics=True, mode=mode,
        include_components=config.volumetric.enable_component_stats,
    )
    shown = selection_mask(headers, config.writer.hidden_barcode_metrics)

    print(f"mode {mode.key} -- {mode.label}")
    print(f"  {mode.description}")
    print(f"\n{len(headers)} metric(s) in the CSV; {sum(shown)} shown on the barcode:\n")
    for header, visible in zip(headers, shown):
        print(f"   {'x' if visible else ' '}  {header}")
    if not all(shown):
        print("\n('x' = on the barcode; all of them are still in the CSV)")
