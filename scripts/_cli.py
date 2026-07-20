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
        "--z-start", type=int, default=0, metavar="SLICE",
        help="first z slice to analyse (default 0)",
    )
    group.add_argument(
        "--z-end", type=int, default=0, metavar="SLICE",
        help="one past the last z slice; 0 means to the end, negatives count back",
    )


def add_metric_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("metrics")
    group.add_argument(
        "--component-stats", action="store_true",
        help="add per-object count, size SD, skewness and median (volumetric modes only)",
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
    v.enable_component_stats = getattr(args, "component_stats", False)
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
