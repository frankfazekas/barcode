"""Shared command-line options for the volumetric scripts.

One definition of each option so the scripts cannot drift apart, and so every setting
reachable in the GUI is reachable from the command line under the same name.
"""
from __future__ import annotations

import argparse
import os
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
        "--rows", default="auto",
        choices=["auto", "file", "timepoint", "slice", "object"],
        help="what ONE ROW of the barcode is. The barcode normalises per column across "
             "rows, so this is what decides what is compared. 'auto' picks from the "
             "data -- many objects -> object, else many timepoints -> timepoint, else "
             "file -- and prints its choice (default: auto)",
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
    group.add_argument(
        "--mask-spacing", type=float, default=0, metavar="UM",
        help="voxel spacing of the segmentation, isotropic. 0 means 'isotropic at the "
             "image xy step', which is right for masks exported on a finer isotropic "
             "grid but WRONG for a mask on the acquired grid -- pass the image's z step "
             "for those, or the mask is resampled to the wrong physical depth",
    )
    group.add_argument(
        "--frame-interval", type=float, default=0, metavar="SECONDS",
        help="seconds between timepoints. Only Speed and Speed Change depend on it, but "
             "they are wrong by exactly this factor if it is left unset: ImageJ's "
             "'finterval' often describes the z acquisition, not the time axis, and the "
             "fallback of 1 s is then reported in um/s as if it were real",
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
        "--curvature-range", action="store_true",
        help="add the minimum and maximum curvature. <H> averages the two principal "
             "curvatures together, so a saddle reads as flat; these do not. Needs the "
             "mesh family",
    )
    group.add_argument(
        "--slice-profile", action="store_true",
        help="add the maximal area slice (index, depth, area) and raise flag digit 6 when "
             "foreground reaches an edge of the analysed field. The only metrics that "
             "say WHERE in depth something is",
    )
    group.add_argument(
        "--mask-intensity", action="store_true",
        help="add per-object in-mask intensity statistics: MFI, SD, CV, skewness, "
             "entropy, normalized entropy and the fraction above twice the median. "
             "Only ENTROPY uses a per-object [0,1] rescaling, which it needs so every "
             "object is binned over the same range; the rest are computed on raw voxels, "
             "because CV and skewness are already scale-invariant and the bright fraction "
             "degenerates on rescaled values for punctate objects. MFI and SD are "
             "therefore in detector units, the rest dimensionless. Needs a segmentation",
    )
    group.add_argument(
        "--packing", action="store_true",
        help="add contact-number statistics (mean, SD, hexagonal fraction): who touches "
             "whom. Needs an INSTANCE segmentation -- in a confluent field connectivity "
             "labelling fuses every cell into one component, so this reports NaN with a "
             "reason rather than a misleading 0",
    )
    group.add_argument(
        "--object-mesh", action="store_true",
        help="mesh EVERY object rather than only the largest, giving per-object "
             "sphericity, solidity, concavity, aspect ratio and curvature. Without it "
             "those columns are empty for object rows, because a whole-field mesh "
             "describes one cell. Costs ~2.5 s per object",
    )
    group.add_argument(
        "--object-mesh-limit", type=int, default=0, metavar="N",
        help="mesh only the first N objects (0 = all). For iterating on settings without "
             "paying ~35 min a field; the rest get NaN shape columns and the run says so",
    )
    group.add_argument(
        "--fingerprint", action="store_true",
        help="write a per-volume card (projections, grouped metrics, distributions). "
             "For inspecting ONE run closely; the barcode is the tool for comparing "
             "many, so this is off unless asked for",
    )
    group.add_argument(
        "--list-metrics", action="store_true",
        help="print the metrics this mode would produce, then exit",
    )


def apply_common(config: BarcodeConfig, args) -> BarcodeConfig:
    """Copy the shared options onto a config."""
    v = config.volumetric
    v.analysis_mode = getattr(args, "mode", v.analysis_mode)
    v.row_axis = getattr(args, "rows", "auto")
    v.z_start = getattr(args, "z_start", 0)
    v.z_end = getattr(args, "z_end", 0)
    v.z_range_units = getattr(args, "z_units", "acquired")
    v.t_start = getattr(args, "t_start", 0)
    v.t_end = getattr(args, "t_end", 0)
    v.t_range_units = getattr(args, "t_units", "index")
    v.axes_override = getattr(args, "axes", "") or ""
    v.xy_step_um = getattr(args, "xy_step", 0) or v.xy_step_um
    v.z_step_um = getattr(args, "z_step", 0) or v.z_step_um
    v.mask_spacing_um = getattr(args, "mask_spacing", 0) or v.mask_spacing_um
    v.enable_component_stats = getattr(args, "component_stats", False)
    v.enable_curvature_range = getattr(args, "curvature_range", False)
    v.enable_slice_profile = getattr(args, "slice_profile", False)
    v.enable_mask_intensity = getattr(args, "mask_intensity", False)
    v.enable_packing_topology = getattr(args, "packing", False)
    v.write_fingerprint = getattr(args, "fingerprint", False)
    v.object_mesh = getattr(args, "object_mesh", False)
    if v.object_mesh:
        # Curvature is cheap next to the meshing itself, and Mean Curvature <H> is one of
        # the per-object columns -- leaving it empty after paying for the meshes is worse
        # than the few percent it costs.
        v.mesh_curvature = True
    v.object_mesh_limit = getattr(args, "object_mesh_limit", 0)
    v.frame_interval_s = getattr(args, "frame_interval", 0) or v.frame_interval_s
    v.intensity_use_mask = getattr(args, "intensity_in_mask", False)
    config.writer.hidden_barcode_metrics = list(getattr(args, "hide_metric", []))
    return config


def default_output_dir(input_path: str, *parts: str) -> str:
    """A results directory beside ``input_path``, always on the data drive.

    ``os.path.dirname(os.path.normpath(folder))`` -- what these scripts used to do -- is
    the empty string for a relative argument like ``data`` or a bare filename, so
    ``os.path.join("", "results", ...)`` resolved against the current working directory.
    Run from the repo, that wrote CSVs, barcodes and figures onto the C: drive, which this
    project forbids: C: holds code, the data drives hold data. Resolving to an absolute
    path first puts the output beside the input it came from, wherever that lives.

    A C: destination is still reachable if the *input* is on C:, so that is called out
    rather than silently accepted.
    """
    base = os.path.dirname(os.path.abspath(input_path))
    out_dir = os.path.join(base, *parts)
    drive = os.path.splitdrive(os.path.abspath(out_dir))[0].upper()
    if drive == "C:":
        print(
            f"Warning: outputs would be written to {out_dir}, on the C: drive. Analysis "
            f"outputs belong next to their data; pass an explicit output path on the "
            f"data drive.",
            flush=True,
        )
    return out_dir


def write_physical_csv(results, path: str, mode, families: dict) -> None:
    """Write results in physical units (um, um^3, um/s) next to the normalised CSV.

    ``utils.writer.results_to_csv`` cannot do this for the time-lapse runner. Its
    physical path asserts that every row's third binarization value is NaN in normalised
    form -- a stand-in for "this row's change metrics were never filled" -- and that
    runner deliberately fills the change metrics afterwards from the series. The
    assertion is right about the ordinary path and simply does not describe this one.

    The distinction matters for validation: the normalised CSV reports every size as a
    fraction of the analysed volume, so it is dimensionless and cannot be compared with
    an externally measured volume. Only this file can be checked against ground truth.
    """
    import csv

    # mode matters here as much as it does for the normalised headers: without it a 3D
    # run is labelled with the 2D names and every volume column claims to be an area.
    headers = results[0].get_physical_headers(mode=mode, **families)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for result in results:
            # get_physical_data, not to_physical_array: the row leads with the file path
            # and flags, which are strings, and coercing the row to a float array to
            # write it as text just fails on the first column.
            writer.writerow(list(result.get_physical_data(mode=mode, **families)))


def family_switches(config: BarcodeConfig) -> dict:
    """Which optional families this configuration will produce.

    The config field names do not map mechanically onto the registry's switch names, so
    the correspondence is written out. Every entry in ``OPTIONAL_FAMILIES`` must appear:
    the assertion below fails loudly when a family is added and this is not updated,
    rather than letting ``--list-metrics`` quietly under-report what the run will emit.
    """
    from core.results import OPTIONAL_FAMILIES

    v = config.volumetric
    switches = {
        "include_mesh": bool(getattr(v, "mesh_enabled", False)),
        "include_components": bool(v.enable_component_stats),
        "include_intensity_magnitude": bool(v.enable_intensity_magnitude),
        "include_ranges": bool(v.record_range_columns),
        "include_packing": bool(v.enable_packing_topology),
        "include_curvature_range": bool(v.enable_curvature_range),
        "include_slice_profile": bool(v.enable_slice_profile),
        "include_mask_intensity": bool(v.enable_mask_intensity),
    }
    missing = {f.switch for f in OPTIONAL_FAMILIES} - set(switches)
    if missing:
        raise AssertionError(
            f"family_switches is missing {sorted(missing)}; add the config field that "
            f"turns each on so --list-metrics matches what a run emits."
        )
    return switches


def print_metrics(config: BarcodeConfig) -> None:
    """Show what this configuration will produce, and what the barcode will show."""
    from core.metrics import selection_mask
    from core.results import ChannelResults

    mode = config.volumetric.mode
    headers = ChannelResults.get_headers(
        just_metrics=True, mode=mode, **family_switches(config))
    shown = selection_mask(headers, config.writer.hidden_barcode_metrics)

    print(f"mode {mode.key} -- {mode.label}")
    print(f"  {mode.description}")
    print(f"\n{len(headers)} metric(s) in the CSV; {sum(shown)} shown on the barcode:\n")
    for header, visible in zip(headers, shown):
        print(f"   {'x' if visible else ' '}  {header}")
    if not all(shown):
        print("\n('x' = on the barcode; all of them are still in the CSV)")
