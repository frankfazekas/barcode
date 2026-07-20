"""xyz mode — for each timepoint, 2D metrics computed over z.

Depth is the progression axis here, not time: each z-slice is analysed as a 2D image and
the Change metrics describe how structure varies *with depth*.

**No new analysis mathematics.** A ``(Z, Y, X)`` array is exactly the shape the existing
2D branch functions already accept, so this module is a reader and a loop. It calls
``analysis.binarization`` and ``analysis.intensity_distribution`` read-only and does not
modify them.

This is also, precisely, what BARCODE has been doing to Z-stacks by accident:
``utils/reader.py`` turns a ``(54,312,303)`` stack into ``(54,312,303,1)`` and the 2D
pipeline analyses the 54 slices as 54 "timepoints". The numbers were right; the labels
were not. Two things are therefore fixed here rather than computed differently:

* **Flow is not run.** Farneback between adjacent focal planes measures how structure
  shifts with depth, in um per um. Reporting that as a velocity in um/s -- which is what
  happens today -- is meaningless. ``core.modes`` marks xyz as not supporting flow and
  the flow columns are omitted from the output entirely.
* **The micron-to-pixel ratio comes from the file's XY spacing.** Every metric here is an
  in-plane 2D quantity, so the z step never enters; using a z-derived scale would be
  wrong by the anisotropy factor (4.6x on the Jurkat data).
"""
from __future__ import annotations

import os
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from analysis.binarization import analyze_binarization
from analysis.intensity_distribution import analyze_intensity_distribution
from analysis.volumetric.reader import VolumeStack, read_volume
from core import BarcodeConfig, ChannelResults
from core.modes import get_mode
from utils.setup import create_channel_output_dir, create_output_directories


@dataclass
class SlicewiseRunDetail:
    """Per-run context the CSV has no column for."""

    stack: VolumeStack = None
    n_timepoints: int = 0
    n_slices: int = 0
    xy_step_um: float = np.nan
    z_step_um: float = np.nan
    z_range: tuple = None
    timepoints: List[int] = field(default_factory=list)


def run_slicewise_analysis(
    filepath: str,
    config: BarcodeConfig,
    channel: int = 0,
) -> Tuple[List[ChannelResults], SlicewiseRunDetail]:
    """Analyse ``filepath`` in xyz mode: one ChannelResults per timepoint.

    Each timepoint's z-stack is handed to the 2D branches as if its slices were frames,
    which is what they are for this analysis -- the progression axis is simply depth.
    """
    mode = get_mode("xyz")
    vcfg = config.volumetric

    stack = read_volume(
        filepath,
        channel=channel,
        z_step_um=vcfg.z_step_um or None,
        xy_step_um=vcfg.xy_step_um or None,
    )
    mode.validate_axes(stack.axes, os.path.basename(filepath))

    # Restrict the depth range before anything is measured. Slices past the object are
    # background, and averaging them into the depth profile flattens exactly the trend
    # this mode exists to show.
    stack = stack.restrict_z(vcfg.z_start, vcfg.z_end)

    detail = SlicewiseRunDetail(
        stack=stack,
        n_timepoints=stack.n_timepoints,
        n_slices=stack.n_slices,
        xy_step_um=stack.xy_step_um,
        z_step_um=stack.z_step_um,
        timepoints=list(range(stack.n_timepoints)),
        z_range=stack.z_range,
    )

    # In-plane metrics only, so the scale is the XY pixel size. The 2D branches read this
    # off ReaderConfig, so give them a copy carrying the file's own spacing rather than
    # whatever the 2D tab happens to hold.
    from dataclasses import replace as _replace

    reader_config = _replace(config.reader, um_pixel_ratio=stack.xy_step_um)

    figure_dir = create_output_directories(filepath) if config.writer.save_visualizations else None

    results: List[ChannelResults] = []
    for t in range(stack.n_timepoints):
        volume = stack.data[t]  # (Z, Y, X) -- the shape the 2D branches expect
        row = ChannelResults(filepath=filepath, channel=channel)

        output_dir = ""
        if figure_dir is not None:
            output_dir = create_channel_output_dir(
                figure_dir, channel if stack.n_timepoints == 1 else f"{channel} t{t}"
            )

        # Each branch is guarded independently, mirroring analysis/run.py: one branch
        # failing on one timepoint should cost that branch's columns, not the whole run.
        # This is not hypothetical -- the 2D binarization branch raises
        # "kth out of bounds" whenever a frame contains exactly one island, because
        # find_island_properties partitions a 1x1 distance matrix (analysis/binarization.py).
        if config.modules.image_binarization:
            try:
                _, row.binarization = analyze_binarization(
                    volume, output_dir, config.image_binarization_parameters,
                    reader_config, config.writer,
                )
            except Exception as exc:
                print(f"  t={t}: binarization failed ({type(exc).__name__}: {exc})", flush=True)

        if config.modules.intensity_distribution:
            try:
                _, row.intensity = analyze_intensity_distribution(
                    volume, output_dir, config.intensity_distribution_parameters, config.writer,
                )
            except Exception as exc:
                print(f"  t={t}: intensity failed ({type(exc).__name__}: {exc})", flush=True)

        # Flow is deliberately not run; see the module docstring. The columns are omitted
        # from the output by core.modes rather than written as NaN.
        results.append(row)

    return results, detail


def run_slicewise_pipeline(
    filepath: str,
    config: BarcodeConfig,
    in_config,
    fail_file_loc: str,
    count: int,
    total: int,
) -> Tuple[List[ChannelResults], int]:
    """Entry point matching ``core.pipeline.process_single_file``'s contract."""
    if total != 1:
        print(f"File {count} of {total}")
        print(filepath)
        count += 1

    try:
        results, detail = run_slicewise_analysis(
            filepath, config, channel=config.channels.selected_channel
        )
    except Exception as exc:
        with open(fail_file_loc, "a", encoding="utf-8") as log_file:
            log_file.write(traceback.format_exc())
            log_file.write(f"File: {filepath}, Module: Slicewise(xyz), Exception: {exc}\n")
        raise

    print(
        f"xyz: {detail.n_slices} slices over z"
        + (f" (of the acquired stack, z[{detail.z_range[0]}:{detail.z_range[1]}])"
           if detail.z_range else "")
        + f", {detail.n_timepoints} timepoint(s) @ xy={detail.xy_step_um:g} um"
        f" -> {len(results)} row(s)"
    )
    return results, count
