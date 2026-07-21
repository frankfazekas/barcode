"""One barcode row per z-slice: 2D metrics as a function of depth.

``slicewise.py`` reduces a whole depth profile to a single row -- useful for comparing
timepoints, but it collapses away the z structure. This module keeps that structure:
every analysed z-slice becomes its own row, so a barcode built from one timepoint reads
top-to-bottom as "how does the cross-section change with depth".

Like ``slicewise``, no new analysis mathematics: it calls the same per-frame primitives
``analysis/binarization.py`` uses internally (``find_island_properties``,
``find_largest_void``, ``check_span``, ``spatial_image_autocorrelation``) and the same
intensity helpers, read-only. The only difference is that results are kept per frame
rather than reduced across frames.

Two consequences of a row being a single slice, both intentional:

* **Change metrics are NaN.** A "change" needs two points on the progression axis; one
  slice has none. They are the depth trend *between* rows, which the barcode shows as a
  gradient down the column rather than as a number in it.
* **Connectivity is 0 or 1**, not a fraction. It is a single yes/no per slice, and the
  fraction-of-frames reading only exists once several frames are pooled.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from analysis.binarization import (
    calculate_area_or_percentage,
    check_span,
    find_island_properties,
    find_largest_void,
    spatial_image_autocorrelation,
)
from analysis.volumetric.binarization import correlation_length_from_radial
from analysis.volumetric.reader import (
    VolumeStack, apply_t_range, apply_z_range, read_volume)
from analysis.volumetric.segmentation import load_mask_on_image_grid
from core import BarcodeConfig, BinarizationResults, ChannelResults, IntensityResults
from core.modes import get_mode
from utils.binarization import binarize, invert_frame
from utils.intensity_distribution import (
    frame_mode,
    histogram,
    kurtosis,
    median_skewness,
    mode_skewness,
)


@dataclass
class PerSliceRunDetail:
    stack: VolumeStack = None
    n_timepoints: int = 0
    n_slices: int = 0
    z_range: tuple = None
    z_step_um: float = np.nan
    xy_step_um: float = np.nan
    mask_path: Optional[str] = None
    slice_indices: List[int] = field(default_factory=list)


def _binarization_for_slice(frame, bin_config, um_pixel_ratio,
                            mask_slice=None) -> BinarizationResults:
    """Structural metrics for one 2D slice, using the 2D branch's own primitives.

    ``mask_slice``, when given, *replaces* intensity thresholding for this slice --
    the same rule the volumetric path follows.
    """
    binning = bin_config.bin_factor
    if mask_slice is not None:
        binary = mask_slice.astype(int)
        if binning > 1:
            # Match the binning the threshold path applies, so the two are comparable.
            from utils import groupAvg
            binary = (groupAvg(binary.astype(float), binning) >= 0.5).astype(int)
    else:
        binary = binarize(frame, bin_config.threshold_offset, binning,
                          bin_config.minimum_island_size)
    if bin_config.invert_binarization:
        binary = invert_frame(binary)

    (largest, second, total, mean_area,
     separation, anisotropy) = find_island_properties(binary, bin_config)
    void = find_largest_void(binary)

    pixels = frame.shape[0] * frame.shape[1] / (binning ** 2)
    convert = bin_config.enable_physical_units

    def fraction_and_quantity(value):
        quantity, fraction = calculate_area_or_percentage(
            value, pixels, convert, um_pixel_ratio)
        return fraction, quantity

    max_island, max_island_q = fraction_and_quantity(largest)
    second_island, second_island_q = fraction_and_quantity(second)
    total_island, total_island_q = fraction_and_quantity(total)
    mean_island, mean_island_q = fraction_and_quantity(mean_area)
    max_void, max_void_q = fraction_and_quantity(void)

    # Same radial autocorrelation and threshold crossing as the 2D branch, capped the
    # same way at half the frame.
    _, radial = spatial_image_autocorrelation(frame)
    radial = radial[: int(frame.shape[0] / 2 * binning)]
    radii = np.arange(len(radial)) * um_pixel_ratio * binning
    correlation = correlation_length_from_radial(radial, radii, float(np.exp(-1)))

    return BinarizationResults(
        connectivity=float(check_span(binary)),
        max_island_size=max_island,
        max_void_size=max_void,
        # A change needs two slices; this row is one. See the module docstring.
        max_island_percent_change=np.nan,
        max_void_percent_change=np.nan,
        island_size_initial=max_island,
        island_size_initial2=second_island,
        island_anisotropy=anisotropy,
        mean_island_size=mean_island,
        total_island_size=total_island,
        mean_island_separation=separation * um_pixel_ratio if separation == separation else np.nan,
        island_correlation_length=correlation,
        max_island_size_quantity=max_island_q,
        max_void_size_quantity=max_void_q,
        island_size_initial_quantity=max_island_q,
        island_size_initial2_quantity=second_island_q,
        mean_island_size_quantity=mean_island_q,
        total_island_size_quantity=total_island_q,
        structural_correlation_flag=int(np.isnan(correlation)),
    )


def _intensity_for_slice(frame, id_config, mask_slice=None) -> IntensityResults:
    """Intensity statistics for one slice, optionally from in-mask pixels only."""
    data = frame if mask_slice is None else frame[mask_slice.astype(bool)]
    if data.size == 0:
        return IntensityResults()
    counts, values = histogram(data, id_config.bin_size, id_config.noise_threshold)
    saturated = bool(
        frame_mode(data, id_config.bin_size, id_config.noise_threshold) == values[-1])
    return IntensityResults(
        max_kurtosis=kurtosis(values, counts),
        max_median_skew=median_skewness(values, counts),
        max_mode_skew=mode_skewness(values, counts),
        kurtosis_diff=np.nan,
        median_skew_diff=np.nan,
        mode_skew_diff=np.nan,
        saturation_flag=int(saturated),
    )


def run_per_slice_analysis(
    filepath: str,
    config: BarcodeConfig,
    channel: int = 0,
    slice_step: int = 1,
) -> Tuple[List[List[ChannelResults]], PerSliceRunDetail]:
    """Analyse each z-slice separately.

    Returns one list of rows *per timepoint*, so a caller can write one barcode per
    timepoint. ``slice_step`` subsamples depth for very tall stacks.
    """
    mode = get_mode("xyz")
    vcfg = config.volumetric

    stack = read_volume(
        filepath, channel=channel,
        z_step_um=vcfg.z_step_um or None, xy_step_um=vcfg.xy_step_um or None,
        axes_override=getattr(vcfg, "axes_override", "") or None,
    )
    mode.validate_axes(stack.axes, os.path.basename(filepath))
    # Indices refer to ACQUIRED slices, before any isotropic resampling.
    # Timepoints first: the t range decides which volumes are analysed at all, so
    # selecting them before any mask or geometry work keeps that work off the ones
    # that were excluded.
    stack = apply_t_range(stack, vcfg)

    # Load the mask against the FULL acquired stack, then restrict both together.
    # Validating it against an already-restricted image would compare the mask's whole
    # depth with a sub-range and reject a perfectly good mask.
    full_mask = load_mask_on_image_grid(filepath, stack, vcfg)

    stack = apply_z_range(stack, vcfg)

    masks = mask_path = None
    if full_mask is not None:
        mask_volume, mask_path = full_mask
        if stack.z_range:
            mask_volume = mask_volume[stack.z_range[0]:stack.z_range[1]]
        masks = mask_volume

    offset = stack.z_range[0] if stack.z_range else 0
    indices = list(range(0, stack.n_slices, max(int(slice_step), 1)))

    detail = PerSliceRunDetail(
        stack=stack, n_timepoints=stack.n_timepoints, n_slices=stack.n_slices,
        z_range=stack.z_range, z_step_um=stack.z_step_um, xy_step_um=stack.xy_step_um,
        mask_path=mask_path,
        slice_indices=[offset + i for i in indices],
    )

    per_timepoint: List[List[ChannelResults]] = []
    for t in range(stack.n_timepoints):
        rows: List[ChannelResults] = []
        for i in indices:
            frame = stack.data[t, i].astype(np.float64)
            # The row is labelled with its absolute slice index and depth, so a row can
            # always be traced back to a plane in the acquired stack.
            depth_um = (offset + i) * stack.z_step_um
            row = ChannelResults(
                filepath=f"{os.path.basename(filepath)} z={offset + i} ({depth_um:.2f}um)",
                channel=channel,
            )
            row.z_range_flag = 1 if (stack.z_range or stack.t_range) else 0
            if config.modules.image_binarization:
                try:
                    row.binarization = _binarization_for_slice(
                        frame, config.image_binarization_parameters, stack.xy_step_um,
                        masks[i] if masks is not None else None)
                except Exception as exc:
                    print(f"  z={offset + i}: binarization failed "
                          f"({type(exc).__name__}: {exc})", flush=True)
            if config.modules.intensity_distribution:
                try:
                    row.intensity = _intensity_for_slice(
                        frame, config.intensity_distribution_parameters,
                        masks[i] if (masks is not None and vcfg.intensity_use_mask) else None)
                except Exception as exc:
                    print(f"  z={offset + i}: intensity failed "
                          f"({type(exc).__name__}: {exc})", flush=True)
            rows.append(row)
        per_timepoint.append(rows)

    return per_timepoint, detail
