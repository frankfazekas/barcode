"""3D intensity distribution — volumetric counterpart of ``analysis/intensity_distribution.py``.

The underlying statistics are already dimension-agnostic: ``utils.intensity_distribution``
builds its histogram with ``np.histogram``, which flattens whatever it is given. So this
module reuses those helpers read-only and differs from the 2D branch only in that a
"frame" is a whole ``(Z, Y, X)`` volume, and in how it behaves at a single timepoint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from core import IntensityResults, VolumetricConfig
from core.results import IntensityMagnitudeResults
from utils.intensity_distribution import (
    frame_mode,
    histogram,
    kurtosis,
    median_skewness,
    mode_skewness,
)


@dataclass
class VolumetricIntensityDetail:
    frame_indices: List[int] = field(default_factory=list)
    kurtosis: List[float] = field(default_factory=list)
    median_skew: List[float] = field(default_factory=list)
    mode_skew: List[float] = field(default_factory=list)
    saturated: List[bool] = field(default_factory=list)


def _average_largest(values: List[float], percent: float = 0.1) -> float:
    """Mean of the top ``percent`` of values.

    Non-finite values are dropped before the sort. ``sorted`` compares with ``<`` and
    every comparison against NaN is False, so a NaN is never moved out of the position it
    started in; with ten timepoints and the default 10% the one selected element was
    whichever value sat at index 0. A single NaN kurtosis -- which ``utils.kurtosis``
    returns whenever a frame's histogram has zero spread -- therefore made the reported
    maximum NaN even though nine finite values were available, and reordering the same
    measurements changed the answer.
    """
    finite = [v for v in np.asarray(values, dtype=np.float64).ravel() if np.isfinite(v)]
    if not finite:
        return np.nan
    ordered = sorted(finite, reverse=True)
    top = max(int(np.ceil(len(ordered) * percent)), 1)
    return float(np.mean(ordered[:top]))


def analyze_intensity_3d(
    volume_series: np.ndarray,
    config: VolumetricConfig,
    frame_indices: List[int],
    masks: Optional[np.ndarray] = None,
) -> Tuple[IntensityResults, VolumetricIntensityDetail]:
    """Intensity-distribution metrics over a ``(T, Z, Y, X)`` series.

    When ``masks`` is given the histogram is built from voxels inside the mask only,
    so that background does not dominate the distribution of a sparsely-filled crop.
    """
    bins = config.bin_size
    noise = config.noise_threshold
    detail = VolumetricIntensityDetail(frame_indices=list(frame_indices))

    for frame_idx in frame_indices:
        data = volume_series[frame_idx]
        if masks is not None:
            data = data[masks[frame_idx].astype(bool)]

        counts, values = histogram(data, bins, noise)
        detail.kurtosis.append(kurtosis(values, counts))
        detail.median_skew.append(median_skewness(values, counts))
        detail.mode_skew.append(mode_skewness(values, counts))
        # `histogram` drops bins below the noise threshold, so `values` is empty when no
        # bin clears it -- routine for a sparse instance mask, where the in-mask voxels
        # are spread thinly over `bin_size` bins. Indexing [-1] then raised IndexError
        # and, unlike the binarization branch, nothing here catches it, so one such frame
        # killed the whole run.
        detail.saturated.append(
            bool(values.size and frame_mode(data, bins, noise) == values[-1])
        )

    n_frames = len(frame_indices)
    single_timepoint = n_frames < 2
    n_eval = max(int(np.ceil(config.percentage_frames_evaluated * n_frames)), 1)

    def change(values: List[float]) -> float:
        # A single timepoint would compare a value against itself; report NaN instead.
        if single_timepoint:
            return np.nan
        return float(np.nanmean(values[-n_eval:]) - np.nanmean(values[:n_eval]))

    results = IntensityResults(
        max_kurtosis=_average_largest(detail.kurtosis),
        max_median_skew=_average_largest(detail.median_skew),
        max_mode_skew=_average_largest(detail.mode_skew),
        kurtosis_diff=change(detail.kurtosis),
        median_skew_diff=change(detail.median_skew),
        mode_skew_diff=change(detail.mode_skew),
        # `all([])` is True, so a run with no analysed frames flagged itself saturated.
        # The `all` itself is kept deliberately: it is the 2D branch's convention
        # (`analysis/intensity_distribution.py:66`), and this column is compared against
        # reference barcodes. Only the empty case changes.
        saturation_flag=int(all(detail.saturated)) if detail.saturated else 0,
    )
    return results, detail


# ---------------------------------------------------------------------------
# Extensive intensity quantities (stream A1, docs/internal/parallel_work_plan.md)
#
# Every other metric in this branch is *intensive*: kurtosis and skewness describe the
# shape of the histogram and do not change if the object doubles in size. Nothing here
# scaled with the amount of material until these existed, which is why "is the intensity
# branch volume based?" had no answer.
# ---------------------------------------------------------------------------


def compute_intensity_magnitude(values, sample_size: float) -> IntensityMagnitudeResults:
    """Extensive intensity statistics for a flat array of sample values.

    ``sample_size`` is the physical size of one sample -- um^3 per voxel in 3D, um^2 per
    pixel in 2D -- which is what makes ``density`` a physical quantity rather than a
    per-sample one.

    The distinction the numbers turn on:

    * ``total`` and ``mean`` are properties of the *samples*, so they do not change when
      the same object is imaged at a different voxel size.
    * ``density`` is per unit physical volume, so it does.

    ``total`` is a **raw sum** and therefore includes background. On a cropped stack that
    can dominate, which is why ``intensity_use_mask`` matters far more here than it does
    for the shape metrics -- in-mask total intensity is usually the quantity wanted. It
    is also meaningless if the detector clipped: check the saturation flag (digit 2)
    before reading it.
    """
    array = np.asarray(values, dtype=np.float64).ravel()
    array = array[np.isfinite(array)]
    if array.size == 0:
        return IntensityMagnitudeResults()

    total = float(array.sum())
    physical = array.size * float(sample_size) if sample_size else np.nan
    return IntensityMagnitudeResults(
        total=total,
        mean=float(array.mean()),
        sd=float(array.std()),
        density=total / physical if physical and np.isfinite(physical) else np.nan,
    )


def analyze_intensity_magnitude(
    volume_series: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
    frame_indices: List[int],
    masks: Optional[np.ndarray] = None,
) -> IntensityMagnitudeResults:
    """Extensive intensity metrics over a ``(T, Z, Y, X)`` series.

    Computed per analysed timepoint and averaged, matching how every other volumetric
    metric is reduced. ``masks`` restricts the sum to in-mask voxels.
    """
    sample_size = float(np.prod(np.asarray(spacing_zyx, dtype=np.float64)))

    per_frame = []
    for frame_idx in frame_indices:
        data = volume_series[frame_idx]
        if masks is not None:
            data = data[masks[frame_idx].astype(bool)]
        per_frame.append(compute_intensity_magnitude(data, sample_size))

    def mean_of(attribute: str) -> float:
        values = np.asarray([getattr(m, attribute) for m in per_frame], dtype=np.float64)
        finite = values[np.isfinite(values)]
        return float(finite.mean()) if finite.size else np.nan

    return IntensityMagnitudeResults(
        total=mean_of("total"), mean=mean_of("mean"),
        sd=mean_of("sd"), density=mean_of("density"),
    )


def is_saturated(values, bins: int, noise_threshold: float) -> bool:
    """Whether the mode of the distribution sits in the top bin.

    Total intensity is not interpretable when the detector clipped, and the existing
    saturation flag is computed inside the shape branch -- so magnitude callers that
    skip that branch need their own way to ask.
    """
    array = np.asarray(values, dtype=np.float64).ravel()
    array = array[np.isfinite(array)]
    if array.size == 0:
        return False
    counts, edges = histogram(array, bins, noise_threshold)
    # `histogram` drops bins below the noise threshold, and on a nearly-empty or
    # single-valued selection that can leave nothing at all -- `edges[-1]` then raised
    # IndexError out of what is only a flag lookup, losing the whole magnitude family.
    # No bins means no mode, which is not the same as saturated.
    if len(edges) == 0:
        return False
    return bool(frame_mode(array, bins, noise_threshold) == edges[-1])
