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
    ordered = sorted(values, reverse=True)
    top = int(np.ceil(len(ordered) * percent))
    return float(np.nanmean(ordered[:top]))


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
        detail.saturated.append(bool(frame_mode(data, bins, noise) == values[-1]))

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
        saturation_flag=int(all(detail.saturated)),
    )
    return results, detail
