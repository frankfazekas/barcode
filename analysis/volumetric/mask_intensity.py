"""Intensity distribution *inside* segmented objects -- the clustering readout.

The intensity branch (``analysis/intensity_distribution.py`` and its 3D counterpart)
describes whatever voxels it is handed. On a cropped stack that is mostly background,
and the background peak dominates every moment it computes. These metrics instead ask a
question about the inside of each object: is the signal spread evenly through it, or
concentrated into foci?

Ported from the T-cell ``clustering_inside_nuc.m`` analysis, with one **deliberate
deviation** that is called out here because it changes numbers:

That source rescales each object's intensities to [0, 1] before *every* statistic, to
make objects comparable. Here only **entropy** is computed on the rescaled values, and
MFI, SD, CV, skewness and the bright fraction are computed on the raw ones. The reasons
are specific, not stylistic:

* **Entropy genuinely needs it.** Binning is only comparable between objects if the bins
  cover the same range, so entropy is taken over a fixed grid on [0, 1].
* **CV and skewness are already scale-invariant.** ``SD/mean`` and the standardised third
  moment are unchanged by a pure scaling, so rescaling buys no comparability -- while
  ``(x - min) / (max - min)`` is affine, not a scaling, so the ``-min`` shift actively
  distorts both. Raw values are the more faithful answer.
* **The bright fraction breaks under rescaling.** In a punctate object most voxels sit at
  the floor, so the rescaled median is 0 and "above twice the median" is undefined --
  degenerate for precisely the objects the metric exists to detect. On raw values the
  median is positive and the quantity means what it says.

``entropy_normalized`` divides by log2(bins), the entropy of a perfectly flat
distribution: 1.0 for uniform fill, falling towards 0 as signal concentrates, readable
without knowing the bin count.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

# Bins for the in-mask histogram. Fixed rather than configurable because
# `entropy_normalized` is only comparable between runs that used the same number.
DEFAULT_BINS = 64


@dataclass
class MaskIntensityDetail:
    """Per-object values, which the reported scalars average over."""

    object_ids: List[int] = field(default_factory=list)
    n_voxels: List[int] = field(default_factory=list)
    mfi: List[float] = field(default_factory=list)
    cv: List[float] = field(default_factory=list)
    entropy: List[float] = field(default_factory=list)
    skipped: int = 0          # objects too small or too flat to describe

    def describe(self) -> str:
        if not self.object_ids:
            return "in-mask intensity: no objects measured"
        return (f"in-mask intensity: {len(self.object_ids)} object(s), "
                f"{self.skipped} skipped, "
                f"mean MFI {np.nanmean(self.mfi):.1f}")


def rescale_unit(values: np.ndarray) -> Optional[np.ndarray]:
    """Rescale to [0, 1]. Returns None when the object has no intensity range.

    A constant object cannot be rescaled -- ``(x - min) / (max - min)`` divides by zero
    -- and it has no distribution to describe either, so it is dropped rather than
    reported as a degenerate 0 or NaN mixed in with real measurements.
    """
    array = np.asarray(values, dtype=np.float64).ravel()
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    low, high = float(array.min()), float(array.max())
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return None
    return (array - low) / (high - low)


def shannon_entropy(values: np.ndarray, bins: int = DEFAULT_BINS) -> float:
    """Entropy in bits of values already on [0, 1], over a fixed grid on that range.

    The range is pinned to [0, 1] rather than taken from the data so that every object
    is binned identically -- otherwise the bin width would vary per object and the
    entropies would not be comparable, which is the whole point of rescaling first.
    """
    array = np.asarray(values, dtype=np.float64).ravel()
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.nan
    counts, _ = np.histogram(array, bins=bins, range=(0.0, 1.0))
    total = counts.sum()
    if total == 0:
        return np.nan
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log2(p)))


def fraction_above_multiple_of_median(values: np.ndarray, multiple: float = 2.0) -> float:
    """Fraction of voxels brighter than ``multiple`` times the median.

    A blunt, threshold-free clustering measure: uniform fill puts almost nothing above
    twice its own median, whereas punctate signal puts a clear fraction there. The
    median is used rather than the mean precisely because foci drag a mean upwards and
    would hide themselves.
    """
    array = np.asarray(values, dtype=np.float64).ravel()
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.nan
    median = float(np.median(array))
    if median <= 0:
        return np.nan
    return float(np.mean(array > multiple * median))


def skewness(values: np.ndarray) -> float:
    """Fisher-Pearson skewness. Positive means a bright tail."""
    array = np.asarray(values, dtype=np.float64).ravel()
    array = array[np.isfinite(array)]
    if array.size < 3:
        return np.nan
    sd = array.std()
    if sd <= 0:
        return np.nan
    return float(np.mean(((array - array.mean()) / sd) ** 3))


def measure_object(raw_values: np.ndarray, bins: int = DEFAULT_BINS) -> Optional[dict]:
    """All in-mask statistics for one object's voxels. None if it cannot be described.

    Entropy is computed on the [0, 1] rescaling; everything else on the raw values, per
    the deviation set out in the module docstring.
    """
    raw = np.asarray(raw_values, dtype=np.float64).ravel()
    raw = raw[np.isfinite(raw)]
    if raw.size == 0:
        return None

    scaled = rescale_unit(raw)
    if scaled is None:
        # A constant object has no distribution to describe, and no range to bin.
        return None

    entropy = shannon_entropy(scaled, bins)
    mean = float(raw.mean())
    sd = float(raw.std())
    return {
        "n_voxels": int(raw.size),
        "mfi": mean,
        "sd": sd,
        "cv": sd / mean if mean > 0 else np.nan,
        "skewness": skewness(raw),
        "entropy": entropy,
        "entropy_normalized": entropy / np.log2(bins) if np.isfinite(entropy) else np.nan,
        "bright_fraction": fraction_above_multiple_of_median(raw),
    }


def analyze_mask_intensity(
    image: np.ndarray,
    labels: np.ndarray,
    bins: int = DEFAULT_BINS,
    min_voxels: int = 8,
):
    """In-mask intensity statistics over every object in one volume.

    ``labels`` may be an integer label volume -- one object per label, which is what
    makes these *per-object* -- or a boolean mask, which is treated as a single object.
    Objects smaller than ``min_voxels`` are skipped: an entropy or skewness computed
    from a handful of voxels is noise, and averaging it in would quietly bias the run.

    Returns ``(MaskIntensityResults, MaskIntensityDetail)``. Every reported scalar is the
    unweighted mean over objects, so one huge object does not outvote the rest -- these
    describe a typical object, not the pooled voxels.
    """
    from core.results import MaskIntensityResults

    image = np.asarray(image)
    labels = np.asarray(labels)
    if image.shape != labels.shape:
        raise ValueError(
            f"image shape {image.shape} does not match label shape {labels.shape}"
        )

    detail = MaskIntensityDetail()
    if labels.dtype == bool:
        ids = [1] if labels.any() else []
        label_volume = labels.astype(np.int32)
    else:
        label_volume = labels
        ids = [int(v) for v in np.unique(label_volume) if v != 0]

    per_object = []
    for object_id in ids:
        voxels = image[label_volume == object_id]
        if voxels.size < min_voxels:
            detail.skipped += 1
            continue
        measured = measure_object(voxels, bins)
        if measured is None:
            detail.skipped += 1
            continue
        per_object.append(measured)
        detail.object_ids.append(object_id)
        detail.n_voxels.append(measured["n_voxels"])
        detail.mfi.append(measured["mfi"])
        detail.cv.append(measured["cv"])
        detail.entropy.append(measured["entropy"])

    if not per_object:
        return MaskIntensityResults(), detail

    def mean_of(key: str) -> float:
        values = np.array([m[key] for m in per_object], dtype=np.float64)
        finite = values[np.isfinite(values)]
        return float(finite.mean()) if finite.size else np.nan

    return (
        MaskIntensityResults(
            mfi=mean_of("mfi"),
            sd=mean_of("sd"),
            cv=mean_of("cv"),
            skewness=mean_of("skewness"),
            entropy=mean_of("entropy"),
            entropy_normalized=mean_of("entropy_normalized"),
            bright_fraction=mean_of("bright_fraction"),
        ),
        detail,
    )


def summarise_mask_intensity(results: Sequence["MaskIntensityResults"]) -> "MaskIntensityResults":
    """Average the per-timepoint scalars, matching how every other family is reduced."""
    from core.results import MaskIntensityResults

    def mean_of(attribute: str) -> float:
        values = np.array([getattr(r, attribute) for r in results], dtype=np.float64)
        finite = values[np.isfinite(values)]
        return float(finite.mean()) if finite.size else np.nan

    return MaskIntensityResults(
        mfi=mean_of("mfi"), sd=mean_of("sd"), cv=mean_of("cv"),
        skewness=mean_of("skewness"), entropy=mean_of("entropy"),
        entropy_normalized=mean_of("entropy_normalized"),
        bright_fraction=mean_of("bright_fraction"),
    )
