"""In-mask intensity statistics -- the clustering readout.

Values are checked against closed-form answers wherever one exists: a flat distribution
has entropy exactly log2(bins), a two-level object has a CV that falls out of the
algebra, and skewness of a symmetric distribution is exactly 0. The load-bearing tests
are the *invariance* ones, which pin the design decisions rather than the arithmetic:
scaling an object must not move CV, skewness or entropy, and the bright fraction must
survive the punctate case that rescaling would have made undefined.

Run: python -m pytest tests/test_mask_intensity.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from analysis.volumetric.mask_intensity import (
    DEFAULT_BINS,
    analyze_mask_intensity,
    fraction_above_multiple_of_median,
    measure_object,
    rescale_unit,
    shannon_entropy,
    skewness,
    summarise_mask_intensity,
)
from core.results import ChannelResults, MaskIntensityResults


# ------------------------------------------------------------------ primitives


def test_rescale_maps_to_the_unit_interval():
    scaled = rescale_unit(np.array([10.0, 20.0, 30.0]))
    assert scaled.tolist() == [0.0, 0.5, 1.0]


def test_rescale_refuses_a_constant_object():
    """No range means no distribution to describe -- dropped, not reported as 0."""
    assert rescale_unit(np.full(50, 7.0)) is None
    assert rescale_unit(np.array([])) is None


def test_entropy_of_a_flat_distribution_is_log2_bins():
    """One value per bin is the maximum-entropy case, known exactly."""
    values = (np.arange(64) + 0.5) / 64          # one per bin
    assert shannon_entropy(values, bins=64) == pytest.approx(6.0)


def test_entropy_of_a_single_value_is_zero():
    assert shannon_entropy(np.full(100, 0.5), bins=64) == pytest.approx(0.0)


def test_entropy_bins_over_the_fixed_unit_range_not_the_data_range():
    """Per-object data ranges would make entropies incomparable, defeating the point."""
    narrow = np.full(100, 0.5)
    narrow[:50] = 0.51
    # Two adjacent bins at most -> low entropy. If the range were taken from the data
    # these two values would spread across all 64 bins and give 1 bit.
    assert shannon_entropy(narrow, bins=64) < 1.01


def test_skewness_is_zero_for_a_symmetric_distribution():
    assert skewness(np.array([1.0, 2, 3, 4, 5])) == pytest.approx(0.0, abs=1e-12)


def test_skewness_is_positive_for_a_bright_tail():
    assert skewness(np.array([1.0] * 20 + [50.0])) > 1.0


def test_bright_fraction_counts_what_it_says():
    values = np.array([1.0] * 8 + [10.0] * 2)     # median 1.0, two above 2.0
    assert fraction_above_multiple_of_median(values) == pytest.approx(0.2)


def test_bright_fraction_of_a_uniform_object_is_near_zero():
    values = np.linspace(0.5, 1.5, 1000)          # median 1.0, none above 2.0
    assert fraction_above_multiple_of_median(values) == pytest.approx(0.0)


# ------------------------------------------------------------------ the deviation


def test_the_bright_fraction_survives_a_punctate_object():
    """The case that motivated computing this on raw values.

    Rescaling to [0, 1] puts the background at exactly 0, so the rescaled median is 0
    and 'above twice the median' is undefined -- for precisely the objects the metric
    exists to detect. On raw values it is well defined.
    """
    raw = np.array([100.0] * 90 + [1000.0] * 10)
    assert rescale_unit(raw)[:90].tolist() == [0.0] * 90
    assert np.isnan(fraction_above_multiple_of_median(rescale_unit(raw)))

    measured = measure_object(raw)
    assert measured["bright_fraction"] == pytest.approx(0.1)


@pytest.mark.parametrize("factor", [0.5, 3.0, 100.0])
def test_scaling_an_object_leaves_the_shape_metrics_alone(factor):
    """CV, skewness and entropy must describe the pattern, not the brightness.

    This is what makes objects comparable, and it holds *without* rescaling because all
    three are already scale-invariant -- which is the argument for computing them raw.
    """
    rng = np.random.default_rng(0)
    raw = rng.gamma(shape=2.0, scale=50.0, size=4000)

    base = measure_object(raw)
    scaled = measure_object(raw * factor)

    assert scaled["cv"] == pytest.approx(base["cv"], rel=1e-12)
    assert scaled["skewness"] == pytest.approx(base["skewness"], rel=1e-12)
    assert scaled["entropy"] == pytest.approx(base["entropy"], rel=1e-12)
    assert scaled["bright_fraction"] == pytest.approx(base["bright_fraction"], rel=1e-12)
    assert scaled["mfi"] == pytest.approx(base["mfi"] * factor, rel=1e-12)


def test_mfi_and_sd_carry_the_detector_units():
    raw = np.array([100.0, 200.0, 300.0, 400.0])
    measured = measure_object(raw)
    assert measured["mfi"] == pytest.approx(250.0)
    assert measured["sd"] == pytest.approx(np.std(raw))
    assert measured["cv"] == pytest.approx(np.std(raw) / 250.0)


# ------------------------------------------------------------------ discrimination


def test_punctate_signal_is_less_uniform_than_even_fill():
    """The whole purpose: two objects of equal mean, told apart by these metrics."""
    rng = np.random.default_rng(1)
    even = rng.normal(500, 20, 8000)
    punctate = np.concatenate([rng.normal(100, 10, 7200),
                               rng.normal(4100, 100, 800)])
    assert np.mean(even) == pytest.approx(np.mean(punctate), rel=0.05), "same brightness"

    a, b = measure_object(even), measure_object(punctate)
    assert b["cv"] > a["cv"]
    assert b["entropy_normalized"] < a["entropy_normalized"]
    assert b["bright_fraction"] > a["bright_fraction"]
    assert b["skewness"] > a["skewness"]


def test_normalized_entropy_is_one_for_a_flat_object_and_bounded():
    flat = (np.arange(DEFAULT_BINS) + 0.5) / DEFAULT_BINS * 1000.0
    measured = measure_object(flat, bins=DEFAULT_BINS)
    assert measured["entropy_normalized"] == pytest.approx(1.0)
    assert 0.0 <= measured["entropy_normalized"] <= 1.0


# ------------------------------------------------------------------ per object


def test_each_label_is_measured_separately():
    """Two objects with the same pattern at different brightness -- one answer each."""
    image = np.zeros((4, 10, 10), np.float64)
    labels = np.zeros((4, 10, 10), np.int32)

    pattern = np.linspace(1.0, 2.0, 100).reshape(10, 10)
    image[1], labels[1] = pattern * 100, 1
    image[2], labels[2] = pattern * 900, 2

    results, detail = analyze_mask_intensity(image, labels)
    assert detail.object_ids == [1, 2]
    assert detail.mfi[1] == pytest.approx(detail.mfi[0] * 9.0, rel=1e-9)
    # ...but the shape statistics are identical, so the average is that shared value.
    assert results.cv == pytest.approx(measure_object(pattern.ravel())["cv"], rel=1e-9)


def test_a_boolean_mask_is_one_object():
    image = np.zeros((4, 8, 8), np.float64)
    image[1:3] = np.linspace(10, 90, 128).reshape(2, 8, 8)
    mask = np.zeros((4, 8, 8), bool)
    mask[1:3] = True

    results, detail = analyze_mask_intensity(image, mask)
    assert detail.object_ids == [1]
    assert results.mfi == pytest.approx(image[mask].mean())


def test_objects_are_averaged_unweighted_so_one_giant_does_not_dominate():
    """These describe a typical object, not the pooled voxels."""
    image = np.zeros((2, 20, 20), np.float64)
    labels = np.zeros((2, 20, 20), np.int32)
    rng = np.random.default_rng(2)

    labels[0, :, :] = 1                       # 400 voxels
    image[0] = rng.normal(1000, 10, (20, 20))
    labels[1, :2, :5] = 2                     # 10 voxels
    image[1, :2, :5] = rng.normal(100, 1, (2, 5))

    results, detail = analyze_mask_intensity(image, labels, min_voxels=8)
    assert detail.object_ids == [1, 2]
    assert results.mfi == pytest.approx((detail.mfi[0] + detail.mfi[1]) / 2, rel=1e-9)
    assert results.mfi < 700, "a voxel-weighted mean would sit near 1000"


def test_tiny_objects_are_skipped_rather_than_averaged_in():
    image = np.zeros((2, 10, 10), np.float64)
    labels = np.zeros((2, 10, 10), np.int32)
    image[0] = np.linspace(1, 100, 100).reshape(10, 10)
    labels[0] = 1
    image[1, 0, :3] = [5.0, 6.0, 7.0]
    labels[1, 0, :3] = 2                      # 3 voxels, below min_voxels

    results, detail = analyze_mask_intensity(image, labels, min_voxels=8)
    assert detail.object_ids == [1]
    assert detail.skipped == 1


def test_a_constant_object_is_skipped():
    image = np.full((2, 10, 10), 42.0)
    labels = np.ones((2, 10, 10), np.int32)
    results, detail = analyze_mask_intensity(image, labels)
    assert detail.skipped == 1
    assert np.isnan(results.cv)


def test_an_empty_mask_reports_nan():
    results, detail = analyze_mask_intensity(np.ones((2, 8, 8)), np.zeros((2, 8, 8), np.int32))
    assert np.isnan(results.mfi)
    assert detail.object_ids == []


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError, match="does not match"):
        analyze_mask_intensity(np.ones((2, 8, 8)), np.ones((2, 4, 4), np.int32))


# ------------------------------------------------------------------ schema


def test_summarising_averages_over_timepoints():
    summary = summarise_mask_intensity([
        MaskIntensityResults(mfi=100.0, cv=0.2),
        MaskIntensityResults(mfi=200.0, cv=0.4),
    ])
    assert summary.mfi == pytest.approx(150.0)
    assert summary.cv == pytest.approx(0.3)


def test_the_family_is_opt_in_and_adds_seven_columns():
    base = ChannelResults.get_headers(just_metrics=False, mode="xyzt")
    with_family = ChannelResults.get_headers(
        just_metrics=False, mode="xyzt", include_mask_intensity=True)
    assert len(with_family) == len(base) + 7
    assert "In-Mask MFI" in with_family
    assert "In-Mask MFI" not in base
    assert len(ChannelResults.get_headers(just_metrics=False)) == 28, "2D must not move"


def test_config_defaults_are_inert():
    from core import BarcodeConfig

    v = BarcodeConfig().volumetric
    assert v.enable_mask_intensity is False
    assert v.mask_intensity_bins == 64
    assert v.mask_intensity_min_voxels == 8
