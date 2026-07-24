"""Stream A: extensive intensity quantities and range provenance.

The property that matters and is easy to get wrong: ``total`` and ``mean`` describe the
*samples* and must not change when the same object is imaged at a different voxel size,
while ``density`` is per unit physical volume and must. Confusing the two is the whole
reason this family did not exist before.

Run: python -m pytest tests/test_intensity_magnitude.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from analysis.volumetric.intensity import (
    analyze_intensity_magnitude,
    compute_intensity_magnitude,
    is_saturated,
)
from analysis.volumetric.provenance import (
    build_range_results,
    describe_range,
    was_restricted,
)
from core.results import IntensityMagnitudeResults, RangeResults


# ------------------------------------------------------------------ analytic


def test_constant_volume_gives_exact_values():
    """Everything is known in closed form, so nothing here is approximate."""
    volume = np.full((10, 10, 10), 7.0)
    sample = 0.001                       # um^3 per voxel

    result = compute_intensity_magnitude(volume, sample)

    assert result.total == 7.0 * 1000
    assert result.mean == 7.0
    assert result.sd == 0.0
    # total / (1000 voxels * 0.001 um^3) = 7000 a.u. per um^3
    assert result.density == pytest.approx(7000.0)


def test_total_is_the_sum_of_an_arbitrary_field():
    rng = np.random.default_rng(0)
    volume = rng.random((6, 8, 9)) * 500
    result = compute_intensity_magnitude(volume, 1.0)
    assert result.total == pytest.approx(volume.sum())
    assert result.mean == pytest.approx(volume.mean())
    assert result.sd == pytest.approx(volume.std())


def test_extensive_and_intensive_quantities_are_not_confused():
    """Doubling the voxel size must move density and leave total/mean alone.

    This is the check that catches a density computed per *sample* rather than per unit
    volume -- the two agree at a spacing of 1.0 and diverge everywhere else, so a test
    at unit spacing would pass either way.
    """
    volume = np.full((8, 8, 8), 3.0)

    fine = compute_intensity_magnitude(volume, sample_size=1.0)
    coarse = compute_intensity_magnitude(volume, sample_size=8.0)   # 2x per axis

    assert coarse.total == fine.total, "total is a property of the samples"
    assert coarse.mean == fine.mean, "mean is a property of the samples"
    assert coarse.density == pytest.approx(fine.density / 8.0), \
        "density is per unit physical volume"


def test_density_uses_the_real_voxel_shape():
    """Anisotropic voxels: the sample size is the product of the three spacings."""
    volume = np.full((5, 5, 5), 2.0)
    spacing = (0.3, 0.065, 0.065)
    result = analyze_intensity_magnitude(volume[None], spacing, [0])

    voxel_volume = 0.3 * 0.065 * 0.065
    assert result.total == pytest.approx(2.0 * 125)
    assert result.density == pytest.approx(2.0 / voxel_volume)


# ------------------------------------------------------------------ masking


def test_masking_restricts_the_sum_to_in_mask_voxels():
    """In-mask total is usually the quantity wanted; the difference must be exact."""
    volume = np.full((4, 6, 6), 10.0)
    volume[0] = 1.0                                   # a background slice
    mask = np.zeros((4, 6, 6), bool)
    mask[1:] = True                                   # exclude that slice

    whole = analyze_intensity_magnitude(volume[None], (1., 1., 1.), [0])
    inside = analyze_intensity_magnitude(volume[None], (1., 1., 1.), [0], masks=mask[None])

    out_of_mask_sum = volume[0].sum()
    assert whole.total - inside.total == pytest.approx(out_of_mask_sum)
    assert inside.mean == pytest.approx(10.0), "background must not drag the mean"
    assert whole.mean < inside.mean


def test_an_empty_mask_yields_nan_rather_than_zero():
    """No voxels is 'not measured', not 'measured as zero'."""
    volume = np.full((4, 4, 4), 5.0)
    empty = np.zeros((4, 4, 4), bool)
    result = analyze_intensity_magnitude(volume[None], (1., 1., 1.), [0], masks=empty[None])
    assert np.isnan(result.total) and np.isnan(result.density)
    assert not result.is_populated()


def test_non_finite_samples_are_ignored():
    """The slice-wise path blanks out-of-mask voxels with NaN."""
    volume = np.full((4, 4, 4), 6.0)
    volume[0] = np.nan
    result = compute_intensity_magnitude(volume, 1.0)
    assert result.total == pytest.approx(6.0 * 48)
    assert result.mean == pytest.approx(6.0)


# ------------------------------------------------------------------ series


def test_series_values_are_averaged_over_analysed_timepoints():
    series = np.stack([np.full((4, 4, 4), 2.0), np.full((4, 4, 4), 4.0)])
    result = analyze_intensity_magnitude(series, (1., 1., 1.), [0, 1])
    assert result.mean == pytest.approx(3.0)
    assert result.total == pytest.approx((2.0 * 64 + 4.0 * 64) / 2)

    # analysing only the first timepoint must report only the first
    first = analyze_intensity_magnitude(series, (1., 1., 1.), [0])
    assert first.mean == pytest.approx(2.0)


# ------------------------------------------------------------------ saturation


def test_saturation_is_detectable_without_the_shape_branch():
    """Total intensity is meaningless when the detector clipped."""
    clipped = np.full((4, 8, 8), 65535.0)
    clipped[0] = 100.0
    assert is_saturated(clipped, bins=300, noise_threshold=5e-4)

    clean = np.linspace(10, 200, 4 * 8 * 8).reshape(4, 8, 8)
    assert not is_saturated(clean, bins=300, noise_threshold=5e-4)


# ------------------------------------------------------------------ provenance


class _Stack:
    """Minimal stand-in carrying only what provenance reads."""

    def __init__(self, n_slices, n_timepoints, z_range=None, t_range=None):
        self.n_slices = n_slices
        self.n_timepoints = n_timepoints
        self.z_range = z_range
        self.t_range = t_range


def test_an_unrestricted_axis_reports_its_full_extent():
    """'0 to 53' and 'no range was set' are the same statement about the data.

    Both ends are reported inclusively, matching the settings a user typed, so a full
    54-slice stack reports its last slice as 53 rather than a one-past-the-end 54.
    """
    result = build_range_results(_Stack(54, 15))
    assert (result.z_start, result.z_end) == (0.0, 53.0)
    assert (result.t_start, result.t_end) == (0.0, 14.0)
    assert not was_restricted(_Stack(54, 15))


def test_a_restricted_axis_reports_the_range_applied():
    # z_range is the INTERNAL half-open pair; a user asking for slices 12..46 produces
    # (12, 47), and the reported range must read back as the inclusive 12..46.
    stack = _Stack(35, 15, z_range=(12, 47))
    result = build_range_results(stack)
    assert (result.z_start, result.z_end) == (12.0, 46.0)
    assert (result.t_start, result.t_end) == (0.0, 14.0)
    assert was_restricted(stack)


def test_both_axes_can_be_restricted():
    stack = _Stack(35, 5, z_range=(12, 47), t_range=(2, 8))
    result = build_range_results(stack)
    assert (result.z_start, result.z_end, result.t_start, result.t_end) == \
        (12.0, 46.0, 2.0, 7.0)
    assert describe_range(result) == "z[12:46] t[2:7]"


def test_range_results_survive_a_csv_round_trip(tmp_path):
    from core.modes import MODES, XYZT
    from core.results import ChannelResults
    from utils.reader import read_csv_to_channel_results
    from utils.writer import results_to_csv

    row = ChannelResults(filepath="cell.tif", channel=0)
    row.ranges = build_range_results(_Stack(35, 15, z_range=(12, 47), t_range=(2, 8)))
    row.intensity_magnitude = IntensityMagnitudeResults(
        total=5000.0, mean=12.5, sd=3.25, density=41.0)

    path = str(tmp_path / "Summary.csv")
    results_to_csv([row], path, just_metrics=False, mode=MODES[XYZT])
    back = read_csv_to_channel_results(path)

    assert len(back) == 1
    assert back[0].ranges.z_start == pytest.approx(12)
    assert back[0].ranges.t_end == pytest.approx(7)
    assert back[0].intensity_magnitude.density == pytest.approx(41.0)
