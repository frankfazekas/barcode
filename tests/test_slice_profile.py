"""Maximal area slice, and the field-of-view clipping flag (digit 6).

The shape tests use an ellipsoid because its widest slice is known in closed form -- the
answer comes from geometry, not from running the code and pinning whatever it printed.

Run: python -m pytest tests/test_slice_profile.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from analysis.volumetric.slice_profile import (
    slice_areas,
    slice_profile,
    summarise_slice_profile,
    touches_xy_border,
    touches_z_border,
)
from core.results import ChannelResults, SliceProfileResults


def ellipsoid(shape=(21, 60, 60), centre=None, radii=(8, 20, 20)):
    """A filled ellipsoid, whose maximal-area slice is exactly its centre plane."""
    centre = centre if centre is not None else tuple(s // 2 for s in shape)
    zz, yy, xx = np.indices(shape)
    return (((zz - centre[0]) / radii[0]) ** 2
            + ((yy - centre[1]) / radii[1]) ** 2
            + ((xx - centre[2]) / radii[2]) ** 2) <= 1.0


# ------------------------------------------------------------------ profile


def test_slice_areas_are_fractions_of_the_slice():
    volume = np.zeros((4, 10, 10), bool)
    volume[2, :5, :] = True                  # half of slice 2
    areas = slice_areas(volume)
    assert areas.tolist() == [0.0, 0.0, 0.5, 0.0]


def test_slice_areas_rejects_a_non_volume():
    with pytest.raises(ValueError, match=r"expected a \(Z, Y, X\) volume"):
        slice_areas(np.zeros((10, 10), bool))


def test_the_max_area_slice_of_an_ellipsoid_is_its_centre():
    volume = ellipsoid(shape=(21, 60, 60), centre=(10, 30, 30), radii=(8, 20, 20))
    results, detail = slice_profile(volume, z_step_um=1.0)
    assert detail.max_area_index == 10
    assert results.max_area_index == 10.0


def test_an_off_centre_object_moves_the_max_area_slice():
    """The metric must track the object, not report the middle of the array."""
    volume = ellipsoid(shape=(31, 60, 60), centre=(7, 30, 30), radii=(5, 20, 20))
    results, _ = slice_profile(volume, z_step_um=1.0)
    assert results.max_area_index == 7.0


def test_depth_scales_with_the_z_step():
    """Index is bookkeeping; depth is physical and must honour the voxel size."""
    volume = ellipsoid(shape=(21, 40, 40), centre=(10, 20, 20), radii=(8, 12, 12))
    assert slice_profile(volume, z_step_um=1.0)[0].max_area_depth == pytest.approx(10.0)
    assert slice_profile(volume, z_step_um=0.235)[0].max_area_depth == pytest.approx(2.35)


def test_the_reported_area_is_the_area_of_that_slice():
    volume = np.zeros((5, 10, 10), bool)
    volume[1, :2, :] = True                  # 20%
    volume[3, :7, :] = True                  # 70% -- the maximal area
    results, _ = slice_profile(volume, z_step_um=1.0)
    assert results.max_area_index == 3.0
    assert results.max_area_area == pytest.approx(0.7)


def test_an_empty_volume_reports_nan_not_slice_zero():
    """argmax of an all-zero profile is 0, which would read as a real measurement."""
    results, detail = slice_profile(np.zeros((6, 8, 8), bool), z_step_um=1.0)
    assert np.isnan(results.max_area_index)
    assert np.isnan(results.max_area_depth)
    assert detail.max_area_index == -1


# ------------------------------------------------------------------ clipping


def test_an_interior_object_is_not_clipped():
    volume = ellipsoid(shape=(21, 60, 60), centre=(10, 30, 30), radii=(6, 15, 15))
    assert not touches_xy_border(volume)
    assert not touches_z_border(volume)
    assert not slice_profile(volume, 1.0)[1].clipped


@pytest.mark.parametrize("index", [(slice(2, 4), 0, slice(None)),
                                   (slice(2, 4), -1, slice(None)),
                                   (slice(2, 4), slice(None), 0),
                                   (slice(2, 4), slice(None), -1)])
def test_every_xy_edge_counts_as_clipping(index):
    # Confined to interior z slices, so this isolates the xy test from the z one.
    volume = np.zeros((6, 10, 10), bool)
    volume[index] = True
    assert touches_xy_border(volume)
    assert not touches_z_border(volume), "an xy edge is not a z edge"


def test_the_first_and_last_slice_count_as_z_clipping():
    for z in (0, -1):
        volume = np.zeros((6, 10, 10), bool)
        volume[z, 4:6, 4:6] = True
        assert touches_z_border(volume)
        assert not touches_xy_border(volume)


def test_an_empty_volume_is_not_clipped():
    assert not touches_xy_border(np.zeros((4, 6, 6), bool))
    assert not touches_z_border(np.zeros((4, 6, 6), bool))


def test_a_clipped_object_still_reports_its_max_area_slice():
    """Clipping is a caveat on the numbers, not a reason to withhold them."""
    volume = np.zeros((7, 10, 10), bool)
    volume[2:5, :, :] = True                 # spans the whole field
    results, detail = slice_profile(volume, 1.0)
    assert detail.clipped and detail.clipped_xy
    assert results.max_area_area == pytest.approx(1.0)


# ------------------------------------------------------------------ reduction


def test_summarising_averages_over_timepoints():
    per_frame = [SliceProfileResults(max_area_index=4.0, max_area_depth=1.0,
                                     max_area_area=0.5),
                 SliceProfileResults(max_area_index=6.0, max_area_depth=1.5,
                                     max_area_area=0.7)]
    summary = summarise_slice_profile(per_frame)
    assert summary.max_area_index == pytest.approx(5.0)
    assert summary.max_area_area == pytest.approx(0.6)


def test_summarising_ignores_empty_timepoints():
    per_frame = [SliceProfileResults(max_area_index=4.0),
                 SliceProfileResults(),          # all NaN
                 SliceProfileResults(max_area_index=6.0)]
    assert summarise_slice_profile(per_frame).max_area_index == pytest.approx(5.0)


# ------------------------------------------------------------------ schema


def test_flag_digit_six_is_separate_from_digit_five():
    """Digit 5 is 'the user restricted the range'; 6 is 'the data is cut off'."""
    results = ChannelResults(filepath="x.tif", channel=0)
    assert results.convert_flags() == "0"

    results.z_range_flag = 1
    assert results.convert_flags() == "5"

    results.fov_clip_flag = 1
    assert results.convert_flags() == "5;6"

    results.z_range_flag = 0
    assert results.convert_flags() == "6"


def test_the_family_is_opt_in_and_adds_three_columns():
    base = ChannelResults.get_headers(just_metrics=False, mode="xyzt")
    with_profile = ChannelResults.get_headers(
        just_metrics=False, mode="xyzt", include_slice_profile=True)
    assert len(with_profile) == len(base) + 3
    assert "Maximal Area Slice Depth" in with_profile
    assert "Maximal Area Slice Depth" not in base
    assert len(ChannelResults.get_headers(just_metrics=False)) == 28, "2D must not move"


def test_config_defaults_are_inert():
    from core import BarcodeConfig

    assert BarcodeConfig().volumetric.enable_slice_profile is False
