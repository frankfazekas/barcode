"""Timepoint selection (stream B1).

Mirrors the z-range work: the same conventions, the same shared-helper structure, and
the same guarantee that stating a range in different units picks the same data.

Run: python -m pytest tests/test_time_range.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

from analysis.volumetric.reader import apply_t_range, read_volume
from core import BarcodeConfig
from core.modes import XYZ


def write_series(path, n_t=5, n_z=12, exposure=2.0, xy_step=0.065, z_step=0.3):
    data = np.zeros((n_t, n_z, 32, 32), np.uint16)
    for t in range(n_t):
        data[t, 4:8, 12:20, 12:20] = 500 + 100 * t   # each timepoint distinguishable
    tifffile.imwrite(str(path), data, imagej=True,
                     resolution=(1 / xy_step, 1 / xy_step),
                     metadata={"axes": "TZYX", "spacing": z_step, "unit": "micron",
                               "finterval": exposure})
    return str(path)


# ------------------------------------------------------------------ mechanics


def test_restrict_t_selects_the_requested_timepoints(tmp_path):
    stack = read_volume(write_series(tmp_path / "s.tif", n_t=10))
    assert stack.n_timepoints == 10 and stack.t_range is None

    # The range is INCLUSIVE of both ends: 3..7 is five timepoints, not four.
    cropped = stack.restrict_t(3, 7)
    assert cropped.n_timepoints == 5
    assert cropped.t_range == (3, 8)          # internally still a half-open pair
    assert stack.n_timepoints == 10, "the original must be untouched"


def test_restrict_t_matches_the_z_conventions(tmp_path):
    stack = read_volume(write_series(tmp_path / "s.tif", n_t=10))
    assert stack.restrict_t(3, 0).n_timepoints == 7       # 0 = to the end
    assert stack.restrict_t(0, -3).n_timepoints == 8      # -3 included
    assert stack.restrict_t(0, -1).n_timepoints == 10     # -1 = the last timepoint
    assert stack.restrict_t(0, 0) is stack                # full range is a no-op


def test_restrict_t_rejects_an_empty_range(tmp_path):
    stack = read_volume(write_series(tmp_path / "s.tif", n_t=10))
    for bad in ((7, 3), (12, 15)):
        with pytest.raises(ValueError, match="selects no timepoints"):
            stack.restrict_t(*bad)

    # start == end is a single timepoint under an inclusive range, not an empty one.
    assert stack.restrict_t(5, 5).n_timepoints == 1


def test_the_right_timepoints_are_kept(tmp_path):
    """Not just the right count -- the right data."""
    stack = read_volume(write_series(tmp_path / "s.tif", n_t=5))
    full = [int(stack.data[t].max()) for t in range(5)]
    assert full == [500, 600, 700, 800, 900]

    cropped = stack.restrict_t(1, 4)
    assert [int(cropped.data[t].max()) for t in range(4)] == [600, 700, 800, 900]


# ------------------------------------------------------------------ units


@pytest.mark.parametrize("units,start,end", [
    ("index", 1, 4),
    ("seconds", 2.0, 8.0),     # the same timepoints at a 2 s exposure
])
def test_both_t_units_select_the_same_timepoints(tmp_path, units, start, end):
    stack = read_volume(write_series(tmp_path / "s.tif", n_t=5, exposure=2.0))
    resolved = stack.resolve_t_range(start, end, units)
    assert resolved == (1, 4)
    assert stack.restrict_t(*resolved).n_timepoints == 4          # 1..4 inclusive


def test_unknown_t_units_raise(tmp_path):
    stack = read_volume(write_series(tmp_path / "s.tif"))
    with pytest.raises(ValueError, match="Unknown t_range_units"):
        stack.resolve_t_range(1, 4, "frames")


def test_seconds_needs_a_frame_interval(tmp_path):
    stack = read_volume(write_series(tmp_path / "s.tif"))
    stack.exposure_time_s = 0
    with pytest.raises(ValueError, match="no frame interval is known"):
        stack.resolve_t_range(2.0, 8.0, "seconds")


def test_seconds_uses_the_configured_interval_not_the_files_tag(tmp_path):
    """The regression: a file whose finterval describes the z acquisition, not time.

    On the Jurkat series the timepoints are 60 s apart and ImageJ's finterval reads 1.
    Converting through the tag turned "the first 600 seconds" into 600 timepoints, which
    clamped to the whole series -- while frame_interval_s sat in the config saying 60.
    """
    stack = read_volume(write_series(tmp_path / "s.tif", n_t=5, exposure=1.0))

    assert stack.resolve_t_range(0, 120.0, "seconds", 60.0) == (0, 2)
    assert stack.resolve_t_range(0, 120.0, "seconds") == (0, 120), \
        "with no configured interval it still falls back to the file's tag"

    config = BarcodeConfig().volumetric
    config.t_start, config.t_end, config.t_range_units = 0, 120.0, "seconds"
    config.frame_interval_s = 60.0
    assert apply_t_range(stack, config).n_timepoints == 3         # 0..2 inclusive


def test_no_timing_anywhere_is_not_reported_as_coming_from_the_file(tmp_path, capsys):
    """The reader's 1.0 fallback must stay distinguishable from a real 1 s tag."""
    from analysis.volumetric.run import resolve_frame_interval

    stack = read_volume(write_series(tmp_path / "s.tif", n_t=3))
    stack.timing_from_file = False
    stack.exposure_time_s = 1.0

    config = BarcodeConfig().volumetric
    config.frame_interval_s = 0
    assert resolve_frame_interval(stack, config) == 1.0
    assert "defaulted" in capsys.readouterr().out


def test_zero_end_means_to_the_end_in_both_units(tmp_path):
    stack = read_volume(write_series(tmp_path / "s.tif", n_t=5, exposure=2.0))
    for units, start in (("index", 1), ("seconds", 2.0)):
        _, end = stack.resolve_t_range(start, 0, units)
        assert end == 0, f"{units}: 0 must stay 'to the end'"
        assert stack.restrict_t(*stack.resolve_t_range(start, 0, units)).n_timepoints == 4


def test_apply_t_range_reads_the_config(tmp_path):
    """All three pipelines go through this, so they cannot diverge."""
    stack = read_volume(write_series(tmp_path / "s.tif", n_t=5, exposure=2.0))
    config = BarcodeConfig().volumetric
    config.t_start, config.t_end, config.t_range_units = 2.0, 8.0, "seconds"
    assert apply_t_range(stack, config).n_timepoints == 4         # 1..4 inclusive


# ------------------------------------------------------------------ series


def test_restricting_time_keeps_a_grouped_series_aligned():
    """A grouped series carries its source files; they must follow the data.

    Otherwise row three of the output would name the wrong file -- a provenance error
    that no metric check would catch.
    """
    from analysis.volumetric.reader import VolumeStack

    stack = VolumeStack(
        data=np.zeros((5, 4, 8, 8)), z_step_um=0.3, xy_step_um=0.065,
        exposure_time_s=1.0, axes="TZYX", source_path="Cell1_1.tif",
        metadata_source={"paths": [f"Cell1_{i}.tif" for i in range(1, 6)],
                         "frames": [1, 2, 3, 4, 5]},
    )
    cropped = stack.restrict_t(1, 4)
    assert cropped.metadata_source["paths"] == [
        "Cell1_2.tif", "Cell1_3.tif", "Cell1_4.tif", "Cell1_5.tif"]
    assert cropped.metadata_source["frames"] == [2, 3, 4, 5]


# ------------------------------------------------------------------ pipeline


def test_xyz_produces_one_barcode_per_selected_timepoint(tmp_path):
    from analysis.volumetric.perslice import run_per_slice_analysis

    path = write_series(tmp_path / "Cell1_1.tif", n_t=5, n_z=12)

    def run(t_start, t_end, units="index"):
        config = BarcodeConfig()
        config.modules.image_binarization = True
        config.image_binarization_parameters.bin_factor = 1
        v = config.volumetric
        v.analysis_mode = XYZ
        v.t_start, v.t_end, v.t_range_units = t_start, t_end, units
        per_timepoint, detail = run_per_slice_analysis(path, config)
        return len(per_timepoint), detail.stack.t_range

    assert run(0, 0) == (5, None)
    assert run(1, 4) == (4, (1, 5))               # 1..4 inclusive
    assert run(2.0, 8.0, "seconds") == (4, (1, 5))


def test_a_restricted_time_range_sets_the_provenance_flag(tmp_path):
    """Flag 5 marks a partial analysis, whichever axis was restricted."""
    from analysis.volumetric.perslice import run_per_slice_analysis

    path = write_series(tmp_path / "Cell1_1.tif", n_t=5, n_z=12)
    config = BarcodeConfig()
    config.modules.image_binarization = True
    config.image_binarization_parameters.bin_factor = 1
    config.volumetric.analysis_mode = XYZ

    # Assert on digit 5 specifically rather than the whole string: this fixture is a
    # small uniform blob whose correlation length is NaN, so digit 3 fires legitimately
    # and has nothing to do with the range.
    per_timepoint, _ = run_per_slice_analysis(path, config)
    assert "5" not in per_timepoint[0][0].convert_flags()

    config.volumetric.t_start, config.volumetric.t_end = 1, 4
    per_timepoint, _ = run_per_slice_analysis(path, config)
    assert "5" in per_timepoint[0][0].convert_flags()
