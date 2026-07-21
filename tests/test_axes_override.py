"""Axis override: letting the user correct a file whose header is wrong.

The module's rule is that BARCODE never *guesses* an axis order. This is the other half
of that rule -- it may be *told*. The motivating case is real: a Drosophila hyperstack
declaring ``ZCYX`` for (150, 14, 1500, 1808) data that is actually a 150-timepoint movie
of 14-slice stacks, because the acquisition software wrote time into ImageJ's
``channels`` field.

A wrong override silently reinterprets everything, so validation is checked as carefully
as the happy path.

Run: python -m pytest tests/test_axes_override.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

from analysis.volumetric.reader import read_volume, validate_axes_override
from core import BarcodeConfig


def write_mislabelled(path, n_t=6, n_z=4, size=16):
    """A T-series written out with T declared as ImageJ 'channels' -- the real bug.

    tifffile reports this as ZCYX with shape (n_t, n_z, ...) because ImageJ's page
    order puts channels fastest.
    """
    data = np.zeros((n_t, n_z, size, size), np.uint16)
    for t in range(n_t):
        for z in range(n_z):
            data[t, z, 4:12, 4:12] = 100 * (t + 1) + z   # every plane identifiable
    tifffile.imwrite(str(path), data, imagej=True,
                     metadata={"axes": "ZCYX", "spacing": 1.0})
    return str(path), data


# ------------------------------------------------------------------ validation


def test_override_must_match_the_data_rank():
    with pytest.raises(ValueError, match="4 axes but the file's data is 3-dimensional"):
        validate_axes_override("TZYX", (4, 8, 8))


def test_override_rejects_unknown_letters():
    with pytest.raises(ValueError, match=r"\['W'\]"):
        validate_axes_override("TWYX", (2, 2, 8, 8))


def test_override_rejects_a_repeated_axis():
    with pytest.raises(ValueError, match="repeats an axis"):
        validate_axes_override("TTYX", (2, 2, 8, 8))


def test_override_requires_y_and_x():
    with pytest.raises(ValueError, match="must include both Y and X"):
        validate_axes_override("TZCY", (2, 2, 8, 8))


def test_override_is_case_insensitive():
    assert validate_axes_override("tzyx", (2, 2, 8, 8)) == "TZYX"


# ------------------------------------------------------------------ reading


def test_without_an_override_the_wrong_header_is_believed(tmp_path):
    """The failure this feature exists to fix, pinned so it cannot be mistaken for fine."""
    path, _ = write_mislabelled(tmp_path / "m.tif", n_t=6, n_z=4)
    stack = read_volume(path)                      # believes ZCYX
    assert stack.axes == "ZCYX"
    assert stack.n_timepoints == 1, "the 6 timepoints are read as z-slices"
    assert stack.n_slices == 6
    assert stack.data.shape == (1, 6, 16, 16)


def test_the_override_recovers_the_true_layout(tmp_path):
    path, _ = write_mislabelled(tmp_path / "m.tif", n_t=6, n_z=4)
    stack = read_volume(path, axes_override="TZYX")
    assert stack.n_timepoints == 6 and stack.n_slices == 4
    assert stack.data.shape == (6, 4, 16, 16)


def test_the_override_moves_the_actual_voxels(tmp_path):
    """Shape alone would pass even if the data were transposed wrongly."""
    path, source = write_mislabelled(tmp_path / "m.tif", n_t=6, n_z=4)
    stack = read_volume(path, axes_override="TZYX")
    for t in range(6):
        for z in range(4):
            assert stack.data[t, z].max() == 100 * (t + 1) + z, f"t={t} z={z} misplaced"
    assert np.array_equal(stack.data, source)


def test_provenance_records_both_orders(tmp_path):
    path, _ = write_mislabelled(tmp_path / "m.tif")
    stack = read_volume(path, axes_override="TZYX")
    assert stack.axes == "TZYX", "the effective order"
    assert stack.declared_axes == "ZCYX", "what the file claimed"


def test_no_override_leaves_declared_axes_reported_as_before(tmp_path):
    """The pre-existing contract: `axes` is the file's own order when nothing is said."""
    path, _ = write_mislabelled(tmp_path / "m.tif")
    stack = read_volume(path)
    assert stack.axes == "ZCYX" == stack.declared_axes


def test_an_override_can_rescue_an_undetermined_axis(tmp_path):
    """Plain page-sequence TIFFs come back as QYX, which the reader otherwise refuses."""
    data = np.zeros((10, 16, 16), np.uint16)
    data[:, 4:12, 4:12] = 300
    path = str(tmp_path / "plain.tif")
    tifffile.imwrite(path, data)

    with pytest.raises(ValueError, match="undetermined axes"):
        read_volume(path)

    stack = read_volume(path, axes_override="ZYX")
    assert stack.data.shape == (1, 10, 16, 16)


def test_a_bad_override_still_fails_the_normal_checks(tmp_path):
    """Overriding to something with no Z must hit the existing 'not volumetric' guard."""
    path, _ = write_mislabelled(tmp_path / "m.tif", n_t=6, n_z=4)
    with pytest.raises(ValueError, match="no Z axis"):
        read_volume(path, axes_override="TCYX")


# ------------------------------------------------------------------ plumbing


def test_the_config_field_reaches_the_reader(tmp_path):
    """All three mode pipelines read the same field, so they cannot diverge."""
    from analysis.volumetric.perslice import run_per_slice_analysis
    from core.modes import XYZ

    path, _ = write_mislabelled(tmp_path / "Cell1_1.tif", n_t=6, n_z=4)

    config = BarcodeConfig()
    config.modules.image_binarization = True
    config.image_binarization_parameters.bin_factor = 1
    config.volumetric.analysis_mode = XYZ
    config.volumetric.axes_override = "TZYX"

    per_timepoint, detail = run_per_slice_analysis(path, config)
    assert len(per_timepoint) == 6, "one barcode per true timepoint"
    assert detail.stack.n_slices == 4


def test_the_default_config_overrides_nothing(tmp_path):
    assert BarcodeConfig().volumetric.axes_override == ""
