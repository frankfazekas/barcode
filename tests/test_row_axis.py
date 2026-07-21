"""Row axis: what one barcode row is, and therefore what is being compared.

The barcode normalises per column across rows, so the rows *are* the comparison. The
right choice depends on the data — a field of 840 cells asks a per-object question, a
single nucleus over time asks a temporal one — and until this existed the answer was
implicit in which script you happened to run.

The load-bearing guarantee is the last test: **without a segmentation the resolver can
never reach 'object'**, so every existing run, including the published 2D reference,
keeps the rows it always had.

Run: python -m pytest tests/test_row_axis.py -v
"""
from __future__ import annotations

import pytest

from core.modes import get_mode
from core.row_axis import (
    FILE,
    OBJECT,
    OBJECT_SCOPE,
    ROW_AXES,
    SLICE,
    TIMEPOINT,
    describe_scope,
    get_row_axis,
    resolve_row_axis,
)

XYZT = get_mode("xyzt")
XYZ = get_mode("xyz")
XYT = get_mode("xyt")


def auto(mode=XYZT, **kwargs):
    return resolve_row_axis("auto", mode, **kwargs).key


# ------------------------------------------------------------------ auto-resolution


def test_many_objects_means_object_rows():
    """The Drosophila case: ~840 cells in a field."""
    assert auto(has_labels=True, n_objects=839, n_timepoints=5) == OBJECT


def test_one_object_over_time_means_timepoint_rows():
    """The Jurkat case: a single nucleus, so the only comparison is temporal."""
    assert auto(has_labels=True, n_objects=1, n_timepoints=15) == TIMEPOINT


def test_a_single_volume_means_file_rows():
    assert auto(has_labels=True, n_objects=1, n_timepoints=1) == FILE
    assert auto(has_labels=False, n_objects=0, n_timepoints=1) == FILE


def test_objects_beat_timepoints():
    """A field of cells asks a per-cell question even when it is also a movie."""
    assert auto(has_labels=True, n_objects=500, n_timepoints=50) == OBJECT


def test_without_a_segmentation_object_is_unreachable():
    """The guarantee that protects every existing run and the 2D reference outputs."""
    for n_timepoints in (1, 2, 15, 1000):
        for n_objects in (0, 1, 500):
            resolved = auto(has_labels=False, n_objects=n_objects,
                            n_timepoints=n_timepoints)
            assert resolved != OBJECT, (n_objects, n_timepoints)


# ------------------------------------------------------------------ explicit choices


def test_an_explicit_choice_is_honoured():
    assert resolve_row_axis("file", XYZT, has_labels=True, n_objects=800).key == FILE
    assert resolve_row_axis("timepoint", XYZT, has_labels=True,
                            n_objects=800, n_timepoints=3).key == TIMEPOINT


def test_object_without_labels_raises_rather_than_falling_back():
    """Silently choosing another axis would change what the figure compares."""
    with pytest.raises(ValueError, match="needs an instance segmentation"):
        resolve_row_axis("object", XYZT, has_labels=False)


def test_object_with_a_single_object_raises():
    with pytest.raises(ValueError, match="more than one object"):
        resolve_row_axis("object", XYZT, has_labels=True, n_objects=1)


def test_slice_rows_are_xyz_only():
    assert resolve_row_axis("slice", XYZ, has_labels=False).key == SLICE
    with pytest.raises(ValueError, match="only available in mode"):
        resolve_row_axis("slice", XYZT, has_labels=False)


def test_an_unknown_axis_lists_the_valid_ones():
    with pytest.raises(ValueError, match="Unknown row axis"):
        get_row_axis("cells")


# ------------------------------------------------------------------ scope


def test_only_object_rows_carry_the_object_scope():
    assert ROW_AXES[OBJECT].scope == OBJECT_SCOPE
    assert ROW_AXES[OBJECT].is_per_object
    for key in (FILE, TIMEPOINT, SLICE):
        assert not ROW_AXES[key].is_per_object


def test_the_scope_description_names_what_was_normalised_over():
    """Two figures normalised over different sets are not comparable; say which."""
    text = describe_scope(ROW_AXES[OBJECT], 4176, 5)
    assert "4176 objects" in text and "5 fields" in text
    assert "1 object " in describe_scope(ROW_AXES[OBJECT], 1, 1)


def test_config_default_is_auto():
    from core import BarcodeConfig
    assert BarcodeConfig().volumetric.row_axis == "auto"
