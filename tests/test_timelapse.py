"""Validation for time-lapse assembly.

The two claims worth pinning down are that files are grouped and *ordered* correctly
(lexicographic sorting would interleave frame 10 before frame 2), and that every
timepoint ends up on one shared grid -- otherwise the fraction-of-volume metrics use a
different denominator per timepoint and cannot be compared along the time axis they
describe.

Run: python -m pytest tests/test_timelapse.py -v
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import tifffile

from analysis.volumetric.run import _prepare_geometry
from analysis.volumetric.reader import VolumeStack
from analysis.volumetric.timelapse import (
    DEFAULT_TIMELAPSE_REGEX,
    group_timelapse,
    read_series,
)
from core import VolumetricConfig


def write_volume(path, shape=(8, 24, 24), value=100, z_step=0.3, xy_step=0.065):
    data = np.full(shape, value, dtype=np.uint16)
    tifffile.imwrite(
        str(path), data, imagej=True,
        resolution=(1 / xy_step, 1 / xy_step),
        metadata={"axes": "ZYX", "spacing": z_step, "unit": "micron"},
    )
    return str(path)


# ------------------------------------------------------------------ grouping


def test_groups_by_series_and_orders_frames_numerically():
    """Cell1_2 must precede Cell1_10; plain string sorting would not."""
    paths = [f"/data/Cell1_{i}.tif" for i in (10, 2, 1, 15, 3)]
    groups, unmatched = group_timelapse(paths, DEFAULT_TIMELAPSE_REGEX)

    assert unmatched == []
    assert len(groups) == 1
    assert groups[0].series == "Cell1"
    assert groups[0].frames == [1, 2, 3, 10, 15]
    assert [p.split("_")[-1] for p in groups[0].paths] == [
        "1.tif", "2.tif", "3.tif", "10.tif", "15.tif"
    ]


def test_separates_distinct_series():
    paths = [f"/d/Cell{c}_{f}.tif" for c in (1, 2, 12) for f in (1, 2)]
    groups, unmatched = group_timelapse(paths, DEFAULT_TIMELAPSE_REGEX)
    assert unmatched == []
    # Natural order, not lexicographic. Frames within a series were always numeric, but
    # the series themselves sorted as strings, so this used to read Cell1, Cell12, Cell2
    # -- and since one series is one barcode row, the picture came out in an order no
    # reader would expect. core.pipeline already natural-sorts the ungrouped path.
    assert [g.series for g in groups] == ["Cell1", "Cell2", "Cell12"]
    assert all(len(g) == 2 for g in groups)


def test_cell1_is_not_confused_with_cell11():
    """A prefix collision would silently merge two cells into one time series."""
    paths = ["/d/Cell1_1.tif", "/d/Cell1_2.tif", "/d/Cell11_1.tif", "/d/Cell11_2.tif"]
    groups, _ = group_timelapse(paths, DEFAULT_TIMELAPSE_REGEX)
    assert {g.series: len(g) for g in groups} == {"Cell1": 2, "Cell11": 2}


def test_channel_suffix_forms_its_own_series():
    """Cell1_centrin_3 groups as 'Cell1_centrin', not as frame 3 of 'Cell1'."""
    paths = ["/d/Cell1_1.tif", "/d/Cell1_2.tif",
             "/d/Cell1_centrin_1.tif", "/d/Cell1_centrin_2.tif"]
    groups, _ = group_timelapse(paths, DEFAULT_TIMELAPSE_REGEX)
    assert {g.series: len(g) for g in groups} == {"Cell1": 2, "Cell1_centrin": 2}


def test_unmatched_files_are_reported_not_dropped_silently():
    paths = ["/d/Cell1_1.tif", "/d/README.tif", "/d/no_number.tif"]
    groups, unmatched = group_timelapse(paths, DEFAULT_TIMELAPSE_REGEX)
    assert [g.series for g in groups] == ["Cell1"]
    assert set(unmatched) == {"/d/README.tif", "/d/no_number.tif"}


def test_duplicate_frame_numbers_raise():
    """Two files in ONE folder claiming the same timepoint: the regex is too greedy."""
    paths = ["/a/Cell1_1.tif", "/a/Cell1_1_dup.tif", "/a/Cell1_2.tif"]
    with pytest.raises(ValueError, match="duplicate frame"):
        group_timelapse(paths, r"^(?P<series>Cell\d+)_(?P<frame>\d+)")


def test_same_name_in_two_folders_is_two_series():
    """find_files walks recursively, so identical numbering in sibling folders is normal.

    Keying on the basename alone merged them: with overlapping frame numbers that raised
    "duplicate frame numbers" from outside run.py's per-series try, aborting the whole
    batch; with disjoint ones (1-15 and 16-30) two experiments silently became a single
    30-timepoint series.
    """
    paths = ["/condA/Cell1_1.tif", "/condA/Cell1_2.tif",
             "/condB/Cell1_1.tif", "/condB/Cell1_2.tif"]
    groups, unmatched = group_timelapse(paths, DEFAULT_TIMELAPSE_REGEX)

    assert not unmatched
    assert len(groups) == 2, "one series per folder"
    assert all(len(g) == 2 for g in groups)
    assert {os.path.dirname(g.paths[0]) for g in groups} == {"/condA", "/condB"}


def test_disjoint_frame_numbers_across_folders_do_not_merge():
    """The silent half of the same defect: no duplicate, so nothing used to complain."""
    paths = [f"/condA/Cell1_{i}.tif" for i in (1, 2)] + \
            [f"/condB/Cell1_{i}.tif" for i in (16, 17)]
    groups, _ = group_timelapse(paths, DEFAULT_TIMELAPSE_REGEX)

    assert len(groups) == 2
    assert sorted(len(g) for g in groups) == [2, 2], \
        "two experiments must not become one 4-timepoint series"


# ------------------------------------------------------------------ reading


def test_read_series_stacks_along_time(tmp_path):
    for i in range(1, 4):
        write_volume(tmp_path / f"Cell1_{i}.tif", value=100 * i)
    groups, _ = group_timelapse([str(p) for p in tmp_path.glob("*.tif")])
    stack = read_series(groups[0])

    assert stack.data.shape == (3, 8, 24, 24)
    assert stack.n_timepoints == 3 and stack.is_timelapse
    assert stack.z_step_um == pytest.approx(0.3)
    assert stack.xy_step_um == pytest.approx(0.065, rel=1e-4)
    # frames must be in ascending order, not filesystem order
    assert [int(stack.data[t].flat[0]) for t in range(3)] == [100, 200, 300]
    assert stack.metadata_source["frames"] == [1, 2, 3]


def test_read_series_rejects_shape_mismatch(tmp_path):
    write_volume(tmp_path / "Cell1_1.tif", shape=(8, 24, 24))
    write_volume(tmp_path / "Cell1_2.tif", shape=(8, 24, 32))
    groups, _ = group_timelapse([str(p) for p in tmp_path.glob("*.tif")])
    with pytest.raises(ValueError, match="not one time series"):
        read_series(groups[0])


def test_read_series_rejects_spacing_mismatch(tmp_path):
    write_volume(tmp_path / "Cell1_1.tif", z_step=0.3)
    write_volume(tmp_path / "Cell1_2.tif", z_step=0.5)
    groups, _ = group_timelapse([str(p) for p in tmp_path.glob("*.tif")])
    with pytest.raises(ValueError, match="z spacing"):
        read_series(groups[0])


# ------------------------------------------------------- shared crop geometry


def moving_mask(shape, t):
    """A block that both moves and changes size between timepoints."""
    mask = np.zeros(shape, bool)
    mask[2 + t : 8 + t, 3 : 9 + t, 4:10] = True
    return mask


def test_every_timepoint_shares_one_grid_and_denominator(monkeypatch):
    """The union bounding box must give all timepoints identical shape.

    Cropping each timepoint to its own mask bbox would produce different array shapes
    (unstackable) and a different denominator per timepoint, so "fraction of volume"
    would drift purely from the crop rather than from the object.
    """
    shape = (20, 20, 20)
    n_t = 4
    masks = np.stack([moving_mask(shape, t) for t in range(n_t)])
    images = np.stack([
        np.where(masks[t], 500, 100).astype(np.uint16) for t in range(n_t)
    ])

    # per-frame bounding boxes genuinely differ, otherwise the test proves nothing
    extents = [tuple(int(c.max() - c.min()) for c in np.where(masks[t])) for t in range(n_t)]
    assert len(set(extents)) > 1, "fixture must have differing per-frame bboxes"

    stack = VolumeStack(
        data=images, z_step_um=1.0, xy_step_um=1.0, exposure_time_s=1.0,
        axes="TZYX", source_path="Cell1_1.tif", channel=0,
        metadata_source={"paths": [f"Cell1_{i}.tif" for i in range(1, n_t + 1)]},
    )

    config = VolumetricConfig()
    config.segmentation_enabled = True
    config.make_isotropic = True
    config.crop_padding_vox = 1

    import analysis.volumetric.run as run_module
    monkeypatch.setattr(
        run_module, "load_segmentation",
        lambda path, shape_zyx, z, xy, cfg: (
            masks[int(path.split("_")[-1].split(".")[0]) - 1], path, 1.0
        ),
    )

    volumes, masks_out, spacing, info, mask_paths = _prepare_geometry(stack, config)

    assert volumes.shape[0] == n_t
    assert masks_out.shape == volumes.shape
    assert info["common_crop"] is True
    # one denominator for the whole series
    assert len({volumes[t].size for t in range(n_t)}) == 1
    # the union box must contain every frame's object
    for t in range(n_t):
        assert masks_out[t].sum() == masks[t].sum(), f"timepoint {t} lost voxels"


def test_single_timepoint_still_crops_to_its_own_mask(monkeypatch):
    """With one timepoint the union is that timepoint, so behaviour is unchanged."""
    shape = (20, 20, 20)
    mask = moving_mask(shape, 0)
    images = np.where(mask, 500, 100).astype(np.uint16)[None]

    stack = VolumeStack(
        data=images, z_step_um=1.0, xy_step_um=1.0, exposure_time_s=1.0,
        axes="ZYX", source_path="Cell1_1.tif", channel=0, metadata_source={},
    )
    config = VolumetricConfig()
    config.segmentation_enabled = True
    config.crop_padding_vox = 1

    import analysis.volumetric.run as run_module
    monkeypatch.setattr(
        run_module, "load_segmentation", lambda *a, **k: (mask, "Cell1_1.tif", 1.0)
    )

    volumes, masks_out, _, info, _ = _prepare_geometry(stack, config)
    assert volumes.shape[0] == 1
    assert info["common_crop"] is False
    assert masks_out[0].sum() == mask.sum()
