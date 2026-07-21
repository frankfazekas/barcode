"""Reading masks from whatever the segmenter wrote (stream B3).

The load-bearing property is that **labels survive**. An instance segmentation of a
confluent tissue is the only thing that distinguishes touching cells; a stray
``astype(bool)`` anywhere on the path turns 500 cells into one component, and every
object metric then reports a plausible number for the wrong object. So the tests assert
on label content, not just shape.

Run: python -m pytest tests/test_mask_io.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

from analysis.volumetric.mask_io import (
    load_mask_array,
    mask_loader,
    supported_mask_extensions,
)


def labelled(n_z=4, size=12):
    """Four stacked slabs carrying four distinct instance labels."""
    volume = np.zeros((n_z, size, size), np.int32)
    volume[:, 1:5, 1:5] = 1
    volume[:, 1:5, 6:10] = 2
    volume[:, 6:10, 1:5] = 3
    volume[:, 6:10, 6:10] = 7      # deliberately not 4: labels need not be contiguous
    return volume


# ------------------------------------------------------------------ formats


def test_tiff_masks_keep_their_labels(tmp_path):
    source = labelled()
    path = tmp_path / "m.tif"
    tifffile.imwrite(str(path), source)
    loaded = load_mask_array(str(path))
    assert np.array_equal(loaded, source)
    assert sorted(np.unique(loaded)) == [0, 1, 2, 3, 7]


def test_a_bare_npy_label_array_loads(tmp_path):
    source = labelled()
    path = tmp_path / "m.npy"
    np.save(str(path), source)
    assert np.array_equal(load_mask_array(str(path)), source)


def test_a_cellpose_seg_npy_bundle_loads(tmp_path):
    """Cellpose np.saves a dict, which numpy returns as a 0-d object array."""
    source = labelled()
    bundle = {
        "masks": source,
        "outlines": np.zeros_like(source),
        "flows": [np.zeros((3,) + source.shape, np.float32)],
        "diams": 30.0,
        "filename": "whatever.tif",
    }
    path = tmp_path / "img_seg.npy"
    np.save(str(path), bundle)                    # exactly how Cellpose writes it

    loaded = load_mask_array(str(path))
    assert np.array_equal(loaded, source), "the 'masks' entry, not 'outlines'"
    assert sorted(np.unique(loaded)) == [0, 1, 2, 3, 7]


def test_npz_with_one_array_loads(tmp_path):
    source = labelled()
    path = tmp_path / "m.npz"
    np.savez(str(path), source)
    assert np.array_equal(load_mask_array(str(path)), source)


def test_npz_with_several_arrays_picks_masks(tmp_path):
    source = labelled()
    path = tmp_path / "m.npz"
    np.savez(str(path), outlines=np.zeros_like(source), masks=source)
    assert np.array_equal(load_mask_array(str(path)), source)


def test_a_bundle_without_a_recognised_key_says_what_it_holds(tmp_path):
    path = tmp_path / "m.npz"
    np.savez(str(path), flows=np.zeros((2, 2)), probability=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="flows"):
        load_mask_array(str(path))


# ------------------------------------------------------------------ registry


def test_an_unknown_extension_lists_what_is_supported(tmp_path):
    path = tmp_path / "m.mha"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match=r"no mask loader for '\.mha'"):
        load_mask_array(str(path))
    try:
        load_mask_array(str(path))
    except ValueError as exc:
        for extension in (".tif", ".npy", ".npz"):
            assert extension in str(exc), "the message must advertise the registry"


def test_a_new_format_needs_only_a_registration(tmp_path):
    """The point of the registry: one decorator, no edits anywhere else."""
    assert ".fake" not in supported_mask_extensions()

    @mask_loader(".fake")
    def _load_fake(path):
        return labelled()

    try:
        assert ".fake" in supported_mask_extensions()
        path = tmp_path / "m.fake"
        path.write_bytes(b"")
        assert np.array_equal(load_mask_array(str(path)), labelled())
    finally:
        from analysis.volumetric import mask_io
        mask_io._LOADERS.pop(".fake", None)


def test_extensions_are_matched_case_insensitively(tmp_path):
    source = labelled()
    path = tmp_path / "m.TIF"
    tifffile.imwrite(str(path), source)
    assert np.array_equal(load_mask_array(str(path)), source)


# ------------------------------------------------------------- through the pipeline


def write_image(path, n_z=4, size=12):
    data = np.zeros((n_z, size, size), np.uint16)
    data[:, 1:10, 1:10] = 400
    tifffile.imwrite(str(path), data, imagej=True,
                     metadata={"axes": "ZYX", "spacing": 1.0},
                     resolution=(1.0, 1.0))
    return str(path)


def segmentation_config(root, template="{stem}_SegMask.npy"):
    from core import BarcodeConfig

    cfg = BarcodeConfig().volumetric
    cfg.segmentation_enabled = True
    cfg.segmentation_root = str(root)
    cfg.segmentation_template = template
    cfg.mask_spacing_um = 0.0
    return cfg


def test_a_cellpose_npy_mask_reaches_load_segmentation_with_labels(tmp_path):
    """End to end: the pairing layer resolves it and the labels are still there."""
    from analysis.volumetric.segmentation import load_segmentation

    image = write_image(tmp_path / "Cell1.tif")
    np.save(str(tmp_path / "Cell1_SegMask.npy"), {"masks": labelled()})

    config = segmentation_config(tmp_path)
    mask, path, spacing = load_segmentation(image, (4, 12, 12), 1.0, 1.0, config)

    assert path.endswith("Cell1_SegMask.npy")
    assert mask.dtype != bool, "an instance mask must not be collapsed"
    assert sorted(np.unique(mask)) == [0, 1, 2, 3, 7]


def test_a_single_object_mask_still_becomes_boolean(tmp_path):
    """'auto' keeps labels only when the mask actually distinguishes objects."""
    from analysis.volumetric.segmentation import load_segmentation

    image = write_image(tmp_path / "Cell1.tif")
    single = np.zeros((4, 12, 12), np.int32)
    single[:, 2:8, 2:8] = 1
    np.save(str(tmp_path / "Cell1_SegMask.npy"), single)

    mask, _, _ = load_segmentation(
        image, (4, 12, 12), 1.0, 1.0, segmentation_config(tmp_path))
    assert mask.dtype == bool


def test_the_z_grid_matcher_preserves_labels():
    """The regression this guards: it used to force bool, undoing the whole feature."""
    from analysis.volumetric.segmentation import match_mask_to_image_grid

    fine = np.repeat(labelled(n_z=4)[None], 1, axis=0)[0]
    fine = np.repeat(fine, 5, axis=0)             # 20 mask planes for 4 image slices
    matched = match_mask_to_image_grid(fine, 4)

    assert matched.shape[0] == 4
    assert matched.dtype != bool
    assert sorted(np.unique(matched)) == [0, 1, 2, 3, 7]


def test_an_equal_depth_mask_is_passed_through_unchanged():
    from analysis.volumetric.segmentation import match_mask_to_image_grid

    source = labelled(n_z=4)
    matched = match_mask_to_image_grid(source, 4)
    assert np.array_equal(matched, source) and matched.dtype != bool


def test_labels_survive_all_the_way_into_packing_topology():
    """The reason any of this matters: contact number needs the instances intact."""
    from analysis.volumetric.packing import packing_topology
    from core import BarcodeConfig

    tile = np.zeros((30, 30), np.int32)
    for i in range(3):
        for j in range(3):
            tile[i * 10:(i + 1) * 10, j * 10:(j + 1) * 10] = i * 3 + j + 1
    labels = np.pad(np.repeat(tile[None], 6, axis=0), ((2, 2), (0, 0), (0, 0)))

    cfg = BarcodeConfig().volumetric
    cfg.packing_contact_dilation_vox = 0
    cfg.packing_min_contact_voxels = 1
    results, detail = packing_topology(labels, cfg)
    assert detail.n_objects == 9

    # The same field collapsed to boolean, which is what the old path produced.
    from skimage.measure import label as cc_label
    collapsed = cc_label(labels > 0, connectivity=3, return_num=True)[1]
    assert collapsed == 1, "one component: every contact number would be 0"
