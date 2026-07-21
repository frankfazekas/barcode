"""Regressions for defects found auditing the volumetric modules.

Each test states the wrong behaviour it pins down, because in every case the old code
produced a plausible number rather than an error -- which is why none of them was caught.

Run: python -m pytest tests/test_volumetric_regressions.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

from core import BarcodeConfig


# ------------------------------------------------------------------ z range + mask


def write_stack(path, n_z=54, z_um=0.3, xy_um=0.065, shape=(24, 24)):
    data = np.random.RandomState(0).randint(100, 4000, (n_z,) + shape).astype(np.uint16)
    tifffile.imwrite(
        str(path), data, imagej=True,
        resolution=(1 / xy_um, 1 / xy_um),
        metadata={"axes": "ZYX", "spacing": z_um, "unit": "micron"},
    )
    return str(path)


def test_a_z_range_keeps_a_finer_mask_aligned_with_the_image(tmp_path):
    """The mask is on a finer grid than the acquisition, so it needs mapping, not slicing.

    The old guard only cropped the mask when it had exactly as many slices as the acquired
    stack. On the real 54-slice/250-slice pairing that was never true, so the image was
    restricted and the mask was not; `prepare_volume` then resampled from origin 0 and
    planted the sub-range at the bottom of the field. With the object at mask z 100..150
    the image landed at z 0..89 -- no overlap at all, and every mask-gated metric measured
    background.
    """
    from analysis.volumetric.run import _mask_z_slice_for_range

    # 54 acquired slices at 0.3 um <-> 250 mask slices at 0.065 um.
    start, stop = _mask_z_slice_for_range(250, 54, 20, 40)

    # Acquired slice 20 sits at 20/53 of the depth, i.e. mask slice ~94.
    assert start == pytest.approx(round(20 * 249 / 53), abs=1)
    assert stop == pytest.approx(round(39 * 249 / 53) + 1, abs=1)
    # And it must actually be a sub-range, which is what the old code never produced.
    assert 0 < start < stop < 250


def test_the_mask_slice_map_is_identity_on_a_matching_grid():
    from analysis.volumetric.run import _mask_z_slice_for_range

    assert _mask_z_slice_for_range(54, 54, 10, 30) == (10, 30)


def test_masked_metrics_still_see_the_object_after_a_z_restriction(tmp_path):
    """End to end: restrict z, and the object must still be inside its own mask."""
    from analysis.volumetric.reader import apply_z_range, read_volume
    from analysis.volumetric.run import _prepare_geometry

    image_path = write_stack(tmp_path / "Cell9_1.tif", n_z=54)

    # A 250-slice isotropic mask holding one object in the middle of the depth.
    mask = np.zeros((250, 24, 24), np.uint8)
    mask[110:150, 8:16, 8:16] = 1
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    tifffile.imwrite(str(masks_dir / "Cell9_1_SegMask.tif"), mask)

    config = BarcodeConfig().volumetric
    config.segmentation_enabled = True
    config.segmentation_root = str(masks_dir)
    config.mask_spacing_um = 0                 # isotropic at the xy step
    # The object spans mask slices 110..150 => acquired slices ~24..32.
    config.z_start, config.z_end = 20, 40

    stack = apply_z_range(read_volume(image_path), config)
    volumes, masks_iso, _, _, _ = _prepare_geometry(stack, config)

    assert masks_iso is not None
    assert masks_iso.shape[1:] == volumes.shape[1:]
    assert masks_iso.any(), "the analysed slab must still contain the segmented object"
    # The image is real data everywhere the mask is, not the zero padding that a
    # misaligned resample leaves behind.
    assert volumes[0][masks_iso[0].astype(bool)].any()


# ------------------------------------------------------------------ flow operators


def uniform_expansion(shape=(9, 12, 12, 3), k=0.5, spacing=(1.0, 1.0, 1.0)):
    """v = k*r about the centre, components ordered (x, y, z) as the module expects."""
    dz, dy, dx = spacing
    nz, ny, nx = shape[:3]
    z = (np.arange(nz) - (nz - 1) / 2) * dz
    y = (np.arange(ny) - (ny - 1) / 2) * dy
    x = (np.arange(nx) - (nx - 1) / 2) * dx
    field = np.zeros(shape)
    field[..., 0] = k * x[None, None, :]
    field[..., 1] = k * y[None, :, None]
    field[..., 2] = k * z[:, None, None]
    return field


def test_divergence_of_a_uniform_expansion_is_three_k():
    """`vy` used to be negated without flipping the y axis, giving dvx/dx - dvy/dy + dvz/dz.

    A uniform expansion then reported k instead of 3k. The sign convention is a reflection:
    doing half of it produces a field that is not in any coordinate frame.
    """
    from analysis.volumetric.flow import _divergence_3d

    spacing = (0.3, 0.065, 0.065)
    field = uniform_expansion(spacing=spacing, k=0.5)
    divergence = _divergence_3d(field, spacing)

    assert np.allclose(divergence, 1.5), "3k for k = 0.5"


def test_curl_of_pure_shear_is_zero():
    """vx = w*y, vy = w*x is irrotational; a lone sign flip reported 2w."""
    from analysis.volumetric.flow import _curl_magnitude_3d

    spacing = (1.0, 1.0, 1.0)
    nz, ny, nx = 7, 10, 10
    y = np.arange(ny) - (ny - 1) / 2
    x = np.arange(nx) - (nx - 1) / 2

    field = np.zeros((nz, ny, nx, 3))
    field[..., 0] = 0.25 * y[None, :, None]
    field[..., 1] = 0.25 * x[None, None, :]

    assert np.allclose(_curl_magnitude_3d(field, spacing), 0.0, atol=1e-12)


def test_the_reported_direction_is_still_y_up():
    """The convention survives; it is applied to the direction vector, not the field."""
    from analysis.volumetric.flow import _assemble_results, VolumetricFlowDetail

    detail = VolumetricFlowDetail(speeds=[1.0], correlation_lengths=[np.nan])
    # A mean unit vector pointing +y in ARRAY coordinates (downwards on screen).
    results = _assemble_results(
        detail, [np.array([0.0, -1.0, 0.0])], [1.0], BarcodeConfig().volumetric)

    assert results.mean_theta == pytest.approx(-np.pi / 2), \
        "array +y is screen-down, so the reported azimuth is negative"


# ------------------------------------------------------------------ xyz mode + mask


def test_masked_intensity_over_z_returns_numbers_not_swallowed_nans():
    """NaN-blanking made np.histogram raise, and the caller's except hid it."""
    from analysis.volumetric.slicewise import _masked_intensity_over_z

    volume = np.random.RandomState(1).randint(100, 4000, (12, 32, 32)).astype(np.float64)
    masks = np.zeros((12, 32, 32), bool)
    masks[3:9, 8:24, 8:24] = True

    results = _masked_intensity_over_z(
        volume, masks, BarcodeConfig().intensity_distribution_parameters)

    assert np.isfinite(results.max_kurtosis)
    assert np.isfinite(results.max_median_skew)
    assert np.isfinite(results.max_mode_skew)


def test_masked_intensity_over_z_survives_an_empty_mask():
    from analysis.volumetric.slicewise import _masked_intensity_over_z

    volume = np.ones((6, 8, 8))
    results = _masked_intensity_over_z(
        volume, np.zeros((6, 8, 8), bool),
        BarcodeConfig().intensity_distribution_parameters)

    assert np.isnan(results.max_kurtosis)


def test_an_empty_mask_slice_is_not_a_full_field_island():
    """binarize thresholds at the frame mean, so an all-zero slice came back all ones."""
    from utils.binarization import binarize

    empty = np.zeros((8, 8))
    assert binarize(empty, 0.0, 1, 1).sum() == 64, \
        "documents the 2D primitive's behaviour, which is deliberately not changed"

    # slicewise therefore drops empty slices before handing the mask over.
    masks = np.zeros((10, 8, 8), bool)
    masks[4:7, 2:6, 2:6] = True
    occupied = masks.any(axis=(1, 2))
    assert occupied.sum() == 3
    kept = masks[occupied].astype(np.float64)
    assert all(binarize(s, 0.0, 1, 1).sum() < 64 for s in kept)


# ------------------------------------------------------------------ voxel size units


@pytest.mark.parametrize("unit, expected", [
    ("micron", 0.065),
    ("nm", 65e-6),
    ("pixel", None),
    ("cm", 650.0),
])
def test_xy_spacing_honours_the_unit_the_file_states(tmp_path, unit, expected):
    """XResolution is a bare number; assuming microns mis-scaled every physical metric.

    "pixel" is the dangerous one: an uncalibrated ImageJ file writes XResolution 1/1, which
    reads as a perfectly ordinary 1.0 um/px, so the "no spacing, assuming 1.0" warning
    could never fire for exactly the files that needed it.
    """
    from analysis.volumetric.reader import _xy_spacing_from_tags

    px_per_unit = 1 / 0.065 if unit != "pixel" else 1.0
    path = tmp_path / f"u_{unit}.tif"
    tifffile.imwrite(str(path), np.zeros((4, 8, 8), np.uint16), imagej=True,
                     resolution=(px_per_unit, px_per_unit),
                     metadata={"axes": "ZYX", "unit": unit})

    with tifffile.TiffFile(str(path)) as handle:
        spacing = _xy_spacing_from_tags(handle.pages[0], unit)

    if expected is None:
        assert spacing is None
    else:
        assert spacing == pytest.approx(expected, rel=1e-4)


def test_a_nanometre_calibrated_stack_is_read_in_microns(tmp_path):
    from analysis.volumetric.reader import read_volume

    path = tmp_path / "nm.tif"
    tifffile.imwrite(str(path), np.zeros((6, 8, 8), np.uint16), imagej=True,
                     resolution=(1 / 65.0, 1 / 65.0),
                     metadata={"axes": "ZYX", "spacing": 300.0, "unit": "nm"})

    stack = read_volume(str(path))
    assert stack.xy_step_um == pytest.approx(0.065, rel=1e-4)
    assert stack.z_step_um == pytest.approx(0.3, rel=1e-6)


# ------------------------------------------------------------------ meshing


def test_an_object_touching_a_face_is_still_meshed_closed():
    """Clipping the one-voxel margin left an open surface, and an open surface has no
    translation-invariant volume -- in field coordinates the error swamped the signal."""
    from analysis.volumetric.mesh_field import iter_objects

    labels = np.zeros((10, 20, 20), np.int32)
    labels[:, 6:14, 6:14] = 1                     # spans the full depth

    (label_id, sub, offset, reason), = list(
        iter_objects(labels, min_voxels=1, exclude_border="xy"))

    assert reason == "" and label_id == 1
    faces = [sub.take(i, axis=axis).any() for axis in range(3) for i in (0, -1)]
    assert not any(faces), "background on every face, so the surface closes"
    assert offset[0] == -1, "the pad shifts the crop origin outside the array, correctly"


def test_written_obj_is_outward_wound():
    """(z,y,x) -> (x,y,z) reverses handedness, so the faces have to be reversed too."""
    import os
    import tempfile

    from analysis.volumetric.mesh import mesh_volume, write_obj

    # A unit tetrahedron, outward-wound in the package's own frame.
    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = np.array([[1, 3, 2], [1, 2, 4], [1, 4, 3], [2, 3, 4]])
    assert mesh_volume(vertices, faces) > 0, "the fixture itself must be outward-wound"

    path = os.path.join(tempfile.mkdtemp(), "t.obj")
    write_obj(path, vertices, faces)

    read_v, read_f = [], []
    for line in open(path, encoding="utf-8"):
        if line.startswith("v "):
            read_v.append([float(v) for v in line.split()[1:]])
        elif line.startswith("f "):
            read_f.append([int(v) for v in line.split()[1:]])

    assert mesh_volume(np.array(read_v), np.array(read_f)) > 0, \
        "outward-wound in the OBJ's right-handed x/y/z frame too"


# ------------------------------------------------------------------ curvature


def test_uncomputable_curvature_is_not_counted_as_convex():
    """NaN comparisons are False, so degenerate faces fell through to CONVEX and then
    padded the invagination denominator without ever reaching a numerator."""
    from analysis.volumetric.curvature import (
        CONCAVE, CONVEX, UNCLASSIFIED, classify_concavity, invagination_ratios)

    k_min = np.array([1.0, -1.0, -1.0, np.nan])
    k_max = np.array([1.0, 1.0, -0.5, np.nan])
    classes = classify_concavity(k_min, k_max)

    assert classes[0] == CONVEX
    assert classes[2] == CONCAVE
    assert classes[3] == UNCLASSIFIED

    areas = np.ones(4)
    excluded = np.zeros(4, bool)
    invagination, concave = invagination_ratios(classes, areas, excluded)

    # Three classifiable faces: one convex, one saddle, one concave.
    assert concave == pytest.approx(1 / 3), "the NaN face leaves the denominator too"
    assert invagination == pytest.approx(2 / 3)


def test_a_mesh_too_flat_to_bin_excludes_nothing():
    """One or two bins made 'lowest or highest bin' cover every face, so every curvature
    metric came back NaN -- reported as missing data rather than as 'too flat'."""
    from analysis.volumetric.curvature import identify_bottom_top_faces

    mask, fraction = identify_bottom_top_faces(np.array([0.0, 0.02, 0.05, 0.08]))
    assert not mask.any() and fraction == 0.0


# ------------------------------------------------------- mesh: closed surfaces


def test_the_pipeline_mesher_closes_an_object_touching_a_face():
    """`mesh_field` padded; `mesh.generate_mesh` -- the path the pipeline uses -- did not.

    `crop_to_mask` is False by default so the mask spans the acquired field, and a nucleus
    clipped by the top or bottom of the stack is the expected case, so this is the normal
    situation rather than an edge case. An open surface has no translation-invariant
    volume: `mesh_volume` is a signed tetrahedron sum from the origin, so a hole at height
    z contributes about A_hole*z/3, of the same order as the object itself in field
    coordinates. It also makes the winding sign unreliable, which `curvature` uses to
    decide whether to flip every face.
    """
    from analysis.volumetric.mesh import mesh_nucleus

    mask = np.zeros((24, 40, 40), bool)
    mask[:, 12:28, 12:28] = True          # spans the full depth: touches both z faces

    geometry = mesh_nucleus(mask, (0.1, 0.1, 0.1), maxrad=2.0).geometry
    assert not geometry.has_holes, "a border-touching object must still mesh closed"
    assert geometry.outward, "a closed outward surface must have positive signed volume"
    assert geometry.volume_um3 > 0


def test_padding_leaves_an_interior_object_unchanged():
    """The pad must be free for meshes that already closed, or it rewrites past results."""
    from analysis.volumetric.mesh import generate_mesh

    mask = np.zeros((30, 30, 30), np.uint8)
    zz, yy, xx = np.ogrid[:30, :30, :30]
    mask[((zz - 15) ** 2 + (yy - 15) ** 2 + (xx - 15) ** 2) <= 81] = 1  # nowhere near a face

    vertices, faces = generate_mesh(mask, maxrad=2.0)
    # Vertices come back in the caller's frame, so they still sit around the ball's centre
    # rather than shifted by the pad.
    assert 10.0 < float(vertices[:, 0].mean()) < 20.0
    assert faces.shape[0] > 0


# ------------------------------------------------------- time-lapse: even sampling


def test_a_series_with_a_missing_timepoint_is_refused():
    """One dropped acquisition silently became a 2*dt step hidden inside the T axis.

    Speed divides by a single frame interval and the flow solver takes a contiguous
    window, so a gap mis-scales both with nothing in the output to reveal it.
    """
    import pytest

    from analysis.volumetric.timelapse import SeriesGroup, read_series

    group = SeriesGroup(series="Cell1",
                        paths=["/d/Cell1_1.tif", "/d/Cell1_2.tif", "/d/Cell1_4.tif"],
                        frames=[1, 2, 4])
    with pytest.raises(ValueError, match=r"missing timepoint\(s\) \[3\]"):
        read_series(group)


# ------------------------------------------------------- mask/image z registration


def test_the_mask_is_mapped_to_the_image_by_physical_depth():
    """Endpoint anchoring imposed (M-1)/(N-1) instead of the true z_step/mask_spacing.

    On the working geometry -- 250 mask planes at 0.065 um against 54 image slices at
    0.3 um -- that is 4.698 against 4.615, a 1.8% stretch that drifts to about one whole
    image slice by the top of the stack, worst exactly where a nuclear mask tapers.
    """
    from analysis.volumetric.segmentation import match_mask_to_image_grid

    mask = np.arange(250).reshape(250, 1, 1)      # each plane labelled by its own index
    matched = match_mask_to_image_grid(mask, 54, 0.3, 0.065)

    assert matched.shape[0] == 54
    # Image slice i sits at i * 0.3 um, i.e. mask plane i * 0.3/0.065.
    for i in (0, 13, 27, 40, 53):
        assert matched[i, 0, 0] == round(i * 0.3 / 0.065)

    # Without the spacings it falls back to the old endpoint mapping, which disagrees.
    legacy = match_mask_to_image_grid(mask, 54)
    assert legacy[53, 0, 0] == 249 and matched[53, 0, 0] == 245
