"""Meshing every object, not just the largest.

``mesh.mesh_object`` begins with ``largest_component``, so feeding it a field of cells
measured one cell and reported its shape as the field's. That is right for a single
nucleus and wrong for 839 of them, and it is silent either way.

Run: python -m pytest tests/test_object_mesh.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from analysis.volumetric.object_mesh import (
    DEFAULT_OBJECT_MAXRAD,
    DEFAULT_OBJECT_MAXRAD_UNITS,
    VOLUME_RATIO_LIMITS,
    bounding_boxes,
    mesh_objects,
)
from core import BarcodeConfig

ISO = (0.2, 0.2, 0.2)


def spheres(radii=(6, 5), shape=(28, 28, 60)):
    """Well-separated spheres of known radius, with non-contiguous label ids."""
    labels = np.zeros(shape, np.int32)
    ids = [7, 19][:len(radii)]
    zz, yy, xx = np.indices(shape)
    for k, (object_id, radius) in enumerate(zip(ids, radii)):
        cz, cy, cx = shape[0] // 2, shape[1] // 2, 14 + 30 * k
        inside = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
        labels[inside] = object_id
    return labels, ids


def config(**kwargs):
    cfg = BarcodeConfig().volumetric
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


def test_bounding_boxes_handle_non_contiguous_ids():
    labels, ids = spheres()
    boxes = bounding_boxes(labels)
    assert sorted(boxes) == sorted(ids), "Cellpose ids are not 1..N"


def test_every_object_is_meshed_not_just_the_largest():
    labels, ids = spheres(radii=(6, 5))
    meshes, detail = mesh_objects(labels, ISO, config())
    assert sorted(meshes) == sorted(ids)
    assert len(detail.meshed) == 2, "the smaller object must not be dropped"


def test_each_mesh_describes_its_own_object():
    """The failure being fixed: one object's shape reported for all of them."""
    labels, ids = spheres(radii=(7, 4))
    meshes, _ = mesh_objects(labels, ISO, config())
    volumes = {i: meshes[i].geometry.volume_um3 for i in meshes}
    assert volumes[ids[0]] > 2 * volumes[ids[1]], (
        "the big and small spheres must not report the same volume")


def test_mesh_volume_tracks_the_voxel_count():
    labels, ids = spheres(radii=(6, 5))
    counts = np.bincount(labels.ravel())
    voxel = float(np.prod(ISO))
    meshes, _ = mesh_objects(labels, ISO, config())
    for object_id, mesh in meshes.items():
        ratio = mesh.geometry.volume_um3 / (counts[object_id] * voxel)
        assert 0.85 < ratio < 1.15, f"object {object_id} ratio {ratio:.3f}"


def test_a_degenerate_mesh_is_rejected_not_reported():
    """A collapsed surface raised nothing and produced a sphericity anyway.

    Measured on real data: an object of 93.7 um^3 meshed to 0.2 um^3.
    """
    labels, _ = spheres(radii=(6,))
    counts = np.bincount(labels.ravel())
    voxel = float(np.prod(ISO))

    meshes, detail = mesh_objects(labels, ISO, config())
    assert meshes, "sanity: this should mesh cleanly"
    for object_id, mesh in meshes.items():
        ratio = mesh.geometry.volume_um3 / (counts[object_id] * voxel)
        assert VOLUME_RATIO_LIMITS[0] <= ratio <= VOLUME_RATIO_LIMITS[1]


def test_tiny_objects_are_skipped_with_a_reason():
    labels = np.zeros((16, 16, 16), np.int32)
    labels[8, 8, 8] = 3                       # one voxel
    meshes, detail = mesh_objects(labels, ISO, config())
    assert meshes == {}
    assert detail.too_small == 1
    assert "voxels" in detail.reasons[3]


def test_border_objects_are_meshed_but_flagged():
    """Their surface is truncated, so sphericity describes a cut shape."""
    labels = np.zeros((20, 20, 20), np.int32)
    labels[:8, 4:14, 4:14] = 5                # runs into z = 0
    _, detail = mesh_objects(labels, ISO, config())
    assert 5 in detail.border
    assert "border" in detail.describe()


def test_an_empty_volume_is_not_an_error():
    meshes, detail = mesh_objects(np.zeros((8, 8, 8), np.int32), ISO, config())
    assert meshes == {} and detail.meshed == []


def test_the_default_triangle_bound_is_relative():
    """5 voxels was tuned for a ~65-voxel-radius nucleus; a cell is 12-17, where it
    costs tens of percent of the volume."""
    assert DEFAULT_OBJECT_MAXRAD_UNITS == "relative"
    assert DEFAULT_OBJECT_MAXRAD == pytest.approx(0.1)
    assert BarcodeConfig().volumetric.object_mesh is False, "must be opt-in: ~2.5 s/object"


def test_shape_columns_reach_object_rows():
    from analysis.volumetric.objects import extract_objects

    labels, ids = spheres(radii=(6, 5))
    meshes, _ = mesh_objects(labels, ISO, config())
    rows = {r.object_id: r for r in extract_objects(labels, ISO, meshes=meshes)}
    for object_id in ids:
        assert np.isfinite(rows[object_id].sphericity)
        assert np.isfinite(rows[object_id].surface_area)
    assert rows[ids[0]].sphericity == pytest.approx(1.0, abs=0.15), "a sphere is spherical"


def test_rows_without_a_mesh_are_nan_not_borrowed():
    from analysis.volumetric.objects import extract_objects

    labels, ids = spheres(radii=(6, 5))
    meshes, _ = mesh_objects(labels, ISO, config())
    partial = {ids[0]: meshes[ids[0]]}          # second object has no mesh
    rows = {r.object_id: r for r in extract_objects(labels, ISO, meshes=partial)}
    assert np.isfinite(rows[ids[0]].sphericity)
    assert np.isnan(rows[ids[1]].sphericity), "must not inherit another object's shape"
