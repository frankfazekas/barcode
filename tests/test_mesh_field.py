"""Per-object meshing of a labelled field (``analysis/volumetric/mesh_field.py``).

The claim worth pinning is that a confluent field stays many objects: the single-object
path collapses it to the largest component, which for an epithelium throws the tissue
away.
"""
import numpy as np
import pytest

from analysis.volumetric.mesh import largest_component
from analysis.volumetric.mesh_field import (
    iter_objects,
    mesh_field,
    touches_border,
)


def _touching_blobs(shape=(24, 60, 60)):
    """Three cuboids sharing faces, plus one running off the x edge."""
    labels = np.zeros(shape, dtype=np.uint16)
    labels[4:20, 10:26, 10:26] = 1
    labels[4:20, 26:42, 10:26] = 2      # shares a face with 1
    labels[4:20, 10:26, 26:42] = 3      # shares a face with 1
    labels[4:20, 44:58, 0:12] = 4       # touches x = 0 -> cut by the frame
    return labels


# --------------------------------------------------------------------------- #
# object partitioning
# --------------------------------------------------------------------------- #
def test_border_detection_ignores_z_by_design():
    """A shallow slab has every object touching top and bottom; only xy cuts count."""
    shape = (14, 100, 100)
    spanning_z = (slice(0, 14), slice(20, 40), slice(20, 40))
    at_x_edge = (slice(3, 10), slice(20, 40), slice(0, 20))

    assert not touches_border(spanning_z, shape, "xy")
    assert touches_border(spanning_z, shape, "all")
    assert touches_border(at_x_edge, shape, "xy")
    assert not touches_border(spanning_z, shape, "none")


def test_iter_objects_keeps_touching_instances_separate():
    labels = _touching_blobs()
    kept = {lab: sub for lab, sub, _, skip in iter_objects(labels) if not skip}

    assert set(kept) == {1, 2, 3}                 # 4 is cut by the x edge
    for lab, sub in kept.items():
        assert sub.sum() == (labels == lab).sum()  # each carries only its own voxels


def test_iter_objects_reports_why_it_skipped():
    labels = _touching_blobs()
    labels[4:6, 50:52, 30:32] = 5                 # a tiny fragment
    reasons = {lab: skip for lab, _, _, skip in iter_objects(labels, min_voxels=64)}
    assert reasons[4] == "border"
    assert reasons[5] == "small"
    assert reasons[1] == ""


def test_offsets_place_objects_back_in_field_coordinates():
    labels = _touching_blobs()
    for lab, sub, offset, skip in iter_objects(labels):
        if skip:
            continue
        # The crop plus its offset must reproduce the object's true bounding box.
        coords = np.argwhere(sub) + offset
        truth = np.argwhere(labels == lab)
        assert coords.min(axis=0).tolist() == truth.min(axis=0).tolist()
        assert coords.max(axis=0).tolist() == truth.max(axis=0).tolist()


def test_single_object_path_would_collapse_the_field():
    """Why this module exists: largest_component keeps one object out of four."""
    labels = _touching_blobs()
    collapsed = largest_component(labels > 0)
    assert collapsed.sum() > (labels == 1).sum()   # the touching blobs merged into one


def test_rejects_a_non_3d_volume():
    from analysis.volumetric.mesh import MeshingError

    with pytest.raises(MeshingError):
        list(iter_objects(np.zeros((10, 10), dtype=np.uint16)))


def test_rejects_an_unknown_border_mode():
    with pytest.raises(ValueError, match="exclude_border"):
        list(iter_objects(_touching_blobs(), exclude_border="sideways"))


# --------------------------------------------------------------------------- #
# meshing the field
# --------------------------------------------------------------------------- #
def _iso2mesh_available():
    try:
        from iso2mesh import v2s  # noqa: F401
    except ImportError:
        return False
    return True


needs_iso2mesh = pytest.mark.skipif(
    not _iso2mesh_available(), reason="pyiso2mesh is not installed"
)


@needs_iso2mesh
def test_mesh_field_returns_one_mesh_per_object():
    labels = _touching_blobs()
    field = mesh_field(labels, (0.2, 0.2, 0.2), min_voxels=64, curvature=False)

    assert len(field) == 3
    assert sorted(m.label for m in field.meshes) == [1, 2, 3]
    assert field.skipped_border == [4]
    assert not field.failed


@needs_iso2mesh
def test_meshed_volumes_track_the_voxel_volumes():
    labels = _touching_blobs()
    field = mesh_field(labels, (0.2, 0.2, 0.2), min_voxels=64, curvature=False)

    voxel_volume = 0.2 ** 3
    for mesh in field.meshes:
        truth = (labels == mesh.label).sum() * voxel_volume
        assert mesh.geometry.volume_um3 == pytest.approx(truth, rel=0.15)
        assert not mesh.geometry.has_holes


@needs_iso2mesh
def test_objects_keep_their_relative_positions():
    """Vertices are in field coordinates, so neighbours must not sit on top of each other."""
    labels = _touching_blobs()
    field = mesh_field(labels, (0.2, 0.2, 0.2), min_voxels=64, curvature=False)
    centroids = {m.label: m.centroid_um for m in field.meshes}

    # Object 2 is offset from 1 along y, object 3 along x.
    assert centroids[2][1] > centroids[1][1] + 1.0
    assert centroids[3][2] > centroids[1][2] + 1.0
    assert centroids[2][2] == pytest.approx(centroids[1][2], abs=0.5)


@needs_iso2mesh
def test_curvature_is_attached_when_requested():
    labels = _touching_blobs()
    field = mesh_field(labels, (0.2, 0.2, 0.2), min_voxels=64, curvature=True)
    assert all(m.curvature is not None for m in field.meshes)
    # A cuboid is convex, so essentially no invaginated area.
    assert all(m.curvature.invagination_ratio < 0.35 for m in field.meshes)


@needs_iso2mesh
def test_anisotropic_spacing_is_refused():
    from analysis.volumetric.mesh import MeshingError

    with pytest.raises(MeshingError, match="isotropic"):
        mesh_field(_touching_blobs(), (0.235, 0.195, 0.195))
