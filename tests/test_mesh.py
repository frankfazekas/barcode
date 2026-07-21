"""Meshing of a segmented 3D nucleus (``analysis/volumetric/mesh.py``).

The geometry helpers are checked against shapes whose volume and area are known
exactly and need no iso2mesh; the meshing chain itself is checked against a
rasterised sphere and skipped when the CGAL binaries are unavailable.
"""
import numpy as np
import pytest
from scipy import ndimage

from analysis.volumetric.mesh import (
    MeshingError,
    aspect_ratio,
    convex_hull_volume,
    convex_hull_voxel_count,
    face_areas,
    face_centroids,
    gibbon_patch_area,
    largest_component,
    mesh_geometry,
    mesh_has_holes,
    mesh_object,
    mip_axis_lengths,
    mesh_series,
    mesh_volume,
    write_obj,
)


def _ball(shape=(48, 48, 48), radius=16.0):
    zz, yy, xx = np.indices(shape)
    centre = (np.array(shape) - 1) / 2.0
    return (
        (zz - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (xx - centre[2]) ** 2
    ) <= radius ** 2


def _unit_cube():
    """Axis-aligned unit cube as 12 triangles with outward 1-based faces."""
    v = np.array(
        [
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
        ],
        dtype=np.float64,
    )
    f = np.array(
        [
            [1, 3, 4], [1, 4, 2],      # one pair of opposite faces
            [5, 6, 8], [5, 8, 7],
            [1, 2, 6], [1, 6, 5],
            [3, 7, 8], [3, 8, 4],
            [1, 5, 7], [1, 7, 3],
            [2, 4, 8], [2, 8, 6],
        ]
    )
    return v, f[:, [0, 2, 1]]          # wound so the normals face outward


def _iso2mesh_available():
    try:
        from iso2mesh import v2s  # noqa: F401
    except ImportError:
        return False
    return True


needs_iso2mesh = pytest.mark.skipif(
    not _iso2mesh_available(), reason="pyiso2mesh is not installed"
)


# --------------------------------------------------------------------------- #
# geometry helpers -- no iso2mesh required
# --------------------------------------------------------------------------- #
def test_cube_geometry_is_exact():
    v, f = _unit_cube()
    assert np.isclose(abs(mesh_volume(v, f)), 1.0)
    assert np.isclose(face_areas(v, f).sum(), 6.0)
    assert not mesh_has_holes(f)
    assert face_centroids(v, f).shape == (12, 3)


def test_geometry_scales_with_vertex_units():
    v, f = _unit_cube()
    geometry = mesh_geometry(v * 2.0, f, voxel_count=8, voxel_volume_um3=1.0)
    assert np.isclose(geometry.volume_um3, 8.0)
    assert np.isclose(geometry.surface_area_um2, 24.0)
    assert np.isclose(geometry.sphericity, np.pi ** (1 / 3) * 48 ** (2 / 3) / 24)
    assert np.isclose(geometry.height_um, 2.0)      # face centroids span 0..2 in z
    assert np.isclose(geometry.volume_ratio, 1.0)
    assert geometry.outward


def test_open_mesh_is_detected():
    v, f = _unit_cube()
    assert mesh_has_holes(f[:-1])  # drop one triangle -> boundary edges appear


def test_inward_winding_is_reported_not_hidden():
    v, f = _unit_cube()
    flipped = f[:, [0, 2, 1]]
    geometry = mesh_geometry(v, flipped)
    assert np.isclose(geometry.volume_um3, 1.0)     # magnitude, not the signed value
    assert not geometry.outward


def test_gibbon_patch_area_matches_triangle_area_for_triangles():
    v, f = _unit_cube()
    assert np.allclose(gibbon_patch_area(v, f), face_areas(v, f))


def test_gibbon_patch_area_fans_polygons():
    # A unit square as one quad, fanned from its centroid: four triangles of 1/4 each.
    v = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]], dtype=np.float64)
    f = np.array([[1, 2, 3, 4]])
    assert np.allclose(gibbon_patch_area(v, f), [1.0])


def test_largest_component_drops_satellites():
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[2:10, 2:10, 2:10] = True          # 512 voxels
    mask[15:17, 15:17, 15:17] = True       # 8 voxels
    kept = largest_component(mask)
    assert kept.sum() == 512
    assert not kept[15:17, 15:17, 15:17].any()


def test_largest_component_rejects_empty_mask():
    with pytest.raises(MeshingError):
        largest_component(np.zeros((4, 4, 4), dtype=bool))


# --------------------------------------------------------------------------- #
# lateral extent and aspect ratio
# --------------------------------------------------------------------------- #
def test_mip_axes_of_a_flat_ellipse_are_its_own():
    """A cylinder's projection is a disc, so both axes equal its diameter."""
    zz, yy, xx = np.indices((10, 41, 41))
    disc = ((yy - 20) ** 2 + (xx - 20) ** 2) <= 15 ** 2
    area, major, minor = mip_axis_lengths(disc, 0.5)

    assert area == pytest.approx(disc[0].sum() * 0.25)
    assert major == pytest.approx(minor, rel=0.02)
    assert major == pytest.approx(2 * 15 * 0.5, rel=0.05)


def test_mip_projects_along_z_not_another_axis():
    """A z-oriented rod projects to a dot; a y-oriented one to a line."""
    rod_z = np.zeros((30, 30, 30), dtype=bool)
    rod_z[:, 14:17, 14:17] = True
    rod_y = np.zeros((30, 30, 30), dtype=bool)
    rod_y[14:17, :, 14:17] = True

    _, major_z, _ = mip_axis_lengths(rod_z, 1.0)
    _, major_y, _ = mip_axis_lengths(rod_y, 1.0)
    assert major_y > 5 * major_z


def test_aspect_ratio_is_lateral_over_axial():
    # A flat object: 20 um across, 2 um tall -> ratio 10.
    assert aspect_ratio(20.0, 20.0, 2.0) == pytest.approx(10.0)
    # An isotropic one -> ratio 1.
    assert aspect_ratio(8.0, 8.0, 8.0) == pytest.approx(1.0)
    assert np.isnan(aspect_ratio(8.0, 8.0, 0.0))


def test_aspect_ratio_is_scale_invariant():
    """Dimensionless: doubling every dimension must not change it."""
    assert aspect_ratio(10.0, 6.0, 4.0) == pytest.approx(aspect_ratio(20.0, 12.0, 8.0))


@needs_iso2mesh
def test_sphere_aspect_ratio_shows_the_known_face_centroid_offset():
    """A sphere gives ~1.08, not 1.00, and that is the definition working as specified.

    The numerator is the mask silhouette (full width) while the denominator is the z
    extent of the mesh face centroids, which sit inside the surface -- see
    ``aspect_ratio``. Pinned so the offset stays a documented property rather than
    drifting silently if the height definition ever changes.
    """
    g = mesh_object(_ball(shape=(60, 60, 60), radius=22.0), (0.1, 0.1, 0.1)).geometry

    assert g.mip_major_um == pytest.approx(g.mip_minor_um, rel=0.02)   # round in xy
    assert g.mip_major_um == pytest.approx(2 * 22.0 * 0.1, rel=0.02)   # true diameter
    assert g.height_um < g.mip_major_um                                # centroids inset
    assert 1.02 < g.aspect_ratio < 1.15


@needs_iso2mesh
def test_oblate_object_reports_a_high_aspect_ratio():
    """Squash a ball in z: lateral stays, axial shrinks, so the ratio rises."""
    zz, yy, xx = np.indices((40, 80, 80))
    flat = (((zz - 19.5) / 8.0) ** 2 + ((yy - 39.5) / 30.0) ** 2
            + ((xx - 39.5) / 30.0) ** 2) <= 1.0
    g = mesh_object(flat, (0.1, 0.1, 0.1)).geometry
    assert g.aspect_ratio > 3.0


# --------------------------------------------------------------------------- #
# convexity
# --------------------------------------------------------------------------- #
def test_convex_shape_has_hull_equal_to_itself():
    """A cuboid is its own convex hull, so solidity is exactly 1."""
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[4:16, 5:15, 6:14] = True
    assert convex_hull_voxel_count(mask) == pytest.approx(mask.sum())


def test_concave_shape_has_hull_larger_than_itself():
    """Carve a bite out of a cuboid: the hull partly refills it, so solidity drops.

    A *face* bite is used rather than a corner one: removing a corner deletes one of the
    box's own vertices, so the hull legitimately shrinks to a corner-cut box rather than
    restoring the original, which makes the bound uninformative.
    """
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[4:16, 5:15, 6:14] = True
    box = mask.sum()
    mask[8:12, 7:13, 6:9] = False               # dent the middle of one face
    dented = mask.sum()

    hull = convex_hull_voxel_count(mask)
    assert dented < hull                         # the hull spans the dent
    assert hull == pytest.approx(box)            # ... back to the full box
    assert dented / hull < 1.0                   # so solidity is below 1


def test_hull_voxel_count_is_bounded_by_the_bounding_box():
    mask = _ball(shape=(30, 30, 30), radius=10.0)
    bbox = np.prod([s.stop - s.start for s in ndimage.find_objects(mask)[0]])
    hull = convex_hull_voxel_count(mask)
    assert mask.sum() <= hull <= bbox


def test_convex_hull_volume_matches_a_cube_analytically():
    v = np.array(
        [[0, 0, 0], [0, 0, 2], [0, 2, 0], [0, 2, 2],
         [2, 0, 0], [2, 0, 2], [2, 2, 0], [2, 2, 2]], dtype=np.float64
    )
    assert convex_hull_volume(v) == pytest.approx(8.0)


def test_degenerate_inputs_do_not_raise():
    """Fewer than four points has no 3-D hull; both helpers must degrade, not crash."""
    tiny = np.zeros((4, 4, 4), dtype=bool)
    tiny[1, 1, 1] = True
    assert convex_hull_voxel_count(tiny) == 1.0
    assert np.isnan(convex_hull_volume(np.zeros((3, 3))))


@needs_iso2mesh
def test_sphere_is_almost_perfectly_solid():
    mesh = mesh_object(_ball(shape=(60, 60, 60), radius=22.0), (0.1, 0.1, 0.1))
    g = mesh.geometry
    assert 0.97 < g.solidity <= 1.0
    assert 0.97 < g.mesh_solidity <= 1.0
    assert g.concavity == pytest.approx(1.0 - g.solidity)
    assert g.convex_hull_volume_um3 >= g.volume_um3


@needs_iso2mesh
def test_solidity_can_be_skipped():
    """The voxel hull is the one costly extra, so it must be optional."""
    mesh = mesh_object(_ball(), (0.1, 0.1, 0.1), solidity=False)
    assert np.isnan(mesh.geometry.solidity)
    assert np.isfinite(mesh.geometry.mesh_solidity)   # geometric hull is free


# --------------------------------------------------------------------------- #
# backend failure handling
# --------------------------------------------------------------------------- #
def test_backend_retries_a_transient_failure():
    """pyiso2mesh's file-based backends fail intermittently under batch load."""
    from analysis.volumetric import mesh as mesh_module

    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("jmeshlib command failed: loadOFF")
        return "meshed"

    assert mesh_module._backend(flaky, "flaky") == "meshed"
    assert len(calls) == 3


def test_backend_reports_a_persistent_failure_as_meshing_error():
    from analysis.volumetric import mesh as mesh_module

    def broken(*args, **kwargs):
        raise RuntimeError("cgalsurf exploded")

    with pytest.raises(MeshingError, match="after 3 attempts"):
        mesh_module._backend(broken, "broken")


# --------------------------------------------------------------------------- #
# the meshing chain
# --------------------------------------------------------------------------- #
@needs_iso2mesh
def test_sphere_mesh_recovers_analytic_geometry():
    radius_vox, spacing = 16.0, 0.1
    mesh = mesh_object(_ball(radius=radius_vox), (spacing, spacing, spacing))
    geometry = mesh.geometry

    radius_um = radius_vox * spacing
    expected_volume = 4 / 3 * np.pi * radius_um ** 3
    expected_area = 4 * np.pi * radius_um ** 2

    assert not geometry.has_holes
    assert geometry.outward
    # The default chain decimates hard, so this is a shape check, not a precision one:
    # a closed sphere-like surface within 25% of the analytic values.
    assert 0.75 < geometry.volume_um3 / expected_volume < 1.15
    assert 0.75 < geometry.surface_area_um2 / expected_area < 1.15
    assert geometry.sphericity > 0.9
    assert np.isclose(
        geometry.equivalent_sphere_radius_um,
        (3 / (4 * np.pi) * geometry.volume_um3) ** (1 / 3),
    )
    assert mesh.faces.min() == 1                      # faces stay 1-based
    assert mesh.faces.max() == mesh.vertices_um.shape[0]


@needs_iso2mesh
def test_smoothing_iterations_shrink_the_surface():
    """HC smoothing pulls the surface in; zero iterations must leave it alone."""
    mask = _ball()
    rough = mesh_object(mask, (0.1, 0.1, 0.1), smoothing_iterations=0)
    smooth = mesh_object(mask, (0.1, 0.1, 0.1), smoothing_iterations=10)
    assert smooth.geometry.surface_area_um2 < rough.geometry.surface_area_um2


@needs_iso2mesh
def test_smoothing_is_humphreys_classes_not_plain_laplacian():
    """HC (Vollmer/Mencl/Muller 1999) de-staircases without collapsing the volume.

    ``generate_mesh`` asks iso2mesh for ``laplacianhc``. Plain Laplacian smoothing at
    the same alpha shrinks the volume far more, so this guards the choice of smoother --
    swapping it would quietly bias every volume metric.

    The margin grows with mesh resolution (measured: 196 faces -> 3x, 1100 faces -> 12x,
    a real ~7000-face nucleus -> 14x, where HC costs 0.06% of the volume and plain
    Laplacian 0.83%). A ball big enough to mesh at a realistic density is used so the
    assertion is not calibrated on a degenerately coarse surface.
    """
    from analysis.volumetric.mesh import _import_iso2mesh

    _, _, smoothsurf, meshconn = _import_iso2mesh()

    mask = _ball(shape=(100, 100, 100), radius=40.0)
    vertices, faces = mesh_module_generate_unsmoothed(mask)
    conn = meshconn(faces, vertices.shape[0])[0]
    reference = abs(mesh_volume(vertices, faces))

    hc = np.asarray(smoothsurf(vertices.copy(), None, conn, 10, 0.1, "laplacianhc", 0.5))
    plain = np.asarray(smoothsurf(vertices.copy(), None, conn, 10, 0.1, "laplacian", 0.5))

    hc_shrink = 1 - abs(mesh_volume(hc, faces)) / reference
    plain_shrink = 1 - abs(mesh_volume(plain, faces)) / reference

    assert 0 <= hc_shrink < 0.01                 # HC barely moves the enclosed volume
    assert plain_shrink > 5 * hc_shrink          # plain Laplacian collapses it faster
    # Both still smooth: surface area drops relative to the staircased input.
    assert face_areas(hc, faces).sum() < face_areas(vertices, faces).sum()


def mesh_module_generate_unsmoothed(mask):
    """The meshing chain stopped just before smoothing, for smoother comparisons."""
    from analysis.volumetric.mesh import generate_mesh as _generate

    return _generate(mask, smoothing_iterations=0)


@needs_iso2mesh
def test_matlab_compat_changes_the_decimation_ratio():
    mask = _ball()
    default = mesh_object(mask, (0.1, 0.1, 0.1))
    compat = mesh_object(mask, (0.1, 0.1, 0.1), matlab_compat=True)
    assert default.geometry.n_faces != compat.geometry.n_faces


@needs_iso2mesh
def test_anisotropic_spacing_is_refused():
    with pytest.raises(MeshingError, match="isotropic"):
        mesh_object(_ball(), (0.3, 0.065, 0.065))


@needs_iso2mesh
def test_mesh_series_meshes_every_requested_timepoint():
    """One mesh per analysed timepoint, as the MATLAB pipeline does per (cell, frame)."""
    from core import BarcodeConfig

    # Three timepoints of a nucleus that shrinks slightly, as a (T, Z, Y, X) mask series.
    masks = np.stack([_ball(radius=r) for r in (16.0, 15.0, 14.0)])

    config = BarcodeConfig().volumetric
    config.mesh_enabled = True
    config.mesh_curvature = True

    meshes = mesh_series(masks, (0.1, 0.1, 0.1), [0, 1, 2], config)

    assert [m.frame_index for m in meshes] == [0, 1, 2]
    # Shrinking mask -> monotonically shrinking mesh.
    volumes = [m.geometry.volume_um3 for m in meshes]
    assert volumes[0] > volumes[1] > volumes[2]
    # Curvature was attached and rises as the radius falls (k ~ 1/R).
    assert all(m.curvature is not None for m in meshes)
    curvatures = [m.curvature.mean_curvature for m in meshes]
    assert curvatures[0] < curvatures[1] < curvatures[2]


@needs_iso2mesh
def test_mesh_series_can_skip_curvature():
    from core import BarcodeConfig

    config = BarcodeConfig().volumetric
    config.mesh_enabled = True
    config.mesh_curvature = False

    meshes = mesh_series(_ball()[None], (0.1, 0.1, 0.1), [0], config)
    assert len(meshes) == 1 and meshes[0].curvature is None


@needs_iso2mesh
def test_write_obj_round_trips_counts_and_axis_order(tmp_path):
    mesh = mesh_object(_ball(), (0.1, 0.1, 0.1))
    path = write_obj(str(tmp_path / "nucleus.obj"), mesh.vertices_um, mesh.faces)

    vertices, faces = [], []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(v) for v in line.split()[1:]])
            elif line.startswith("f "):
                faces.append([int(v) for v in line.split()[1:]])

    assert len(vertices) == mesh.vertices_um.shape[0]
    assert len(faces) == mesh.faces.shape[0]
    # OBJ is (x, y, z); the mesh is (z, y, x), so the columns are reversed.
    assert np.allclose(np.array(vertices), mesh.vertices_um[:, ::-1])


# --------------------------------------------------------------------------- #
# maxrad units
# --------------------------------------------------------------------------- #
def test_maxrad_in_voxels_is_passed_through_unchanged():
    """The stored meaning, and the historical one: no conversion at all."""
    from analysis.volumetric.mesh import resolve_maxrad

    assert resolve_maxrad(5.0, "voxels", 0.108333) == 5.0
    assert resolve_maxrad(5.0, "", 0.5) == 5.0          # unset falls back to voxels


def test_maxrad_in_microns_converts_with_the_voxel_size():
    """The point of the option: one physical size across different acquisitions.

    0.5 um is 5 voxels on a 0.1 um grid and 2 voxels on a 0.25 um grid, so stating it
    in microns is what keeps the triangle bound physically comparable between datasets.
    """
    from analysis.volumetric.mesh import resolve_maxrad

    assert resolve_maxrad(0.5, "um", 0.1) == pytest.approx(5.0)
    assert resolve_maxrad(0.5, "um", 0.25) == pytest.approx(2.0)
    assert resolve_maxrad(0.5, "microns", 0.1) == pytest.approx(5.0)


def test_maxrad_in_microns_without_a_voxel_size_raises():
    """Silently treating microns as voxels would change the mesh by whatever the
    voxel size happens to be -- refuse instead."""
    from analysis.volumetric.mesh import resolve_maxrad

    with pytest.raises(MeshingError, match="cannot convert"):
        resolve_maxrad(0.5, "um", 0.0)


def test_unknown_maxrad_units_are_rejected():
    from analysis.volumetric.mesh import resolve_maxrad

    with pytest.raises(MeshingError, match="Unknown maxrad units"):
        resolve_maxrad(5.0, "nanometres", 0.1)


def test_maxrad_from_config_reads_both_fields():
    from types import SimpleNamespace

    from analysis.volumetric.mesh import maxrad_from_config

    assert maxrad_from_config(
        SimpleNamespace(mesh_maxrad=2.0, mesh_maxrad_units="voxels"), 0.1) == 2.0
    assert maxrad_from_config(
        SimpleNamespace(mesh_maxrad=0.2, mesh_maxrad_units="um"), 0.1
    ) == pytest.approx(2.0)
    # A config predating the field must keep behaving as it always did.
    assert maxrad_from_config(SimpleNamespace(mesh_maxrad=5.0), 0.1) == 5.0


def test_relative_maxrad_scales_with_the_object():
    """The point of "relative": a fixed fraction of each object's own radius.

    A 4/3 pi r^3 ball of radius 20 has ~33500 voxels, so 0.1 relative is ~2 voxels; the
    same setting on a radius-5 object is ~0.5. That is what keeps mesh accuracy constant
    across a field whose objects differ in size, which neither voxels nor microns do.
    """
    from analysis.volumetric.mesh import equivalent_radius_voxels, resolve_maxrad

    big = 4 / 3 * np.pi * 20 ** 3
    small = 4 / 3 * np.pi * 5 ** 3
    assert equivalent_radius_voxels(big) == pytest.approx(20, rel=1e-6)
    assert resolve_maxrad(0.1, "relative", 0.1, big) == pytest.approx(2.0, rel=1e-6)
    assert resolve_maxrad(0.1, "relative", 0.1, small) == pytest.approx(0.5, rel=1e-6)


def test_relative_maxrad_has_a_floor():
    """Below about a quarter voxel the mesher gains nothing and costs a great deal."""
    from analysis.volumetric.mesh import resolve_maxrad

    assert resolve_maxrad(0.01, "relative", 0.1, 8.0) == 0.25


def test_relative_maxrad_without_the_object_size_raises():
    from analysis.volumetric.mesh import resolve_maxrad

    with pytest.raises(MeshingError, match="needs the object's size"):
        resolve_maxrad(0.1, "relative", 0.1)


def test_coarse_maxrad_is_reported_not_swallowed():
    """A too-coarse bound yields a closed, plausible mesh that is simply too small.

    Nothing downstream fails, so if this is not said out loud it reaches a figure.
    """
    from analysis.volumetric.mesh import warn_if_maxrad_coarse

    tiny = 4 / 3 * np.pi * 6 ** 3          # radius 6: maxrad 5 is 83% of it
    assert "maxrad" in warn_if_maxrad_coarse(5.0, tiny)
    big = 4 / 3 * np.pi * 65 ** 3          # radius 65: maxrad 5 is 8%, which is fine
    assert warn_if_maxrad_coarse(5.0, big) is None
