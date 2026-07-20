"""Meshing of a segmented 3D nucleus (``analysis/volumetric/mesh.py``).

The geometry helpers are checked against shapes whose volume and area are known
exactly and need no iso2mesh; the meshing chain itself is checked against a
rasterised sphere and skipped when the CGAL binaries are unavailable.
"""
import numpy as np
import pytest

from analysis.volumetric.mesh import (
    MeshingError,
    face_areas,
    face_centroids,
    gibbon_patch_area,
    largest_component,
    mesh_geometry,
    mesh_has_holes,
    mesh_nucleus,
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
    mesh = mesh_nucleus(_ball(radius=radius_vox), (spacing, spacing, spacing))
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
    rough = mesh_nucleus(mask, (0.1, 0.1, 0.1), smoothing_iterations=0)
    smooth = mesh_nucleus(mask, (0.1, 0.1, 0.1), smoothing_iterations=10)
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
    default = mesh_nucleus(mask, (0.1, 0.1, 0.1))
    compat = mesh_nucleus(mask, (0.1, 0.1, 0.1), matlab_compat=True)
    assert default.geometry.n_faces != compat.geometry.n_faces


@needs_iso2mesh
def test_anisotropic_spacing_is_refused():
    with pytest.raises(MeshingError, match="isotropic"):
        mesh_nucleus(_ball(), (0.3, 0.065, 0.065))


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
    mesh = mesh_nucleus(_ball(), (0.1, 0.1, 0.1))
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
