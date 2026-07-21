"""Principal curvatures over a nucleus mesh (``analysis/volumetric/curvature.py``).

The analytic checks use shapes whose curvature is known in closed form -- a sphere
(k1 = k2 = 1/R everywhere) and a cylinder (k1 = 1/R, k2 = 0) -- built here directly
rather than through iso2mesh, so most of the file runs without the CGAL binaries.
"""
import numpy as np
import pytest

from analysis.volumetric.curvature import (
    CONCAVE,
    CONVEX,
    HYPERBOLOID,
    analyze_curvature,
    area_weighted_mean_curvature,
    classify_concavity,
    face_normals,
    identify_bottom_top_faces,
    invagination_ratios,
    principal_curvatures,
    vertex_to_face,
)


def _icosphere(subdivisions=3, radius=1.0):
    """Geodesic sphere with outward-wound faces, vertices in (z, y, x)."""
    t = (1 + 5 ** 0.5) / 2
    v = np.array(
        [
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
        ],
        dtype=np.float64,
    )
    f = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ]
    )

    for _ in range(subdivisions):
        midpoint, new_faces = {}, []

        def middle(a, b):
            key = (min(a, b), max(a, b))
            if key not in midpoint:
                midpoint[key] = len(v_list)
                v_list.append((v_list[a] + v_list[b]) / 2)
            return midpoint[key]

        v_list = list(v)
        for a, b, c in f:
            ab, bc, ca = middle(a, b), middle(b, c), middle(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        v = np.array(v_list)
        f = np.array(new_faces)

    v = v / np.linalg.norm(v, axis=1, keepdims=True) * radius
    return v, f + 1  # 1-based faces, as the rest of the package uses


def _cylinder(radius=1.0, height=4.0, n_theta=64, n_z=40):
    """Open cylinder along z, outward normals. Vertices are (z, y, x)."""
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    z = np.linspace(-height / 2, height / 2, n_z)
    zz, tt = np.meshgrid(z, theta, indexing="ij")
    v = np.stack([zz, radius * np.sin(tt), radius * np.cos(tt)], axis=-1).reshape(-1, 3)

    faces = []
    for i in range(n_z - 1):
        for j in range(n_theta):
            j2 = (j + 1) % n_theta
            a = i * n_theta + j
            b = i * n_theta + j2
            c = (i + 1) * n_theta + j
            d = (i + 1) * n_theta + j2
            faces += [[a, c, b], [b, c, d]]
    return v, np.array(faces) + 1


def _interior_faces(vertices, faces, z_margin):
    """Faces whose corners are all away from an open mesh's z boundaries."""
    z = vertices[:, 0]
    inner = (z > z.min() + z_margin) & (z < z.max() - z_margin)
    return inner[np.asarray(faces) - 1].all(axis=1)


# --------------------------------------------------------------------------- #
# analytic shapes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("radius", [1.0, 5.0])
def test_sphere_has_uniform_positive_curvature(radius):
    v, f = _icosphere(subdivisions=3, radius=radius)
    k_min, k_max = principal_curvatures(v, f)

    expected = 1.0 / radius
    # Outward normals give a convex surface positive curvature -- the sign convention
    # classify_concavity depends on.
    assert np.allclose(k_min, expected, rtol=0.02)
    assert np.allclose(k_max, expected, rtol=0.02)


def test_sphere_scales_curvature_inversely_with_radius():
    small = principal_curvatures(*_icosphere(3, 1.0))[0].mean()
    large = principal_curvatures(*_icosphere(3, 4.0))[0].mean()
    assert np.isclose(small / large, 4.0, rtol=0.02)


def test_cylinder_has_one_zero_principal_curvature():
    radius = 2.0
    v, f = _cylinder(radius=radius, height=6.0)
    k_min, k_max = principal_curvatures(v, f)

    # Only away from the open ends, where the estimator has no neighbourhood.
    interior = (v[:, 0] > v[:, 0].min() + 0.5) & (v[:, 0] < v[:, 0].max() - 0.5)
    assert np.allclose(k_min[interior], 0.0, atol=0.02)
    assert np.allclose(k_max[interior], 1.0 / radius, rtol=0.02)


def test_inward_wound_sphere_is_flipped_not_misclassified():
    v, f = _icosphere(subdivisions=3, radius=2.0)
    inward = f[:, [0, 2, 1]]

    result = analyze_curvature(v, inward)
    assert result.faces_flipped
    assert result.mean_curvature > 0
    assert result.invagination_ratio == 0.0

    outward = analyze_curvature(v, f)
    assert not outward.faces_flipped
    assert np.isclose(result.mean_curvature, outward.mean_curvature)


def test_sphere_reports_no_invagination():
    result = analyze_curvature(*_icosphere(subdivisions=3, radius=3.0))
    assert result.invagination_ratio == 0.0
    assert result.concave_ratio == 0.0
    assert (result.concavity_classes == CONVEX).all()


def test_saddle_is_classified_hyperboloid():
    """A hyperbolic paraboloid z = (x^2 - y^2)/2 is a saddle everywhere."""
    n = 40
    x = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    zz = (xx ** 2 - yy ** 2) / 2
    v = np.stack([zz, yy, xx], axis=-1).reshape(-1, 3)

    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b = i * n + j, i * n + j + 1
            c, d = (i + 1) * n + j, (i + 1) * n + j + 1
            faces += [[a, c, b], [b, c, d]]
    f = np.array(faces) + 1

    k_min, k_max = principal_curvatures(v, f)
    inner = (np.abs(v[:, 1]) < 0.7) & (np.abs(v[:, 2]) < 0.7)
    classes = classify_concavity(k_min[inner], k_max[inner])
    assert (classes == HYPERBOLOID).mean() > 0.95


# --------------------------------------------------------------------------- #
# the reductions built on the curvatures
# --------------------------------------------------------------------------- #
def test_classify_concavity_boundaries():
    k_min = np.array([-1.0, -1.0, 1.0, 0.0])
    k_max = np.array([-0.5, 1.0, 2.0, 0.0])
    assert list(classify_concavity(k_min, k_max)) == [
        CONCAVE, HYPERBOLOID, CONVEX, CONCAVE
    ]


def test_vertex_to_face_averages_corners():
    faces = np.array([[1, 2, 3]])
    assert vertex_to_face(faces, np.array([3.0, 6.0, 9.0])) == pytest.approx(6.0)


def test_bottom_top_faces_are_the_extreme_z_bins():
    z = np.linspace(0.0, 1.0, 101)          # 10 bins of 0.1 um
    mask, fraction = identify_bottom_top_faces(z)
    assert mask[0] and mask[-1]
    assert not mask[50]
    assert fraction == pytest.approx(mask.sum() / z.size)


def test_bottom_top_handles_a_flat_mesh():
    mask, fraction = identify_bottom_top_faces(np.zeros(10))
    assert not mask.any() and fraction == 0.0


def test_area_weighted_mean_excludes_outliers_only_when_a_limit_is_given():
    """The outlier band is opt-in: 0 keeps every face, a positive limit trims.

    Both directions are pinned because the default changed. Face exclusion used to be
    unconditional; measuring the whole surface is now the default and MATLAB's behaviour
    is reached by asking for it, so a silent return to always-trimming would be as much a
    regression as a silent failure to trim.
    """
    values = np.array([1.0, 1.0, 99.0, 1.0])     # 99 is outside the +/-2 band
    areas = np.array([1.0, 1.0, 1.0, 5.0])
    excluded = np.array([False, False, False, True])

    # limit 2.0 -> the 99 is dropped, leaving two unit faces
    assert area_weighted_mean_curvature(
        values, values, areas, excluded, 2.0) == pytest.approx(1.0)

    # limit 0 (the default) -> the 99 is kept: (1 + 1 + 99) / 3
    assert area_weighted_mean_curvature(
        values, values, areas, excluded) == pytest.approx(101.0 / 3.0)
    assert area_weighted_mean_curvature(
        values, values, areas, excluded, 0.0) == pytest.approx(101.0 / 3.0)


def test_analyze_curvature_excludes_no_faces_by_default():
    """Caps are kept unless asked for, and the reported fraction says so."""
    v, f = _icosphere(subdivisions=3, radius=3.0)

    default = analyze_curvature(v, f)
    assert default.fraction_faces_bottom_top == 0.0
    assert not default.bottom_top_faces.any()

    matlab = analyze_curvature(v, f, exclude_caps=True, outlier_limit=2.0)
    assert matlab.fraction_faces_bottom_top > 0.0
    assert matlab.bottom_top_faces.any()

    # A sphere is smooth enough that neither rule changes <H> materially, so the switch
    # is visible in what was excluded rather than in the mean.
    assert default.mean_curvature == pytest.approx(1.0 / 3.0, rel=0.02)
    assert matlab.mean_curvature == pytest.approx(1.0 / 3.0, rel=0.02)


def test_area_weighted_mean_is_nan_when_everything_is_excluded():
    values = np.array([1.0, 1.0])
    assert np.isnan(
        area_weighted_mean_curvature(values, values, np.ones(2), np.ones(2, dtype=bool))
    )


def test_invagination_ratio_counts_concave_and_saddle_by_area():
    classes = np.array([CONCAVE, HYPERBOLOID, CONVEX, CONCAVE])
    areas = np.array([1.0, 2.0, 7.0, 100.0])
    excluded = np.array([False, False, False, True])       # the big one is excluded
    invagination, concave = invagination_ratios(classes, areas, excluded)
    assert invagination == pytest.approx(3 / 10)
    assert concave == pytest.approx(1 / 10)


def test_face_normals_point_outward_for_an_outward_sphere():
    v, f = _icosphere(subdivisions=2, radius=1.0)
    normals = face_normals(v, f)
    centroids = v[f - 1].mean(axis=1)
    assert (np.einsum("ij,ij->i", normals, centroids) > 0).all()
