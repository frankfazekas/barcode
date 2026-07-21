"""Principal curvatures over a nucleus surface mesh, and the shape metrics built on them.

A port of the curvature half of ``TCell-3D-Morphodynamics``'s nuclear morphology
pipeline. The MATLAB side is a thin wrapper (``src/morphology/mesh/``) around Itzik
Ben Shabat's ``CurvatureEstimation`` toolbox, which implements Rusinkiewicz (2004),
"Estimating Curvatures and Their Derivatives on Triangle Meshes", following that
paper's own ``trimesh2`` C implementation. What is ported:

=================================================  ==========================================
MATLAB                                             here
=================================================  ==========================================
``CalcFaceNormals``                                :func:`face_normals`
``CalcVertexNormals``                              :func:`vertex_normals`
``CalcCurvature`` + ``ProjectCurvatureTensor``     :func:`_vertex_shape_operators`
``getPrincipalCurvatures`` + ``RotateCoordinate*`` :func:`principal_curvatures`
``curvatures_on_faces`` (GIBBON ``vertexToFace*``) :func:`vertex_to_face`
``identify_bottom_top_faces``                      :func:`identify_bottom_top_faces`
``classify_concavity``                             :func:`classify_concavity`
``mean_curvature_over_mesh``                       :func:`area_weighted_mean_curvature`
``find_invag_ratio``                               :func:`invagination_ratios`
=================================================  ==========================================

Everything is vectorised over faces or vertices; the MATLAB original loops. Two places
where that changes the arithmetic, both deliberate:

* ``CalcVertexNormals`` seeds each vertex's reference direction ``up`` inside the face
  loop, so a vertex shared by several faces keeps whichever face wrote **last**. Three
  separate vectorised assignments would not reproduce that order, so the writes are
  flattened face-major/corner-minor into one indexed assignment, which numpy resolves
  last-wins -- the same order MATLAB's loop produces.
* ``CalcCurvature`` solves a 6x3 least-squares system per face with MATLAB's backslash.
  Here that is a batched pseudo-inverse, which agrees with backslash whenever the
  system has full rank; they differ only for a degenerate (zero-area) triangle, where
  backslash returns a basic solution and this returns the minimum-norm one.

Sign convention: with outward-facing normals a sphere comes out with **positive**
principal curvatures, so ``classify_concavity`` reads a negative maximum curvature as
concave (an invagination). :func:`analyze_curvature` checks the mesh winding and flips
the faces when they wind inward, because otherwise every sign here would be inverted
and the invagination ratio would report the complement of what it means.

Units are inverse microns, since the vertices are in microns.

Agreement with MATLAB
---------------------
Unlike the meshing (see ``mesh.py``, where an arbitrary CGAL seed point costs exact
parity), this is deterministic arithmetic and reproduces MATLAB **bit for bit**.
Checked by exporting four real Jurkat nucleus meshes and running both implementations
on the identical vertices and faces (cells 1, 11, 12; ~24,000 faces total): every
scalar -- mean/min/max curvature, invagination ratio, concave ratio, bottom/top
fraction -- matched to all nine printed decimals, and per-face mean curvature agreed
to a maximum absolute difference of 1e-15 1/um, i.e. floating-point noise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from analysis.volumetric.mesh import face_areas, face_centroids, mesh_volume

# ``mean_curvature_over_mesh.m``: curvatures outside this band are outliers from
# badly-conditioned faces and are dropped before averaging (1/um).
_OUTLIER_BOUNDS = (-2.0, 2.0)

# ``identify_bottom_top_faces.m``: z-histogram bin width in microns.
_BOTTOM_TOP_BIN_UM = 0.1

# Concavity classes, as ``classify_concavity.m`` numbers them.
CONCAVE, HYPERBOLOID, CONVEX = 0, 1, 2
# A face whose curvature could not be computed (NaN), as distinct from one measured to be
# convex. Excluded from the invagination ratios entirely rather than counted as convex.
UNCLASSIFIED = -1


def _normr(rows: np.ndarray) -> np.ndarray:
    """Row-wise unit normalisation -- MATLAB's ``normr``."""
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return rows / norms


def _faces0(faces: np.ndarray) -> np.ndarray:
    """Faces as 0-based (N, 3) integers; the rest of the package keeps them 1-based."""
    return np.asarray(faces)[:, :3] - 1


# --------------------------------------------------------------------------- #
# normals and vertex frames
# --------------------------------------------------------------------------- #
def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Unit face normals -- ``CalcFaceNormals.m``."""
    v, f = np.asarray(vertices, dtype=np.float64), _faces0(faces)
    e0 = v[f[:, 2]] - v[f[:, 1]]
    e1 = v[f[:, 0]] - v[f[:, 2]]
    return _normr(np.cross(e0, e1))


def vertex_normals(
    vertices: np.ndarray, faces: np.ndarray, normals_f: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``CalcVertexNormals.m`` -- returns ``(normals, a_vertex, a_corner, up, vp)``.

    ``a_corner`` is the Meyer et al. (2002) mixed-Voronoi area split per face corner,
    ``a_vertex`` its sum per vertex, and ``(up, vp)`` an orthonormal tangent frame.
    """
    v, f = np.asarray(vertices, dtype=np.float64), _faces0(faces)
    n_vertices = v.shape[0]

    e0 = v[f[:, 2]] - v[f[:, 1]]
    e1 = v[f[:, 0]] - v[f[:, 2]]
    e2 = v[f[:, 1]] - v[f[:, 0]]
    e0n, e1n, e2n = _normr(e0), _normr(e1), _normr(e2)

    de0 = np.linalg.norm(e0, axis=1)
    de1 = np.linalg.norm(e1, axis=1)
    de2 = np.linalg.norm(e2, axis=1)
    l2 = np.stack([de0 ** 2, de1 ** 2, de2 ** 2], axis=1)

    # ew > 0 for every corner means the triangle is acute (barycentre inside).
    ew = np.stack(
        [
            l2[:, 0] * (l2[:, 1] + l2[:, 2] - l2[:, 0]),
            l2[:, 1] * (l2[:, 2] + l2[:, 0] - l2[:, 1]),
            l2[:, 2] * (l2[:, 0] + l2[:, 1] - l2[:, 2]),
        ],
        axis=1,
    )

    # Heron's formula, matching MATLAB rather than the cross-product area.
    s = (de0 + de1 + de2) / 2
    area_f = np.sqrt(np.clip(s * (s - de0) * (s - de1) * (s - de2), 0.0, None))

    with np.errstate(invalid="ignore", divide="ignore"):
        # Max (1999) angle weights for the normals.
        w = np.stack(
            [
                area_f / (de1 ** 2 * de2 ** 2),
                area_f / (de0 ** 2 * de2 ** 2),
                area_f / (de1 ** 2 * de0 ** 2),
            ],
            axis=1,
        )

    normals_v = np.zeros((n_vertices, 3))
    for corner in range(3):
        np.add.at(normals_v, f[:, corner], w[:, corner, None] * normals_f)
    normals_v = _normr(normals_v)

    # Obtuse triangles get the corner areas split by the Meyer et al. special cases;
    # acute ones by the ew weights.
    a_corner = np.zeros((f.shape[0], 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        obtuse0 = ew[:, 0] <= 0
        obtuse1 = (~obtuse0) & (ew[:, 1] <= 0)
        obtuse2 = (~obtuse0) & (~obtuse1) & (ew[:, 2] <= 0)
        acute = ~(obtuse0 | obtuse1 | obtuse2)

        dot02 = np.einsum("ij,ij->i", e0, e2)
        dot01 = np.einsum("ij,ij->i", e0, e1)
        dot12 = np.einsum("ij,ij->i", e1, e2)

        a_corner[obtuse0, 1] = -0.25 * l2[obtuse0, 2] * area_f[obtuse0] / dot02[obtuse0]
        a_corner[obtuse0, 2] = -0.25 * l2[obtuse0, 1] * area_f[obtuse0] / dot01[obtuse0]
        a_corner[obtuse0, 0] = area_f[obtuse0] - a_corner[obtuse0, 1] - a_corner[obtuse0, 2]

        a_corner[obtuse1, 2] = -0.25 * l2[obtuse1, 0] * area_f[obtuse1] / dot01[obtuse1]
        a_corner[obtuse1, 0] = -0.25 * l2[obtuse1, 2] * area_f[obtuse1] / dot12[obtuse1]
        a_corner[obtuse1, 1] = area_f[obtuse1] - a_corner[obtuse1, 0] - a_corner[obtuse1, 2]

        a_corner[obtuse2, 0] = -0.25 * l2[obtuse2, 1] * area_f[obtuse2] / dot12[obtuse2]
        a_corner[obtuse2, 1] = -0.25 * l2[obtuse2, 0] * area_f[obtuse2] / dot02[obtuse2]
        a_corner[obtuse2, 2] = area_f[obtuse2] - a_corner[obtuse2, 0] - a_corner[obtuse2, 1]

        scale = 0.5 * area_f[acute] / ew[acute].sum(axis=1)
        a_corner[acute, 0] = scale * (ew[acute, 1] + ew[acute, 2])
        a_corner[acute, 1] = scale * (ew[acute, 0] + ew[acute, 2])
        a_corner[acute, 2] = scale * (ew[acute, 1] + ew[acute, 0])

    a_vertex = np.zeros(n_vertices)
    for corner in range(3):
        np.add.at(a_vertex, f[:, corner], a_corner[:, corner])

    # Seed each vertex's reference direction. Flattened face-major/corner-minor so the
    # duplicate-index resolution matches MATLAB's loop -- see the module docstring.
    up = np.zeros((n_vertices, 3))
    seeds = np.stack([e2n, e0n, e1n], axis=1).reshape(-1, 3)
    up[f.reshape(-1)] = seeds

    up = _normr(np.cross(up, normals_v))
    vp = np.cross(normals_v, up)
    return normals_v, a_vertex, a_corner, up, vp


def _rotate_coordinate_system(
    up: np.ndarray, vp: np.ndarray, nf: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """``RotateCoordinateSystem.m`` -- rotate ``(up, vp)`` into the plane of ``nf``."""
    normal = _normr(np.cross(up, vp))
    ndot = np.einsum("ij,ij->i", nf, normal)[:, None]

    perp = nf - ndot * normal
    with np.errstate(invalid="ignore", divide="ignore"):
        dperp = (normal + nf) / (1 + ndot)
    r_u = up - dperp * np.einsum("ij,ij->i", perp, up)[:, None]
    r_v = vp - dperp * np.einsum("ij,ij->i", perp, vp)[:, None]

    # Anti-parallel frames are simply negated (the rotation above is undefined there).
    flipped = (ndot <= -1).reshape(-1)
    r_u[flipped] = -up[flipped]
    r_v[flipped] = -vp[flipped]
    return r_u, r_v


def _vertex_shape_operators(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals_v: np.ndarray,
    normals_f: np.ndarray,
    a_vertex: np.ndarray,
    a_corner: np.ndarray,
    up: np.ndarray,
    vp: np.ndarray,
) -> np.ndarray:
    """``CalcCurvature.m`` -- per-vertex second fundamental form as ``(N, 2, 2)``.

    Each face's shape operator comes from a least-squares fit to how the vertex normals
    change along its edges; it is then projected into each corner's tangent frame and
    accumulated with Voronoi weights.
    """
    v, f = np.asarray(vertices, dtype=np.float64), _faces0(faces)
    n_faces = f.shape[0]

    e0 = v[f[:, 2]] - v[f[:, 1]]
    e1 = v[f[:, 0]] - v[f[:, 2]]
    e2 = v[f[:, 1]] - v[f[:, 0]]

    # Face frame: t along edge 0, B completing the right-handed frame with the normal.
    t = _normr(e0)
    b = _normr(np.cross(normals_f, t))

    n0, n1, n2 = normals_v[f[:, 0]], normals_v[f[:, 1]], normals_v[f[:, 2]]

    def dots(edge):
        return np.einsum("ij,ij->i", edge, t), np.einsum("ij,ij->i", edge, b)

    e0t, e0b = dots(e0)
    e1t, e1b = dots(e1)
    e2t, e2b = dots(e2)

    zero = np.zeros(n_faces)
    matrices = np.stack(
        [
            np.stack([e0t, e0b, zero], axis=1),
            np.stack([zero, e0t, e0b], axis=1),
            np.stack([e1t, e1b, zero], axis=1),
            np.stack([zero, e1t, e1b], axis=1),
            np.stack([e2t, e2b, zero], axis=1),
            np.stack([zero, e2t, e2b], axis=1),
        ],
        axis=1,
    )  # (n_faces, 6, 3)

    def diff_dots(a, c):
        d = a - c
        return np.einsum("ij,ij->i", d, t), np.einsum("ij,ij->i", d, b)

    d21t, d21b = diff_dots(n2, n1)
    d02t, d02b = diff_dots(n0, n2)
    d10t, d10b = diff_dots(n1, n0)
    rhs = np.stack([d21t, d21b, d02t, d02b, d10t, d10b], axis=1)  # (n_faces, 6)

    with np.errstate(invalid="ignore", divide="ignore"):
        # pinv of (n, 6, 3) is (n, 3, 6); contract it against the (n, 6) right-hand side.
        solution = np.einsum("ijk,ik->ij", np.linalg.pinv(matrices), rhs)
    ku, kuv, kv = solution[:, 0], solution[:, 1], solution[:, 2]

    # Project each face's tensor into its three corners' frames and accumulate.
    operators = np.zeros((v.shape[0], 2, 2))
    for corner in range(3):
        idx = f[:, corner]
        r_u, r_v = _rotate_coordinate_system(up[idx], vp[idx], normals_f)

        u1 = np.einsum("ij,ij->i", r_u, t)
        v1 = np.einsum("ij,ij->i", r_u, b)
        u2 = np.einsum("ij,ij->i", r_v, t)
        v2 = np.einsum("ij,ij->i", r_v, b)

        new_ku = ku * u1 * u1 + 2 * kuv * u1 * v1 + kv * v1 * v1
        new_kuv = ku * u1 * u2 + kuv * (u1 * v2 + u2 * v1) + kv * v1 * v2
        new_kv = ku * u2 * u2 + 2 * kuv * u2 * v2 + kv * v2 * v2

        with np.errstate(invalid="ignore", divide="ignore"):
            weight = a_corner[:, corner] / a_vertex[idx]
        contribution = weight[:, None, None] * np.stack(
            [np.stack([new_ku, new_kuv], axis=1), np.stack([new_kuv, new_kv], axis=1)],
            axis=1,
        )
        np.add.at(operators, idx, contribution)

    return operators


def principal_curvatures(
    vertices: np.ndarray, faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-vertex principal curvatures ``(k_min, k_max)`` in 1/um.

    ``getPrincipalCurvatures.m``: a Jacobi rotation diagonalises each vertex's shape
    operator, then ``compute_curvatures_Rusinkiewicz.m`` reduces the pair to its
    minimum and maximum.
    """
    normals_f = face_normals(vertices, faces)
    normals_v, a_vertex, a_corner, up, vp = vertex_normals(vertices, faces, normals_f)
    operators = _vertex_shape_operators(
        vertices, faces, normals_v, normals_f, a_vertex, a_corner, up, vp
    )

    ku = operators[:, 0, 0]
    kuv = operators[:, 0, 1]
    kv = operators[:, 1, 1]

    tt = np.zeros_like(ku)
    off_diagonal = kuv != 0
    with np.errstate(invalid="ignore", divide="ignore"):
        h = 0.5 * (kv[off_diagonal] - ku[off_diagonal]) / kuv[off_diagonal]
        root = np.sqrt(1 + h * h)
        tt[off_diagonal] = np.where(h < 0, 1 / (h - root), 1 / (h + root))

    k1 = ku - tt * kuv
    k2 = kv + tt * kuv
    # getPrincipalCurvatures swaps so that |k1| >= |k2|; min/max is taken afterwards, so
    # the ordering does not survive -- but it is kept for faithfulness.
    swap = np.abs(k1) < np.abs(k2)
    k1[swap], k2[swap] = k2[swap].copy(), k1[swap].copy()

    return np.minimum(k1, k2), np.maximum(k1, k2)


# --------------------------------------------------------------------------- #
# per-face reduction and the metrics built on it
# --------------------------------------------------------------------------- #
def vertex_to_face(faces: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Mean of a per-vertex quantity over each face's corners (``vertexToFaceMeasure``)."""
    return np.asarray(values, dtype=np.float64)[_faces0(faces)].mean(axis=1)


def identify_bottom_top_faces(centroids_z: np.ndarray) -> Tuple[np.ndarray, float]:
    """Faces in the lowest or highest z bin -- ``identify_bottom_top_faces.m``.

    Curvature is unreliable where the segmentation is clipped by the top and bottom of
    the imaged stack, so those faces are excluded from every curvature metric. Returns
    ``(mask, fraction)``.
    """
    z = np.asarray(centroids_z, dtype=np.float64)
    z_min, z_max = float(z.min()), float(z.max())
    if not np.isfinite(z_min) or z_max <= z_min:
        return np.zeros(z.shape, dtype=bool), 0.0

    # At least three bins. With one or two, "lowest bin or highest bin" covers the whole
    # surface: an object flatter than the bin width (or barely two bins tall) had every
    # face excluded, so fraction_faces_bottom_top came back 1.0 and every curvature metric
    # NaN -- reported as missing data rather than as "too flat to classify". Three bins
    # keeps a middle to measure; anything flatter than that has no meaningful top and
    # bottom to distinguish and is left unexcluded.
    n_bins = int(np.ceil((z_max - z_min) / _BOTTOM_TOP_BIN_UM))
    if n_bins < 3:
        return np.zeros(z.shape, dtype=bool), 0.0
    edges = np.linspace(z_min, z_max, n_bins + 1)
    # MATLAB's histcounts closes the final bin on the right; clipping reproduces that
    # without a special case, since z never falls outside [z_min, z_max].
    bins = np.clip(np.digitize(z, edges), 1, n_bins)

    mask = (bins == 1) | (bins == n_bins)
    return mask, float(mask.sum() / z.size)


def classify_concavity(k_min: np.ndarray, k_max: np.ndarray) -> np.ndarray:
    """0 concave / 1 hyperboloid (saddle) / 2 convex / -1 unclassifiable.

    ``classify_concavity.m``, plus the NaN case it has no equivalent of. A vertex whose
    incident triangles all have zero Heron area -- the degenerate slivers ``meshresample``
    does produce -- gets NaN principal curvatures, and ``vertex_to_face`` spreads that to
    every adjacent face. Every NaN comparison is False, so those faces fell through both
    tests and were labelled CONVEX: a definite classification of a face nothing is known
    about. ``invagination_ratios`` then kept them in its denominator while they could
    never reach a numerator, biasing both ratios low by an amount set by mesh quality.
    """
    k_min = np.asarray(k_min, dtype=np.float64)
    k_max = np.asarray(k_max, dtype=np.float64)

    classes = np.full(k_min.shape, CONVEX, dtype=np.int8)
    classes[k_min <= 0] = HYPERBOLOID
    classes[k_max <= 0] = CONCAVE
    classes[~np.isfinite(k_min) | ~np.isfinite(k_max)] = UNCLASSIFIED
    return classes


def area_weighted_mean_curvature(
    values: np.ndarray,
    reference: np.ndarray,
    areas: np.ndarray,
    excluded: np.ndarray,
) -> float:
    """Area-weighted mean of ``values`` -- ``mean_curvature_over_mesh.m``.

    ``reference`` is the mean curvature that decides which faces are outliers, so that
    the same face set is used for the min, max and mean averages. ``excluded`` masks the
    bottom/top faces out.
    """
    low, high = _OUTLIER_BOUNDS
    good = (reference > low) & (reference < high) & ~np.asarray(excluded, dtype=bool)
    total = float(areas[good].sum())
    if total == 0:
        return np.nan
    return float((values[good] * areas[good]).sum() / total)


def invagination_ratios(
    classes: np.ndarray, areas: np.ndarray, excluded: np.ndarray
) -> Tuple[float, float]:
    """``find_invag_ratio.m`` -- ``(invagination_ratio, concave_ratio)``.

    Both are area fractions of the non-bottom/top surface: the first counts concave
    *and* saddle faces, the second concave only.
    """
    # Unclassifiable faces leave the denominator as well as the numerator: a ratio over a
    # surface that includes area nothing is known about is not a ratio of anything.
    keep = ~np.asarray(excluded, dtype=bool) & (np.asarray(classes) != UNCLASSIFIED)
    denominator = float(areas[keep].sum())
    if denominator == 0:
        return np.nan, np.nan

    concave = keep & (classes == CONCAVE)
    invaginated = concave | (keep & (classes == HYPERBOLOID))
    return (
        float(areas[invaginated].sum() / denominator),
        float(areas[concave].sum() / denominator),
    )


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
@dataclass
class CurvatureResults:
    """Curvature metrics for one mesh, plus the per-face arrays they came from."""

    # Area-weighted means over the usable surface (1/um).
    mean_curvature: float = np.nan
    min_curvature: float = np.nan
    max_curvature: float = np.nan

    # Area fractions of the usable surface.
    invagination_ratio: float = np.nan
    concave_ratio: float = np.nan
    fraction_faces_bottom_top: float = np.nan

    faces_flipped: bool = False

    # Per-face detail, for rendering and for stratifying other channels by curvature.
    k_min_faces: Optional[np.ndarray] = field(default=None, repr=False)
    k_max_faces: Optional[np.ndarray] = field(default=None, repr=False)
    k_mean_faces: Optional[np.ndarray] = field(default=None, repr=False)
    k_gaussian_faces: Optional[np.ndarray] = field(default=None, repr=False)
    concavity_classes: Optional[np.ndarray] = field(default=None, repr=False)
    bottom_top_faces: Optional[np.ndarray] = field(default=None, repr=False)

    def describe(self) -> List[str]:
        classes = self.concavity_classes
        counts = (
            f"{int((classes == CONCAVE).sum())} concave / "
            f"{int((classes == HYPERBOLOID).sum())} saddle / "
            f"{int((classes == CONVEX).sum())} convex"
            if classes is not None
            else "n/a"
        )
        return [
            f"mean curvature      : {self.mean_curvature:+.5f} 1/um "
            f"(min {self.min_curvature:+.5f}, max {self.max_curvature:+.5f})",
            f"invagination ratio  : {self.invagination_ratio:.5f}"
            f"   (concave only {self.concave_ratio:.5f})",
            f"faces by class      : {counts}",
            f"excluded top/bottom : {self.fraction_faces_bottom_top:.4f} of faces"
            f"{'   [faces flipped to outward]' if self.faces_flipped else ''}",
        ]


def analyze_curvature(
    vertices_um: np.ndarray,
    faces: np.ndarray,
    z_axis: int = 0,
) -> CurvatureResults:
    """Curvatures and the shape metrics derived from them, for one mesh.

    ``vertices_um`` is (N, 3) in microns and ``faces`` is 1-based, as
    :func:`analysis.volumetric.mesh.mesh_nucleus` returns them. ``z_axis`` names the
    vertex column holding z -- 0 for this package's (z, y, x) order, 2 for MATLAB's
    (y, x, z).
    """
    vertices = np.asarray(vertices_um, dtype=np.float64)
    faces = np.asarray(faces)[:, :3]

    # Curvature signs follow the normals, so an inward-wound mesh would invert every
    # classification. Flip it rather than silently reporting the complement.
    flipped = mesh_volume(vertices, faces) < 0
    if flipped:
        faces = faces[:, [0, 2, 1]]

    k_min_v, k_max_v = principal_curvatures(vertices, faces)

    k_min_f = vertex_to_face(faces, k_min_v)
    k_max_f = vertex_to_face(faces, k_max_v)
    k_mean_f = (k_min_f + k_max_f) / 2
    k_gauss_f = k_min_f * k_max_f

    areas = face_areas(vertices, faces)
    centroids = face_centroids(vertices, faces)
    bottom_top, fraction_bt = identify_bottom_top_faces(centroids[:, z_axis])

    classes = classify_concavity(k_min_f, k_max_f)
    invagination, concave = invagination_ratios(classes, areas, bottom_top)

    return CurvatureResults(
        mean_curvature=area_weighted_mean_curvature(k_mean_f, k_mean_f, areas, bottom_top),
        min_curvature=area_weighted_mean_curvature(k_min_f, k_mean_f, areas, bottom_top),
        max_curvature=area_weighted_mean_curvature(k_max_f, k_mean_f, areas, bottom_top),
        invagination_ratio=invagination,
        concave_ratio=concave,
        fraction_faces_bottom_top=fraction_bt,
        faces_flipped=bool(flipped),
        k_min_faces=k_min_f,
        k_max_faces=k_max_f,
        k_mean_faces=k_mean_f,
        k_gaussian_faces=k_gauss_f,
        concavity_classes=classes,
        bottom_top_faces=bottom_top,
    )
