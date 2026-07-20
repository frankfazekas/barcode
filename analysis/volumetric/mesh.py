"""Surface meshing of a segmented 3D nucleus.

A port of the meshing step in ``TCell-3D-Morphodynamics``
(``src/morphology/mesh/generate_mesh.m`` plus the mesh-derived scalars computed at
``src/processing/process_nucleus_channel.m:244-268``), built on ``pyiso2mesh`` --
the native-Python reimplementation of the same iso2mesh toolbox the MATLAB code calls.

The MATLAB chain, and what each step maps to here:

===========================================  ==========================================
MATLAB                                       here
===========================================  ==========================================
``v2s(I_bin, .99, maxrad)``                  ``iso2mesh.v2s`` (same binary, ``cgalsurf``)
``patch_area(F, V)`` (GIBBON)                :func:`face_areas` / :func:`gibbon_patch_area`
``meshresample(V, F, keep_ratio)``           ``iso2mesh.meshresample`` (``cgalsimp2``)
``meshconn`` + ``smoothsurf(...'laplacianhc')``  ``iso2mesh.trait.meshconn`` + ``smoothsurf``
``triSurfVolume`` / ``patch_area`` / ...     :func:`mesh_geometry`
===========================================  ==========================================

Conventions worth knowing before reading further:

* **Array axis order is (Z, Y, X)** throughout the volumetric package, and vertices
  here follow it: ``V[:, 0]`` is z, ``V[:, 1]`` y, ``V[:, 2]`` x. iso2mesh returns
  vertices in image-axis order, so this is what it hands back unchanged --
  in MATLAB the same code yields (y, x, z) because MATLAB stacks are (Y, X, Z),
  which is why ``process_nucleus_channel.m`` reads z out of column 3 and this module
  reads it out of column 0. :func:`write_obj` reverses to (x, y, z) on the way out.
* **Faces are 1-based** (MATLAB convention), because that is what pyiso2mesh emits
  and what its own ``meshconn`` expects. They are kept 1-based end to end; the
  geometry helpers here index with ``F - 1``.
* **Vertices come out in voxel units** and are scaled by the isotropic voxel size at
  the end, exactly as MATLAB does (``V = V * params.psize``). ``maxrad`` is therefore
  in voxels, and the mask must be on an isotropic grid for it to mean the same thing
  in every direction -- :func:`mesh_nucleus` enforces that.
* iso2mesh's vertices sit in a 1-based voxel frame (a vertex at array index 0 comes
  back as 1.0). That offset is left in place for parity with MATLAB; it cancels out of
  every extent- or size-based metric, and only shifts absolute ``z_min``/``z_max``
  by one voxel.

External binaries
-----------------
``v2s`` and ``meshresample`` shell out to ``cgalsurf`` and ``cgalsimp2``. pyiso2mesh
looks for them under ``~/iso2mesh-tools/iso2mesh-<ver>/bin`` and downloads them from
GitHub if that directory does not exist. :func:`ensure_iso2mesh_binaries` can instead
stage a local copy (e.g. the ones vendored in the MATLAB repo, which is what the
reference results were produced with). There is deliberately no marching-cubes
fallback: it would silently produce meshes that are not comparable to the MATLAB ones.

Agreement with MATLAB, and where it comes from
----------------------------------------------
Checked against ``Jurkats_live_Control_04142022_results.mat``, meshing the same
``Cell_N_SegMask.tif`` the MATLAB run used (cells 1, 11, 12; frames 1, 2, 8):

===================  =======================  =======================
metric               default                  ``matlab_compat=True``
===================  =======================  =======================
mesh volume          within 0.11%             within 0.03%
surface area         within 0.5%              within 0.05%
sphericity           within 0.5%              within 0.05%
height               within 0.4%              within 0.25%
===================  =======================  =======================

Two reproducibility limits, both upstream in the CGAL executables and both measured
here rather than assumed:

* **Concurrency is unsafe without isolation.** pyiso2mesh drives the executables
  through files with fixed names in one shared directory, so two meshing processes
  silently corrupt each other -- a nucleus that meshes to 949 um^3 alone came back at
  3,897,051 um^3 while another meshing job was running, with no error raised.
  :func:`_isolate_temp_dir` gives each process its own directory, which removes this.
* **cgalsurf is not bit-reproducible across processes**, even with its seed fixed and
  the temp directory isolated: repeated runs of the same nucleus land on one of two
  outcomes (949.50177 vs 949.53861 um^3 -- 0.004% in volume, 0.02% in surface area and
  sphericity). Within a single process it is stable. This is an order of magnitude
  below the MATLAB agreement below, so it is documented rather than fought.

Bit-parity with MATLAB is *not* achievable either, and the reason is worth knowing
before chasing it:
``cgalsurf`` takes an arbitrary interior seed point (from ``surfinterior``) and
centres its bounding sphere on it, so the whole meshing is seeded by that point.
MATLAB's ``surfinterior`` and pyiso2mesh's return different -- both valid -- interior
points, and on a small object that alone changes the output substantially (on a
40-voxel ball: 205 nodes from one seed, 416 from the other). At real nucleus scale the
effect washes out to the fractions of a percent tabulated above.
"""
from __future__ import annotations

import atexit
import contextlib
import io
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

if TYPE_CHECKING:  # avoids a cycle: curvature.py imports the geometry helpers here
    from analysis.volumetric.curvature import CurvatureResults

import numpy as np
from scipy import ndimage

# Spacings this close together count as isotropic (um).
_ISOTROPY_TOLERANCE_UM = 1e-9


class MeshingError(RuntimeError):
    """Raised when a mesh cannot be produced or is unusable."""


# --------------------------------------------------------------------------- #
# iso2mesh availability
# --------------------------------------------------------------------------- #
def ensure_iso2mesh_binaries(bin_dir: str = "") -> str:
    """Make sure the CGAL executables pyiso2mesh needs are on disk.

    ``bin_dir`` is an existing iso2mesh ``bin/`` directory to stage from (for
    instance ``.../TCell-3D-Morphodynamics/src/external/iso2mesh-1.9.6/bin``). It is
    copied into the location pyiso2mesh searches, which also suppresses the
    auto-download. With ``bin_dir`` empty, nothing is staged and pyiso2mesh's own
    lookup -- local directory, then download, then ``PATH`` -- is left alone.

    Returns the directory pyiso2mesh will search.
    """
    try:
        from iso2mesh.utils import ISO2MESH_BIN_VER
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise MeshingError(
            "The volumetric meshing branch needs pyiso2mesh: pip install iso2mesh"
        ) from exc

    target = os.path.join(
        os.path.expanduser("~"), "iso2mesh-tools", f"iso2mesh-{ISO2MESH_BIN_VER}", "bin"
    )
    if not bin_dir:
        return target

    if not os.path.isdir(bin_dir):
        raise MeshingError(f"iso2mesh binary directory {bin_dir!r} does not exist.")

    os.makedirs(target, exist_ok=True)
    for name in os.listdir(bin_dir):
        if not name.lower().endswith((".exe", ".dll", ".so", ".dylib")) and "." in name:
            continue
        destination = os.path.join(target, name)
        if not os.path.exists(destination):
            shutil.copy2(os.path.join(bin_dir, name), destination)
    return target


def _isolate_temp_dir() -> str:
    """Give this process its own pyiso2mesh scratch directory.

    pyiso2mesh drives the CGAL executables through files with **fixed** names
    (``pre_extract.inr``, ``post_remesh.off``, ...) in one shared directory, so two
    processes meshing at the same time overwrite each other's intermediates. The
    symptom is a parse error on a file that was perfectly valid when written --
    ``ValueError: cannot reshape array of size 518 into shape (3)`` out of ``readoff``,
    or ``jmeshlib command failed: loadOFF`` -- on a mesh that succeeds when re-run alone.

    ``ISO2MESH_TEMP`` relocates that directory. ``ISO2MESH_SESSION`` would also work but
    must not be used: ``vol2restrictedtri`` reads the *same* variable as the CGAL random
    seed (``os.getenv("ISO2MESH_SESSION", 0x623F9A9E)``), so setting it would silently
    change the seed and therefore every mesh.
    """
    existing = os.environ.get("ISO2MESH_TEMP")
    if existing and os.path.isdir(existing):
        return existing

    path = tempfile.mkdtemp(prefix=f"barcode-iso2mesh-{os.getpid()}-")
    os.environ["ISO2MESH_TEMP"] = path
    atexit.register(shutil.rmtree, path, True)
    return path


def _import_iso2mesh():
    """Import pyiso2mesh, turning the usual failures into :class:`MeshingError`."""
    try:
        from iso2mesh import meshresample, smoothsurf, v2s
        from iso2mesh.trait import meshconn
    except ImportError as exc:
        raise MeshingError(
            "The volumetric meshing branch needs pyiso2mesh: pip install iso2mesh"
        ) from exc
    _isolate_temp_dir()
    return v2s, meshresample, smoothsurf, meshconn


# --------------------------------------------------------------------------- #
# mask preparation
# --------------------------------------------------------------------------- #
def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected object, as ``process_nucleus_channel.m:207``.

    Mesh- and voxel-derived metrics have to describe the same object, so a mask that
    arrives with satellite fragments is reduced before anything is measured.
    """
    binary = np.asarray(mask).astype(bool)
    if not binary.any():
        raise MeshingError("Cannot mesh an empty mask.")

    labelled, count = ndimage.label(binary, ndimage.generate_binary_structure(3, 3))
    if count <= 1:
        return binary
    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0
    return labelled == int(np.argmax(sizes))


# --------------------------------------------------------------------------- #
# geometry helpers (numpy ports of the GIBBON / iso2mesh functions)
# --------------------------------------------------------------------------- #
def _triangles(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """(M, 3, 3) array of triangle corner coordinates from 1-based ``faces``."""
    return np.asarray(vertices, dtype=np.float64)[np.asarray(faces)[:, :3] - 1]


def face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Triangle areas -- ``patch_area`` for the already-triangular case."""
    tri = _triangles(vertices, faces)
    return 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )


def gibbon_patch_area(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Faithful port of GIBBON's ``patch_area`` *including* its polygon branch.

    Only needed for ``matlab_compat`` meshing. ``v2s`` returns faces with a fourth
    region-id column, and ``generate_mesh.m`` passes that straight to ``patch_area``,
    which sees ``size(F,2) > 3`` and treats each row as a quadrilateral -- fanning it
    from its own centroid, with the region id (always 1) read as a vertex index. The
    resulting "areas" are dominated by the distance to vertex 1 rather than by
    triangle size. Reproduced here so the MATLAB keep-ratio can be recovered exactly;
    see :func:`generate_mesh` for why it is not the default.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    if f.shape[1] <= 3:
        return face_areas(v, f)

    corners = v[f - 1]                       # (M, k, 3)
    centroids = corners.mean(axis=1)         # fan apex, as GIBBON does
    nxt = np.roll(corners, -1, axis=1)
    e1 = nxt - corners
    e2 = centroids[:, None, :] - corners
    return 0.5 * np.linalg.norm(np.cross(e1, e2), axis=2).sum(axis=1)


def face_centroids(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """``meshcentroid`` -- the mean of each face's corners."""
    return _triangles(vertices, faces).mean(axis=1)


def mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Signed enclosed volume via the divergence theorem (``triSurfVolume``).

    The sign reports orientation: negative means the faces wind inward. Callers take
    the magnitude and keep the sign as a diagnostic.
    """
    tri = _triangles(vertices, faces)
    return float(
        np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0
    )


def mesh_has_holes(faces: np.ndarray) -> bool:
    """``mesh_has_holes.m`` -- true if any edge belongs to only one face."""
    f = np.asarray(faces)[:, :3]
    edges = np.sort(
        np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0), axis=1
    )
    counts = np.unique(edges, axis=0, return_counts=True)[1]
    return bool(np.any(counts == 1))


# --------------------------------------------------------------------------- #
# meshing
# --------------------------------------------------------------------------- #
def generate_mesh(
    mask: np.ndarray,
    maxrad: float = 5.0,
    area_frac: float = 0.2,
    smoothing_iterations: int = 10,
    alpha: float = 0.1,
    beta: float = 0.5,
    matlab_compat: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Port of ``generate_mesh.m``. Returns ``(vertices_voxels, faces_1based)``.

    ``mask`` is a boolean/0-1 volume on an **isotropic** grid; ``maxrad`` is
    ``cgalsurf``'s ``radbound`` in voxels.

    The area filter deserves a note. Both variants compute the same thing in spirit --
    the fraction of faces that are not part of the small spurious node clusters
    ``v2s`` leaves behind, used as the decimation ratio -- but:

    * default (``matlab_compat=False``) filters on real triangle areas;
    * ``matlab_compat=True`` reproduces MATLAB's quad-fan reading of the 4-column face
      array (see :func:`gibbon_patch_area`), which yields a different ratio and hence
      a different mesh.

    The default is the corrected one. Use the flag when reproducing MATLAB numbers.
    """
    v2s, meshresample, smoothsurf, meshconn = _import_iso2mesh()

    binary = np.asarray(mask).astype(np.uint8)
    if binary.ndim != 3:
        raise MeshingError(f"Expected a 3-D mask, got shape {binary.shape}.")
    if not binary.any():
        raise MeshingError("Cannot mesh an empty mask.")

    # 1) Surface extraction. Isovalue 0.99 on a 0/1 volume, radbound = maxrad.
    #    v2s returns faces as (M, 4): three 1-based node indices plus a region id.
    vertices, faces_raw = _backend(
        v2s, "v2s", binary, 0.99, float(maxrad), verbose=verbose
    )[:2]
    vertices = np.asarray(vertices, dtype=np.float64)[:, :3]
    faces_raw = np.asarray(faces_raw)
    if faces_raw.size == 0:
        raise MeshingError("v2s produced no faces; is the mask a single voxel?")

    # 2) Fraction of faces large enough to be real surface rather than a node cluster.
    areas = (
        gibbon_patch_area(vertices, faces_raw)
        if matlab_compat
        else face_areas(vertices, faces_raw)
    )
    largest = float(np.max(areas))
    keep_ratio = float(np.count_nonzero(areas > area_frac * largest) / areas.size)
    if keep_ratio <= 0:
        raise MeshingError("Area filter kept no faces; check area_frac.")

    # 3) Decimate to that ratio. Only the three node columns are meshable; passing the
    #    region column through would write quads into the OFF file cgalsimp2 reads.
    faces = faces_raw[:, :3]
    vertices, faces = _backend(
        meshresample, "meshresample", vertices, faces, keep_ratio, verbose=verbose
    )
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)[:, :3]

    # 4) Laplacian-HC smoothing (Vollmer/Mencl/Muller). MATLAB passes the *face* count
    #    to meshconn where a node count belongs; that only over-allocates the neighbour
    #    list on a closed surface, so the corrected node count is used here.
    if smoothing_iterations > 0:
        conn = meshconn(faces, vertices.shape[0])[0]
        vertices = np.asarray(
            smoothsurf(
                vertices, None, conn, int(smoothing_iterations),
                float(alpha), "laplacianhc", float(beta),
            ),
            dtype=np.float64,
        )

    return vertices, faces


def _backend(func, name: str, *args, verbose: bool = False, attempts: int = 3, **kwargs):
    """Call a pyiso2mesh entry point, retrying transient failures.

    :func:`_isolate_temp_dir` removes the cross-process cause of these failures, but the
    file round-trip through the CGAL executables can still fail occasionally within one
    process (``jmeshlib command failed: ERROR- loadOFF ...`` on a mesh that meshes
    cleanly on the very next attempt). Retrying costs a few seconds and turns a
    batch-ending crash into a hiccup. A genuinely bad mesh fails every attempt and is
    reported as a :class:`MeshingError`, so nothing is swallowed.

    ``ValueError`` is caught alongside the others because a truncated intermediate file
    surfaces from ``readoff`` as a reshape error rather than as an I/O error.
    """
    last: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            return _quiet(func, *args, verbose=verbose, **kwargs)
        except (RuntimeError, OSError, ValueError) as exc:
            last = exc
            if verbose or attempt > 1:
                print(
                    f"  {name} failed (attempt {attempt}/{attempts}): "
                    f"{type(exc).__name__}: {str(exc).strip()[:160]}",
                    flush=True,
                )
    raise MeshingError(f"iso2mesh {name} failed after {attempts} attempts: {last}")


def _quiet(func, *args, verbose: bool = False, **kwargs):
    """Call ``func`` with the meshing chatter suppressed.

    The MATLAB code wraps these same calls in ``evalc`` for the same reason. Two
    streams have to be caught: pyiso2mesh's own ``print`` calls, and the CGAL
    binaries writing to the process's stdout file descriptor -- redirecting only the
    descriptor leaves Python's buffered text to be flushed once it is restored.
    """
    if verbose:
        return func(*args, **kwargs)

    stdout_fd = 1
    sys.stdout.flush()
    saved = os.dup(stdout_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, stdout_fd)
        with contextlib.redirect_stdout(io.StringIO()):
            return func(*args, **kwargs)
    finally:
        sys.stdout.flush()
        os.dup2(saved, stdout_fd)
        os.close(saved)
        os.close(devnull)


# --------------------------------------------------------------------------- #
# mesh-derived scalars
# --------------------------------------------------------------------------- #
@dataclass
class MeshGeometry:
    """The cheap mesh-derived scalars from ``process_nucleus_channel.m:244-272``."""

    n_vertices: int = 0
    n_faces: int = 0
    has_holes: bool = False
    outward: bool = True

    volume_um3: float = np.nan
    surface_area_um2: float = np.nan
    sphericity: float = np.nan
    equivalent_sphere_radius_um: float = np.nan

    height_um: float = np.nan          # z extent of the face centroids
    extent_zyx_um: Tuple[float, float, float] = (np.nan, np.nan, np.nan)
    z_min_um: float = np.nan
    z_max_um: float = np.nan

    # Independent voxel-side cross-check on volume_um3.
    voxel_count: int = 0
    voxel_volume_um3: float = np.nan

    @property
    def volume_ratio(self) -> float:
        """``volume_um3`` over the voxel-counted volume; ~1 for a faithful mesh."""
        reference = self.voxel_count * self.voxel_volume_um3
        return self.volume_um3 / reference if reference else np.nan

    def describe(self) -> List[str]:
        return [
            f"vertices/faces      : {self.n_vertices} / {self.n_faces}"
            f"{'' if not self.has_holes else '   (OPEN BOUNDARY)'}"
            f"{'' if self.outward else '   (inward normals)'}",
            f"volume              : {self.volume_um3:.4f} um^3"
            f"   (voxels: {self.voxel_count * self.voxel_volume_um3:.4f},"
            f" ratio {self.volume_ratio:.4f})",
            f"surface area        : {self.surface_area_um2:.4f} um^2",
            f"sphericity          : {self.sphericity:.4f}",
            f"equiv sphere radius : {self.equivalent_sphere_radius_um:.4f} um",
            f"height (z of faces) : {self.height_um:.4f} um",
            f"extent (z, y, x)    : "
            + ", ".join(f"{e:.4f}" for e in self.extent_zyx_um)
            + " um",
        ]


def mesh_geometry(
    vertices_um: np.ndarray,
    faces: np.ndarray,
    voxel_count: int = 0,
    voxel_volume_um3: float = np.nan,
) -> MeshGeometry:
    """Scalars for a mesh whose vertices are already in microns, ordered (z, y, x)."""
    signed = mesh_volume(vertices_um, faces)
    volume = abs(signed)
    area = float(face_areas(vertices_um, faces).sum())
    sphericity = (
        float(np.pi ** (1 / 3) * (6 * volume) ** (2 / 3) / area) if area > 0 else np.nan
    )

    centroids = face_centroids(vertices_um, faces)
    extent = centroids.max(axis=0) - centroids.min(axis=0)

    return MeshGeometry(
        n_vertices=int(vertices_um.shape[0]),
        n_faces=int(np.asarray(faces).shape[0]),
        has_holes=mesh_has_holes(faces),
        outward=signed >= 0,
        volume_um3=volume,
        surface_area_um2=area,
        sphericity=sphericity,
        equivalent_sphere_radius_um=float((3 / (4 * np.pi) * volume) ** (1 / 3)),
        height_um=float(extent[0]),                      # (z, y, x) -- z is column 0
        extent_zyx_um=tuple(float(e) for e in extent),
        z_min_um=float(vertices_um[:, 0].min()),
        z_max_um=float(vertices_um[:, 0].max()),
        voxel_count=int(voxel_count),
        voxel_volume_um3=float(voxel_volume_um3),
    )


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
@dataclass
class NucleusMesh:
    """A meshed nucleus: vertices in microns (z, y, x), 1-based faces, and scalars."""

    vertices_um: np.ndarray
    faces: np.ndarray
    geometry: MeshGeometry
    frame_index: int = 0
    curvature: Optional["CurvatureResults"] = None


def mesh_nucleus(
    mask_zyx: np.ndarray,
    spacing_zyx_um: Sequence[float],
    maxrad: float = 5.0,
    area_frac: float = 0.2,
    smoothing_iterations: int = 10,
    alpha: float = 0.1,
    beta: float = 0.5,
    matlab_compat: bool = False,
    verbose: bool = False,
    frame_index: int = 0,
) -> NucleusMesh:
    """Mesh one segmented nucleus and measure it.

    ``mask_zyx`` must be on an isotropic grid -- which is what
    ``analysis.volumetric.resample.prepare_nucleus`` produces -- because ``maxrad``
    is a single radius in voxels and would otherwise mean a different physical
    distance along each axis.
    """
    spacing = np.asarray(spacing_zyx_um, dtype=np.float64)
    if spacing.size != 3:
        raise MeshingError(f"spacing_zyx_um must have 3 entries, got {spacing.size}.")
    if float(spacing.max() - spacing.min()) > _ISOTROPY_TOLERANCE_UM:
        raise MeshingError(
            f"Meshing needs an isotropic grid but spacing (z, y, x) is "
            f"{tuple(spacing)} um. Run with make_isotropic enabled, or resample first."
        )

    binary = largest_component(mask_zyx)
    vertices_vox, faces = generate_mesh(
        binary,
        maxrad=maxrad,
        area_frac=area_frac,
        smoothing_iterations=smoothing_iterations,
        alpha=alpha,
        beta=beta,
        matlab_compat=matlab_compat,
        verbose=verbose,
    )

    voxel_size = float(spacing[0])
    vertices_um = vertices_vox * voxel_size
    geometry = mesh_geometry(
        vertices_um,
        faces,
        voxel_count=int(binary.sum()),
        voxel_volume_um3=float(np.prod(spacing)),
    )
    return NucleusMesh(
        vertices_um=vertices_um, faces=faces, geometry=geometry, frame_index=frame_index
    )


def write_obj(path: str, vertices_um: np.ndarray, faces: np.ndarray) -> str:
    """Write a Wavefront OBJ, reordering vertices from (z, y, x) to (x, y, z).

    OBJ face indices are 1-based, which is what the faces already are.
    """
    vertices = np.asarray(vertices_um, dtype=np.float64)[:, ::-1]
    faces = np.asarray(faces)[:, :3]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# BARCODE volumetric nucleus mesh\n")
        for x, y, z in vertices:
            handle.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in faces:
            handle.write(f"f {a} {b} {c}\n")
    return path


def mesh_series(
    masks: np.ndarray,
    spacing_zyx_um: Sequence[float],
    frame_indices: Sequence[int],
    config,
    verbose: bool = False,
) -> List[NucleusMesh]:
    """Mesh the analysed timepoints of a ``(T, Z, Y, X)`` mask series."""
    ensure_iso2mesh_binaries(getattr(config, "mesh_iso2mesh_bin", "") or "")
    meshes = [
        mesh_nucleus(
            masks[index],
            spacing_zyx_um,
            maxrad=config.mesh_maxrad,
            area_frac=config.mesh_area_frac,
            smoothing_iterations=config.mesh_smoothing_iterations,
            alpha=config.mesh_smoothing_alpha,
            beta=config.mesh_smoothing_beta,
            matlab_compat=config.mesh_matlab_compat,
            verbose=verbose,
            frame_index=int(index),
        )
        for index in frame_indices
    ]

    if getattr(config, "mesh_curvature", False):
        # Imported here rather than at module scope: curvature.py depends on the
        # geometry helpers above, so a top-level import would be circular.
        from analysis.volumetric.curvature import analyze_curvature

        for mesh in meshes:
            mesh.curvature = analyze_curvature(mesh.vertices_um, mesh.faces)

    return meshes
