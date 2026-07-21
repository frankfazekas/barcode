"""Per-object meshing for a field of many objects.

``mesh.py`` meshes *one* object: ``mesh_object`` calls ``largest_component`` and throws
the rest away. That is right for an isolated nucleus and wrong for a confluent field --
an epithelium segmented by Cellpose is hundreds of touching cells, and collapsing it to
the largest one discards the tissue.

This module is the many-object counterpart. It is deliberately object-generic: BARCODE
analyses soft active materials, so an "object" here is any labelled instance -- a cell, a
nucleus, a droplet, a grain.

Two things make this cheap enough to run on a real field:

* each object is meshed inside **its own bounding box**, so cost scales with the object
  rather than with the field (a 1500x1808 frame with 300 cells would otherwise mesh a
  27-megavoxel volume 300 times);
* vertices are shifted back into field coordinates afterwards, so the objects still sit
  correctly relative to one another and can be rendered as one scene.

Touching instances are the normal case, so the input must be an **integer label volume**,
not a binary one: connectivity labelling would merge a confluent sheet into a single
object. ``segmentation_label_mode="labels"`` in the volumetric config preserves those
integers when the mask is loaded.

Choosing ``maxrad`` for thin objects
------------------------------------
``maxrad`` is the CGAL surface radbound in voxels, and the default of 5 is tuned for
isolated nuclei ~200 voxels across. It is **badly wrong for a thin object**, because the
triangles then span a large fraction of the object's depth. Measured on a cell-shaped
ellipsoid 14 voxels deep and 32 wide -- the Drosophila slab geometry -- against the true
voxel volume:

======  =====  =============
maxrad  faces  mesh / voxels
======  =====  =============
5.0        96         0.51
3.0       334         0.73
2.0       706         0.78
1.5      1256         0.80
1.0      3090         0.81
======  =====  =============

So the default loses half the volume, and the curve flattens by ~1.5. Scale ``maxrad`` to
the object's *thinnest* dimension (roughly depth/8), not to its width. The residual ~20%
is inherent: the surface passes through voxel centres and HC smoothing pulls it in, which
costs proportionally more on a shallow object. ``MeshGeometry.volume_ratio`` reports this
per object, so it is visible rather than assumed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from analysis.volumetric.mesh import (
    DEFAULT_ISOVALUE,
    resolve_maxrad,
    warn_if_maxrad_coarse,
    MeshGeometry,
    MeshingError,
    convex_hull_voxel_count,
    ensure_iso2mesh_binaries,
    generate_mesh,
    mesh_geometry,
)

# Which faces of the volume count as "the edge of the field".
BORDER_MODES = ("none", "xy", "all")


@dataclass
class ObjectMesh:
    """One meshed object: vertices in microns (z, y, x), 1-based faces, and scalars."""

    label: int
    vertices_um: np.ndarray
    faces: np.ndarray
    geometry: MeshGeometry
    frame_index: int = 0
    curvature: Optional["object"] = None      # CurvatureResults, when computed

    @property
    def centroid_um(self) -> np.ndarray:
        return self.vertices_um.mean(axis=0)


@dataclass
class FieldMeshes:
    """Every object meshed from one frame, plus what was skipped and why."""

    meshes: List[ObjectMesh] = field(default_factory=list)
    frame_index: int = 0
    n_labels: int = 0
    skipped_small: List[int] = field(default_factory=list)
    skipped_border: List[int] = field(default_factory=list)
    failed: Dict[int, str] = field(default_factory=dict)
    # Objects meshed with a triangle bound too coarse for their size. Not failures --
    # they produce a closed, plausible mesh whose volume is simply too small, which is
    # precisely why they need saying out loud rather than being left to inspection.
    warnings: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.meshes)

    def describe(self) -> List[str]:
        lines = [
            f"objects meshed      : {len(self.meshes)} of {self.n_labels} labels",
        ]
        if self.skipped_small:
            lines.append(f"skipped (too small) : {len(self.skipped_small)}")
        if self.skipped_border:
            lines.append(f"skipped (at border) : {len(self.skipped_border)}")
        if self.warnings:
            lines.append(f"maxrad too coarse   : {len(self.warnings)} object(s); "
                         f"e.g. {self.warnings[0]}")
        if self.failed:
            lines.append(f"failed to mesh      : {len(self.failed)} "
                         f"{sorted(self.failed)[:5]}")
        if self.meshes:
            volumes = np.array([m.geometry.volume_um3 for m in self.meshes])
            lines.append(f"volume um^3         : median {np.median(volumes):.3f}  "
                         f"range [{volumes.min():.3f}, {volumes.max():.3f}]")
        return lines


def touches_border(bbox: Tuple[slice, ...], shape: Sequence[int], mode: str) -> bool:
    """Does this bounding box reach the edge of the volume?

    ``mode='xy'`` ignores the z faces on purpose. A shallow acquisition -- the Drosophila
    slab is 14 slices -- has *every* object touching the top and bottom, so excluding
    z-border objects would exclude the entire field. Objects cut by the frame edge in xy
    are genuinely incomplete and are the ones worth dropping.
    """
    if mode == "none":
        return False
    axes = (1, 2) if mode == "xy" else (0, 1, 2)
    return any(bbox[a].start == 0 or bbox[a].stop == shape[a] for a in axes)


def iter_objects(
    labels: np.ndarray,
    min_voxels: int = 1,
    exclude_border: str = "xy",
) -> Iterator[Tuple[int, np.ndarray, np.ndarray, str]]:
    """Yield ``(label, sub_mask, offset_zyx, skip_reason)`` for each label.

    ``sub_mask`` is the object alone, cropped to its bounding box with a one-voxel margin
    so the surface closes cleanly, and ``offset_zyx`` is where that crop sits in the
    field. A non-empty ``skip_reason`` means the object was rejected; the caller records
    it rather than having it silently vanish.
    """
    if exclude_border not in BORDER_MODES:
        raise ValueError(f"exclude_border must be one of {BORDER_MODES}")

    array = np.asarray(labels)
    if array.ndim != 3:
        raise MeshingError(f"Expected a 3-D label volume, got shape {array.shape}.")

    boxes = ndimage.find_objects(array)
    for index, bbox in enumerate(boxes):
        if bbox is None:                       # label absent from the volume
            continue
        label_id = index + 1

        if touches_border(bbox, array.shape, exclude_border):
            yield label_id, None, None, "border"
            continue

        # One-voxel margin of background on every side, so the object never abuts the
        # crop and the surface closes.
        #
        # The margin used to be clipped to the volume (`max(sl.start - 1, 0)`), which
        # gave objects touching a face no margin there and so an OPEN surface. With
        # exclude_border="xy" -- the recommended setting, because in a shallow slab every
        # object touches z -- that was the normal case, not an exception. An open surface
        # breaks `mesh_volume`: the signed-tetrahedron sum is translation-invariant only
        # when the surface closes, and vertices here are in FIELD coordinates, so an
        # object 100 um from the origin with a ~100 um^2 hole picked up an offset term of
        # 10^3-10^4 um^3 against a real cell volume of ~10^3. `mesh_geometry` takes the
        # absolute value, so sphericity and solidity inherited it silently, and
        # `curvature` decides winding from the sign -- flipping some objects in a field
        # and not others, inverting their curvature.
        #
        # Zero-padding where the clip would have bitten restores the margin. `offset` is
        # then the field coordinate of the padded sub-array's first voxel, which is -1 on
        # a clipped face: correct, and the arithmetic downstream is unchanged.
        window, offset, pad_width = [], [], []
        for axis, sl in enumerate(bbox):
            start, stop = sl.start - 1, sl.stop + 1
            low, high = max(start, 0), min(stop, array.shape[axis])
            window.append(slice(low, high))
            pad_width.append((low - start, stop - high))
            offset.append(start)

        sub = array[tuple(window)] == label_id
        if int(sub.sum()) < max(min_voxels, 1):
            yield label_id, None, None, "small"
            continue
        if any(before or after for before, after in pad_width):
            sub = np.pad(sub, pad_width, mode="constant", constant_values=False)

        yield label_id, sub, np.asarray(offset, dtype=np.float64), ""


def mesh_field(
    labels: np.ndarray,
    spacing_zyx_um: Sequence[float],
    maxrad: float = 5.0,
    area_frac: float = 0.2,
    smoothing_iterations: int = 10,
    alpha: float = 0.1,
    beta: float = 0.5,
    min_voxels: int = 64,
    exclude_border: str = "xy",
    curvature: bool = True,
    solidity: bool = False,
    frame_index: int = 0,
    iso2mesh_bin: str = "",
    verbose: bool = False,
    isovalue: float = DEFAULT_ISOVALUE,
    maxrad_units: str = "voxels",
) -> FieldMeshes:
    """Mesh every labelled object in one field.

    ``min_voxels`` drops fragments that cannot carry a meaningful surface -- Cellpose
    leaves slivers at frame edges, and meshing a 12-voxel crumb produces a degenerate
    surface whose curvature is noise. ``solidity`` defaults to False here (unlike the
    single-object path) because the voxel convex hull costs about a second per object,
    which is minutes across a field.

    One object failing never aborts the field: the label and its error are recorded in
    ``FieldMeshes.failed`` and the rest continue.
    """
    spacing = np.asarray(spacing_zyx_um, dtype=np.float64)
    # Arity first, as mesh.mesh_object does. Without it a scalar or 2-tuple sails through
    # the isotropy test (min == max) and fails later on `spacing[0]` with an IndexError
    # instead of the MeshingError the caller handles.
    if spacing.size != 3:
        raise MeshingError(
            f"spacing_zyx_um must have three elements (z, y, x); got {spacing.size}."
        )
    if float(spacing.max() - spacing.min()) > 1e-9:
        raise MeshingError(
            f"Meshing needs an isotropic grid but spacing (z, y, x) is {tuple(spacing)} "
            f"um; resample first (analysis.volumetric.resample.prepare_volume)."
        )
    voxel_size = float(spacing[0])

    ensure_iso2mesh_binaries(iso2mesh_bin or "")
    if curvature:
        from analysis.volumetric.curvature import analyze_curvature

    array = np.asarray(labels)
    # How many labels there ARE, not the largest id. Cellpose ids are routinely
    # non-contiguous after any filtering step, so `array.max()` made describe() report
    # "180 of 4200 labels" when 200 labels existed -- which reads as a mass failure.
    result = FieldMeshes(frame_index=frame_index,
                         n_labels=int(np.count_nonzero(np.unique(array))))

    for label_id, sub, offset, skip in iter_objects(array, min_voxels, exclude_border):
        if skip == "border":
            result.skipped_border.append(label_id)
            continue
        if skip == "small":
            result.skipped_small.append(label_id)
            continue

        try:
            # Per object: "voxels" and "um" give the same answer for every object,
            # but "relative" deliberately does not -- that is the point of it.
            maxrad_vox = resolve_maxrad(maxrad, maxrad_units, float(voxel_size),
                                        int(sub.sum()))
            warning = warn_if_maxrad_coarse(maxrad_vox, int(sub.sum()),
                                            f"label {label_id}")
            if warning:
                result.warnings.append(warning)
            vertices_vox, faces = generate_mesh(
                sub, maxrad=maxrad_vox, area_frac=area_frac,
                smoothing_iterations=smoothing_iterations,
                alpha=alpha, beta=beta, verbose=verbose, isovalue=isovalue,
            )
            # Back into field coordinates, then into microns, so every object shares one
            # frame of reference and the field can be rendered as a single scene.
            vertices_um = (vertices_vox + offset) * voxel_size
            geometry = mesh_geometry(
                vertices_um, faces,
                voxel_count=int(sub.sum()),
                voxel_volume_um3=float(np.prod(spacing)),
                hull_voxel_count=convex_hull_voxel_count(sub) if solidity else np.nan,
                mask=sub,
                voxel_size_um=voxel_size,
            )
            mesh = ObjectMesh(
                label=label_id, vertices_um=vertices_um, faces=faces,
                geometry=geometry, frame_index=frame_index,
            )
            if curvature:
                mesh.curvature = analyze_curvature(vertices_um, faces)
            result.meshes.append(mesh)
        except Exception as exc:                # noqa: BLE001 - recorded, not swallowed
            result.failed[label_id] = f"{type(exc).__name__}: {exc}"

    return result
