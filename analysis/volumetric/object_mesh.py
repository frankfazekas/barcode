"""Mesh every object in a label volume, not just the biggest one.

``mesh.mesh_object`` measures one surface, and ``mesh_series`` fed it a whole mask —
which begins ``largest_component(mask_zyx)``, so a field of 839 Cellpose cells produced
exactly one mesh and every shape metric described the single largest cell. For a
single-nucleus dataset that is right. For a field of cells it silently answers a
different question from the one asked.

This meshes each label separately, so sphericity, solidity, concavity and curvature
become per-object quantities and can be rows of a barcode.

Two things make it affordable. Each object is **cropped to its own bounding box** before
meshing, so the cost scales with the object rather than the field — a 6 um cell in a
350 um embryo is a ~40^3 crop, not 1500x1808x15. And it is **opt-in**, because hundreds
of iso2mesh calls per timepoint is real time however well it is cropped.

Objects touching the field border are meshed but flagged: their surface is cut off, so
sphericity and solidity describe a truncated shape. That is reported, not hidden — the
same rule the packing family follows when it excludes border objects from statistics.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from analysis.volumetric.mesh import MeshingError, ObjectMesh, mesh_object

# Below this an object is a fragment, not a cell: a marching-cubes surface over a handful
# of voxels is dominated by the voxel grid and its sphericity is meaningless.
DEFAULT_MIN_VOXELS = 64

# Triangle-size bound, as a FRACTION of each object's own radius rather than in voxels.
# The package default (5 voxels) was tuned for a nucleus of ~65-voxel radius; a Drosophila
# cell is 12-17, where 5 voxels is 30-40% of the radius and the mesh loses tens of percent
# of the volume -- measured: one cell of 208 um^3 by voxel count meshed to 146. "relative"
# is the only unit that holds accuracy constant across a field of differently-sized
# objects, which is exactly what per-object meshing walks into. mesh.resolve_maxrad's own
# table says 0.1 is a good value.
DEFAULT_OBJECT_MAXRAD = 0.1
DEFAULT_OBJECT_MAXRAD_UNITS = "relative"

# A mesh whose volume disagrees with the voxel count by more than this has not closed.
# Deliberately wide -- marching cubes legitimately differs by a few percent -- so it
# catches collapse, not ordinary discretisation error.
VOLUME_RATIO_LIMITS = (0.5, 2.0)


@dataclass
class ObjectMeshDetail:
    """What happened, so a run can say why an object has no shape metrics."""

    meshed: List[int] = None
    too_small: int = 0
    failed: int = 0
    border: List[int] = None
    reasons: Dict[int, str] = None
    limited_to: int = 0        # non-zero when only the first N objects were meshed

    def __post_init__(self):
        self.meshed = self.meshed if self.meshed is not None else []
        self.border = self.border if self.border is not None else []
        self.reasons = self.reasons if self.reasons is not None else {}

    def describe(self) -> str:
        parts = [f"object meshes: {len(self.meshed)} built"]
        if self.too_small:
            parts.append(f"{self.too_small} too small")
        if self.failed:
            parts.append(f"{self.failed} failed")
        if self.border:
            parts.append(f"{len(self.border)} touch the border (surface truncated)")
        if self.limited_to:
            parts.append(f"LIMITED to the first {self.limited_to} objects; "
                         f"the rest have no shape metrics")
        return ", ".join(parts)


def bounding_boxes(labels: np.ndarray) -> Dict[int, Tuple[slice, slice, slice]]:
    """One pass for every object's bounding box.

    ``scipy.ndimage.find_objects`` indexes by ``label - 1`` and returns ``None`` for ids
    that are absent, which is exactly right for the non-contiguous ids Cellpose produces.
    Doing this per object with ``np.where(labels == i)`` would be a full pass over the
    volume per object -- 839 passes over 40 million voxels.
    """
    from scipy import ndimage

    found = ndimage.find_objects(labels)
    return {index + 1: box for index, box in enumerate(found) if box is not None}


def _touches_border(box: Tuple[slice, slice, slice], shape: Sequence[int]) -> bool:
    return any(s.start == 0 or s.stop == n for s, n in zip(box, shape))


def mesh_objects(
    labels: np.ndarray,
    spacing_zyx_um: Sequence[float],
    config,
    min_voxels: int = DEFAULT_MIN_VOXELS,
    pad: int = 2,
    verbose: bool = False,
    frame_index: int = 0,
    maxrad: float = DEFAULT_OBJECT_MAXRAD,
    maxrad_units: str = DEFAULT_OBJECT_MAXRAD_UNITS,
    limit: int = 0,
) -> Tuple[Dict[int, ObjectMesh], ObjectMeshDetail]:
    """``({object id: ObjectMesh}, detail)`` for every object worth meshing.

    One object failing must not lose the rest: a mask can contain a shape iso2mesh
    cannot close, and in a field of hundreds that is close to certain.
    """
    labels = np.asarray(labels)
    if labels.dtype == bool:
        labels = labels.astype(np.int32)

    detail = ObjectMeshDetail()
    meshes: Dict[int, ObjectMesh] = {}
    if labels.max() == 0:
        return meshes, detail

    counts = np.bincount(labels.ravel())
    boxes = bounding_boxes(labels)
    curvature_on = bool(getattr(config, "mesh_curvature", False))

    # `limit` meshes only the first N objects, for iterating on settings without paying
    # ~35 minutes a field. The rest get NaN shape columns, and the count is reported --
    # a partial run must never look like a complete one.
    ordered = sorted(boxes.items())
    if limit and limit > 0:
        ordered = ordered[:limit]
        detail.limited_to = int(limit)

    for object_id, box in ordered:
        if object_id >= counts.size or counts[object_id] < min_voxels:
            detail.too_small += 1
            detail.reasons[object_id] = f"{int(counts[object_id])} voxels < {min_voxels}"
            continue

        if _touches_border(box, labels.shape):
            detail.border.append(object_id)

        # Crop with padding so marching cubes closes the surface instead of running into
        # the array edge, which would leave a hole exactly where the padding is missing.
        padded = tuple(
            slice(max(0, s.start - pad), min(n, s.stop + pad))
            for s, n in zip(box, labels.shape)
        )
        crop = labels[padded] == object_id
        crop = np.pad(crop, pad)      # guarantee background on every side

        try:
            mesh = mesh_object(
                crop,
                spacing_zyx_um,
                maxrad=maxrad,
                maxrad_units=maxrad_units,
                isovalue=getattr(config, "mesh_isovalue", 0.5),
                area_frac=getattr(config, "mesh_area_frac", 0.2),
                smoothing_iterations=getattr(config, "mesh_smoothing_iterations", 10),
                alpha=getattr(config, "mesh_smoothing_alpha", 0.1),
                beta=getattr(config, "mesh_smoothing_beta", 0.5),
                matlab_compat=getattr(config, "mesh_matlab_compat", False),
                verbose=False,
                frame_index=frame_index,
                solidity=bool(getattr(config, "mesh_solidity", True)),
            )
        except (MeshingError, Exception) as exc:      # noqa: B014 - iso2mesh raises broadly
            detail.failed += 1
            detail.reasons[object_id] = str(exc)[:120]
            continue

        # The voxel count is ground truth for volume, so a mesh that disagrees with it by
        # a wide margin has failed to close whatever iso2mesh reported. Measured: one
        # object of 93.7 um^3 meshed to 0.2 -- no exception, no warning, and a sphericity
        # that would have gone straight into the barcode as if it meant something.
        voxel_volume = float(counts[object_id]) * float(np.prod(np.asarray(spacing_zyx_um)))
        meshed_volume = float(getattr(mesh.geometry, "volume_um3", np.nan))
        ratio = meshed_volume / voxel_volume if voxel_volume > 0 else np.nan
        if not np.isfinite(ratio) or not (VOLUME_RATIO_LIMITS[0] <= ratio <= VOLUME_RATIO_LIMITS[1]):
            detail.failed += 1
            detail.reasons[object_id] = (
                f"mesh volume {meshed_volume:.3g} um^3 vs {voxel_volume:.3g} by voxel "
                f"count (ratio {ratio:.3g}); degenerate surface"
            )
            continue

        if curvature_on:
            try:
                from analysis.volumetric.curvature import analyze_curvature

                mesh.curvature = analyze_curvature(
                    mesh.vertices_um, mesh.faces,
                    exclude_caps=bool(getattr(config, "curvature_exclude_caps", False)),
                    outlier_limit=float(getattr(config, "curvature_outlier_limit", 0.0)),
                )
            except Exception as exc:
                detail.reasons[object_id] = f"curvature: {str(exc)[:100]}"

        meshes[object_id] = mesh
        detail.meshed.append(object_id)

    if verbose:
        print(f"  {detail.describe()}", flush=True)
    return meshes, detail
