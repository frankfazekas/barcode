"""One barcode row per segmented object.

For a field of many objects -- a Drosophila embryo is ~840 cells -- the interesting
comparison is between the objects, not between fields. This turns a run into per-object
rows so the barcode's per-column normalisation runs across the population of cells, which
is what makes a size or packing gradient visible at all.

**Nothing here measures anything new.** Every value already exists per object somewhere in
the run: contact numbers on ``VolumetricPackingDetail``, the seven in-mask statistics on
``MaskIntensityDetail``, and volumes from a ``bincount`` of the label array. This module
only joins them, by object id, into rows.

The column set is deliberately smaller than a field row's. Most metrics are field-level
by definition -- there is no per-object connectivity, correlation length, kurtosis or
optical flow -- and following the rule the analysis modes already use, a column that
cannot mean anything is omitted rather than filled with NaN or, worse, filled with the
field's value repeated down every row.

That leaves ~10 columns today. It becomes ~22 when per-object meshing lands: ``mesh.py``
still calls ``largest_component(mask)``, so shape metrics exist for one object per volume.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from core.metrics import Metrics, Units


@dataclass
class ObjectResults:
    """One object: its identity, and the metrics defined for a single object."""

    filepath: str = ""
    fov: str = ""
    object_id: int = 0
    # The renderer splits rows by channel; objects inherit their field's channel.
    channel: int = 0

    volume: float = np.nan            # um^3
    anisotropy: float = np.nan        # principal-axis major/minor ratio, >= 1
    contact_number: float = np.nan

    mfi: float = np.nan
    intensity_sd: float = np.nan
    intensity_cv: float = np.nan
    intensity_skew: float = np.nan
    entropy: float = np.nan
    entropy_normalized: float = np.nan
    bright_fraction: float = np.nan

    # Shape, from this object's OWN mesh. Absent until per-object meshing is enabled --
    # `mesh.mesh_object` starts with `largest_component`, so a whole-field mesh describes
    # one cell and these would otherwise be that cell's numbers on every row.
    surface_area: float = np.nan
    sphericity: float = np.nan
    solidity: float = np.nan
    aspect_ratio: float = np.nan
    mean_curvature: float = np.nan

    # ---- the surface the writer and the barcode renderer expect -------------------
    #
    # The in-mask intensity family is GATED behind ``include_mask_intensity`` (default
    # off). "Intensity inside the mask" only means something when the fluorophore labels
    # what fills the object -- chromatin in a nucleus, say. For a membrane marker (the
    # Drosophila case) the mask IS the cell and the signal is its boundary, so those seven
    # columns measure noise dressed up as a readout. Off by default, on when the caller
    # opts in; the switch rides the same ``--mask-intensity`` flag that computes them, so a
    # run that never computed them never shows them either.

    _MASK_INTENSITY_METRICS = [
        Metrics.MASK_INTENSITY_MFI,
        Metrics.MASK_INTENSITY_SD,
        Metrics.MASK_INTENSITY_CV,
        Metrics.MASK_INTENSITY_SKEW,
        Metrics.MASK_INTENSITY_ENTROPY,
        Metrics.MASK_INTENSITY_ENTROPY_NORM,
        Metrics.MASK_INTENSITY_BRIGHT_FRACTION,
    ]

    @classmethod
    def get_metrics(cls, mode=None, include_mask_intensity: bool = False, **_) -> List[Metrics]:
        metrics = [
            Metrics.OBJECT_VOLUME,
            Metrics.OBJECT_ANISOTROPY,
            Metrics.OBJECT_CONTACT_NUMBER,
        ]
        if include_mask_intensity:
            metrics += list(cls._MASK_INTENSITY_METRICS)
        metrics += [
            Metrics.MESH_SURFACE_AREA,
            Metrics.MESH_SPHERICITY,
            Metrics.MESH_SOLIDITY,
            Metrics.MESH_ASPECT_RATIO,
            Metrics.CURVATURE_MEAN,
        ]
        return metrics

    @classmethod
    def get_units(cls, mode=None, include_mask_intensity: bool = False, **_) -> List[Units]:
        units = [Units.VOLUME, Units.NONE, Units.NONE]
        if include_mask_intensity:
            units += [Units.INTENSITY, Units.INTENSITY, Units.NONE, Units.NONE,
                      Units.NONE, Units.NONE, Units.NONE]
        units += [Units.AREA, Units.NONE, Units.NONE, Units.NONE, Units.CURVATURE]
        return units

    @classmethod
    def get_headers(cls, just_metrics: bool = True, mode=None,
                    include_mask_intensity: bool = False, **_) -> List[str]:
        headers = [metric.value for metric in
                   cls.get_metrics(mode=mode, include_mask_intensity=include_mask_intensity)]
        if just_metrics:
            return headers
        return ["File", "FOV", "Object"] + headers

    def get_data(self, include_mask_intensity: bool = False, **_) -> List[float]:
        data = [self.volume, self.anisotropy, self.contact_number]
        if include_mask_intensity:
            data += [self.mfi, self.intensity_sd, self.intensity_cv, self.intensity_skew,
                     self.entropy, self.entropy_normalized, self.bright_fraction]
        data += [self.surface_area, self.sphericity, self.solidity,
                 self.aspect_ratio, self.mean_curvature]
        return data

    def to_array(self, just_metrics: bool = True, mode=None,
                 include_mask_intensity: bool = False, **_) -> np.ndarray:
        return np.array(self.get_data(include_mask_intensity=include_mask_intensity), dtype=float)

    def to_physical_array(self, **kwargs) -> np.ndarray:
        """Already physical: volumes are um^3, not fractions of the field."""
        return self.to_array(**kwargs)

    # Object rows are physical to begin with -- a volume in um^3, not a fraction of the
    # field -- so the physical variants are the same columns. Defined so the renderer's
    # physical_units path does not have to special-case which schema it was handed.
    @classmethod
    def get_physical_metrics(cls, mode=None, include_mask_intensity: bool = False, **_) -> List[Metrics]:
        return cls.get_metrics(mode=mode, include_mask_intensity=include_mask_intensity)

    @classmethod
    def get_physical_headers(cls, just_metrics: bool = True, mode=None,
                             include_mask_intensity: bool = False, **_) -> List[str]:
        return cls.get_headers(just_metrics, mode, include_mask_intensity=include_mask_intensity)

    @classmethod
    def family_switches_for(cls, results) -> Dict[str, bool]:
        """Which gated object families actually carry data.

        ``ObjectResults`` is a flat schema, so the renderer's ``OPTIONAL_FAMILIES``
        detection (which reads a sub-object per family) cannot size the picture for it.
        This reports the same answer by inspecting the rows: in-mask columns are shown only
        when at least one object was actually measured for them, i.e. ``--mask-intensity``
        was on. A run that never computed them never draws them.
        """
        has_inmask = any(np.isfinite(getattr(r, "mfi", np.nan)) for r in results)
        return {"include_mask_intensity": bool(has_inmask)}

    def get_row(self, include_mask_intensity: bool = False) -> List:
        return ([self.filepath, self.fov, self.object_id]
                + self.get_data(include_mask_intensity=include_mask_intensity))

    def convert_flags(self) -> str:
        """Objects carry no flags of their own; the field's flags describe the run."""
        return "0"

    def is_populated(self) -> bool:
        return bool(np.any(np.isfinite(np.array(self.get_data(), dtype=float))))


def _paired(detail_list: Optional[Sequence], attribute: str) -> Dict[int, float]:
    """``{object_id: value}`` from the first frame's detail, or empty.

    Joining on the id rather than on position is the point: the packing and in-mask
    details enumerate objects independently, and a positional join would silently
    attribute one cell's intensity to another whenever the two lists differ -- which they
    do, because packing keeps border objects that in-mask may skip as too small.
    """
    if not detail_list:
        return {}
    first = detail_list[0]
    ids = getattr(first, "object_ids", None)
    values = getattr(first, attribute, None)
    if not ids or values is None or len(values) != len(ids):
        return {}
    return {int(i): float(v) for i, v in zip(ids, values)}


def _object_anisotropy(labels: np.ndarray, spacing_zyx_um) -> Dict[int, float]:
    """``{object_id: anisotropy}`` — the inertia-ellipsoid major/minor ratio per object.

    Reuses the field branch's ``_anisotropy_from_eigvals`` so the per-object number and the
    field-level Mean Island Anisotropy cannot drift apart, and passes ``spacing`` so the
    ratio describes the object rather than the sampling grid (the same fix documented for
    the field metric). Needs no mesh, which is why this is a default 3D object metric.
    """
    from skimage.measure import regionprops_table

    from analysis.volumetric.binarization import _anisotropy_from_eigvals

    integer = labels if np.issubdtype(labels.dtype, np.integer) else labels.astype(np.int32)
    if integer.max() < 1:
        return {}
    props = regionprops_table(
        integer,
        properties=["label", "inertia_tensor_eigvals"],
        spacing=tuple(float(s) for s in spacing_zyx_um),
    )
    eigvals = np.stack(
        [props[f"inertia_tensor_eigvals-{i}"] for i in range(3)], axis=1
    )
    ratios = _anisotropy_from_eigvals(eigvals)
    return {int(lab): float(r) for lab, r in zip(props["label"], ratios)}


def extract_objects(
    labels: np.ndarray,
    spacing_zyx_um,
    detail=None,
    filepath: str = "",
    fov: str = "",
    meshes=None,
) -> List[ObjectResults]:
    """Join the per-object values a run already produced into one row per object."""
    labels = np.asarray(labels)
    if labels.dtype == bool:
        labels = labels.astype(np.int32)

    counts = np.bincount(labels.ravel())
    if counts.size <= 1:
        return []
    counts[0] = 0
    present = np.nonzero(counts)[0]
    if present.size == 0:
        return []

    voxel_um3 = float(np.prod(np.asarray(spacing_zyx_um, dtype=float)))
    anisotropy = _object_anisotropy(labels, spacing_zyx_um)
    contacts = _paired(getattr(detail, "packing", None), "contact_numbers")
    intensity = {
        name: _paired(getattr(detail, "mask_intensity", None), name)
        for name in ("mfi", "sd", "cv", "skewness", "entropy",
                     "entropy_normalized", "bright_fraction")
    }

    meshes = meshes or {}

    rows: List[ObjectResults] = []
    for object_id in present:
        object_id = int(object_id)
        volume = float(counts[object_id]) * voxel_um3
        rows.append(ObjectResults(
            filepath=filepath,
            fov=fov,
            object_id=object_id,
            volume=volume,
            anisotropy=anisotropy.get(object_id, np.nan),
            contact_number=contacts.get(object_id, np.nan),
            mfi=intensity["mfi"].get(object_id, np.nan),
            intensity_sd=intensity["sd"].get(object_id, np.nan),
            intensity_cv=intensity["cv"].get(object_id, np.nan),
            intensity_skew=intensity["skewness"].get(object_id, np.nan),
            entropy=intensity["entropy"].get(object_id, np.nan),
            entropy_normalized=intensity["entropy_normalized"].get(object_id, np.nan),
            bright_fraction=intensity["bright_fraction"].get(object_id, np.nan),
            **_shape_of(meshes.get(object_id)),
        ))
    return rows


def _shape_of(mesh) -> Dict[str, float]:
    """Shape columns from one object's mesh, all NaN when it has none.

    An object can legitimately lack a mesh -- too small, or rejected because its meshed
    volume disagreed with its voxel count -- and NaN is the honest answer there. Falling
    back to the field mesh would put one cell's shape on every row.
    """
    if mesh is None:
        return {}
    geometry = mesh.geometry
    curvature = getattr(mesh, "curvature", None)
    return {
        "surface_area": float(getattr(geometry, "surface_area_um2", np.nan)),
        "sphericity": float(getattr(geometry, "sphericity", np.nan)),
        "solidity": float(getattr(geometry, "mesh_solidity", np.nan)),
        "aspect_ratio": float(getattr(geometry, "aspect_ratio", np.nan)),
        "mean_curvature": float(getattr(curvature, "mean_curvature", np.nan))
                          if curvature is not None else np.nan,
    }


def objects_to_csv(rows: Sequence[ObjectResults], path: str) -> str:
    """Write per-object rows, identity columns first.

    The in-mask intensity family is written only when it was actually measured (same gate
    as the barcode), so a membrane-marker run does not carry seven NaN columns pretending
    to be a chromatin readout.
    """
    import csv

    include_mask_intensity = (
        ObjectResults.family_switches_for(rows)["include_mask_intensity"] if rows else False
    )
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(ObjectResults.get_headers(
            just_metrics=False, include_mask_intensity=include_mask_intensity))
        for row in rows:
            writer.writerow(row.get_row(include_mask_intensity=include_mask_intensity))
    return path
