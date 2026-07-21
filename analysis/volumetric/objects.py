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
    diameter: float = np.nan          # um, equivalent-sphere
    contact_number: float = np.nan

    mfi: float = np.nan
    intensity_sd: float = np.nan
    intensity_cv: float = np.nan
    intensity_skew: float = np.nan
    entropy: float = np.nan
    entropy_normalized: float = np.nan
    bright_fraction: float = np.nan

    # ---- the surface the writer and the barcode renderer expect -------------------

    @classmethod
    def get_metrics(cls, mode=None, **_) -> List[Metrics]:
        return [
            Metrics.OBJECT_VOLUME,
            Metrics.OBJECT_DIAMETER,
            Metrics.OBJECT_CONTACT_NUMBER,
            Metrics.MASK_INTENSITY_MFI,
            Metrics.MASK_INTENSITY_SD,
            Metrics.MASK_INTENSITY_CV,
            Metrics.MASK_INTENSITY_SKEW,
            Metrics.MASK_INTENSITY_ENTROPY,
            Metrics.MASK_INTENSITY_ENTROPY_NORM,
            Metrics.MASK_INTENSITY_BRIGHT_FRACTION,
        ]

    @classmethod
    def get_units(cls, mode=None, **_) -> List[Units]:
        return [
            Units.VOLUME, Units.LENGTH, Units.NONE,
            Units.INTENSITY, Units.INTENSITY, Units.NONE, Units.NONE,
            Units.NONE, Units.NONE, Units.NONE,
        ]

    @classmethod
    def get_headers(cls, just_metrics: bool = True, mode=None, **_) -> List[str]:
        headers = [metric.value for metric in cls.get_metrics(mode)]
        if just_metrics:
            return headers
        return ["File", "FOV", "Object"] + headers

    def get_data(self, **_) -> List[float]:
        return [
            self.volume, self.diameter, self.contact_number,
            self.mfi, self.intensity_sd, self.intensity_cv, self.intensity_skew,
            self.entropy, self.entropy_normalized, self.bright_fraction,
        ]

    def to_array(self, just_metrics: bool = True, mode=None, **_) -> np.ndarray:
        return np.array(self.get_data(), dtype=float)

    def to_physical_array(self, **kwargs) -> np.ndarray:
        """Already physical: volumes are um^3 and diameters um, not fractions."""
        return self.to_array(**kwargs)

    # Object rows are physical to begin with -- a volume in um^3, not a fraction of the
    # field -- so the physical variants are the same columns. Defined so the renderer's
    # physical_units path does not have to special-case which schema it was handed.
    @classmethod
    def get_physical_metrics(cls, mode=None, **_) -> List[Metrics]:
        return cls.get_metrics(mode)

    @classmethod
    def get_physical_headers(cls, just_metrics: bool = True, mode=None, **_) -> List[str]:
        return cls.get_headers(just_metrics, mode)

    def get_row(self) -> List:
        return [self.filepath, self.fov, self.object_id] + self.get_data()

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


def extract_objects(
    labels: np.ndarray,
    spacing_zyx_um,
    detail=None,
    filepath: str = "",
    fov: str = "",
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
    contacts = _paired(getattr(detail, "packing", None), "contact_numbers")
    intensity = {
        name: _paired(getattr(detail, "mask_intensity", None), name)
        for name in ("mfi", "sd", "cv", "skewness", "entropy",
                     "entropy_normalized", "bright_fraction")
    }

    rows: List[ObjectResults] = []
    for object_id in present:
        object_id = int(object_id)
        volume = float(counts[object_id]) * voxel_um3
        rows.append(ObjectResults(
            filepath=filepath,
            fov=fov,
            object_id=object_id,
            volume=volume,
            # Equivalent-sphere diameter: a size in microns, comparable between objects
            # of different shape, and far easier to read than a cubic micron.
            diameter=2.0 * (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0),
            contact_number=contacts.get(object_id, np.nan),
            mfi=intensity["mfi"].get(object_id, np.nan),
            intensity_sd=intensity["sd"].get(object_id, np.nan),
            intensity_cv=intensity["cv"].get(object_id, np.nan),
            intensity_skew=intensity["skewness"].get(object_id, np.nan),
            entropy=intensity["entropy"].get(object_id, np.nan),
            entropy_normalized=intensity["entropy_normalized"].get(object_id, np.nan),
            bright_fraction=intensity["bright_fraction"].get(object_id, np.nan),
        ))
    return rows


def objects_to_csv(rows: Sequence[ObjectResults], path: str) -> str:
    """Write per-object rows, identity columns first."""
    import csv

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(ObjectResults.get_headers(just_metrics=False))
        for row in rows:
            writer.writerow(row.get_row())
    return path
