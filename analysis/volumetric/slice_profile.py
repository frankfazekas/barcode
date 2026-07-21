"""Where through a stack the object is widest, and whether it is cut off.

Every other volumetric metric reduces a stack to one number per timepoint, which
answers "how much" and "what shape" but never "where in depth". Two things fall out of
the per-slice foreground profile that nothing else in the branch can say:

* **the maximal-area slice** -- for a stack through a curved surface or a rounded object it
  locates the equator, and it moves when the object flattens, tilts, or drifts out of
  the focal range;
* **field-of-view clipping** -- whether the foreground runs off an edge of the analysed
  field, in which case every size and shape metric describes a truncated object.

Clipping is reported as flag digit 6, kept separate from digit 5 on purpose. Digit 5
means the *user* restricted the analysis; digit 6 means the *data* is cut off. They look
similar in a CSV and mean opposite things about whose decision it was.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class SliceProfileDetail:
    """The per-slice foreground profile the scalars summarise."""

    areas: List[float] = field(default_factory=list)      # fraction of the slice
    max_area_index: int = -1
    max_area_depth_um: float = np.nan
    clipped_xy: bool = False
    clipped_z: bool = False

    @property
    def clipped(self) -> bool:
        return bool(self.clipped_xy or self.clipped_z)

    def describe(self) -> str:
        if self.max_area_index < 0:
            return "slice profile: empty volume"
        where = []
        if self.clipped_xy:
            where.append("xy border")
        if self.clipped_z:
            where.append("first/last slice")
        edge = f", clipped at {' and '.join(where)}" if where else ""
        return (f"slice profile: max area at z={self.max_area_index} "
                f"({self.max_area_depth_um:.3g} um), "
                f"{100 * self.areas[self.max_area_index]:.1f}% of the slice{edge}")


def slice_areas(binary: np.ndarray) -> np.ndarray:
    """Foreground fraction of each z slice of a ``(Z, Y, X)`` volume."""
    volume = np.asarray(binary).astype(bool)
    if volume.ndim != 3:
        raise ValueError(f"expected a (Z, Y, X) volume, got shape {volume.shape}")
    per_slice = volume.reshape(volume.shape[0], -1)
    return per_slice.mean(axis=1)


def touches_xy_border(binary: np.ndarray) -> bool:
    """Whether foreground reaches the first or last row or column of any slice."""
    volume = np.asarray(binary).astype(bool)
    if not volume.any():
        return False
    return bool(volume[:, 0, :].any() or volume[:, -1, :].any()
                or volume[:, :, 0].any() or volume[:, :, -1].any())


def touches_z_border(binary: np.ndarray) -> bool:
    """Whether foreground is present in the first or last analysed slice.

    This is about the object continuing past the stack, so it is judged on the slices
    actually analysed -- a z-range restriction moves where the boundary is, which is
    correct: metrics computed on the restricted stack really are truncated there.
    """
    volume = np.asarray(binary).astype(bool)
    if not volume.any():
        return False
    return bool(volume[0].any() or volume[-1].any())


def slice_profile(binary: np.ndarray, z_step_um: float) -> Tuple["SliceProfileResults",
                                                                 SliceProfileDetail]:
    """Maximal-area-slice statistics and clipping status for one ``(Z, Y, X)`` volume.

    Depth is measured from the first analysed slice, so it is unaffected by a z-range
    restriction -- and correspondingly is not an absolute stage position.
    """
    from core.results import SliceProfileResults

    areas = slice_areas(binary)
    detail = SliceProfileDetail(areas=[float(a) for a in areas])

    if not np.any(areas > 0):
        return SliceProfileResults(), detail

    max_area = int(np.argmax(areas))
    detail.max_area_index = max_area
    detail.max_area_depth_um = float(max_area * float(z_step_um))
    detail.clipped_xy = touches_xy_border(binary)
    detail.clipped_z = touches_z_border(binary)

    return (
        SliceProfileResults(
            max_area_index=float(max_area),
            max_area_depth=detail.max_area_depth_um,
            max_area_area=float(areas[max_area]),
        ),
        detail,
    )


def summarise_slice_profile(results: Sequence["SliceProfileResults"]) -> "SliceProfileResults":
    """Average the per-timepoint scalars, matching how every other family is reduced."""
    from core.results import SliceProfileResults

    def mean_of(attribute: str) -> float:
        values = np.array([getattr(r, attribute) for r in results], dtype=np.float64)
        finite = values[np.isfinite(values)]
        return float(finite.mean()) if finite.size else np.nan

    return SliceProfileResults(
        max_area_index=mean_of("max_area_index"),
        max_area_depth=mean_of("max_area_depth"),
        max_area_area=mean_of("max_area_area"),
    )
