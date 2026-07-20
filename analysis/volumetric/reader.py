"""Axis-aware loading of volumetric TIFF stacks.

The 2D reader (``utils/reader.py``) infers layout from array shape — it appends a
channel axis to any 3-D array and uses ``min(file.shape)`` to find the channel axis.
For a ``(Z, Y, X)`` stack that silently yields "Z timepoints, 1 channel", which runs
without error and analyses Z as if it were time. This module never guesses: it reads
the axis order that the file declares and raises if the file does not declare one.

Returns ``(T, Z, Y, X)`` for a single channel, plus the physical voxel spacing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Optional, Tuple

import numpy as np
import tifffile

# Axes tifffile uses for "I don't know what this is": a plain page sequence (I/Q),
# or a generic sample axis (S). Guessing what these mean is exactly the bug this
# module exists to avoid.
_UNKNOWN_AXES = set("IQS")

# Axes we understand and can map onto (T, Z, Y, X, C).
_KNOWN_AXES = set("TZCYX")


@dataclass
class VolumeStack:
    """A single-channel volumetric stack in canonical ``(T, Z, Y, X)`` order."""

    data: np.ndarray
    z_step_um: float
    xy_step_um: float
    exposure_time_s: float
    axes: str
    source_path: str
    channel: int = 0
    metadata_source: dict = field(default_factory=dict)
    # (start, stop) of the analysed slices within the acquired stack; None = all of it.
    z_range: tuple = None

    @property
    def n_timepoints(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_slices(self) -> int:
        return int(self.data.shape[1])

    @property
    def is_timelapse(self) -> bool:
        return self.n_timepoints > 1

    @property
    def spacing_zyx_um(self) -> Tuple[float, float, float]:
        return (self.z_step_um, self.xy_step_um, self.xy_step_um)

    @property
    def spacing_xyz_um(self) -> Tuple[float, float, float]:
        """SimpleITK orders spacing (x, y, z) while arrays are (Z, Y, X)."""
        return (self.xy_step_um, self.xy_step_um, self.z_step_um)

    def volume(self, t: int = 0) -> np.ndarray:
        """The ``(Z, Y, X)`` volume at timepoint ``t``."""
        return self.data[t]

    def restrict_z(self, z_start: int = 0, z_end: int = 0) -> "VolumeStack":
        """Return a copy limited to a range of z slices.

        ``z_end`` of 0 means "to the last slice"; negatives index from the end, matching
        Python slicing. Raises on an empty or reversed range rather than returning a
        zero-slice stack, which would surface much later as an unexplained NaN.
        """
        n_z = self.n_slices
        start = z_start + n_z if z_start < 0 else z_start
        stop = n_z if z_end == 0 else (z_end + n_z if z_end < 0 else z_end)
        start, stop = max(start, 0), min(stop, n_z)

        if stop <= start:
            raise ValueError(
                f"{os.path.basename(self.source_path)}: z range [{z_start}, {z_end}) "
                f"selects no slices from a {n_z}-slice stack."
            )
        if (start, stop) == (0, n_z):
            return self

        restricted = replace(self, data=self.data[:, start:stop])
        restricted.z_range = (start, stop)
        return restricted

    def describe(self) -> str:
        anisotropy = self.z_step_um / self.xy_step_um if self.xy_step_um else float("nan")
        return (
            f"{os.path.basename(self.source_path)}: axes={self.axes} "
            f"(T,Z,Y,X)={self.data.shape} dtype={self.data.dtype} "
            f"z={self.z_step_um:g}um xy={self.xy_step_um:g}um "
            f"anisotropy={anisotropy:.3f}x channel={self.channel}"
            + (f" z[{self.z_range[0]}:{self.z_range[1]}]" if self.z_range else "")
        )


def _rational_to_float(value) -> Optional[float]:
    """TIFF resolution tags are (numerator, denominator) rationals."""
    try:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            num, den = float(value[0]), float(value[1])
            return num / den if den else None
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _xy_spacing_from_tags(page) -> Optional[float]:
    """Microns per pixel from XResolution (which is *pixels per unit*)."""
    tag = page.tags.get("XResolution")
    if tag is None:
        return None
    px_per_unit = _rational_to_float(tag.value)
    if not px_per_unit:
        return None
    return 1.0 / px_per_unit


def read_axes(path: str) -> Tuple[str, Tuple[int, ...]]:
    """Return the declared axis order and shape without loading pixel data."""
    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        return series.axes, tuple(int(v) for v in series.shape)


def read_volume(
    path: str,
    channel: int = 0,
    z_step_um: Optional[float] = None,
    xy_step_um: Optional[float] = None,
    exposure_time_s: Optional[float] = None,
) -> VolumeStack:
    """Load a volumetric TIFF as ``(T, Z, Y, X)`` for one channel.

    Physical spacing is read from ImageJ metadata when present. Explicit arguments
    always win over file metadata; if neither supplies a value the spacing falls back
    to 1.0 with a printed warning, because silently assuming a voxel size would make
    every physical metric wrong by an unknown factor.
    """
    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        axes = series.axes
        array = series.asarray()
        ij = tf.imagej_metadata or {}
        tag_xy = _xy_spacing_from_tags(tf.pages[0])

    unknown = set(axes) & _UNKNOWN_AXES
    if unknown:
        raise ValueError(
            f"{os.path.basename(path)}: TIFF declares axes {axes!r}, which contains "
            f"undetermined axes {sorted(unknown)!r}. BARCODE will not guess whether "
            f"these are Z or T — re-save as an ImageJ hyperstack with explicit axes."
        )
    unsupported = set(axes) - _KNOWN_AXES
    if unsupported:
        raise ValueError(
            f"{os.path.basename(path)}: unsupported TIFF axes {sorted(unsupported)!r} "
            f"in {axes!r}; expected some arrangement of T, Z, C, Y, X."
        )
    if "Y" not in axes or "X" not in axes:
        raise ValueError(f"{os.path.basename(path)}: axes {axes!r} lack Y and/or X.")
    if "Z" not in axes:
        raise ValueError(
            f"{os.path.basename(path)}: axes {axes!r} have no Z axis — this is not a "
            f"volumetric stack. Use the standard (2D) BARCODE pipeline for this file."
        )

    # Reorder to (T, Z, C, Y, X), inserting length-1 axes for anything absent.
    for missing in ("T", "Z", "C"):
        if missing not in axes:
            array = np.expand_dims(array, axis=0)
            axes = missing + axes
    array = np.transpose(array, [axes.index(a) for a in "TZCYX"])

    n_channels = array.shape[2]
    if not -n_channels <= channel < n_channels:
        raise ValueError(
            f"{os.path.basename(path)}: channel {channel} out of range for "
            f"{n_channels} channel(s)."
        )
    data = np.ascontiguousarray(array[:, :, channel])

    # Physical spacing: explicit argument > file metadata > warn-and-default.
    z_um = z_step_um if z_step_um is not None else ij.get("spacing")
    xy_um = xy_step_um if xy_step_um is not None else tag_xy
    exposure = exposure_time_s if exposure_time_s is not None else ij.get("finterval")

    if z_um is None:
        print(f"Warning: {os.path.basename(path)} has no z spacing; assuming 1.0 um.", flush=True)
        z_um = 1.0
    if xy_um is None:
        print(f"Warning: {os.path.basename(path)} has no xy spacing; assuming 1.0 um.", flush=True)
        xy_um = 1.0

    return VolumeStack(
        data=data,
        z_step_um=float(z_um),
        xy_step_um=float(xy_um),
        exposure_time_s=float(exposure if exposure is not None else 1.0),
        axes=read_axes(path)[0],
        source_path=path,
        channel=channel,
        metadata_source={
            "imagej": {k: v for k, v in ij.items() if k not in ("Labels", "LUTs")},
            "xresolution_um_per_px": tag_xy,
        },
    )
