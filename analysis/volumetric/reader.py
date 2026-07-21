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


def validate_axes_override(override: str, shape: Tuple[int, ...], filename: str = "") -> str:
    """Check a user-supplied axis order against the file's actual shape.

    Acquisition software mislabels hyperstacks — writing a time series into ImageJ's
    ``channels`` field is common, and the file then declares ``ZCYX`` for data that is
    really ``TZYX``. The module's rule is that BARCODE never *guesses* the axis order;
    it does not forbid the user from *stating* it. This is that statement, and it is
    checked hard, because a wrong override silently reinterprets every axis.
    """
    label = f"{filename}: " if filename else ""
    axes = str(override).strip().upper()

    if len(axes) != len(shape):
        raise ValueError(
            f"{label}axis override {axes!r} has {len(axes)} axes but the file's data is "
            f"{len(shape)}-dimensional {shape}. Give one letter per dimension."
        )
    unsupported = set(axes) - _KNOWN_AXES
    if unsupported:
        raise ValueError(
            f"{label}axis override {axes!r} contains {sorted(unsupported)!r}; "
            f"only T, Z, C, Y and X are understood."
        )
    if len(set(axes)) != len(axes):
        raise ValueError(f"{label}axis override {axes!r} repeats an axis.")
    if "Y" not in axes or "X" not in axes:
        raise ValueError(f"{label}axis override {axes!r} must include both Y and X.")
    return axes


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
    # Whether ``exposure_time_s`` came from the file at all, or is the 1.0 fallback.
    # Without it the two are indistinguishable downstream, which is how a file carrying no
    # timing at all could be reported as "1 s from the file".
    timing_from_file: bool = False
    # What the file itself claimed, when that differs from `axes` because the user
    # supplied an override. Kept so provenance records the reinterpretation.
    declared_axes: str = None
    metadata_source: dict = field(default_factory=dict)
    # (start, stop) of the analysed slices within the acquired stack; None = all of it.
    z_range: tuple = None
    # How tall the stack was as acquired. A mask covers the whole acquisition, so
    # validating it against an already-restricted image compares its full depth with a
    # sub-range and rejects a perfectly good mask.
    n_slices_acquired: int = None
    # (start, stop) of the analysed timepoints within the acquired series.
    t_range: tuple = None

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

    def resolve_z_range(self, z_start, z_end, units: str = "acquired",
                        isotropic_step_um: float = None) -> Tuple[int, int]:
        """Convert a z range in any supported unit to acquired-slice indices.

        "Slice 46" means different planes depending on the grid: the acquired stack and
        the isotropic grid a segmentation lives on differ by the anisotropy factor (54
        vs ~249 slices on 0.3/0.065 um data). Stating the unit removes the guesswork.
        """
        units = (units or "acquired").strip().lower()
        if units not in ("acquired", "isotropic", "microns"):
            raise ValueError(
                f"Unknown z_range_units {units!r}; expected 'acquired', 'isotropic' "
                f"or 'microns'."
            )
        if units == "acquired":
            return int(z_start), int(z_end)

        step = (isotropic_step_um or self.xy_step_um) if units == "isotropic" else 1.0
        if not self.z_step_um:
            raise ValueError("Cannot convert a z range without a z step.")

        def to_acquired(value, is_end):
            if value == 0 and is_end:
                return 0                      # 0 always means "to the end"
            depth = float(value) * step       # microns from the bottom (or from the end)
            index = depth / self.z_step_um
            # Negative index units count back from the end, matching Python slicing.
            return int(round(index))

        return to_acquired(z_start, False), to_acquired(z_end, True)

    def resolve_t_range(self, t_start, t_end, units: str = "index",
                        frame_interval_s: float = 0.0) -> Tuple[int, int]:
        """Convert a timepoint range to indices.

        ``index`` counts timepoints; ``seconds`` converts through the interval between
        timepoints, so a range can be stated in the units the experiment was designed in
        rather than in frame numbers that change if the acquisition rate does.

        ``frame_interval_s`` is the configured interval and is what "seconds" means here.
        This used to convert through ``exposure_time_s`` -- the file's ImageJ ``finterval``
        -- and guard with ``if not self.exposure_time_s``, a guard that could never fire
        because ``read_volume`` had already substituted 1.0 for a missing tag. On the
        Jurkat series (60 s between timepoints, ``finterval`` = 1) asking for the first
        600 seconds selected 600 timepoints instead of 10, silently clamped to the whole
        series, with the right number sitting in the config the whole time.
        """
        units = (units or "index").strip().lower()
        if units not in ("index", "seconds"):
            raise ValueError(
                f"Unknown t_range_units {units!r}; expected 'index' or 'seconds'."
            )
        if units == "index":
            return int(t_start), int(t_end)

        interval = float(frame_interval_s or 0.0)
        if interval <= 0 and self.timing_from_file:
            interval = float(self.exposure_time_s or 0.0)
        if interval <= 0:
            raise ValueError(
                "Cannot convert a t range in seconds: no frame interval is known. Set "
                "Frame Interval (--frame-interval) to the seconds between timepoints, or "
                "use t_range_units='index'."
            )

        def to_index(value, is_end):
            if value == 0 and is_end:
                return 0                       # 0 always means "to the end"
            return int(round(float(value) / interval))

        return to_index(t_start, False), to_index(t_end, True)

    def restrict_t(self, t_start: int = 0, t_end: int = 0) -> "VolumeStack":
        """Return a copy limited to a range of timepoints.

        Same conventions as ``restrict_z``: ``t_end`` of 0 means "to the last", negatives
        index from the end, and an empty or reversed range raises rather than yielding a
        zero-timepoint stack that would surface later as an unexplained NaN.
        """
        n_t = self.n_timepoints
        start = t_start + n_t if t_start < 0 else t_start
        stop = n_t if t_end == 0 else (t_end + n_t if t_end < 0 else t_end)
        start, stop = max(start, 0), min(stop, n_t)

        if stop <= start:
            raise ValueError(
                f"{os.path.basename(self.source_path)}: t range [{t_start}, {t_end}) "
                f"selects no timepoints from a {n_t}-timepoint series."
            )
        if (start, stop) == (0, n_t):
            return self

        restricted = replace(self, data=self.data[start:stop])
        restricted.t_range = (start, stop)
        # A grouped series carries its source files; keep them aligned with the data.
        paths = self.metadata_source.get("paths")
        if paths:
            metadata = dict(self.metadata_source)
            metadata["paths"] = list(paths[start:stop])
            frames = metadata.get("frames")
            if frames:
                metadata["frames"] = list(frames[start:stop])
            restricted.metadata_source = metadata
        return restricted

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
        restricted.n_slices_acquired = n_z
        return restricted

    def describe(self) -> str:
        anisotropy = self.z_step_um / self.xy_step_um if self.xy_step_um else float("nan")
        return (
            f"{os.path.basename(self.source_path)}: axes={self.axes} "
            f"(T,Z,Y,X)={self.data.shape} dtype={self.data.dtype} "
            f"z={self.z_step_um:g}um xy={self.xy_step_um:g}um "
            f"anisotropy={anisotropy:.3f}x channel={self.channel}"
            + (f" z[{self.z_range[0]}:{self.z_range[1]}]" if self.z_range else "")
            + (f" t[{self.t_range[0]}:{self.t_range[1]}]" if self.t_range else "")
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


# How many microns one of the unit a file names actually is. ImageJ writes the name into
# `unit`; a plain TIFF names it numerically in ResolutionUnit (1 = none, 2 = inch, 3 = cm).
# Anything not listed here is not a length this module is willing to guess at.
_UNIT_TO_UM = {
    "um": 1.0, "µm": 1.0, "μm": 1.0, "micron": 1.0, "microns": 1.0,
    "micrometer": 1.0, "micrometre": 1.0, "microm": 1.0,
    "nm": 1e-3, "nanometer": 1e-3, "nanometre": 1e-3,
    "mm": 1e3, "millimeter": 1e3, "millimetre": 1e3,
    "cm": 1e4, "centimeter": 1e4, "centimetre": 1e4,
    "m": 1e6, "meter": 1e6, "metre": 1e6,
    "inch": 25400.0, "in": 25400.0, '"': 25400.0,
}

# Units that explicitly mean "this file is NOT spatially calibrated". Believing them is
# the whole point: a pixel-calibrated ImageJ file reports XResolution 1/1, which is
# indistinguishable from a genuine 1.0 um/px calibration unless the name is read.
_UNCALIBRATED_UNITS = {"pixel", "pixels", "px", "none", "", "unit", "units", "a.u."}


def unit_to_um(name) -> Optional[float]:
    """Microns per unit for a unit *name*, or None when it is not a known length.

    Returns None both for an unrecognised name and for one that explicitly means
    uncalibrated ("pixel"), because the caller must treat those the same way: fall back
    and say so, never silently multiply by 1.
    """
    if name is None:
        return None
    text = str(name).strip().lower()
    if text in _UNCALIBRATED_UNITS:
        return None
    return _UNIT_TO_UM.get(text)


def _resolution_unit_name(page) -> Optional[str]:
    """The TIFF ResolutionUnit tag as a unit name, when the file carries one."""
    tag = page.tags.get("ResolutionUnit")
    if tag is None:
        return None
    value = getattr(tag.value, "value", tag.value)
    return {1: "none", 2: "inch", 3: "cm"}.get(int(value))


def _xy_spacing_from_tags(page, unit_name=None) -> Optional[float]:
    """Microns per pixel from XResolution (which is *pixels per unit*).

    ``XResolution`` is a bare number; the unit it counts per lives elsewhere, and this
    used to assume microns unconditionally. That silently mis-scaled every physical
    metric for any file not calibrated in microns -- a nm-calibrated stack read as
    300 um/slice, an inch-calibrated plain TIFF as 0.0139 -- and, worse, it made the
    "no spacing found, assuming 1.0 um" warning unreachable for a *pixel*-calibrated
    ImageJ file, whose XResolution of 1/1 is a perfectly ordinary-looking 1.0.

    ``unit_name`` is ImageJ's `unit` when present, else the TIFF ResolutionUnit. An
    unrecognised or uncalibrated unit returns None, i.e. "no usable spacing here".
    """
    tag = page.tags.get("XResolution")
    if tag is None:
        return None
    px_per_unit = _rational_to_float(tag.value)
    if not px_per_unit:
        return None

    name = unit_name if unit_name is not None else _resolution_unit_name(page)
    if name is None:
        # No unit stated anywhere. Microns is the microscopy convention and the previous
        # behaviour, so it stays the assumption -- but only when nothing contradicts it.
        return 1.0 / px_per_unit
    um_per_unit = unit_to_um(name)
    if um_per_unit is None:
        return None
    return um_per_unit / px_per_unit


TIFF_SUFFIXES = (".tif", ".tiff")
ND2_SUFFIX = ".nd2"
SUPPORTED_SUFFIXES = TIFF_SUFFIXES + (ND2_SUFFIX,)


def require_supported(path: str) -> None:
    """Reject input this module cannot read, with a message that says what to do.

    ``utils.setup.find_files`` also accepts ``.mp4``/``.avi``, so a folder of those is
    happily discovered and then handed to a reader that fails per file with something
    unhelpful about file magic.
    """
    suffix = os.path.splitext(path)[1].lower()
    if suffix in SUPPORTED_SUFFIXES:
        return
    raise ValueError(
        f"The volumetric modes read TIFF and ND2, but got "
        f"'{suffix or 'no extension'}': {os.path.basename(path)}. Convert it in "
        f"ImageJ/FIJI first (File > Save As > Tiff), saving as a hyperstack so the axis "
        f"order and z spacing are carried in the metadata."
    )


# Kept as an alias: this used to be the TIFF-only gate, and external callers (and the
# tests that pin the ND2 rejection) still name it.
def require_tiff(path: str) -> None:
    """Deprecated alias for :func:`require_supported`. ND2 is now read directly."""
    require_supported(path)


def _nd2_frame_interval_s(handle) -> Optional[float]:
    """Seconds between timepoints, from the ND2's own acquisition loop.

    Preferred over the per-frame event timestamps: the loop states the *requested*
    period, which is what "the spacing between timepoints" means, whereas the events are
    wall-clock arrival times that drift and, on a z-series, carry one entry per PLANE
    rather than per timepoint -- so differencing them yields the z dwell, which is
    exactly the confusion ``resolve_frame_interval`` exists to prevent.

    Returns None when the file declares no time loop, which is the honest answer for a
    single-timepoint z-stack.
    """
    try:
        from nd2 import structures as nd2_structures
    except Exception:                                   # pragma: no cover - import guard
        return None

    for loop in getattr(handle, "experiment", None) or []:
        params = getattr(loop, "parameters", None)
        if isinstance(loop, getattr(nd2_structures, "TimeLoop", ())):
            period_ms = getattr(params, "periodMs", None)
            if period_ms:
                return float(period_ms) / 1000.0
        if isinstance(loop, getattr(nd2_structures, "NETimeLoop", ())):
            # A multi-period ("ND" time) acquisition. Only a single uniform period can be
            # represented by one frame interval, so anything else is declined rather than
            # averaged -- an averaged period is wrong for every timepoint individually.
            periods = [p for p in (getattr(params, "periods", None) or [])
                       if getattr(p, "periodMs", None)]
            unique = {round(float(p.periodMs), 6) for p in periods}
            if len(unique) == 1:
                return float(unique.pop()) / 1000.0
            if len(unique) > 1:
                print(
                    f"  {os.path.basename(handle.path)}: the ND2 declares "
                    f"{len(unique)} different time periods, so no single frame interval "
                    f"describes it; set Frame Interval explicitly.",
                    flush=True,
                )
            return None
    return None


def _nd2_z_step_um(handle) -> Optional[float]:
    """Z step from the ND2's z-stack loop, as a fallback for ``voxel_size().z``."""
    try:
        from nd2 import structures as nd2_structures
    except Exception:                                   # pragma: no cover - import guard
        return None

    for loop in getattr(handle, "experiment", None) or []:
        if isinstance(loop, getattr(nd2_structures, "ZStackLoop", ())):
            step = getattr(getattr(loop, "parameters", None), "stepUm", None)
            if step:
                return abs(float(step))
    return None


def _read_nd2(path: str):
    """Load an ND2 as ``(array, axes, z_um, xy_um, interval_s, metadata)``.

    ND2 states its own axis order and voxel size, which is what this module wants and
    what a TIFF so often lacks -- so there is no guessing here either.

    Note this is not the same as the 2D branch's ND2 path in ``utils/reader.py``, which
    *rejects* anything with a Z axis ("Z-stack identified, skipping to next file"). That
    left a volumetric ND2 readable by neither pipeline.
    """
    import nd2

    with nd2.ND2File(path) as handle:
        sizes = dict(handle.sizes)
        axes = "".join(sizes.keys())

        # 'P' (multi-point) and friends are not a data axis this module can collapse:
        # each position is a different field of view and averaging or concatenating them
        # would silently pool unrelated cells.
        unsupported = set(axes) - _KNOWN_AXES
        if unsupported:
            raise ValueError(
                f"{os.path.basename(path)}: ND2 declares axes {axes!r}, which contains "
                f"{sorted(unsupported)!r}. Multi-point and other extra loops are not "
                f"supported -- split the file into one series per position first "
                f"(NIS-Elements: File > Save As, or Fiji's Bio-Formats importer with "
                f"'Split positions')."
            )

        array = np.asarray(handle.asarray())
        voxel = handle.voxel_size()
        xy_um = float(voxel.x) if getattr(voxel, "x", 0) else None
        z_um = float(voxel.z) if getattr(voxel, "z", 0) else None
        if z_um is None:
            z_um = _nd2_z_step_um(handle)
        interval_s = _nd2_frame_interval_s(handle)
        metadata = {
            "nd2": {
                "sizes": sizes,
                "voxel_size_xyz_um": tuple(float(v) for v in voxel),
                "frame_interval_s": interval_s,
            },
        }

    if array.ndim != len(axes):
        raise ValueError(
            f"{os.path.basename(path)}: ND2 declares axes {axes!r} but the array is "
            f"{array.ndim}-dimensional {array.shape}."
        )
    return array, axes, z_um, xy_um, interval_s, metadata


def read_axes(path: str) -> Tuple[str, Tuple[int, ...]]:
    """Return the declared axis order and shape without loading pixel data."""
    require_supported(path)
    if os.path.splitext(path)[1].lower() == ND2_SUFFIX:
        import nd2

        with nd2.ND2File(path) as handle:
            sizes = dict(handle.sizes)
        return "".join(sizes.keys()), tuple(int(v) for v in sizes.values())
    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        return series.axes, tuple(int(v) for v in series.shape)


def read_volume(
    path: str,
    channel: int = 0,
    z_step_um: Optional[float] = None,
    xy_step_um: Optional[float] = None,
    exposure_time_s: Optional[float] = None,
    axes_override: Optional[str] = None,
) -> VolumeStack:
    """Load a volumetric TIFF as ``(T, Z, Y, X)`` for one channel.

    Physical spacing is read from ImageJ metadata when present. Explicit arguments
    always win over file metadata; if neither supplies a value the spacing falls back
    to 1.0 with a printed warning, because silently assuming a voxel size would make
    every physical metric wrong by an unknown factor.

    ``axes_override`` states the true axis order for a file whose header is wrong (or
    undeclared), one letter per data dimension. It replaces the declared order outright,
    so it also rescues the ``IQS`` "undetermined axis" files this module otherwise
    refuses. The distinction the module keeps is between guessing and being told.
    """
    require_supported(path)
    if os.path.splitext(path)[1].lower() == ND2_SUFFIX:
        array, axes, z_declared, tag_xy, file_interval, metadata_source = _read_nd2(path)
        unit_name = "micron"          # nd2 reports voxel size in microns by definition
    else:
        with tifffile.TiffFile(path) as tf:
            series = tf.series[0]
            axes = series.axes
            array = series.asarray()
            ij = tf.imagej_metadata or {}
            unit_name = ij.get("unit")
            tag_xy = _xy_spacing_from_tags(tf.pages[0], unit_name)

        # ImageJ's `spacing` is stated in the same `unit` as the lateral calibration, so a
        # nm- or inch-calibrated stack needs converting rather than reading as microns.
        z_declared = ij.get("spacing")
        if z_declared is not None and unit_name is not None:
            um_per_unit = unit_to_um(unit_name)
            z_declared = None if um_per_unit is None else float(z_declared) * um_per_unit
        file_interval = ij.get("finterval")
        metadata_source = {
            "imagej": {k: v for k, v in ij.items() if k not in ("Labels", "LUTs")},
            "xresolution_um_per_px": tag_xy,
            "unit": unit_name,
        }

    declared = axes
    if axes_override:
        axes = validate_axes_override(axes_override, array.shape, os.path.basename(path))
        if axes != declared:
            print(f"{os.path.basename(path)}: reading axes as {axes} "
                  f"(file declares {declared}).", flush=True)

    unknown = set(axes) & _UNKNOWN_AXES
    if unknown:
        raise ValueError(
            f"{os.path.basename(path)}: file declares axes {axes!r}, which contains "
            f"undetermined axes {sorted(unknown)!r}. BARCODE will not guess whether "
            f"these are Z or T — re-save as an ImageJ hyperstack with explicit axes."
        )
    unsupported = set(axes) - _KNOWN_AXES
    if unsupported:
        raise ValueError(
            f"{os.path.basename(path)}: unsupported axes {sorted(unsupported)!r} "
            f"in {axes!r}; expected some arrangement of T, Z, C, Y, X."
        )
    if "Y" not in axes or "X" not in axes:
        raise ValueError(f"{os.path.basename(path)}: axes {axes!r} lack Y and/or X.")
    if "Z" not in axes:
        raise ValueError(
            f"{os.path.basename(path)}: axes {axes!r} have no Z axis — this is not a "
            f"volumetric stack. Use the standard (2D) BARCODE pipeline for this file."
        )

    # The loop below rewrites `axes` as it pads, so keep what the data really is.
    effective_axes = axes

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
    # Both readers have already reduced their own metadata to microns above.
    z_um = z_step_um if z_step_um is not None else z_declared
    xy_um = xy_step_um if xy_step_um is not None else tag_xy
    # `timing_from_file` must describe the FILE, so it is decided from the file's own
    # value before the caller's override is folded in. Deriving it from the merged value
    # made a caller-supplied exposure_time_s report as coming from the file, which is the
    # one claim the flag exists to make honestly. It is load-bearing for Speed.
    #
    # ND2 is the format where this finally pays off: an ImageJ `finterval` is routinely
    # the z dwell or a leftover 1, which is why `resolve_frame_interval` distrusts it,
    # whereas an ND2's time loop states the real period between timepoints. Both arrive
    # here as `file_interval`; the difference is that an ND2 z-stack with no time loop
    # reports None rather than a misleading number.
    timing_from_file = file_interval is not None
    exposure = exposure_time_s if exposure_time_s is not None else file_interval

    if z_um is None:
        print(
            f"Warning: {os.path.basename(path)} has no usable z spacing"
            f"{f' (unit {unit_name!r})' if unit_name else ''}; assuming 1.0 um.",
            flush=True,
        )
        z_um = 1.0
    if xy_um is None:
        print(
            f"Warning: {os.path.basename(path)} has no usable xy spacing"
            f"{f' (unit {unit_name!r})' if unit_name else ''}; assuming 1.0 um. "
            f"Set Micron to Pixel Ratio (or --xy-step) if that is not the real scale.",
            flush=True,
        )
        xy_um = 1.0

    return VolumeStack(
        data=data,
        z_step_um=float(z_um),
        xy_step_um=float(xy_um),
        exposure_time_s=float(exposure if exposure is not None else 1.0),
        timing_from_file=timing_from_file,
        axes=effective_axes,
        declared_axes=declared,
        source_path=path,
        channel=channel,
        metadata_source=metadata_source,
    )


def apply_t_range(stack: VolumeStack, config) -> VolumeStack:
    """Restrict ``stack`` to the config's timepoint range, in whatever unit it is stated.

    Paired with ``apply_z_range`` so no pipeline can interpret either setting its own way.
    """
    start, end = stack.resolve_t_range(
        getattr(config, "t_start", 0),
        getattr(config, "t_end", 0),
        getattr(config, "t_range_units", "index"),
        getattr(config, "frame_interval_s", 0) or 0.0,
    )
    return stack.restrict_t(start, end)


def apply_z_range(stack: VolumeStack, config) -> VolumeStack:
    """Restrict ``stack`` to the config's z range, in whatever unit it is stated.

    One place for this so xyz, xyzt and the per-slice path cannot interpret the same
    setting differently.
    """
    start, end = stack.resolve_z_range(
        getattr(config, "z_start", 0),
        getattr(config, "z_end", 0),
        getattr(config, "z_range_units", "acquired"),
        getattr(config, "mask_spacing_um", 0) or None,
    )
    return stack.restrict_z(start, end)
