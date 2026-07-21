"""Analysis modes — what BARCODE is being asked to measure, and along which axis.

BARCODE can analyse the same file in three genuinely different ways, and until now the
choice was implicit: the 2D pipeline assumed the third axis was time, and a Z-stack fed
to it was silently analysed as a time series (optical flow between focal planes, reported
in um/s). Naming the modes makes the choice explicit and lets each one emit only the
metrics it can actually support.

The three modes are not arbitrary options; they are two orthogonal properties:

* **spatial_dims** -- is the analysed unit a plane or a volume? This decides area vs
  volume, 2D vs 3D connectivity/anisotropy/correlation, and whether meshing is possible.
* **progression** -- along which axis do things change? This decides whether the Change
  metrics mean anything, and whether flow is a *velocity* (progression is time) or a
  displacement per unit depth (progression is z, which is not motion at all).

Every applicability rule elsewhere in the codebase should be derived from these fields
rather than re-deriving the reasoning, so the rules cannot drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

XYT = "xyt"
XYZ = "xyz"
XYZT = "xyzt"


@dataclass(frozen=True)
class AnalysisMode:
    """One way of analysing a file."""

    key: str
    label: str
    description: str

    spatial_dims: int          # 2 = each analysed unit is a plane, 3 = a volume
    progression: str           # "time" or "depth" -- the axis Change metrics run along
    progression_noun: str      # what to call one step, for log messages
    required_axes: Tuple[str, ...]

    supports_flow: bool
    supports_mesh: bool

    @property
    def is_volumetric(self) -> bool:
        return self.spatial_dims == 3

    @property
    def supports_component_stats(self) -> bool:
        """Whether the per-object size-distribution family can be computed.

        Only where BARCODE owns the labelling: the volumetric branch exposes every
        component's size, whereas the 2D branch computes sizes internally and returns
        only summaries. Adding it for the 2D modes would mean labelling each slice a
        second time, which is a cost worth paying only if the family proves useful.
        """
        return self.spatial_dims == 3

    @property
    def progresses_in_time(self) -> bool:
        """True when flow is a velocity rather than a per-depth displacement."""
        return self.progression == "time"

    def validate_axes(self, axes: Sequence[str], filename: str = "") -> None:
        """Raise unless the file declares every axis this mode needs.

        Deliberately strict: falling back to another mode would reintroduce exactly the
        silent misreading this module exists to prevent.
        """
        present = set(axes)
        missing = [a for a in self.required_axes if a not in present]
        if missing:
            where = f"{filename}: " if filename else ""
            raise ValueError(
                f"{where}mode '{self.key}' needs axis/axes {', '.join(missing)} but the "
                f"file declares axes {''.join(axes)!r}. Choose a mode that matches the "
                f"data, or re-save the file with the axis present."
            )


MODES: Dict[str, AnalysisMode] = {
    XYT: AnalysisMode(
        key=XYT,
        label="xy over time (2D)",
        description=(
            "Planar images analysed as a time series. The original BARCODE behaviour: "
            "2D areas, and optical flow as a true velocity."
        ),
        spatial_dims=2,
        progression="time",
        progression_noun="frame",
        required_axes=("T",),
        supports_flow=True,
        supports_mesh=False,
    ),
    XYZ: AnalysisMode(
        key=XYZ,
        label="xy over z (2D slice-wise)",
        description=(
            "For each timepoint, every z-slice is analysed as a 2D image and depth is "
            "the progression axis. Reports 2D areas; Change metrics describe variation "
            "with depth. Flow is disabled -- displacement between focal planes is not "
            "motion, so a velocity would be meaningless."
        ),
        spatial_dims=2,
        progression="depth",
        progression_noun="slice",
        required_axes=("Z",),
        supports_flow=False,
        supports_mesh=False,
    ),
    XYZT: AnalysisMode(
        key=XYZT,
        label="xyz over time (3D volumetric)",
        description=(
            "Volumes analysed as a time series: true 3D metrics (volumes, "
            "26-connectivity, 3D correlation lengths), surface meshing, and 3D flow. "
            "A single volume is this mode with one timepoint -- Change metrics are then "
            "NaN, everything else is live."
        ),
        spatial_dims=3,
        progression="time",
        progression_noun="timepoint",
        required_axes=("Z",),
        supports_flow=True,
        supports_mesh=True,
    ),
}

MODE_KEYS = tuple(MODES)


def guard_declared_axes(mode: AnalysisMode, filepath: str) -> None:
    """Warn when a file's declared axes contradict the chosen mode.

    Deliberately narrow. Most planar TIFFs declare no meaningful axis at all -- a plain
    multipage movie comes back as ``QYX`` -- so demanding a ``T`` axis for xyt would
    flag ordinary 2D data, including the published reference set. Only ``Z`` without
    ``T`` is called out, because analysing that as a time series is the silent misreading
    modes exist to prevent.

    It *warns* rather than raises. This used to raise, which turned out to reject
    legitimate input: Fiji saves a 2D movie as an ImageJ stack with ``slices=N``, which
    reads back as ``ZYX``, so the declared axes cannot separate "a Z-stack in the wrong
    mode" from "an ordinary 2D movie saved the usual way". A hard error on the
    reference-validated 2D path is the worse failure of the two.

    Anything unreadable or undeclared passes through untouched; this is a guard against
    a known wrong answer, not a general gatekeeper.
    """
    if mode.key != XYT or not filepath.lower().endswith((".tif", ".tiff")):
        return
    try:
        import tifffile

        with tifffile.TiffFile(filepath) as handle:
            axes = handle.series[0].axes
            shape = tuple(int(v) for v in handle.series[0].shape)
    except Exception:
        return

    if "Z" in axes and "T" not in axes:
        import os

        # A WARNING, not an exception. Fiji writes an ordinary saved stack with
        # `slices=N` rather than `frames=N`, and tifffile reports that as ZYX -- so a
        # perfectly normal 2D time series exported from Fiji declares Z and no T, and
        # raising here refused data that analysed fine before the modes existed,
        # potentially including the published reference set. The misreading this guards
        # against is real, but it is not distinguishable from that case by the axes alone,
        # so the honest response is to say so loudly and let the run proceed.
        # The EXTENT along Z, from the shape -- `axes.count("Z")` counts the letter in the
        # axis string, which is 1 for any z-stack, and the message it produced was
        # nonsense for that reason.
        n_slices = shape[axes.index("Z")] if len(shape) == len(axes) else "?"
        print(
            f"WARNING: {os.path.basename(filepath)} declares axes {axes!r} — "
            f"{n_slices} Z slice(s) and no T axis. Mode 'xyt' will analyse those "
            f"slices as timepoints and report optical flow between focal planes as a "
            f"velocity. If this is a Z-stack rather than a time series, choose 'xyz' for "
            f"2D metrics over depth or 'xyzt' for true 3D. (Fiji writes ordinary 2D "
            f"movies this way too, in which case this is expected.)",
            flush=True,
        )


def get_mode(key: str) -> AnalysisMode:
    """Look up a mode, failing with the valid options rather than a KeyError."""
    try:
        return MODES[str(key).strip().lower()]
    except KeyError:
        raise ValueError(
            f"Unknown analysis mode {key!r}. Valid modes: {', '.join(MODE_KEYS)}."
        ) from None
