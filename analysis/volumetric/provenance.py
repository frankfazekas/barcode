"""Which data a row's numbers were computed over.

Stream A2 of docs/parallel_work_plan.md.

Flag digit 5 already marks *that* an analysis covered part of the acquired data. This
records *which* part, per row, so a CSV separated from its Settings.yaml still describes
itself — the same failure that made a stale summary unreadable earlier in this work,
where nothing in the file recorded that its correlation lengths predated a bug fix.

It also makes per-file ranges representable. ``z_start``/``z_end`` are global settings,
so today every row of a batch shares one range; once a per-file range exists these
columns are already the place it goes.

Indices are into the **acquired** data, before any isotropic resampling. A mask often
lives on a much finer grid (250 planes at 0.065 um for a 54-slice acquisition at 0.3),
so "slice 46" is ambiguous unless the grid is stated.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from core.results import RangeResults


def _span(explicit: Optional[Tuple[int, int]], length: int) -> Tuple[float, float]:
    """The (start, end) actually analysed, whether or not a range was applied.

    An unrestricted axis reports its full extent rather than NaN: "0 to 54" and "no
    range was set" are the same statement about the data, and a reader should not have
    to know which one produced the row.
    """
    if explicit:
        return float(explicit[0]), float(explicit[1])
    return 0.0, float(length)


def build_range_results(stack, n_timepoints: int = None) -> RangeResults:
    """Record the z and t extents ``stack`` was reduced to.

    ``stack`` is a ``VolumeStack``; its ``z_range``/``t_range`` are set by the restrict
    helpers and are None when the whole axis was kept.
    """
    z_start, z_end = _span(getattr(stack, "z_range", None), stack.n_slices)
    t_start, t_end = _span(
        getattr(stack, "t_range", None),
        stack.n_timepoints if n_timepoints is None else n_timepoints,
    )
    return RangeResults(z_start=z_start, z_end=z_end, t_start=t_start, t_end=t_end)


def was_restricted(stack) -> bool:
    """True when either axis was reduced — what flag digit 5 reports."""
    return bool(getattr(stack, "z_range", None) or getattr(stack, "t_range", None))


def describe_range(results: RangeResults) -> str:
    """One-line human summary, for logs and the harness."""
    def fmt(start: float, end: float) -> str:
        if not np.isfinite(start) or not np.isfinite(end):
            return "?"
        return f"[{start:g}:{end:g}]"

    return f"z{fmt(results.z_start, results.z_end)} t{fmt(results.t_start, results.t_end)}"
