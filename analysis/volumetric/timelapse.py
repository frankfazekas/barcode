"""Assemble a time series from per-timepoint volume files.

Volumetric time-lapses are frequently exported one file per timepoint
(``Cell1_1.tif`` ... ``Cell1_15.tif``) rather than as a single TZYX hyperstack.
Analysed individually those give 15 independent rows with every change metric NaN,
which throws away the dynamics BARCODE exists to measure. This module groups such
files back into one ``(T, Z, Y, X)`` series.

Grouping is driven by a regex with two named groups:

* ``series`` -- files sharing this value belong to the same time-lapse
* ``frame``  -- ordering within the series (compared numerically)

The default handles the ``Cell{n}_{frame}`` convention. Files that do not match the
regex are left alone and reported, never silently dropped.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from analysis.volumetric.reader import VolumeStack, read_volume

DEFAULT_TIMELAPSE_REGEX = r"^(?P<series>.+?)_(?P<frame>\d+)$"


@dataclass
class SeriesGroup:
    """One time-lapse: an ordered set of single-volume files."""

    series: str
    paths: List[str] = field(default_factory=list)
    frames: List[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.paths)

    def describe(self) -> str:
        return (
            f"{self.series}: {len(self.paths)} timepoints "
            f"(frames {self.frames[0]}..{self.frames[-1]})"
        )


def group_timelapse(
    paths: Sequence[str], regex: str = DEFAULT_TIMELAPSE_REGEX
) -> Tuple[List[SeriesGroup], List[str]]:
    """Group ``paths`` into ordered series.

    Returns ``(groups, unmatched)``. Groups are sorted by series name and their frames
    ordered numerically, so ``Cell1_2`` precedes ``Cell1_10`` (plain lexicographic
    sorting, which ``utils.setup.find_files`` uses, would not).
    """
    pattern = re.compile(regex)
    buckets: Dict[str, List[Tuple[int, str]]] = {}
    unmatched: List[str] = []

    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        match = pattern.search(stem)
        if match is None or "series" not in match.groupdict() or "frame" not in match.groupdict():
            unmatched.append(path)
            continue
        try:
            frame = int(match.group("frame"))
        except (TypeError, ValueError):
            unmatched.append(path)
            continue
        buckets.setdefault(match.group("series"), []).append((frame, path))

    groups = []
    for series in sorted(buckets):
        entries = sorted(buckets[series])
        frames = [f for f, _ in entries]
        if len(set(frames)) != len(frames):
            duplicates = sorted({f for f in frames if frames.count(f) > 1})
            raise ValueError(
                f"Series {series!r} has duplicate frame numbers {duplicates}; "
                f"the grouping regex is matching more files than intended."
            )
        groups.append(SeriesGroup(series=series, paths=[p for _, p in entries], frames=frames))

    return groups, unmatched


def read_series(
    group: SeriesGroup,
    channel: int = 0,
    z_step_um: Optional[float] = None,
    xy_step_um: Optional[float] = None,
) -> VolumeStack:
    """Read every file in ``group`` and stack them along T.

    All timepoints must share a shape and voxel spacing; a mismatch means the files do
    not belong to one series, so it raises rather than silently padding or cropping.
    """
    if not group.paths:
        raise ValueError(f"Series {group.series!r} has no files.")

    volumes, reference = [], None
    for path in group.paths:
        stack = read_volume(path, channel=channel, z_step_um=z_step_um, xy_step_um=xy_step_um)
        if stack.n_timepoints != 1:
            raise ValueError(
                f"{os.path.basename(path)} already contains {stack.n_timepoints} "
                f"timepoints; per-file grouping expects one volume per file."
            )
        if reference is None:
            reference = stack
        else:
            if stack.data.shape[1:] != reference.data.shape[1:]:
                raise ValueError(
                    f"{os.path.basename(path)} has shape {stack.data.shape[1:]} but "
                    f"{os.path.basename(reference.source_path)} has "
                    f"{reference.data.shape[1:]}; these are not one time series."
                )
            for name, a, b in (
                ("z", stack.z_step_um, reference.z_step_um),
                ("xy", stack.xy_step_um, reference.xy_step_um),
            ):
                if not np.isclose(a, b, rtol=1e-6):
                    raise ValueError(
                        f"{os.path.basename(path)} has {name} spacing {a} but "
                        f"{os.path.basename(reference.source_path)} has {b}."
                    )
        volumes.append(stack.data[0])

    return VolumeStack(
        data=np.stack(volumes),
        z_step_um=reference.z_step_um,
        xy_step_um=reference.xy_step_um,
        exposure_time_s=reference.exposure_time_s,
        axes=reference.axes,
        # The series is identified by its first file, so per-file outputs and the CSV
        # row point at something that exists on disk.
        source_path=group.paths[0],
        channel=channel,
        metadata_source={
            "series": group.series,
            "frames": list(group.frames),
            "paths": list(group.paths),
        },
    )
