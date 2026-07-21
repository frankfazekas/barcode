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

    Returns ``(groups, unmatched)``. Groups are sorted by directory then series name and
    their frames ordered numerically, so ``Cell1_2`` precedes ``Cell1_10`` (plain
    lexicographic sorting, which ``utils.setup.find_files`` uses, would not).

    A series is identified by its directory *and* the matched name. Files in different
    folders are never one series, however alike their names: ``find_files`` walks
    recursively, and the same numbering convention in two condition folders is the normal
    case rather than an exotic one.
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
        # Keyed by directory as well as by the matched series name. `find_files` walks
        # recursively, so a run pointed at a parent folder sees every subfolder's files;
        # keying on the basename alone merged them. Two conditions each holding
        # Cell1_1..15.tif collapsed into one bucket and raised "duplicate frame numbers"
        # -- outside the per-series try in run.py, so it aborted the whole batch -- and if
        # their frame numbers happened not to overlap (1-15 and 16-30) they merged
        # silently into a single 30-timepoint "series" spanning two experiments.
        key = (os.path.dirname(os.path.abspath(path)), match.group("series"))
        buckets.setdefault(key, []).append((frame, path))

    groups = []
    for key in sorted(buckets):
        directory, series = key
        entries = sorted(buckets[key])
        frames = [f for f, _ in entries]
        if len(set(frames)) != len(frames):
            duplicates = sorted({f for f in frames if frames.count(f) > 1})
            raise ValueError(
                f"Series {series!r} in {directory} has duplicate frame numbers "
                f"{duplicates}; the grouping regex is matching more files than intended."
            )
        groups.append(SeriesGroup(series=series, paths=[p for _, p in entries], frames=frames))

    return groups, unmatched


def read_series(
    group: SeriesGroup,
    channel: int = 0,
    z_step_um: Optional[float] = None,
    xy_step_um: Optional[float] = None,
    axes_override: Optional[str] = None,
) -> VolumeStack:
    """Read every file in ``group`` and stack them along T.

    All timepoints must share a shape and voxel spacing; a mismatch means the files do
    not belong to one series, so it raises rather than silently padding or cropping.

    ``axes_override`` is passed through to every file. It used to be missing here alone,
    so a grouped series whose header names the wrong axes -- acquisition software writing
    a time series into ImageJ's "channels" field is the common case -- could not be
    rescued in the one mode built for per-timepoint files, though every other entry point
    honoured the setting.
    """
    if not group.paths:
        raise ValueError(f"Series {group.series!r} has no files.")

    volumes, reference = [], None
    for path in group.paths:
        stack = read_volume(path, channel=channel, z_step_um=z_step_um,
                            xy_step_um=xy_step_um, axes_override=axes_override)
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
        # The series is (T, Z, Y, X) however each file was stored, so say so. Inheriting
        # the per-file "ZYX" made describe() print `axes=ZYX (T,Z,Y,X)=(15,54,312,303)`
        # and recorded a single-volume axis order as the provenance of a constructed
        # series. `declared_axes` keeps what the files themselves claimed.
        axes="TZYX",
        declared_axes=reference.axes,
        # Inherited from file 0, where it describes that file's z acquisition rather than
        # the spacing between timepoints. `timing_from_file` stays False so nothing
        # downstream mistakes it for a real interval — set Frame Interval for that.
        timing_from_file=False,
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
