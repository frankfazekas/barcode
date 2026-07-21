"""Row axes — what one row of a barcode *is*, and therefore what is being compared.

``core/modes.py`` names what BARCODE measures. This names what it compares, which until
now was implicit in which script you happened to run: ``core/pipeline.py`` emits one row
per file, ``scripts/run_xyz_slice_barcodes.py`` one per z-slice, the time-lapse runner
one per series. Every emitter ends in the same two calls on a list of results, and the
list's meaning was recorded nowhere.

That matters because **the barcode normalises per column across rows**. The rows are the
comparison. Get them wrong and the picture is either empty of information (one row is a
flat stripe) or quietly misleading (two figures normalised over different sets invite a
comparison the colours do not support).

The right axis depends on the data, not on taste:

* a Drosophila embryo is ~840 cells in one field -- the comparison is between **objects**
* a Jurkat nucleus is one object per field -- the only comparison is between **timepoints**

Both are xyzt runs. So the axis is resolved from the data when the user has not chosen,
and the choice is printed and recorded rather than assumed.

**Scope decides the metric set.** Most metrics are field-level by definition -- there is
no per-object connectivity, correlation length, or optical flow -- so object rows carry a
smaller, different column set. That follows the rule the modes already use: a column that
cannot mean anything is omitted, not filled with NaN.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

AUTO = "auto"
FILE = "file"
TIMEPOINT = "timepoint"
SLICE = "slice"
OBJECT = "object"

# What a row can carry.
FIELD_SCOPE = "field"      # every metric the analysis mode produces
OBJECT_SCOPE = "object"    # only metrics defined for a single object


@dataclass(frozen=True)
class RowAxis:
    """One way of splitting an analysis into comparable rows."""

    key: str
    label: str
    description: str

    noun: str                  # what a single row is, for log messages
    scope: str                 # FIELD_SCOPE or OBJECT_SCOPE
    requires_labels: bool      # needs an instance segmentation
    allowed_modes: Tuple[str, ...] = ()   # empty = any mode

    @property
    def is_per_object(self) -> bool:
        return self.scope == OBJECT_SCOPE

    def validate(self, mode, *, has_labels: bool, n_objects: int = 0) -> None:
        """Raise unless this axis is actually available for the data at hand.

        Explicit and impossible is an error, not a cue to fall back. Silently choosing a
        different axis would change what the figure compares without saying so, which is
        the whole failure this module exists to prevent.
        """
        if self.requires_labels and not has_labels:
            raise ValueError(
                f"Row axis '{self.key}' needs an instance segmentation: one row is one "
                f"{self.noun}, and without labels there are no objects to be rows. "
                f"Supply a mask (segmentation_enabled) or choose another row axis."
            )
        if self.requires_labels and has_labels and n_objects < 2:
            raise ValueError(
                f"Row axis '{self.key}' needs more than one object; the segmentation "
                f"resolved {n_objects}. A single object is one row, which a barcode "
                f"cannot normalise -- use 'timepoint' or 'file'."
            )
        if self.allowed_modes and mode is not None and mode.key not in self.allowed_modes:
            raise ValueError(
                f"Row axis '{self.key}' is only available in mode(s) "
                f"{', '.join(self.allowed_modes)}, not '{mode.key}'."
            )


ROW_AXES: Dict[str, RowAxis] = {
    FILE: RowAxis(
        key=FILE,
        label="one row per file",
        description=(
            "Each input file (field of view) is a row. BARCODE's original behaviour and "
            "the right choice when the comparison is between acquisitions."
        ),
        noun="file",
        scope=FIELD_SCOPE,
        requires_labels=False,
    ),
    TIMEPOINT: RowAxis(
        key=TIMEPOINT,
        label="one row per timepoint",
        description=(
            "Each timepoint is a row, so a column reads down the page as a time course. "
            "The only comparison available when a field holds a single object."
        ),
        noun="timepoint",
        scope=FIELD_SCOPE,
        requires_labels=False,
    ),
    SLICE: RowAxis(
        key=SLICE,
        label="one row per z-slice",
        description=(
            "Each z-slice is a row, so a column reads as a depth profile within one "
            "timepoint. Planar metrics only, which is what mode xyz produces."
        ),
        noun="z-slice",
        scope=FIELD_SCOPE,
        requires_labels=False,
        allowed_modes=("xyz",),
    ),
    OBJECT: RowAxis(
        key=OBJECT,
        label="one row per object",
        description=(
            "Each segmented object is a row, pooled across fields so one colour scale "
            "covers every object. Carries only metrics defined for a single object -- "
            "there is no per-object connectivity, correlation length or flow."
        ),
        noun="object",
        scope=OBJECT_SCOPE,
        requires_labels=True,
    ),
}

ROW_AXIS_KEYS = (AUTO,) + tuple(ROW_AXES)


def get_row_axis(key: str) -> RowAxis:
    """Look up a row axis, failing with the valid options rather than a KeyError."""
    try:
        return ROW_AXES[str(key).strip().lower()]
    except KeyError:
        raise ValueError(
            f"Unknown row axis {key!r}. Valid: {', '.join(ROW_AXIS_KEYS)}."
        ) from None


def resolve_row_axis(
    requested: str,
    mode,
    *,
    has_labels: bool = False,
    n_objects: int = 0,
    n_timepoints: int = 1,
) -> RowAxis:
    """The row axis for this run, inferring one when the user has not chosen.

    The inference is deliberately conservative and, importantly, **never reaches
    'object' without a segmentation** -- so a run with no mask resolves to 'file' exactly
    as BARCODE has always behaved, and the published 2D reference outputs are unaffected.

    Order: many objects beats many timepoints, because a field of cells is almost always
    asking a per-cell question; a single object over time is asking a temporal one.
    """
    requested = (requested or AUTO).strip().lower()

    if requested != AUTO:
        axis = get_row_axis(requested)
        axis.validate(mode, has_labels=has_labels, n_objects=n_objects)
        return axis

    if has_labels and n_objects > 1:
        return ROW_AXES[OBJECT]
    if n_timepoints > 1:
        return ROW_AXES[TIMEPOINT]
    return ROW_AXES[FILE]


def describe_scope(axis: RowAxis, n_rows: int, n_sources: int) -> str:
    """One line naming what the colours were normalised over.

    The barcode's colour scale is meaningless without this, and two figures built over
    different sets are not comparable -- so it belongs on the figure and in the settings,
    not in someone's memory.
    """
    source = "field" if n_sources == 1 else "fields"
    return (f"normalised across {n_rows} {axis.noun}"
            f"{'' if n_rows == 1 else 's'} from {n_sources} {source}")
