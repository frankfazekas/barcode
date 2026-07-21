"""Ordering a set of per-timepoint files the way a human numbered them.

In the volumetric modes one file is one timepoint, so the file list *is* the barcode's
vertical axis. ``utils.setup.find_files`` sorts lexicographically, which puts a numbered
series in the order 1, 10, 11, 12, ... 2, 3 -- and because every row still carries its
own filename, the resulting barcode looks entirely plausible while showing the time
course out of order.

Kept out of ``utils/setup.py`` deliberately: changing the sort there would reorder the
rows of the published 2D reference CSVs, which are compared byte-for-byte.
"""
from __future__ import annotations

import os
import re
from typing import List, Sequence


def natural_key(text: str):
    """Split into digit and non-digit runs so numbers compare as numbers."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", text)]


def sort_numerically(paths: Sequence[str]) -> List[str]:
    """Sort paths so embedded numbers order numerically rather than as text.

    Sorts on the directory and the basename separately, so a file's position never
    depends on how deep its folder happens to be nested.
    """
    return sorted(paths, key=lambda p: (natural_key(os.path.dirname(p)),
                                        natural_key(os.path.basename(p))))
