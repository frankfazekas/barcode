"""Packing topology: how objects are arranged relative to each other.

BARCODE describes objects individually — volume, sphericity, curvature, the spread of
their sizes — and describes their spacing with a single scalar nearest-neighbour
distance. Nothing described *who touches whom*.

For a space-filling monolayer that is the whole question. Sizes and separations are
near-uniform; what changes during morphogenesis is the neighbour-number distribution,
which is the canonical epithelial readout (hexagonal fraction, packing disorder).

**Needs an integer label volume.** In a confluent field every cell touches its
neighbours, so deriving objects by connectivity collapses the entire tissue into one
component and there is no graph to build. See ``object_partition`` in
``core/config.py`` and the mask loader in ``segmentation.py``.

Two decisions that shape every number here:

* **Face adjacency, not 26-connectivity.** Contact means shared surface. Two cells
  meeting along an edge or at a corner are not neighbours, and 26-connectivity would
  call them one. Adjacency is built from 6-connected face contacts only, then filtered
  by a minimum shared-face area so a one-voxel segmentation nick does not invent an edge.
* **Border objects are excluded from the statistics but kept in the graph.** An object
  touching the edge of the array has an artificially low contact number and would drag
  the mean down. Deleting them outright is worse: it strips real neighbours from the
  interior objects beside them. So every degree is computed from the full graph and only
  the *reported statistics* are restricted to interior objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class VolumetricPackingDetail:
    """Per-object contact numbers, which the three scalars summarise."""

    object_ids: List[int] = field(default_factory=list)
    contact_numbers: List[int] = field(default_factory=list)
    interior_ids: List[int] = field(default_factory=list)
    border_ids: List[int] = field(default_factory=list)
    n_objects: int = 0
    n_edges: int = 0
    reason: str = ""          # why the scalars are NaN, when they are

    def describe(self) -> str:
        if self.reason:
            return f"packing: not computed ({self.reason})"
        return (f"packing: {self.n_objects} objects, {self.n_edges} contacts, "
                f"{len(self.interior_ids)} interior")


def contact_graph(
    labels: np.ndarray,
    dilation_vox: int = 1,
    min_contact_voxels: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pairs of labels sharing a face, and the number of voxel faces each pair shares.

    Compares ``labels[:-1]`` against ``labels[1:]`` along each axis in turn, which finds
    every 6-connected contact in three passes rather than per-object neighbourhood
    queries. ``np.unique(..., return_counts=True)`` then gives each pair and its shared
    face area together.

    ``dilation_vox`` grows the labels first so objects separated by a thin background
    gap — a segmentation that left a one-voxel membrane between cells — still register as
    neighbours. ``min_contact_voxels`` then discards pairs whose shared area is too small
    to be a real interface.

    **What dilation does to the face-adjacency rule.** The module docstring's promise that
    edge- and corner-only touches are not neighbours holds only at ``dilation_vox=0``.
    With the shipped default of 1, two labels meeting along an edge each grow one voxel
    into the shared diagonal and afterwards share a real 6-connected face running the
    whole length of that edge — ~30 voxels for a 30-voxel-tall cell, far above the default
    ``min_contact_voxels=5``, which cannot filter it because it measures the DILATED
    interface. That is defensible when the dilation is there to bridge a segmented
    membrane, and wrong when it is not, so it is stated here rather than left for a reader
    to infer from a docstring that says the opposite. Set ``dilation_vox=0`` for strict
    face adjacency; the tests cover that setting.
    """
    labels = np.asarray(labels)
    if dilation_vox and dilation_vox > 0:
        from skimage.segmentation import expand_labels

        # NOTE: `distance` is in VOXELS, and the shared-face counts below weight a
        # z-normal face (area dy*dx) the same as an xy-normal one (dz*dx). Both are only
        # meaningful on an isotropic grid: at the Jurkat anisotropy of 4.6x one voxel of
        # dilation bridges 0.3 um in z and 0.065 um in xy, and the same physical contact
        # passes or fails `min_contact_voxels` depending on its orientation alone. The
        # caller is warned in `packing_topology` rather than silently rescaled here,
        # because there is no single voxel distance that means one physical distance on
        # an anisotropic grid.
        labels = expand_labels(labels, distance=dilation_vox)

    pairs = []
    for axis in range(labels.ndim):
        lower = np.moveaxis(labels, axis, 0)[:-1]
        upper = np.moveaxis(labels, axis, 0)[1:]
        touching = (lower > 0) & (upper > 0) & (lower != upper)
        if touching.any():
            pairs.append(np.stack([lower[touching], upper[touching]], axis=1))

    if not pairs:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64)

    stacked = np.concatenate(pairs).astype(np.int64)
    stacked.sort(axis=1)          # (a, b) and (b, a) are the same interface
    unique_pairs, counts = np.unique(stacked, axis=0, return_counts=True)

    keep = counts >= max(int(min_contact_voxels), 1)
    return unique_pairs[keep], counts[keep]


def border_labels(labels: np.ndarray, mode: str = "xy") -> set:
    """Labels appearing on the faces of the array.

    ``mode="xy"`` (the default) ignores the two z faces, matching
    ``mesh_field.touches_border``. A packing is a space-filling monolayer or epithelium,
    which is exactly the geometry whose segmentation spans the full acquired depth: with
    all six faces every object was a border object, so ``packing_topology`` returned NaN
    for every metric with the reason "every object touches the array border" -- which
    reads like a data problem rather than the axis-handling one it was. An object cut by
    the frame edge in xy is genuinely incomplete and still worth dropping.

    ``mode="all"`` restores the six-face behaviour for a volume where z is a real
    boundary; ``mode="none"`` excludes nothing.
    """
    if mode not in ("xy", "all", "none"):
        raise ValueError(f"border mode must be 'xy', 'all' or 'none', got {mode!r}")
    labels = np.asarray(labels)
    if mode == "none":
        return set()

    axes = (1, 2) if mode == "xy" else tuple(range(labels.ndim))
    found = set()
    for axis in axes:
        moved = np.moveaxis(labels, axis, 0)
        found.update(np.unique(moved[0]).tolist())
        found.update(np.unique(moved[-1]).tolist())
    found.discard(0)
    return found


def contact_numbers(
    labels: np.ndarray,
    dilation_vox: int = 1,
    min_contact_voxels: int = 5,
) -> Dict[int, int]:
    """Number of distinct objects each object touches, for every object present."""
    labels = np.asarray(labels)
    present = [int(v) for v in np.unique(labels) if v != 0]
    degrees = {label: 0 for label in present}

    pairs, _ = contact_graph(labels, dilation_vox, min_contact_voxels)
    for a, b in pairs:
        degrees[int(a)] = degrees.get(int(a), 0) + 1
        degrees[int(b)] = degrees.get(int(b), 0) + 1
    return degrees


def packing_topology(
    labels: np.ndarray, config, spacing_zyx_um=None
) -> Tuple["PackingResults", VolumetricPackingDetail]:
    """Contact-number statistics for a labelled volume.

    Returns NaN with a stated reason rather than a misleading number when the topology is
    undefined: fewer than two objects, or every object touching the array border.

    ``spacing_zyx_um`` is only used to emit the anisotropy warning ``contact_graph``
    promises. That warning did not exist: ``contact_graph`` said "the caller is warned in
    ``packing_topology``" and this function took no spacing at all, so a reader of that
    comment believed a safeguard was in place that was never written.
    """
    from core.results import PackingResults

    labels = np.asarray(labels)
    dilation = getattr(config, "packing_contact_dilation_vox", 1)
    minimum = getattr(config, "packing_min_contact_voxels", 5)
    exclude_border = getattr(config, "packing_exclude_border_objects", True)
    border_mode = getattr(config, "packing_border_mode", "xy")

    if spacing_zyx_um is not None:
        spacing = np.asarray(spacing_zyx_um, dtype=np.float64)
        if spacing.size == 3 and spacing.min() > 0:
            anisotropy = float(spacing.max() / spacing.min())
            if anisotropy > 1.01:
                print(
                    f"  packing: the labels are on a {anisotropy:.1f}x anisotropic grid. "
                    f"Contact is counted in VOXEL faces and bridged by a voxel dilation, "
                    f"so the same physical interface passes or fails "
                    f"min_contact_voxels={minimum} depending on its orientation, and "
                    f"contact number and hexagonal fraction are grid-dependent. Enable "
                    f"Resample to Isotropic Voxels for comparable numbers.",
                    flush=True,
                )

    degrees = contact_numbers(labels, dilation, minimum)
    detail = VolumetricPackingDetail(
        object_ids=sorted(degrees),
        contact_numbers=[degrees[k] for k in sorted(degrees)],
        n_objects=len(degrees),
        n_edges=int(sum(degrees.values()) // 2),
    )

    if len(degrees) < 2:
        detail.reason = f"{len(degrees)} object(s): a packing needs at least two"
        return PackingResults(), detail

    on_border = border_labels(labels, border_mode) if exclude_border else set()
    detail.border_ids = sorted(on_border)
    detail.interior_ids = sorted(set(degrees) - on_border)

    if not detail.interior_ids:
        detail.reason = (
            f"every object reaches the {border_mode} edge of the field, so none is a "
            f"complete cell; set packing_border_mode='none' to report them anyway"
        )
        return PackingResults(), detail

    # Degrees come from the full graph; only the reported set is restricted, so interior
    # objects keep the neighbours they have on the border side.
    interior = np.array([degrees[label] for label in detail.interior_ids], dtype=np.float64)
    return (
        PackingResults(
            contact_number_mean=float(interior.mean()),
            contact_number_sd=float(interior.std()),
            hexagonal_fraction=float(np.mean(interior == 6)),
        ),
        detail,
    )


def summarise_packing(details: Sequence[VolumetricPackingDetail],
                      results: Sequence["PackingResults"]) -> "PackingResults":
    """Average the per-timepoint scalars, matching how every other family is reduced."""
    from core.results import PackingResults

    def mean_of(attribute: str) -> float:
        values = np.array([getattr(r, attribute) for r in results], dtype=np.float64)
        finite = values[np.isfinite(values)]
        return float(finite.mean()) if finite.size else np.nan

    return PackingResults(
        contact_number_mean=mean_of("contact_number_mean"),
        contact_number_sd=mean_of("contact_number_sd"),
        hexagonal_fraction=mean_of("hexagonal_fraction"),
    )
