"""Packing topology — contact-number statistics for a labelled volume.

The load-bearing test is the Poisson-Voronoi one: a regular lattice passes even for a
naive implementation, whereas a random tessellation has a known mean (6, by Euler) and a
known hexagonal fraction (~0.30) that a subtly wrong adjacency rule will miss.

Run: python -m pytest tests/test_packing.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from analysis.volumetric.packing import (
    border_labels,
    contact_graph,
    contact_numbers,
    packing_topology,
)
from core import BarcodeConfig


def config(dilation=0, minimum=1, exclude_border=True):
    """Defaults here are deliberately permissive; the real defaults are tested below."""
    cfg = BarcodeConfig().volumetric
    cfg.packing_contact_dilation_vox = dilation
    cfg.packing_min_contact_voxels = minimum
    cfg.packing_exclude_border_objects = exclude_border
    return cfg


def extruded(tile_2d, depth=6, pad=0):
    """Extrude a 2D label tiling into a monolayer, optionally padded with background."""
    volume = np.repeat(tile_2d[None], depth, axis=0)
    if pad:
        volume = np.pad(volume, ((pad, pad), (0, 0), (0, 0)))
    return volume


def voronoi_labels(shape=(120, 120), n_seeds=60, seed=0):
    """A Poisson-Voronoi tessellation, built by nearest-seed assignment.

    Its expected mean contact number is 6 (Euler's formula for any planar tessellation
    where three edges meet at a vertex) and its hexagonal fraction is about 0.30 -- both
    known independently of this implementation, which is what makes it a real check.
    """
    rng = np.random.default_rng(seed)
    seeds = rng.random((n_seeds, 2)) * np.array(shape)
    yy, xx = np.indices(shape)
    points = np.stack([yy.ravel(), xx.ravel()], axis=1)
    distances = ((points[:, None, :] - seeds[None, :, :]) ** 2).sum(axis=2)
    return (distances.argmin(axis=1) + 1).reshape(shape).astype(np.int32)


# ------------------------------------------------------------------ adjacency


def test_face_contact_is_found_and_counted():
    labels = np.zeros((6, 6, 12), np.int32)
    labels[:, :, :6] = 1
    labels[:, :, 6:] = 2
    pairs, counts = contact_graph(labels, dilation_vox=0, min_contact_voxels=1)
    assert pairs.tolist() == [[1, 2]]
    assert counts[0] == 36, "the shared face is 6x6 voxels"


def test_a_corner_touch_is_not_a_contact():
    """Contact means shared surface. 26-connectivity would wrongly join these."""
    labels = np.zeros((10, 10, 10), np.int32)
    labels[0:4, 0:4, 0:4] = 1
    labels[4:8, 4:8, 4:8] = 2      # shares only the corner vertex
    pairs, _ = contact_graph(labels, dilation_vox=0, min_contact_voxels=1)
    assert len(pairs) == 0


def test_an_edge_touch_is_not_a_contact():
    labels = np.zeros((10, 10, 10), np.int32)
    labels[0:4, 0:4, 0:8] = 1
    labels[4:8, 4:8, 0:8] = 2      # shares an edge line, no face
    pairs, _ = contact_graph(labels, dilation_vox=0, min_contact_voxels=1)
    assert len(pairs) == 0


def test_minimum_contact_area_rejects_a_segmentation_nick():
    """A one-voxel touch is noise, not an interface."""
    labels = np.zeros((6, 6, 12), np.int32)
    labels[:, :, :6] = 1
    labels[3, 3, 6] = 2            # a single voxel abutting one face of object 1

    accepted, _ = contact_graph(labels, dilation_vox=0, min_contact_voxels=1)
    rejected, _ = contact_graph(labels, dilation_vox=0, min_contact_voxels=5)
    assert len(accepted) == 1
    assert len(rejected) == 0


def test_dilation_bridges_a_thin_background_gap():
    """Segmentations often leave a membrane voxel between touching cells."""
    labels = np.zeros((6, 6, 13), np.int32)
    labels[:, :, :6] = 1
    labels[:, :, 7:] = 2           # one background plane between them

    without = contact_graph(labels, dilation_vox=0, min_contact_voxels=1)[0]
    with_bridge = contact_graph(labels, dilation_vox=1, min_contact_voxels=1)[0]
    assert len(without) == 0
    assert with_bridge.tolist() == [[1, 2]]


# ------------------------------------------------------------------ border


def test_border_labels_finds_objects_on_every_face():
    labels = np.zeros((8, 8, 8), np.int32)
    labels[0, 3, 3] = 1            # on the low z face
    labels[3:5, 3:5, 3:5] = 2      # interior
    labels[3, 3, 7] = 3            # on the high x face
    assert border_labels(labels) == {1, 3}


def test_interior_objects_keep_neighbours_that_sit_on_the_border():
    """Border objects are excluded from the statistics, not deleted from the graph.

    Removing them would strip real neighbours from the interior objects beside them,
    which distorts the very number being measured.
    """
    tile = np.zeros((30, 30), np.int32)
    for i in range(3):
        for j in range(3):
            tile[i * 10:(i + 1) * 10, j * 10:(j + 1) * 10] = i * 3 + j + 1
    labels = extruded(tile, depth=6, pad=2)

    degrees = contact_numbers(labels, dilation_vox=0, min_contact_voxels=1)
    centre = tile[15, 15]
    assert degrees[centre] == 4, "the centre tile touches all four of its edge-neighbours"

    results, detail = packing_topology(labels, config())
    assert centre in detail.interior_ids
    assert set(detail.border_ids) == set(range(1, 10)) - {centre}


# ------------------------------------------------------------------ lattices


def test_hexagonal_lattice_gives_six_contacts_and_full_hexagonal_fraction():
    """The canonical monolayer: every interior cell has exactly six neighbours."""
    shape = (140, 140)
    rows, cols, spacing = 10, 10, 13
    seeds = []
    for r in range(rows):
        for c in range(cols):
            y = 8 + r * spacing
            x = 8 + c * spacing + (spacing // 2 if r % 2 else 0)
            seeds.append((y, x))
    seeds = np.array(seeds, dtype=float)

    yy, xx = np.indices(shape)
    points = np.stack([yy.ravel(), xx.ravel()], axis=1)
    distances = ((points[:, None, :] - seeds[None, :, :]) ** 2).sum(axis=2)
    tile = (distances.argmin(axis=1) + 1).reshape(shape).astype(np.int32)

    labels = extruded(tile, depth=5, pad=2)
    results, detail = packing_topology(labels, config())

    assert results.contact_number_mean == pytest.approx(6.0, abs=0.05)
    assert results.contact_number_sd == pytest.approx(0.0, abs=0.15)
    assert results.hexagonal_fraction == pytest.approx(1.0, abs=0.05)


def test_poisson_voronoi_matches_the_known_values():
    """A random tessellation: mean 6 by Euler, hexagonal fraction ~0.30.

    This is the test a naive adjacency rule fails while still passing the lattice case.
    """
    tile = voronoi_labels((120, 120), n_seeds=70, seed=3)
    labels = extruded(tile, depth=5, pad=2)
    results, detail = packing_topology(labels, config())

    assert len(detail.interior_ids) > 20, "need a decent interior population"
    assert results.contact_number_mean == pytest.approx(6.0, abs=0.6)
    assert results.hexagonal_fraction == pytest.approx(0.30, abs=0.15)
    assert results.contact_number_sd > 0.5, "a random packing is disordered"


def test_border_inclusion_drags_the_mean_below_six():
    tile = voronoi_labels((120, 120), n_seeds=70, seed=3)
    labels = extruded(tile, depth=5, pad=2)

    interior_only, _ = packing_topology(labels, config(exclude_border=True))
    everything, _ = packing_topology(labels, config(exclude_border=False))
    assert everything.contact_number_mean < interior_only.contact_number_mean
    assert interior_only.contact_number_mean == pytest.approx(6.0, abs=0.6)


# ------------------------------------------------------------------ degenerate


def test_a_single_object_reports_nan_with_a_reason():
    labels = np.zeros((10, 10, 10), np.int32)
    labels[2:8, 2:8, 2:8] = 1
    results, detail = packing_topology(labels, config())
    assert np.isnan(results.contact_number_mean)
    assert "at least two" in detail.reason


def test_all_objects_on_the_border_reports_nan_with_a_reason():
    labels = np.zeros((4, 4, 8), np.int32)
    labels[:, :, :4] = 1
    labels[:, :, 4:] = 2           # both span the array, so both are border objects
    results, detail = packing_topology(labels, config())
    assert np.isnan(results.contact_number_mean)
    assert "border" in detail.reason


def test_an_empty_volume_reports_nan():
    results, detail = packing_topology(np.zeros((8, 8, 8), np.int32), config())
    assert np.isnan(results.hexagonal_fraction)
    assert detail.n_objects == 0


def test_a_confluent_field_is_why_labels_are_required():
    """Connectivity labelling collapses a packed tissue into one object.

    This is the reason the family needs a supplied label volume rather than deriving
    objects itself, and it is worth pinning because the failure is silent: a binarized
    tissue yields one component and a contact number of zero, which reads as a
    measurement rather than a misconfiguration.
    """
    from skimage.measure import label as cc_label

    tile = voronoi_labels((60, 60), n_seeds=20, seed=1)
    labels = extruded(tile, depth=4, pad=1)

    assert len(np.unique(labels)) - 1 == 20, "20 distinct instances"
    collapsed = cc_label(labels > 0, connectivity=3, return_num=True)[1]
    assert collapsed == 1, "connectivity sees a single confluent blob"


# ------------------------------------------------------------------ schema


def test_packing_family_is_opt_in_and_composes():
    from core.results import ChannelResults, PackingResults

    base = ChannelResults.get_headers(just_metrics=False, mode="xyzt")
    with_packing = ChannelResults.get_headers(
        just_metrics=False, mode="xyzt", include_packing=True)
    assert len(with_packing) == len(base) + 3
    assert "Mean Contact Number" in with_packing
    assert "Mean Contact Number" not in base
    assert len(ChannelResults.get_headers(just_metrics=False)) == 28, "2D must not move"

    assert not PackingResults().is_populated()
    assert PackingResults(contact_number_mean=6.0).is_populated()


def test_packing_config_defaults_are_inert():
    cfg = BarcodeConfig().volumetric
    assert cfg.enable_packing_topology is False
    assert cfg.packing_contact_dilation_vox == 1
    assert cfg.packing_min_contact_voxels == 5
    assert cfg.packing_exclude_border_objects is True
