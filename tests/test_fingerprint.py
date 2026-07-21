"""The fingerprint card: a readable summary of ONE analysed volume.

The barcode normalises per column across rows, so a single volume is a single row and
every stripe is a flat colour. The card is what a one-volume run gets read from instead,
and the property that matters is that it shows **only what the data supports** — a panel
drawn from absent data would be a confident-looking lie, and an absent panel where data
exists silently hides a result.

Run: python -m pytest tests/test_fingerprint.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from core.modes import get_mode
from core.results import ChannelResults
from visualization.fingerprint import (
    FLAG_MEANINGS,
    build_fingerprint,
    distributions,
    metric_groups,
)

SPACING = (0.235, 0.195, 0.195)


def volume_and_labels(n_objects=1, shape=(20, 60, 72)):
    rng = np.random.default_rng(0)
    volume = (rng.random(shape) * 200 + 300).astype(np.uint16)
    labels = np.zeros(shape, np.int32)
    if n_objects == 1:
        labels[5:15, 15:45, 20:55] = 1
    else:
        k = 1
        for y in range(6, shape[1] - 12, 14):
            for x in range(6, shape[2] - 12, 14):
                if k > n_objects:
                    break
                labels[5:15, y:y + 10, x:x + 10] = k
                k += 1
    volume[labels > 0] += 400
    return volume, labels


def kinds(volume, labels, detail=None):
    return [kind for kind, _ in distributions(volume, labels, SPACING, detail)]


# ------------------------------------------------------------------ panels appear only when earned


def test_a_single_object_gets_no_population_panels():
    volume, labels = volume_and_labels(1)
    found = kinds(volume, labels)
    assert "depth" in found and "intensity" in found
    assert "sizes" not in found, "one object is not a size distribution"
    assert "contacts" not in found, "one object has no contact graph"


def test_many_objects_gain_a_size_panel():
    volume, labels = volume_and_labels(12)
    found = kinds(volume, labels)
    assert "sizes" in found
    assert "contacts" not in found, "contacts need the packing family, not just labels"


def test_no_mask_means_no_mask_derived_panels():
    volume, _ = volume_and_labels(1)
    found = kinds(volume, None)
    assert found == [] or found == ["depth"]
    assert "intensity" not in found and "sizes" not in found


def test_curvature_panel_needs_a_mesh():
    volume, labels = volume_and_labels(1)
    assert "curvature" not in kinds(volume, labels, detail=None)

    class Curvature:
        k_mean_faces = np.linspace(-0.5, 0.5, 500)

    class Mesh:
        curvature = Curvature()

    class Detail:
        meshes = [Mesh()]
        slice_profile = []
        packing = []

    assert "curvature" in kinds(volume, labels, detail=Detail())


def test_contacts_panel_comes_from_the_packing_detail():
    volume, labels = volume_and_labels(12)

    class Packing:
        object_ids = list(range(1, 13))
        interior_ids = list(range(1, 13))
        contact_numbers = [5, 6, 6, 7, 6, 6, 5, 7, 6, 6, 6, 8]

    class Detail:
        meshes = []
        slice_profile = []
        packing = [Packing()]

    assert "contacts" in kinds(volume, labels, detail=Detail())


def test_the_depth_profile_prefers_the_computed_one():
    """When the slice-profile family ran, use its numbers, not a re-derivation."""
    volume, labels = volume_and_labels(1)

    class Profile:
        areas = [0.1, 0.4, 0.9, 0.4, 0.1]

    class Detail:
        meshes = []
        packing = []
        slice_profile = [Profile()]

    payload = dict(distributions(volume, labels, SPACING, Detail()))["depth"]
    assert np.allclose(payload["area"], Profile.areas)
    assert len(payload["depth_um"]) == 5


# ------------------------------------------------------------------ grouping


def test_groups_use_display_names_not_enum_reprs():
    results = ChannelResults(filepath="x.tif", channel=0)
    groups = metric_groups(results, get_mode("xyzt"))
    names = [name for _, rows in groups for name, _, _ in rows]
    assert "Connectivity" in names
    assert not any(n.startswith("Metrics.") for n in names), "enum repr leaked to the card"


def test_absent_families_leave_no_empty_heading():
    results = ChannelResults(filepath="x.tif", channel=0)
    titles = [title for title, _ in metric_groups(results, get_mode("xyzt"))]
    assert "Packing" not in titles and "In-mask intensity" not in titles
    assert "Structure" in titles


def test_xyz_drops_the_motion_group():
    """Flow is disabled in xyz, so a Motion heading there would be a lie."""
    results = ChannelResults(filepath="x.tif", channel=0)
    titles = [title for title, _ in metric_groups(results, get_mode("xyz"))]
    assert "Motion" not in titles


# ------------------------------------------------------------------ rendering


def test_it_renders_for_all_three_input_shapes(tmp_path):
    results = ChannelResults(filepath="x.tif", channel=0)
    mode = get_mode("xyzt")
    for name, n_objects, use_mask in (("single", 1, True), ("many", 12, True),
                                      ("nomask", 1, False)):
        volume, labels = volume_and_labels(n_objects)
        path = build_fingerprint(
            volume, labels if use_mask else None, SPACING, results, None, mode,
            title=name, figpath=str(tmp_path / f"{name}.png"))
        assert path.endswith(".png")
        assert (tmp_path / f"{name}.png").stat().st_size > 5000


def test_a_run_with_nothing_optional_still_produces_a_card(tmp_path):
    """One bad file must not abort a batch, and an empty run is not an error."""
    results = ChannelResults(filepath="x.tif", channel=0)
    path = build_fingerprint(np.zeros((8, 20, 24), np.uint16), None, SPACING,
                             results, None, get_mode("xyzt"),
                             figpath=str(tmp_path / "empty.png"))
    assert (tmp_path / "empty.png").exists()


def test_flags_are_explained_not_just_printed():
    for digit in "01234567":
        assert digit in FLAG_MEANINGS, f"flag {digit} has no explanation"


# ------------------------------------------------------------------ the helper move


def test_moved_panel_helpers_are_importable_from_visualization():
    """analysis/ and visualization/ must not import from scripts/."""
    from visualization.panels import (  # noqa: F401
        MU, add_scale_bar, apply_style, as_zyx, boundary_rgba, label_projections,
        nice_bar_length, object_count, projections, show_panel, stretch,
    )

    volume = np.zeros((4, 6, 8))
    volume[1:3, 2:4, 3:6] = 1
    views = projections(volume)
    assert views["xy"].shape == (6, 8)
    assert views["xz"].shape == (4, 8)
    assert views["yz"].shape == (4, 6)


def test_the_open_data_script_uses_the_moved_helpers():
    """It must not have kept its own copy, or the two will drift."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "odf_check", "scripts/make_open_data_figures.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["odf_check"] = module
    spec.loader.exec_module(module)

    import visualization.panels as panels
    assert module.projections is panels.projections
    assert module.add_scale_bar is panels.add_scale_bar
    assert module.MU is panels.MU
