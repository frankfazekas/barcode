"""The optional-family contract that the parallel work streams build against.

Phase 0 of docs/parallel_work_plan.md: every new column, config field and call site
exists as an inert stub, so streams A and B can fill in behind it without either editing
the schema layer. These tests pin that contract.

The properties worth pinning are that families compose in a fixed order, that a family
with no data never reaches the output, and above all that the 2D schema has not moved --
the published reference set is compared against it.

Run: python -m pytest tests/test_family_registry.py -v
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from core import BarcodeConfig, Metrics, Units
from core.modes import MODES, XYT, XYZ, XYZT
from core.results import (
    OPTIONAL_FAMILIES,
    ChannelResults,
    IntensityMagnitudeResults,
    RangeResults,
    _resolve,
)


# --------------------------------------------------------------- the registry


def test_registry_is_well_formed():
    switches = [f.switch for f in OPTIONAL_FAMILIES]
    attributes = [f.attribute for f in OPTIONAL_FAMILIES]
    assert len(set(switches)) == len(switches), "switch names must be unique"
    assert len(set(attributes)) == len(attributes), "attributes must be unique"

    row = ChannelResults(filepath="x.tif", channel=0)
    for family in OPTIONAL_FAMILIES:
        assert hasattr(row, family.attribute), f"{family.attribute} missing on ChannelResults"
        assert callable(family.supported)
        # every family exposes the same interface, so the registry can drive them all
        assert family.results_cls.get_metrics(MODES[XYZT])
        assert len(family.results_cls.get_metrics(MODES[XYZT])) == \
            len(family.results_cls.get_units(MODES[XYZT]))


def test_resolve_defaults_to_what_the_mode_supports():
    _, _, enabled = _resolve(XYZT)
    assert enabled["include_mesh"] is True, "xyzt supports meshing"
    _, _, enabled = _resolve(XYZ)
    assert enabled["include_mesh"] is False, "a 2D mode has no surface to mesh"


def test_resolve_switches_override_the_default():
    _, _, enabled = _resolve(XYZT, include_mesh=False)
    assert enabled["include_mesh"] is False
    _, _, enabled = _resolve(XYZ, include_ranges=True)
    assert enabled["include_ranges"] is True


def test_flow_follows_the_mode_not_a_switch():
    _, with_flow, _ = _resolve(XYT)
    assert with_flow is True
    _, with_flow, _ = _resolve(XYZ)
    assert with_flow is False, "xyz has no velocity"


# --------------------------------------------------------------- composition


def test_every_family_combination_keeps_headers_and_data_aligned():
    """The check that catches a family shifting the ones after it."""
    row = ChannelResults(filepath="x.tif", channel=0)
    for mode in MODES:
        for combo in itertools.product((False, True), repeat=len(OPTIONAL_FAMILIES)):
            switches = {f.switch: on for f, on in zip(OPTIONAL_FAMILIES, combo)}
            headers = ChannelResults.get_headers(just_metrics=False, mode=mode, **switches)
            units = ChannelResults.get_units(just_metrics=False, mode=mode, **switches)
            data = row.get_data(just_metrics=False, mode=mode, **switches)
            assert len(headers) == len(units) == len(data), (mode, switches)
            assert len(set(headers)) == len(headers), f"duplicate header in {mode} {switches}"


def test_families_append_in_registry_order():
    base = ChannelResults.get_headers(just_metrics=True, mode=XYZT, **{
        f.switch: False for f in OPTIONAL_FAMILIES})
    full = ChannelResults.get_headers(just_metrics=True, mode=XYZT, **{
        f.switch: True for f in OPTIONAL_FAMILIES})
    assert full[:len(base)] == base, "enabling a family must not disturb the base columns"

    tail, cursor = full[len(base):], 0
    for family in OPTIONAL_FAMILIES:
        expected = [m.value for m in family.results_cls.get_metrics(MODES[XYZT])]
        assert tail[cursor:cursor + len(expected)] == expected, family.switch
        cursor += len(expected)


@pytest.mark.parametrize("switch,width", [
    ("include_intensity_magnitude", 4),
    ("include_ranges", 4),
])
def test_new_families_add_the_expected_width(switch, width):
    base = len(ChannelResults.get_headers(just_metrics=False, mode=XYZT))
    with_family = len(ChannelResults.get_headers(
        just_metrics=False, mode=XYZT, **{switch: True}))
    assert with_family == base + width


# --------------------------------------------------------------- 2D is frozen


def test_the_2d_schema_has_not_moved():
    """The published reference set is compared against these exact 28 columns."""
    assert len(ChannelResults.get_headers(just_metrics=False)) == 28
    assert len(ChannelResults.get_headers(just_metrics=False, mode=XYT)) == 28
    assert ChannelResults.get_headers(just_metrics=False) == \
        ChannelResults.get_headers(just_metrics=False, mode=XYT)


def test_new_families_are_off_for_every_mode_by_default():
    """Phase 0 is inert: nothing populates these, so nothing may emit them."""
    for mode in MODES:
        headers = ChannelResults.get_headers(just_metrics=False, mode=mode)
        for name in ("Total Intensity", "Mean Intensity", "Intensity SD",
                     "Z Range Start", "T Range End"):
            assert name not in headers, f"{name} leaked into {mode} by default"


# --------------------------------------------------------------- stub state


def test_the_new_families_are_stubs_returning_nan():
    for cls in (IntensityMagnitudeResults, RangeResults):
        values = cls().get_data()
        assert all(np.isnan(v) for v in values), f"{cls.__name__} should start empty"
        assert not cls().is_populated()


def test_a_populated_family_is_detected():
    magnitude = IntensityMagnitudeResults(total=1234.0)
    assert magnitude.is_populated()
    assert RangeResults(z_start=12, z_end=46).is_populated()


def test_density_is_named_and_united_per_mode():
    """Its unit genuinely differs, so it follows the Area/Volume precedent."""
    volumetric = IntensityMagnitudeResults.get_headers(mode=MODES[XYZT])
    planar = IntensityMagnitudeResults.get_headers(mode=MODES[XYT])
    assert "Intensity Density (per volume)" in volumetric
    assert "Intensity Density (per area)" in planar

    assert Units.INTENSITY_PER_VOLUME in IntensityMagnitudeResults.get_units(MODES[XYZT])
    assert Units.INTENSITY_PER_AREA in IntensityMagnitudeResults.get_units(MODES[XYT])


def test_every_new_unit_is_classified_by_get_data_limits():
    """get_data_limits raises on an unrecognised unit, so a missing one is a crash."""
    from core.metrics import get_data_limits

    for mode in MODES:
        metrics = ChannelResults.get_metrics(just_metrics=True, mode=mode, **{
            f.switch: True for f in OPTIONAL_FAMILIES})
        units = ChannelResults.get_units(just_metrics=True, mode=mode, **{
            f.switch: True for f in OPTIONAL_FAMILIES})
        data = np.ones((2, len(metrics)))
        limits = get_data_limits(data, metrics, units)   # must not raise
        assert len(limits) == len(metrics)


# --------------------------------------------------------------- config


def test_phase0_config_fields_exist_with_inert_defaults():
    config = BarcodeConfig().volumetric
    assert (config.t_start, config.t_end, config.t_range_units) == (0, 0, "index")
    assert config.segmentation_label_mode == "binary"
    assert config.segmentation_secondary_root == ""
    assert config.mesh_aggregation == "largest"
    assert config.enable_intensity_magnitude is False
    assert config.record_range_columns is False


def test_phase0_config_round_trips_through_yaml(tmp_path):
    from core import BarcodeConfig as Config

    config = Config()
    config.volumetric.t_start, config.volumetric.t_end = 2, 9
    config.volumetric.t_range_units = "seconds"
    config.volumetric.mesh_aggregation = "mean"
    config.volumetric.enable_intensity_magnitude = True
    path = str(tmp_path / "settings.yaml")
    config.save_to_yaml(path)

    back = Config.load_from_yaml(path).volumetric
    assert (back.t_start, back.t_end, back.t_range_units) == (2, 9, "seconds")
    assert back.mesh_aggregation == "mean"
    assert back.enable_intensity_magnitude is True


# --------------------------------------------------------------- round-trip


def test_csv_round_trip_with_every_family_populated(tmp_path):
    """The reader must recognise and rebuild any family combination.

    A header list the reader does not know used to make it drop every row silently,
    which is the failure this whole registry exists to prevent.
    """
    from utils.reader import read_csv_to_channel_results
    from utils.writer import results_to_csv

    row = ChannelResults(filepath="cell.tif", channel=0)
    row.intensity_magnitude = IntensityMagnitudeResults(
        total=1000.0, mean=2.5, sd=0.5, density=7.25)
    row.ranges = RangeResults(z_start=12, z_end=46, t_start=0, t_end=15)
    row.binarization.max_island_size = 0.25

    path = str(tmp_path / "Summary.csv")
    results_to_csv([row], path, just_metrics=False, mode=MODES[XYZT])

    back = read_csv_to_channel_results(path)
    assert len(back) == 1, "the row must survive the round-trip"
    assert back[0].intensity_magnitude.total == pytest.approx(1000.0)
    assert back[0].intensity_magnitude.density == pytest.approx(7.25)
    assert back[0].ranges.z_start == pytest.approx(12)
    assert back[0].ranges.t_end == pytest.approx(15)


def test_unpopulated_families_stay_out_of_the_csv(tmp_path):
    from utils.writer import results_to_csv
    import csv

    row = ChannelResults(filepath="cell.tif", channel=0)
    row.binarization.max_island_size = 0.25
    path = str(tmp_path / "Summary.csv")
    results_to_csv([row], path, just_metrics=False, mode=MODES[XYZT])

    headers = next(csv.reader(open(path)))
    assert "Total Intensity" not in headers
    assert "Z Range Start" not in headers


# --------------------------------------------------- instance segmentation


def test_supplied_labels_define_objects_rather_than_connectivity():
    """BARCODE does not segment: a supplied instance partition is authoritative.

    Re-deriving objects by connectivity merges instances that touch, which is the normal
    case in a confluent field. Cellpose separating two adjacent cells must survive.
    """
    from analysis.volumetric.binarization import find_island_properties_3d

    labels = np.zeros((20, 20, 40), np.uint16)
    labels[5:15, 5:15, 5:20] = 1
    labels[5:15, 5:15, 20:35] = 2        # shares a face with instance 1
    binary = labels > 0

    merged = find_island_properties_3d(binary, (1., 1., 1.), 0.5)
    kept = find_island_properties_3d(binary, (1., 1., 1.), 0.5, labelled=labels)

    assert merged["count"] == 1, "connectivity merges touching instances (the old bug)"
    assert kept["count"] == 2, "supplied labels must be respected"
    assert kept["largest"] == 1500 and kept["second_largest"] == 1500
    assert np.isnan(merged["separation"]) and np.isfinite(kept["separation"])


def test_connectivity_remains_the_default_for_binary_masks():
    """A binary mask has no instances, so connectivity is still the right partition."""
    from analysis.volumetric.binarization import find_island_properties_3d

    binary = np.zeros((20, 20, 40), bool)
    binary[5:15, 5:15, 5:12] = True
    binary[5:15, 5:15, 25:32] = True     # genuinely separate
    props = find_island_properties_3d(binary, (1., 1., 1.), 0.5)
    assert props["count"] == 2


def test_object_identity_is_in_the_contract():
    """Per-object rows must be a behaviour change later, not a schema change."""
    assert Metrics.OBJECT_ID.value == "Object ID"
    config = BarcodeConfig().volumetric
    assert config.object_partition == "auto"
    assert config.per_object_rows is False
    assert config.mask_format == "auto"


def test_every_optional_family_survives_a_csv_round_trip(tmp_path):
    """Read-back must reproduce what was written, for ALL eight families.

    The reader rebuilt each family by zipping its CSV block onto
    ``__dataclass_fields__``. That works only while the two are the same length, and
    MeshResults writes a DERIVED column -- Concavity = 1 - Solidity -- so its block is 12
    wide against 11 fields. ``zip`` truncated in silence and shifted every column from
    Concavity onward, which read Mean Curvature <H> back as 1 - Solidity: a positive
    dimensionless number in a column declared 1/um, where the real value is signed.

    Only mesh was affected, and only mesh has a derived column -- which is exactly why a
    test over one family could not have caught it. This covers all of them, and asserts
    the width invariant directly so the next derived column fails loudly.
    """
    from core.modes import get_mode
    from core.results import OPTIONAL_FAMILIES
    from utils.reader import read_csv_to_channel_results
    from utils.writer import results_to_csv

    mode = get_mode("xyzt")
    result = ChannelResults(filepath="cell.tif", channel=0)
    result.binarization.max_island_size = 0.25

    # Distinct, non-symmetric values so any shift shows up as a mismatch rather than
    # coincidentally agreeing.
    for family in OPTIONAL_FAMILIES:
        populated = family.results_cls()
        for offset, name in enumerate(populated.__dataclass_fields__):
            setattr(populated, name, -1.5 + offset * 0.37)
        setattr(result, family.attribute, populated)

    path = str(tmp_path / "Summary.csv")
    switches = {f.switch: True for f in OPTIONAL_FAMILIES}
    results_to_csv([result, result], path, just_metrics=False, mode=mode, **switches)
    back = read_csv_to_channel_results(path)[0]

    for family in OPTIONAL_FAMILIES:
        written = getattr(result, family.attribute)
        read = getattr(back, family.attribute)
        for name in written.__dataclass_fields__:
            assert getattr(read, name) == pytest.approx(getattr(written, name)), (
                f"{family.results_cls.__name__}.{name} did not survive the round trip"
            )


def test_a_family_with_a_derived_column_must_define_from_values():
    """The invariant that makes the generic read-back path safe."""
    from core.results import OPTIONAL_FAMILIES

    for family in OPTIONAL_FAMILIES:
        widths = len(family.results_cls.get_metrics())
        fields = len(family.results_cls.__dataclass_fields__)
        if widths != fields:
            assert hasattr(family.results_cls, "from_values"), (
                f"{family.results_cls.__name__} writes {widths} columns for {fields} "
                f"fields, so it needs a from_values naming its CSV order"
            )
