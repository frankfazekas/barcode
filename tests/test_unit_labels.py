"""A unit label must describe the number actually stored in the column.

Three labels were wrong, and all three were wrong in the direction that flatters the
reader rather than alarming them:

* "% of FOV" on values in 0-1 -- a void filling 94% of the field was drawn on the
  barcode as "0.94 % of FOV", two orders of magnitude out.
* "% of Frames" on Connectivity, which is `n_connected / n_frames`.
* "Fractional Change" on `final / initial`, where 1.0 means NO CHANGE. A fractional
  change conventionally has 0 as its no-change point, so that label was wrong about the
  magnitude *and* the zero.

The values are reference-validated and were NOT rescaled to fit the old wording; the
wording was corrected to fit the values. These tests pin that direction.
"""
import pytest

from core.metrics import Units
from core.modes import MODES
from core.results import ChannelResults

# Units is not an Enum -- it is a UnitsNum(ABC) whose members are plain string class
# attributes, so introspection goes through vars() and comparison is by string value.
UNIT_MEMBERS = {name: value for name, value in vars(Units).items()
                if not name.startswith("_") and isinstance(value, str)}


def test_no_unit_label_claims_percent():
    """The whole class of bug: a '%' label on a 0-1 quantity."""
    offenders = {n: v for n, v in UNIT_MEMBERS.items() if "%" in v}
    assert not offenders, (
        f"{offenders} say percent. If a metric really is 0-100 this test needs "
        f"updating; if it is 0-1 the label is wrong.")


def test_fractional_units_are_not_named_percent():
    """The label was wrong because the MEMBER was named PERCENT_*, and the string was
    written to match the name instead of the data. Keep the names honest."""
    for name, value in UNIT_MEMBERS.items():
        if value.startswith("fraction"):
            assert name.startswith("FRACTION"), f"{name} = {value!r}"
    assert "PERCENT_FOV" not in UNIT_MEMBERS
    assert "PERCENT_FRAMES" not in UNIT_MEMBERS
    assert "PERCENT_CHANGE" not in UNIT_MEMBERS


def test_the_ratio_unit_does_not_call_itself_a_change():
    """`final / initial` is a ratio; 1.0 is no change, not 100% change."""
    label = Units.RATIO_TO_INITIAL.lower()
    assert "ratio" in label
    assert "fractional change" not in label


@pytest.mark.parametrize("mode_key", ["xyt", "xyz", "xyzt"])
def test_every_column_has_a_unit_in_every_mode(mode_key):
    """A mismatch here means headers and units have drifted apart, which is how a label
    ends up describing the column next to it."""
    results = ChannelResults(filepath="x", channel=0)
    headers = ChannelResults.get_headers(just_metrics=True, mode=MODES[mode_key])
    units = results.get_units(just_metrics=True, mode=MODES[mode_key])
    assert len(headers) == len(units), f"{len(headers)} headers vs {len(units)} units"


def test_area_and_volume_fractions_are_labelled_as_fractions():
    """Spot-check the columns that were actually wrong, by name, in both modes."""
    for mode_key, word in (("xyt", "Area"), ("xyzt", "Volume")):
        results = ChannelResults(filepath="x", channel=0)
        headers = ChannelResults.get_headers(just_metrics=True, mode=MODES[mode_key])
        units = dict(zip(headers, results.get_units(just_metrics=True, mode=MODES[mode_key])))
        for name in (f"Maximum Island {word}", f"Maximum Void {word}",
                     f"Mean Island {word}", f"Total Island {word}"):
            assert units[name] == Units.FRACTION_FOV, f"{name} -> {units[name]}"
        for name in (f"Maximum Island {word} Change", f"Maximum Void {word} Change"):
            assert units[name] == Units.RATIO_TO_INITIAL, f"{name} -> {units[name]}"
