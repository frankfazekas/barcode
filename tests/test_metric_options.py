"""Z-range selection, the per-component family, and metric on/off.

Three things a user controls that change what comes out, so each needs to be pinned:
the depth range analysed, whether the object-size distribution is reported, and which
metrics reach the barcode image.

Run: python -m pytest tests/test_metric_options.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

from analysis.volumetric.binarization import find_island_properties_3d
from analysis.volumetric.reader import read_volume
from core import BarcodeConfig
from core.metrics import selection_mask
from core.modes import MODES, XYT, XYZ, XYZT
from core.results import ChannelResults, ComponentResults


def write_stack(path, n_z=20, xy_step=0.065, z_step=0.3):
    """Bright object in the middle slices only, background above and below."""
    zz, yy, xx = np.indices((n_z, 32, 32))
    inside = (np.abs(zz - n_z // 2) < 4) & ((yy - 16) ** 2 + (xx - 16) ** 2 <= 25)
    tifffile.imwrite(str(path), np.where(inside, 800, 100).astype(np.uint16), imagej=True,
                     resolution=(1 / xy_step, 1 / xy_step),
                     metadata={"axes": "ZYX", "spacing": z_step, "unit": "micron"})
    return str(path)


# ------------------------------------------------------------------ z range


def test_restrict_z_selects_the_requested_slices(tmp_path):
    stack = read_volume(write_stack(tmp_path / "s.tif", n_z=20))
    assert stack.n_slices == 20 and stack.z_range is None

    cropped = stack.restrict_z(5, 15)
    assert cropped.n_slices == 10
    assert cropped.z_range == (5, 15)
    # the original is untouched -- restrict_z returns a view-backed copy
    assert stack.n_slices == 20


def test_restrict_z_handles_negative_and_open_ends(tmp_path):
    stack = read_volume(write_stack(tmp_path / "s.tif", n_z=20))
    assert stack.restrict_z(5, 0).n_slices == 15        # 0 = to the end
    assert stack.restrict_z(0, -5).n_slices == 15       # negative counts back
    assert stack.restrict_z(0, 0) is stack              # full range is a no-op


def test_restrict_z_rejects_an_empty_range(tmp_path):
    stack = read_volume(write_stack(tmp_path / "s.tif", n_z=20))
    for bad in ((15, 5), (10, 10), (25, 30)):
        with pytest.raises(ValueError, match="selects no slices"):
            stack.restrict_z(*bad)


def test_z_range_changes_the_measured_values(tmp_path):
    """The point of the setting: background slices dilute the metrics.

    The fixture is bright only in the middle, so restricting to those slices must move
    the intensity distribution. If it did not, the range would not be being applied.
    """
    from analysis.volumetric.slicewise import run_slicewise_analysis

    path = write_stack(tmp_path / "Cell1_1.tif", n_z=20)

    def kurtosis_for(z_start, z_end):
        config = BarcodeConfig()
        config.modules.intensity_distribution = True
        config.volumetric.analysis_mode = XYZ
        config.volumetric.z_start, config.volumetric.z_end = z_start, z_end
        rows, detail = run_slicewise_analysis(path, config)
        return rows[0].intensity.max_kurtosis, detail.n_slices

    full, n_full = kurtosis_for(0, 0)
    middle, n_middle = kurtosis_for(6, 14)

    assert n_full == 20 and n_middle == 8
    assert not np.isclose(full, middle), "restricting z must change the result"


# --------------------------------------------------- per-component statistics


def test_component_stats_describe_the_size_distribution():
    """One big object plus debris must be distinguishable from several even ones."""
    lopsided = np.zeros((24, 40, 40), bool)
    lopsided[8:16, 8:32, 8:32] = True                      # one dominant object
    for i, (y, x) in enumerate([(2, 2), (2, 36), (36, 2)]):
        lopsided[2:4, y:y + 2, x:x + 2] = True             # specks

    # Similar but not identical: four identical objects have zero variance, so their
    # skewness is 0/0 and correctly comes back NaN (see the graceful-degradation test).
    even = np.zeros((24, 40, 40), bool)
    for (cz, cy, cx), r in zip([(6, 10, 10), (6, 10, 30), (16, 30, 10), (16, 30, 30)],
                               (3, 4, 3, 4)):
        even[cz - r:cz + r, cy - r:cy + r, cx - r:cx + r] = True

    a = find_island_properties_3d(lopsided, (1.0, 1.0, 1.0), 0.5)
    b = find_island_properties_3d(even, (1.0, 1.0, 1.0), 0.5)

    assert a["count"] == 4 and b["count"] == 4
    # same object count, wildly different distribution shape
    assert a["skew"] > 1.0, "one dominant object should skew right"
    assert abs(b["skew"]) < 0.5, "even objects should be near-symmetric"
    assert a["median"] < b["median"], "the lopsided set is mostly specks"


def test_component_stats_degrade_gracefully():
    single = np.zeros((16, 16, 16), bool)
    single[4:12, 4:12, 4:12] = True
    props = find_island_properties_3d(single, (1.0, 1.0, 1.0), 0.5)
    assert props["count"] == 1
    assert props["sd"] == 0.0
    assert np.isnan(props["skew"]), "skewness is undefined for fewer than three objects"

    empty = find_island_properties_3d(np.zeros((8, 8, 8), bool), (1.0, 1.0, 1.0), 0.5)
    assert empty["count"] == 0
    for key in ("sd", "skew", "median"):
        assert np.isnan(empty[key])


def test_component_family_is_opt_in_and_volumetric_only():
    assert MODES[XYZT].supports_component_stats
    assert not MODES[XYT].supports_component_stats
    assert not MODES[XYZ].supports_component_stats

    base = ChannelResults.get_headers(just_metrics=False, mode=XYZT)
    with_stats = ChannelResults.get_headers(
        just_metrics=False, mode=XYZT, include_components=True)
    assert len(with_stats) == len(base) + 4
    assert "Island Count" in with_stats and "Island Count" not in base


def test_component_metric_names_follow_the_mode():
    assert "Island Volume SD" in ComponentResults.get_headers(mode=MODES[XYZT])
    assert "Island Area SD" in ComponentResults.get_headers(mode=MODES[XYT])


def test_headers_and_data_agree_with_components_on():
    row = ChannelResults(filepath="x.tif", channel=0)
    headers = ChannelResults.get_headers(
        just_metrics=False, mode=XYZT, include_components=True)
    data = row.get_data(just_metrics=False, mode=XYZT, include_components=True)
    assert len(headers) == len(data)


# ------------------------------------------------------------ metric on/off


def test_selection_mask_hides_by_name():
    headers = ChannelResults.get_headers(just_metrics=True, mode=XYZT)
    mask = selection_mask(headers, ["Connectivity", "Curl"])
    assert len(mask) == len(headers)
    assert sum(mask) == len(headers) - 2
    assert [h for h, on in zip(headers, mask) if not on] == ["Connectivity", "Curl"]


def test_selection_mask_defaults_to_everything():
    headers = ChannelResults.get_headers(just_metrics=True, mode=XYT)
    for empty in (None, [], ["", "  "]):
        assert all(selection_mask(headers, empty))


def test_selection_mask_ignores_names_from_another_mode():
    """A selection saved for one mode is routinely reused for another."""
    headers = ChannelResults.get_headers(just_metrics=True, mode=XYZ)
    mask = selection_mask(headers, ["Speed", "Sphericity", "Connectivity"])
    assert len(mask) == len(headers)
    # only the one that exists here is hidden
    assert sum(mask) == len(headers) - 1


def test_hiding_never_changes_the_csv_schema():
    """Selection is a display concern; the CSV must stay round-trippable."""
    config = BarcodeConfig()
    config.writer.hidden_barcode_metrics = ["Connectivity", "Curl"]
    headers = ChannelResults.get_headers(just_metrics=False, mode=XYZT)
    assert "Connectivity" in headers and "Curl" in headers
