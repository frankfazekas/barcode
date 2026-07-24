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

    # The range is INCLUSIVE of both ends: 5..15 is eleven slices, not ten.
    cropped = stack.restrict_z(5, 15)
    assert cropped.n_slices == 11
    assert cropped.z_range == (5, 16)          # internally still a half-open pair
    # the original is untouched -- restrict_z returns a view-backed copy
    assert stack.n_slices == 20


def test_restrict_z_handles_negative_and_open_ends(tmp_path):
    stack = read_volume(write_stack(tmp_path / "s.tif", n_z=20))
    assert stack.restrict_z(5, 0).n_slices == 15        # 0 = to the end
    assert stack.restrict_z(0, -5).n_slices == 16       # -5 = sixth-from-last, included
    assert stack.restrict_z(0, -1).n_slices == 20       # -1 = the last slice
    assert stack.restrict_z(0, 0) is stack              # full range is a no-op


def test_restrict_z_rejects_an_empty_range(tmp_path):
    stack = read_volume(write_stack(tmp_path / "s.tif", n_z=20))
    for bad in ((15, 5), (25, 30)):
        with pytest.raises(ValueError, match="selects no slices"):
            stack.restrict_z(*bad)

    # start == end is a single slice under an inclusive range, not an empty one.
    assert stack.restrict_z(10, 10).n_slices == 1


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

    assert n_full == 20 and n_middle == 9      # 6..14 inclusive
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
    # XYT carries no mesh family, so no always-hidden QC metric is in play here and the
    # count reflects exactly what was named.
    headers = ChannelResults.get_headers(just_metrics=True, mode=XYT)
    mask = selection_mask(headers, ["Connectivity", "Curl"])
    assert len(mask) == len(headers)
    assert sum(mask) == len(headers) - 2
    assert [h for h, on in zip(headers, mask) if not on] == ["Connectivity", "Curl"]


def test_selection_mask_always_hides_qc_metrics():
    """Mesh Volume Ratio is a fidelity check, never a barcode column -- excluded even when
    the user hides nothing. It stays in the CSV; only the picture drops it."""
    headers = ChannelResults.get_headers(just_metrics=True, mode=XYZT)
    assert "Mesh Volume Ratio" in headers
    shown = [h for h, on in zip(headers, selection_mask(headers, [])) if on]
    assert "Mesh Volume Ratio" not in shown


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


# --------------------------------------------------- z range units


@pytest.mark.parametrize("units,start,end", [
    ("acquired", 12, 46),
    ("isotropic", 55, 212),      # the same planes on a 0.065 um grid
    ("microns", 3.6, 13.8),      # the same depth in physical units
])
def test_all_z_units_select_the_same_slices(tmp_path, units, start, end):
    """'Slice 46' is ambiguous on anisotropic data, so the unit must be stated.

    An acquired stack at 0.3 um and the isotropic grid a mask lives on at 0.065 um
    differ by the anisotropy factor -- 54 slices versus ~249 for the same object. All
    three ways of naming the same physical range must land on the same slices.
    """
    path = write_stack(tmp_path / "s.tif", n_z=54, z_step=0.3, xy_step=0.065)
    stack = read_volume(path)

    resolved = stack.resolve_z_range(start, end, units)
    restricted = stack.restrict_z(*resolved)

    assert resolved == (12, 46)
    assert restricted.n_slices == 35                       # 12..46 inclusive
    assert restricted.n_slices * stack.z_step_um == pytest.approx(10.5, abs=1e-6)


def test_z_units_default_is_acquired_and_unknown_units_raise(tmp_path):
    stack = read_volume(write_stack(tmp_path / "s.tif", n_z=54))
    assert stack.resolve_z_range(12, 46) == (12, 46)          # default
    assert stack.resolve_z_range(12, 46, "acquired") == (12, 46)
    with pytest.raises(ValueError, match="Unknown z_range_units"):
        stack.resolve_z_range(12, 46, "planes")


def test_zero_end_still_means_to_the_end_in_every_unit(tmp_path):
    stack = read_volume(write_stack(tmp_path / "s.tif", n_z=54))
    for units, start in (("acquired", 12), ("isotropic", 55), ("microns", 3.6)):
        _, end = stack.resolve_z_range(start, 0, units)
        assert end == 0, f"{units}: 0 must stay 'to the end', not convert to a depth"
        assert stack.restrict_z(*stack.resolve_z_range(start, 0, units)).n_slices == 42


def test_apply_z_range_reads_the_config(tmp_path):
    """All three pipelines go through this, so they cannot diverge."""
    from analysis.volumetric.reader import apply_z_range

    stack = read_volume(write_stack(tmp_path / "s.tif", n_z=54))
    config = BarcodeConfig().volumetric
    config.z_start, config.z_end, config.z_range_units = 3.6, 13.8, "microns"
    assert apply_z_range(stack, config).n_slices == 35


# ------------------------------------------------------- intensity masking


def write_masked_pair(tmp_path, n_z=20):
    """An image with a bright object, and a mask on a FINER z grid, as real data is."""
    zz, yy, xx = np.indices((n_z, 32, 32))
    inside = (np.abs(zz - n_z // 2) < 4) & ((yy - 16) ** 2 + (xx - 16) ** 2 <= 25)
    image = np.where(inside, 800, 100).astype(np.uint16)
    tifffile.imwrite(str(tmp_path / "Cell1_1.tif"), image, imagej=True,
                     resolution=(1 / 0.065, 1 / 0.065),
                     metadata={"axes": "ZYX", "spacing": 0.3, "unit": "micron"})

    masks = tmp_path / "masks"
    masks.mkdir(exist_ok=True)
    fine = int(round(n_z * 0.3 / 0.065))              # the isotropic grid a mask lives on
    fz, fy, fx = np.indices((fine, 32, 32))
    centre = fine // 2
    span = int(round(4 * 0.3 / 0.065))
    fine_mask = ((np.abs(fz - centre) < span) &
                 ((fy - 16) ** 2 + (fx - 16) ** 2 <= 25))
    tifffile.imwrite(str(masks / "Cell1_1_SegMask.tif"), fine_mask.astype(np.uint8) * 255)
    return str(tmp_path / "Cell1_1.tif"), str(masks), fine


def test_mask_is_matched_to_the_acquired_slice_grid():
    """Masks are stored finer than the acquisition; the 2D modes bring them down."""
    from analysis.volumetric.segmentation import match_mask_to_image_grid

    fine = np.zeros((250, 8, 8), bool)
    fine[100:150] = True
    matched = match_mask_to_image_grid(fine, 54)
    assert matched.shape == (54, 8, 8)
    assert matched.dtype == bool
    assert matched.any(), "the object must survive the regrid"
    # already-matching input is returned untouched
    assert match_mask_to_image_grid(fine, 250).shape == (250, 8, 8)


def test_xyz_can_use_a_mask_for_binarization_and_intensity(tmp_path):
    """The three configurations must give three different answers."""
    from analysis.volumetric.perslice import run_per_slice_analysis

    image, masks, _ = write_masked_pair(tmp_path)

    def run(use_mask, in_mask_intensity):
        config = BarcodeConfig()
        config.modules.image_binarization = True
        config.modules.intensity_distribution = True
        v = config.volumetric
        v.analysis_mode = XYZ
        config.image_binarization_parameters.bin_factor = 1
        if use_mask:
            v.segmentation_enabled = True
            v.segmentation_root = masks
        v.intensity_use_mask = in_mask_intensity
        per_timepoint, detail = run_per_slice_analysis(image, config)
        middle = per_timepoint[0][len(per_timepoint[0]) // 2]
        return detail.mask_path, middle.binarization.max_island_size, \
            middle.intensity.max_kurtosis

    no_mask_path, _, plain_kurtosis = run(False, False)
    mask_path, _, mask_kurtosis = run(True, False)
    _, _, in_mask_kurtosis = run(True, True)

    assert no_mask_path is None and mask_path is not None
    # binarization source changed but the histogram did not
    assert mask_kurtosis == pytest.approx(plain_kurtosis)
    # restricting the histogram to in-mask voxels must move it
    assert not np.isclose(in_mask_kurtosis, mask_kurtosis), \
        "in-mask intensity should differ from whole-slice intensity"


def test_mask_is_restricted_alongside_the_image(tmp_path):
    """A z range must not make a whole-stack mask look like the wrong depth.

    The mask is validated against the full acquired stack and only then cropped to the
    same range; validating against an already-restricted image compared the mask's whole
    depth with a sub-range and rejected a perfectly good mask.
    """
    from analysis.volumetric.perslice import run_per_slice_analysis

    image, masks, _ = write_masked_pair(tmp_path, n_z=20)
    config = BarcodeConfig()
    config.modules.image_binarization = True
    v = config.volumetric
    v.analysis_mode = XYZ
    v.segmentation_enabled = True
    v.segmentation_root = masks
    v.z_start, v.z_end = 5, 15            # a sub-range of the stack
    config.image_binarization_parameters.bin_factor = 1

    per_timepoint, detail = run_per_slice_analysis(image, config)
    assert detail.mask_path is not None, "the mask must survive a restricted z range"
    assert len(per_timepoint[0]) == 11                 # 5..15 inclusive
    assert detail.z_range == (5, 16)                   # internal half-open pair


# --------------------------------------------------- the inclusive range contract


def test_z_and_t_ranges_are_inclusive_of_both_ends(tmp_path):
    """The range a user types includes the end index they typed.

    An exclusive end cost a silent off-by-one: "analyse slices 12 to 46" quietly dropped
    slice 46, and nothing in the GUI, the CLI or the provenance columns said so. Two
    separate readers of this pipeline read the range as inclusive before it was, which is
    a fact about the interface rather than about them.

    The internal ``z_range``/``t_range`` pairs stay ordinary Python half-open slices --
    only the setting and the reported provenance are inclusive.
    """
    from analysis.volumetric.provenance import build_range_results

    stack = read_volume(write_stack(tmp_path / "s.tif", n_z=54))

    restricted = stack.restrict_z(12, 46)
    assert restricted.n_slices == 35, "12..46 inclusive is 35 slices, not 34"
    assert restricted.z_range == (12, 47), "internally still a half-open pair"

    # The end index the user typed is actually in the data.
    first = int(stack.data[0, 12].max())
    last = int(stack.data[0, 46].max())
    present = [int(restricted.data[0, i].max()) for i in range(restricted.n_slices)]
    assert present[0] == first and present[-1] == last

    # ...and the provenance columns report that same inclusive pair back.
    reported = build_range_results(restricted)
    assert (reported.z_start, reported.z_end) == (12.0, 46.0)


def test_zero_still_means_to_the_end(tmp_path):
    """The 0 sentinel outlives the change, so existing config files keep working."""
    stack = read_volume(write_stack(tmp_path / "s.tif", n_z=54))
    assert stack.restrict_z(0, 0) is stack                 # the default: whole stack
    assert stack.restrict_z(10, 0).n_slices == 44          # 10..end
    assert stack.restrict_z(0, -1).n_slices == 54          # -1 = the last slice
