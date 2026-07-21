"""The four additions, checked where they are actually consumed.

Unit tests prove the maths; these prove the wiring. The failure this guards against has
happened twice in this codebase: a family computes correctly, and the CSV, the reader or
the barcode disagrees about whether its columns exist -- silently dropping rows or
rendering the wrong header set.

Run: python -m pytest tests/test_new_families_end_to_end.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

from core import BarcodeConfig
from core.results import (
    OPTIONAL_FAMILIES,
    ChannelResults,
    CurvatureRangeResults,
    MaskIntensityResults,
    SliceProfileResults,
)

NEW_SWITCHES = ("include_curvature_range", "include_slice_profile",
                "include_mask_intensity")


Z_STEP, XY_STEP = 0.3, 0.065
CENTRES = (12, 28)          # x centres of the two objects, shared by image and mask


def _objects(shape, z_centre, z_half, radius=5, y_centre=20):
    """Two cylinders, defined identically on whatever z grid is passed in.

    Image and mask must describe the *same* objects on their own grids -- the mask is
    stored on the finer isotropic grid, as real segmentations are.
    """
    zz, yy, xx = np.indices(shape)
    in_z = np.abs(zz - z_centre) < z_half
    for i, centre_x in enumerate(CENTRES):
        yield i + 1, in_z & (((yy - y_centre) ** 2 + (xx - centre_x) ** 2) <= radius ** 2)


def write_volume(path, n_t=2, n_z=12, size=40):
    """Objects that sit clear of every edge, so nothing is clipped by construction."""
    data = np.zeros((n_t, n_z, size, size), np.uint16)
    _, yy, xx = np.indices((n_z, size, size))
    # Texture *inside* the objects, not a flat fill: a constant object has no intensity
    # distribution, so the in-mask family would correctly skip it and measure nothing.
    texture = 700 + 150 * np.sin(xx / 2.0) * np.cos(yy / 2.0)
    for t in range(n_t):
        frame = np.full((n_z, size, size), 100.0)
        for _, region in _objects((n_z, size, size), n_z // 2, 3):
            frame[region] = (texture + 50 * t)[region]
        data[t] = frame.astype(np.uint16)
    tifffile.imwrite(str(path), data, imagej=True,
                     resolution=(1 / XY_STEP, 1 / XY_STEP),
                     metadata={"axes": "TZYX", "spacing": Z_STEP, "unit": "micron"})
    return str(path)


def write_mask(path, n_z=12, size=40):
    """A LABEL volume on the finer isotropic z grid, matching how masks really arrive."""
    fine = int(round(n_z * Z_STEP / XY_STEP))
    mask = np.zeros((fine, size, size), np.int32)
    for label, region in _objects((fine, size, size), fine // 2,
                                  3 * Z_STEP / XY_STEP):
        mask[region] = label
    tifffile.imwrite(str(path), mask)
    return str(path)


# ------------------------------------------------------------------ registry


def test_every_new_family_is_registered():
    registered = {f.switch for f in OPTIONAL_FAMILIES}
    assert set(NEW_SWITCHES) <= registered


def test_every_new_family_is_off_by_default_in_every_mode():
    """A new column appearing unasked would break the reference 2D schema."""
    from core.modes import MODES

    for mode in MODES.values():
        headers = ChannelResults.get_headers(just_metrics=False, mode=mode)
        for name in ("Minimum Curvature", "Broadest Slice Depth", "In-Mask MFI"):
            assert name not in headers, f"{name} leaked into {mode.key}"


def test_the_2d_schema_is_unchanged():
    assert len(ChannelResults.get_headers(just_metrics=False)) == 28


def test_families_compose_additively():
    """Enabling all three adds exactly the sum of their widths, in registry order."""
    base = ChannelResults.get_headers(just_metrics=False, mode="xyzt")
    both = ChannelResults.get_headers(
        just_metrics=False, mode="xyzt",
        **{switch: True for switch in NEW_SWITCHES})
    assert len(both) == len(base) + 2 + 3 + 7


def test_switch_names_are_validated():
    with pytest.raises(TypeError, match="Unknown optional-family switch"):
        ChannelResults.get_headers(just_metrics=False, mode="xyzt", include_slice=True)


# ------------------------------------------------------------------ round trip


@pytest.mark.parametrize("switch,populated", [
    ("include_curvature_range", CurvatureRangeResults(min_curvature=-0.5, max_curvature=2.0)),
    ("include_slice_profile", SliceProfileResults(broadest_index=6.0, broadest_depth=1.8,
                                                  broadest_area=0.25)),
    ("include_mask_intensity", MaskIntensityResults(mfi=880.0, sd=40.0, cv=0.045,
                                                    skewness=0.3, entropy=4.2,
                                                    entropy_normalized=0.7,
                                                    bright_fraction=0.02)),
])
def test_a_populated_family_survives_a_csv_round_trip(tmp_path, switch, populated):
    """The reader identifies a layout by its header set; a new family must be findable.

    When this broke before, every row was silently dropped rather than erroring.
    """
    from utils.reader import read_csv_to_channel_results
    from utils.writer import results_to_csv

    attribute = next(f.attribute for f in OPTIONAL_FAMILIES if f.switch == switch)
    results = ChannelResults(filepath="cell.tif", channel=0)
    setattr(results, attribute, populated)
    assert getattr(results, attribute).is_populated()

    out = tmp_path / "Summary.csv"
    results_to_csv([results], str(out), just_metrics=False, mode="xyzt")

    recovered = read_csv_to_channel_results(str(out))
    assert len(recovered) == 1, "the reader dropped the row -- header set unrecognised"
    round_tripped = getattr(recovered[0], attribute)
    np.testing.assert_allclose(round_tripped.get_data(), populated.get_data(), rtol=1e-9)


def test_the_clipping_flag_survives_a_csv_round_trip(tmp_path):
    from utils.reader import read_csv_to_channel_results
    from utils.writer import results_to_csv

    results = ChannelResults(filepath="cell.tif", channel=0)
    results.fov_clip_flag = 1

    out = tmp_path / "Summary.csv"
    results_to_csv([results], str(out), just_metrics=False, mode="xyzt")
    assert "6" in read_csv_to_channel_results(str(out))[0].total_flags


# ------------------------------------------------------------------ pipeline


def test_slice_profile_and_clipping_come_out_of_a_real_run(tmp_path):
    from analysis.volumetric.run import run_volumetric_analysis

    path = write_volume(tmp_path / "Cell1_1.tif")

    config = BarcodeConfig()
    config.modules.image_binarization = True
    config.modules.intensity_distribution = False
    v = config.volumetric
    v.analysis_mode = "xyzt"
    v.enable_slice_profile = True

    results, detail = run_volumetric_analysis(path, config)

    assert results.slice_profile.is_populated()
    # These objects are cylinders spanning acquired z 4..8, so every one of those slices
    # is equally broad and argmax legitimately returns the first. The exact-centre claim
    # is pinned on an ellipsoid in tests/test_slice_profile.py, where it is well posed;
    # here the honest assertion is that it lands inside the object.
    #
    # Asserted in MICRONS, not slice indices. A run with no segmentation now resamples to
    # an isotropic grid (12 acquired slices -> 51 at 0.065 um), so an index-based bound is
    # a statement about which grid happens to be in use; the physical depth is not.
    # The cylinders occupy z 4..8 x 0.3 um = 1.2..2.4 um, and linear interpolation ramps
    # the boundary over roughly one acquired slice either side, so the first thresholded
    # slice sits a little below 1.2.
    assert 0.85 <= results.slice_profile.broadest_depth <= 2.45, (
        f"broadest slice at {results.slice_profile.broadest_depth} um is outside the "
        f"cylinders' 1.2-2.4 um extent")
    assert results.slice_profile.broadest_area > 0
    assert results.fov_clip_flag == 0
    assert "6" not in results.convert_flags()
    assert len(detail.slice_profile) == len(detail.frame_indices)


def test_an_object_touching_the_edge_raises_the_clipping_flag(tmp_path):
    from analysis.volumetric.run import run_volumetric_analysis

    # Fills the whole field, so foreground reaches every boundary.
    data = np.full((1, 8, 30, 30), 900, np.uint16)
    data[:, :, ::3, ::3] = 100          # some texture so it still binarizes
    path = str(tmp_path / "Cell2_1.tif")
    tifffile.imwrite(path, data, imagej=True,
                     metadata={"axes": "TZYX", "spacing": 0.3, "unit": "micron"})

    config = BarcodeConfig()
    config.modules.image_binarization = True
    config.modules.intensity_distribution = False
    config.volumetric.analysis_mode = "xyzt"
    config.volumetric.enable_slice_profile = True

    results, _ = run_volumetric_analysis(path, config)
    assert results.fov_clip_flag == 1
    assert "6" in results.convert_flags()


def test_mask_intensity_comes_out_of_a_real_run(tmp_path):
    from analysis.volumetric.run import run_volumetric_analysis

    path = write_volume(tmp_path / "Cell3_1.tif", n_t=1)
    masks = tmp_path / "masks"
    masks.mkdir()
    write_mask(masks / "Cell3_1_SegMask.tif")

    config = BarcodeConfig()
    config.modules.image_binarization = True
    config.modules.intensity_distribution = False
    v = config.volumetric
    v.analysis_mode = "xyzt"
    v.segmentation_enabled = True
    v.segmentation_root = str(masks)
    v.enable_mask_intensity = True

    results, detail = run_volumetric_analysis(path, config)

    assert results.mask_intensity.is_populated()
    assert results.mask_intensity.mfi > 0
    assert detail.mask_intensity[0].object_ids == [1, 2], "both labels measured"


def test_mask_intensity_reports_rather_than_crashes_without_a_segmentation(tmp_path, capsys):
    """One misconfigured file must not abort a batch."""
    from analysis.volumetric.run import run_volumetric_analysis

    path = write_volume(tmp_path / "Cell4_1.tif", n_t=1)

    config = BarcodeConfig()
    config.modules.image_binarization = True
    config.modules.intensity_distribution = False
    config.volumetric.analysis_mode = "xyzt"
    config.volumetric.enable_mask_intensity = True

    results, _ = run_volumetric_analysis(path, config)
    assert not results.mask_intensity.is_populated()
    assert "needs a segmentation" in capsys.readouterr().out


def test_curvature_range_reports_rather_than_crashes_without_a_mesh(tmp_path, capsys):
    from analysis.volumetric.run import run_volumetric_analysis

    path = write_volume(tmp_path / "Cell5_1.tif", n_t=1)

    config = BarcodeConfig()
    config.modules.image_binarization = True
    config.modules.intensity_distribution = False
    config.volumetric.analysis_mode = "xyzt"
    config.volumetric.enable_curvature_range = True

    results, _ = run_volumetric_analysis(path, config)
    assert not results.curvature_range.is_populated()
    assert "needs the mesh family" in capsys.readouterr().out


# ------------------------------------------------------------------ cli parity


def test_list_metrics_covers_every_registered_family():
    """The guard that stops --list-metrics under-reporting when a family is added."""
    from scripts._cli import family_switches

    switches = family_switches(BarcodeConfig())
    assert set(switches) == {f.switch for f in OPTIONAL_FAMILIES}


def test_cli_switches_follow_the_config():
    from scripts._cli import family_switches

    config = BarcodeConfig()
    config.volumetric.enable_slice_profile = True
    config.volumetric.enable_mask_intensity = True

    switches = family_switches(config)
    assert switches["include_slice_profile"] is True
    assert switches["include_mask_intensity"] is True
    assert switches["include_curvature_range"] is False


# ------------------------------------------- meshing / isotropy without a mask

def test_a_run_without_a_segmentation_is_still_resampled_to_isotropic(tmp_path):
    """make_isotropic used to be a no-op without a mask, silently.

    Resampling targeted the mask's grid, so with no segmentation the run fell through to
    the acquired voxels -- 4.6x anisotropic on typical confocal data -- and every 3D
    shape and connectivity metric was measured on non-cubic voxels. The isotropic grid
    is fixed by the acquired spacing alone and needs no mask.
    """
    from analysis.volumetric.run import run_volumetric_analysis

    path = write_volume(tmp_path / "Cell1_1.tif")
    config = BarcodeConfig()
    config.modules.image_binarization = True
    config.modules.intensity_distribution = False
    config.volumetric.analysis_mode = "xyzt"
    config.volumetric.make_isotropic = True

    _results, detail = run_volumetric_analysis(path, config)

    spacing = detail.spacing_zyx_um
    assert max(spacing) - min(spacing) < 1e-9, f"grid is not isotropic: {spacing}"
    assert abs(spacing[0] - XY_STEP) < 1e-9, "should resample z up to the xy step"
    # 12 acquired slices at 0.3 um span 3.3 um, which is 51 slices at 0.065 um.
    assert detail.shape_zyx[0] == 51


def test_make_isotropic_off_leaves_the_acquired_grid_alone(tmp_path):
    from analysis.volumetric.run import run_volumetric_analysis

    path = write_volume(tmp_path / "Cell1_1.tif")
    config = BarcodeConfig()
    config.modules.image_binarization = True
    config.modules.intensity_distribution = False
    config.volumetric.analysis_mode = "xyzt"
    config.volumetric.make_isotropic = False

    _results, detail = run_volumetric_analysis(path, config)

    assert detail.shape_zyx[0] == 12, "opting out must not resample"
    assert abs(detail.spacing_zyx_um[0] - Z_STEP) < 1e-9


def test_meshing_without_a_segmentation_meshes_the_binarized_volume(tmp_path):
    """Previously this printed "skipping the mesh" and returned nothing.

    The binarization branch already computes the surface; refusing to mesh it withheld a
    result the pipeline was one call away from producing.
    """
    from analysis.volumetric.run import run_volumetric_analysis

    path = write_volume(tmp_path / "Cell1_1.tif")
    config = BarcodeConfig()
    config.modules.image_binarization = True
    config.modules.intensity_distribution = False
    v = config.volumetric
    v.analysis_mode = "xyzt"
    v.mesh_enabled = True
    v.mesh_curvature = False          # curvature is the slow part and not what is under test

    results, detail = run_volumetric_analysis(path, config)

    assert detail.meshes, "no mesh produced without a segmentation"
    assert results.mesh.is_populated()
    assert results.mesh.get_data()[0] > 0, "mesh volume should be positive"
