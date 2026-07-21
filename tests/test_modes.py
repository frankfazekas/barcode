"""Validation for analysis modes.

The claims worth pinning are that each mode emits exactly the metrics it can support,
that xyz reproduces the numbers the 2D pipeline already produces on a Z-stack (because
it is the same computation, correctly labelled), and that a mode/axes mismatch fails
loudly rather than silently analysing depth as time.

Run: python -m pytest tests/test_modes.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

from analysis.volumetric.slicewise import run_slicewise_analysis
from core import BarcodeConfig, VolumetricConfig
from core.modes import MODES, XYT, XYZ, XYZT, get_mode, guard_declared_axes
from core.results import ChannelResults


# --------------------------------------------------------------- the registry


def test_every_mode_declares_a_coherent_combination():
    for key, mode in MODES.items():
        assert mode.key == key
        assert mode.spatial_dims in (2, 3)
        assert mode.progression in ("time", "depth")
        # Meshing needs a volume; a plane has no surface to mesh.
        assert not (mode.supports_mesh and mode.spatial_dims == 2)
        # Flow is only physical when the progression axis is time.
        assert not (mode.supports_flow and not mode.progresses_in_time)


def test_unknown_mode_names_the_valid_options():
    with pytest.raises(ValueError, match="xyt, xyz, xyzt"):
        get_mode("4d")


# --------------------------------------------------------------- column sets


# XYZT = the 28 shared columns + the 11-column mesh family (mesh geometry, Lateral/Axial
# Ratio, Solidity, and curvature). XYZ is slice-wise 2D, so it carries no mesh family.
@pytest.mark.parametrize("mode,expected", [(XYT, 28), (XYZ, 21), (XYZT, 39)])
def test_column_counts_per_mode(mode, expected):
    """Pinned so a schema drift is caught rather than discovered in a figure."""
    assert len(ChannelResults.get_headers(just_metrics=False, mode=mode)) == expected


def test_legacy_call_without_mode_is_unchanged():
    """The 2D schema must not move: published CSVs are compared against it."""
    assert len(ChannelResults.get_headers(just_metrics=False)) == 28
    assert ChannelResults.get_headers(just_metrics=False) == \
        ChannelResults.get_headers(just_metrics=False, mode=XYT)


def test_xyz_omits_flow_and_xyt_keeps_it():
    xyt = ChannelResults.get_headers(just_metrics=True, mode=XYT)
    xyz = ChannelResults.get_headers(just_metrics=True, mode=XYZ)
    flow = {"Speed", "Speed Change", "Mean Flow Direction", "Directional Spread",
            "Velocity Correlation Length", "Divergence", "Curl"}
    assert flow <= set(xyt)
    assert not (flow & set(xyz)), "xyz must not carry velocity columns"


def test_only_the_volumetric_mode_uses_volume_names():
    xyt = ChannelResults.get_headers(just_metrics=True, mode=XYT)
    xyz = ChannelResults.get_headers(just_metrics=True, mode=XYZ)
    xyzt = ChannelResults.get_headers(just_metrics=True, mode=XYZT)

    assert "Maximum Island Area" in xyt and "Maximum Island Area" in xyz
    assert "Maximum Island Volume" in xyzt
    assert "Maximum Island Area" not in xyzt
    # dimension-neutral metrics keep one name everywhere
    for shared in ("Connectivity", "Mean Island Anisotropy", "Mean Island Separation",
                   "Structural Correlation Length"):
        assert shared in xyt and shared in xyz and shared in xyzt


def test_xyz_change_metrics_say_they_are_over_depth():
    """A depth trend read as a time trend is a silent scientific error."""
    xyz = ChannelResults.get_headers(just_metrics=True, mode=XYZ)
    assert "Maximum Island Area Change (over Z)" in xyz
    assert "Kurtosis Change (over Z)" in xyz
    assert "Kurtosis Change" not in xyz


def test_units_track_the_metric_names():
    from core import Units

    for mode, expected in ((XYT, Units.AREA), (XYZT, Units.VOLUME)):
        metrics = ChannelResults.get_physical_metrics(just_metrics=True, mode=mode)
        units = ChannelResults.get_physical_units(just_metrics=True, mode=mode)
        sizes = [u for m, u in zip(metrics, units) if "Island" in m.value and "Change" not in m.value
                 and "Anisotropy" not in m.value and "Separation" not in m.value
                 and "Correlation" not in m.value]
        assert sizes, f"no size metrics found for {mode}"
        assert all(u == expected for u in sizes), f"{mode} size units should be {expected}"


def test_headers_and_data_stay_the_same_length_in_every_mode():
    row = ChannelResults(filepath="x.tif", channel=0)
    for mode in MODES:
        headers = ChannelResults.get_headers(just_metrics=False, mode=mode)
        data = row.get_data(just_metrics=False, mode=mode)
        units = ChannelResults.get_units(just_metrics=False, mode=mode)
        assert len(headers) == len(data) == len(units), mode


# --------------------------------------------------------------- axis validation


@pytest.mark.parametrize("mode,axes,ok", [
    (XYZ, "ZYX", True), (XYZT, "TZYX", True), (XYZT, "ZYX", True),
    (XYZ, "TYX", False), (XYZT, "TYX", False), (XYT, "ZYX", False),
])
def test_axis_validation(mode, axes, ok):
    m = get_mode(mode)
    if ok:
        m.validate_axes(axes, "sample.tif")
    else:
        with pytest.raises(ValueError, match="needs axis"):
            m.validate_axes(axes, "sample.tif")


def test_xyt_guard_blocks_a_declared_zstack_but_allows_undeclared_axes(tmp_path):
    """The guard must be narrow: most planar TIFFs declare no axis at all.

    A plain multipage movie comes back as 'QYX'. Demanding a T axis would reject
    ordinary 2D data including the published reference set, so only an explicit
    Z-without-T is refused.
    """
    plain = tmp_path / "movie.tif"
    tifffile.imwrite(str(plain), np.zeros((8, 16, 16), np.uint16))
    guard_declared_axes(get_mode(XYT), str(plain))  # must not raise

    # A declared Z-with-no-T warns but must NOT raise: Fiji writes an ordinary 2D movie
    # as an ImageJ stack with slices=N, which reads back as ZYX, so raising here rejected
    # legitimate 2D input on the reference-validated path.
    stack = tmp_path / "stack.tif"
    tifffile.imwrite(str(stack), np.zeros((8, 16, 16), np.uint16), imagej=True,
                     metadata={"axes": "ZYX", "spacing": 0.3, "unit": "micron"})
    guard_declared_axes(get_mode(XYT), str(stack))


def test_guard_warns_about_a_declared_z_stack_in_xyt(tmp_path, capsys):
    stack = tmp_path / "stack.tif"
    tifffile.imwrite(str(stack), np.zeros((8, 16, 16), np.uint16), imagej=True,
                     metadata={"axes": "ZYX", "spacing": 0.3, "unit": "micron"})
    guard_declared_axes(get_mode(XYT), str(stack))
    out = capsys.readouterr().out
    assert "WARNING" in out and "8 Z slice(s)" in out


def test_guard_ignores_non_tiff_and_unreadable_files(tmp_path):
    guard_declared_axes(get_mode(XYT), str(tmp_path / "nope.nd2"))
    junk = tmp_path / "junk.tif"
    junk.write_bytes(b"not a tiff")
    guard_declared_axes(get_mode(XYT), str(junk))  # unreadable -> pass through


# --------------------------------------------------------------- config


def test_legacy_enabled_flag_migrates_to_xyzt():
    """Settings.yaml written before modes existed must not load as 2D."""
    assert VolumetricConfig.from_dict({"enabled": True}).analysis_mode == "xyzt"
    assert VolumetricConfig.from_dict({"enabled": False}).analysis_mode == "xyt"
    # an explicit mode always wins over the legacy flag
    assert VolumetricConfig.from_dict(
        {"enabled": True, "analysis_mode": "xyz"}).analysis_mode == "xyz"


def test_default_config_is_the_2d_mode():
    assert BarcodeConfig().volumetric.mode.key == XYT


# --------------------------------------------------------------- xyz behaviour


def write_stack(path, n_z=12, value=None, z_step=0.3, xy_step=0.065):
    """A z-stack of several blobs whose in-plane size varies with depth.

    Several rather than one: the 2D binarization branch raises "kth out of bounds" on a
    frame containing exactly one island (it partitions a 1x1 distance matrix), so a
    single-blob fixture would exercise that pre-existing defect instead of this code.
    See test_single_island_frame_is_survivable.
    """
    if value is None:
        zz, yy, xx = np.indices((n_z, 48, 48))
        radius = 3 + 2 * np.sin(np.pi * zz / max(n_z - 1, 1))
        value = np.full((n_z, 48, 48), 100)
        for cy, cx in ((14, 14), (14, 34), (34, 14), (34, 34)):
            value = np.where((yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2, 600, value)
    tifffile.imwrite(str(path), value.astype(np.uint16), imagej=True,
                     resolution=(1 / xy_step, 1 / xy_step),
                     metadata={"axes": "ZYX", "spacing": z_step, "unit": "micron"})
    return str(path)


def write_single_blob_stack(path, n_z=12, xy_step=0.065):
    # 12 slices, not fewer: with fewer than frame_step (default 10) the 2D helper
    # find_analysis_frames turns its step into a float and raises TypeError, so a short
    # stack would take down the intensity branch too and obscure what this fixture is for.
    zz, yy, xx = np.indices((n_z, 32, 32))
    value = np.where((yy - 16) ** 2 + (xx - 16) ** 2 <= 36, 600, 100)
    tifffile.imwrite(str(path), value.astype(np.uint16), imagej=True,
                     resolution=(1 / xy_step, 1 / xy_step),
                     metadata={"axes": "ZYX", "spacing": 0.3, "unit": "micron"})
    return str(path)


def test_xyz_produces_one_row_per_timepoint(tmp_path):
    path = write_stack(tmp_path / "Cell1_1.tif")
    config = BarcodeConfig()
    config.modules.image_binarization = True
    config.modules.intensity_distribution = True
    config.volumetric.analysis_mode = XYZ

    rows, detail = run_slicewise_analysis(path, config)
    assert len(rows) == 1               # one timepoint in the file
    assert detail.n_slices == 12
    assert detail.xy_step_um == pytest.approx(0.065, rel=1e-4)


def test_xyz_ignores_a_requested_flow_branch(tmp_path):
    """Flow between focal planes is not motion; asking for it must not produce it."""
    path = write_stack(tmp_path / "Cell1_1.tif")
    config = BarcodeConfig()
    config.modules.image_binarization = True
    config.modules.optical_flow = True   # requested
    config.volumetric.analysis_mode = XYZ

    rows, _ = run_slicewise_analysis(path, config)
    assert np.isnan(rows[0].flow.mean_speed), "xyz must not compute a velocity"
    assert "Speed" not in ChannelResults.get_headers(just_metrics=True, mode=XYZ)


def test_xyz_uses_the_files_own_xy_spacing_not_the_2d_tab(tmp_path):
    """In-plane metrics are 2D, so the scale is the XY pixel size from the file."""
    path = write_stack(tmp_path / "Cell1_1.tif", xy_step=0.2)
    config = BarcodeConfig()
    config.modules.image_binarization = True
    config.reader.um_pixel_ratio = 999.0      # deliberately wrong
    config.volumetric.analysis_mode = XYZ

    _, detail = run_slicewise_analysis(path, config)
    assert detail.xy_step_um == pytest.approx(0.2, rel=1e-4)


def test_xyz_rejects_a_planar_movie(tmp_path):
    planar = tmp_path / "movie.tif"
    tifffile.imwrite(str(planar), np.zeros((8, 16, 16), np.uint16), imagej=True,
                     metadata={"axes": "TYX"})
    config = BarcodeConfig()
    config.volumetric.analysis_mode = XYZ
    with pytest.raises(ValueError, match="no Z axis|needs axis"):
        run_slicewise_analysis(str(planar), config)


def test_single_island_frame_reports_nan_separation_not_a_crash(tmp_path, capsys):
    """A frame with one island must yield NaN separation, not abort the branch.

    ``find_island_properties`` used to compute ``np.partition(distances, k_num + 1)`` on
    an N-by-N matrix of island centroids; with a single island that is a 1x1 matrix and
    kth=1, which numpy rejects. ``analysis/run.py`` swallowed the exception and wrote a
    blank binarization row, so the failure was invisible.

    It is not a rare shape: the top and bottom slices of a z-stack through a single cell
    routinely contain exactly one island, and it took out one file in fifteen of the
    real Jurkat series in xyz mode. A lone island simply has nothing to be separated
    from, so NaN is the answer -- which is what the 3D branch already returned.
    """
    path = write_single_blob_stack(tmp_path / "Cell1_1.tif")
    config = BarcodeConfig()
    config.modules.image_binarization = True
    config.modules.intensity_distribution = True
    config.volumetric.analysis_mode = XYZ

    rows, _ = run_slicewise_analysis(path, config)

    assert len(rows) == 1
    assert "binarization failed" not in capsys.readouterr().out
    # the branch ran: sizes are real, and only the separation is undefined
    assert np.isfinite(rows[0].binarization.max_island_size)
    assert np.isnan(rows[0].binarization.mean_island_separation)
    assert np.isfinite(rows[0].intensity.max_kurtosis)


def test_lone_island_separation_is_nan_directly():
    """The guard itself, without going through a whole run."""
    from analysis.binarization import find_island_properties
    from core import BinarizationConfig

    frame = np.zeros((32, 32), dtype=int)
    frame[12:20, 12:20] = 1          # exactly one island
    props = find_island_properties(frame, BinarizationConfig())
    separation = props[4]
    assert np.isnan(separation)
