"""Validation for the 3D optical flow branch.

Organised the same way as ``tests/test_volumetric.py``, because "the code runs" is not
evidence that the code is right:

1. ANALYTIC     — the correct answer is known in closed form (a known translation, the
                  divergence of a radial field, the curl of a solid-body rotation).
2. CROSS-CHECK  — the same quantity computed a second, independent way (the FFT
                  correlation against brute-force circular shifting).
3. INVARIANCE   — properties that must hold whatever the implementation (velocities
                  scale with voxel size, inversely with exposure time).
4. BEHAVIOUR    — documented edge-case handling (short series, reliability and mask
                  gating, degenerate volumes).

A note on what the ANALYTIC tests can fairly assert. The vendored Lucas-Kanade solver
is a *gradient-based* estimator, so it systematically under-reports displacement when
the image's feature scale is comparable to its own smoothing scale ``xyzSig``: measured
gain is ~0.55 for features at 2 px and ~0.92 for features at 6 px, both with the default
``xyzSig=3``. That is a property of the published method, not a defect introduced here,
so these tests pin down direction exactly and magnitude to a documented tolerance.

Run: python -m pytest tests/test_flow_3d.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from analysis.volumetric.binarization import correlation_length_from_radial
from analysis.volumetric.flow import (
    _curl_magnitude_3d,
    _divergence_3d,
    analyze_optical_flow_3d,
    velocity_correlation_3d,
    window_size,
)
from analysis.volumetric.flow_lucas_kanade import calc_flow3D
from core import VolumetricConfig

# Features well above xyzSig=3 so the estimator is in its accurate regime, and realistic
# intensity amplitudes. At near-zero contrast the solver's eps regularisation dominates
# the structure-tensor inverse and every velocity collapses towards zero — a real
# property of the method that the reliability mask is there to catch.
BLUR = 6.0
AMPLITUDE = 1000.0
# Interior only: 'nearest' edge handling in the correlation filters biases the boundary.
CORE = (slice(10, 30), slice(15, 45), slice(15, 45))


def make_series(shift_zyx, n_frames=7, shape=(40, 60, 60), seed=0):
    """A blurred random volume translated by a fixed sub-voxel step each frame."""
    rng = np.random.default_rng(seed)
    base = ndimage.gaussian_filter(rng.normal(size=shape), BLUR)
    base = base / base.std() * AMPLITUDE
    return np.stack([
        ndimage.shift(base, tuple(s * t for s in shift_zyx), order=3, mode="wrap")
        for t in range(n_frames)
    ])


def flow_config(**overrides) -> VolumetricConfig:
    config = VolumetricConfig()
    config.flow_reliability_percentile = 75.0
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def reliable_median(values, rel, quantile=75):
    core_values, core_rel = values[CORE], rel[CORE]
    return float(np.median(core_values[core_rel >= np.percentile(core_rel, quantile)]))


# --------------------------------------------------------------------------------
# 1. ANALYTIC
# --------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "shift_zyx, expect",
    [
        ((0.0, 0.0, 0.5), "x"),
        ((0.0, 0.5, 0.0), "y"),
        ((0.5, 0.0, 0.0), "z"),
    ],
)
def test_translation_is_recovered_on_the_right_axis(shift_zyx, expect):
    """A pure translation shows up on its own axis and nowhere else."""
    vx, vy, vz, rel = calc_flow3D(make_series(shift_zyx), 3, 1, 4)
    recovered = {
        "x": reliable_median(vx, rel),
        "y": reliable_median(vy, rel),
        "z": reliable_median(vz, rel),
    }
    moving = recovered.pop(expect)
    # Gain is below 1 by construction (see the module docstring); direction is exact.
    assert 0.7 * 0.5 <= moving <= 1.05 * 0.5
    for axis, value in recovered.items():
        assert abs(value) < 0.1 * moving, f"leaked {value:.4f} into {axis}"


def test_recovered_speed_is_linear_in_the_true_speed():
    """The estimator's gain is a constant, not a function of displacement."""
    gains = []
    for shift in (0.2, 0.4, 0.8):
        vx, _, _, rel = calc_flow3D(make_series((0.0, 0.0, shift)), 3, 1, 4)
        gains.append(reliable_median(vx, rel) / shift)
    assert max(gains) - min(gains) < 0.1 * np.mean(gains)


def test_divergence_of_a_radial_field_is_three_k():
    """div(k*r) == 3k exactly in three dimensions."""
    k, spacing = 0.7, (0.5, 0.25, 0.25)
    z, y, x = np.meshgrid(
        *(np.arange(n) * d for n, d in zip((12, 16, 16), spacing)), indexing="ij"
    )
    field = k * np.stack([x, y, z], axis=-1)
    assert np.allclose(_divergence_3d(field, spacing), 3 * k)


def test_curl_of_solid_body_rotation_is_twice_omega():
    """curl of a rigid rotation at rate omega has magnitude 2*omega everywhere."""
    omega, spacing = 0.3, (0.5, 0.25, 0.25)
    z, y, x = np.meshgrid(
        *(np.arange(n) * d for n, d in zip((12, 16, 16), spacing)), indexing="ij"
    )
    # Rotation about z: v = omega * (-y, x, 0)
    field = omega * np.stack([-y, x, np.zeros_like(x)], axis=-1)
    assert np.allclose(_curl_magnitude_3d(field, spacing), 2 * omega)
    assert np.allclose(_divergence_3d(field, spacing), 0.0, atol=1e-12)


def test_uniform_flow_has_no_divergence_or_curl():
    spacing = (0.4, 0.2, 0.2)
    field = np.tile(np.array([1.5, -0.5, 0.25]), (10, 12, 12, 1))
    assert np.allclose(_divergence_3d(field, spacing), 0.0, atol=1e-12)
    assert np.allclose(_curl_magnitude_3d(field, spacing), 0.0, atol=1e-12)


def test_correlation_starts_at_one_and_a_uniform_field_never_decays():
    """C(0)=1 by normalisation; a perfectly coherent field stays correlated."""
    field = np.tile(np.array([1.0, 2.0, -1.0]), (16, 16, 16, 1))
    radii, corr = velocity_correlation_3d(field, (1.0, 1.0, 1.0))
    assert radii.size > 0
    assert corr[0] == pytest.approx(1.0, abs=0.02)
    assert np.nanmin(corr) > 0.5


# --------------------------------------------------------------------------------
# 2. CROSS-CHECK
# --------------------------------------------------------------------------------

def test_fft_correlation_matches_brute_force_shifting():
    """The FFT form equals the explicit shift-and-dot the 2D branch uses.

    ``utils.optical_flow.velocity_correlation`` loops over every shift; that is O(N^2)
    and hopeless on a volume, so ``velocity_correlation_3d`` uses Wiener-Khinchin. The
    FFT is circular, so the brute-force oracle rolls rather than crops.
    """
    rng = np.random.default_rng(3)
    field = ndimage.gaussian_filter(rng.normal(size=(12, 12, 12, 3)), (2, 2, 2, 0))

    mean_square = np.mean(np.sum(field ** 2, axis=-1))
    n = field[..., 0].size
    brute = np.zeros(field.shape[:3])
    for dz in range(12):
        for dy in range(12):
            for dx in range(12):
                shifted = np.roll(field, (-dz, -dy, -dx), axis=(0, 1, 2))
                brute[dz, dy, dx] = np.sum(field * shifted) / (n * mean_square)
    brute = np.fft.fftshift(brute)

    # Reproduce the module's own binning on the oracle so only the correlation itself,
    # not the histogramming, is under test.
    radius = np.sqrt(sum(
        ((np.arange(12) - 6) ** 2).reshape([-1 if i == a else 1 for i in range(3)])
        for a in range(3)
    ))
    edges = np.arange(0, 6.0, 1.0)
    counts = np.histogram(radius, edges)[0]
    expected = np.histogram(radius, edges, weights=brute)[0] / np.maximum(counts, 1)

    _, actual = velocity_correlation_3d(field, (1.0, 1.0, 1.0))
    assert np.allclose(actual[: len(expected)], expected, atol=1e-10)


def test_window_size_matches_the_solvers_own_requirement():
    """window_size must be exactly what calc_flow3D refuses to run below."""
    for t_sigma in (1, 2, 3):
        size = window_size(t_sigma)
        assert size >= 6 * t_sigma + 1
        assert size % 2 == 1
        # One frame short must be rejected by the solver (it calls sys.exit).
        with pytest.raises(SystemExit):
            calc_flow3D(np.zeros((6 * t_sigma, 4, 4, 4)), 3, t_sigma, 4)


# --------------------------------------------------------------------------------
# 3. INVARIANCE
# --------------------------------------------------------------------------------

def test_speed_scales_with_voxel_size():
    """Same voxels, twice the physical spacing: twice the speed."""
    series = make_series((0.0, 0.0, 0.5))
    config = flow_config()
    fine, _ = analyze_optical_flow_3d(series, (1.0, 1.0, 1.0), 1.0, config, [3], None)
    coarse, _ = analyze_optical_flow_3d(series, (2.0, 2.0, 2.0), 1.0, config, [3], None)
    assert coarse.mean_speed == pytest.approx(2 * fine.mean_speed, rel=1e-9)


def test_correlation_length_scales_with_voxel_size():
    """Twice the spacing, twice the correlation length.

    Driven from a synthetic decaying field rather than from a translating series: a rigid
    translation is coherent across the whole volume, so its correlation never crosses the
    threshold and the length is legitimately NaN (see the flag-4 test below).
    """
    rng = np.random.default_rng(11)
    field = ndimage.gaussian_filter(rng.normal(size=(24, 24, 24, 3)), (3, 3, 3, 0))

    fine_r, fine_c = velocity_correlation_3d(field, (1.0, 1.0, 1.0))
    coarse_r, coarse_c = velocity_correlation_3d(field, (2.0, 2.0, 2.0))
    fine = correlation_length_from_radial(fine_c, fine_r, 0.5)
    coarse = correlation_length_from_radial(coarse_c, coarse_r, 0.5)
    assert np.isfinite(fine)
    assert coarse == pytest.approx(2 * fine, rel=1e-9)


def test_configured_frame_interval_overrides_the_file_and_scales_speed():
    """The interval is an input; a configured value must win over file metadata.

    Guards the concrete failure this was written for: a stack whose ImageJ 'finterval'
    says 1 s when timepoints are really 60 s apart reports speeds 60x too fast.
    """
    from analysis.volumetric.reader import VolumeStack
    from analysis.volumetric.run import resolve_frame_interval

    stack = VolumeStack(
        data=np.zeros((1, 2, 2, 2)), z_step_um=1.0, xy_step_um=1.0,
        exposure_time_s=1.0, axes="TZYX", source_path="x.tif",
    )
    config = flow_config()
    assert resolve_frame_interval(stack, config) == 1.0  # falls back to the file

    config.frame_interval_s = 60.0
    assert resolve_frame_interval(stack, config) == 60.0  # configured value wins

    series = make_series((0.0, 0.0, 0.5))
    per_frame, _ = analyze_optical_flow_3d(series, (1.0,) * 3, 1.0, config, [3], None)
    per_minute, detail = analyze_optical_flow_3d(series, (1.0,) * 3, 60.0, config, [3], None)
    assert detail.frame_interval_s == 60.0
    assert per_minute.mean_speed == pytest.approx(per_frame.mean_speed / 60, rel=1e-9)


def test_speed_is_inverse_in_exposure_time():
    series = make_series((0.0, 0.0, 0.5))
    config = flow_config()
    fast, _ = analyze_optical_flow_3d(series, (1.0, 1.0, 1.0), 1.0, config, [3], None)
    slow, _ = analyze_optical_flow_3d(series, (1.0, 1.0, 1.0), 2.0, config, [3], None)
    assert slow.mean_speed == pytest.approx(fast.mean_speed / 2, rel=1e-9)


def test_divergence_and_curl_are_invariant_to_grid_rotation():
    """Divergence is a scalar and ||curl|| a pseudovector norm; a proper rotation of the
    frame must leave both alone.

    The rotation used is the cyclic relabelling (z, y, x) -> (y, x, z), which is an even
    permutation and therefore a genuine rotation. A plain axis *transpose* would be a
    reflection, which flips the sign of the curl vector, so it is not a valid check.
    """
    rng = np.random.default_rng(5)
    field = ndimage.gaussian_filter(rng.normal(size=(16, 16, 16, 3)), (2, 2, 2, 0))
    spacing = (1.0, 1.0, 1.0)  # must be isotropic, or the frames are not equivalent

    rotated = np.transpose(field, (1, 2, 0, 3))
    # Components follow the same relabelling: x' <- z, y' <- x, z' <- y.
    rotated = np.stack(
        [rotated[..., 2], rotated[..., 0], rotated[..., 1]], axis=-1
    )

    assert np.nanmean(_curl_magnitude_3d(rotated, spacing)) == pytest.approx(
        np.nanmean(_curl_magnitude_3d(field, spacing)), rel=1e-9
    )
    assert np.nanmean(_divergence_3d(rotated, spacing)) == pytest.approx(
        np.nanmean(_divergence_3d(field, spacing)), rel=1e-9
    )


def test_results_are_deterministic():
    series = make_series((0.0, 0.0, 0.5))
    config = flow_config()
    first, _ = analyze_optical_flow_3d(series, (1.0, 1.0, 1.0), 1.0, config, [3], None)
    second, _ = analyze_optical_flow_3d(series, (1.0, 1.0, 1.0), 1.0, config, [3], None)
    assert first.get_data() == second.get_data()


# --------------------------------------------------------------------------------
# 4. BEHAVIOUR
# --------------------------------------------------------------------------------

def test_short_series_returns_nan_rather_than_raising():
    """Fewer than 6*t_sigma+1 timepoints: skip the branch, do not approximate it."""
    series = make_series((0.0, 0.0, 0.5), n_frames=5, shape=(8, 16, 16))
    results, detail = analyze_optical_flow_3d(
        series, (1.0, 1.0, 1.0), 1.0, flow_config(), [0, 2, 4], None
    )
    assert detail.window_size == 7
    assert detail.centres == []
    assert all(np.isnan(v) for v in results.get_data())


def test_centres_without_a_full_window_are_dropped():
    """A centre closer to an end than half a window has no window and is skipped."""
    series = make_series((0.0, 0.0, 0.5), n_frames=9, shape=(12, 24, 24))
    _, detail = analyze_optical_flow_3d(
        series, (1.0, 1.0, 1.0), 1.0, flow_config(), [0, 4, 8], None
    )
    assert detail.centres == [4]
    assert detail.skipped_centres == [0, 8]


def test_single_window_gives_nan_speed_change_not_zero():
    """One window would compare a value against itself, as in the 2D-vs-3D convention."""
    series = make_series((0.0, 0.0, 0.5))
    results, detail = analyze_optical_flow_3d(
        series, (1.0, 1.0, 1.0), 1.0, flow_config(), [3], None
    )
    assert len(detail.centres) == 1
    assert np.isnan(results.delta_speed)
    assert np.isfinite(results.mean_speed)


def test_reliability_percentile_controls_how_many_voxels_are_used():
    series = make_series((0.0, 0.0, 0.5))
    _, none = analyze_optical_flow_3d(
        series, (1.0, 1.0, 1.0), 1.0, flow_config(flow_reliability_percentile=0.0), [3], None
    )
    _, half = analyze_optical_flow_3d(
        series, (1.0, 1.0, 1.0), 1.0, flow_config(flow_reliability_percentile=50.0), [3], None
    )
    assert none.valid_fractions[0] == pytest.approx(1.0)
    assert half.valid_fractions[0] == pytest.approx(0.5, abs=0.02)


def test_mask_restricts_the_metrics_and_only_when_asked():
    series = make_series((0.0, 0.0, 0.5))
    masks = np.zeros(series.shape, dtype=bool)
    masks[:, 10:30, 15:45, 15:45] = True

    config = flow_config(flow_reliability_percentile=0.0, flow_use_mask=True)
    _, masked = analyze_optical_flow_3d(series, (1.0, 1.0, 1.0), 1.0, config, [3], masks)
    assert masked.used_mask
    assert masked.valid_fractions[0] == pytest.approx(masks[0].mean(), abs=1e-9)

    config.flow_use_mask = False
    _, unmasked = analyze_optical_flow_3d(series, (1.0, 1.0, 1.0), 1.0, config, [3], masks)
    assert not unmasked.used_mask
    assert unmasked.valid_fractions[0] == pytest.approx(1.0)


def test_reliability_masking_does_not_corrupt_the_spatial_operators():
    """Divergence, curl and correlation must be computed before the mask is applied.

    A smooth translating field has near-zero divergence and curl everywhere. If the
    masked-out voxels were punched to zero *before* differentiating, the scattered holes
    left by a 50th-percentile reliability cut would create large spurious gradients at
    every hole edge, and curl would jump by orders of magnitude relative to the unmasked
    run. Here the mask changes which voxels are averaged, not what is differentiated, so
    the two stay in the same ballpark.
    """
    series = make_series((0.0, 0.0, 0.5))
    unmasked, _ = analyze_optical_flow_3d(
        series, (1.0, 1.0, 1.0), 1.0, flow_config(flow_reliability_percentile=0.0), [3], None
    )
    masked, detail = analyze_optical_flow_3d(
        series, (1.0, 1.0, 1.0), 1.0, flow_config(flow_reliability_percentile=50.0), [3], None
    )
    assert detail.valid_fractions[0] == pytest.approx(0.5, abs=0.02)
    assert masked.curl == pytest.approx(unmasked.curl, rel=1.0)
    # The correlation reads the full field, so the reliability cut cannot touch it at all.
    assert masked.velocity_correlation_flag == unmasked.velocity_correlation_flag
    assert np.array_equal(
        [masked.velocity_correlation_length], [unmasked.velocity_correlation_length],
        equal_nan=True,
    )


def test_coherent_flow_exceeds_the_field_of_view_and_raises_flag_4():
    """A rigid translation stays correlated everywhere, so there is no crossing to find.

    NaN plus flag 4 ("velocity correlation length > field of view") is the honest report.
    This is also the regression guard for masking-before-differentiation: punching the
    reliability holes out first would decorrelate the field artificially and yield a
    confident, wrong, finite length here.
    """
    series = make_series((0.0, 0.0, 0.5))
    results, _ = analyze_optical_flow_3d(
        series, (1.0, 1.0, 1.0), 1.0, flow_config(), [3], None
    )
    assert np.isfinite(results.mean_speed)
    assert np.isnan(results.velocity_correlation_length)
    assert results.velocity_correlation_flag == 1


def test_static_series_reports_near_zero_speed():
    """Nothing moves, so nothing should be reported as moving."""
    still = np.repeat(make_series((0.0, 0.0, 0.0), n_frames=1)[0][None], 7, axis=0)
    results, _ = analyze_optical_flow_3d(
        still, (1.0, 1.0, 1.0), 1.0, flow_config(), [3], None
    )
    assert results.mean_speed < 1e-6


def test_constant_volume_degrades_gracefully():
    """No texture at all: no crash, no correlation length, and flag 4 raised."""
    flat = np.full((7, 12, 24, 24), 500.0)
    results, detail = analyze_optical_flow_3d(
        flat, (1.0, 1.0, 1.0), 1.0, flow_config(), [3], None
    )
    assert detail.centres == [3]
    assert np.isnan(results.velocity_correlation_length)
    assert results.velocity_correlation_flag == 1


def test_fully_masked_out_window_does_not_warn_or_raise():
    series = make_series((0.0, 0.0, 0.5), n_frames=7, shape=(12, 24, 24))
    masks = np.zeros(series.shape, dtype=bool)
    config = flow_config(flow_reliability_percentile=0.0, flow_use_mask=True)
    results, detail = analyze_optical_flow_3d(
        series, (1.0, 1.0, 1.0), 1.0, config, [3], masks
    )
    assert detail.valid_fractions == [0.0]
    # Every metric must be NaN. In particular the directional spread must not come back
    # as a large finite number: an empty window has no direction, and a resultant length
    # of 0 would otherwise be indistinguishable from genuinely isotropic flow.
    assert all(np.isnan(v) for v in results.get_data())


def test_downsampling_keeps_the_direction_and_the_physical_scale():
    """Block-averaging trades resolution for speed; it must not change what is reported."""
    series = make_series((0.0, 0.0, 0.5), shape=(24, 48, 48))
    full, _ = analyze_optical_flow_3d(
        series, (0.5, 0.5, 0.5), 1.0, flow_config(), [3], None
    )
    down, detail = analyze_optical_flow_3d(
        series, (0.5, 0.5, 0.5), 1.0, flow_config(flow_downsample=2), [3], None
    )
    assert detail.spacing_zyx_um == (1.0, 1.0, 1.0)
    # The motion is pure +x, so the azimuth should sit near 0 either way. Its *sign* is
    # pure noise at that magnitude, which is why this checks the angle, not the sign.
    assert abs(full.mean_theta) < 0.15
    assert abs(down.mean_theta) < 0.15
    assert down.mean_speed == pytest.approx(full.mean_speed, rel=0.5)
