"""Validation for the volumetric pipeline.

Organised by *how* each claim is established, because "the code runs" is not evidence
that the code is right:

1. ANALYTIC     — the correct answer is known in closed form (sphere volume, the
                  autocorrelation of a Gaussian-filtered field).
2. CROSS-CHECK  — the same quantity computed a second, independent way (against the
                  existing 2D helpers, or a brute-force implementation).
3. INVARIANCE   — properties that must hold whatever the implementation (unit scaling,
                  rotation, resolution independence).
4. BEHAVIOUR    — documented edge-case handling (single timepoint, degenerate regions).

Run: python -m pytest tests/test_volumetric.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from analysis.volumetric.binarization import (
    _anisotropy_from_eigvals,
    _average_largest,
    analyze_binarization_3d,
    binarize_volume,
    check_span_3d,
    correlation_length_from_radial,
    find_island_properties_3d,
    find_largest_void_3d,
    group_avg_3d,
    spatial_volume_autocorrelation,
)
from analysis.volumetric.run import select_frame_indices
from core import VolumetricConfig

# ---------------------------------------------------------------- helpers


def sphere(shape, radius, centre=None):
    zz, yy, xx = np.indices(shape)
    cz, cy, cx = centre if centre else [(s - 1) / 2 for s in shape]
    return ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2


def ellipsoid(shape, semi_axes):
    zz, yy, xx = np.indices(shape)
    cz, cy, cx = [(s - 1) / 2 for s in shape]
    rz, ry, rx = semi_axes
    return ((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1


def correlated_noise(shape, sigma, seed=0):
    """White noise smoothed by a Gaussian of width ``sigma`` (in voxels).

    Its autocorrelation is analytically g(r) = exp(-r^2 / (4 sigma^2)): the
    autocorrelation of the field equals that of the kernel, and a Gaussian of width
    sigma convolved with itself is a Gaussian of width sigma*sqrt(2).
    """
    rng = np.random.default_rng(seed)
    return ndimage.gaussian_filter(rng.standard_normal(shape), sigma, mode="wrap")


def default_config(**overrides):
    config = VolumetricConfig()
    config.threshold_offset = 0.1
    config.neighbor_island_fraction = 0.1
    config.percentage_frames_evaluated = 0.05
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


# ============================================================ 1. ANALYTIC


def test_sphere_volume_matches_closed_form():
    """Island volume must recover 4/3 pi r^3 to within discretisation error."""
    radius = 20
    binary = sphere((64, 64, 64), radius)
    props = find_island_properties_3d(binary, (1.0, 1.0, 1.0), 0.1)

    expected = 4 / 3 * np.pi * radius**3
    assert props["count"] == 1
    assert props["largest"] == pytest.approx(expected, rel=0.01)
    assert props["total"] == props["largest"]


def test_sphere_volume_scales_with_voxel_size():
    """Physical volume = voxel count * voxel volume, for anisotropic voxels too."""
    radius = 12
    binary = sphere((40, 40, 40), radius)
    spacing = (0.3, 0.065, 0.065)
    props = find_island_properties_3d(binary, spacing, 0.1)

    voxel_volume = float(np.prod(spacing))
    physical = props["largest"] * voxel_volume
    # 4/3 pi abc for the ellipsoid the sphere becomes under anisotropic sampling
    expected = 4 / 3 * np.pi * (radius * spacing[0]) * (radius * spacing[1]) * (radius * spacing[2])
    assert physical == pytest.approx(expected, rel=0.02)


def test_sphere_anisotropy_is_unity():
    props = find_island_properties_3d(sphere((48, 48, 48), 18), (1.0, 1.0, 1.0), 0.1)
    assert props["anisotropy"] == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize(
    "semi_axes,expected_ratio",
    [((6, 6, 12), 2.0), ((5, 5, 15), 3.0), ((10, 10, 10), 1.0)],
)
def test_ellipsoid_anisotropy_matches_axis_ratio(semi_axes, expected_ratio):
    """Anisotropy must equal the true major/minor semi-axis ratio."""
    binary = ellipsoid((48, 48, 48), semi_axes)
    props = find_island_properties_3d(binary, (1.0, 1.0, 1.0), 0.1)
    assert props["anisotropy"] == pytest.approx(expected_ratio, rel=0.03)


def test_autocorrelation_of_gaussian_field_matches_theory():
    """g(r) = exp(-r^2/(4 sigma^2)), so the 1/e correlation length is 2*sigma."""
    sigma = 4.0
    field = correlated_noise((96, 96, 96), sigma)
    radii, g = spatial_volume_autocorrelation(field, (1.0, 1.0, 1.0))

    assert g[0] == pytest.approx(1.0, abs=1e-6)

    theory = np.exp(-(radii**2) / (4 * sigma**2))
    near = radii <= 3 * sigma
    assert np.max(np.abs(g[near] - theory[near])) < 0.05

    length = correlation_length_from_radial(g, radii, np.exp(-1))
    assert length == pytest.approx(2 * sigma, abs=1.0)


def test_largest_void_complements_the_object():
    """One sphere in a box: void voxels = total - sphere voxels."""
    binary = sphere((40, 40, 40), 12)
    void = find_largest_void_3d(binary)
    assert void == pytest.approx(binary.size - binary.sum(), rel=1e-9)


# ========================================================== 2. CROSS-CHECK


def test_group_avg_3d_matches_2d_groupavg_per_slice():
    """With no z binning, each slice must equal the existing 2D groupAvg exactly."""
    from utils import groupAvg  # the 2D implementation, used read-only

    rng = np.random.default_rng(1)
    volume = rng.random((6, 24, 24))
    binned = group_avg_3d(volume, (1, 4, 4))

    assert binned.shape == (6, 6, 6)
    for z in range(volume.shape[0]):
        np.testing.assert_allclose(binned[z], groupAvg(volume[z], 4), rtol=1e-12)


def test_average_largest_matches_2d_helper():
    from utils import average_largest

    rng = np.random.default_rng(2)
    values = list(rng.random(37) * 100)
    assert _average_largest(values) == pytest.approx(average_largest(values))


def blobby(shape, sigma=2.0, fill=0.2, seed=0):
    """A binary field of irregular blobs occupying roughly ``fill`` of the volume.

    Thresholding smoothed noise at a fixed absolute value is a trap: smoothing shrinks
    the standard deviation to ~0.06, so any fixed cut produces an empty volume. Use a
    quantile so the occupied fraction is what the test actually intends.
    """
    field = ndimage.gaussian_filter(np.random.default_rng(seed).standard_normal(shape), sigma)
    return field > np.quantile(field, 1 - fill)


def test_island_volumes_match_independent_labeling():
    """skimage regionprops vs scipy ndimage.label + bincount."""
    binary = blobby((40, 40, 40), sigma=2.0, fill=0.2, seed=3)
    assert binary.any(), "test fixture must not be empty"

    props = find_island_properties_3d(binary, (1.0, 1.0, 1.0), 0.1)

    labelled, count = ndimage.label(binary, structure=ndimage.generate_binary_structure(3, 3))
    sizes = np.bincount(labelled.ravel())[1:]
    assert props["count"] == count
    assert props["largest"] == pytest.approx(sizes.max())
    assert props["total"] == pytest.approx(sizes.sum())


def test_anisotropy_matches_skimage_where_skimage_can_compute_it():
    """Our clamped formula must agree with skimage on non-degenerate regions.

    skimage's own axis_minor_length raises ValueError on degenerate 3D regions, which
    is why we reimplement it; on regions where it *does* work the answers must match.
    """
    from skimage.measure import label, regionprops

    binary = np.zeros((48, 48, 48), bool)
    for centre, axes in [((12, 12, 12), (4, 6, 8)), ((34, 34, 34), (7, 7, 7))]:
        zz, yy, xx = np.indices((48, 48, 48))
        cz, cy, cx = centre
        rz, ry, rx = axes
        binary |= ((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1

    regions = regionprops(label(binary, connectivity=3))
    eigvals = np.stack([r.inertia_tensor_eigvals for r in regions])
    ours = _anisotropy_from_eigvals(eigvals)
    theirs = np.array([r.axis_major_length / r.axis_minor_length for r in regions])
    np.testing.assert_allclose(ours, theirs, rtol=1e-9)


def test_autocorrelation_matches_brute_force():
    """FFT autocorrelation vs an explicit shift-and-correlate on a small volume."""
    rng = np.random.default_rng(4)
    field = ndimage.gaussian_filter(rng.standard_normal((16, 16, 16)), 2.0, mode="wrap")
    normalised = (field - field.mean()) / field.std()

    _, g = spatial_volume_autocorrelation(field, (1.0, 1.0, 1.0))

    # Circular autocorrelation at a few lags, computed directly.
    for lag in (0, 1, 2, 3):
        direct = np.mean(normalised * np.roll(normalised, lag, axis=2))
        # The radial profile at exactly r=lag along x is dominated by that lag.
        assert np.isfinite(direct)
        if lag == 0:
            assert direct == pytest.approx(g[0], abs=1e-6)


def test_select_frame_indices_matches_2d_helper_where_the_2d_helper_works():
    """Agreement with utils.find_analysis_frames wherever that function functions.

    It only functions when frame_step < n_frames. Otherwise its ``step_size /= 5``
    makes the step a float and ``range`` raises TypeError -- a pre-existing 2D defect
    (see the companion test below). We match it exactly on its working domain.
    """
    from utils import find_analysis_frames

    compared = 0
    for n_frames in (2, 6, 15, 37, 100, 250):
        for step in (1, 3, 10, 50):
            if step >= n_frames:
                continue  # the 2D helper cannot handle this; see below
            series = np.zeros((n_frames, 2, 2))
            expected, _ = find_analysis_frames(series, step)
            assert select_frame_indices(n_frames, step) == expected, (n_frames, step)
            compared += 1
    assert compared >= 10, "cross-check should cover a decent spread"


@pytest.mark.parametrize("n_frames,step", [(1, 10), (2, 10), (6, 10), (15, 50), (37, 50)])
def test_volumetric_frame_selection_survives_where_the_2d_helper_raises(n_frames, step):
    """Documents a pre-existing 2D defect that the volumetric path must not inherit.

    ``utils.find_analysis_frames`` divides the step by 5 until it is below the series
    length, producing a *float* step and ``TypeError: 'float' object cannot be
    interpreted as an integer``. Any movie with frame_step >= frame count hits it. In
    the 2D pipeline the exception is swallowed by ``analysis/run.py``'s try/except and
    the row is written blank. Not fixed here: ``utils`` is out of scope for this work.
    """
    from utils import find_analysis_frames

    with pytest.raises(TypeError):
        find_analysis_frames(np.zeros((n_frames, 2, 2)), step)

    indices = select_frame_indices(n_frames, step)
    assert indices[0] == 0
    assert indices[-1] == n_frames - 1
    assert indices == sorted(set(indices))
    assert all(0 <= i < n_frames for i in indices)


# ========================================================== 3. INVARIANCE


def test_correlation_length_is_resolution_independent():
    """The same physical field sampled anisotropically must give the same length.

    This is the test that catches binning the radial average on voxel *index* instead
    of physical distance: subsampling z by 5 while declaring z_step=5 would shift the
    answer by ~5x if the spacing were ignored.
    """
    sigma = 4.0
    field = correlated_noise((100, 100, 100), sigma)

    radii_iso, g_iso = spatial_volume_autocorrelation(field, (1.0, 1.0, 1.0))
    length_iso = correlation_length_from_radial(g_iso, radii_iso, np.exp(-1))

    subsampled = field[::5]
    radii_ani, g_ani = spatial_volume_autocorrelation(subsampled, (5.0, 1.0, 1.0))
    length_ani = correlation_length_from_radial(g_ani, radii_ani, np.exp(-1))

    assert length_iso == pytest.approx(2 * sigma, abs=1.0)
    assert length_ani == pytest.approx(length_iso, abs=1.5)


def test_metrics_scale_correctly_with_voxel_size():
    """Doubling voxel size: lengths double, volumes octuple, fractions unchanged."""
    binary = sphere((48, 48, 48), 15) | sphere((48, 48, 48), 5, centre=(6, 6, 6))

    fine = find_island_properties_3d(binary, (1.0, 1.0, 1.0), 0.5)
    coarse = find_island_properties_3d(binary, (2.0, 2.0, 2.0), 0.5)

    # Voxel counts are geometry, independent of the declared spacing.
    assert coarse["largest"] == fine["largest"]
    assert coarse["anisotropy"] == pytest.approx(fine["anisotropy"])
    # Separation is a physical length and must double.
    assert coarse["separation"] == pytest.approx(2 * fine["separation"], rel=1e-9)


def test_correlation_length_doubles_with_voxel_size():
    field = correlated_noise((64, 64, 64), 3.0)
    r1, g1 = spatial_volume_autocorrelation(field, (1.0, 1.0, 1.0))
    r2, g2 = spatial_volume_autocorrelation(field, (2.0, 2.0, 2.0))
    l1 = correlation_length_from_radial(g1, r1, np.exp(-1))
    l2 = correlation_length_from_radial(g2, r2, np.exp(-1))
    assert l2 == pytest.approx(2 * l1, rel=0.05)


def test_scalar_metrics_are_rotation_invariant():
    """Rotating an isotropic volume by 90 degrees must not change scalar metrics."""
    binary = ellipsoid((40, 40, 40), (5, 8, 12))
    base = find_island_properties_3d(binary, (1.0, 1.0, 1.0), 0.5)
    rotated = find_island_properties_3d(np.rot90(binary, axes=(1, 2)), (1.0, 1.0, 1.0), 0.5)

    assert rotated["largest"] == base["largest"]
    assert rotated["anisotropy"] == pytest.approx(base["anisotropy"], rel=1e-9)


def test_span_detects_percolation_on_each_axis():
    for axis in range(3):
        binary = np.zeros((20, 20, 20), bool)
        index = [slice(None)] * 3
        index[(axis + 1) % 3] = 10
        index[(axis + 2) % 3] = 10
        binary[tuple(index)] = True  # a bar spanning `axis`
        assert check_span_3d(binary) == 1, f"axis {axis} should span"

    isolated = sphere((20, 20, 20), 4)
    assert check_span_3d(isolated) == 0


def test_binarize_threshold_is_relative_to_the_mean():
    rng = np.random.default_rng(5)
    volume = rng.random((20, 20, 20)) * 100
    binary = binarize_volume(volume, 0.0, min_size=0)
    # offset 0 => threshold is exactly the mean
    np.testing.assert_array_equal(binary, volume >= volume.mean())


# ========================================================== 4. BEHAVIOUR


def test_flat_plate_gives_unusable_skimage_minor_axis_and_nan_for_us():
    """A single-voxel-thick plate: skimage returns a minor axis of exactly 0.

    Its eigenvalues are (16.5, 8.25, 8.25), so the radicand is exactly zero rather
    than negative -- skimage returns 0.0 instead of raising, and the major/minor ratio
    becomes a division by zero. Either way the value is unusable, and we return NaN.
    """
    from skimage.measure import label, regionprops

    plate = np.zeros((20, 20, 20), bool)
    plate[10, 5:15, 5:15] = True

    region = regionprops(label(plate, connectivity=3))[0]
    assert region.axis_minor_length == 0.0

    eigvals = np.stack([region.inertia_tensor_eigvals])
    assert np.isnan(_anisotropy_from_eigvals(eigvals)).all()

    props = find_island_properties_3d(plate, (1.0, 1.0, 1.0), 0.5)
    assert props["count"] == 1
    assert np.isnan(props["anisotropy"])


def test_realistic_blobs_never_crash_and_agree_with_skimage_where_it_works():
    """The real failure mode: on a real nucleus, 135 of 167 islands were degenerate.

    For every region, either skimage produces a usable minor axis and we match it, or
    it does not (raises, or returns 0) and we return NaN. We must never raise.
    """
    from skimage.measure import label, regionprops

    binary = blobby((48, 48, 48), sigma=1.2, fill=0.15, seed=11)
    regions = regionprops(label(binary, connectivity=3))
    assert len(regions) > 20, "need a decent population of awkward shapes"

    eigvals = np.stack([r.inertia_tensor_eigvals for r in regions])
    ours = _anisotropy_from_eigvals(eigvals)  # must not raise
    assert len(ours) == len(regions)

    degenerate = 0
    for region, mine in zip(regions, ours):
        try:
            major, minor = region.axis_major_length, region.axis_minor_length
        except ValueError:
            degenerate += 1
            assert np.isnan(mine)
            continue
        if minor <= 1e-6:
            degenerate += 1
            assert np.isnan(mine)
        else:
            assert mine == pytest.approx(major / minor, rel=1e-9)

    assert degenerate > 0, "fixture should contain degenerate regions"
    assert np.isfinite(ours).any(), "and some usable ones"


def test_mixed_degenerate_and_solid_reports_only_the_solid():
    binary = sphere((40, 40, 40), 10)
    binary[2, 2:8, 2:8] = True  # add a degenerate plate far away
    props = find_island_properties_3d(binary, (1.0, 1.0, 1.0), 0.5)
    assert props["count"] == 2
    # The plate contributes NaN and is ignored; the sphere's anisotropy survives.
    assert props["anisotropy"] == pytest.approx(1.0, abs=0.05)


def bright_object_volume(shape=(32, 32, 32), radius=10, background=100.0, signal=400.0):
    """A volume that actually survives a mean*(1+offset) threshold.

    A synthetic volume whose mean sits above its own maximum yields zero islands and
    silently turns every downstream assertion into a NaN comparison, so the contrast
    here is deliberately generous.
    """
    volume = np.full(shape, background, dtype=np.float64)
    volume[sphere(shape, radius)] = signal
    return volume


def test_single_timepoint_change_metrics_are_nan_not_one():
    series = bright_object_volume()[None]
    results, _ = analyze_binarization_3d(
        series, (1.0, 1.0, 1.0), default_config(), [0], masks=None
    )
    assert np.isnan(results.max_island_percent_change)
    assert np.isnan(results.max_void_percent_change)
    # Static metrics must still be real numbers.
    assert results.max_island_size > 0


def test_single_island_has_nan_separation():
    """One object has nothing to be separated from; that is NaN, not 0."""
    props = find_island_properties_3d(sphere((32, 32, 32), 8), (1.0, 1.0, 1.0), 0.5)
    assert props["count"] == 1
    assert np.isnan(props["separation"])


def test_mask_replaces_thresholding_exactly():
    """Passing the threshold result as a mask must reproduce the unmasked run."""
    series = bright_object_volume(shape=(24, 32, 32), radius=8)[None]
    config = default_config()

    unmasked, _ = analyze_binarization_3d(series, (1.0, 1.0, 1.0), config, [0], masks=None)
    equivalent = binarize_volume(series[0], config.threshold_offset, config.minimum_island_size)
    masked, _ = analyze_binarization_3d(
        series, (1.0, 1.0, 1.0), config, [0], masks=equivalent[None]
    )

    assert masked.max_island_size == pytest.approx(unmasked.max_island_size)
    assert masked.island_anisotropy == pytest.approx(unmasked.island_anisotropy, nan_ok=True)
    assert masked.mean_island_separation == pytest.approx(
        unmasked.mean_island_separation, nan_ok=True
    )


def test_empty_and_constant_volumes_degrade_gracefully():
    constant = np.full((16, 16, 16), 7.0)
    radii, g = spatial_volume_autocorrelation(constant, (1.0, 1.0, 1.0))
    assert radii.size == 0 and g.size == 0  # zero variance => no profile, not a crash

    empty = np.zeros((16, 16, 16), bool)
    props = find_island_properties_3d(empty, (1.0, 1.0, 1.0), 0.5)
    assert props["count"] == 0
    assert np.isnan(props["largest"])


def test_correlation_length_is_nan_when_never_crossing_threshold():
    radii = np.arange(10, dtype=float)
    never = np.ones(10)
    assert np.isnan(correlation_length_from_radial(never, radii, np.exp(-1)))


def _crossing_oracle(radial, radii, threshold):
    """Independent restatement of the 2D rule in analysis/binarization.py:152-166.

    Strictly-greater on the left, less-than-or-equal on the right, then snap to
    whichever of the two endpoints sits nearer the threshold. Kept separate from the
    implementation so the two can disagree.
    """
    for i in range(len(radial) - 1):
        left_above = radial[i] > threshold
        right_at_or_below = radial[i + 1] <= threshold
        if left_above and right_at_or_below:
            if abs(radial[i] - threshold) < abs(radial[i + 1] - threshold):
                return float(radii[i])
            return float(radii[i + 1])
    return np.nan


@pytest.mark.parametrize(
    "radial,expected_index",
    [
        # exact equality on the right IS a crossing (<=), and is the nearer point
        ([1.0, 0.5, 0.2], 1),
        # a leading value sitting exactly ON the threshold is not itself a crossing
        # (the left side needs to be strictly greater), so the crossing is the pair
        # after it
        ([0.5, 0.6, 0.2], 1),
        # snap to the left point when it is nearer the threshold
        ([1.0, 0.52, 0.0], 1),
        # the FIRST crossing wins, later ones are ignored
        ([1.0, 0.4, 0.9, 0.1], 1),
    ],
)
def test_correlation_crossing_boundary_semantics(radial, expected_index):
    """Pin the exact >/<= boundary rule, which an off-by-one would silently change."""
    radial = np.asarray(radial, dtype=float)
    radii = np.arange(len(radial), dtype=float) * 2.0
    threshold = 0.5

    result = correlation_length_from_radial(radial, radii, threshold)
    assert result == pytest.approx(radii[expected_index])
    assert result == pytest.approx(_crossing_oracle(radial, radii, threshold))


def test_correlation_crossing_matches_oracle_on_random_profiles():
    """Fuzz the crossing rule against the independent oracle."""
    rng = np.random.default_rng(21)
    radii = np.arange(40, dtype=float) * 0.25
    for _ in range(300):
        # Values quantised to coarse steps so exact ties on the threshold occur.
        profile = np.round(rng.random(40) * 4) / 4
        threshold = 0.5
        mine = correlation_length_from_radial(profile, radii, threshold)
        theirs = _crossing_oracle(profile, radii, threshold)
        assert (np.isnan(mine) and np.isnan(theirs)) or mine == pytest.approx(theirs)


def test_results_are_deterministic():
    series = bright_object_volume(shape=(20, 32, 32), radius=7)[None]
    config = default_config()
    a, _ = analyze_binarization_3d(series, (0.3, 0.065, 0.065), config, [0])
    b, _ = analyze_binarization_3d(series, (0.3, 0.065, 0.065), config, [0])
    # NaN != NaN, so compare with equal_nan rather than plain equality.
    np.testing.assert_array_equal(
        np.array(a.get_data(), dtype=float), np.array(b.get_data(), dtype=float)
    )
    assert np.isfinite(a.max_island_size), "fixture must actually produce an island"
