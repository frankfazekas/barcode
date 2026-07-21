"""3D structural metrics — the volumetric counterpart of ``analysis/binarization.py``.

Deliberately a separate implementation sharing no code with the 2D branch, so that
nothing here can change 2D results. It mirrors the 2D module's structure function for
function to keep the two readable side by side.

Three things genuinely differ in 3D, and each was found by running against real data:

* ``regionprops``' ``axis_major_length`` / ``axis_minor_length`` raise
  ``ValueError: math domain error`` on flat or thin 3D regions (the inertia-eigenvalue
  formula goes negative). On a real nucleus 135 of 167 islands were degenerate, so
  anisotropy is computed from clamped ``inertia_tensor_eigvals`` with degenerate
  regions excluded rather than via those properties.
* The correlation radius is bounded by the *physical* half-extent of the smallest
  axis. With anisotropic voxels that is usually z, not y — the 2D code's
  ``video.shape[1] / 2`` is a y-index assumption that does not carry over.
* Island counts explode in 3D, so nearest-neighbour separation uses a KD-tree instead
  of the 2D branch's dense pairwise distance matrix.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.fft import fftn, fftshift, ifftn
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.measure import label, regionprops_table
from skimage.morphology import remove_small_holes, remove_small_objects

from core import BinarizationResults, VolumetricConfig

# skimage derives axis lengths from inertia eigenvalues; below this the region is
# effectively planar/linear and its "anisotropy" is a division by ~zero.
_MIN_AXIS_LENGTH = 1e-6


def group_avg_3d(volume: np.ndarray, factors: Tuple[int, int, int]) -> np.ndarray:
    """Block-mean ``volume`` by ``factors`` per axis (the 3D analogue of groupAvg)."""
    fz, fy, fx = (max(int(f), 1) for f in factors)
    if (fz, fy, fx) == (1, 1, 1):
        return volume
    nz, ny, nx = (s // f * f for s, f in zip(volume.shape, (fz, fy, fx)))
    trimmed = volume[:nz, :ny, :nx].astype(np.float64)
    return trimmed.reshape(nz // fz, fz, ny // fy, fy, nx // fx, fx).mean(axis=(1, 3, 5))


def binarize_volume(
    volume: np.ndarray, offset_threshold: float, min_size: int = 1
) -> np.ndarray:
    """Threshold at ``mean * (1 + offset)``, then clean, using 3D connectivity.

    A volume with no contrast has no structure to find, and is reported as empty rather
    than as one object filling the field. The threshold is relative to the volume's own
    mean, so on a constant volume it lands exactly on that constant: an all-zero volume
    gives threshold 0 and ``volume >= 0`` is true everywhere, so a dropped or blank
    acquisition -- routine at the top and bottom of a stack -- came back 100% foreground
    with connectivity 1, and (via ``find_largest_void_3d``) a maximal void as well. In the
    CSV that is indistinguishable from a genuinely percolating sample.
    """
    volume = np.asarray(volume)
    if volume.size == 0 or float(np.ptp(volume)) == 0.0:
        return np.zeros(volume.shape, dtype=bool)
    threshold = float(np.mean(volume)) * (1 + offset_threshold)
    binary = volume >= threshold
    binary = remove_small_objects(binary, min_size + 1, connectivity=3)
    binary = remove_small_holes(binary, min_size + 1, connectivity=3)
    return binary


def invert_volume(binary: np.ndarray) -> np.ndarray:
    return ~binary.astype(bool)


def check_span_3d(binary: np.ndarray) -> int:
    """1 if any single object touches both faces of any axis (percolation)."""
    structure = ndimage.generate_binary_structure(3, 3)
    labelled, count = ndimage.label(binary, structure=structure)
    if count == 0:
        return 0
    for axis in range(3):
        first = set(np.unique(np.take(labelled, 0, axis=axis))) - {0}
        last = set(np.unique(np.take(labelled, -1, axis=axis))) - {0}
        if first & last:
            return 1
    return 0


def find_largest_void_3d(binary: np.ndarray) -> float:
    """Largest connected background region, in voxels."""
    inverted = invert_volume(binary)
    labelled, count = label(inverted, connectivity=3, return_num=True)
    if count == 0:
        # No background components means the mask is entirely foreground, so the largest
        # void is 0 -- the minimum. Returning binary.size reported the maximum instead,
        # inverting the metric on exactly the edge case, and fed max_void_percent_change
        # too.
        return 0.0
    areas = regionprops_table(labelled, properties=["area"])["area"]
    return float(np.max(areas))


def _anisotropy_from_eigvals(eigvals: np.ndarray) -> np.ndarray:
    """Major/minor axis-length ratio from 3D inertia eigenvalues.

    Reimplements skimage's ``axis_major_length``/``axis_minor_length`` with the
    negative-radicand clamp those properties lack — see the module docstring.
    Returns NaN for degenerate (planar or linear) regions rather than a huge ratio.
    """
    ev = np.sort(np.asarray(eigvals, dtype=np.float64), axis=1)[:, ::-1]  # descending
    # skimage: major = sqrt(10*(ev0 + ev1 - ev2)), minor = sqrt(10*(-ev0 + ev1 + ev2)).
    major = np.sqrt(10 * np.clip(ev[:, 0] + ev[:, 1] - ev[:, 2], 0.0, None))
    minor = np.sqrt(10 * np.clip(-ev[:, 0] + ev[:, 1] + ev[:, 2], 0.0, None))
    ratio = np.full(major.shape, np.nan)
    valid = minor > _MIN_AXIS_LENGTH
    ratio[valid] = major[valid] / minor[valid]
    return ratio


def _mean_island_separation(
    centroids_zyx: np.ndarray, spacing_zyx: Tuple[float, float, float], fraction: float
) -> float:
    """Mean distance to the nearest ``fraction`` of other islands, in microns.

    The 2D branch builds a dense N-by-N distance matrix; 3D island counts make that
    prohibitive, so this uses a KD-tree over physically-scaled coordinates.
    """
    n = len(centroids_zyx)
    if n < 2:
        return np.nan
    points = centroids_zyx * np.asarray(spacing_zyx, dtype=np.float64)
    k = int(np.ceil(fraction * n)) - 1
    k = max(min(k, n - 1), 1)
    distances, _ = cKDTree(points).query(points, k=k + 1)
    return float(np.nanmean(distances[:, 1:]))


def find_island_properties_3d(
    binary: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
    neighbor_fraction: float,
    labelled: np.ndarray = None,
) -> Dict[str, float]:
    """Volume/anisotropy/separation statistics for the objects in a field.

    ``labelled`` supplies the object partition directly -- every distinct positive
    integer is one object. Pass it whenever a segmentation provided instances:
    re-deriving objects by connectivity merges instances that touch, which is the normal
    case in a confluent field and silently corrupts count, separation and the whole size
    distribution. BARCODE does not segment; a supplied partition is authoritative.

    Without it, objects are derived from ``binary`` by 26-connectivity, which is correct
    for a binary mask or an intensity threshold.
    """
    if labelled is not None:
        labelled = np.asarray(labelled)
        count = int(np.count_nonzero(np.unique(labelled)))
    else:
        labelled, count = label(binary, connectivity=3, return_num=True)
    if count == 0:
        return {
            "largest": np.nan,
            "second_largest": np.nan,
            "total": np.nan,
            "mean": np.nan,
            "sd": np.nan,
            "skew": np.nan,
            "median": np.nan,
            "separation": np.nan,
            "anisotropy": np.nan,
            "count": 0,
        }

    regions = regionprops_table(labelled, properties=["area", "centroid"])
    areas = np.sort(np.asarray(regions["area"], dtype=np.float64))[::-1]
    centroids = np.stack([regions[f"centroid-{i}"] for i in range(3)], axis=1)

    # The inertia tensor MUST be built in physical coordinates, and it needs its own
    # regionprops pass to get there: `spacing` rescales every property in the call, so
    # folding it into the one above would turn `area` into a physical volume (it is
    # multiplied by voxel_volume downstream) and `centroid` into microns (it is
    # multiplied by spacing in _mean_island_separation) -- both double-counted.
    #
    # Without spacing the eigenvalues describe the object in VOXEL INDEX space, so
    # Anisotropy measured the sampling grid rather than the object: a physically perfect
    # sphere on the Jurkat grid (z 0.3 / xy 0.065 um) reported 4.56, which is just the
    # 4.6x grid ratio. With spacing it reports 1.01.
    #
    # On an isotropic grid this changes nothing at all -- uniform scaling multiplies
    # every eigenvalue by s^2, so the major/minor ratio is bit-identical -- which is why
    # this is a fix rather than a new setting.
    eigen_regions = regionprops_table(
        labelled, properties=["inertia_tensor_eigvals"], spacing=tuple(spacing_zyx)
    )
    eigvals = np.stack(
        [eigen_regions[f"inertia_tensor_eigvals-{i}"] for i in range(3)], axis=1
    )

    # Spread of the size distribution, not just its summary: one dominant object plus
    # debris and a handful of even objects give the same mean.
    skew = np.nan
    if len(areas) > 2 and areas.std() > 0:
        skew = float(np.mean(((areas - areas.mean()) / areas.std()) ** 3))

    return {
        "largest": float(areas[0]),
        "second_largest": float(areas[1]) if len(areas) > 1 else 0.0,
        "total": float(areas.sum()),
        "mean": float(np.nanmean(areas)),
        "sd": float(areas.std()) if len(areas) > 1 else 0.0,
        "skew": skew,
        "median": float(np.median(areas)),
        "separation": _mean_island_separation(centroids, spacing_zyx, neighbor_fraction),
        "anisotropy": float(np.nanmean(_anisotropy_from_eigvals(eigvals))),
        "count": int(count),
    }


def spatial_volume_autocorrelation(
    volume: np.ndarray, spacing_zyx: Tuple[float, float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Radially averaged 3D autocorrelation g(r), binned on physical distance.

    Returns ``(radii_um, g)``. Unlike the 2D branch, which bins on pixel index, this
    bins on true distance so anisotropic voxels do not distort the profile.
    """
    field = volume.astype(np.float64)
    std = field.std()
    if std == 0:
        return np.array([]), np.array([])
    field = (field - field.mean()) / std

    spectrum = fftn(field)
    corr = np.real(fftshift(ifftn(spectrum * np.conj(spectrum)))) / field.size

    dz, dy, dx = spacing_zyx
    nz, ny, nx = corr.shape
    z = (np.arange(nz) - nz // 2) * dz
    y = (np.arange(ny) - ny // 2) * dy
    x = (np.arange(nx) - nx // 2) * dx
    radius = np.sqrt(z[:, None, None] ** 2 + y[None, :, None] ** 2 + x[None, None, :] ** 2)

    # Bounded by the smallest physical half-extent; with anisotropic voxels this is
    # typically z, not y.
    r_max = min(nz // 2 * dz, ny // 2 * dy, nx // 2 * dx)
    step = min(dz, dy, dx)
    if r_max <= step:
        return np.array([]), np.array([])

    edges = np.arange(0, r_max, step)
    counts = np.histogram(radius, edges)[0]
    totals = np.histogram(radius, edges, weights=corr)[0]
    # Report each shell at its own mean radius, not at the bin's left edge. A shell
    # spans a range of radii and its population grows as r^2, so its mean radius sits
    # well above the left edge; anchoring g to the edge makes the profile look like it
    # decays faster than it does (~0.07 error against the analytic Gaussian case).
    radius_totals = np.histogram(radius, edges, weights=radius)[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        safe = np.maximum(counts, 1)
        radial = np.where(counts > 0, totals / safe, np.nan)
        centres = np.where(counts > 0, radius_totals / safe, np.nan)
    return centres, radial


def correlation_length_from_radial(
    radial: np.ndarray, radii: np.ndarray, threshold: float
) -> float:
    """First downward crossing of ``threshold``, snapped to the nearer grid point.

    Copied verbatim from ``analysis/binarization.py`` (the loop at lines 152-166) and
    ``analysis/optical_flow.py`` (lines 85-99) so that the 3D branch behaves exactly
    like the 2D one. Note that CLAUDE.md records this snapping rule as a post-2.0
    rewrite that does *not* reproduce the reference release 22f98df, which took the
    first x value where the correlation fell to or below the threshold. Kept as-is
    intentionally: changing it here would make 2D and 3D disagree.
    """
    for i in range(len(radial) - 1):
        if radial[i] > threshold and radial[i + 1] <= threshold:
            d1 = abs(radial[i] - threshold)
            d2 = abs(radial[i + 1] - threshold)
            return float(radii[i] if d1 < d2 else radii[i + 1])
    return np.nan


@dataclass
class VolumetricBinarizationDetail:
    """Per-timepoint values, kept for reporting and the single-frame harness."""

    frame_indices: List[int] = None
    island_voxels: List[float] = None
    void_voxels: List[float] = None
    total_island_voxels: List[float] = None
    mean_island_voxels: List[float] = None
    separations: List[float] = None
    anisotropies: List[float] = None
    correlation_lengths: List[float] = None
    connected: List[int] = None
    island_counts: List[int] = None
    size_sds: List[float] = None
    size_skews: List[float] = None
    size_medians: List[float] = None
    voxel_count: int = 0
    voxel_volume_um3: float = np.nan
    used_mask: bool = False
    r_max_um: float = np.nan


def analyze_binarization_3d(
    volume_series: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
    config: VolumetricConfig,
    frame_indices: List[int],
    masks: Optional[np.ndarray] = None,
) -> Tuple[BinarizationResults, VolumetricBinarizationDetail]:
    """Structural metrics over a ``(T, Z, Y, X)`` series.

    When ``masks`` is supplied it *replaces* intensity thresholding, per the design
    decision that a segmentation is the binarization rather than a modifier of it.
    Autocorrelation always runs on the raw voxels either way.
    """
    dz, dy, dx = spacing_zyx
    voxel_volume = dz * dy * dx
    threshold = np.exp(-1)

    detail = VolumetricBinarizationDetail(
        frame_indices=list(frame_indices),
        island_voxels=[], void_voxels=[], total_island_voxels=[],
        mean_island_voxels=[], separations=[], anisotropies=[],
        correlation_lengths=[], connected=[], island_counts=[],
        size_sds=[], size_skews=[], size_medians=[],
        voxel_volume_um3=voxel_volume,
        used_mask=masks is not None,
    )

    second_largest: List[float] = []

    for frame_idx in frame_indices:
        raw = volume_series[frame_idx]

        label_partition = None
        if masks is not None:
            frame_mask = masks[frame_idx]
            binary = frame_mask.astype(bool)
            # A label mask supplies the object partition directly. Re-deriving it by
            # connectivity would merge every touching instance, which in a confluent
            # field means the whole tissue becomes one object.
            if frame_mask.dtype != bool and int(np.count_nonzero(np.unique(frame_mask))) > 1:
                label_partition = frame_mask
        else:
            binary = binarize_volume(
                raw, config.threshold_offset, config.minimum_island_size
            )
            if config.invert_binarization:
                binary = invert_volume(binary)

        props = find_island_properties_3d(
            binary, spacing_zyx, config.neighbor_island_fraction, label_partition)
        radii, radial = spatial_volume_autocorrelation(raw, spacing_zyx)
        correlation_length = (
            correlation_length_from_radial(radial, radii, threshold)
            if radial.size
            else np.nan
        )
        if radii.size:
            # The outermost radial bin is the likeliest to be empty, and an empty bin
            # carries NaN, so `radii[-1]` could report the cut-off radius as NaN. Take
            # the largest radius that actually has data.
            finite_radii = radii[np.isfinite(radii)]
            detail.r_max_um = float(finite_radii[-1]) if finite_radii.size else np.nan

        detail.island_voxels.append(props["largest"])
        second_largest.append(props["second_largest"])
        detail.total_island_voxels.append(props["total"])
        detail.mean_island_voxels.append(props["mean"])
        detail.separations.append(props["separation"])
        detail.anisotropies.append(props["anisotropy"])
        detail.island_counts.append(props["count"])
        detail.size_sds.append(props["sd"])
        detail.size_skews.append(props["skew"])
        detail.size_medians.append(props["median"])
        detail.void_voxels.append(find_largest_void_3d(binary))
        detail.connected.append(check_span_3d(binary))
        detail.correlation_lengths.append(correlation_length)

    detail.voxel_count = int(volume_series[frame_indices[0]].size)
    return (
        _assemble_results(detail, second_largest, config, voxel_volume),
        detail,
    )


def _assemble_results(
    detail: VolumetricBinarizationDetail,
    second_largest: List[float],
    config: VolumetricConfig,
    voxel_volume: float,
) -> BinarizationResults:
    """Reduce per-timepoint values to the metrics the CSV expects.

    Change metrics compare the last and first ``percentage_frames_evaluated`` of the
    series. With a single timepoint that comparison is between a value and itself, so
    they are NaN rather than a meaningless 1.0.
    """
    n_frames = len(detail.frame_indices)
    single_timepoint = n_frames < 2

    island = np.asarray(detail.island_voxels, dtype=np.float64)
    void = np.asarray(detail.void_voxels, dtype=np.float64)
    voxels = float(detail.voxel_count)

    n_eval = max(int(np.ceil(n_frames * config.percentage_frames_evaluated)), 1)

    island_initial = _nanmean(island[:n_eval])
    island_initial2 = _nanmean(np.asarray(second_largest)[:n_eval])
    void_initial = _nanmean(void[:n_eval])

    if single_timepoint:
        island_change = np.nan
        void_change = np.nan
    else:
        # A zero or non-finite baseline gives inf (plus a RuntimeWarning) written into the
        # CSV as though it were a measurement. "Grew from nothing" has no ratio; NaN is the
        # honest answer, and it is what every other undefined value in this row already is.
        def ratio(final: float, initial: float) -> float:
            if not np.isfinite(initial) or initial == 0:
                return np.nan
            return final / initial

        island_change = ratio(_nanmean(island[-n_eval:]), island_initial)
        void_change = ratio(_nanmean(void[-n_eval:]), void_initial)

    def fraction(value: float) -> float:
        return value / voxels if voxels else np.nan

    def quantity(value: float) -> float:
        return value * voxel_volume

    max_island = _average_largest(detail.island_voxels)
    max_void = _average_largest(detail.void_voxels)
    mean_island = _nanmean(detail.mean_island_voxels)
    total_island = _nanmean(detail.total_island_voxels)

    correlation_lengths = np.asarray(detail.correlation_lengths, dtype=np.float64)

    return BinarizationResults(
        connectivity=float(np.mean(detail.connected)),
        max_island_size=fraction(max_island),
        max_void_size=fraction(max_void),
        max_island_percent_change=island_change,
        max_void_percent_change=void_change,
        island_size_initial=fraction(island_initial),
        island_size_initial2=fraction(island_initial2),
        island_anisotropy=_nanmean(detail.anisotropies),
        mean_island_size=fraction(mean_island),
        total_island_size=fraction(total_island),
        mean_island_separation=_nanmean(detail.separations),
        island_correlation_length=_nanmean(correlation_lengths),
        max_island_size_quantity=quantity(max_island),
        max_void_size_quantity=quantity(max_void),
        island_size_initial_quantity=quantity(island_initial),
        island_size_initial2_quantity=quantity(island_initial2),
        mean_island_size_quantity=quantity(mean_island),
        total_island_size_quantity=quantity(total_island),
        structural_correlation_flag=int(np.isnan(np.sum(correlation_lengths))),
    )


def _average_largest(values: List[float], percent: float = 0.1) -> float:
    """Mean of the top ``percent`` of values — matches ``utils.average_largest``.

    Non-finite values are dropped *before* the sort, not left for ``_nanmean`` to skip
    afterwards. ``sorted`` orders with ``<``, and every comparison against NaN is False,
    so a NaN is never moved -- it simply stays where it started. NaN is routine in this
    list (a frame with zero islands records NaN for its largest), so with ten frames and
    the default 10% the single selected element was whichever value happened to sit at
    index 0. That made the result order-dependent: [nan, 5, 3] gave NaN while [5, nan, 3]
    gave 5.0, from the same measurements.
    """
    finite = [v for v in np.asarray(values, dtype=np.float64).ravel() if np.isfinite(v)]
    if not finite:
        return np.nan
    ordered = sorted(finite, reverse=True)
    top = max(int(np.ceil(len(ordered) * percent)), 1)
    return _nanmean(ordered[:top])


def _nanmean(values) -> float:
    """``np.nanmean`` that returns NaN quietly when everything is NaN.

    All-NaN is an expected outcome, not an anomaly: a volume containing a single island
    has no island separation to report, so numpy's "Mean of empty slice" warning would
    fire on every well-formed single-object frame and train the reader to ignore it.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).any():
        return np.nan
    return float(np.nanmean(array[np.isfinite(array)]))
