"""3D optical flow — the volumetric counterpart of ``analysis/optical_flow.py``.

The 2D branch uses ``cv.calcOpticalFlowFarneback``, which is 2D only and has no OpenCV
3D equivalent. This module instead wraps the Gaussian-derivative Lucas-Kanade solver
vendored in ``flow_lucas_kanade.py`` from aicjanelia/OpticalFlow3D (BSD-3), and reduces
its velocity fields to the same seven ``FlowResults`` metrics the CSV already has
columns for.

Four things genuinely differ from the 2D branch, and each forced a decision:

* **Windows, not frame pairs.** Farneback needs two frames; ``calc_flow3D`` needs a
  *contiguous* run of ``6*tSig+1`` volumes and returns flow for the middle one only.
  With the default ``tSig=1`` that is 7. The branch therefore centres a window on each
  frame ``select_frame_indices`` picked and drops centres too close to either end —
  which is what upstream's own ``process_flow`` does. Series shorter than one window
  are skipped rather than approximated.
* **Curl is a 3-vector in 3D.** There is one ``Curl`` column, so this reports the mean
  of ``||curl v||``. That is rotation-invariant but unsigned, unlike the 2D scalar,
  which carries handedness.
* **Direction needs two angles in 3D.** ``Mean Flow Direction`` stays the XY azimuth so
  the column remains comparable with 2D runs; ``Flow Directional Spread`` uses the
  resultant length of the full *3D* unit vectors, so out-of-plane scatter still widens
  it.
* **Velocity correlation cannot be brute-forced.** ``utils.optical_flow.velocity_correlation``
  loops explicitly over every shift, which is O(N^2) and hopeless on a volume. The FFT
  form here is the same quantity; see ``velocity_correlation_3d``.

Sign convention: upstream puts (0,0) at the top-left with +y pointing *down*. The 2D
branch flips to y-up (``-1 * np.flipud(downV)``), so ``vy`` is negated here to match.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.fft import fftn, fftshift, ifftn

from analysis.volumetric.binarization import (
    correlation_length_from_radial,
    group_avg_3d,
)
from analysis.volumetric.flow_lucas_kanade import calc_flow3D
from core import FlowResults, VolumetricConfig

FLOW_3D_AVAILABLE = True

# The 2D flow branch calls the velocity correlation length the first downward crossing
# of 0.5 (analysis/optical_flow.py:85). Kept identical so 2D and 3D mean the same thing.
_CORRELATION_THRESHOLD = 0.5


@dataclass
class VolumetricFlowDetail:
    """Per-window values, for the verbose harness. The CSV has no column for these."""

    centres: List[int] = field(default_factory=list)
    skipped_centres: List[int] = field(default_factory=list)
    window_size: int = 0
    t_sigma: int = 0
    speeds: List[float] = field(default_factory=list)
    divergences: List[float] = field(default_factory=list)
    curls: List[float] = field(default_factory=list)
    correlation_lengths: List[float] = field(default_factory=list)
    valid_fractions: List[float] = field(default_factory=list)
    used_mask: bool = False
    downsample: int = 1
    spacing_zyx_um: Tuple[float, float, float] = (np.nan, np.nan, np.nan)
    r_max_um: float = np.nan


def window_size(t_sigma: int) -> int:
    """Contiguous timepoints ``calc_flow3D`` needs per analysed frame.

    ``6*tSig+1`` is upstream's own requirement (three sigma of temporal kernel either
    side of a central pixel), forced odd because only the central frame is returned.
    """
    size = 6 * int(t_sigma) + 1
    return size if size % 2 else size + 1


def velocity_correlation_3d(
    field_zyx: np.ndarray, spacing_zyx: Tuple[float, float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Radially averaged velocity autocorrelation C(r), binned on physical distance.

    ``field_zyx`` is ``(Z, Y, X, 3)``; NaNs (masked-out voxels) contribute zero.
    Returns ``(radii_um, C)`` with ``C(0) == 1``.

    This is the FFT form of what ``utils.optical_flow.velocity_correlation`` computes by
    explicit shifting: ``C(r) = <v(x).v(x+r)> / <|v|^2>``, the dot product summed over
    components. Wiener-Khinchin turns each component's shift-and-multiply into
    ``|FFT|^2``, which is the only reason this is tractable on a volume.

    The radial binning is deliberately identical to
    ``binarization.spatial_volume_autocorrelation`` — bounded by the smallest *physical*
    half-extent (with anisotropic voxels that is usually z, not y) and reporting each
    shell at its own mean radius rather than its bin's left edge.
    """
    field = np.nan_to_num(np.asarray(field_zyx, dtype=np.float64), nan=0.0)
    mean_square = float(np.mean(np.sum(field ** 2, axis=-1)))
    if mean_square == 0:
        return np.array([]), np.array([])

    corr = np.zeros(field.shape[:3], dtype=np.float64)
    for component in range(field.shape[-1]):
        spectrum = fftn(field[..., component])
        corr += np.real(fftshift(ifftn(spectrum * np.conj(spectrum))))
    # At zero lag the sum over components is N * <|v|^2>, so this normalises C(0) to 1.
    corr /= corr.size * mean_square

    dz, dy, dx = spacing_zyx
    nz, ny, nx = corr.shape
    z = (np.arange(nz) - nz // 2) * dz
    y = (np.arange(ny) - ny // 2) * dy
    x = (np.arange(nx) - nx // 2) * dx
    radius = np.sqrt(z[:, None, None] ** 2 + y[None, :, None] ** 2 + x[None, None, :] ** 2)

    r_max = min(nz // 2 * dz, ny // 2 * dy, nx // 2 * dx)
    step = min(dz, dy, dx)
    if r_max <= step:
        return np.array([]), np.array([])

    edges = np.arange(0, r_max, step)
    counts = np.histogram(radius, edges)[0]
    totals = np.histogram(radius, edges, weights=corr)[0]
    radius_totals = np.histogram(radius, edges, weights=radius)[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        safe = np.maximum(counts, 1)
        radial = np.where(counts > 0, totals / safe, np.nan)
        centres = np.where(counts > 0, radius_totals / safe, np.nan)
    return centres, radial


def _divergence_3d(
    field_zyx: np.ndarray, spacing_zyx: Tuple[float, float, float]
) -> np.ndarray:
    """div v = dvx/dx + dvy/dy + dvz/dz, on the physical grid."""
    dz, dy, dx = spacing_zyx
    vz, vy, vx = field_zyx[..., 2], field_zyx[..., 1], field_zyx[..., 0]
    return (
        np.gradient(vx, dx, axis=2)
        + np.gradient(vy, dy, axis=1)
        + np.gradient(vz, dz, axis=0)
    )


def _curl_magnitude_3d(
    field_zyx: np.ndarray, spacing_zyx: Tuple[float, float, float]
) -> np.ndarray:
    """||curl v||, on the physical grid.

    Array axes are (Z, Y, X) while the vector components are ordered (x, y, z), so the
    axis each derivative is taken along has to be read off carefully: axis 0 is z,
    axis 1 is y, axis 2 is x.
    """
    dz, dy, dx = spacing_zyx
    vx, vy, vz = field_zyx[..., 0], field_zyx[..., 1], field_zyx[..., 2]
    curl_x = np.gradient(vz, dy, axis=1) - np.gradient(vy, dz, axis=0)
    curl_y = np.gradient(vx, dz, axis=0) - np.gradient(vz, dx, axis=2)
    curl_z = np.gradient(vy, dx, axis=2) - np.gradient(vx, dy, axis=1)
    return np.sqrt(curl_x ** 2 + curl_y ** 2 + curl_z ** 2)


def _nanmean(values) -> float:
    """``np.nanmean`` that returns NaN quietly when everything is NaN.

    Same rationale as ``binarization._nanmean``: an all-NaN reduction is an expected
    outcome here (a fully masked-out window), not an anomaly worth a warning on.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).any():
        return np.nan
    return float(np.nanmean(array[np.isfinite(array)]))


def _select_centres(n_frames: int, frame_indices: List[int], size: int) -> Tuple[List[int], List[int]]:
    """Split the requested centres into those a full window fits around, and the rest."""
    half = size // 2
    kept, dropped = [], []
    for centre in frame_indices:
        (kept if half <= centre <= n_frames - 1 - half else dropped).append(int(centre))
    return kept, dropped


def analyze_optical_flow_3d(
    volume_series: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
    exposure_time_s: float,
    config: VolumetricConfig,
    frame_indices: List[int],
    masks: Optional[np.ndarray] = None,
) -> Tuple[FlowResults, VolumetricFlowDetail]:
    """Flow metrics over a ``(T, Z, Y, X)`` series.

    ``masks`` is the per-timepoint segmentation on the same grid, or None. When
    ``config.flow_use_mask`` is set it restricts which voxels enter the metrics; the
    flow itself is always solved on the whole volume so gradients at the mask boundary
    are computed from real neighbours rather than from zeros.
    """
    size = window_size(config.flow_t_sigma)
    n_frames = int(volume_series.shape[0])

    detail = VolumetricFlowDetail(
        window_size=size,
        t_sigma=int(config.flow_t_sigma),
        used_mask=bool(masks is not None and config.flow_use_mask),
        downsample=max(int(config.flow_downsample), 1),
    )

    if n_frames < size:
        print(
            f"3D optical flow needs {size} contiguous timepoints "
            f"(6*t_sigma+1, t_sigma={config.flow_t_sigma}) but the series has "
            f"{n_frames}; skipping the flow branch. Structural and intensity metrics "
            f"are unaffected.",
            flush=True,
        )
        return FlowResults(), detail

    centres, dropped = _select_centres(n_frames, frame_indices, size)
    detail.centres, detail.skipped_centres = centres, dropped
    if dropped:
        print(
            f"  flow: {len(dropped)} timepoint(s) sit within {size // 2} frames of a "
            f"series end and have no full window; skipped {dropped}.",
            flush=True,
        )
    if not centres:
        print("  flow: no timepoint has a full window; skipping the flow branch.", flush=True)
        return FlowResults(), detail

    down = max(int(config.flow_downsample), 1)
    dz, dy, dx = (s * down for s in spacing_zyx)
    detail.spacing_zyx_um = (dz, dy, dx)

    mean_units: List[np.ndarray] = []
    resultants: List[float] = []

    for centre in centres:
        window = volume_series[centre - size // 2 : centre + size // 2 + 1]
        if down > 1:
            window = np.stack([group_avg_3d(v, (down, down, down)) for v in window])

        vx, vy, vz, rel = calc_flow3D(
            window,
            float(config.flow_xyz_sigma),
            int(config.flow_t_sigma),
            float(config.flow_w_sigma),
        )

        # voxels/frame -> um/s. The window is contiguous, so one frame is one exposure.
        dt = float(exposure_time_s) if exposure_time_s else 1.0
        vx = vx * dx / dt
        vy = -vy * dy / dt  # upstream is y-down; BARCODE reports y-up.
        vz = vz * dz / dt

        valid = np.ones(vx.shape, dtype=bool)
        if config.flow_reliability_percentile > 0:
            cutoff = np.nanpercentile(rel, float(config.flow_reliability_percentile))
            valid &= rel >= cutoff
        if masks is not None and config.flow_use_mask:
            mask = masks[centre].astype(bool)
            if down > 1:
                mask = group_avg_3d(mask.astype(np.float64), (down, down, down)) >= 0.5
            valid &= mask
        detail.valid_fractions.append(float(valid.mean()))

        field = np.stack([vx, vy, vz], axis=-1)
        field[~valid] = np.nan

        speed = np.sqrt(np.nansum(field ** 2, axis=-1))
        speed[~valid] = np.nan
        detail.speeds.append(_nanmean(speed))

        with np.errstate(invalid="ignore", divide="ignore"):
            unit = field / speed[..., None]
        mean_unit = np.array([_nanmean(unit[..., i]) for i in range(3)])
        mean_units.append(mean_unit)
        resultants.append(float(np.sqrt(np.nansum(mean_unit ** 2))))

        # Gradients need a filled field; NaNs would spread to every neighbour. Masked
        # voxels are zeroed for the derivative and then dropped from the mean, so an
        # excluded region contributes nothing rather than contributing noise.
        filled = np.nan_to_num(field, nan=0.0)
        divergence = _divergence_3d(filled, (dz, dy, dx))
        curl = _curl_magnitude_3d(filled, (dz, dy, dx))
        detail.divergences.append(_nanmean(np.where(valid, divergence, np.nan)))
        detail.curls.append(_nanmean(np.where(valid, curl, np.nan)))

        radii, radial = velocity_correlation_3d(field, (dz, dy, dx))
        if radii.size:
            detail.r_max_um = float(radii[-1])
            detail.correlation_lengths.append(
                correlation_length_from_radial(radial, radii, _CORRELATION_THRESHOLD)
            )
        else:
            detail.correlation_lengths.append(np.nan)

    return _assemble_results(detail, mean_units, resultants, config), detail


def _assemble_results(
    detail: VolumetricFlowDetail,
    mean_units: List[np.ndarray],
    resultants: List[float],
    config: VolumetricConfig,
) -> FlowResults:
    """Reduce per-window values to the seven metrics the CSV expects.

    ``delta_speed`` compares the last and first ``percentage_frames_evaluated`` of the
    windows. With a single window that would compare a value against itself, so it is
    NaN — the same convention as ``binarization._assemble_results`` and
    ``intensity.analyze_intensity_3d``.
    """
    speeds = np.asarray(detail.speeds, dtype=np.float64)
    n_windows = len(speeds)
    n_eval = max(int(np.ceil(n_windows * config.percentage_frames_evaluated)), 1)

    delta_speed = (
        np.nan
        if n_windows < 2
        else _nanmean(speeds[-n_eval:]) - _nanmean(speeds[:n_eval])
    )

    # Average the per-window mean unit vectors, then take the angle — matching how the
    # 2D branch averages components before calling arctan2 (optical_flow.py:113-136).
    units = np.asarray(mean_units, dtype=np.float64)
    theta = (
        float(np.arctan2(_nanmean(units[:, 1]), _nanmean(units[:, 0])))
        if units.size
        else np.nan
    )

    # Circular spread from the resultant length of the *3D* unit vectors: R -> 1 for a
    # coherent flow (spread 0), R -> 0 for isotropic scatter (spread -> inf).
    resultant = np.clip(np.asarray(resultants, dtype=np.float64), 1e-12, 1.0)
    sigma_theta = _nanmean(np.sqrt(-2 * np.log(resultant)))

    correlation_lengths = np.asarray(detail.correlation_lengths, dtype=np.float64)

    return FlowResults(
        mean_speed=_nanmean(speeds),
        delta_speed=delta_speed,
        mean_theta=theta,
        mean_sigma_theta=sigma_theta,
        velocity_correlation_length=_nanmean(correlation_lengths),
        divergence=_nanmean(detail.divergences),
        curl=_nanmean(detail.curls),
        velocity_correlation_flag=int(np.isnan(np.sum(correlation_lengths))),
    )
