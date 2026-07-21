"""Shared drawing helpers for physically-calibrated volume panels.

Moved VERBATIM from ``scripts/make_open_data_figures.py``, which is where they grew and
which now imports them from here. The move exists because ``analysis/`` and
``visualization/`` must not import from ``scripts/``, and the fingerprint card needs the
same panels the open-data figures use -- two copies of a scale-bar convention would drift
and quietly disagree about what a micron is.

Everything here works in PHYSICAL units: panels are drawn with a micron extent rather
than a pixel one, so anisotropic voxels look anisotropic and a scale bar means the same
thing in xy and xz.
"""
from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


MU = "µm"

INSTANCE_COLORS = np.array(
    [
        (0.902, 0.624, 0.000),  # orange
        (0.337, 0.706, 0.914),  # sky blue
        (0.000, 0.620, 0.451),  # bluish green
        (0.941, 0.894, 0.259),  # yellow
        (0.000, 0.447, 0.698),  # blue
        (0.835, 0.369, 0.000),  # vermillion
        (0.800, 0.475, 0.655),  # reddish purple
    ]
)

NICE_BAR_UM = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

def apply_style() -> None:
    """Slide-legible matplotlib defaults; white background, no chart junk."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "lines.linewidth": 2.2,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )

def as_zyx(volume: np.ndarray) -> np.ndarray:
    volume = np.asarray(volume)
    while volume.ndim > 3:
        volume = volume[0]
    if volume.ndim == 2:
        volume = volume[None]
    return volume

def stretch(image: np.ndarray, low: float = 0.5, high: float = 99.8) -> np.ndarray:
    """Percentile contrast stretch to [0, 1]; robust to empty/flat panels."""
    data = np.asarray(image, dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros_like(data)
    lo, hi = np.percentile(finite, [low, high])
    if not np.isfinite(hi) or hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros_like(data)
    return np.clip((data - lo) / (hi - lo), 0.0, 1.0)

def projections(volume: np.ndarray) -> Dict[str, np.ndarray]:
    """Max-intensity projections. xy is (y,x); xz is (z,x); yz is (z,y)."""
    volume = as_zyx(volume)
    return {
        "xy": volume.max(axis=0),
        "xz": volume.max(axis=1),
        "yz": volume.max(axis=2),
    }

def label_projections(labels: np.ndarray) -> Dict[str, np.ndarray]:
    labels = as_zyx(labels)
    return {
        "xy": labels.max(axis=0),
        "xz": labels.max(axis=1),
        "yz": labels.max(axis=2),
    }

def boundary_rgba(labels2d: np.ndarray, thickness: int = 2) -> np.ndarray:
    """RGBA overlay of instance boundaries, one colour per label (cycled)."""
    from scipy import ndimage
    from skimage.segmentation import find_boundaries

    labels2d = np.asarray(labels2d)
    rgba = np.zeros(labels2d.shape + (4,), dtype=np.float32)
    if labels2d.max() == 0:
        return rgba
    edges = find_boundaries(labels2d, mode="inner")
    painted = np.where(edges, labels2d, 0)
    if thickness > 1:
        painted = ndimage.grey_dilation(painted, size=(thickness, thickness))
    hit = painted > 0
    if not hit.any():
        return rgba
    colors = INSTANCE_COLORS[(painted[hit].astype(np.int64) - 1) % len(INSTANCE_COLORS)]
    rgba[hit, :3] = colors
    rgba[hit, 3] = 1.0
    return rgba

def object_count(labels: np.ndarray) -> int:
    values = np.unique(np.asarray(labels))
    return int((values > 0).sum())

def nice_bar_length(span_um: float) -> float:
    target = span_um * 0.22
    choices = [c for c in NICE_BAR_UM if c <= max(target, NICE_BAR_UM[0])]
    return choices[-1] if choices else NICE_BAR_UM[0]

def add_scale_bar(ax, span_um: float, color: str = "white", region=None) -> None:
    """Scale bar in the lower left of the image region (data coordinates, µm)."""
    length = nice_bar_length(span_um)
    if region is None:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
    else:
        x0, x1, y_lo, y_hi = region
        y0, y1 = y_hi, y_lo  # images are drawn with an inverted y axis
    width = abs(x1 - x0)
    height = abs(y1 - y0)
    xs = min(x0, x1) + 0.06 * width
    # data coordinates: y axis is inverted for images (origin upper)
    ys = (max(y0, y1) - 0.12 * height) if y0 > y1 else (min(y0, y1) + 0.12 * height)
    ax.plot(
        [xs, xs + length],
        [ys, ys],
        color=color,
        linewidth=4,
        solid_capstyle="butt",
        zorder=6,
    )
    ax.text(
        xs + length / 2,
        ys - 0.035 * height if y0 > y1 else ys + 0.035 * height,
        f"{length:g} {MU}",
        color=color,
        ha="center",
        va="bottom" if y0 > y1 else "top",
        fontsize=12,
        zorder=6,
    )

def show_panel(
    ax,
    image: np.ndarray,
    extent_x_um: float,
    extent_y_um: float,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    ax.imshow(
        stretch(image),
        cmap="gray",
        vmin=0,
        vmax=1,
        origin="upper",
        extent=(0.0, extent_x_um, extent_y_um, 0.0),
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_title(title, pad=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=3)
