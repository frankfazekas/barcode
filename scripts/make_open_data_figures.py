#!/usr/bin/env python
"""Presentation-quality figures and movies for the staged open 3-D datasets.

Walks ``L:\\FF\\Hackathon\\full_datasets`` for the staged ``ctc_*`` (Cell Tracking
Challenge time-lapses) and ``allen_*`` (Allen Institute single-timepoint fields)
folders and, for each one, writes into ``_figures/<dataset>/``:

    overview.png          max-intensity projections along z, y and x, with a
                          physical scale bar and the dataset geometry as caption
    mask_overlay.png      the same projections with instance-mask boundaries drawn
                          on top in distinct colours, plus the object count
    z_flythrough.gif/.avi slice-by-slice walk down z with the mask contour and the
                          physical depth annotated per frame
    t_flythrough.gif/.avi (time-lapses only) the max-z projection across timepoints
                          with the mask contour and elapsed time in real units
    metrics_timeseries.png (time-lapses only) BARCODE metrics against real elapsed
                          time, read from the "(physical)" results CSV

and one cross-dataset grid at ``_figures/overview/dataset_grid.png``.

Geometry is never hard-coded: every staged volume was rewritten with ImageJ metadata,
so the xy step comes from the TIFF ``XResolution`` tag, the z step from ImageJ
``spacing`` and the frame interval from ImageJ ``finterval``; the mask's own
``spacing`` gives the (finer, isotropic) mask grid. The README is only consulted as a
fallback.

Outputs go to the data drive -- never to C:.

Usage::

    python scripts/make_open_data_figures.py
    python scripts/make_open_data_figures.py --only ctc_Fluo-N3DH-CHO_01
    python scripts/make_open_data_figures.py --limit 12 --datasets-limit 2
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tifffile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts._staging import read_tiff_any  # noqa: E402

DEFAULT_ROOT = r"L:\FF\Hackathon\full_datasets"
MU = "µm"

# Okabe-Ito: colourblind-safe qualitative palette.
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

SERIES_COLOR = "#0072B2"

NICE_BAR_UM = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

# Metrics worth showing against time, in preference order, with display units.
TIMESERIES_METRICS: List[Tuple[str, str]] = [
    ("Island Count", "objects"),
    ("Total Island Volume Quantity", f"{MU}$^3$"),
    ("Mean Island Volume Quantity", f"{MU}$^3$"),
    ("Maximum Island Volume Quantity", f"{MU}$^3$"),
    ("Mean Island Separation", MU),
    ("Mean Island Anisotropy", "ratio"),
    ("Structural Correlation Length", MU),
    ("Speed", f"{MU}/s"),
    ("Median Island Volume", "fraction of FOV"),
]


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


# --------------------------------------------------------------------------------------
# discovery + geometry
# --------------------------------------------------------------------------------------


@dataclass
class Dataset:
    name: str
    root: str
    data_dir: str
    mask_dir: str
    results_dir: str
    frames: List[str]
    masks: List[str]
    xy_um: float
    z_um: float
    mask_um: float
    frame_interval_s: float
    shape: Tuple[int, int, int]
    dtype: str

    @property
    def is_timelapse(self) -> bool:
        return self.name.startswith("ctc_") and len(self.frames) > 1

    @property
    def anisotropy(self) -> float:
        return self.z_um / self.xy_um if self.xy_um else float("nan")

    @property
    def geometry_caption(self) -> str:
        nz, ny, nx = self.shape
        parts = [
            f"{nx}×{ny}×{nz} voxels (x×y×z)",
            f"voxel {self.xy_um:.4g}×{self.xy_um:.4g}×{self.z_um:.4g} {MU}"
            f" (anisotropy {self.anisotropy:.1f}×)",
            f"FOV {nx * self.xy_um:.1f}×{ny * self.xy_um:.1f}×{nz * self.z_um:.1f} {MU}",
        ]
        if self.is_timelapse:
            span = (len(self.frames) - 1) * self.frame_interval_s
            parts.append(
                f"{len(self.frames)} timepoints × {fmt_time(self.frame_interval_s)}"
                f" (span {fmt_time(span)})"
            )
        else:
            parts.append(f"{len(self.frames)} single-timepoint fields")
        return " • ".join(parts)


def fmt_time(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds <= 0:
        return "n/a"
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


def _rational(value) -> Optional[float]:
    try:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return float(value[0]) / float(value[1])
        return float(value)
    except Exception:
        return None


def read_geometry(path: str) -> Dict[str, Optional[float]]:
    """xy/z spacing and frame interval as declared by a staged TIFF itself."""
    with tifffile.TiffFile(path) as handle:
        series = handle.series[0]
        shape = tuple(int(v) for v in series.shape)
        dtype = str(series.dtype)
        meta = handle.imagej_metadata or {}
        tag = handle.pages[0].tags.get("XResolution")
        xres = _rational(tag.value) if tag is not None else None
    xy = 1.0 / xres if xres else None
    return {
        "xy_um": xy,
        "z_um": _rational(meta.get("spacing")),
        "frame_interval_s": _rational(meta.get("finterval")),
        "shape": shape,
        "dtype": dtype,
    }


def readme_fallback(root: str) -> Dict[str, float]:
    """Scrape xy/z/frame-interval out of README.txt when a TIFF is silent."""
    path = os.path.join(root, "README.txt")
    out: Dict[str, float] = {}
    if not os.path.isfile(path):
        return out
    text = open(path, "r", encoding="utf-8", errors="replace").read()
    patterns = {
        "xy_um": r"xy(?:\s+step)?\s+([0-9.]+)\s*um",
        "z_um": r"z(?:\s+step)?\s+([0-9.]+)\s*um",
        "frame_interval_s": r"frame\s+interval\s+([0-9.]+)\s*s",
    }
    for key, pattern in patterns.items():
        found = re.search(pattern, text, re.IGNORECASE)
        if found:
            out[key] = float(found.group(1))
    return out


def list_tifs(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    names = [
        n
        for n in sorted(os.listdir(folder))
        if n.lower().endswith((".tif", ".tiff")) and not n.startswith("._")
    ]
    return [os.path.join(folder, n) for n in names]


def discover(root: str) -> List[Dataset]:
    datasets: List[Dataset] = []
    for name in sorted(os.listdir(root)):
        if not (name.startswith("ctc_") or name.startswith("allen_")):
            continue
        base = os.path.join(root, name)
        data_dir = os.path.join(base, "BARCODE", "data")
        frames = list_tifs(data_dir)
        if not frames:
            continue
        mask_dir = os.path.join(base, "BARCODE", "masks")
        masks = list_tifs(mask_dir)
        geom = read_geometry(frames[0])
        fallback = readme_fallback(base)
        xy = geom["xy_um"] or fallback.get("xy_um")
        z = geom["z_um"] or fallback.get("z_um")
        interval = geom["frame_interval_s"] or fallback.get("frame_interval_s") or 0.0
        if not xy or not z:
            print(f"  ! {name}: no usable spacing metadata, skipping")
            continue
        mask_um = xy
        if masks:
            mask_geom = read_geometry(masks[0])
            mask_um = mask_geom["z_um"] or xy
        datasets.append(
            Dataset(
                name=name,
                root=base,
                data_dir=data_dir,
                mask_dir=mask_dir,
                results_dir=os.path.join(base, "BARCODE", "results"),
                frames=frames,
                masks=masks,
                xy_um=float(xy),
                z_um=float(z),
                mask_um=float(mask_um),
                frame_interval_s=float(interval),
                shape=geom["shape"],  # type: ignore[arg-type]
                dtype=str(geom["dtype"]),
            )
        )
    return datasets


def mask_for(dataset: Dataset, frame_path: str) -> Optional[str]:
    stem = os.path.splitext(os.path.basename(frame_path))[0]
    candidate = os.path.join(dataset.mask_dir, f"{stem}_SegMask.tif")
    return candidate if os.path.isfile(candidate) else None


def representative_index(dataset: Dataset) -> int:
    """Middle timepoint for a time-lapse, first field otherwise."""
    return len(dataset.frames) // 2 if dataset.is_timelapse else 0


# --------------------------------------------------------------------------------------
# array helpers
# --------------------------------------------------------------------------------------


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


def physical_row_figure(
    panels: Sequence[Tuple[float, float]],
    target_width_in: float = 15.0,
    max_band_in: float = 8.0,
    top_in: float = 1.15,
    bottom_in: float = 1.45,
):
    """A figure whose axes boxes have exactly the panels' physical aspect ratios.

    Equal-aspect imshow panels inside a uniform grid leave large empty margins when
    the projections are strongly anisotropic, so the axes are placed by hand in inch
    space at a single common ``inches per micron`` scale -- which also makes the three
    projections directly comparable.
    """
    left, right, gap = 1.05, 0.4, 1.05
    widths_um = [w for w, _h in panels]
    heights_um = [h for _w, h in panels]
    available = target_width_in - left - right - gap * (len(panels) - 1)
    scale = available / sum(widths_um)
    scale = min(scale, max_band_in / max(heights_um))
    widths_in = [w * scale for w in widths_um]
    heights_in = [h * scale for h in heights_um]
    band = max(heights_in)
    fig_w = left + right + gap * (len(panels) - 1) + sum(widths_in)
    fig_h = band + top_in + bottom_in
    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = []
    cursor = left
    for width_in, height_in in zip(widths_in, heights_in):
        y = bottom_in + (band - height_in) / 2.0
        axes.append(
            fig.add_axes([cursor / fig_w, y / fig_h, width_in / fig_w, height_in / fig_h])
        )
        cursor += width_in + gap
    return fig, axes, fig_w, fig_h


def panel_specs(dataset: Dataset, volume_shape: Tuple[int, int, int]):
    nz, ny, nx = volume_shape
    x_um, y_um, z_um = nx * dataset.xy_um, ny * dataset.xy_um, nz * dataset.z_um
    return [
        ("xy", "XY — max over z", f"x ({MU})", f"y ({MU})", x_um, y_um),
        ("xz", "XZ — max over y", f"x ({MU})", f"z ({MU})", x_um, z_um),
        ("yz", "YZ — max over x", f"y ({MU})", f"z ({MU})", y_um, z_um),
    ]


def add_captions(
    fig: Figure, fig_h: float, title: str, caption: str, source: str, sub: str = ""
) -> None:
    """Title above, caption lines below -- placed in inch space so they never collide."""
    fig.text(
        0.5, 1.0 - 0.42 / fig_h, title, ha="center", va="top", fontsize=20, fontweight="bold"
    )
    lines = [caption] + ([sub] if sub else [])
    top = 0.40 + 0.30 * len(lines)
    for index, line in enumerate(lines):
        fig.text(
            0.5,
            (top - 0.30 * index) / fig_h,
            line,
            ha="center",
            va="top",
            fontsize=12.0,
            color="0.25",
        )
    fig.text(
        0.5,
        0.10 / fig_h,
        f"representative volume: {source}",
        ha="center",
        va="bottom",
        fontsize=10.0,
        color="0.5",
    )


# --------------------------------------------------------------------------------------
# A. overview figure
# --------------------------------------------------------------------------------------


def figure_overview(dataset: Dataset, volume: np.ndarray, out_path: str, source: str) -> str:
    projs = projections(volume)
    specs = panel_specs(dataset, as_zyx(volume).shape)
    fig, axes, fig_w, fig_h = physical_row_figure([(s[4], s[5]) for s in specs])
    for ax, (key, title, xlabel, ylabel, ex, ey) in zip(axes, specs):
        show_panel(ax, projs[key], ex, ey, title, xlabel, ylabel)
        if key == "xy":
            add_scale_bar(ax, ex)
    add_captions(fig, fig_h, dataset.name, dataset.geometry_caption, source)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------------
# B. mask overlay figure
# --------------------------------------------------------------------------------------


def figure_mask_overlay(
    dataset: Dataset,
    volume: np.ndarray,
    labels: np.ndarray,
    out_path: str,
    source: str,
) -> str:
    projs = projections(volume)
    label_projs = label_projections(labels)
    specs = panel_specs(dataset, as_zyx(volume).shape)
    nzm = as_zyx(labels).shape[0]
    mask_z_um = nzm * dataset.mask_um
    count = object_count(labels)

    fig, axes, fig_w, fig_h = physical_row_figure(
        [(s[4], s[5]) for s in specs], bottom_in=1.75
    )
    for ax, (key, title, xlabel, ylabel, ex, ey) in zip(axes, specs):
        show_panel(ax, projs[key], ex, ey, title, xlabel, ylabel)
        # The mask lives on its own (finer, isotropic) z grid, so it is drawn at its
        # own physical extent rather than resampled onto the image grid.
        overlay_ey = mask_z_um if key in ("xz", "yz") else ey
        ax.imshow(
            boundary_rgba(label_projs[key]),
            origin="upper",
            extent=(0.0, ex, overlay_ey, 0.0),
            interpolation="nearest",
            aspect="equal",
            zorder=4,
        )
        ax.set_xlim(0.0, ex)
        ax.set_ylim(ey, 0.0)
        if key == "xy":
            add_scale_bar(ax, ex)
    add_captions(
        fig,
        fig_h,
        f"{dataset.name}  —  {count} instance labels",
        "instance-mask boundaries over the max-intensity projections • "
        f"mask grid isotropic at {dataset.mask_um:.4g} {MU}",
        source,
        sub=dataset.geometry_caption,
    )
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------------
# movie plumbing
# --------------------------------------------------------------------------------------


def fig_to_rgb(fig: Figure) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def write_gif(frames: Sequence[np.ndarray], path: str, fps: float, max_dim: int = 640) -> None:
    """GIF via Pillow (ffmpeg is not reliably available); downscaled to stay shareable."""
    import cv2
    from PIL import Image

    if not frames:
        return
    scale = max_dim / float(max(frames[0].shape[:2]))
    if scale < 1.0:
        size = (
            max(int(frames[0].shape[1] * scale), 2),
            max(int(frames[0].shape[0] * scale), 2),
        )
        frames = [cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames]
    images = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE, colors=96) for f in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=max(int(round(1000.0 / max(fps, 0.1))), 20),
        loop=0,
        optimize=True,
        disposal=2,
    )


def write_avi(frames: Sequence[np.ndarray], path: str, fps: float) -> None:
    import cv2

    if not frames:
        return
    height, width = frames[0].shape[:2]
    # MJPG needs even dimensions on some builds; pad rather than rescale.
    pad_h, pad_w = height % 2, width % 2
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"MJPG"), max(fps, 1.0), (width + pad_w, height + pad_h)
    )
    if not writer.isOpened():
        print(f"    ! could not open {path} for writing")
        return
    for frame in frames:
        if pad_h or pad_w:
            frame = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        writer.write(frame[:, :, ::-1])
    writer.release()


def stride_for(count: int, limit: int) -> int:
    return max(1, int(math.ceil(count / float(max(limit, 1)))))


def movie_figure(width_um: float, height_um: float):
    """A fixed-size single-panel figure whose axes box matches the physical aspect."""
    left, right, top, bottom = 1.0, 0.4, 1.25, 0.95
    scale = min(8.6 / width_um, 6.2 / height_um)
    width_in, height_in = width_um * scale, height_um * scale
    fig_w, fig_h = left + right + width_in, top + bottom + height_in
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes(
        [left / fig_w, bottom / fig_h, width_in / fig_w, height_in / fig_h]
    )
    return fig, ax


# --------------------------------------------------------------------------------------
# C. z fly-through
# --------------------------------------------------------------------------------------


def movie_z_flythrough(
    dataset: Dataset,
    volume: np.ndarray,
    labels: Optional[np.ndarray],
    out_dir: str,
    max_frames: int,
    source: str,
) -> List[str]:
    volume = as_zyx(volume)
    labels = as_zyx(labels) if labels is not None else None
    nz, ny, nx = volume.shape
    x_um, y_um = nx * dataset.xy_um, ny * dataset.xy_um
    # One global contrast for the whole stack so brightness does not pump.
    finite = volume[np.isfinite(volume)]
    lo, hi = np.percentile(finite, [1.0, 99.7])
    if hi <= lo:
        lo, hi = float(finite.min()), float(max(finite.max(), finite.min() + 1))

    step = stride_for(nz, max_frames)
    indices = list(range(0, nz, step))
    fig, ax = movie_figure(x_um, y_um)
    frames: List[np.ndarray] = []
    for index in indices:
        ax.clear()
        plane = np.clip((volume[index].astype(np.float32) - lo) / (hi - lo), 0, 1)
        ax.imshow(
            plane,
            cmap="gray",
            vmin=0,
            vmax=1,
            origin="upper",
            extent=(0.0, x_um, y_um, 0.0),
            interpolation="nearest",
            aspect="equal",
        )
        depth_um = index * dataset.z_um
        if labels is not None:
            mask_index = int(round(depth_um / dataset.mask_um))
            mask_index = min(max(mask_index, 0), labels.shape[0] - 1)
            ax.imshow(
                boundary_rgba(labels[mask_index]),
                origin="upper",
                extent=(0.0, x_um, y_um, 0.0),
                interpolation="nearest",
                aspect="equal",
                zorder=4,
            )
        ax.set_xlabel(f"x ({MU})")
        ax.set_ylabel(f"y ({MU})")
        # Two lines: a long dataset name would otherwise run off a narrow frame.
        ax.set_title(
            f"{dataset.name}\nz = {depth_um:.2f} {MU}   (slice {index + 1}/{nz})",
            fontsize=13.5,
            linespacing=1.4,
        )
        for spine in ax.spines.values():
            spine.set_visible(False)
        add_scale_bar(ax, x_um)
        frames.append(fig_to_rgb(fig))
    plt.close(fig)

    written = []
    gif_path = os.path.join(out_dir, "z_flythrough.gif")
    avi_path = os.path.join(out_dir, "z_flythrough.avi")
    write_gif(frames, gif_path, fps=8)
    write_avi(frames, avi_path, fps=8)
    written += [gif_path, avi_path]
    print(f"    z fly-through: {len(frames)} frames (step {step}) from {source}")
    return written


# --------------------------------------------------------------------------------------
# D. t fly-through
# --------------------------------------------------------------------------------------


def movie_t_flythrough(
    dataset: Dataset, out_dir: str, max_frames: int
) -> List[str]:
    step = stride_for(len(dataset.frames), max_frames)
    indices = list(range(0, len(dataset.frames), step))
    nz, ny, nx = dataset.shape
    x_um, y_um = nx * dataset.xy_um, ny * dataset.xy_um

    fig, ax = movie_figure(x_um, y_um)
    frames: List[np.ndarray] = []
    lo = hi = None
    for index in indices:
        path = dataset.frames[index]
        volume = as_zyx(read_tiff_any(path))
        plane = volume.max(axis=0).astype(np.float32)
        del volume  # never hold more than one timepoint
        if lo is None:
            lo, hi = np.percentile(plane, [1.0, 99.7])
            if hi <= lo:
                lo, hi = float(plane.min()), float(max(plane.max(), plane.min() + 1))
        # Extents follow the frame actually read, in case a series is ragged.
        frame_x_um = plane.shape[1] * dataset.xy_um
        frame_y_um = plane.shape[0] * dataset.xy_um
        ax.clear()
        ax.imshow(
            np.clip((plane - lo) / (hi - lo), 0, 1),
            cmap="gray",
            vmin=0,
            vmax=1,
            origin="upper",
            extent=(0.0, frame_x_um, frame_y_um, 0.0),
            interpolation="nearest",
            aspect="equal",
        )
        mask_path = mask_for(dataset, path)
        count = None
        if mask_path:
            labels = as_zyx(read_tiff_any(mask_path))
            count = object_count(labels)
            ax.imshow(
                boundary_rgba(labels.max(axis=0)),
                origin="upper",
                extent=(0.0, frame_x_um, frame_y_um, 0.0),
                interpolation="nearest",
                aspect="equal",
                zorder=4,
            )
            del labels
        ax.set_xlim(0.0, frame_x_um)
        ax.set_ylim(frame_y_um, 0.0)
        elapsed = index * dataset.frame_interval_s
        title = (
            f"{dataset.name}\nt = {fmt_time(elapsed) if elapsed else '0 s'}"
            f"   (frame {index + 1}/{len(dataset.frames)})"
        )
        if count is not None:
            title += f"   —   {count} objects"
        ax.set_title(title, fontsize=13.5, linespacing=1.4)
        ax.set_xlabel(f"x ({MU})")
        ax.set_ylabel(f"y ({MU})")
        for spine in ax.spines.values():
            spine.set_visible(False)
        add_scale_bar(ax, frame_x_um)
        frames.append(fig_to_rgb(fig))
    plt.close(fig)

    gif_path = os.path.join(out_dir, "t_flythrough.gif")
    avi_path = os.path.join(out_dir, "t_flythrough.avi")
    write_gif(frames, gif_path, fps=6)
    write_avi(frames, avi_path, fps=6)
    print(f"    t fly-through: {len(frames)} frames (step {step})")
    return [gif_path, avi_path]


# --------------------------------------------------------------------------------------
# E. metric time series
# --------------------------------------------------------------------------------------


def find_results_csv(dataset: Dataset) -> Optional[str]:
    """Prefer the "(physical)" CSV -- it carries real units."""
    if not os.path.isdir(dataset.results_dir):
        return None
    physical, plain = [], []
    for folder, _dirs, names in os.walk(dataset.results_dir):
        for name in names:
            if not name.lower().endswith(".csv"):
                continue
            path = os.path.join(folder, name)
            (physical if "(physical)" in name.lower() else plain).append(path)
    pool = physical or plain
    if not pool:
        return None
    return max(pool, key=os.path.getmtime)


def read_csv_columns(path: str) -> Tuple[List[str], Dict[str, List[str]]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        columns: Dict[str, List[str]] = {name: [] for name in fields}
        for row in reader:
            for name in fields:
                columns[name].append(row.get(name, ""))
    return fields, columns


def to_float(values: Sequence[str]) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for i, value in enumerate(values):
        try:
            out[i] = float(value)
        except (TypeError, ValueError):
            pass
    return out


def figure_metrics(dataset: Dataset, csv_path: str, out_path: str) -> Optional[str]:
    fields, columns = read_csv_columns(csv_path)
    if not fields:
        return None
    n_rows = len(columns[fields[0]])
    if n_rows < 2:
        print("    metrics CSV has <2 rows, skipping time series")
        return None

    # Order rows by the numeric suffix of the file name so the x axis is real time.
    file_col = columns.get("File") or columns.get("file") or []
    order = list(range(n_rows))
    frame_index = np.arange(n_rows, dtype=float)
    if file_col:
        keys = []
        for i, value in enumerate(file_col):
            found = re.findall(r"(\d+)", os.path.basename(str(value)))
            keys.append((int(found[-1]) if found else i, i))
        keys.sort()
        order = [i for _key, i in keys]
        frame_index = np.array([float(key) for key, _i in keys])
        frame_index -= frame_index.min()

    interval = dataset.frame_interval_s or 1.0
    elapsed_s = frame_index * interval
    span = float(elapsed_s.max())
    if span >= 7200:
        time_axis, time_label = elapsed_s / 3600.0, "elapsed time (h)"
    elif span >= 180:
        time_axis, time_label = elapsed_s / 60.0, "elapsed time (min)"
    else:
        time_axis, time_label = elapsed_s, "elapsed time (s)"

    chosen: List[Tuple[str, str, np.ndarray]] = []
    for name, unit in TIMESERIES_METRICS:
        if name not in columns:
            continue
        values = to_float(columns[name])[order]
        good = np.isfinite(values)
        if good.sum() < 2 or np.nanstd(values[good]) == 0:
            continue
        chosen.append((name, unit, values))
        if len(chosen) == 4:
            break
    if not chosen:
        print("    no informative metric columns found, skipping time series")
        return None

    rows = 2 if len(chosen) > 2 else 1
    cols = 2 if len(chosen) > 1 else 1
    fig, axes = plt.subplots(
        rows, cols, figsize=(6.6 * cols, 4.3 * rows), squeeze=False, sharex=True
    )
    flat = axes.ravel()
    for ax, (name, unit, values) in zip(flat, chosen):
        good = np.isfinite(values)
        ax.plot(time_axis[good], values[good], color=SERIES_COLOR, marker="o", markersize=3.2)
        ax.set_title(name, fontsize=14)
        ax.set_ylabel(unit)
        ax.grid(True, axis="y", color="0.9", linewidth=0.8)
        ax.set_axisbelow(True)
    for ax in flat[len(chosen) :]:
        ax.set_visible(False)
    for ax in flat[max(0, len(chosen) - cols) :][:cols]:
        ax.set_xlabel(time_label)
    if rows == 2:
        for ax in flat[:cols]:
            ax.set_xlabel("")
    fig.suptitle(
        f"{dataset.name} — BARCODE metrics vs. real time", fontsize=18, fontweight="bold"
    )
    fig.text(
        0.5,
        0.005,
        f"frame interval {fmt_time(dataset.frame_interval_s)} • "
        f"source: {os.path.basename(csv_path)}",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="0.45",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------------
# cross-dataset grid
# --------------------------------------------------------------------------------------


def figure_grid(entries: List[Dict], out_path: str) -> Optional[str]:
    if not entries:
        return None
    entries = sorted(entries, key=lambda e: e["name"])
    cols = min(4, len(entries))
    rows = int(math.ceil(len(entries) / cols))
    target_aspect = 0.75  # every cell gets the same box, so nothing collides
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("white")
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.9 * rows), squeeze=False)
    flat = axes.ravel()
    for ax, entry in zip(flat, entries):
        thumb, x_um, y_um, region = pad_to_aspect(
            entry["thumb"], entry["x_um"], entry["y_um"], target_aspect
        )
        ax.imshow(
            np.ma.masked_invalid(thumb),
            cmap=cmap,
            vmin=0,
            vmax=1,
            origin="upper",
            extent=(0.0, x_um, y_um, 0.0),
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_title(entry["name"], fontsize=11.5, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        add_scale_bar(ax, entry["x_um"], region=region)
        ax.set_xlabel(
            f"{entry['xy_um']:.3g}×{entry['xy_um']:.3g}×{entry['z_um']:.3g} {MU}\n"
            f"anisotropy {entry['aniso']:.1f}× • {entry['nframes']} "
            f"{'timepoints' if entry['timelapse'] else 'fields'}",
            fontsize=10.5,
            color="0.3",
        )
    for ax in flat[len(entries) :]:
        ax.set_visible(False)
    fig.suptitle(
        "Staged open 3-D datasets — XY max projections, per-panel physical scale bar",
        fontsize=19,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def pad_to_aspect(
    image: np.ndarray, x_um: float, y_um: float, aspect: float
) -> Tuple[np.ndarray, float, float, Tuple[float, float, float, float]]:
    """Pad with NaN (drawn white) so every grid cell has the same box shape.

    Padding rather than stretching keeps the pixels square, so the per-panel scale bar
    stays physically honest.
    """
    height, width = image.shape[:2]
    px_per_um = width / x_um if x_um else 1.0
    if y_um / x_um > aspect:  # too tall -> widen
        new_x = y_um / aspect
        pad = int(round((new_x - x_um) * px_per_um / 2.0))
        image = np.pad(image, ((0, 0), (pad, pad)), constant_values=np.nan)
        offset = (new_x - x_um) / 2.0
        region = (offset, offset + x_um, 0.0, y_um)
        x_um = new_x
    else:  # too wide -> heighten
        new_y = x_um * aspect
        pad = int(round((new_y - y_um) * px_per_um / 2.0))
        image = np.pad(image, ((pad, pad), (0, 0)), constant_values=np.nan)
        offset = (new_y - y_um) / 2.0
        region = (0.0, x_um, offset, offset + y_um)
        y_um = new_y
    return image, x_um, y_um, region


def thumbnail(volume: np.ndarray, max_dim: int = 420) -> np.ndarray:
    import cv2

    flat = stretch(volume.max(axis=0))
    scale = max_dim / float(max(flat.shape))
    if scale < 1.0:
        flat = cv2.resize(
            flat, (max(int(flat.shape[1] * scale), 1), max(int(flat.shape[0] * scale), 1)),
            interpolation=cv2.INTER_AREA,
        )
    return flat


# --------------------------------------------------------------------------------------
# per-dataset driver
# --------------------------------------------------------------------------------------


def process(dataset: Dataset, fig_root: str, max_frames: int, skip_movies: bool) -> Dict:
    out_dir = os.path.join(fig_root, dataset.name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n== {dataset.name}")
    print(f"   {dataset.geometry_caption}")

    index = representative_index(dataset)
    frame_path = dataset.frames[index]
    source = os.path.basename(frame_path)
    volume = as_zyx(read_tiff_any(frame_path))
    written: List[str] = []

    written.append(
        figure_overview(dataset, volume, os.path.join(out_dir, "overview.png"), source)
    )
    print(f"    overview.png")

    mask_path = mask_for(dataset, frame_path)
    labels = as_zyx(read_tiff_any(mask_path)) if mask_path else None
    if labels is not None:
        written.append(
            figure_mask_overlay(
                dataset, volume, labels, os.path.join(out_dir, "mask_overlay.png"), source
            )
        )
        print(f"    mask_overlay.png ({object_count(labels)} objects)")
    else:
        print("    ! no matching mask, skipping overlay")

    if not skip_movies:
        written += movie_z_flythrough(dataset, volume, labels, out_dir, max_frames, source)

    entry = {
        "name": dataset.name,
        "thumb": thumbnail(volume),
        "x_um": dataset.shape[2] * dataset.xy_um,
        "y_um": dataset.shape[1] * dataset.xy_um,
        "xy_um": dataset.xy_um,
        "z_um": dataset.z_um,
        "aniso": dataset.anisotropy,
        "nframes": len(dataset.frames),
        "timelapse": dataset.is_timelapse,
    }
    del volume, labels

    if dataset.is_timelapse:
        if not skip_movies:
            written += movie_t_flythrough(dataset, out_dir, max_frames)
        csv_path = find_results_csv(dataset)
        if csv_path is None:
            print("    ! no results CSV yet, skipping metric time series")
        else:
            made = figure_metrics(
                dataset, csv_path, os.path.join(out_dir, "metrics_timeseries.png")
            )
            if made:
                written.append(made)
                print(f"    metrics_timeseries.png  <- {os.path.basename(csv_path)}")

    entry["written"] = written
    return entry


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=DEFAULT_ROOT, help="staged dataset root (data drive)")
    parser.add_argument("--out", default=None, help="figure root (default <root>/_figures)")
    parser.add_argument("--only", action="append", default=None, help="dataset name (repeatable)")
    parser.add_argument(
        "--limit", type=int, default=120, help="max frames per movie (frames are strided)"
    )
    parser.add_argument(
        "--datasets-limit", type=int, default=0, help="process at most N datasets (0 = all)"
    )
    parser.add_argument("--skip-movies", action="store_true", help="figures only")
    parser.add_argument("--skip-grid", action="store_true", help="do not write the summary grid")
    args = parser.parse_args(argv)

    if os.path.splitdrive(os.path.abspath(args.out or args.root))[0].upper().startswith("C"):
        parser.error("refusing to write outputs to the C: drive")

    apply_style()
    fig_root = args.out or os.path.join(args.root, "_figures")
    os.makedirs(fig_root, exist_ok=True)

    datasets = discover(args.root)
    if args.only:
        wanted = set(args.only)
        datasets = [d for d in datasets if d.name in wanted]
        missing = wanted - {d.name for d in datasets}
        for name in sorted(missing):
            print(f"! --only {name}: not found under {args.root}")
    if args.datasets_limit:
        datasets = datasets[: args.datasets_limit]
    if not datasets:
        print("no datasets to process")
        return 1

    entries, failures = [], []
    for dataset in datasets:
        try:
            entries.append(process(dataset, fig_root, args.limit, args.skip_movies))
        except Exception as error:  # one bad dataset must not lose the rest
            failures.append((dataset.name, repr(error)))
            print(f"    !! FAILED {dataset.name}: {error}")
            traceback.print_exc()

    if entries and not args.skip_grid:
        grid_dir = os.path.join(fig_root, "overview")
        os.makedirs(grid_dir, exist_ok=True)
        made = figure_grid(entries, os.path.join(grid_dir, "dataset_grid.png"))
        if made:
            print(f"\ngrid: {made}")

    print(f"\ndone: {len(entries)} ok, {len(failures)} failed")
    for name, error in failures:
        print(f"  FAILED {name}: {error}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
