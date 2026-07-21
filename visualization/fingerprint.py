"""A one-page summary of a single analysed volume.

The barcode normalises each column across rows, so it needs a population: one timepoint
is one row and every stripe comes out a flat, uninformative colour. That is the common
case for volumetric work -- a single stack, or a single object imaged over time -- and it
is what this module exists for.

The card carries three things a barcode cannot:

* **image context** -- orthogonal projections of the volume that was actually analysed
  (after any z range and isotropic resampling), so the numbers can be checked against
  what they describe;
* **units and grouping** -- 40+ scalars are unreadable as a strip but fine as a grouped
  table;
* **the distributions behind the scalars** -- a single object still contains populations
  (per-face curvature, in-mask voxel intensities, the depth profile), and those usually
  say more than the mean that summarises them.

Group membership is read from the ``OPTIONAL_FAMILIES`` registry in ``core/results.py``
rather than a hand-written list, so a new metric family appears here without editing this
file -- the same rule that keeps the CSV, the reader and the barcode in step.

Nothing here judges a metric. Caveats come from the ``describe()`` strings the detail
objects already produce ("packing: not computed (...)"), so the card reports what the
analysis said about itself and invents no verdicts of its own.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np

from visualization.panels import (
    MU,
    add_scale_bar,
    apply_style,
    boundary_rgba,
    label_projections,
    projections,
    show_panel,
)

# Flag digits, in the order ChannelResults.convert_flags emits them. Kept here because
# nothing else in the codebase spells them out, and a bare "5;6" on a figure is noise.
FLAG_MEANINGS: Dict[str, str] = {
    "0": "none",
    "1": "dim channel",
    "2": "saturated",
    "3": "structural correlation length > field",
    "4": "velocity correlation length > field",
    "5": "partial range analysed",
    "6": "foreground clipped at field edge",
    "7": "open mesh surface",
}

# Display names for the optional families. A family missing from this map still appears,
# titled from its attribute -- the registry decides what exists, this only decides wording.
FAMILY_TITLES: Dict[str, str] = {
    "mesh": "Shape (mesh)",
    "components": "Objects",
    "packing": "Packing",
    "curvature_range": "Curvature range",
    "slice_profile": "Depth profile",
    "mask_intensity": "In-mask intensity",
    "intensity_magnitude": "Intensity (extensive)",
    "ranges": "Provenance",
}


def _fmt(value) -> str:
    """Numbers a human can scan; NaN is an em dash, not 'nan'."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "—"
    if number == 0:
        return "0"
    magnitude = abs(number)
    if 1e-3 <= magnitude < 1e5:
        return f"{number:,.4g}"
    return f"{number:.3e}"


def metric_groups(results, mode) -> List[Tuple[str, List[Tuple[str, str, str]]]]:
    """``[(group title, [(metric, value, unit), ...]), ...]`` for a populated result.

    Only groups the run actually produced are returned, so an absent family leaves no
    empty heading behind.
    """
    from core.results import OPTIONAL_FAMILIES

    groups: List[Tuple[str, List[Tuple[str, str, str]]]] = []

    def add(title: str, holder, metrics, units) -> None:
        if holder is None:
            return
        values = holder.get_data()
        # `.value` is the display name the CSV header uses; str() on the enum gives
        # "Metrics.CONNECTIVITY", which is the internal name and not what anyone reads.
        rows = [
            (getattr(metric, "value", str(metric)), _fmt(value),
             getattr(unit, "value", str(unit)))
            for metric, value, unit in zip(metrics, values, units)
        ]
        if rows:
            groups.append((title, rows))

    add("Structure", results.binarization,
        type(results.binarization).get_metrics(mode=mode),
        type(results.binarization).get_units(mode=mode))
    add("Intensity", results.intensity,
        type(results.intensity).get_metrics(), type(results.intensity).get_units())
    if mode is None or mode.supports_flow:
        add("Motion", results.flow,
            type(results.flow).get_metrics(), type(results.flow).get_units())

    for family in OPTIONAL_FAMILIES:
        holder = getattr(results, family.attribute, None)
        if holder is None or not holder.is_populated():
            continue
        title = FAMILY_TITLES.get(family.attribute, family.attribute.replace("_", " ").title())
        add(title, holder, family.results_cls.get_metrics(), family.results_cls.get_units())

    return groups


def _labels_of(mask) -> Optional[np.ndarray]:
    if mask is None:
        return None
    mask = np.asarray(mask)
    return mask.astype(np.int32) if mask.dtype != bool else mask.astype(np.int32)


def distributions(volume, mask, spacing_zyx_um, detail) -> List[Tuple[str, dict]]:
    """The populations worth plotting, each as ``(kind, payload)``.

    Only what the data supports: no curvature panel without a mesh, no contact-number
    panel with a single object. A panel that would be empty is simply not produced,
    which is why the caller can trust the count.
    """
    found: List[Tuple[str, dict]] = []
    labels = _labels_of(mask)

    # Depth profile. Prefer what the slice-profile family already computed; otherwise
    # derive it from the mask, so the panel does not depend on an opt-in family.
    areas: Optional[Sequence[float]] = None
    profile = getattr(detail, "slice_profile", None) if detail is not None else None
    if profile:
        candidate = getattr(profile[0], "areas", None)
        if candidate is not None and len(candidate):
            areas = candidate
    if areas is None and labels is not None and labels.ndim == 3:
        areas = (labels > 0).reshape(labels.shape[0], -1).mean(axis=1)
    if areas is not None and len(areas) > 1:
        depth = np.arange(len(areas)) * float(spacing_zyx_um[0])
        found.append(("depth", {"depth_um": depth, "area": np.asarray(areas, float)}))

    # Per-face curvature: a real distribution even for a single object.
    meshes = getattr(detail, "meshes", None) if detail is not None else None
    if meshes:
        curvature = getattr(meshes[0], "curvature", None)
        faces = getattr(curvature, "k_mean_faces", None) if curvature is not None else None
        if faces is not None and np.size(faces):
            found.append(("curvature", {"values": np.asarray(faces, float).ravel()}))

    # In-mask voxel intensities.
    if labels is not None and volume is not None:
        inside = np.asarray(volume)[labels > 0]
        if inside.size:
            found.append(("intensity", {"values": inside.astype(float).ravel()}))

    # Population panels, only when there is a population.
    if labels is not None:
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        sizes = counts[counts > 0]
        if sizes.size > 1:
            voxel_um3 = float(np.prod(np.asarray(spacing_zyx_um, float)))
            found.append(("sizes", {"values": sizes * voxel_um3}))

    packing = getattr(detail, "packing", None) if detail is not None else None
    if packing:
        first = packing[0]
        interior = set(getattr(first, "interior_ids", []) or [])
        ids = getattr(first, "object_ids", []) or []
        degrees = getattr(first, "contact_numbers", []) or []
        chosen = [d for i, d in zip(ids, degrees) if i in interior] or list(degrees)
        if len(chosen) > 1:
            found.append(("contacts", {"values": np.asarray(chosen, float)}))

    return found


def _draw_distribution(ax, kind: str, payload: dict) -> None:
    color = "#0072B2"
    if kind == "depth":
        ax.plot(payload["depth_um"], 100 * payload["area"], color=color)
        ax.set_xlabel(f"depth ({MU})")
        ax.set_ylabel("foreground (% of slice)")
        ax.set_title("Depth profile")
        return

    values = payload["values"]
    values = values[np.isfinite(values)]
    label = {
        "curvature": ("Curvature per face", f"mean curvature (1/{MU})"),
        "intensity": ("In-mask intensity", "intensity"),
        "sizes": ("Object volumes", f"volume ({MU}$^3$)"),
        "contacts": ("Contact number", "neighbours"),
    }[kind]
    if kind == "contacts" and values.size:
        lo, hi = int(values.min()), int(values.max())
        bins = np.arange(lo - 0.5, hi + 1.5)
    else:
        bins = 40
    ax.hist(values, bins=bins, color=color, edgecolor="white", linewidth=0.4)
    if values.size:
        ax.axvline(float(np.median(values)), color="#D55E00", linewidth=2,
                   label=f"median {_fmt(np.median(values))}")
        ax.legend(frameon=False, fontsize=9)
    ax.set_title(label[0])
    ax.set_xlabel(label[1])
    ax.set_ylabel("count")


def _notes(results, detail) -> List[str]:
    """What the analysis said about itself: flags, plus each family's own describe()."""
    lines: List[str] = []
    flags = results.convert_flags()
    described = ", ".join(
        f"{d}={FLAG_MEANINGS.get(d, 'unknown')}" for d in flags.split(";")
    )
    lines.append(f"Flags {flags}  ({described})")

    if detail is not None:
        for attribute in ("packing", "slice_profile", "mask_intensity"):
            entries = getattr(detail, attribute, None)
            if entries:
                describe = getattr(entries[0], "describe", None)
                if callable(describe):
                    lines.append(str(describe()))
    return lines


def build_fingerprint(
    volume,
    mask,
    spacing_zyx_um,
    results,
    detail=None,
    mode=None,
    title: str = "",
    figpath: Optional[str] = None,
    dpi: int = 110,
):
    """Render the card. Returns the written path, or the figure when ``figpath`` is None."""
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    apply_style()

    volume = None if volume is None else np.asarray(volume)
    labels = _labels_of(mask)
    spacing_zyx_um = tuple(float(v) for v in spacing_zyx_um)

    groups = metric_groups(results, mode)
    panels = distributions(volume, labels, spacing_zyx_um, detail)
    notes = _notes(results, detail)

    n_cols = max(len(groups), 1)
    n_panels = max(len(panels), 1)
    fig = plt.figure(figsize=(max(16, 3.1 * n_cols), 13.5))
    grid = fig.add_gridspec(
        3, 1, height_ratios=[1.15, 1.25, 0.95], hspace=0.42,
        left=0.045, right=0.985, top=0.925, bottom=0.055,
    )

    # ---- band 1: projections of the analysed volume -------------------------------
    if volume is not None and volume.ndim == 3:
        band = grid[0].subgridspec(1, 3, wspace=0.22)
        views = projections(volume)
        overlays = label_projections(labels) if labels is not None else None
        nz, ny, nx = volume.shape
        dz, dy, dx = spacing_zyx_um
        geometry = {
            "xy": (views["xy"], nx * dx, ny * dy, f"x ({MU})", f"y ({MU})"),
            "xz": (views["xz"], nx * dx, nz * dz, f"x ({MU})", f"z ({MU})"),
            "yz": (views["yz"], ny * dy, nz * dz, f"y ({MU})", f"z ({MU})"),
        }
        for column, name in enumerate(("xy", "xz", "yz")):
            image, ex, ey, xlabel, ylabel = geometry[name]
            ax = fig.add_subplot(band[0, column])
            show_panel(ax, image, ex, ey, f"{name} projection", xlabel, ylabel)
            if overlays is not None:
                ax.imshow(boundary_rgba(overlays[name]), origin="upper",
                          extent=(0.0, ex, ey, 0.0), interpolation="nearest")
            # A thin slab drawn to scale is an unreadable sliver -- the Drosophila stack
            # is 2.9 um deep against 350 um wide, which at equal aspect is a few pixels
            # high with its tick labels on top of each other. Stretch z and SAY so, the
            # usual convention; the alternative is a panel nobody can see.
            if ey / ex < 0.15:
                exaggeration = 0.25 * ex / ey
                ax.set_aspect(exaggeration)
                ax.set_title(f"{name} projection  (z × {exaggeration:.0f})", pad=8)
            if column == 0:
                add_scale_bar(ax, ex, region=(0.0, ex, 0.0, ey))

    # ---- band 2: grouped metric table ---------------------------------------------
    table = fig.add_subplot(grid[1])
    table.axis("off")
    if groups:
        width = 1.0 / len(groups)
        # Label text has to fit beside its value inside one column, and the column gets
        # narrower with every extra group. Budget characters from the actual column
        # width rather than a fixed truncation, or long names overrun their value --
        # "Maximum Island Volume [% of FOV]" against "0.0893" is unreadable.
        column_inches = fig.get_size_inches()[0] * width
        label_chars = max(14, int((column_inches * 0.62) / 0.075))
        for column, (heading, rows) in enumerate(groups):
            x = column * width
            table.text(x, 1.0, heading, transform=table.transAxes,
                       fontsize=12, fontweight="bold", va="top")
            for row, (metric, value, unit) in enumerate(rows):
                y = 0.925 - row * 0.062
                if y < -0.02:
                    table.text(x, y, "…", transform=table.transAxes, fontsize=9)
                    break
                suffix = "" if unit in ("", "None", "none") else f" [{unit}]"
                text = f"{metric}{suffix}"
                if len(text) > label_chars:
                    # Drop the unit before mangling the name; the name identifies the
                    # metric, the unit is recoverable from the CSV.
                    text = metric if len(metric) <= label_chars else metric[:label_chars - 1] + "…"
                table.text(x, y, text, transform=table.transAxes,
                           fontsize=8, va="top", color="#444444")
                table.text(x + width * 0.97, y, value, transform=table.transAxes,
                           fontsize=8, va="top", ha="right", fontweight="bold")

    # ---- band 3: distributions -----------------------------------------------------
    if panels:
        band = grid[2].subgridspec(1, n_panels, wspace=0.32)
        for column, (kind, payload) in enumerate(panels):
            _draw_distribution(fig.add_subplot(band[0, column]), kind, payload)

    fig.suptitle(title or "BARCODE volumetric fingerprint", fontsize=16, y=0.975)
    fig.text(0.045, 0.012, "   |   ".join(notes), fontsize=9, color="#333333")

    if figpath is None:
        return fig
    if not figpath.lower().endswith(".png"):
        figpath += ".png"
    os.makedirs(os.path.dirname(os.path.abspath(figpath)) or ".", exist_ok=True)
    fig.savefig(figpath, dpi=dpi, facecolor="white")
    plt.close(fig)
    return figpath
