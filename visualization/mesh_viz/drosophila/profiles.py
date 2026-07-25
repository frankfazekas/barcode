"""Dorsoventral profiles of cell shape across all five Drosophila frames.

The tissue maps show a band of large, elongated cells across the middle. Rather than
outlining it by eye and calling it the furrow, this plots each metric against position
along the dorsoventral axis: the band then appears as a peak that is either there or
not, with its width and amplitude measurable and comparable between timepoints.

All metrics here come from the label volume alone -- apical footprint and in-plane
shape -- because the 3.1 um slab truncates every cell in z, so closed-body quantities
would describe a truncated column rather than a cell.
"""
import os
import sys

sys.path.insert(0, r"C:\Users\Upadhyaya_Lab\Code\barcode")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from skimage.measure import regionprops_table

STAGED = r"L:\FF\Hackathon\full_datasets\drosophila_Erika\BARCODE\staged"
OUT = r"L:\FF\Hackathon\full_datasets\drosophila_Erika\BARCODE\results\tissue_maps"
XY, Z = 0.195, 0.235
FRAMES = [1, 2, 3, 4, 5]

os.makedirs(OUT, exist_ok=True)


def measure(frame):
    labels = tifffile.imread(os.path.join(STAGED, "masks", f"emb_{frame}_SegMask.tif"))
    flat = labels.max(axis=0)
    props = regionprops_table(
        flat, properties=["label", "area", "centroid", "axis_major_length",
                          "axis_minor_length", "perimeter", "orientation"])
    depth = np.array([np.count_nonzero(np.any(labels == lab, axis=(1, 2)))
                      for lab in props["label"]]) * Z
    return {
        "y_um": props["centroid-0"] * XY,
        "x_um": props["centroid-1"] * XY,
        "area": props["area"] * XY ** 2,
        "elong": props["axis_major_length"] / np.maximum(props["axis_minor_length"], 1e-9),
        "circ": 4 * np.pi * props["area"] / np.maximum(props["perimeter"], 1e-9) ** 2,
        "depth": depth,
        # Orientation relative to the DV axis: 0 = aligned with the furrow (mediolateral),
        # 90 = perpendicular. Cells intercalating in the germband align systematically.
        "angle": np.abs(np.degrees(props["orientation"])),
        "shape": flat.shape,
    }


data = {f: measure(f) for f in FRAMES}
for f in FRAMES:
    d = data[f]
    print(f"frame {f}: {len(d['area'])} cells, area med {np.median(d['area']):.1f} um^2, "
          f"elong med {np.median(d['elong']):.2f}", flush=True)


def profile(d, key, bins):
    """Median of ``key`` in each DV bin, with the interquartile band."""
    idx = np.digitize(d["y_um"], bins) - 1
    med, lo, hi, n = [], [], [], []
    for b in range(len(bins) - 1):
        v = d[key][idx == b]
        if v.size < 5:
            med.append(np.nan); lo.append(np.nan); hi.append(np.nan); n.append(v.size)
            continue
        med.append(np.median(v)); lo.append(np.percentile(v, 25))
        hi.append(np.percentile(v, 75)); n.append(v.size)
    return np.array(med), np.array(lo), np.array(hi), np.array(n)


extent = data[1]["shape"][0] * XY
bins = np.linspace(0, extent, 41)
centres = (bins[:-1] + bins[1:]) / 2

METRICS = [
    ("area", "apical area (µm$^2$)"),
    ("elong", "elongation (major/minor)"),
    ("circ", "circularity 4$\\pi$A/P$^2$"),
    ("depth", "depth spanned (µm)"),
]
colours = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(FRAMES)))

fig, axes = plt.subplots(1, 4, figsize=(22, 5.4), facecolor="white")
for ax, (key, label) in zip(axes, METRICS):
    for colour, f in zip(colours, FRAMES):
        med, lo, hi, n = profile(data[f], key, bins)
        ax.plot(centres, med, color=colour, lw=2, label=f"t{f}")
        if f == FRAMES[0]:
            ax.fill_between(centres, lo, hi, color=colour, alpha=0.18, lw=0)
    ax.set_xlabel("position along dorsoventral axis (µm)")
    ax.set_ylabel(label)
    ax.grid(alpha=0.25, lw=0.5)
axes[0].legend(title="frame", fontsize=9, ncol=2)
axes[0].text(0.02, 0.97, "shaded band = IQR at t1", transform=axes[0].transAxes,
             va="top", fontsize=9, color="#555")

fig.suptitle("Drosophila germband: cell shape along the dorsoventral axis, 5 timepoints",
             fontsize=15)
fig.tight_layout()
path = os.path.join(OUT, "dv_profiles.png")
fig.savefig(path, dpi=120, facecolor="white")
print("wrote", path, flush=True)

# Where is the furrow, objectively? The elongation peak, located per frame.
print("\nfurrow position from the elongation profile:")
for f in FRAMES:
    med, _, _, n = profile(data[f], "elong", bins)
    valid = np.isfinite(med)
    peak = centres[valid][np.argmax(med[valid])]
    baseline = np.nanmedian(med[valid])
    print(f"  t{f}: peak elongation {np.nanmax(med):.2f} at y = {peak:.1f} um "
          f"(baseline {baseline:.2f}, {np.nanmax(med) / baseline:.2f}x)", flush=True)
