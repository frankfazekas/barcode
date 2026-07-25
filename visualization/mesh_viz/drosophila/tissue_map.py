"""Tissue-level maps of the Drosophila epithelium, straight from the Cellpose labels.

With ~840 cells a per-cell 3D render is unreadable, so the first useful view is the
tissue itself: every cell painted by a scalar, in place. These metrics need only the
label volume, so they are available immediately -- the mesh/curvature maps come later.

One caveat drives the metric choice: the stack is 13 slices at 0.235 um, a 3.1 um slab,
so every cell is truncated in z. Apical *area* and in-plane shape are trustworthy;
anything that treats the cell as a closed 3-D body (volume, sphericity) describes a
truncated column, not a cell.
"""
import os
import sys

sys.path.insert(0, r"C:\Users\Upadhyaya_Lab\Code\barcode")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.colors import ListedColormap
from skimage.measure import regionprops_table

STAGED = r"L:\FF\Hackathon\full_datasets\drosophila_Erika\BARCODE\staged"
OUT = r"L:\FF\Hackathon\full_datasets\drosophila_Erika\BARCODE\results\tissue_maps"
XY, Z = 0.195, 0.235

os.makedirs(OUT, exist_ok=True)
frame = sys.argv[1] if len(sys.argv) > 1 else "1"
labels = tifffile.imread(os.path.join(STAGED, "masks", f"emb_{frame}_SegMask.tif"))
image = tifffile.imread(os.path.join(STAGED, f"emb_{frame}.tif"))
print(f"emb_{frame}: {labels.shape}, {labels.max()} labels", flush=True)

# Apical projection: the cell footprint seen from above.
flat = labels.max(axis=0)
props = regionprops_table(
    flat,
    properties=["label", "area", "centroid", "eccentricity", "solidity",
                "axis_major_length", "axis_minor_length", "orientation",
                "perimeter", "euler_number"],
)
n_slices = np.array([np.count_nonzero(np.any(labels == lab, axis=(1, 2)))
                     for lab in props["label"]])

area_um2 = props["area"] * XY ** 2
elong = props["axis_major_length"] / np.maximum(props["axis_minor_length"], 1e-9)
# Circularity 4*pi*A/P^2: 1 for a circle, lower for a ragged or elongated outline.
circ = 4 * np.pi * props["area"] / np.maximum(props["perimeter"], 1e-9) ** 2
depth_um = n_slices * Z

print(f"cells {len(area_um2)}   apical area med {np.median(area_um2):.2f} um^2   "
      f"elongation med {np.median(elong):.2f}   depth med {np.median(depth_um):.2f} um",
      flush=True)


def paint(values, vmin=None, vmax=None, cmap="viridis"):
    """Paint each cell's footprint with its scalar; background stays NaN."""
    lut = np.full(int(flat.max()) + 1, np.nan)
    lut[props["label"]] = values
    return lut[flat]


PANELS = [
    ("apical area", area_um2, "um$^2$", "viridis",
     np.percentile(area_um2, 2), np.percentile(area_um2, 98)),
    ("elongation (major/minor)", elong, "", "magma", 1.0, np.percentile(elong, 98)),
    ("circularity 4$\\pi$A/P$^2$", circ, "", "cividis",
     np.percentile(circ, 2), np.percentile(circ, 98)),
    ("depth spanned", depth_um, "um", "plasma",
     np.percentile(depth_um, 2), np.percentile(depth_um, 98)),
]

fig, axes = plt.subplots(2, 3, figsize=(21, 12), facecolor="black")
axes = axes.ravel()

# Reference panels: the membrane signal and the segmentation itself.
mip = image.max(axis=0).astype(np.float32)
axes[0].imshow(mip, cmap="gray",
               vmin=np.percentile(mip, 1), vmax=np.percentile(mip, 99.5))
axes[0].set_title("gap43-mCherry membrane (max projection)", color="white", fontsize=13)

rng = np.random.default_rng(0)
glasbey = np.vstack([[0, 0, 0], rng.uniform(0.25, 1.0, (int(flat.max()), 3))])
axes[1].imshow(flat, cmap=ListedColormap(glasbey), interpolation="nearest")
axes[1].set_title(f"Cellpose labels ({len(area_um2)} cells)", color="white", fontsize=13)

for ax, (name, values, unit, cmap, lo, hi) in zip(axes[2:], PANELS):
    painted = paint(values)
    im = ax.imshow(painted, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_title(f"{name}" + (f"  [{unit}]" if unit else ""), color="white", fontsize=13)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01)
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=10)
    cb.outline.set_edgecolor("#888")

for ax in axes:
    ax.set_facecolor("black")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

# One scale bar for the whole figure.
bar_um = 20.0
axes[0].plot([60, 60 + bar_um / XY], [flat.shape[0] - 70] * 2, color="white", lw=4)
axes[0].text(60 + bar_um / XY / 2, flat.shape[0] - 95, f"{bar_um:g} µm",
             color="white", ha="center", va="bottom", fontsize=12)

fig.suptitle(
    f"Drosophila germband, gap43-mCherry -- frame {frame} of 5   "
    f"({len(area_um2)} cells, 3.1 µm slab so cells are z-truncated)",
    color="white", fontsize=15)
fig.tight_layout()
path = os.path.join(OUT, f"emb_{frame}_tissue_map.png")
fig.savefig(path, dpi=110, facecolor="black")
print("wrote", path, flush=True)
