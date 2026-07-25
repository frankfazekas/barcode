"""Curvature of the apical surface, and how the furrow deepens across the five frames.

Per-cell curvature is not available here -- the cells are truncated pancakes -- but the
*surface* they form is well posed. Treating the apical height as a Monge patch z(x, y),
the mean curvature is

    H = ((1 + zx^2) zyy - 2 zx zy zxy + (1 + zy^2) zxx) / (2 (1 + zx^2 + zy^2)^{3/2})

which is exact for a height field and needs no meshing. Negative H is a groove, positive
is a ridge, so the furrow should read as a single strong negative band.

Where the tissue has invaginated below the acquired stack there is no surface to
differentiate, so those columns stay masked rather than being filled and then measured.
"""
import os
import sys

sys.path.insert(0, r"C:\Users\Upadhyaya_Lab\Code\barcode")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.colors import TwoSlopeNorm
from scipy import ndimage

STAGED = r"L:\FF\Hackathon\full_datasets\drosophila_Erika\BARCODE\staged"
OUT = r"L:\FF\Hackathon\full_datasets\drosophila_Erika\BARCODE\results\apical_surface"
XY, Z = 0.195, 0.235
FRAMES = [1, 2, 3, 4, 5]
SMOOTH_UM = 3.0          # cell-scale roughness to remove before differentiating
CLIM = 0.06              # 1/um; the tissue-scale groove, not cell-scale wrinkle

os.makedirs(OUT, exist_ok=True)


def apical(frame):
    labels = tifffile.imread(os.path.join(STAGED, "masks", f"emb_{frame}_SegMask.tif"))
    occupied = labels > 0
    z_index = np.arange(labels.shape[0])[:, None, None]
    top = np.where(occupied, z_index, -1).max(axis=0)
    valid = top >= 0
    height = np.where(valid, top * Z, 0.0).astype(np.float32)

    # Normalised (NaN-aware) smoothing: the empty furrow core must not bleed inward and
    # invent a surface where none was imaged.
    sigma = SMOOTH_UM / XY
    weight = valid.astype(np.float32)
    smooth = ndimage.gaussian_filter(height * weight, sigma) / np.maximum(
        ndimage.gaussian_filter(weight, sigma), 1e-6)
    return smooth, valid


def mean_curvature(z, valid):
    """Mean curvature of the height field, in 1/um."""
    zy, zx = np.gradient(z, XY, XY)
    zyy, zyx = np.gradient(zy, XY, XY)
    zxy, zxx = np.gradient(zx, XY, XY)
    num = (1 + zx ** 2) * zyy - 2 * zx * zy * zxy + (1 + zy ** 2) * zxx
    den = 2 * (1 + zx ** 2 + zy ** 2) ** 1.5
    H = num / den
    # Differentiation reaches a couple of pixels, so trust only well-inside columns.
    trust = ndimage.binary_erosion(valid, np.ones((9, 9), bool))
    return np.where(trust, H, np.nan)


data = {}
for f in FRAMES:
    z, valid = apical(f)
    data[f] = (z, valid, mean_curvature(z, valid))
    below = 100 * (1 - valid.mean())
    print(f"frame {f}: {below:.1f}% of columns below the stack", flush=True)

norm = TwoSlopeNorm(vmin=-CLIM, vcenter=0.0, vmax=CLIM)
cmap = plt.get_cmap("RdBu_r")
shape = data[1][0].shape
ys = np.arange(shape[0]) * XY

fig = plt.figure(figsize=(22, 11), facecolor="black")
gs = fig.add_gridspec(3, 5, height_ratios=[1.0, 1.0, 0.85], hspace=0.13, wspace=0.04)

for i, f in enumerate(FRAMES):
    z, valid, H = data[f]
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(np.where(valid, z, np.nan), cmap="turbo", vmin=0, vmax=2.9,
              interpolation="nearest")
    ax.set_title(f"t{f}", color="white", fontsize=13)
    if i == 0:
        ax.set_ylabel("apical height", color="white", fontsize=12)

    ax2 = fig.add_subplot(gs[1, i])
    ax2.imshow(H, cmap=cmap, norm=norm, interpolation="nearest")
    if i == 0:
        ax2.set_ylabel("surface curvature", color="white", fontsize=12)
    for a in (ax, ax2):
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)

# Profiles: the groove as a single curve per frame.
axh = fig.add_subplot(gs[2, :2])
axc = fig.add_subplot(gs[2, 2:4])
colours = plt.get_cmap("viridis")(np.linspace(0.15, 0.9, len(FRAMES)))
for colour, f in zip(colours, FRAMES):
    z, valid, H = data[f]
    zp = np.where(valid, z, np.nan)
    axh.plot(ys, np.nanmedian(zp, axis=1), color=colour, lw=2, label=f"t{f}")
    axc.plot(ys, np.nanmedian(H, axis=1), color=colour, lw=2)
for a, label in ((axh, "median apical height (µm)"),
                 (axc, "median surface curvature (1/µm)")):
    a.set_xlabel("dorsoventral position (µm)", color="white")
    a.set_ylabel(label, color="white")
    a.set_facecolor("black")
    a.tick_params(colors="#bbb")
    a.grid(alpha=0.25, lw=0.5, color="#555")
    for s in a.spines.values():
        s.set_color("#666")
axc.axhline(0, color="#888", lw=1, ls="--")
axh.legend(fontsize=9, ncol=5, facecolor="black", labelcolor="white", edgecolor="#444")

cax = fig.add_axes([0.915, 0.42, 0.011, 0.24])
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax)
cb.set_label("Mean Curvature (1/µm)", color="white", fontsize=12)
cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=9)
cb.outline.set_edgecolor("white")

fig.suptitle("Drosophila germband — apical surface and its curvature across 5 timepoints "
             "(blue = groove, red = ridge)", color="white", fontsize=16)
path = os.path.join(OUT, "surface_curvature_timeseries.png")
fig.savefig(path, dpi=105, facecolor="black", bbox_inches="tight")
print("wrote", path, flush=True)

print("\nfurrow depth, from the height profile:")
for f in FRAMES:
    z, valid, H = data[f]
    prof = np.nanmedian(np.where(valid, z, np.nan), axis=1)
    core = slice(int(140 / XY), int(240 / XY))
    depth = np.nanmax(prof[core]) - np.nanmin(prof[core])
    trough = ys[core][np.nanargmin(prof[core])]
    hp = np.nanmedian(H, axis=1)
    print(f"  t{f}: trough at y={trough:.0f} um, relief {depth:.2f} um, "
          f"min curvature {np.nanmin(hp):.4f} 1/um", flush=True)
