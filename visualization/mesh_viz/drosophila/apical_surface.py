"""The apical surface of the Drosophila germband, as a height field.

Why not per-cell meshes: the stack is 13 slices at 0.235 um, so a cell that is ~10 um
wide is only ~3 um deep and is cut top and bottom by the acquisition. Meshing one gives
a pancake whose curvature is dominated by the artificial rim, not by its shape -- the
side view in furrow_3d makes that obvious.

What the slab *does* capture honestly is where the apical surface sits. The furrow is an
invagination, so the tissue surface dips; that dip is real 3-D morphology and is visible
even in a shallow stack. This reconstructs the apical height z(x, y) from the
segmentation, smooths it, and renders it as a surface -- which is the shape the embryo
is actually making.
"""
import os
import sys

sys.path.insert(0, r"C:\Users\Upadhyaya_Lab\Code\barcode")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage

STAGED = r"L:\FF\Hackathon\full_datasets\drosophila_Erika\BARCODE\staged"
OUT = r"L:\FF\Hackathon\full_datasets\drosophila_Erika\BARCODE\results\apical_surface"
XY, Z = 0.195, 0.235

os.makedirs(OUT, exist_ok=True)
frame = sys.argv[1] if len(sys.argv) > 1 else "1"
labels = tifffile.imread(os.path.join(STAGED, "masks", f"emb_{frame}_SegMask.tif"))
image = tifffile.imread(os.path.join(STAGED, f"emb_{frame}.tif")).astype(np.float32)
occupied = labels > 0
print(f"emb_{frame}: {labels.shape}", flush=True)

# Apical height: the topmost segmented slice at each (x, y). Where the tissue has
# invaginated below the acquired stack the column is empty, and that absence is itself
# the signal -- it is masked rather than filled with a guess.
z_index = np.arange(labels.shape[0])[:, None, None]
top = np.where(occupied, z_index, -1).max(axis=0).astype(np.float32)
valid = top >= 0
height = np.where(valid, top * Z, np.nan)
print(f"columns with tissue: {100 * valid.mean():.1f}%", flush=True)

# Smooth over the cell-scale roughness so the tissue-scale shape is visible. NaNs are
# handled by normalised convolution, so the empty furrow core does not bleed inward.
sigma = 6.0 / XY * 0.25
filled = np.nan_to_num(height)
weight = valid.astype(np.float32)
smooth = ndimage.gaussian_filter(filled, sigma) / np.maximum(
    ndimage.gaussian_filter(weight, sigma), 1e-6)
smooth[~valid] = np.nan

# Downsample for a renderable surface.
step = 8
hs = smooth[::step, ::step]
mip = image.max(axis=0)[::step, ::step]
ys = np.arange(hs.shape[0]) * step * XY
xs = np.arange(hs.shape[1]) * step * XY
X, Y = np.meshgrid(xs, ys)

fig = plt.figure(figsize=(21, 8), facecolor="black")

ax = fig.add_subplot(1, 3, 1)
im = ax.imshow(height, cmap="turbo", interpolation="nearest")
ax.set_title("apical height z(x, y)", color="white", fontsize=13)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01)
cb.set_label("µm above the lowest slice", color="white", fontsize=11)
cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=9)

ax2 = fig.add_subplot(1, 3, 2)
ax2.imshow(mip, cmap="gray", vmin=np.percentile(mip, 1), vmax=np.percentile(mip, 99.5))
missing = np.ma.masked_where(valid[::step, ::step], np.ones_like(hs))
ax2.imshow(missing, cmap="autumn", alpha=0.55, interpolation="nearest")
ax2.set_title("membrane, with tissue below the stack in red", color="white", fontsize=13)

ax3 = fig.add_subplot(1, 3, 3, projection="3d", facecolor="black")
norm = plt.Normalize(np.nanpercentile(hs, 2), np.nanpercentile(hs, 98))
ax3.plot_surface(X, Y, np.nan_to_num(hs, nan=np.nanmin(hs)),
                 facecolors=plt.get_cmap("turbo")(norm(hs)),
                 rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
ax3.set_box_aspect((1.0, 0.85, 0.22))
ax3.view_init(elev=52, azim=-72)
ax3.set_xlabel("x (µm)", color="white"); ax3.set_ylabel("y (µm)", color="white")
ax3.set_zlabel("z (µm)", color="white")
for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
    axis.set_pane_color((0, 0, 0, 1.0))
    axis.line.set_color("#666")
ax3.tick_params(colors="#aaa", labelsize=8)
ax3.set_title("apical surface in 3D", color="white", fontsize=13)

for a in (ax, ax2):
    a.set_xticks([]); a.set_yticks([])
    for spine in a.spines.values():
        spine.set_visible(False)

fig.suptitle(
    f"Drosophila germband, frame {frame} — apical surface topography "
    f"(13-slice stack, so the furrow core drops out of the acquired volume)",
    color="white", fontsize=15)
fig.tight_layout()
path = os.path.join(OUT, f"emb_{frame}_apical_surface.png")
fig.savefig(path, dpi=115, facecolor="black")
print("wrote", path, flush=True)
