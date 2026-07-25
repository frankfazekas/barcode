"""Mesh a strip across the ventral furrow and render it in 3D, curvature-coloured.

All ~800 cells at once is unreadable; the informative object is the furrow and the
elongated cells flanking it, which the DV profiles place at y ~ 150-250 um. This meshes
that strip finely and shows the apical topography.

maxrad is 1.2 voxels, not the 5 that suits a nucleus. These cells sit in a 16-slice
isotropic slab, and a triangle bound that is a large fraction of an object's *thinnest*
dimension collapses the surface: at 2.5 voxels, 87 of 784 cells meshed to ~0 volume.
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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from analysis.volumetric.mesh_field import mesh_field

STAGED = r"L:\FF\Hackathon\full_datasets\drosophila_Erika\BARCODE\staged"
OUT = r"L:\FF\Hackathon\full_datasets\drosophila_Erika\BARCODE\results\furrow_3d"
XY, Z = 0.195, 0.235
CLIM = 1.5      # 1/um; cells are ~8 um across, so curvature runs higher than a nucleus

os.makedirs(OUT, exist_ok=True)
frame = sys.argv[1] if len(sys.argv) > 1 else "1"
y_lo, y_hi = (float(v) for v in (sys.argv[2:4] if len(sys.argv) > 3 else (150, 250)))
x_lo, x_hi = (float(v) for v in (sys.argv[4:6] if len(sys.argv) > 5 else (60, 175)))
y0, y1 = int(y_lo / XY), int(y_hi / XY)
x0, x1 = int(x_lo / XY), int(x_hi / XY)

labels = tifffile.imread(os.path.join(STAGED, "masks", f"emb_{frame}_SegMask.tif"))
n_z = int(round(labels.shape[0] * Z / XY))
index = np.clip(np.round(np.linspace(0, labels.shape[0] - 1, n_z)).astype(int),
                0, labels.shape[0] - 1)
strip = labels[index][:, y0:y1, x0:x1]
print(f"strip {strip.shape} = {y_hi - y_lo:.0f} x {x_hi - x_lo:.0f} um, "
      f"{len(np.unique(strip)) - 1} labels", flush=True)

field = mesh_field(strip, (XY,) * 3, maxrad=1.2, min_voxels=400,
                   curvature=True, exclude_border="xy")
print(f"meshed {len(field)} cells", flush=True)
for line in field.describe():
    print("   ", line, flush=True)
if not field.meshes:
    raise SystemExit("no cells meshed")

# Drop any mesh that collapsed, rather than letting it show as a dark speck.
good = [m for m in field.meshes if m.geometry.volume_ratio > 0.6]
print(f"kept {len(good)} of {len(field)} (volume ratio > 0.6)", flush=True)

norm = TwoSlopeNorm(vmin=-CLIM, vcenter=0.0, vmax=CLIM)
cmap = plt.get_cmap("RdBu_r")
all_v = np.concatenate([m.vertices_um for m in good])
centre = (all_v.min(0) + all_v.max(0)) / 2
span = float(np.abs(all_v - centre).max())

VIEWS = [(34, -62, "oblique"), (4, -90, "side view (x-z)"), (80, -90, "from above")]
fig = plt.figure(figsize=(21, 7.6), facecolor="black")
for i, (elev, azim, title) in enumerate(VIEWS, start=1):
    ax = fig.add_subplot(1, 3, i, projection="3d", facecolor="black")
    ax.set_proj_type("ortho")
    for m in good:
        v = m.vertices_um[:, ::-1] - centre[::-1]        # (z, y, x) -> plot as x, y, z
        ax.add_collection3d(Poly3DCollection(
            v[m.faces - 1],
            facecolors=cmap(norm(np.clip(m.curvature.k_mean_faces, -CLIM, CLIM))),
            edgecolor="none"))
    for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        setter(-span, span)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, color="white", fontsize=13)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cb = fig.colorbar(sm, ax=fig.axes, shrink=0.5, pad=0.01, aspect=26)
cb.set_label("Mean Curvature (1/µm)", color="white", fontsize=14)
cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=11)
cb.outline.set_edgecolor("white")

fig.suptitle(
    f"Drosophila ventral furrow, frame {frame} — {len(good)} cells meshed, "
    f"coloured by surface curvature",
    color="white", fontsize=15)
path = os.path.join(OUT, f"emb_{frame}_furrow_3d.png")
fig.savefig(path, dpi=115, facecolor="black", bbox_inches="tight")
print("wrote", path, flush=True)
