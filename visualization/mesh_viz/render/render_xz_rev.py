"""Large mean-curvature render of Cell1 timepoint 1 in the MATLAB 'xz_rev' view.

'xz_rev' is defined in TCell-3D-Morphodynamics/src/plotting/mesh_with_channel/scene/
setup_mesh_scene.m:37-52 as MATLAB view([-90 0]) with orthographic projection, equal
data aspect and axes off. MATLAB's vertices are (image-Y, image-X, image-Z), so that
camera sits at LOW image-Y looking along +image-Y: horizontal screen = image-X,
vertical screen = image-Z, depth = image-Y.

Our OBJ is (x, y, z) = (image-X, image-Y, image-Z), so the equivalent matplotlib
camera is elev=0, azim=-90 with an orthographic projection.
"""
import os
import sys

sys.path.insert(0, r"C:\Users\Upadhyaya_Lab\Code\barcode")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from analysis.volumetric.curvature import analyze_curvature
from analysis.volumetric.mesh import mesh_geometry

OUT = (r"F:\FF\nucleus_live_cell\jurkat_nucleus_centrosome\NaBu800 Experiments"
       r"\Control_3D_CD3\all_cells_together\BARCODE\results\mesh_timepoints")
OBJ = os.path.join(OUT, "objs", "Cell1_1.obj")


def read_obj(path):
    v, f = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("v "):
                v.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                f.append([int(t.split("/")[0]) for t in line.split()[1:4]])
    return np.array(v), np.array(f)


V, F = read_obj(OBJ)
curv = analyze_curvature(V, F, z_axis=2)         # OBJ carries z in column 2
geom = mesh_geometry(V[:, ::-1], F)
print(f"{os.path.basename(OBJ)}: {V.shape[0]} vertices / {F.shape[0]} faces")

k = curv.k_mean_faces
lim = float(np.percentile(np.abs(k), 98))
norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
cmap = plt.get_cmap("RdBu_r")

fig = plt.figure(figsize=(11, 11.5), facecolor="black")
ax = fig.add_axes([0.02, 0.06, 0.80, 0.86], projection="3d", facecolor="black")
ax.set_proj_type("ortho")                        # MATLAB Projection = orthographic
ax.add_collection3d(
    Poly3DCollection(V[F - 1], facecolors=cmap(norm(k)), edgecolor="none")
)

lo, hi = V.min(axis=0), V.max(axis=0)
centre, span = (lo + hi) / 2, (hi - lo).max() / 2 * 1.02
for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), centre):
    setter(c - span, c + span)
ax.set_box_aspect((1, 1, 1))                     # daspect [1 1 1]
ax.view_init(elev=0, azim=-90)                   # MATLAB view([-90 0])
ax.set_axis_off()

# Scale bar: 5 um along the horizontal screen axis (image-X), drawn at the
# bottom-front so it is not occluded by the surface.
bar = 5.0
x0 = centre[0] - span * 0.85
z0 = centre[2] - span * 0.90
ax.plot([x0, x0 + bar], [lo[1] - span * 0.5] * 2, [z0, z0],
        color="white", lw=3, zorder=10)
ax.text(x0 + bar / 2, lo[1] - span * 0.5, z0 - span * 0.06, f"{bar:g} µm",
        color="white", ha="center", va="top", fontsize=13)

cax = fig.add_axes([0.86, 0.20, 0.030, 0.58])
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax)
cb.set_label("mean curvature (1/µm)", color="white", fontsize=13)
cb.ax.yaxis.set_tick_params(color="white", labelcolor="white")
cb.outline.set_edgecolor("white")

fig.text(0.02, 0.975,
         "NaBu800 Control_3D_CD3 — Cell1, timepoint 1 — mean curvature, xz_rev view",
         color="white", fontsize=15, va="top")
fig.text(0.02, 0.945,
         f"volume {geom.volume_um3:.1f} µm³   SA {geom.surface_area_um2:.1f} µm²   "
         f"sphericity {geom.sphericity:.3f}   invagination {curv.invagination_ratio:.3f}   "
         f"{F.shape[0]} faces",
         color="0.75", fontsize=11.5, va="top")
fig.text(0.02, 0.020,
         "blue = concave (invagination)   red = convex   "
         "camera: MATLAB view([-90 0]), orthographic, daspect [1 1 1]",
         color="0.6", fontsize=10, va="bottom")

path = os.path.join(OUT, "Cell1_1 Mean Curvature xz_rev.png")
fig.savefig(path, dpi=200, facecolor="black")
print("wrote", path)
