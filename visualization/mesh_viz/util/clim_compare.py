"""Compare candidate colour limits for the curvature figures, on one frame."""
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
from analysis.volumetric.mesh import face_areas

O = (r"F:\FF\nucleus_live_cell\jurkat_nucleus_centrosome\NaBu800 Experiments"
     r"\Control_3D_CD3\all_cells_together\BARCODE\results\mesh_timepoints")


def read_obj(p):
    v, f = [], []
    for line in open(p, encoding="utf-8"):
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            f.append([int(t.split("/")[0]) for t in line.split()[1:4]])
    return np.array(v), np.array(f)


V, F = read_obj(os.path.join(O, "objs", "Cell1_1.obj"))
curv = analyze_curvature(V, F, z_axis=2)
k = curv.k_mean_faces
areas = face_areas(V, F)
R_eq = 5.1085                      # equivalent-sphere radius of this nucleus

matrix = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)   # yz
centre = (V.min(0) + V.max(0)) / 2
span = float(np.abs(V - centre).max()) * 1.06 / 1.6
rotated = (V - centre) @ matrix.T

options = [
    ("current: p98, symmetric", -0.687, 0.687),
    ("fixed round, symmetric", -0.75, 0.75),
    ("asymmetric, covers furrows", -1.50, 0.75),
    ("units of 1/R_eq (x0.196)", -1.50, 0.75),      # same numbers, relabelled
]

fig = plt.figure(figsize=(19, 5.4), facecolor="black")
for i, (title, lo, hi) in enumerate(options, start=1):
    ax = fig.add_subplot(1, 4, i, projection="3d", facecolor="black")
    norm = TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)
    cmap = plt.get_cmap("RdBu_r")
    ax.set_proj_type("ortho")
    ax.add_collection3d(Poly3DCollection(rotated[F - 1],
                                         facecolors=cmap(norm(np.clip(k, lo, hi))),
                                         edgecolor="none"))
    for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        setter(-span, span)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=0, azim=-90)
    ax.set_axis_off()

    clipped = 100 * areas[(k < lo) | (k > hi)].sum() / areas.sum()
    if "R_eq" in title:
        label = f"[{lo * R_eq:.1f}, {hi * R_eq:.1f}] H*R_eq"
    else:
        label = f"[{lo:g}, {hi:g}] 1/um"
    ax.set_title(f"{title}\n{label}   clips {clipped:.1f}% of area",
                 color="white", fontsize=11)

    cax = fig.add_axes([0.055 + (i - 1) * 0.25, 0.10, 0.012, 0.30])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax)
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=8)
    cb.outline.set_edgecolor("white")

fig.suptitle("Candidate colour limits, same mesh and camera (Cell1 t1)",
             color="white", fontsize=13)
out = os.path.join(O, "movies", "_clim_options.png")
fig.savefig(out, dpi=125, facecolor="black", bbox_inches="tight")
print("wrote", out)
