"""Curvature-coloured nucleus meshes: one cell over time, and several cells.

Every panel shares one colour scale so they can be compared directly.
"""
import sys
sys.path.insert(0, r"C:\Users\Upadhyaya_Lab\Code\barcode")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from analysis.volumetric.curvature import analyze_curvature
from analysis.volumetric.mesh import mesh_nucleus

B = (r"F:\FF\nucleus_live_cell\jurkat_nucleus_centrosome\GFP-Centrin_SiR-DNA"
     r"\Control\cells\all_cells_together\prog_live_cells")
SC = (r"C:\Users\UPADHY~1\AppData\Local\Temp\claude"
      r"\C--Users-Upadhyaya-Lab-Code-barcode"
      r"\9f48b303-899e-4482-ab3b-afb87486e1b4\scratchpad")
PSIZE = 0.065


def build(cell, frame):
    path = rf"{B}\Cell{cell}\frame{frame}\nucleus\3D_seg\Cell_{cell}_SegMask.tif"
    mask = tifffile.imread(path) > 0
    m = mesh_nucleus(mask, (PSIZE,) * 3)
    return m, analyze_curvature(m.vertices_um, m.faces)


def draw(ax, mesh, curvature, norm, cmap, elev=18, azim=-55):
    v = mesh.vertices_um[:, ::-1]                 # (z,y,x) -> x,y,z for plotting
    ax.add_collection3d(
        Poly3DCollection(v[mesh.faces - 1],
                         facecolors=cmap(norm(curvature.k_mean_faces)),
                         edgecolor="none")
    )
    lo, hi = v.min(axis=0), v.max(axis=0)
    centre, span = (lo + hi) / 2, (hi - lo).max() / 2
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), centre):
        setter(c - span, c + span)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def finish(fig, norm, cmap, title, path):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=fig.axes, shrink=0.4, pad=0.01, aspect=28)
    cb.set_label("mean curvature (1/µm)")
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=125, bbox_inches="tight")
    print("wrote", path, flush=True)


cmap = plt.get_cmap("RdBu_r")

# ---- one cell through time -------------------------------------------------
frames = [1, 3, 5, 7, 9, 11, 13, 15]
built = [(fr, *build(11, fr)) for fr in frames]
lim = np.percentile(np.abs(np.concatenate([c.k_mean_faces for _, _, c in built])), 98)
norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

fig = plt.figure(figsize=(19, 5.6))
for i, (fr, mesh, curv) in enumerate(built, start=1):
    ax = fig.add_subplot(2, 4, i, projection="3d")
    draw(ax, mesh, curv, norm, cmap)
    ax.set_title(f"frame {fr} — vol {mesh.geometry.volume_um3:.0f} µm³, "
                 f"invag {curv.invagination_ratio:.2f}", fontsize=8.5)
finish(fig, norm, cmap,
       "Jurkat nucleus cell11 over time — surface coloured by mean curvature\n"
       "blue = concave (invagination), red = convex; shared colour scale",
       rf"{SC}\gallery_cell11_time.png")

# ---- several cells at one timepoint ---------------------------------------
cells = [1, 3, 5, 11, 12, 14]
built = []
for c in cells:
    try:
        built.append((c, *build(c, 1)))
    except Exception as exc:
        print(f"cell{c} frame1 skipped: {type(exc).__name__}: {exc}", flush=True)

lim = np.percentile(np.abs(np.concatenate([c.k_mean_faces for _, _, c in built])), 98)
norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

fig = plt.figure(figsize=(19, 6.2))
for i, (cell, mesh, curv) in enumerate(built, start=1):
    ax = fig.add_subplot(2, 3, i, projection="3d")
    draw(ax, mesh, curv, norm, cmap)
    ax.set_title(f"cell{cell} — vol {mesh.geometry.volume_um3:.0f} µm³, "
                 f"sph {mesh.geometry.sphericity:.2f}, "
                 f"invag {curv.invagination_ratio:.2f}", fontsize=9)
finish(fig, norm, cmap,
       "Six Jurkat nuclei at frame 1 — surface coloured by mean curvature\n"
       "blue = concave (invagination), red = convex; shared colour scale",
       rf"{SC}\gallery_cells.png")
