"""Curvature-coloured montage + metric trajectories for the NaBu800 Cell1 series.

Meshes are read back from the OBJs written by run_mesh_timeseries.py, so the figures
are a check on the files themselves. Output goes on the data drive, never C:.
"""
import csv
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

OUT = (r"F:\FF\nucleus_live_cell\jurkat_nucleus_centrosome\NaBu800 Experiments"
       r"\Control_3D_CD3\all_cells_together\BARCODE\results\mesh_timepoints")


def read_obj(path):
    v, f = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("v "):
                v.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                f.append([int(t.split("/")[0]) for t in line.split()[1:4]])
    return np.array(v), np.array(f)


frames = list(range(1, 16))
built = []
for fr in frames:
    V, F = read_obj(os.path.join(OUT, "objs", f"Cell1_{fr}.obj"))
    built.append((fr, V, F, analyze_curvature(V, F, z_axis=2)))   # OBJ carries z in col 2
print(f"read {len(built)} meshes back from OBJ", flush=True)

lim = float(np.percentile(np.abs(np.concatenate([c.k_mean_faces for *_, c in built])), 98))
norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
cmap = plt.get_cmap("RdBu_r")

fig = plt.figure(figsize=(19, 7.2))
for i, (fr, V, F, curv) in enumerate(built, start=1):
    ax = fig.add_subplot(3, 5, i, projection="3d")
    ax.add_collection3d(
        Poly3DCollection(V[F - 1], facecolors=cmap(norm(curv.k_mean_faces)),
                         edgecolor="none")
    )
    lo, hi = V.min(axis=0), V.max(axis=0)
    centre, span = (lo + hi) / 2, (hi - lo).max() / 2
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), centre):
        setter(c - span, c + span)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-55)
    ax.set_axis_off()
    ax.set_title(f"t={fr} min  invag {curv.invagination_ratio:.2f}", fontsize=8.5)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cb = fig.colorbar(sm, ax=fig.axes, shrink=0.4, pad=0.01, aspect=30)
cb.set_label("mean curvature (1/µm)")
fig.suptitle(
    "NaBu800 Control_3D_CD3 — Cell1 nucleus, every timepoint, coloured by mean curvature\n"
    "blue = concave (invagination), red = convex; shared colour scale; rendered from the OBJs",
    fontsize=12,
)
p1 = os.path.join(OUT, "Cell1 Mesh Curvature Montage.png")
fig.savefig(p1, dpi=125, bbox_inches="tight")
print("wrote", p1, flush=True)

# ---- metric trajectories ---------------------------------------------------
with open(os.path.join(OUT, "Cell1 Mesh Timepoints.csv"), newline="", encoding="utf-8") as fh:
    rows = sorted(csv.DictReader(fh), key=lambda r: int(r["frame"]))
t = [int(r["frame"]) for r in rows]

panels = [
    ("volume_um3", "mesh volume (µm³)"),
    ("surface_area_um2", "surface area (µm²)"),
    ("sphericity", "sphericity"),
    ("height_um", "height (µm)"),
    ("invagination_ratio", "invagination ratio"),
    ("mean_curvature_inv_um", "mean curvature (1/µm)"),
]
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for ax, (col, label) in zip(axes.ravel(), panels):
    ax.plot(t, [float(r[col]) for r in rows], "-o", ms=4, color="#2b6a9a")
    ax.set_xlabel("frame (min)")
    ax.set_ylabel(label)
    ax.grid(alpha=0.25, lw=0.5)
fig.suptitle("NaBu800 Control_3D_CD3 — Cell1 mesh metrics over time", fontsize=12)
fig.tight_layout()
p2 = os.path.join(OUT, "Cell1 Mesh Trajectories.png")
fig.savefig(p2, dpi=130)
print("wrote", p2, flush=True)
