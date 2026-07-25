"""Read a written OBJ back off disk, verify it, and render it to PNG.

Rendering from the file (not from the in-memory mesh) makes the PNG a check on the
OBJ itself: if the writer were wrong, the picture would be wrong too.
"""
import os
import shutil
import sys

sys.path.insert(0, r"C:\Users\Upadhyaya_Lab\Code\barcode")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from analysis.volumetric.curvature import analyze_curvature
from analysis.volumetric.mesh import face_areas, mesh_geometry

SRC = (r"C:\Users\UPADHY~1\AppData\Local\Temp\claude"
       r"\C--Users-Upadhyaya-Lab-Code-barcode"
       r"\9f48b303-899e-4482-ab3b-afb87486e1b4\scratchpad\objs\cell11_2.obj")
OUT = r"C:\Users\Upadhyaya_Lab\Documents\nucleus_mesh_inspect"


def read_obj(path):
    vertices, faces = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(t.split("/")[0]) for t in line.split()[1:4]])
    return np.array(vertices), np.array(faces)


os.makedirs(OUT, exist_ok=True)
obj_path = os.path.join(OUT, os.path.basename(SRC))
shutil.copy2(SRC, obj_path)

V, F = read_obj(obj_path)               # OBJ is (x, y, z) and 1-based
print(f"read back: {V.shape[0]} vertices, {F.shape[0]} faces, faces 1-based="
      f"{F.min() == 1}, max index == n_vertices={F.max() == V.shape[0]}")

# The OBJ carries z in column 2 (the package's arrays carry it in column 0).
geom = mesh_geometry(V[:, ::-1], F)     # reverse to (z, y, x) for the geometry helpers
curv = analyze_curvature(V, F, z_axis=2)
print(f"from file: volume {geom.volume_um3:.3f} um^3   SA {geom.surface_area_um2:.3f} um^2"
      f"   sphericity {geom.sphericity:.4f}   holes {geom.has_holes}")
print(f"           mean curvature {curv.mean_curvature:+.5f} 1/um"
      f"   invagination {curv.invagination_ratio:.4f}")

lim = float(np.percentile(np.abs(curv.k_mean_faces), 98))
norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
cmap = plt.get_cmap("RdBu_r")
tris = V[F - 1]

fig = plt.figure(figsize=(16, 4.6))
for i, (elev, azim) in enumerate([(18, -55), (18, 35), (18, 125), (80, -90)], start=1):
    ax = fig.add_subplot(1, 4, i, projection="3d")
    ax.add_collection3d(
        Poly3DCollection(tris, facecolors=cmap(norm(curv.k_mean_faces)),
                         edgecolor="none")
    )
    lo, hi = V.min(axis=0), V.max(axis=0)
    centre, span = (lo + hi) / 2, (hi - lo).max() / 2
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), centre):
        setter(c - span, c + span)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(f"elev {elev}°, azim {azim}°", fontsize=9)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cb = fig.colorbar(sm, ax=fig.axes, shrink=0.55, pad=0.01, aspect=26)
cb.set_label("mean curvature (1/µm)")
fig.suptitle(
    f"{os.path.basename(obj_path)} — rendered from the file on disk\n"
    f"{V.shape[0]} vertices / {F.shape[0]} faces   "
    f"volume {geom.volume_um3:.1f} µm³ (MATLAB 948.8)   "
    f"SA {geom.surface_area_um2:.1f} µm²   sphericity {geom.sphericity:.3f}   "
    f"invagination {curv.invagination_ratio:.3f}",
    fontsize=11,
)
png_path = os.path.join(OUT, "cell11_2_curvature.png")
fig.savefig(png_path, dpi=140, bbox_inches="tight")
print("OBJ:", obj_path)
print("PNG:", png_path)
