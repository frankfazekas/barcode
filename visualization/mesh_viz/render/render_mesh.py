"""Render the ported nucleus mesh from three angles, next to the voxel mask."""
import sys
sys.path.insert(0, r"C:\Users\Upadhyaya_Lab\Code\barcode")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from analysis.volumetric.mesh import mesh_nucleus

B = (r"F:\FF\nucleus_live_cell\jurkat_nucleus_centrosome\GFP-Centrin_SiR-DNA"
     r"\Control\cells\all_cells_together\prog_live_cells")
OUT = (r"C:\Users\UPADHY~1\AppData\Local\Temp\claude"
       r"\C--Users-Upadhyaya-Lab-Code-barcode"
       r"\9f48b303-899e-4482-ab3b-afb87486e1b4\scratchpad")

mask = tifffile.imread(rf"{B}\Cell11\frame2\nucleus\3D_seg\Cell_11_SegMask.tif") > 0
mesh = mesh_nucleus(mask, (0.065, 0.065, 0.065))
g = mesh.geometry
V, F = mesh.vertices_um, mesh.faces - 1

fig = plt.figure(figsize=(15, 5.6))
views = [(20, -60), (0, 0), (89, -90)]
titles = ["oblique", "side (x-z)", "top (x-y)"]

for i, ((elev, azim), title) in enumerate(zip(views, titles), start=1):
    ax = fig.add_subplot(1, 3, i, projection="3d")
    # vertices are (z, y, x); plot as x, y, z
    tris = V[:, ::-1][F]
    coll = Poly3DCollection(tris, facecolor="#7aa6c2", edgecolor="#20303a",
                            linewidths=0.12, alpha=1.0)
    ax.add_collection3d(coll)
    lo = V[:, ::-1].min(axis=0)
    hi = V[:, ::-1].max(axis=0)
    centre = (lo + hi) / 2
    span = (hi - lo).max() / 2
    ax.set_xlim(centre[0] - span, centre[0] + span)
    ax.set_ylim(centre[1] - span, centre[1] + span)
    ax.set_zlim(centre[2] - span, centre[2] + span)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)"); ax.set_zlabel("z (µm)")

fig.suptitle(
    f"Jurkat nucleus cell11 frame2 — {g.n_vertices} vertices / {g.n_faces} faces\n"
    f"volume {g.volume_um3:.1f} µm³ (MATLAB 948.8)   "
    f"SA {g.surface_area_um2:.1f} µm²   sphericity {g.sphericity:.3f}   "
    f"height {g.height_um:.2f} µm",
    fontsize=11,
)
fig.tight_layout()
path = rf"{OUT}\cell11_frame2_mesh.png"
fig.savefig(path, dpi=130)
print("wrote", path)
