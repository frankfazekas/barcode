"""Render the nucleus mesh coloured by mean curvature and by concavity class."""
import sys
sys.path.insert(0, r"C:\Users\Upadhyaya_Lab\Code\barcode")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.colors import ListedColormap, Normalize, TwoSlopeNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from analysis.volumetric.curvature import CONCAVE, CONVEX, HYPERBOLOID, analyze_curvature
from analysis.volumetric.mesh import mesh_nucleus

B = (r"F:\FF\nucleus_live_cell\jurkat_nucleus_centrosome\GFP-Centrin_SiR-DNA"
     r"\Control\cells\all_cells_together\prog_live_cells")
OUT = (r"C:\Users\UPADHY~1\AppData\Local\Temp\claude"
       r"\C--Users-Upadhyaya-Lab-Code-barcode"
       r"\9f48b303-899e-4482-ab3b-afb87486e1b4\scratchpad")

mask = tifffile.imread(rf"{B}\Cell11\frame2\nucleus\3D_seg\Cell_11_SegMask.tif") > 0
m = mesh_nucleus(mask, (0.065, 0.065, 0.065))
r = analyze_curvature(m.vertices_um, m.faces)

V = m.vertices_um[:, ::-1]          # (z,y,x) -> plot as x,y,z
tris = V[m.faces - 1]

k = r.k_mean_faces
lim = np.percentile(np.abs(k), 98)
norm_k = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
cmap_k = plt.get_cmap("RdBu_r")

class_cmap = ListedColormap(["#c0392b", "#e0b040", "#4a80a8"])   # concave/saddle/convex

fig = plt.figure(figsize=(15, 7.2))
views = [(18, -55), (18, 125)]

for col, (elev, azim) in enumerate(views):
    for row, (facecolors, title) in enumerate([
        (cmap_k(norm_k(k)), "mean curvature"),
        (class_cmap(r.concavity_classes), "concavity class"),
    ]):
        ax = fig.add_subplot(2, 2, row * 2 + col + 1, projection="3d")
        ax.add_collection3d(
            Poly3DCollection(tris, facecolors=facecolors, edgecolor="none")
        )
        lo, hi = V.min(axis=0), V.max(axis=0)
        centre, span = (lo + hi) / 2, (hi - lo).max() / 2
        for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), centre):
            setter(c - span, c + span)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(f"{title} — view {col + 1}", fontsize=10)

sm = plt.cm.ScalarMappable(cmap=cmap_k, norm=norm_k)
cb = fig.colorbar(sm, ax=fig.axes[0:3:2], shrink=0.55, pad=0.02)
cb.set_label("mean curvature (1/µm)")

handles = [
    plt.Line2D([], [], marker="s", ls="", color=class_cmap(i), label=name)
    for i, name in [(CONCAVE, "concave"), (HYPERBOLOID, "saddle"), (CONVEX, "convex")]
]
fig.axes[3].legend(handles=handles, loc="lower left", fontsize=8, frameon=False)

fig.suptitle(
    f"Jurkat nucleus cell11 frame2 — curvature port (matches MATLAB to ~1e-16)\n"
    f"mean {r.mean_curvature:+.4f} 1/µm   invagination ratio {r.invagination_ratio:.3f}"
    f"   concave-only {r.concave_ratio:.3f}   "
    f"{int((r.concavity_classes == CONCAVE).sum())} concave / "
    f"{int((r.concavity_classes == HYPERBOLOID).sum())} saddle / "
    f"{int((r.concavity_classes == CONVEX).sum())} convex faces",
    fontsize=11,
)
path = rf"{OUT}\cell11_frame2_curvature.png"
fig.savefig(path, dpi=130, bbox_inches="tight")
print("wrote", path)
