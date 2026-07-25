"""Mesh Kiet coculture cell 1, colour by per-face mean curvature, save PNG + OBJ.

Object mask = (Simple Segmentation == 1). Calibration is USER-CORRECTED:
xy = 0.065 um/px (the file's dCalibration 0.1625 is wrong), z = 0.3 um.
Outputs go to the data drive (Y:), never C:.
"""
import os, sys
import numpy as np
import tifffile
from scipy import ndimage as ndi

sys.path.insert(0, r"C:\Users\Upadhyaya_Lab\Code\barcode")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from analysis.volumetric.mesh import mesh_object, write_obj, largest_component
from analysis.volumetric.curvature import analyze_curvature

SEG = r"Y:\User_data\Kiet\03102026_coculture_hek_CD19gfp_Hoescht_CAR_HA_APC\FMC_30min_\converted\cells\channels\C2-cell1_Simple Segmentation.tiff"
OUT = r"Y:\User_data\Kiet\03102026_coculture_hek_CD19gfp_Hoescht_CAR_HA_APC\FMC_30min_\converted\cells\channels\mesh_test"
XY, Z = 0.065, 0.3
os.makedirs(OUT, exist_ok=True)

# 1. object mask
seg = tifffile.imread(SEG)
mask = largest_component(seg == 1)
print(f"mask voxels {int(mask.sum())}  shape {mask.shape}")

# 2. isotropic resample to 0.065 um (nearest-neighbour on the label)
zf = Z / XY
mask_iso = ndi.zoom(mask.astype(np.uint8), (zf, 1, 1), order=0).astype(bool)
print(f"iso shape {mask_iso.shape}  (z x{zf:.3f})")

# 3. mesh
om = mesh_object(mask_iso, (XY, XY, XY), maxrad=5.0, verbose=True)
g = om.geometry
print(f"\nvertices {g.n_vertices}  faces {g.n_faces}  holes {g.has_holes}")
print(f"volume {g.volume_um3:.1f} um^3  (voxel {g.voxel_volume_um3:.1f})")
print(f"surface {g.surface_area_um2:.1f} um^2  sphericity {g.sphericity:.3f}")
print(f"solidity {g.solidity:.3f}  aspect ratio {g.aspect_ratio:.3f}")

# 4. curvature per face
cv = analyze_curvature(om.vertices_um, om.faces, z_axis=0)
k = cv.k_mean_faces
print(f"mean-curv per face: p5 {np.percentile(k,5):.2f}  med {np.median(k):.2f}  "
      f"p95 {np.percentile(k,95):.2f}  (1/um)")

# 5. OBJ
write_obj(os.path.join(OUT, "cell1.obj"), om.vertices_um, om.faces)

# 6. render 4 orientations, coloured by clipped mean curvature
V, F = om.vertices_um, om.faces
tris = V[F - 1]                       # (nfaces, 3, 3) in (z,y,x)
# plot as (x, y, z)
tris_xyz = tris[..., ::-1]
centre = V[:, ::-1].mean(axis=0)
span = np.ptp(V[:, ::-1], axis=0).max() / 2 * 1.1

CLIM = 0.5
norm = TwoSlopeNorm(vmin=-CLIM, vcenter=0.0, vmax=CLIM)
cmap = plt.cm.coolwarm
colors = cmap(norm(np.clip(k, -CLIM, CLIM)))

views = [("front", 10, -90), ("side", 10, 0), ("top", 88, -90), ("oblique", 25, -55)]
fig = plt.figure(figsize=(11, 11), facecolor="white")
for i, (name, elev, azim) in enumerate(views):
    ax = fig.add_subplot(2, 2, i + 1, projection="3d", facecolor="white")
    ax.set_proj_type("ortho")
    ax.add_collection3d(Poly3DCollection(tris_xyz, facecolors=colors, edgecolor="none"))
    for lo, hi, s in zip(centre - span, centre + span,
                         (ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        s(lo, hi)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(name, fontsize=14)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cax = fig.add_axes([0.35, 0.05, 0.3, 0.02])
cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
cb.set_label("Mean curvature (1/µm)", fontsize=13)
fig.suptitle(f"C2-cell1   V={g.volume_um3:.0f} µm³   sphericity={g.sphericity:.2f}   "
             f"solidity={g.solidity:.2f}   AR={g.aspect_ratio:.2f}", fontsize=14)
png = os.path.join(OUT, "cell1_curvature.png")
fig.savefig(png, dpi=150, facecolor="white", bbox_inches="tight")
print("\nwrote", png)
print("wrote", os.path.join(OUT, "cell1.obj"))
