"""PowerPoint-ready exports of the Cell1 curvature meshes.

Produces, all from the OBJs already on disk so the figures match the widget exactly:

  hero        one large 300-dpi still, transparent background so it drops onto any slide
  timelapse   animated GIF scrubbing t1..t15 (PowerPoint plays GIFs natively)
  rotation    animated GIF spinning one timepoint, for a "here is the 3D shape" slide
  strip       a wide 16:9 filmstrip of selected timepoints for a static slide

Everything is written next to the data on F:, never to C: (CLAUDE.md).
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
from PIL import Image

from analysis.volumetric.curvature import analyze_curvature

BASE = (r"F:\FF\nucleus_live_cell\jurkat_nucleus_centrosome\NaBu800 Experiments"
        r"\Control_3D_CD3\all_cells_together\BARCODE\results\mesh_timepoints")
OUT = os.path.join(BASE, "ppt_export")
FRAMES = list(range(1, 16))
AZIM = {"xz": 90, "xz_rev": -90, "yz": 180, "yz_rev": 0}


def read_obj(path):
    v, f = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("v "):
                v.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                f.append([int(t.split("/")[0]) for t in line.split()[1:4]])
    return np.array(v), np.array(f)


os.makedirs(OUT, exist_ok=True)
meshes = []
for fr in FRAMES:
    V, F = read_obj(os.path.join(BASE, "objs", f"Cell1_{fr}.obj"))
    meshes.append((fr, V, F, analyze_curvature(V, F, z_axis=2)))
print(f"loaded {len(meshes)} meshes", flush=True)

# One shared frame and colour scale, exactly as the widget uses.
all_v = np.concatenate([m[1] for m in meshes])
centre = (all_v.min(axis=0) + all_v.max(axis=0)) / 2
span = float(np.abs(all_v - centre).max()) * 1.05
lim = float(np.percentile(np.abs(np.concatenate([m[3].k_mean_faces for m in meshes])), 98))
norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
cmap = plt.get_cmap("RdBu_r")
print(f"shared colour limit +/-{lim:.4f} 1/um", flush=True)


def panel(ax, V, F, curv, azim, elev=0, transparent=False):
    ax.set_proj_type("ortho")
    ax.add_collection3d(
        Poly3DCollection(V[F - 1], facecolors=cmap(norm(curv.k_mean_faces)),
                         edgecolor="none")
    )
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), centre):
        setter(c - span, c + span)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    if not transparent:
        ax.set_facecolor("black")


def scalebar(ax, bar=5.0, colour="white"):
    from matplotlib.lines import Line2D
    frac = bar / (2 * span)
    ax.add_line(Line2D([0.05, 0.05 + frac], [0.05, 0.05], transform=ax.transAxes,
                       color=colour, lw=3.5, zorder=100, solid_capstyle="butt"))
    ax.text2D(0.05 + frac / 2, 0.025, f"{bar:g} µm", transform=ax.transAxes,
              color=colour, ha="center", va="top", fontsize=14, zorder=100)


# --------------------------------------------------------------- hero still
fr, V, F, curv = meshes[0]
fig = plt.figure(figsize=(9, 9), facecolor="none")
ax = fig.add_axes([0.0, 0.0, 0.86, 1.0], projection="3d", facecolor="none")
panel(ax, V, F, curv, AZIM["yz"], transparent=True)
scalebar(ax, colour="#222222")
cax = fig.add_axes([0.88, 0.22, 0.030, 0.56])
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax)
cb.set_label("mean curvature (1/µm)", fontsize=13, color="#222222")
cb.ax.yaxis.set_tick_params(color="#222222", labelcolor="#222222")
cb.outline.set_edgecolor("#666666")
hero = os.path.join(OUT, "hero_t1_curvature_transparent.png")
fig.savefig(hero, dpi=300, transparent=True)
plt.close(fig)
print("wrote", os.path.basename(hero), flush=True)

# --------------------------------------------------------------- filmstrip 16:9
picks = [1, 3, 5, 7, 9, 11, 13, 15]
fig = plt.figure(figsize=(13.333, 7.5), facecolor="black")   # exact 16:9 slide
for i, fr in enumerate(picks):
    _, V, F, curv = meshes[fr - 1]
    ax = fig.add_subplot(2, 4, i + 1, projection="3d", facecolor="black")
    panel(ax, V, F, curv, AZIM["yz"])
    ax.set_title(f"t = {fr - 1} min", color="white", fontsize=13, pad=0)
cax = fig.add_axes([0.93, 0.28, 0.014, 0.44])
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax)
cb.set_label("mean curvature (1/µm)", color="white", fontsize=12)
cb.ax.yaxis.set_tick_params(color="white", labelcolor="white")
cb.outline.set_edgecolor("white")
fig.subplots_adjust(left=0.01, right=0.91, top=0.94, bottom=0.02, wspace=0.0, hspace=0.10)
strip = os.path.join(OUT, "filmstrip_16x9.png")
fig.savefig(strip, dpi=200, facecolor="black")
plt.close(fig)
print("wrote", os.path.basename(strip), flush=True)


def frames_to_gif(images, path, duration_ms):
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=duration_ms, loop=0, optimize=True)
    print(f"wrote {os.path.basename(path)} ({len(images)} frames, "
          f"{os.path.getsize(path) / 1e6:.1f} MB)", flush=True)


def render_rgb(V, F, curv, azim, elev=0, size=6.0, dpi=110):
    fig = plt.figure(figsize=(size, size), facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1], projection="3d", facecolor="black")
    panel(ax, V, F, curv, azim, elev)
    scalebar(ax)
    fig.canvas.draw()
    image = Image.frombytes("RGBA", fig.canvas.get_width_height(),
                            bytes(fig.canvas.buffer_rgba())).convert("RGB")
    plt.close(fig)
    return image


# --------------------------------------------------------------- timelapse GIF
images = []
for fr, V, F, curv in meshes:
    images.append(render_rgb(V, F, curv, AZIM["yz"]))
frames_to_gif(images, os.path.join(OUT, "timelapse_t1-15_yz.gif"), 400)

# --------------------------------------------------------------- rotation GIF
_, V, F, curv = meshes[0]
images = [render_rgb(V, F, curv, a) for a in range(0, 360, 10)]
frames_to_gif(images, os.path.join(OUT, "rotation_t1.gif"), 80)

print("\nall exports ->", OUT)
