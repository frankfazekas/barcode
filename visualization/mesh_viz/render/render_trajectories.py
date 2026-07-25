"""Per-timepoint mesh metrics over time: Python side-car vs the MATLAB grid."""
import csv
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SC = (r"C:\Users\UPADHY~1\AppData\Local\Temp\claude"
      r"\C--Users-Upadhyaya-Lab-Code-barcode"
      r"\9f48b303-899e-4482-ab3b-afb87486e1b4\scratchpad")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def num(row, key):
    try:
        return float(row[key])
    except (TypeError, ValueError):
        return np.nan


py = defaultdict(dict)
for r in read_csv(rf"{SC}\mesh_series.csv"):
    cell = int(re.search(r"(\d+)", r["series"]).group(1))
    py[cell][int(r["frame"])] = r

ml = defaultdict(dict)
for r in read_csv(rf"{SC}\matlab_mesh_grid.csv"):
    ml[int(r["cell"])][int(r["frame"])] = r

cells = sorted(py)
cmap = plt.get_cmap("turbo", max(len(cells), 2))

panels = [
    ("volume_um3", "ml_volume_mesh", "mesh volume (µm³)"),
    ("sphericity", "ml_sphericity", "sphericity"),
    ("height_um", "ml_height", "height (µm)"),
    ("invagination_ratio", None, "invagination ratio"),
    ("mean_curvature_inv_um", None, "mean curvature (1/µm)"),
    ("surface_area_um2", None, "surface area (µm²)"),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
for ax, (py_col, ml_col, label) in zip(axes.ravel(), panels):
    for i, cell in enumerate(cells):
        frames = sorted(py[cell])
        ax.plot(frames, [num(py[cell][f], py_col) for f in frames],
                "-o", ms=2.5, lw=1.1, color=cmap(i), label=f"cell{cell}")
        if ml_col:
            mf = [f for f in frames if f in ml[cell] and np.isfinite(num(ml[cell][f], ml_col))]
            if mf:
                ax.plot(mf, [num(ml[cell][f], ml_col) for f in mf],
                        "x", ms=4, mew=0.9, color="k", alpha=0.55)
    ax.set_xlabel("frame (min)")
    ax.set_ylabel(label)
    ax.set_title(label + ("   (× = MATLAB)" if ml_col else "   (no MATLAB reference)"),
                 fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)

axes[0, 0].legend(fontsize=6, ncol=2, frameon=False, loc="best")
fig.suptitle(
    "Per-timepoint nucleus meshes — Python side-car (lines) vs MATLAB pipeline (black ×)\n"
    f"{len(cells)} cells, {sum(len(v) for v in py.values())} timepoints",
    fontsize=12,
)
fig.tight_layout()
path = rf"{SC}\mesh_trajectories.png"
fig.savefig(path, dpi=130)
print("wrote", path)
