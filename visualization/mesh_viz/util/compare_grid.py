"""Compare the Python per-timepoint mesh side-car against the MATLAB results grid.
stdlib + numpy only (the barcode env has no pandas)."""
import csv
import re
from collections import defaultdict

import numpy as np

SC = (r"C:\Users\UPADHY~1\AppData\Local\Temp\claude"
      r"\C--Users-Upadhyaya-Lab-Code-barcode"
      r"\9f48b303-899e-4482-ab3b-afb87486e1b4\scratchpad")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def f(row, key):
    try:
        return float(row[key])
    except (TypeError, ValueError):
        return np.nan


py_rows = read_csv(rf"{SC}\mesh_series.csv")
ml_rows = read_csv(rf"{SC}\matlab_mesh_grid.csv")

py = {}
for r in py_rows:
    cell = int(re.search(r"(\d+)", r["series"]).group(1))
    py[(cell, int(r["frame"]))] = r

ml = {(int(r["cell"]), int(r["frame"])): r for r in ml_rows}
ml_valid = {k: r for k, r in ml.items() if np.isfinite(f(r, "ml_volume_mesh"))}

print(f"python meshes        : {len(py)}")
print(f"matlab meshes        : {len(ml_valid)} of {len(ml)} grid cells")
both = sorted(set(py) & set(ml_valid))
print(f"python-only          : {len(set(py) - set(ml_valid))}")
print(f"matlab-only          : {len(set(ml_valid) - set(py))}")
print(f"comparable timepoints: {len(both)}\n")

# MATLAB stores no surface area; derive it from the volume/sphericity pair it does.
def ml_surface_area(r):
    v, s = f(r, "ml_volume_mesh"), f(r, "ml_sphericity")
    return np.pi ** (1 / 3) * (6 * v) ** (2 / 3) / s

pairs = [
    ("volume_um3", lambda r: f(r, "ml_volume_mesh"), "mesh volume (um^3)"),
    ("surface_area_um2", ml_surface_area, "surface area (um^2)"),
    ("sphericity", lambda r: f(r, "ml_sphericity"), "sphericity"),
    ("height_um", lambda r: f(r, "ml_height"), "height (um)"),
    ("equivalent_sphere_radius_um", lambda r: f(r, "ml_sphere_rad"), "equiv sphere radius"),
    ("voxel_volume_um3", lambda r: f(r, "ml_volume_from_seg"), "voxel volume (um^3)"),
]

print(f"{'metric':<24} {'median |rel|':>13} {'p95 |rel|':>10} {'max |rel|':>10}"
      f" {'py mean':>10} {'ml mean':>10}")
for py_col, ml_get, label in pairs:
    a = np.array([f(py[k], py_col) for k in both])
    b = np.array([ml_get(ml[k]) for k in both])
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 0)
    rel = np.abs(a[ok] - b[ok]) / np.abs(b[ok])
    print(f"{label:<24} {np.median(rel):12.4%} {np.percentile(rel, 95):9.4%}"
          f" {rel.max():9.4%} {a[ok].mean():10.4f} {b[ok].mean():10.4f}")

rel_v = np.array([
    abs(f(py[k], "volume_um3") - f(ml[k], "ml_volume_mesh")) / f(ml[k], "ml_volume_mesh")
    for k in both
])
print("\nworst 5 by volume disagreement:")
for idx in np.argsort(rel_v)[::-1][:5]:
    cell, frame = both[idx]
    r = py[(cell, frame)]
    print(f"  cell{cell}_{frame}: py {f(r,'volume_um3'):8.3f}  "
          f"ml {f(ml[(cell,frame)],'ml_volume_mesh'):8.3f}  "
          f"rel {rel_v[idx]:.3%}  faces {r['n_faces']}  holes {r['has_holes']}")

print("\nmesh health across all python rows:")
holes = sum(int(r["has_holes"]) for r in py_rows)
flipped = sum(int(r["faces_flipped"] or 0) for r in py_rows)
ratio = np.array([f(r, "volume_ratio") for r in py_rows])
invag = np.array([f(r, "invagination_ratio") for r in py_rows])
curv = np.array([f(r, "mean_curvature_inv_um") for r in py_rows])
print(f"  meshes with holes : {holes} of {len(py_rows)}")
print(f"  faces flipped     : {flipped}")
print(f"  volume ratio      : mean {ratio.mean():.4f} "
      f"min {ratio.min():.4f} max {ratio.max():.4f}")
print(f"  invagination ratio: mean {invag.mean():.4f} "
      f"range [{invag.min():.4f}, {invag.max():.4f}]")
print(f"  mean curvature    : mean {curv.mean():+.4f} "
      f"range [{curv.min():+.4f}, {curv.max():+.4f}] 1/um")

# A nucleus should not jump in volume between consecutive minutes.
series = defaultdict(list)
for r in py_rows:
    series[r["series"]].append((int(r["frame"]), f(r, "volume_um3")))
steps = []
for values in series.values():
    v = np.array([x for _, x in sorted(values)])
    if v.size > 1:
        steps.append(np.abs(np.diff(v)) / v[:-1])
steps = np.concatenate(steps)
print(f"\nframe-to-frame volume change within a series: median {np.median(steps):.3%}"
      f"  p95 {np.percentile(steps, 95):.3%}  max {steps.max():.3%}")
