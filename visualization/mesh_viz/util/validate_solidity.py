"""Both solidity definitions vs MATLAB's stored regionprops3 nuc_solidity."""
import csv
import sys
import time

sys.path.insert(0, r"C:\Users\Upadhyaya_Lab\Code\barcode")
import numpy as np
import tifffile

from analysis.volumetric.mesh import (
    convex_hull_voxel_count,
    convex_hull_volume,
    largest_component,
    mesh_nucleus,
)

SC = (r"C:\Users\UPADHY~1\AppData\Local\Temp\claude"
      r"\C--Users-Upadhyaya-Lab-Code-barcode"
      r"\9f48b303-899e-4482-ab3b-afb87486e1b4\scratchpad")
B = (r"F:\FF\nucleus_live_cell\jurkat_nucleus_centrosome\GFP-Centrin_SiR-DNA"
     r"\Control\cells\all_cells_together\prog_live_cells")

ml = {}
with open(rf"{SC}\matlab_mesh_grid.csv", newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        ml[(int(r["cell"]), int(r["frame"]))] = r

# The grid CSV has no solidity column; the MATLAB values come from a companion dump.
sol = {}
try:
    with open(rf"{SC}\matlab_solidity.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                sol[(int(r["cell"]), int(r["frame"]))] = float(r["ml_solidity"])
            except ValueError:
                pass
except FileNotFoundError:
    print("matlab_solidity.csv missing - run matlab_dump_solidity.m first")
    raise SystemExit(1)

cases = [k for k in sorted(sol) if np.isfinite(sol[k])]
print(f"MATLAB solidity values available: {len(cases)}")
cases = cases[: int(sys.argv[1])] if len(sys.argv) > 1 else cases

rows = []
for cell, frame in cases:
    path = rf"{B}\Cell{cell}\frame{frame}\nucleus\3D_seg\Cell_{cell}_SegMask.tif"
    mask = largest_component(tifffile.imread(path) > 0)
    t0 = time.time()
    hull_vox = convex_hull_voxel_count(mask)
    t_hull = time.time() - t0
    vox_solidity = mask.sum() / hull_vox

    m = mesh_nucleus(mask, (0.065,) * 3, solidity=False)
    hull_um3 = convex_hull_volume(m.vertices_um)
    mesh_sol = m.geometry.volume_um3 / hull_um3

    rows.append((cell, frame, sol[(cell, frame)], vox_solidity, mesh_sol, t_hull))
    print(f"  cell{cell}_{frame}: MATLAB {sol[(cell,frame)]:.6f}  "
          f"voxel {vox_solidity:.6f} ({abs(vox_solidity/sol[(cell,frame)]-1):.4%})  "
          f"mesh {mesh_sol:.6f} ({abs(mesh_sol/sol[(cell,frame)]-1):.4%})  "
          f"[hull {t_hull:.1f}s]", flush=True)

a = np.array([[r[2], r[3], r[4]] for r in rows])
for name, col in (("voxel-count hull", 1), ("mesh-geometric hull", 2)):
    rel = np.abs(a[:, col] - a[:, 0]) / a[:, 0]
    print(f"{name:22s} median {np.median(rel):.4%}  p95 {np.percentile(rel,95):.4%}  "
          f"max {rel.max():.4%}")
print(f"hull time: mean {np.mean([r[5] for r in rows]):.1f}s")
