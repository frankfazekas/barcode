"""Mesh real nuclei, run the Python curvature port, and export the identical mesh
for MATLAB so the two curvature implementations can be compared face by face."""
import sys
sys.path.insert(0, r"C:\Users\Upadhyaya_Lab\Code\barcode")
import numpy as np
import tifffile
from scipy.io import savemat

from analysis.volumetric.mesh import mesh_nucleus
from analysis.volumetric.curvature import analyze_curvature

B = (r"F:\FF\nucleus_live_cell\jurkat_nucleus_centrosome\GFP-Centrin_SiR-DNA"
     r"\Control\cells\all_cells_together\prog_live_cells")
OUT = (r"C:\Users\UPADHY~1\AppData\Local\Temp\claude"
       r"\C--Users-Upadhyaya-Lab-Code-barcode"
       r"\9f48b303-899e-4482-ab3b-afb87486e1b4\scratchpad")

cases = [(11, 2), (11, 8), (1, 1), (12, 1)]

for cell, frame in cases:
    mask = tifffile.imread(
        rf"{B}\Cell{cell}\frame{frame}\nucleus\3D_seg\Cell_{cell}_SegMask.tif") > 0
    m = mesh_nucleus(mask, (0.065, 0.065, 0.065))
    r = analyze_curvature(m.vertices_um, m.faces)

    print(f"PY cell{cell}_{frame} faces={m.geometry.n_faces} "
          f"mean={r.mean_curvature:.9f} min={r.min_curvature:.9f} "
          f"max={r.max_curvature:.9f} invag={r.invagination_ratio:.9f} "
          f"concave={r.concave_ratio:.9f} bt={r.fraction_faces_bottom_top:.9f}",
          flush=True)

    # MATLAB's code reads z from column 3; our vertices are (z, y, x). The reorder to
    # (y, x, z) is a cyclic permutation, so handedness -- and therefore every curvature
    # sign -- is preserved.
    savemat(
        rf"{OUT}\mesh_cell{cell}_{frame}.mat",
        {"V": m.vertices_um[:, [1, 2, 0]], "F": m.faces.astype(np.float64),
         "py_k_mean_F": r.k_mean_faces, "py_k_min_F": r.k_min_faces,
         "py_k_max_F": r.k_max_faces},
        do_compression=True,
    )
