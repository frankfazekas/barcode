"""Mesh real Jurkat nuclear segmentations with the Python port, for comparison
against the MATLAB pipeline's stored results."""
import sys, time
sys.path.insert(0, r"C:\Users\Upadhyaya_Lab\Code\barcode")
import numpy as np
import tifffile

from analysis.volumetric.mesh import mesh_nucleus, write_obj

B = (r"F:\FF\nucleus_live_cell\jurkat_nucleus_centrosome\GFP-Centrin_SiR-DNA"
     r"\Control\cells\all_cells_together\prog_live_cells")
PSIZE = 0.065

cases = [(int(c), int(f)) for c, f in (a.split(",") for a in sys.argv[1:])] or [(11, 1)]

for cell, frame in cases:
    path = rf"{B}\Cell{cell}\frame{frame}\nucleus\3D_seg\Cell_{cell}_SegMask.tif"
    mask = tifffile.imread(path) > 0
    for compat in (False, True):
        t0 = time.time()
        m = mesh_nucleus(mask, (PSIZE, PSIZE, PSIZE), matlab_compat=compat)
        g = m.geometry
        print(
            f"cell{cell}_{frame} compat={int(compat)} "
            f"faces={g.n_faces:6d} vol={g.volume_um3:9.4f} "
            f"voxvol={g.voxel_count * g.voxel_volume_um3:9.4f} "
            f"ratio={g.volume_ratio:.4f} SA={g.surface_area_um2:9.4f} "
            f"sph={g.sphericity:.4f} h={g.height_um:7.4f} "
            f"holes={int(g.has_holes)} {time.time()-t0:5.1f}s",
            flush=True,
        )
    if len(cases) == 1:
        out = write_obj(
            r"C:\Users\UPADHY~1\AppData\Local\Temp\claude"
            r"\C--Users-Upadhyaya-Lab-Code-barcode"
            r"\9f48b303-899e-4482-ab3b-afb87486e1b4\scratchpad"
            rf"\cell{cell}_{frame}_nucleus.obj",
            m.vertices_um, m.faces,
        )
        print("wrote", out)
