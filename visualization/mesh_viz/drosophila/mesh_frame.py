"""Mesh every cell of one Drosophila frame and save the meshes + per-cell metrics.

The mask is on the acquired grid (z 0.235, xy 0.195 um) and meshing needs an isotropic
one, so z is resampled by nearest-neighbour index mapping first -- nearest, not linear,
because the volume carries integer instance labels that must not be blended.
"""
import os
import pickle
import sys
import time

sys.path.insert(0, r"C:\Users\Upadhyaya_Lab\Code\barcode")
import numpy as np
import tifffile

from analysis.volumetric.mesh_field import mesh_field

STAGED = r"L:\FF\Hackathon\full_datasets\drosophila_Erika\BARCODE\staged"
OUT = r"L:\FF\Hackathon\full_datasets\drosophila_Erika\BARCODE\results\meshes"
XY, Z = 0.195, 0.235

frame = sys.argv[1] if len(sys.argv) > 1 else "1"
maxrad = float(sys.argv[2]) if len(sys.argv) > 2 else 2.5

os.makedirs(OUT, exist_ok=True)
mask = tifffile.imread(os.path.join(STAGED, "masks", f"emb_{frame}_SegMask.tif"))
n_z = int(round(mask.shape[0] * Z / XY))
index = np.clip(np.round(np.linspace(0, mask.shape[0] - 1, n_z)).astype(int),
                0, mask.shape[0] - 1)
iso = mask[index]
print(f"emb_{frame}: {mask.shape} -> {iso.shape} isotropic at {XY} um, "
      f"{mask.max()} labels, maxrad {maxrad}", flush=True)

started = time.time()
field = mesh_field(iso, (XY,) * 3, maxrad=maxrad, min_voxels=200,
                   curvature=True, exclude_border="xy")
print(f"meshed {len(field)} cells in {time.time() - started:.0f}s", flush=True)
for line in field.describe():
    print("   ", line, flush=True)

path = os.path.join(OUT, f"emb_{frame}_field.pkl")
with open(path, "wb") as fh:
    pickle.dump(
        {
            "frame": frame,
            "spacing": (XY, XY, XY),
            "shape": iso.shape,
            "maxrad": maxrad,
            "cells": [
                {
                    "label": m.label,
                    "vertices": m.vertices_um.astype(np.float32),
                    "faces": m.faces.astype(np.int32),
                    "k_mean_faces": m.curvature.k_mean_faces.astype(np.float32),
                    "geometry": m.geometry,
                    "curvature_scalars": {
                        "mean": m.curvature.mean_curvature,
                        "invagination": m.curvature.invagination_ratio,
                        "concave": m.curvature.concave_ratio,
                    },
                }
                for m in field.meshes
            ],
            "skipped_border": field.skipped_border,
            "skipped_small": field.skipped_small,
            "failed": field.failed,
        },
        fh,
    )
print("wrote", path, f"({os.path.getsize(path) / 1e6:.1f} MB)", flush=True)
