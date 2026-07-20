#!/usr/bin/env python3
"""Mesh every timepoint of every series and write a side-car CSV.

The 3D counterpart of what ``TCell-3D-Morphodynamics`` does per (cell, frame): one
surface mesh per segmented nucleus per timepoint, plus the mesh- and curvature-derived
scalars. Output is a **side-car** -- its own CSV (and optionally one OBJ per timepoint)
-- and never touches the BARCODE Summary CSV or its 25 metrics.

Only the segmentations are read; the fluorescence volumes are not needed, since the
mesh describes the segmented object. That makes this far cheaper than the full
volumetric pipeline and is what the MATLAB code meshes too.

    python scripts/run_mesh_timeseries.py <folder-of-timepoint-files> \\
        --seg-root ".../prog_live_cells" \\
        --seg-regex "cell(?P<cell>\\d+)_(?P<frame>\\d+)" \\
        --seg-template "Cell{cell}/frame{frame}/nucleus/3D_seg/Cell_{cell}_SegMask.tif" \\
        --mask-spacing 0.065 --csv meshes.csv --obj-dir meshes/

``--seg-regex`` resolves the mask path and ``--timelapse-regex`` decides which files
belong to the same series and in what order; they are separate because the mask layout
and the file-naming convention need not agree.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tifffile

from analysis.volumetric.curvature import CONCAVE, CONVEX, HYPERBOLOID, analyze_curvature
from analysis.volumetric.mesh import (
    MeshingError,
    ensure_iso2mesh_binaries,
    mesh_nucleus,
    write_obj,
)
from analysis.volumetric.segmentation import coerce_to_zyx, resolve_segmentation_path
from analysis.volumetric.timelapse import DEFAULT_TIMELAPSE_REGEX, group_timelapse
from core import BarcodeConfig

COLUMNS = [
    "series", "frame", "mask_path",
    "n_vertices", "n_faces", "has_holes", "faces_flipped",
    "volume_um3", "voxel_volume_um3", "volume_ratio",
    "surface_area_um2", "sphericity", "equivalent_sphere_radius_um",
    "height_um", "extent_z_um", "extent_y_um", "extent_x_um",
    "mean_curvature_inv_um", "min_curvature_inv_um", "max_curvature_inv_um",
    "invagination_ratio", "concave_ratio", "fraction_faces_bottom_top",
    "n_concave_faces", "n_saddle_faces", "n_convex_faces",
    "seconds",
]


def build_config(args) -> BarcodeConfig:
    config = BarcodeConfig()
    v = config.volumetric
    v.enabled = True
    v.segmentation_enabled = True
    v.segmentation_root = args.seg_root or ""
    if args.seg_regex:
        v.segmentation_regex = args.seg_regex
    if args.seg_template:
        v.segmentation_template = args.seg_template

    v.mesh_enabled = True
    v.mesh_maxrad = args.mesh_maxrad
    v.mesh_area_frac = args.mesh_area_frac
    v.mesh_smoothing_iterations = args.mesh_smooth_iters
    v.mesh_matlab_compat = args.mesh_matlab_compat
    v.mesh_curvature = not args.no_curvature
    v.mesh_iso2mesh_bin = args.mesh_iso2mesh_bin or ""
    return config


def row_for(series, frame, mask_path, mesh, curvature, elapsed):
    g = mesh.geometry
    classes = curvature.concavity_classes if curvature is not None else None
    row = {
        "series": series,
        "frame": frame,
        "mask_path": mask_path,
        "n_vertices": g.n_vertices,
        "n_faces": g.n_faces,
        "has_holes": int(g.has_holes),
        "faces_flipped": int(curvature.faces_flipped) if curvature else "",
        "volume_um3": g.volume_um3,
        "voxel_volume_um3": g.voxel_count * g.voxel_volume_um3,
        "volume_ratio": g.volume_ratio,
        "surface_area_um2": g.surface_area_um2,
        "sphericity": g.sphericity,
        "equivalent_sphere_radius_um": g.equivalent_sphere_radius_um,
        "height_um": g.height_um,
        "extent_z_um": g.extent_zyx_um[0],
        "extent_y_um": g.extent_zyx_um[1],
        "extent_x_um": g.extent_zyx_um[2],
        "seconds": round(elapsed, 2),
    }
    if curvature is None:
        row.update({k: "" for k in COLUMNS if k not in row})
        return row

    row.update({
        "mean_curvature_inv_um": curvature.mean_curvature,
        "min_curvature_inv_um": curvature.min_curvature,
        "max_curvature_inv_um": curvature.max_curvature,
        "invagination_ratio": curvature.invagination_ratio,
        "concave_ratio": curvature.concave_ratio,
        "fraction_faces_bottom_top": curvature.fraction_faces_bottom_top,
        "n_concave_faces": int((classes == CONCAVE).sum()),
        "n_saddle_faces": int((classes == HYPERBOLOID).sum()),
        "n_convex_faces": int((classes == CONVEX).sum()),
    })
    return row


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("folder", help="folder of per-timepoint files naming the series")
    p.add_argument("--pattern", default="*.tif")
    p.add_argument("--timelapse-regex", default=DEFAULT_TIMELAPSE_REGEX,
                   help="groups files into series; needs <series> and <frame>")
    p.add_argument("--seg-root", default=None)
    p.add_argument("--seg-regex", default=None, help="default: (?P<stem>.+)")
    p.add_argument("--seg-template", default=None)
    p.add_argument("--mask-spacing", type=float, default=None,
                   help="um per isotropic mask voxel (required; masks carry no metadata)")
    p.add_argument("--mesh-maxrad", type=float, default=5.0)
    p.add_argument("--mesh-area-frac", type=float, default=0.2)
    p.add_argument("--mesh-smooth-iters", type=int, default=10)
    p.add_argument("--mesh-matlab-compat", action="store_true")
    p.add_argument("--no-curvature", action="store_true")
    p.add_argument("--mesh-iso2mesh-bin", default=None)
    p.add_argument("--csv", default=None, help="output CSV (default: <folder>/Mesh Series.csv)")
    p.add_argument("--obj-dir", default=None, help="also write one OBJ per timepoint")
    p.add_argument("--limit-series", type=int, default=None, help="only the first N series")
    args = p.parse_args()

    if not args.mask_spacing:
        p.error("--mask-spacing is required: segmentation masks carry no voxel size.")

    config = build_config(args)
    spacing = (args.mask_spacing,) * 3

    paths = sorted(glob.glob(os.path.join(args.folder, args.pattern)))
    groups, unmatched = group_timelapse(paths, args.timelapse_regex)
    if unmatched:
        print(f"{len(unmatched)} file(s) did not match {args.timelapse_regex!r}; skipped.")
    if not groups:
        print("No series found.")
        return 1
    if args.limit_series:
        groups = groups[: args.limit_series]

    ensure_iso2mesh_binaries(config.volumetric.mesh_iso2mesh_bin)
    csv_path = args.csv or os.path.join(args.folder, "Mesh Series.csv")
    if args.obj_dir:
        os.makedirs(args.obj_dir, exist_ok=True)

    rows, failures = [], []
    started = time.time()
    total = sum(len(g) for g in groups)
    done = 0

    for group in groups:
        print(f"\n{group.describe()}", flush=True)
        for path, frame in zip(group.paths, group.frames):
            done += 1
            label = f"{group.series}_{frame}"
            try:
                mask_path = resolve_segmentation_path(path, config.volumetric)
                mask = coerce_to_zyx(tifffile.imread(mask_path), label) > 0

                t0 = time.time()
                mesh = mesh_nucleus(mask, spacing,
                                    maxrad=args.mesh_maxrad,
                                    area_frac=args.mesh_area_frac,
                                    smoothing_iterations=args.mesh_smooth_iters,
                                    matlab_compat=args.mesh_matlab_compat,
                                    frame_index=frame)
                curvature = (
                    None if args.no_curvature
                    else analyze_curvature(mesh.vertices_um, mesh.faces)
                )
                elapsed = time.time() - t0
            except (MeshingError, OSError, ValueError, FileNotFoundError) as exc:
                # One unusable timepoint must not abandon the rest of the series.
                print(f"  [{done}/{total}] {label}: FAILED {type(exc).__name__}: {exc}",
                      flush=True)
                failures.append((label, f"{type(exc).__name__}: {exc}"))
                continue

            rows.append(row_for(group.series, frame, mask_path, mesh, curvature, elapsed))
            g = mesh.geometry
            extra = (
                f" invag {curvature.invagination_ratio:.3f}"
                f" curv {curvature.mean_curvature:+.4f}" if curvature else ""
            )
            print(f"  [{done}/{total}] {label}: {g.n_faces:5d} faces "
                  f"vol {g.volume_um3:8.3f} um^3  SA {g.surface_area_um2:8.3f}"
                  f"  sph {g.sphericity:.4f}{extra}  ({elapsed:.1f}s)", flush=True)

            if args.obj_dir:
                write_obj(os.path.join(args.obj_dir, f"{label}.obj"),
                          mesh.vertices_um, mesh.faces)

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} timepoint(s) meshed in {time.time() - started:.1f}s "
          f"-> {csv_path}")
    if failures:
        print(f"{len(failures)} failure(s):")
        for label, message in failures[:20]:
            print(f"  {label}: {message}")

    if rows:
        volumes = np.array([r["volume_um3"] for r in rows], dtype=float)
        ratios = np.array([r["volume_ratio"] for r in rows], dtype=float)
        holes = sum(r["has_holes"] for r in rows)
        print(f"volume  mean {volumes.mean():.3f} um^3  sd {volumes.std():.3f}  "
              f"range [{volumes.min():.3f}, {volumes.max():.3f}]")
        print(f"mesh/voxel volume ratio  mean {ratios.mean():.4f}  "
              f"min {ratios.min():.4f}   meshes with holes: {holes}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
