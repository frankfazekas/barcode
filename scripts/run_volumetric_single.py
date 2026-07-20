#!/usr/bin/env python3
"""Run the volumetric pipeline on a single file and print every metric.

A GUI-free gate for checking the 3D path on one volume before batch processing or
before any GUI wiring exists. Example::

    python scripts/run_volumetric_single.py path/to/Cell1_1.tif

    python scripts/run_volumetric_single.py path/to/Cell1_1.tif \\
        --seg-root ".../prog_live_cells" \\
        --seg-regex "Cell(?P<cell>\\d+)_(?P<frame>\\d+)" \\
        --seg-template "Cell{cell}/frame{frame}/nucleus/3D_seg/Cell_{cell}_SegMask_origFOV.tif"
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from analysis.volumetric.mesh import write_obj
from analysis.volumetric.run import run_volumetric_analysis
from core import BarcodeConfig, Metrics


def build_config(args) -> BarcodeConfig:
    config = BarcodeConfig()
    config.modules.image_binarization = not args.no_binarization
    config.modules.intensity_distribution = not args.no_intensity
    config.modules.optical_flow = args.flow

    v = config.volumetric
    v.enabled = True
    v.flow_xyz_sigma = args.flow_xyz_sigma
    v.flow_t_sigma = args.flow_t_sigma
    v.flow_w_sigma = args.flow_w_sigma
    v.flow_reliability_percentile = args.flow_reliability
    v.flow_downsample = args.flow_downsample
    v.flow_use_mask = not args.flow_ignore_mask
    v.z_step_um = args.z_step or 0.0
    v.xy_step_um = args.xy_step or 0.0
    v.threshold_offset = args.threshold_offset
    v.crop_padding_vox = args.crop_padding
    v.make_isotropic = not args.no_isotropic
    v.intensity_use_mask = args.intensity_in_mask

    v.mesh_enabled = args.mesh
    v.mesh_maxrad = args.mesh_maxrad
    v.mesh_area_frac = args.mesh_area_frac
    v.mesh_smoothing_iterations = args.mesh_smooth_iters
    v.mesh_matlab_compat = args.mesh_matlab_compat
    v.mesh_curvature = not args.no_curvature
    v.mesh_iso2mesh_bin = args.mesh_iso2mesh_bin or ""

    # Either flag turns segmentation on: the default regex/template pair
    # ({stem} -> {stem}_SegMask.tif) already resolves a flat mask folder.
    if args.seg_template or args.seg_root:
        v.segmentation_enabled = True
        v.segmentation_root = args.seg_root or ""
        if args.seg_regex:
            v.segmentation_regex = args.seg_regex
        if args.seg_template:
            v.segmentation_template = args.seg_template
        v.mask_spacing_um = args.mask_spacing or 0.0
    return config


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", help="volumetric TIFF to analyse")
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--z-step", type=float, default=None, help="um per z slice (default: from file metadata)")
    p.add_argument("--xy-step", type=float, default=None, help="um per pixel (default: from file metadata)")
    p.add_argument("--threshold-offset", type=float, default=0.1)
    p.add_argument("--crop-padding", type=int, default=2)
    p.add_argument("--no-isotropic", action="store_true", help="downsample the mask to the image grid instead of resampling the image")
    p.add_argument("--seg-root", default=None)
    p.add_argument("--seg-regex", default=None, help="default: (?P<stem>.+)")
    p.add_argument("--seg-template", default=None, help="enables segmentation when given")
    p.add_argument("--mask-spacing", type=float, default=None, help="um per mask voxel (default: the image xy spacing)")
    p.add_argument("--no-binarization", action="store_true")
    p.add_argument("--no-intensity", action="store_true")
    p.add_argument("--intensity-in-mask", action="store_true",
                   help="build the intensity histogram from in-mask voxels only")
    p.add_argument("--flow", action="store_true",
                   help="run the 3D optical flow branch (needs 6*t_sigma+1 contiguous timepoints)")
    p.add_argument("--flow-xyz-sigma", type=float, default=3.0)
    p.add_argument("--flow-t-sigma", type=int, default=1)
    p.add_argument("--flow-w-sigma", type=float, default=4.0)
    p.add_argument("--flow-reliability", type=float, default=50.0, metavar="PERCENTILE",
                   help="drop voxels below this percentile of solver reliability; 0 keeps all")
    p.add_argument("--flow-downsample", type=int, default=1)
    p.add_argument("--flow-ignore-mask", action="store_true",
                   help="compute flow metrics over the whole volume, not just inside the mask")
    p.add_argument("--mesh", action="store_true",
                   help="surface-mesh the segmented nucleus (needs a segmentation)")
    p.add_argument("--mesh-maxrad", type=float, default=5.0,
                   help="cgalsurf radbound in isotropic voxels; smaller = finer mesh")
    p.add_argument("--mesh-area-frac", type=float, default=0.2)
    p.add_argument("--mesh-smooth-iters", type=int, default=10)
    p.add_argument("--mesh-matlab-compat", action="store_true",
                   help="reproduce the MATLAB pipeline's face-area filter")
    p.add_argument("--no-curvature", action="store_true",
                   help="skip the principal-curvature and invagination metrics")
    p.add_argument("--mesh-iso2mesh-bin", default=None,
                   help="an iso2mesh bin/ directory to stage the CGAL executables from")
    p.add_argument("--mesh-obj", default=None, metavar="DIR",
                   help="also write each mesh to DIR as an OBJ")
    args = p.parse_args()

    config = build_config(args)

    started = time.time()
    results, detail = run_volumetric_analysis(args.image, config, channel=args.channel)
    elapsed = time.time() - started

    stack = detail.stack
    print()
    print("=" * 78)
    print(f"INPUT   {stack.describe()}")
    if detail.mask_paths:
        print(f"MASK    {detail.mask_paths[0]}"
              + (f"  (+{len(detail.mask_paths) - 1} more)" if len(detail.mask_paths) > 1 else ""))
        print(f"        resample: {detail.resample_info}")
    else:
        print("MASK    none (intensity threshold used for binarization)")
    spacing = tuple(round(float(s), 5) for s in detail.spacing_zyx_um)
    print(f"ANALYSED shape(Z,Y,X)={detail.shape_zyx} spacing(z,y,x)={spacing} um")
    print(f"        timepoints analysed: {detail.frame_indices}")
    print("=" * 78)

    b = detail.binarization
    if b is not None:
        voxel_um3 = b.voxel_volume_um3
        total_vox = b.voxel_count
        print()
        print(f"-- structural (per timepoint), {total_vox:,} voxels @ {voxel_um3:.3e} um^3 --")
        print(f"   islands found      : {b.island_counts}")
        print(f"   largest island vox : {[f'{v:,.0f}' for v in b.island_voxels]}")
        print(f"   largest island um^3: {[f'{v * voxel_um3:.3f}' for v in b.island_voxels]}")
        print(f"   largest island %vol: {[f'{v / total_vox:.4%}' for v in b.island_voxels]}")
        print(f"   largest void   %vol: {[f'{v / total_vox:.4%}' for v in b.void_voxels]}")
        print(f"   spans field        : {b.connected}")
        print(f"   correlation length : {[f'{v:.4f}' for v in b.correlation_lengths]} um (r_max {b.r_max_um:.3f})")

    f = detail.flow
    if f is not None:
        print()
        print(f"-- flow (per window), {f.window_size}-frame windows (t_sigma={f.t_sigma}) --")
        print(f"   window centres     : {f.centres or 'none — series too short or all centres at an edge'}")
        if f.skipped_centres:
            print(f"   skipped (no window): {f.skipped_centres}")
        if f.centres:
            print(f"   solved on grid     : {tuple(round(s, 4) for s in f.spacing_zyx_um)} um"
                  f"{f'  (downsampled {f.downsample}x)' if f.downsample > 1 else ''}")
            print(f"   voxels used        : {[f'{v:.1%}' for v in f.valid_fractions]}"
                  f"{'  (reliability + mask)' if f.used_mask else '  (reliability only)'}")
            print(f"   mean speed         : {[f'{v:.4f}' for v in f.speeds]} um/s")
            print(f"   divergence         : {[f'{v:.4g}' for v in f.divergences]}")
            print(f"   curl ||rot v||     : {[f'{v:.4g}' for v in f.curls]}")
            print(f"   correlation length : {[f'{v:.4f}' for v in f.correlation_lengths]} um (r_max {f.r_max_um:.3f})")

    if detail.meshes:
        print()
        print(f"-- nucleus mesh ({len(detail.meshes)} timepoint(s)) --")
        for mesh in detail.meshes:
            print(f"   timepoint {mesh.frame_index}")
            for line in mesh.geometry.describe():
                print(f"     {line}")
            if mesh.curvature is not None:
                for line in mesh.curvature.describe():
                    print(f"     {line}")
            if args.mesh_obj:
                path = os.path.join(
                    args.mesh_obj,
                    f"{os.path.splitext(os.path.basename(args.image))[0]}"
                    f"_t{mesh.frame_index}.obj",
                )
                print(f"     wrote {write_obj(path, mesh.vertices_um, mesh.faces)}")

    print()
    print("-- BARCODE metrics --")
    for metric, value in results.get_dict_data(just_metrics=True).items():
        name = metric.value if isinstance(metric, Metrics) else str(metric)
        if isinstance(value, float) and np.isnan(value):
            print(f"   {name:38s} nan")
        else:
            print(f"   {name:38s} {value:.6g}")
    print(f"   {'Flags':38s} {results.convert_flags()}")
    print()
    print(f"elapsed {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
