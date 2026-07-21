"""How much of a shape metric survives the imaging grid? Ask a 4 nm nucleus.

Our live Jurkat volumes sit on a coarse, strongly anisotropic grid (0.065 um in xy,
0.3 um in z -- 4.6x). Every geometry metric BARCODE reports from them -- sphericity,
mean curvature, invagination ratio, surface area -- is therefore measured on a nucleus
that the grid itself has already smoothed and stair-stepped. Synthetic phantoms
(``scripts/validate_phantoms.py``) bound that error for spheres, ellipsoids and tori,
but a real nucleus is none of those, and the error depends on the shape.

Janelia's OpenOrganelle ``jrc_jurkat-1`` closes that gap: real Jurkat nuclei, segmented
at 4 nm on a near-isotropic grid. Treating the finest rung as truth, this script
resamples the SAME nucleus down a ladder of progressively coarser and more anisotropic
grids -- ending at our live acquisition geometry -- and re-runs the pipeline at each
rung. The output says, per metric, how much of the number is the nucleus and how much
is the microscope.

Run it on the output of ``scripts/stage_openorganelle.py``, with the BARCODE interpreter
(this one needs ``core``, unlike the stager which needs zarr)::

    ~/miniforge3/envs/barcode/python.exe scripts/validate_openorganelle.py \
        --staged L:/FF/Hackathon/full_datasets/jrc_jurkat-1/BARCODE

Reading the result: a metric whose drift stays within a few percent across the ladder is
being measured, not manufactured. One that moves monotonically and lands far from the
truth rung at the live geometry is telling you the live-cell number is dominated by the
grid, and that only *relative* comparisons between equally-sampled datasets are safe.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts._staging import write_volume

# The rungs. The last one is this lab's live acquisition geometry (CLAUDE.md: z step
# 0.3 um from the ImageJ 'spacing' tag, xy 0.065 um from XResolution), which is the whole
# reason for the exercise. The ones before it separate the two effects that arrive
# together there: coarsening alone, then coarsening plus anisotropy.
DEFAULT_LADDER: List[Tuple[str, float, float]] = [
    # label,           xy_um,  z_um
    ("native",          0.000,  0.000),   # 0,0 = leave the staged grid untouched (truth)
    ("iso_0p128",       0.128,  0.128),
    ("iso_0p256",       0.256,  0.256),
    ("xy0p065_z0p065",  0.065,  0.065),   # live xy, but isotropic: coarsening only
    ("xy0p065_z0p150",  0.065,  0.150),   # 2.3x anisotropic
    ("live_z0p300",     0.065,  0.300),   # the real acquisition geometry, 4.6x
]

# Reported in physical units, so they are comparable across rungs. Fractional metrics are
# deliberately left out: they are normalised by the analysed volume, which changes with the
# grid, so a drift in them would not mean what it appears to mean.
METRICS_OF_INTEREST = [
    "Mesh Volume", "Mesh Surface Area", "Sphericity", "Equivalent Sphere Radius",
    "Lateral/Axial Ratio", "Solidity", "Mean Curvature <H>",
    "Invagination Ratio", "Concave Area Fraction", "Mesh Height",
    "Maximum Island Volume Quantity", "Mean Island Separation",
    "Structural Correlation Length",
]


def _read_staged(staged: str) -> Tuple[str, str, str]:
    """Locate the one image/mask pair written by stage_openorganelle.py."""
    images = sorted(glob.glob(os.path.join(staged, "data", "*.tif")))
    if not images:
        raise SystemExit(f"no staged volumes in {os.path.join(staged, 'data')}")
    if len(images) > 1:
        raise SystemExit(
            f"{len(images)} volumes in {staged}/data; this compares ONE nucleus field "
            f"down the ladder, so stage them into separate folders")
    image_path = images[0]
    stem = os.path.splitext(os.path.basename(image_path))[0]
    mask_path = os.path.join(staged, "masks", f"{stem}_SegMask.tif")
    if not os.path.isfile(mask_path):
        raise SystemExit(f"no mask beside the image: {mask_path}")
    return image_path, mask_path, stem


def _spacing_of(path: str) -> Tuple[float, float]:
    """(xy_um, z_um) as the file declares them -- the same tags the reader trusts."""
    with tifffile.TiffFile(path) as handle:
        meta = handle.imagej_metadata or {}
        z_um = float(meta.get("spacing", 1.0))
        tags = handle.pages[0].tags
        x_res = tags["XResolution"].value if "XResolution" in tags else (1, 1)
        xy_um = float(x_res[1]) / float(x_res[0])
    return xy_um, z_um


def _resample(volume: np.ndarray, src: Tuple[float, float], dst: Tuple[float, float],
              *, is_mask: bool) -> np.ndarray:
    """Put ``volume`` on the (xy, z) grid ``dst``, by physical size.

    Nearest neighbour for the mask, always: these carry integer instance labels and
    averaging label 7 with label 8 invents an object that was never segmented -- the same
    reason ``_staging.mask_z_to_isotropic`` refuses to interpolate. Linear for the EM,
    which is a real intensity field.

    Node-aligned like ``resample._reference_shape_for_spacing``: n planes span (n-1)
    steps, not n. Using n would stretch the volume by about one step per axis, which lands
    straight in Mesh Volume and Height.
    """
    import SimpleITK as sitk

    (src_xy, src_z), (dst_xy, dst_z) = src, dst
    image = sitk.GetImageFromArray(volume)          # sitk is x,y,z; numpy is z,y,x
    image.SetSpacing((src_xy, src_xy, src_z))

    shape = []
    for extent, step in ((volume.shape[0], (src_z, dst_z)),
                         (volume.shape[1], (src_xy, dst_xy)),
                         (volume.shape[2], (src_xy, dst_xy))):
        span = (extent - 1) * step[0]
        shape.append(max(1, int(np.floor(span / step[1])) + 1))

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing((dst_xy, dst_xy, dst_z))
    resampler.SetSize([int(shape[2]), int(shape[1]), int(shape[0])])
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    out = sitk.GetArrayFromImage(resampler.Execute(image))
    return out.astype(volume.dtype)


def build_ladder(staged: str, out_root: str,
                 ladder: List[Tuple[str, float, float]]) -> Dict[str, str]:
    """Write one image+mask pair per rung, each declaring its own geometry."""
    image_path, mask_path, stem = _read_staged(staged)
    src = _spacing_of(image_path)
    print(f"truth grid: xy {src[0]} um, z {src[1]} um   ({stem})")

    image = np.asarray(tifffile.imread(image_path))
    mask = np.asarray(tifffile.imread(mask_path))
    mask_src = _spacing_of(mask_path)

    rungs: Dict[str, str] = {}
    for label, xy_um, z_um in ladder:
        target = src if (xy_um == 0 and z_um == 0) else (xy_um, z_um)
        rung_dir = os.path.join(out_root, label)
        os.makedirs(os.path.join(rung_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(rung_dir, "masks"), exist_ok=True)

        rung_image = image if target == src else _resample(image, src, target, is_mask=False)

        # Simulating the acquisition, in two steps, because one step tests nothing.
        #
        # A segmentation of live data is drawn on the ACQUIRED anisotropic grid -- it can
        # only ever know the nucleus at 0.3 um in z -- and BARCODE then resamples it up to
        # isotropic. The information lost in that round trip is precisely what anisotropy
        # costs, and it is not recoverable by the upsampling.
        #
        # Going straight to isotropic-at-xy instead (the obvious one-liner) makes every
        # rung with the same xy step produce a BYTE-IDENTICAL mask, so the mesh metrics
        # come out equal at 0.065/0.065 and 0.065/0.300 and the ladder silently reports
        # that anisotropy is free. It is not.
        #
        # The mask's own source spacing is (xy, xy), not the image's -- it was staged onto
        # an isotropic grid -- so resampling it from the image's z step would move it
        # bodily in depth.
        acquired = _resample(mask, mask_src, target, is_mask=True)
        rung_mask = _resample(acquired, target, (target[0], target[0]), is_mask=True)

        if not rung_mask.any():
            print(f"  {label:16s} SKIPPED -- mask vanished at this grid")
            continue

        image_out = os.path.join(rung_dir, "data", f"{stem}.tif")
        mask_out = os.path.join(rung_dir, "masks", f"{stem}_SegMask.tif")
        # The mask stays isotropic at the rung's xy step, which is what mask_spacing_um: 0
        # means, so no override is needed at any rung.
        write_volume(image_out, rung_image, xy_um=target[0], z_um=target[1])
        write_volume(mask_out, rung_mask, xy_um=target[0], z_um=target[0])
        print(f"  {label:16s} xy {target[0]:.3f} z {target[1]:.3f}  "
              f"image {rung_image.shape}  mask {rung_mask.shape}")
        rungs[label] = rung_dir
    return rungs


def run_rung(rung_dir: str, label: str) -> Optional[Dict[str, float]]:
    """Run the volumetric pipeline on one rung and return its physical metrics."""
    csv_path = os.path.join(rung_dir, f"{label}.csv")
    command = [
        sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "run_volumetric_batch.py"),
        os.path.join(rung_dir, "data"), "--mode", "xyzt", "--mesh",
        "--seg-root", os.path.join(rung_dir, "masks"), "--csv", csv_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  {label}: pipeline failed\n{result.stdout[-1500:]}\n{result.stderr[-1500:]}")
        return None

    physical = os.path.join(rung_dir, f"{label} (physical).csv")
    if not os.path.isfile(physical):
        print(f"  {label}: no physical CSV at {physical}")
        return None
    with open(physical, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None

    out = {}
    for key, value in rows[0].items():
        if key in METRICS_OF_INTEREST:
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                out[key] = float("nan")
    return out


def report(results: Dict[str, Dict[str, float]], truth_label: str, out_path: str) -> None:
    truth = results.get(truth_label)
    if truth is None:
        raise SystemExit(f"the truth rung {truth_label!r} produced no metrics")

    labels = [k for k in results if k != truth_label]
    metrics = [m for m in METRICS_OF_INTEREST if m in truth]

    width = max(len(m) for m in metrics) + 2
    print(f"\n{'metric':{width}}{truth_label:>14}" +
          "".join(f"{l:>18}" for l in labels))
    print("-" * (width + 14 + 18 * len(labels)))
    for metric in metrics:
        base = truth[metric]
        line = f"{metric:{width}}{base:14.4f}"
        for label in labels:
            value = results[label].get(metric, float("nan"))
            if base and np.isfinite(base) and np.isfinite(value):
                line += f"{value:11.4f}{100 * (value - base) / abs(base):+6.1f}%"
            else:
                line += f"{value:>18.4f}"
        print(line)

    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Metric", truth_label] +
                        [c for label in labels for c in (label, f"{label} drift %")])
        for metric in metrics:
            base = truth[metric]
            row = [metric, base]
            for label in labels:
                value = results[label].get(metric, float("nan"))
                drift = (100 * (value - base) / abs(base)
                         if base and np.isfinite(base) and np.isfinite(value) else "")
                row += [value, drift]
            writer.writerow(row)
    print(f"\nwrote {out_path}")
    print("\nDrift is against the finest rung, which is treated as truth. Metrics that stay "
          "\nwithin a few percent are measured; ones that move monotonically toward the live "
          "\ngeometry are grid artefacts and must only be compared like-for-like.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--staged", required=True,
        help="a stage_openorganelle.py output root (holding data/ and masks/)")
    parser.add_argument("--out", default="", help="ladder root (default: <staged>/ladder)")
    parser.add_argument(
        "--build-only", action="store_true",
        help="write the ladder but do not run the pipeline")
    args = parser.parse_args()

    out_root = args.out or os.path.join(args.staged, "ladder")
    if os.path.splitdrive(os.path.abspath(out_root))[0].upper() == "C:":
        raise SystemExit(f"refusing to write {out_root}: outputs belong on a data drive.")
    os.makedirs(out_root, exist_ok=True)

    rungs = build_ladder(args.staged, out_root, DEFAULT_LADDER)
    if args.build_only:
        return

    print("\nrunning the pipeline at each rung ...")
    results: Dict[str, Dict[str, float]] = {}
    for label, rung_dir in rungs.items():
        print(f"  {label} ...", flush=True)
        metrics = run_rung(rung_dir, label)
        if metrics:
            results[label] = metrics

    truth = next(iter(rungs))
    report(results, truth, os.path.join(out_root, "resolution_ladder.csv"))


if __name__ == "__main__":
    main()
