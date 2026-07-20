#!/usr/bin/env python3
"""One barcode covering every timepoint of a volumetric time series.

The time-lapse pipeline reduces a series to a *single* row, which is the right answer
for a dataset-level summary but makes a degenerate barcode: the PNG normalises each
column across rows, so one row has no contrast to show.

This emits one row **per timepoint** instead, while still assembling the series first
so that every timepoint shares one crop box and therefore one denominator. Cropping
each timepoint to its own mask bounding box -- what you get analysing the files
independently -- would make the fraction-of-volume columns drift with the crop rather
than with the nucleus, and those are exactly the columns the barcode colours.

    python scripts/run_volumetric_timelapse_barcode.py <folder> [--seg-root ...]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from analysis.volumetric.binarization import analyze_binarization_3d
from analysis.volumetric.intensity import analyze_intensity_3d
from analysis.volumetric.run import _prepare_geometry
from analysis.volumetric.timelapse import group_timelapse, read_series
from core import BarcodeConfig, ChannelResults
from utils.writer import generate_combined_barcode, results_to_csv


def build_config(args) -> BarcodeConfig:
    config = BarcodeConfig()
    config.modules.image_binarization = True
    config.modules.intensity_distribution = True
    config.modules.optical_flow = False
    v = config.volumetric
    v.enabled = True
    v.timelapse_enabled = True
    v.threshold_offset = args.threshold_offset
    v.crop_padding_vox = args.crop_padding
    if args.seg_root or args.seg_template:
        v.segmentation_enabled = True
        v.segmentation_root = args.seg_root or ""
        if args.seg_regex:
            v.segmentation_regex = args.seg_regex
        if args.seg_template:
            v.segmentation_template = args.seg_template
    return config


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--pattern", default="*.tif")
    p.add_argument("--threshold-offset", type=float, default=0.1)
    p.add_argument("--crop-padding", type=int, default=2)
    p.add_argument("--seg-root", default=None)
    p.add_argument("--seg-regex", default=None)
    p.add_argument("--seg-template", default=None)
    p.add_argument("--timelapse-regex", default=None)
    p.add_argument("--out", default=None, help="output basename (default: <folder>/<series> Timepoints)")
    args = p.parse_args()

    files = glob.glob(os.path.join(args.folder, args.pattern))
    if not files:
        print(f"No files matching {args.pattern} in {args.folder}")
        return 1

    config = build_config(args)
    vcfg = config.volumetric
    regex = args.timelapse_regex or vcfg.timelapse_regex
    groups, unmatched = group_timelapse(files, regex)
    if unmatched:
        print(f"skipped {len(unmatched)} unmatched file(s)")
    if not groups:
        print("No series found.")
        return 1

    masked = vcfg.segmentation_enabled
    print(f"{'with' if masked else 'without'} masks")

    for group in groups:
        started = time.time()
        print(f"\n{group.describe()}")
        stack = read_series(group, channel=config.channels.selected_channel)
        volumes, masks, spacing, info, mask_paths = _prepare_geometry(stack, vcfg)
        print(f"  grid {volumes.shape[1:]} @ {tuple(round(s, 4) for s in spacing)} um"
              f"{'  (common crop box)' if info.get('common_crop') else ''}")

        results, island, void, kurt, med, mode = [], [], [], [], [], []
        for t in range(volumes.shape[0]):
            # Analyse timepoint t alone, but on the shared grid established above, so
            # the static metrics describe this timepoint and nothing else.
            binar, detail = analyze_binarization_3d(volumes, spacing, vcfg, [t], masks)
            inten, idet = analyze_intensity_3d(
                volumes, vcfg, [t], masks if vcfg.intensity_use_mask else None
            )
            row = ChannelResults(filepath=group.paths[t], channel=0)
            row.binarization = binar
            row.intensity = inten
            results.append(row)

            island.append(detail.island_voxels[0])
            void.append(detail.void_voxels[0])
            kurt.append(idet.kurtosis[0])
            med.append(idet.median_skew[0])
            mode.append(idet.mode_skew[0])

            print(f"    t={group.frames[t]:<3d} "
                  f"vol {detail.island_voxels[0] * detail.voxel_volume_um3:8.2f} um^3  "
                  f"{detail.island_voxels[0] / detail.voxel_count:6.2%}  "
                  f"aniso {binar.island_anisotropy:5.3f}  "
                  f"corr {binar.island_correlation_length:6.4f} um")

        # Each row above analysed a single timepoint, so its change metrics came back
        # NaN -- there was nothing to compare against *within* that call. The series
        # does have the comparison, so fill them in now, each timepoint relative to the
        # first. Without this, six of the twenty-five barcode columns are dead.
        for t, row in enumerate(results):
            row.binarization.max_island_percent_change = island[t] / island[0]
            row.binarization.max_void_percent_change = void[t] / void[0]
            row.intensity.kurtosis_diff = kurt[t] - kurt[0]
            row.intensity.median_skew_diff = med[t] - med[0]
            row.intensity.mode_skew_diff = mode[t] - mode[0]

        suffix = "with masks" if masked else "no masks"
        if args.out:
            base = args.out
        else:
            # Default beside the input folder, not inside it: mixing generated
            # output in with the images is how a data folder turns into a pile
            # of stale CSVs nobody can date.
            out_dir = os.path.join(
                os.path.dirname(os.path.normpath(args.folder)),
                "results", f"timepoints_{'with_masks' if masked else 'no_masks'}")
            os.makedirs(out_dir, exist_ok=True)
            base = os.path.join(out_dir, f"{group.series} Timepoints ({suffix})")
        results_to_csv(results, base + ".csv", just_metrics=False, physical_units=False)
        generate_combined_barcode(results, base)
        print(f"  wrote {base}.csv")
        print(f"  wrote {base}*.png   ({len(results)} rows, one per timepoint)")
        print(f"  {time.time() - started:.1f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
