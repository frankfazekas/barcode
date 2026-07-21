#!/usr/bin/env python3
"""Extract single timepoints from a large volumetric TIFF, calibrated, without loading it.

Some acquisitions ship as one very large hyperstack -- the Drosophila membrane movie is
11.4 GB -- and ``analysis.volumetric.reader.read_volume`` calls ``series.asarray()``,
which would pull the whole thing into memory. A single timepoint is a few tens of MB, so
the fix is to slice timepoints out through a memory map and write them as ordinary small
volumes that every downstream tool already understands.

Two things are written into each output that the source file lacks:

* **the true axis order.** ImageJ headers are often wrong. The Drosophila file declares
  ``channels=14, slices=150`` (i.e. 14 fluorescent markers) when it is really 150
  timepoints of a 14-slice stack -- measured, not assumed: along the 150-axis the mean
  decays monotonically (photobleaching) while along the 14-axis it peaks mid-stack and
  falls off a cliff at the last slice. ``--axes`` states the real order.
* **voxel size.** The source carries ``XResolution=(1,1)`` and no ImageJ ``spacing``, so
  every physical metric downstream would silently be in pixels. The outputs are written
  as ImageJ TIFFs with ``spacing``/``unit``/``XResolution`` set, so the reader picks the
  calibration up on its own and no override is needed again.

    python scripts/extract_timepoints.py <big.tif> --axes TZYX \\
        --xy-step 0.195 --z-step 0.235 --timepoints 40,41,70,100 --out <dir>
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tifffile


def parse_timepoints(spec: str, n_t: int) -> list:
    """Accept ``40,41`` or ``0-12`` or ``0:150:10``, always bounded by ``n_t``."""
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            bits = [int(b) if b else None for b in part.split(":")]
            start = bits[0] or 0
            stop = bits[1] if len(bits) > 1 and bits[1] is not None else n_t
            step = bits[2] if len(bits) > 2 and bits[2] is not None else 1
            out.extend(range(start, min(stop, n_t), step))
        elif "-" in part.lstrip("-"):
            lo, hi = part.split("-")
            out.extend(range(int(lo), min(int(hi) + 1, n_t)))
        else:
            out.append(int(part))
    bad = [t for t in out if not 0 <= t < n_t]
    if bad:
        raise SystemExit(f"timepoints out of range 0..{n_t - 1}: {sorted(set(bad))}")
    return sorted(set(out))


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("image", help="the large source TIFF")
    p.add_argument("--axes", required=True,
                   help="true axis order of the stored array, e.g. TZYX")
    p.add_argument("--timepoints", required=True,
                   help="which to extract: '40,41', '0-12' or 'start:stop:step'")
    p.add_argument("--out", required=True, help="output directory (use a data drive)")
    p.add_argument("--xy-step", type=float, required=True, help="um per pixel in xy")
    p.add_argument("--z-step", type=float, required=True, help="um per z slice")
    p.add_argument("--channel", type=int, default=None,
                   help="channel index, if the axes include C")
    p.add_argument("--prefix", default=None, help="output name stem (default: source stem)")
    args = p.parse_args()

    axes = args.axes.upper()
    memmap = tifffile.memmap(args.image, mode="r")
    if len(axes) != memmap.ndim:
        raise SystemExit(
            f"--axes {axes} has {len(axes)} letters but the array has {memmap.ndim} "
            f"dimensions {memmap.shape}"
        )
    for required in "ZYX":
        if required not in axes:
            raise SystemExit(f"--axes {axes} must contain {required}")
    if "T" not in axes:
        raise SystemExit("--axes must contain T to extract timepoints")

    order = {letter: i for i, letter in enumerate(axes)}
    n_t = memmap.shape[order["T"]]
    timepoints = parse_timepoints(args.timepoints, n_t)

    print(f"source {os.path.basename(args.image)}: {memmap.shape} {memmap.dtype} "
          f"as {axes}  ->  T={n_t}, Z={memmap.shape[order['Z']]}, "
          f"Y={memmap.shape[order['Y']]}, X={memmap.shape[order['X']]}")
    print(f"extracting {len(timepoints)} timepoint(s): {timepoints}")

    os.makedirs(args.out, exist_ok=True)
    stem = args.prefix or os.path.splitext(os.path.basename(args.image))[0]

    for t in timepoints:
        index = [slice(None)] * memmap.ndim
        index[order["T"]] = t
        if "C" in order:
            if args.channel is None:
                raise SystemExit("--channel is required when the axes include C")
            index[order["C"]] = args.channel
        volume = np.asarray(memmap[tuple(index)])

        # Reorder whatever is left to (Z, Y, X).
        remaining = [a for a in axes if a not in ("T", "C")]
        volume = np.transpose(volume, [remaining.index(a) for a in "ZYX"])

        path = os.path.join(args.out, f"{stem}_T{t}.tif")
        tifffile.imwrite(
            path,
            np.ascontiguousarray(volume),
            imagej=True,
            resolution=(1.0 / args.xy_step, 1.0 / args.xy_step),
            metadata={"axes": "ZYX", "spacing": args.z_step, "unit": "micron"},
        )
        print(f"  T{t:<4d} {volume.shape} {volume.dtype}  "
              f"{os.path.getsize(path) / 1e6:.1f} MB  -> {os.path.basename(path)}",
              flush=True)

    print(f"\nwrote {len(timepoints)} volume(s) to {args.out}")
    print(f"calibration baked in: xy {args.xy_step} um, z {args.z_step} um "
          f"-- downstream needs no --axes or step overrides")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
