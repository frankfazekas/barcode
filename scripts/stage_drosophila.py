"""Stage matched image+mask timepoints from the Drosophila 4D pair.

The image is one 10.5 GB TIFF and the segmentation is one 21 GB ``.npy``, both
``(T=149, Z=13, 1500, 1808)``. Neither fits the per-file mask pairing BARCODE uses, and
loading both whole is 31 GB before any resampling. So write a chosen timepoint range out
as ordinary per-timepoint files that the existing time-lapse path already handles:

    <out>/emb_1.tif ... emb_N.tif          image volumes  (Z, Y, X) uint16
    <out>/masks/emb_1_SegMask.tif ...      label volumes  (Z, Y, X) int32

Both are read lazily -- ``tifffile.memmap`` for the TIFF (one contiguous ImageJ page)
and ``np.load(mmap_mode='r')`` for the ``.npy`` -- so peak memory is one timepoint.

The image's header declares ``ZCYX`` but the data is really ``TZYX``; that is a property
of the file, not a guess (see the dataset README). Slicing ``[t]`` is correct either way
here because T is the first axis under both readings -- what the header gets wrong is the
*meaning* of axes 0 and 1, and this script only ever indexes axis 0.

Outputs go next to the data on L:, never to C:.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import tifffile

IMAGE = (r"L:\FF\Hackathon\full_datasets\_open_data\drosophila_Erika"
         r"\20241021_gap43mCh_CyO_ZipWT_mem_barcode.tif")
MASK = (r"L:\FF\Hackathon\full_datasets\_open_data\drosophila_Erika"
        r"\20241021_gap43mCh_CyO_ZipWT_4Dseg_zstitch.npy")
OUT = (r"L:\FF\Hackathon\full_datasets\_open_data\drosophila_Erika\BARCODE\staged")

XY_STEP, Z_STEP = 0.195, 0.235


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", default=IMAGE)
    p.add_argument("--mask", default=MASK)
    p.add_argument("--out", default=OUT)
    p.add_argument("--t-start", type=int, default=40)
    p.add_argument("--t-end", type=int, default=45, help="exclusive")
    p.add_argument("--min-object-um3", type=float, default=0.0,
                   help="drop labels smaller than this; sub-cell fragments otherwise "
                        "dominate the object counts and drag the contact statistics")
    args = p.parse_args()

    images = tifffile.memmap(args.image)
    masks = np.load(args.mask, mmap_mode="r")
    print(f"image {images.shape} {images.dtype}")
    print(f"mask  {masks.shape} {masks.dtype}")

    if images.shape[0] != masks.shape[0] or images.shape[1] != masks.shape[1]:
        raise SystemExit(
            f"image {images.shape[:2]} and mask {masks.shape[:2]} disagree on (T, Z); "
            f"these are not the matched pair."
        )
    if images.shape[2:] != masks.shape[2:]:
        raise SystemExit(f"XY differs: {images.shape[2:]} vs {masks.shape[2:]}")

    mask_dir = os.path.join(args.out, "masks")
    os.makedirs(mask_dir, exist_ok=True)
    voxel_um3 = XY_STEP * XY_STEP * Z_STEP

    for n, t in enumerate(range(args.t_start, args.t_end), start=1):
        volume = np.asarray(images[t])
        labels = np.asarray(masks[t]).astype(np.int32)

        if args.min_object_um3 > 0:
            counts = np.bincount(labels.ravel())
            counts[0] = 0
            keep = counts * voxel_um3 >= args.min_object_um3
            lut = np.where(keep, np.arange(counts.size), 0).astype(np.int32)
            dropped = int(np.count_nonzero(counts) - keep.sum())
            labels = lut[labels]
        else:
            dropped = 0

        n_objects = int(np.count_nonzero(np.unique(labels)))
        tifffile.imwrite(
            os.path.join(args.out, f"emb_{n}.tif"), volume, imagej=True,
            resolution=(1 / XY_STEP, 1 / XY_STEP),
            metadata={"axes": "ZYX", "spacing": Z_STEP, "unit": "micron"},
        )
        tifffile.imwrite(os.path.join(mask_dir, f"emb_{n}_SegMask.tif"), labels)
        print(f"  t={t:3d} -> emb_{n}: {n_objects} objects"
              + (f" ({dropped} dropped below {args.min_object_um3:g} um^3)" if dropped else ""))

    print(f"\nstaged {args.t_end - args.t_start} timepoint(s) to {args.out}")
    print("mask_spacing_um must be the IMAGE z step (0.235), not the xy step: these masks"
          "\nare on the acquired grid, not a finer isotropic one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
