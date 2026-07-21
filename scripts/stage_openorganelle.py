"""Stage Janelia OpenOrganelle FIB-SEM volumes into BARCODE's conventions.

OpenOrganelle (https://openorganelle.janelia.org/) publishes near-isotropic FIB-SEM
volumes with dense organelle segmentations. ``jrc_jurkat-1`` is a volume of *Jurkat
T cells* -- the same cell type as this lab's live nucleus/centrosome imaging, but fixed,
at 4 nm, and near-isotropic. That makes it the closest thing available to ground truth
for the geometry metrics (mesh volume, surface area, sphericity, mean curvature,
invagination) and the only second source, after the Allen FOVs, of a genuine INTEGER
INSTANCE mask for the packing metrics: ``nucleus_seg`` holds 11 separate nuclei.

Two things make staging necessary rather than optional.

*Format.* The data ships as chunked ``.zarr`` (EM, zstd/blosc) and ``.n5`` (segmentations,
gzip). The pinned ``barcode`` environment has no zarr, numcodecs or s3fs, and must not be
pip-installed into. So this script runs under a DIFFERENT interpreter that already has
them -- see "Running it" below -- and hands BARCODE nothing but plain ImageJ TIFF.

*Metadata.* ``analysis/volumetric/reader.py`` deliberately refuses to guess axis order or
voxel size. Rather than pushing ``--axes``/``--xy-step``/``--z-step`` overrides through
every downstream call, staging writes files that state their own geometry, via the same
``write_volume`` helper the Cell Tracking Challenge and Allen stagers use.

Running it -- NOT the barcode interpreter::

    ~/miniforge3/envs/napari_test/python.exe scripts/stage_openorganelle.py \
        --source L:/FF/Hackathon/full_datasets/jrc_jurkat-1 --scale s3 --label nucleus_seg

``--source s3`` reads straight from the public bucket instead of a local copy, which is
useful for a quick look but re-downloads on every run.

Note on the EM channel: FIB-SEM contrast is INVERTED relative to fluorescence -- membranes
are dark on a bright background. The intensity-distribution metrics are therefore not
comparable with fluorescence runs, and BARCODE's own mean-relative thresholding will select
the background, not the structure. This staging path is for MASK-driven analysis; pass the
segmentation and leave the EM as the intensity channel only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts._staging import mask_z_to_isotropic, write_volume

BUCKET = "janelia-cosem-datasets"
NM_PER_UM = 1000.0


def _require_zarr():
    """Fail with the fix, not with a bare ImportError.

    The single most likely way to run this script is with the barcode interpreter out of
    habit, which cannot work and must not be made to work by installing into that env.
    """
    try:
        import zarr  # noqa: F401
    except ImportError:
        raise SystemExit(
            "zarr is not available in this interpreter.\n"
            "Do NOT pip install it into the pinned 'barcode' env. Run this script with an\n"
            "interpreter that already has zarr + numcodecs (+ s3fs for --source s3), e.g.\n"
            "  ~/miniforge3/envs/napari_test/python.exe scripts/stage_openorganelle.py ..."
        )
    return __import__("zarr")


# --- opening the two stores -------------------------------------------------------------
#
# The EM is OME-NGFF zarr and the segmentations are COSEM-flavoured N5. They spell their
# multiscale metadata differently, so each gets its own reader and both are normalised to
# the same {level: (z, y, x) nanometres} mapping. Everything downstream matches levels by
# PHYSICAL SCALE rather than by name -- the EM has s0..s6 and the labels only s0..s4, and
# assuming "s3 means s3 in both" is the kind of thing that silently misregisters a mask.


def _open_group(source: str, dataset: str, kind: str):
    zarr = _require_zarr()
    if source == "s3":
        if kind == "zarr":
            import s3fs

            fs = s3fs.S3FileSystem(anon=True)
            path = f"{BUCKET}/{dataset}/{dataset}.zarr"
            return zarr.open(s3fs.S3Map(path, s3=fs), mode="r")
        return zarr.open(
            zarr.N5FSStore(f"s3://{BUCKET}/{dataset}/{dataset}.n5", anon=True), mode="r")

    path = os.path.join(source, f"{dataset}.{kind}")
    if not os.path.isdir(path):
        raise SystemExit(f"not found: {path}\n(is the rclone copy complete?)")
    store = zarr.N5Store(path) if kind == "n5" else zarr.DirectoryStore(path)
    return zarr.open(store, mode="r")


def _scales_ngff(group) -> Dict[str, Tuple[float, float, float]]:
    """Level -> (z, y, x) nanometres, from OME-NGFF ``multiscales``."""
    multiscales = group.attrs["multiscales"][0]
    units = {a["name"]: a.get("unit", "nanometer") for a in multiscales.get("axes", [])}
    if any(u != "nanometer" for u in units.values()):
        raise SystemExit(f"expected nanometre axes, got {units}")
    out = {}
    for entry in multiscales["datasets"]:
        scale = next(t["scale"] for t in entry["coordinateTransformations"]
                     if t["type"] == "scale")
        out[entry["path"]] = tuple(float(v) for v in scale)
    return out


def _scales_n5(group) -> Dict[str, Tuple[float, float, float]]:
    """Level -> (z, y, x) nanometres, from COSEM N5 ``multiscales``."""
    out = {}
    for entry in group.attrs["multiscales"][0]["datasets"]:
        transform = entry["transform"]
        if list(transform["axes"]) != ["z", "y", "x"]:
            raise SystemExit(f"unexpected N5 axis order {transform['axes']}")
        if any(u != "nm" for u in transform["units"]):
            raise SystemExit(f"expected nm units, got {transform['units']}")
        out[entry["path"]] = tuple(float(v) for v in transform["scale"])
    return out


def _match_level(scales: Dict[str, Tuple[float, float, float]],
                 target: Tuple[float, float, float]) -> str:
    """The level whose voxel scale matches ``target``, within a hair.

    Exact equality is not safe: the published numbers carry float noise (16 nm appears as
    13.760000000000002 at one level). A 1% tolerance is far tighter than the 2x gap
    between neighbouring levels, so a match is unambiguous when it exists.
    """
    for name, scale in scales.items():
        if all(abs(a - b) <= 0.01 * b for a, b in zip(scale, target)):
            return name
    raise SystemExit(
        f"no level at scale {target} nm; available: "
        + ", ".join(f"{k}={v}" for k, v in sorted(scales.items()))
    )


# --- cropping ---------------------------------------------------------------------------


def _bbox_of_labels(volume: np.ndarray, pad: int) -> Tuple[slice, slice, slice]:
    """Tight box around every non-zero voxel, padded and clipped."""
    nonzero = np.nonzero(volume)
    if not len(nonzero[0]):
        raise SystemExit("segmentation is empty at this level -- nothing to crop to")
    box = []
    for axis, idx in enumerate(nonzero):
        lo = max(0, int(idx.min()) - pad)
        hi = min(volume.shape[axis], int(idx.max()) + 1 + pad)
        box.append(slice(lo, hi))
    return tuple(box)


def _parse_crop(text: str) -> Tuple[slice, slice, slice]:
    parts = [int(v) for v in text.replace(" ", "").split(",")]
    if len(parts) != 6:
        raise SystemExit("--crop wants six integers: z0,z1,y0,y1,x0,x1")
    return tuple(slice(parts[i], parts[i + 1]) for i in (0, 2, 4))


# --- main -------------------------------------------------------------------------------


def _select_object(seg: np.ndarray, object_id: int) -> np.ndarray:
    """Keep one instance, drop the rest.

    Needed for the resolution ladder and for any per-nucleus shape reference: the field
    holds 11 nuclei and several are cut by its edge, so a clean measurement means choosing
    one interior object rather than meshing whatever happens to be largest.
    """
    present = np.unique(seg)
    present = present[present != 0]
    if object_id not in present:
        raise SystemExit(
            f"object {object_id} is not in this segmentation; present: "
            f"{', '.join(str(int(v)) for v in present)}")
    return np.where(seg == object_id, seg, 0)


def _touches_edge(seg: np.ndarray, object_id: int) -> bool:
    """Is this object cut by the edge of the field?

    A clipped nucleus has a flat face the biology never gave it, which lands directly in
    surface area, sphericity and solidity. Worth saying out loud rather than silently
    averaging it in.
    """
    mask = seg == object_id
    faces = [mask[0], mask[-1], mask[:, 0], mask[:, -1], mask[:, :, 0], mask[:, :, -1]]
    return any(face.any() for face in faces)


def stage(source: str, dataset: str, scale: str, label: str, out_dir: str,
          crop: Optional[str], crop_to_labels: bool, pad: int,
          keep_instances: bool, em_group: str, object_id: Optional[int] = None) -> None:
    em_root = _open_group(source, dataset, "zarr")
    em_multiscale = em_root[f"recon-1/em/{em_group}"]
    em_scales = _scales_ngff(em_multiscale)
    if scale not in em_scales:
        raise SystemExit(
            f"unknown EM level {scale!r}; available: {', '.join(sorted(em_scales))}")
    target = em_scales[scale]

    seg_root = _open_group(source, dataset, "n5")
    seg_multiscale = seg_root[f"labels/{label}"]
    seg_scales = _scales_n5(seg_multiscale)
    seg_level = _match_level(seg_scales, target)

    z_nm, y_nm, x_nm = target
    if abs(y_nm - x_nm) > 1e-9:
        raise SystemExit(f"anisotropic xy ({y_nm} x {x_nm} nm) is not supported")
    xy_um, z_um = x_nm / NM_PER_UM, z_nm / NM_PER_UM

    em_array = em_multiscale[scale]
    seg_array = seg_multiscale[seg_level]
    if em_array.shape != seg_array.shape:
        raise SystemExit(
            f"shape mismatch at matched scale: EM {scale} {em_array.shape} vs "
            f"{label} {seg_level} {seg_array.shape}"
        )

    print(f"{dataset}: EM {scale} / {label} {seg_level}  shape {em_array.shape}  "
          f"voxel {z_nm} x {y_nm} x {x_nm} nm")

    # The segmentation is read first and in full at this level, because --crop-to-labels
    # needs to see all of it to find the box. Levels are chosen so this fits in RAM; s0 is
    # 257 Gvoxel and is not a sane choice here.
    print("  reading segmentation ...", flush=True)
    seg = np.asarray(seg_array[:])

    if object_id is not None:
        clipped = _touches_edge(seg, object_id)
        seg = _select_object(seg, object_id)
        print(f"  isolated object {object_id}"
              + ("  WARNING: it touches the field edge, so its surface area, sphericity "
                 "and solidity are not trustworthy" if clipped else "  (interior)"))

    if crop_to_labels:
        box = _bbox_of_labels(seg, pad)
    elif crop:
        box = _parse_crop(crop)
    else:
        box = (slice(None), slice(None), slice(None))

    seg = seg[box]
    print("  reading EM ...", flush=True)
    # Cropped identically and at the same level, so image and mask share a grid exactly.
    # BARCODE requires the mask to match the image XY exactly and to agree in z extent to
    # within 1 um; anything less than one shared box risks failing that check.
    image = np.asarray(em_array[box])

    labels_present = np.unique(seg)
    labels_present = labels_present[labels_present != 0]
    print(f"  cropped to {seg.shape}  ({len(labels_present)} object(s) in the mask)")

    if not keep_instances:
        seg = (seg != 0).astype(np.uint8)
    elif seg.dtype not in (np.uint8, np.uint16, np.uint32):
        seg = seg.astype(np.uint16)

    # A mask on the ACQUIRED anisotropic grid cannot be described to BARCODE:
    # mask_spacing_um is a single scalar meaning "isotropic at this spacing". Putting the
    # mask on an isotropic grid at the xy step here means the default mask_spacing_um: 0
    # describes the file correctly and nothing has to be overridden downstream. Same
    # reasoning as the CTC stager. Here z (3.44 nm) is FINER than xy (4.0 nm), so this
    # reduces the plane count rather than increasing it.
    if abs(z_um - xy_um) > 1e-12:
        before = seg.shape[0]
        seg = mask_z_to_isotropic(seg, z_um=z_um, xy_um=xy_um)
        print(f"  mask z resampled to isotropic at {xy_um} um: {before} -> {seg.shape[0]}")

    os.makedirs(os.path.join(out_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)

    # Trailing "_<digits>" would be read as a timelapse frame number by the default
    # timelapse_regex, so the resolution suffix is spelled with its unit.
    nm = f"{x_nm:g}nm".replace(".", "p")
    stem = f"{dataset}_{label}_{nm}"
    image_path = os.path.join(out_dir, "data", f"{stem}.tif")
    mask_path = os.path.join(out_dir, "masks", f"{stem}_SegMask.tif")

    write_volume(image_path, image, xy_um=xy_um, z_um=z_um)
    write_volume(mask_path, seg, xy_um=xy_um, z_um=xy_um)

    sidecar = {
        "source": f"s3://{BUCKET}/{dataset}/" if source == "s3" else os.path.abspath(source),
        "dataset": dataset,
        "em_level": scale,
        "segmentation": label,
        "segmentation_level": seg_level,
        "voxel_nm": {"z": z_nm, "y": y_nm, "x": x_nm},
        "image_um": {"xy": xy_um, "z": z_um},
        "mask_um_isotropic": xy_um,
        "image_shape_zyx": list(image.shape),
        "mask_shape_zyx": list(seg.shape),
        "object_labels": [int(v) for v in labels_present],
        "instances_kept": bool(keep_instances),
        "crop": None if box[0] == slice(None) else [
            [b.start, b.stop] for b in box],
        "note": "FIB-SEM contrast is inverted vs fluorescence; use the mask, not "
                "mean-relative thresholding, to define structure.",
    }
    with open(os.path.join(out_dir, f"{stem}.provenance.json"), "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2)

    print(f"\n  image -> {image_path}")
    print(f"  mask  -> {mask_path}")
    print(f"\nRun it with the barcode interpreter, no geometry overrides needed:\n"
          f"  ~/miniforge3/envs/barcode/python.exe scripts/run_volumetric_batch.py \\\n"
          f"      \"{os.path.join(out_dir, 'data')}\" --mode xyzt --mesh --packing "
          f"--component-stats \\\n"
          f"      --seg-root \"{os.path.join(out_dir, 'masks')}\"")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source", default="s3",
        help="local directory holding <dataset>.zarr and <dataset>.n5, or 's3' to read "
             "the public bucket directly (default: s3)")
    parser.add_argument("--dataset", default="jrc_jurkat-1", help="OpenOrganelle dataset ID")
    parser.add_argument(
        "--scale", default="s3",
        help="EM multiscale level. s0 is 4 nm and 257 Gvoxel -- far too large to hold; s3 "
             "(32 nm, ~0.5 Gvoxel) covers the whole volume comfortably (default: s3)")
    parser.add_argument(
        "--label", default="nucleus_seg",
        help="segmentation layer under <dataset>.n5/labels/ (default: nucleus_seg)")
    parser.add_argument(
        "--em-group", default="fibsem-uint8",
        help="EM array group under recon-1/em/ (default: fibsem-uint8)")
    parser.add_argument("--out", default="", help="output root (default: <source>/BARCODE)")
    parser.add_argument(
        "--crop", default="", metavar="z0,z1,y0,y1,x0,x1",
        help="voxel crop box at the chosen level")
    parser.add_argument(
        "--crop-to-labels", action="store_true",
        help="crop to the bounding box of the segmentation, padded by --pad")
    parser.add_argument("--pad", type=int, default=8, help="padding for --crop-to-labels")
    parser.add_argument(
        "--object", type=int, default=None, metavar="ID",
        help="keep only this instance label and drop the rest. Warns if it is cut by the "
             "edge of the field, which several nuclei in jrc_jurkat-1 are")
    parser.add_argument(
        "--binary", action="store_true",
        help="collapse instance labels to a single binary foreground. Off by default: the "
             "instance labels are the whole point for the packing and per-object metrics")
    args = parser.parse_args()

    if args.crop and args.crop_to_labels:
        raise SystemExit("--crop and --crop-to-labels are mutually exclusive")

    out_dir = args.out
    if not out_dir:
        if args.source == "s3":
            raise SystemExit("--out is required with --source s3 (and must not be on C:)")
        out_dir = os.path.join(args.source, "BARCODE")

    drive = os.path.splitdrive(os.path.abspath(out_dir))[0].upper()
    if drive == "C:":
        raise SystemExit(
            f"refusing to write outputs to {out_dir}: C: holds code, the data drives hold "
            f"data. Pass --out on a data drive.")

    stage(source=args.source, dataset=args.dataset, scale=args.scale, label=args.label,
          out_dir=out_dir, crop=args.crop, crop_to_labels=args.crop_to_labels,
          pad=args.pad, keep_instances=not args.binary, em_group=args.em_group,
          object_id=args.object)


if __name__ == "__main__":
    main()
