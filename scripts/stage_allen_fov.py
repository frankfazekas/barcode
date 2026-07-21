#!/usr/bin/env python3
"""Stage Allen Institute FOV datasets (image + published segmentation) for BARCODE.

Complements ``fetch_ctc.py``. CTC supplies *time*: per-timepoint volumes with dense
masks, which is what the change and flow metrics need. Allen supplies *number and
crowding*: hundreds of independent 3D fields of view, each with an INSTANCE
segmentation of a confluent hiPSC colony -- dozens of labelled, touching cells per
field. That is the case ``analysis/volumetric/packing.py`` and ``ordering.py`` were
written for and which no dataset on this machine has previously exercised: in a
confluent field, connectivity labelling fuses every cell into one component, so only a
real instance segmentation distinguishes a correct answer from a plausible one.

These FOVs are ``SizeT=1`` -- single timepoints, not time-lapses. Every change, speed
and flow metric is therefore out of scope here; use CTC for those.

Expected source layout (as found on L:)::

    <source>/raw_image/<id>_original.tiff              ZCYX, 7ch, the raw acquisition
            /dye/<id>.tiff                             CZYX, 2ch: ch0 membrane, ch1 nucleus
            /structure/<id>.tiff                       ZYX,  the tagged structure
            /fov_segmentation/<id>_fov_segmentation.tiff   ZCYX, 4ch INSTANCE labels
            /structure_segmentation/<id>_structure_segmentation.tiff  ZYX, binary

The ``fov_segmentation`` channels were identified from the data rather than from the
OME channel names (which are unreliable in these files -- the raw stack's names repeat
and misattribute). Measured on a sample FOV, per-channel occupancy and the intensity
contrast of each candidate image channel inside each mask:

    ch0   9.8% of volume, labels 1..N   nucleus        (dye ch1 brighter inside)
    ch1  36.3% of volume, labels 1..N   whole cell     (dye ch0 brighter inside, and
                                                        dimmer inside the nucleus --
                                                        a membrane stain, as expected)
    ch2   0.45%                          nuclear contour shell
    ch3   1.17%                          cell contour shell

Output, per BARCODE's defaults so nothing needs overriding::

    <root>/allen_<name>_<target>/BARCODE/
        data/    <id>.tif             ZYX, declaring xy/z spacing from the OME header
        masks/   <id>_SegMask.tif     instance labels, isotropic at the xy step
        results/
        README.txt

Usage::

    python scripts/stage_allen_fov.py --source L:/FF/AllenCell_data/centrosome/subset
    python scripts/stage_allen_fov.py --source L:/FF/AllenCell_data/centrosome/full \\
        --target cell --limit 50
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tifffile

from scripts._staging import mask_z_to_isotropic, read_tiff_any, write_volume

DEFAULT_ROOT = r"L:\FF\Hackathon\full_datasets"

# Fallbacks only. The real values are read per file from each OME header; these are what
# the sampled FOVs carry, kept so a file with a stripped header fails loudly against a
# sane reference rather than silently analysing voxels as if they were microns.
FALLBACK_XY_UM = 0.10833333333333332
FALLBACK_Z_UM = 0.29


@dataclass(frozen=True)
class Target:
    """One image/mask pairing available in an Allen FOV set."""

    key: str
    image_dir: str
    image_suffix: str
    image_channel: Optional[int]     # None: the file is already single-channel ZYX
    mask_dir: str
    mask_suffix: str
    mask_channel: Optional[int]
    description: str


TARGETS: Tuple[Target, ...] = (
    Target("nucleus", "dye", ".tiff", 1,
           "fov_segmentation", "_fov_segmentation.tiff", 0,
           "Hoechst nuclei + per-cell nucleus instance labels; the direct analogue of "
           "the Jurkat nucleus work, but confluent and dozens per field"),
    Target("cell", "dye", ".tiff", 0,
           "fov_segmentation", "_fov_segmentation.tiff", 1,
           "CellMask membrane + whole-cell instance labels; the packing/ordering case "
           "-- touching cells that only labels can separate"),
    Target("structure", "structure", ".tiff", None,
           "structure_segmentation", "_structure_segmentation.tiff", None,
           "tagged structure (centrosome) + its binary segmentation; tiny, sparse "
           "objects rather than filled bodies"),
)

BY_KEY: Dict[str, Target] = {t.key: t for t in TARGETS}


def read_spacing(*candidates: str) -> Tuple[float, float]:
    """Pull (xy, z) micron spacing from the first OME header that states it.

    Several candidates because the derived files in these sets are inconsistent: the
    ``dye`` and ``fov_segmentation`` exports carry no PhysicalSize at all, while
    ``raw_image`` and ``structure_segmentation`` -- same voxels, same field -- do. Every
    physical metric scales off these numbers, so it is worth looking in the sibling
    file rather than reaching for a constant.

    Falls back with a warning rather than raising: one header-stripped field should not
    abort a 632-file run, but it must not pass unremarked either.
    """
    for path in candidates:
        if not path or not os.path.isfile(path):
            continue
        try:
            with tifffile.TiffFile(path) as handle:
                meta = handle.ome_metadata or ""
        except Exception:
            continue
        x = re.search(r'PhysicalSizeX="([\d.eE+-]+)"', meta)
        z = re.search(r'PhysicalSizeZ="([\d.eE+-]+)"', meta)
        if x and z:
            return float(x.group(1)), float(z.group(1))
    shown = os.path.basename(candidates[0]) if candidates else "?"
    print(f"    ! {shown}: no PhysicalSize in any sibling header; "
          f"assuming {FALLBACK_XY_UM} / {FALLBACK_Z_UM} um")
    return FALLBACK_XY_UM, FALLBACK_Z_UM


def find_ids(source: str, target: Target) -> List[str]:
    """FOV ids that have BOTH an image and a mask for this target."""
    image_dir = os.path.join(source, target.image_dir)
    mask_dir = os.path.join(source, target.mask_dir)
    if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
        raise FileNotFoundError(
            f"expected {image_dir!r} and {mask_dir!r}; is --source the folder that "
            f"contains raw_image/, dye/, fov_segmentation/ ...?"
        )

    images = {
        entry[: -len(target.image_suffix)]
        for entry in os.listdir(image_dir) if entry.endswith(target.image_suffix)
    }
    masks = {
        entry[: -len(target.mask_suffix)]
        for entry in os.listdir(mask_dir) if entry.endswith(target.mask_suffix)
    }
    paired = sorted(images & masks, key=lambda s: (len(s), s))
    for orphan in sorted(images ^ masks):
        print(f"    ! unpaired: {orphan}")
    return paired


def _pick_channel(volume: np.ndarray, channel: Optional[int], label: str) -> np.ndarray:
    """Reduce a loaded array to ZYX, taking one channel of a ZCYX/CZYX stack.

    Which axis is the channel axis is decided by the ARRAY, not by convention: these
    files genuinely differ (``dye`` is CZYX, ``fov_segmentation`` is ZCYX), and picking
    the wrong axis returns a plane-count slice that still looks like a volume.
    """
    volume = np.asarray(volume)
    if channel is None:
        if volume.ndim == 3:
            return volume
        raise ValueError(f"{label}: expected a 3-D volume, got {volume.shape}")
    if volume.ndim != 4:
        raise ValueError(f"{label}: expected a 4-D stack to take channel {channel} "
                         f"from, got {volume.shape}")
    # The channel axis is the short one; Z is far longer than the handful of channels.
    axis = int(np.argmin(volume.shape[:2]))
    if channel >= volume.shape[axis]:
        raise ValueError(f"{label}: channel {channel} out of range for axis {axis} "
                         f"of {volume.shape}")
    return volume[channel] if axis == 0 else volume[:, channel]


def stage(source: str, target: Target, root: str, name: str,
          limit: Optional[int]) -> Optional[str]:
    ids = find_ids(source, target)
    if limit:
        ids = ids[:limit]
    if not ids:
        print("  no paired FOVs found")
        return None

    folder = os.path.join(root, f"allen_{name}_{target.key}")
    data_dir = os.path.join(folder, "BARCODE", "data")
    mask_dir = os.path.join(folder, "BARCODE", "masks")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(os.path.join(folder, "BARCODE", "results"), exist_ok=True)

    print(f"  staging {len(ids)} FOV(s) -> {folder}")
    xy_um = z_um = None
    instances: List[int] = []
    for n, fid in enumerate(ids, 1):
        image_path = os.path.join(source, target.image_dir, fid + target.image_suffix)
        mask_path = os.path.join(source, target.mask_dir, fid + target.mask_suffix)
        xy_um, z_um = read_spacing(
            image_path, mask_path,
            os.path.join(source, "raw_image", f"{fid}_original.tiff"),
            os.path.join(source, "structure_segmentation",
                         f"{fid}_structure_segmentation.tiff"),
        )

        image = _pick_channel(read_tiff_any(image_path), target.image_channel, fid)
        mask = _pick_channel(read_tiff_any(mask_path), target.mask_channel, fid)
        if mask.shape != image.shape:
            print(f"    ! {fid}: mask {mask.shape} != image {image.shape}, skipping")
            continue
        instances.append(int(np.count_nonzero(np.unique(mask))))

        write_volume(os.path.join(data_dir, f"{fid}.tif"), image, xy_um, z_um)
        # Isotropic at xy so mask_spacing_um stays at its default -- see _staging.
        write_volume(os.path.join(mask_dir, f"{fid}_SegMask.tif"),
                     mask_z_to_isotropic(mask, z_um, xy_um), xy_um, xy_um)
        if n % 25 == 0 or n == len(ids):
            print(f"    {n}/{len(ids)}")

    if not instances:
        print("  nothing staged")
        return None
    write_readme(folder, source, target, name, len(instances), instances, xy_um, z_um)
    print(f"  objects per field: min {min(instances)}  "
          f"median {int(np.median(instances))}  max {max(instances)}")
    return folder


README = """\
Allen Institute {name} FOVs -- {key}
{underline}

{description}

Source      {source}
            Allen Institute for Cell Science, allencell.org
            https://www.allencell.org/data-downloading.html
            Allen Cell imaging collections are released for research use; cite the
            Allen Institute and the dataset's own publication.

Contents    {fields} fields of view, each a SINGLE TIMEPOINT 3-D volume (OME SizeT=1).
            image  {image_dir}/{image_ch}
            mask   {mask_dir}/{mask_ch}
            {objects} objects per field (min {omin}, median {omed}, max {omax}).

Geometry    xy {xy} um/pixel, z {z} um  (anisotropy {aniso:.1f}x), read per file from
            each OME header rather than assumed.

Staged by scripts/stage_allen_fov.py. Volumes were rewritten with ImageJ metadata
(axes=ZYX, spacing, XResolution) so they declare their own geometry -- no --axes /
--xy-step / --z-step override is needed. Masks were resampled to isotropic at the xy
step, so LEAVE mask_spacing_um AT 0 (its default) and do not pass --mask-spacing.

BECAUSE THESE ARE SINGLE TIMEPOINTS: every change / speed / flow metric is out of
scope. Do not run the time-lapse script against them -- it would group unrelated fields
into a fake "series" and report differences between different cells as dynamics. Use
the per-file batch runner, which treats each field as the independent sample it is:

    python scripts/run_volumetric_batch.py "{data_path}" \\
        --seg-root "{mask_path}" --mode xyz --packing --component-stats

The masks are INSTANCE labels, which is the point: in a confluent colony connectivity
labelling fuses touching cells into one component, so the packing and ordering metrics
are only meaningful against labels somebody else assigned.

Write run outputs to results/<run name>/ in this folder -- never to C:.
"""


def write_readme(folder: str, source: str, target: Target, name: str, fields: int,
                 instances: List[int], xy_um: float, z_um: float) -> None:
    title = f"Allen Institute {name} FOVs -- {target.key}"
    text = README.format(
        name=name, key=target.key, underline="=" * len(title),
        description=target.description, source=source, fields=fields,
        image_dir=target.image_dir,
        image_ch=("single channel" if target.image_channel is None
                  else f"channel {target.image_channel}"),
        mask_dir=target.mask_dir,
        mask_ch=("single channel" if target.mask_channel is None
                 else f"channel {target.mask_channel}"),
        objects="Instance-labelled" if max(instances) > 1 else "Binary,",
        omin=min(instances), omed=int(np.median(instances)), omax=max(instances),
        xy=xy_um, z=z_um, aniso=z_um / xy_um,
        data_path=os.path.join(folder, "BARCODE", "data"),
        mask_path=os.path.join(folder, "BARCODE", "masks"),
    )
    with open(os.path.join(folder, "README.txt"), "w", encoding="utf-8") as handle:
        handle.write(text)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage Allen Institute FOV image/segmentation pairs for BARCODE.")
    parser.add_argument("--source", required=True,
                        help="folder containing dye/, structure/, fov_segmentation/ ...")
    parser.add_argument("--target", default="nucleus", choices=list(BY_KEY),
                        help="; ".join(f"{t.key}: {t.description}" for t in TARGETS))
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help=f"staging root, on a data drive (default: {DEFAULT_ROOT})")
    parser.add_argument("--name", default=None,
                        help="dataset name for the output folder (default: from --source)")
    parser.add_argument("--limit", type=int, default=None,
                        help="stage only the first N fields; use for a quick trial")
    args = parser.parse_args()

    if os.path.abspath(args.root)[:2].upper() == "C:":
        print("Refusing to stage data on C: -- that drive holds code, not data.")
        return 1

    source = os.path.normpath(args.source)
    name = args.name or "_".join(
        p for p in source.replace("/", "\\").split("\\")[-2:] if p)
    target = BY_KEY[args.target]

    print(f"{name}  --  {target.key}: {target.description}")
    folder = stage(source, target, args.root, name, args.limit)
    if folder is None:
        return 1
    print(f"\nStaged -> {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
