#!/usr/bin/env python3
"""Download Cell Tracking Challenge 3D+time datasets and stage them for BARCODE.

Why this exists: the volumetric branch has only ever run on one dataset (the Jurkat
nuclei on F:), one geometry, and masks we produced ourselves. CTC gives per-timepoint
3D TIFFs -- structurally identical to ``Cell1_N.tif`` -- across a 1x-to-11x anisotropy
range, with dense silver-standard masks somebody else made, and *published* voxel size
and time step. Two of those (spacing, dt) are exactly what our TIFFs cannot tell us.

Two things make raw CTC data unusable as-shipped:

* The TIFFs carry no axis or spacing metadata. ``analysis/volumetric/reader.py`` refuses
  to guess an axis order -- correctly -- so a plain 3-D page series is rejected.
* Filenames are ``01/t000.tif`` with masks at ``01_ST/SEG/man_seg000.tif``, which needs
  bespoke ``timelapse_regex`` / ``segmentation_template`` values at every call site.

Rather than push both problems onto every downstream command, staging fixes them once:
each volume is rewritten with honest ImageJ metadata (``axes=ZYX``, ``spacing``,
``XResolution``, ``finterval``) taken from the published table below, and renamed into
the conventions BARCODE already defaults to::

    <root>/ctc_<dataset>_<seq>/BARCODE/
        data/   <dataset>_<seq>_000.tif           <- default timelapse_regex groups these
        masks/  <dataset>_<seq>_000_SegMask.tif   <- default segmentation_template finds these
        results/                                   <- one folder per run
        README.txt                                 <- provenance + the command to run

The file then *declares* its geometry instead of being overridden on the command line,
which is the same footing the Jurkat data is on.

Masks are the silver standard (``_ST/SEG``), not gold. Gold ``_GT/SEG`` for 3D data is
sparse -- annotated on scattered slices of scattered frames -- and a missing frame makes
mask resolution raise. Silver is dense on every frame. For the simulated datasets the
gold truth *is* dense (the phantoms are known), so those prefer ``_GT/SEG``.

Usage::

    python scripts/fetch_ctc.py --list
    python scripts/fetch_ctc.py Fluo-N3DH-CHO --root L:/FF/Hackathon/full_datasets/_open_data
    python scripts/fetch_ctc.py --all --root L:/FF/Hackathon/full_datasets/_open_data

Downloads are cached in ``<root>/_ctc_downloads`` and resumed, so re-running is cheap.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts._staging import mask_z_to_isotropic, read_tiff_any, write_volume

BASE_URL = "https://data.celltrackingchallenge.net/training-datasets"

# Never stage onto C: -- that drive holds code, not data (see CLAUDE.md).
DEFAULT_ROOT = r"L:\FF\Hackathon\full_datasets\_open_data"


@dataclass(frozen=True)
class Dataset:
    """One CTC dataset, with the geometry the files themselves omit.

    ``xy_um``/``z_um``/``dt_s`` are transcribed from celltrackingchallenge.net/3d-datasets
    and are the whole reason this table exists: without them Speed is wrong by an
    arbitrary factor and every physical length is in voxels pretending to be microns.
    """

    name: str
    xy_um: float
    z_um: float
    dt_s: float
    approx_mb: int
    description: str
    prefer_gold: bool = False   # simulated sets have dense gold truth; real ones do not

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{self.name}.zip"

    @property
    def anisotropy(self) -> float:
        return self.z_um / self.xy_um


DATASETS: Tuple[Dataset, ...] = (
    Dataset("Fluo-N3DH-CHO", 0.202, 1.0, 570.0, 98,
            "CHO nuclei overexpressing GFP-PCNA; closest analogue to the Jurkat data"),
    Dataset("Fluo-C3DL-MDA231", 1.242, 6.0, 4800.0, 182,
            "MDA231 breast carcinoma in collagen; 4.8x anisotropy over only 30 slices"),
    Dataset("Fluo-C3DH-A549", 0.126, 1.0, 120.0, 244,
            "GFP-actin A549 in Matrigel; texture rather than blobs"),
    Dataset("Fluo-C3DH-A549-SIM", 0.126, 1.0, 20.0, 314,
            "simulated A549: object shape and motion are known exactly", prefer_gold=True),
    Dataset("Fluo-N3DH-SIM+", 0.125, 0.200, 1740.0, 3100,
            "simulated HL60 nuclei with exact ground truth", prefer_gold=True),
    Dataset("Fluo-N3DH-CE", 0.09, 1.0, 60.0, 3100,
            "C. elegans developing embryo; 11x anisotropy"),
    Dataset("Fluo-N3DL-DRO", 0.406, 2.03, 30.0, 5800,
            "Drosophila embryo; densely packed nuclei for the packing/ordering metrics"),
    Dataset("Fluo-C3DH-H157", 0.126, 0.5, 120.0, 7000,
            "GFP-transfected H157 lung carcinoma in Matrigel"),
)

# Deliberately absent: Fluo-N3DL-TRIF (320 GB training) and Fluo-N3DL-TRIC (cartographic
# projection, so the site publishes no voxel size -- there is nothing to calibrate with).

BY_NAME: Dict[str, Dataset] = {d.name: d for d in DATASETS}


# --------------------------------------------------------------------------- download

def download(dataset: Dataset, cache_dir: str) -> str:
    """Fetch the training zip, resuming a partial download rather than restarting."""
    os.makedirs(cache_dir, exist_ok=True)
    target = os.path.join(cache_dir, f"{dataset.name}.zip")

    if os.path.isfile(target) and _is_complete_zip(target):
        print(f"  cached: {target}")
        return target

    print(f"  downloading {dataset.url}  (~{dataset.approx_mb} MB)")
    # curl over urllib for the resume: these run to 7 GB over a link that does drop.
    result = subprocess.run(
        ["curl.exe", "-L", "-C", "-", "--fail", "--retry", "3", "-o", target, dataset.url],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"download failed for {dataset.name} (curl {result.returncode})")
    if not _is_complete_zip(target):
        raise RuntimeError(f"{target} is not a readable zip; delete it and retry")
    return target


def _is_complete_zip(path: str) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    except (zipfile.BadZipFile, OSError):
        return False


def extract(zip_path: str, work_dir: str) -> str:
    """Unzip into ``work_dir/<dataset>``, skipping if it is already there."""
    name = os.path.splitext(os.path.basename(zip_path))[0]
    out = os.path.join(work_dir, name)
    if os.path.isdir(out) and os.listdir(out):
        print(f"  extracted already: {out}")
        return out
    print(f"  extracting -> {out}")
    os.makedirs(work_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(work_dir)
    if not os.path.isdir(out):
        raise RuntimeError(f"expected {out} inside {zip_path}; found {os.listdir(work_dir)}")
    return out


# ---------------------------------------------------------------------------- staging

def find_sequences(extracted: str) -> List[str]:
    """CTC ships each dataset as numbered sequences: 01, 02, ..."""
    return sorted(
        entry for entry in os.listdir(extracted)
        if entry.isdigit() and os.path.isdir(os.path.join(extracted, entry))
    )


def mask_directory(extracted: str, sequence: str, prefer_gold: bool) -> Optional[str]:
    """Pick the mask folder: silver is dense on every frame, gold usually is not."""
    gold = os.path.join(extracted, f"{sequence}_GT", "SEG")
    silver = os.path.join(extracted, f"{sequence}_ST", "SEG")
    order = (gold, silver) if prefer_gold else (silver, gold)
    for candidate in order:
        if os.path.isdir(candidate) and _volume_masks(candidate):
            return candidate
    return None


def _volume_masks(directory: str) -> Dict[str, str]:
    """Map frame index -> mask path, keeping only whole-volume masks.

    Sparse gold truth is written per *slice* as ``man_seg_000_012.tif`` (frame, slice).
    Those are not volumes and pairing one to a frame would silently analyse a single
    plane as if it were the stack, so they are dropped by the two-underscore shape.
    """
    masks: Dict[str, str] = {}
    for entry in sorted(os.listdir(directory)):
        if not entry.startswith("man_seg") or not entry.endswith((".tif", ".tiff")):
            continue
        stem = os.path.splitext(entry)[0]
        digits = stem[len("man_seg"):]
        if not digits.isdigit():        # "_000_012" -> per-slice annotation, skip
            continue
        masks[digits] = os.path.join(directory, entry)
    return masks


def stage_sequence(dataset: Dataset, extracted: str, sequence: str, root: str) -> Optional[str]:
    """Stage one sequence into its own dataset folder. Returns the folder, or None."""
    source = os.path.join(extracted, sequence)
    frames = sorted(
        entry for entry in os.listdir(source)
        if entry.startswith("t") and entry.endswith((".tif", ".tiff"))
    )
    if not frames:
        print(f"  {sequence}: no t*.tif frames, skipping")
        return None

    mask_dir = mask_directory(extracted, sequence, dataset.prefer_gold)
    masks = _volume_masks(mask_dir) if mask_dir else {}
    if mask_dir:
        print(f"  {sequence}: masks from {os.path.basename(os.path.dirname(mask_dir))}"
              f" ({len(masks)} volume masks)")
    else:
        print(f"  {sequence}: no whole-volume masks found -- staging images only")

    # Only frames that have a mask, when masks exist at all: a frame whose mask is
    # missing makes resolve_segmentation_path raise mid-run, which is a worse failure
    # than never staging it.
    if masks:
        frames = [f for f in frames if os.path.splitext(f)[0][1:] in masks]
        if not frames:
            print(f"  {sequence}: masks exist but none pair with a frame, skipping")
            return None

    folder = os.path.join(root, f"ctc_{dataset.name}_{sequence}")
    data_dir = os.path.join(folder, "BARCODE", "data")
    mask_out = os.path.join(folder, "BARCODE", "masks")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(folder, "BARCODE", "results"), exist_ok=True)
    if masks:
        os.makedirs(mask_out, exist_ok=True)

    print(f"  {sequence}: staging {len(frames)} timepoint(s) -> {folder}")
    for entry in frames:
        index = os.path.splitext(entry)[0][1:]          # "t000" -> "000"
        stem = f"{dataset.name}_{sequence}_{index}"
        volume = _as_zyx(read_tiff_any(os.path.join(source, entry)), entry)
        write_volume(os.path.join(data_dir, f"{stem}.tif"), volume,
                     dataset.xy_um, dataset.z_um, dataset.dt_s)
        if masks:
            # Staged isotropic at xy so mask_spacing_um can stay at its default.
            mask = mask_z_to_isotropic(
                _as_zyx(read_tiff_any(masks[index]), masks[index]),
                dataset.z_um, dataset.xy_um)
            write_volume(os.path.join(mask_out, f"{stem}_SegMask.tif"), mask,
                         dataset.xy_um, dataset.xy_um, dataset.dt_s)

    write_readme(dataset, sequence, folder, len(frames), bool(masks), mask_dir)
    return folder


def _as_zyx(array: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 3:
        squeezed = np.squeeze(array)
        if squeezed.ndim != 3:
            raise ValueError(f"{label}: expected a 3-D volume, got {array.shape}")
        return squeezed
    return array


README = """\
{name}, sequence {sequence}
{underline}

{description}

Source      Cell Tracking Challenge training data
            {url}
            https://celltrackingchallenge.net/3d-datasets/
Citation    Maska M et al., "The Cell Tracking Challenge: 10 years of objective
            benchmarking", Nature Methods 20, 1010-1020 (2023).
            Datasets are free for non-commercial research; cite the challenge and the
            dataset's own reference as listed on the site.

Geometry (published by CTC -- the raw TIFFs carry no metadata at all)
    xy step         {xy} um/pixel
    z step          {z} um            (anisotropy {aniso:.1f}x)
    frame interval  {dt} s

Staged by scripts/fetch_ctc.py, which rewrote every volume with ImageJ metadata
(axes=ZYX, spacing, XResolution, finterval) carrying the numbers above. The files now
declare their own geometry, so no --axes / --xy-step / --z-step override is needed --
the volumetric reader reads them the same way it reads the Jurkat stacks.

    data/   {frames} timepoints, {name}_{sequence}_NNN.tif
    masks/  {mask_note}

CTC masks ship on the image's own ANISOTROPIC grid, which BARCODE cannot describe:
mask_spacing_um is one scalar meaning "isotropic at this spacing", and prepare_volume
resamples the image onto whatever grid it names. Setting it to the z step therefore
does not describe the mask -- it resamples everything to {z} um cubes and discards the
xy resolution ({xy} um), which still yields perfectly plausible-looking numbers.

Staging resampled the masks instead, to isotropic at the xy step -- the same footing as
the Jurkat masks. So LEAVE mask_spacing_um AT 0 (its default); do not pass
--mask-spacing. Mask z slices were multiplied by {z}/{xy} accordingly.

Run it:

    python scripts/run_volumetric_timelapse_barcode.py "{data_path}" \\
        --seg-root "{mask_path}" \\
        --frame-interval {dt} --component-stats --mesh

Filenames were chosen so BARCODE's DEFAULTS apply: timelapse_regex groups
"{name}_{sequence}" as one series, and segmentation_template "{{stem}}_SegMask.tif"
resolves each mask. Nothing bespoke to remember.

Write run outputs to results/<run name>/ in this folder -- never to C:.
"""


def write_readme(dataset: Dataset, sequence: str, folder: str, frames: int,
                 has_masks: bool, mask_dir: Optional[str]) -> None:
    title = f"{dataset.name}, sequence {sequence}"
    standard = ""
    if mask_dir:
        standard = "silver standard (_ST)" if "_ST" in mask_dir else "gold truth (_GT)"
    text = README.format(
        name=dataset.name,
        sequence=sequence,
        underline="=" * len(title),
        description=dataset.description,
        url=dataset.url,
        xy=dataset.xy_um,
        z=dataset.z_um,
        dt=dataset.dt_s,
        aniso=dataset.anisotropy,
        frames=frames,
        mask_note=(f"{frames} masks, {standard}, {dataset.name}_{sequence}_NNN_SegMask.tif"
                   if has_masks else "(none -- no whole-volume masks in this dataset)"),
        data_path=os.path.join(folder, "BARCODE", "data"),
        mask_path=os.path.join(folder, "BARCODE", "masks"),
    )
    with open(os.path.join(folder, "README.txt"), "w", encoding="utf-8") as handle:
        handle.write(text)


# ------------------------------------------------------------------------------- main

def fetch(dataset: Dataset, root: str, keep_raw: bool) -> List[str]:
    print(f"\n{dataset.name}  --  {dataset.description}")
    cache = os.path.join(root, "_ctc_downloads")
    zip_path = download(dataset, cache)
    extracted = extract(zip_path, cache)

    staged: List[str] = []
    sequences = find_sequences(extracted)
    if not sequences:
        print(f"  no numbered sequence folders in {extracted}")
        return staged
    for sequence in sequences:
        folder = stage_sequence(dataset, extracted, sequence, root)
        if folder:
            staged.append(folder)

    if not keep_raw:
        # The staged copies are self-describing and complete; the extracted tree is a
        # duplicate of the zip, which is itself kept for re-staging.
        shutil.rmtree(extracted, ignore_errors=True)
    return staged


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and stage Cell Tracking Challenge 3D+time datasets.")
    parser.add_argument("datasets", nargs="*", help="dataset names (see --list)")
    parser.add_argument("--all", action="store_true", help="every dataset in the table")
    parser.add_argument("--list", action="store_true", help="show the table and exit")
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help=f"staging root, on a data drive (default: {DEFAULT_ROOT})")
    parser.add_argument("--keep-raw", action="store_true",
                        help="keep the extracted CTC tree next to the zip")
    args = parser.parse_args()

    if args.list:
        print(f"{'dataset':<22}{'xy um':>8}{'z um':>8}{'aniso':>7}{'dt s':>9}{'MB':>7}  description")
        for dataset in DATASETS:
            print(f"{dataset.name:<22}{dataset.xy_um:>8}{dataset.z_um:>8}"
                  f"{dataset.anisotropy:>6.1f}x{dataset.dt_s:>9}{dataset.approx_mb:>7}"
                  f"  {dataset.description}")
        return 0

    chosen = list(DATASETS) if args.all else [BY_NAME[n] for n in args.datasets if n in BY_NAME]
    unknown = [n for n in args.datasets if n not in BY_NAME]
    if unknown:
        print(f"unknown dataset(s): {unknown}; see --list")
        return 1
    if not chosen:
        parser.print_help()
        return 1

    if os.path.abspath(args.root)[:2].upper() == "C:":
        print("Refusing to stage data on C: -- that drive holds code, not data.")
        return 1

    os.makedirs(args.root, exist_ok=True)
    staged: List[str] = []
    failed: List[str] = []
    for dataset in chosen:
        try:
            staged.extend(fetch(dataset, args.root, args.keep_raw))
        except Exception as error:                      # one bad download != lose the rest
            print(f"  FAILED {dataset.name}: {error}")
            failed.append(dataset.name)

    print(f"\nStaged {len(staged)} sequence folder(s) under {args.root}:")
    for folder in staged:
        print(f"  {folder}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    return 1 if failed and not staged else 0


if __name__ == "__main__":
    raise SystemExit(main())
