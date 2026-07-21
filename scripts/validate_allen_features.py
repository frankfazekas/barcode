#!/usr/bin/env python3
"""Compare object volumes against the Allen Institute's own published measurements.

This is the check that the mask comparisons in ``validate_open_data.py`` cannot be. There
a mask is used as the binarization and then measured against itself, so agreement is
guaranteed by construction. Here the reference was produced by a different laboratory
running different software over the same pixels, and can therefore disagree.

Allen publish ``metadata.csv`` for the hiPSC single-cell image dataset: 215,081 cells,
each with volume, surface area and depth measured by their pipeline, keyed by ``CellId``
and ``FOVId``. Every field of view held locally under ``AllenCell_data`` appears in it.

    curl -L -o <root>/_allen_metadata/metadata.csv \\
        https://allencell.s3.amazonaws.com/aics/hipsc_single_cell_image_dataset/metadata.csv
    python scripts/validate_allen_features.py --root L:/FF/Hackathon/full_datasets

**The QC filter is the trap.** Allen's table keeps only cells that passed quality
control -- complete cells, away from the field edge -- while the segmentation labels
*everything*, including cells clipped by the boundary, which are smaller by construction.

Cells are therefore joined by IDENTITY, not by size: ``this_cell_index`` is the label an
object carries in the FOV segmentation, verified against the published ``roi`` bounding
boxes. Matching by size rank instead pairs the wrong cells and is badly misleading -- on
one field it turned a 2% agreement into a 40% disagreement, purely as an artefact.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts._staging import read_tiff_any

DEFAULT_ROOT = r"L:\FF\Hackathon\full_datasets"
METADATA = os.path.join("_allen_metadata", "metadata.csv")

# Allen report volumes and areas as voxel counts on the isotropic grid their
# ``scale_micron`` column names. Read it from the file rather than assuming.
DEFAULT_SCALE_UM = 0.108333

# staged folder -> (Allen volume column, Allen area column)
# Suffix -> (Allen volume column, Allen area column). Matched by suffix because the
# staged folder carries whatever --name the stager was given (subset, full, ...).
TARGETS = {
    "nucleus": ("NUC_shape_volume", "NUC_roundness_surface_area"),
    "cell": ("MEM_shape_volume", "MEM_roundness_surface_area"),
}


def staged_datasets(root: str) -> Dict[str, Tuple[str, str]]:
    """Every staged allen_* folder, paired with the Allen columns it should match."""
    found = {}
    for path in sorted(glob.glob(os.path.join(root, "allen_*"))):
        if not os.path.isdir(os.path.join(path, "BARCODE", "masks")):
            continue
        for suffix, columns in TARGETS.items():
            if os.path.basename(path).endswith("_" + suffix):
                found[os.path.basename(path)] = (path, columns)
    return found


def read_allen(path: str, fovs: set, columns: List[str]) -> Dict[str, Dict[str, List[float]]]:
    """Per-FOV lists of the requested columns, for the FOVs we hold."""
    csv.field_size_limit(10 ** 7)
    out: Dict[str, Dict[str, List[float]]] = {f: {c: [] for c in columns} for f in fovs}
    scale: Optional[float] = None
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        missing = [c for c in columns + ["FOVId", "this_cell_index"] if c not in header]
        if missing:
            raise KeyError(f"metadata.csv lacks {missing}")
        index = {c: header.index(c) for c in columns + ["FOVId", "this_cell_index"]}
        scale_i = header.index("scale_micron") if "scale_micron" in header else None
        for row in reader:
            fov = row[index["FOVId"]]
            if fov not in out:
                continue
            if scale is None and scale_i is not None:
                try:
                    scale = float(str(row[scale_i]).strip("[] ").split(",")[0])
                except (ValueError, IndexError):
                    pass
            try:
                label = int(float(row[index["this_cell_index"]]))
            except ValueError:
                continue
            for c in columns:
                try:
                    out[fov][c].append((label, float(row[index[c]])))
                except ValueError:
                    pass
    return out, (scale or DEFAULT_SCALE_UM)


def measured_by_label(folder: str, fov: str, voxel_um3: float) -> Dict[int, float]:
    """Volume of every labelled object in the staged mask, keyed by its label.

    The label is the join key. Allen's ``this_cell_index`` is the value that object
    carries in the FOV segmentation, and staging preserves labels (the z resampling is
    nearest-neighbour precisely so it cannot invent or merge them). Verified against the
    published ``roi`` bounding boxes before being relied on.

    Matching cells by SIZE RANK instead -- which an earlier version of this script did --
    is not a substitute. Allen's table is quality-filtered, so the n cells they kept are
    not the n largest in the field, and ranking pairs the wrong cells together. On one
    field that alone turned a 2% agreement into a 40% disagreement.
    """
    path = os.path.join(folder, "BARCODE", "masks", f"{fov}_SegMask.tif")
    if not os.path.isfile(path):
        return {}
    mask = read_tiff_any(path)
    labels, counts = np.unique(mask[mask > 0], return_counts=True)
    return {int(l): float(c) * voxel_um3 for l, c in zip(labels, counts)}



def meshed_by_label(folder: str, fov: str, xy_um: float, maxrad: float
                    ) -> Dict[int, Tuple[float, float]]:
    """Mesh every object in the field; return {label: (volume_um3, area_um2)}.

    Surface area has no voxel-counted equivalent -- a voxelised surface is all axis-
    aligned faces and overestimates any curved boundary badly -- so the mesh is the only
    thing to compare against Allen's published areas. ``mesh_field`` meshes each labelled
    object separately, which is what keeps the labels available to join on.
    """
    from analysis.volumetric.mesh_field import mesh_field

    path = os.path.join(folder, "BARCODE", "masks", f"{fov}_SegMask.tif")
    if not os.path.isfile(path):
        return {}
    labels = read_tiff_any(path)
    field = mesh_field(labels, (xy_um,) * 3, maxrad=maxrad, min_voxels=512,
                       curvature=False, solidity=False)
    return {int(m.label): (m.geometry.volume_um3, m.geometry.surface_area_um2)
            for m in field.meshes}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="BARCODE volumes vs the Allen Institute's published per-cell values.")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--metadata", default=None, help="path to Allen metadata.csv")
    parser.add_argument("--mesh", action="store_true",
                        help="also mesh every object and compare MESH volume and SURFACE "
                             "AREA against Allen's published values. Slow (seconds per "
                             "object) but the only way to check surface area, which has "
                             "no voxel-counted equivalent")
    parser.add_argument("--mesh-maxrad", type=float, default=2.0,
                        help="triangle-size bound for --mesh. The pipeline default of 5 "
                             "is too coarse for cells this size (see validate_phantoms)")
    parser.add_argument("--mesh-fields", type=int, default=8,
                        help="how many fields to mesh (default 8; meshing is the slow part)")
    args = parser.parse_args()

    metadata = args.metadata or os.path.join(args.root, METADATA)
    if not os.path.isfile(metadata):
        print(f"No Allen metadata at {metadata}\nDownload it first -- see this file's docstring.")
        return 1

    discovered = staged_datasets(args.root)
    staged = {name: path for name, (path, _) in discovered.items()}
    if not staged:
        print(f"No allen_* datasets staged under {args.root}")
        return 1

    fovs = sorted({os.path.splitext(os.path.basename(p))[0]
                   for folder in staged.values()
                   for p in glob.glob(os.path.join(folder, "BARCODE", "data", "*.tif"))})
    print(f"{len(fovs)} staged field(s): {', '.join(fovs)}")

    columns = sorted({c for _, cols in discovered.values() for c in cols})
    allen, scale_um = read_allen(metadata, set(fovs), columns)
    voxel_um3 = scale_um ** 3
    print(f"Allen grid {scale_um} um isotropic -> voxel {voxel_um3:.7f} um^3\n")

    summary: Dict[str, List[float]] = {}
    per_cell: Dict[str, List[Tuple[float, float]]] = {}
    for name, folder in staged.items():
        volume_column = discovered[name][1][0]
        pairs: List[Tuple[float, float]] = []
        fields = 0
        for fov in fovs:
            published = allen.get(fov, {}).get(volume_column, [])
            ours = measured_by_label(folder, fov, voxel_um3)
            if not published or not ours:
                continue
            fields += 1
            for label, value in published:
                if label in ours:
                    pairs.append((value * voxel_um3, ours[label]))
        if not pairs:
            continue
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])
        ratio = b / a
        per_cell[name] = pairs
        summary[name] = list(ratio)
        print(f"=== {name}   vs {volume_column}")
        print(f"  {len(pairs):,} cells across {fields} field(s), matched by "
              f"this_cell_index")
        print(f"  BARCODE / Allen   median {np.median(ratio):.4f}   "
              f"IQR [{np.percentile(ratio, 25):.3f}, {np.percentile(ratio, 75):.3f}]")
        print(f"  correlation       r = {np.corrcoef(a, b)[0, 1]:.5f}")
        print(f"  within 5%         {100 * np.mean(np.abs(ratio - 1) < 0.05):.1f}% "
              f"of cells\n")

    # The nuclear-to-cell ratio is unit-free, so it survives every convention question
    # about what a "voxel" or a "volume" means on either side of the comparison.
    nuc = next((p for n, p in staged.items() if n.endswith("_nucleus")), None)
    cell = next((p for n, p in staged.items() if n.endswith("_cell")), None)
    if nuc and cell:
        an, am, bn, bm = [], [], [], []
        for fov in fovs:
            pn = dict(allen.get(fov, {}).get("NUC_shape_volume", []))
            pm = dict(allen.get(fov, {}).get("MEM_shape_volume", []))
            on = measured_by_label(nuc, fov, voxel_um3)
            om = measured_by_label(cell, fov, voxel_um3)
            for label in set(pn) & set(pm) & set(on) & set(om):
                an.append(pn[label] * voxel_um3); am.append(pm[label] * voxel_um3)
                bn.append(on[label]); bm.append(om[label])
        if an:
            ra = np.array(an) / np.array(am)
            rb = np.array(bn) / np.array(bm)
            print(f"nuclear-to-cell volume ratio, per cell (n={len(an):,})")
            print(f"  Allen  median {np.median(ra):.4f}     "
                  f"BARCODE median {np.median(rb):.4f}     "
                  f"{abs(np.median(rb) - np.median(ra)) / np.median(ra):.1%} apart")
            print("  A ratio cancels every voxel-size and volume-convention "
                  "question, which is why it is the strongest line here.")
    if args.mesh:
        print("\n" + "=" * 74)
        print("MESH metrics vs Allen -- volume and SURFACE AREA, per cell")
        for name, folder in staged.items():
            vol_col, area_col = discovered[name][1]
            pv, pa, mv, ma = [], [], [], []
            for fov in fovs[: args.mesh_fields]:
                published_v = dict(allen.get(fov, {}).get(vol_col, []))
                published_a = dict(allen.get(fov, {}).get(area_col, []))
                meshed = meshed_by_label(folder, fov, scale_um, args.mesh_maxrad)
                for label, (volume, area) in meshed.items():
                    if label in published_v and label in published_a:
                        pv.append(published_v[label] * voxel_um3)
                        pa.append(published_a[label] * scale_um ** 2)
                        mv.append(volume)
                        ma.append(area)
            if not pv:
                continue
            pv, pa, mv, ma = map(np.asarray, (pv, pa, mv, ma))
            print(f"\n=== {name}   ({len(pv)} cells, maxrad {args.mesh_maxrad})")
            for label, published, measured in (("mesh volume", pv, mv),
                                               ("surface area", pa, ma)):
                ratio = measured / published
                print(f"  {label:<14} BARCODE/Allen median {np.median(ratio):.4f}   "
                      f"IQR [{np.percentile(ratio, 25):.3f}, "
                      f"{np.percentile(ratio, 75):.3f}]   "
                      f"r = {np.corrcoef(published, measured)[0, 1]:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
