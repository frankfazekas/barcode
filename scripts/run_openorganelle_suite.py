"""One barcode of a Jurkat cell's organelles, from Janelia's FIB-SEM volume.

``stage_openorganelle.py`` extracts one segmentation layer. This drives it across many
layers of the same volume and then runs BARCODE once over all of them, so the result is a
single barcode with **one row per organelle class** -- nucleus, chromatin, nucleolus,
mitochondria, ER, microtubules, centrosome -- measured on identical voxels by identical
code. The barcode normalises per column, which is exactly the right reading here: within a
column, how does chromatin's island separation compare with mitochondria's?

This is the one artifact the synthetic phantoms cannot produce. Phantoms tell you the
geometry code is correct; this tells you what the metrics actually say about real
subcellular structure across a range of morphologies -- compact blobs (nucleolus), networks
(ER), dispersed populations (mitochondria, vesicles) and filaments (microtubules).

Two interpreters are involved, and neither can do the other's job: staging needs zarr,
which the pinned ``barcode`` env does not have and must not be given; analysis needs
``core``. So this script runs under the zarr interpreter and shells out to the barcode one.

    ~/miniforge3/envs/napari_test/python.exe scripts/run_openorganelle_suite.py \
        --source L:/FF/Hackathon/full_datasets/jrc_jurkat-1 \
        --out L:/FF/Hackathon/full_datasets/jrc_jurkat-1/BARCODE_suite

Note the EM volume is written once per layer, because BARCODE pairs ``data/X.tif`` with
``masks/X_SegMask.tif`` by name. At s4 that is ~60 MB a copy, which is the cheap way to
avoid special-casing a shared-image layout inside the pipeline.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Chosen to span morphologies, not to be exhaustive: a compact blob, a dense network, a
# dispersed population and a filament system all stress different metrics. Layers that are
# predictions of a MEMBRANE rather than a filled body (``pm_seg``, ``*-mem_seg``) are left
# out -- they come back as one connected sheet plus fragments and the per-object metrics
# describe the sheet, not the cells. See docs/internal/volumetric_validation.md.
DEFAULT_LAYERS = [
    "nucleus_seg",     # 11 whole nuclei -- the shape reference
    "chrom_seg",       # chromatin: network-like, fills the nucleus
    "nhchrom_seg",     # heterochromatin: denser, more clustered than chrom
    "nucleolus_seg",   # compact blobs
    "mito_seg",        # dispersed population, elongated
    "er_seg",          # connected network, high surface-to-volume
    "golgi_seg",       # stacked sheets
    "lyso_seg",        # small compact population
    "vesicle_seg",     # many tiny objects -- stresses the object-count path
    "cent_seg",        # centrosome: matches our live jurkat_nucleus_centrosome data
    "mt-out_seg",      # microtubules: filaments, the anisotropy extreme
    "np_seg",          # nuclear pores: punctate, on a surface
]

BARCODE_PYTHON = os.path.expanduser("~/miniforge3/envs/barcode/python.exe")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="local OpenOrganelle dataset root")
    parser.add_argument("--dataset", default="jrc_jurkat-1")
    parser.add_argument("--out", required=True, help="output root (must not be on C:)")
    parser.add_argument(
        "--scale", default="s4",
        help="EM level. s4 (~62 Mvoxel) sweeps all layers in minutes; s3 is 8x finer and "
             "8x slower (default: s4)")
    parser.add_argument("--layers", nargs="*", default=DEFAULT_LAYERS)
    parser.add_argument("--barcode-python", default=BARCODE_PYTHON)
    parser.add_argument(
        "--stage-only", action="store_true", help="stage the layers but do not analyse")
    args = parser.parse_args()

    if os.path.splitdrive(os.path.abspath(args.out))[0].upper() == "C:":
        raise SystemExit(f"refusing to write {args.out}: outputs belong on a data drive.")

    here = os.path.dirname(os.path.abspath(__file__))
    staged, failed = [], []

    for layer in args.layers:
        print(f"\n=== staging {layer} ===", flush=True)
        command = [
            sys.executable, os.path.join(here, "stage_openorganelle.py"),
            "--source", args.source, "--dataset", args.dataset,
            "--scale", args.scale, "--label", layer, "--out", args.out,
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        print(result.stdout.strip()[-900:] or result.stderr.strip()[-900:])
        # A missing layer is normal -- the layer list is shared across datasets and not
        # every dataset has every organelle. Keep going and report at the end.
        (staged if result.returncode == 0 else failed).append(layer)

    print(f"\nstaged {len(staged)}/{len(args.layers)} layer(s)")
    if failed:
        print(f"  unavailable in {args.dataset}: {', '.join(failed)}")
    if args.stage_only or not staged:
        return

    print("\n=== running BARCODE over every staged layer ===", flush=True)
    command = [
        args.barcode_python, os.path.join(here, "run_barcode.py"),
        os.path.join(args.out, "data"), "--mode", "xyzt",
        "--seg-root", os.path.join(args.out, "masks"),
        "--mesh", "--component-stats",
    ]
    result = subprocess.run(command, text=True)
    if result.returncode != 0:
        raise SystemExit(f"BARCODE run failed ({result.returncode})")

    print(f"\nOne row per organelle class, in {os.path.join(args.out, 'data')}.")
    print("The barcode normalises PER COLUMN, so compare classes down a column, never "
          "across columns.")


if __name__ == "__main__":
    main()
