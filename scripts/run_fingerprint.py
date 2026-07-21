"""Build a fingerprint card for one volume without re-running a whole analysis.

The card is written automatically at the end of a volumetric run. This exists for the
other case: iterating on the layout, or producing a card for a volume whose run happened
before the card did. It re-runs the analysis for that single file (cheap next to a batch)
and renders the result.

    python scripts/run_fingerprint.py path/to/Cell1_1.tif --seg-root path/to/masks --mesh
    python scripts/run_fingerprint.py emb_1.tif --xy-step 0.195 --z-step 0.235 \
        --seg-root masks --packing --mask-intensity

Outputs land beside the input unless --out says otherwise. Per the repo convention that
means the data drive, never C:.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import BarcodeConfig  # noqa: E402
from scripts._cli import add_metric_arguments, add_mode_arguments, apply_common  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="a single volumetric file")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--out", default=None, metavar="PNG",
                        help="output path (default: '<stem> Fingerprint.png' beside the input)")
    parser.add_argument("--seg-root", default=None)
    parser.add_argument("--seg-regex", default=None)
    parser.add_argument("--seg-template", default=None)
    parser.add_argument("--mesh", action="store_true", help="surface meshing + curvature")
    add_mode_arguments(parser, default="xyzt")
    add_metric_arguments(parser)
    args = parser.parse_args()

    if not os.path.isfile(args.path):
        print(f"No such file: {args.path}")
        return 1

    config = apply_common(BarcodeConfig(), args)
    config.modules.image_binarization = True
    config.modules.intensity_distribution = True
    volumetric = config.volumetric
    volumetric.write_fingerprint = True
    if args.seg_root:
        volumetric.segmentation_enabled = True
        volumetric.segmentation_root = args.seg_root
    if args.seg_regex:
        volumetric.segmentation_regex = args.seg_regex
    if args.seg_template:
        volumetric.segmentation_template = args.seg_template
    for attribute in ("mesh_enabled", "enable_mesh", "make_mesh"):
        if hasattr(volumetric, attribute):
            setattr(volumetric, attribute, bool(args.mesh))
    if args.mesh:
        volumetric.mesh_curvature = True

    from analysis.volumetric.run import run_volumetric_analysis
    from visualization.fingerprint import build_fingerprint

    results, detail = run_volumetric_analysis(args.path, config, channel=args.channel)

    stem = os.path.splitext(os.path.basename(args.path))[0]
    figpath = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.path)), f"{stem} Fingerprint.png")
    frame = detail.representative_frame
    written = build_fingerprint(
        detail.analysed_volume, detail.analysed_mask, detail.spacing_zyx_um,
        results, detail, volumetric.mode,
        title=f"{stem}  —  {volumetric.mode.key}"
              + (f", timepoint {frame}" if frame is not None else ""),
        figpath=figpath, dpi=volumetric.fingerprint_dpi,
    )
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
