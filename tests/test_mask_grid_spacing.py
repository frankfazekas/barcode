"""A mask on the image's own grid takes the image's (anisotropic) spacing.

``mask_spacing_um`` is a single scalar meaning "isotropic at this spacing". That suits a
mask exported on a fine isotropic grid, but Cellpose masks arrive on the ACQUIRED grid,
which is anisotropic (0.195 xy, 0.235 z on the Drosophila set). Neither available scalar
is right: the z step relabels xy and inflates every in-plane length by z/xy, and 0
relabels z and shrinks the analysed depth. Both fail silently -- shapes still match and
nothing raises.

Same shape means same grid, so no guess is needed.

Run: python -m pytest tests/test_mask_grid_spacing.py -v
"""
from __future__ import annotations

import numpy as np
import tifffile

from core import BarcodeConfig

XY, Z = 0.195, 0.235


def write_pair(tmp_path, n_z=13, size=48):
    """An image and a label mask on the SAME acquired anisotropic grid."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    data = np.zeros((1, n_z, size, size), np.uint16)
    labels = np.zeros((n_z, size, size), np.int32)
    for i, (y, x) in enumerate(((12, 12), (12, 32), (32, 12), (32, 32)), start=1):
        data[0, 2:n_z - 2, y - 6:y + 6, x - 6:x + 6] = 400 + 100 * i
        labels[2:n_z - 2, y - 6:y + 6, x - 6:x + 6] = i

    image = str(tmp_path / "emb_1.tif")
    tifffile.imwrite(image, data, imagej=True, resolution=(1 / XY, 1 / XY),
                     metadata={"axes": "TZYX", "spacing": Z, "unit": "micron"})
    masks = tmp_path / "masks"
    masks.mkdir()
    tifffile.imwrite(str(masks / "emb_1_SegMask.tif"), labels)
    return image, str(masks)


def config_for(masks, mask_spacing_um=0.0):
    config = BarcodeConfig()
    config.modules.image_binarization = True
    config.modules.intensity_distribution = False
    v = config.volumetric
    v.analysis_mode = "xyzt"
    v.segmentation_enabled = True
    v.segmentation_root = masks
    v.xy_step_um, v.z_step_um = XY, Z
    v.mask_spacing_um = mask_spacing_um
    return config


def test_the_image_grid_wins_over_the_scalar(tmp_path):
    """Both plausible scalars give the same, correct grid.

    0.0 ("isotropic at xy") and the z step are the two a user would actually reach for,
    and each was wrong in a different direction before. A wildly wrong scalar is still
    rejected upstream by the z-extent check in ``load_segmentation`` -- that gate is
    unaffected; what changed is only which spacing the RESAMPLING uses.
    """
    from analysis.volumetric.run import run_volumetric_analysis

    spacings = []
    for tag, scalar in (("iso_at_xy", 0.0), ("z_step", Z)):
        image, masks = write_pair(tmp_path / tag)
        _, detail = run_volumetric_analysis(image, config_for(masks, scalar))
        spacings.append(detail.spacing_zyx_um)

    for spacing in spacings:
        assert np.allclose(spacing, (XY, XY, XY)), (
            f"expected isotropic {XY} um, got {spacing}: the scalar overrode the grid")
    assert spacings[0] == spacings[1], "the two plausible scalars must agree"


def test_the_depth_is_upsampled_not_relabelled(tmp_path):
    """13 slices at 0.235 um span 2.82 um; at 0.195 um that is ~15 slices, not 13."""
    from analysis.volumetric.run import run_volumetric_analysis

    image, masks = write_pair(tmp_path, n_z=13)
    _, detail = run_volumetric_analysis(image, config_for(masks))

    n_z = detail.shape_zyx[0]
    assert n_z > 13, "relabelling the spacing would leave the slice count untouched"
    expected = int(np.floor((13 - 1) * Z / XY)) + 1
    assert n_z == expected == 15


def test_a_finer_isotropic_mask_still_uses_the_scalar(tmp_path):
    """The Jurkat case must keep working: a mask on its own finer grid."""
    from analysis.volumetric.segmentation import _mask_spacing_um

    config = BarcodeConfig().volumetric
    config.mask_spacing_um = 0.065
    assert _mask_spacing_um(config, XY) == 0.065
    config.mask_spacing_um = 0.0
    assert _mask_spacing_um(config, XY) == XY, "0 still means 'isotropic at xy'"
