"""prepare_nucleus: anisotropic channel + isotropic mask -> common isotropic grid.

Ported unmodified (bar the import path) from chromatin-analysis
``tests/test_resample.py``, so the port can be checked against its origin.
"""
import numpy as np

from analysis.volumetric.resample import prepare_nucleus


def _sphere(shape, radius):
    zz, yy, xx = np.indices(shape)
    cz, cy, cx = (np.array(shape) - 1) / 2.0
    return ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2


def test_prepare_nucleus_makes_isotropic_aligned_grid():
    # Isotropic mask (0.065 um), anisotropic channel (z = 0.3 um) covering the
    # same physical extent.
    mask = _sphere((40, 40, 40), 14).astype(np.uint8)
    channel = np.ones((9, 40, 40), dtype=np.uint16) * 100

    images_iso, mask_iso, spacing_iso, info = prepare_nucleus(
        images={"hoechst": channel},
        image_spacings={"hoechst": (0.065, 0.065, 0.3)},
        mask=mask,
        mask_spacing=(0.065, 0.065, 0.065),
        crop_padding=1,
    )

    assert np.allclose(spacing_iso, (0.065, 0.065, 0.065))
    # channel resampled onto the mask grid, then both cropped identically
    assert images_iso["hoechst"].shape == mask_iso.shape
    assert mask_iso.dtype == np.uint8
    assert set(np.unique(mask_iso)).issubset({0, 1})
    assert mask_iso.max() == 1
    assert info["hoechst_canon"] == "resampled_to_mask_geometry"
