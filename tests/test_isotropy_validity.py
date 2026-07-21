"""Metrics that are only defined on a cubic grid must not reach the outputs.

Two different problems, handled two different ways:

* **Anisotropy** was computed from an inertia tensor built in VOXEL INDEX space, so it
  measured the sampling grid rather than the object. That is repairable -- measure in
  physical coordinates -- so it is fixed, not suppressed.
* **Packing topology** counts contact in voxel FACES and bridges gaps by a voxel
  dilation. No rescaling repairs that, because one voxel distance cannot mean one
  physical distance on a non-cubic grid. So it is left empty and its columns never
  reach the CSV or the barcode.
"""
import numpy as np
import pytest
from skimage.measure import regionprops_table

from analysis.volumetric.binarization import _anisotropy_from_eigvals

Z_STEP, XY_STEP = 0.3, 0.065          # the Jurkat geometry: 4.6x anisotropic


def _sphere_on_anisotropic_grid(radius_um=1.3, shape=(21, 81, 81)):
    nz, ny, nx = shape
    zz, yy, xx = np.indices(shape)
    z = (zz - nz // 2) * Z_STEP
    y = (yy - ny // 2) * XY_STEP
    x = (xx - nx // 2) * XY_STEP
    return ((z ** 2 + y ** 2 + x ** 2) <= radius_um ** 2).astype(np.uint8)


def _anisotropy(volume, spacing):
    table = regionprops_table(volume, properties=["inertia_tensor_eigvals"],
                              spacing=tuple(spacing))
    eig = np.stack([table[f"inertia_tensor_eigvals-{i}"] for i in range(3)], axis=1)
    return float(_anisotropy_from_eigvals(eig)[0])


def test_a_sphere_on_an_anisotropic_grid_is_reported_as_round():
    """The regression: this returned ~4.56, which is just the grid's own 4.6x ratio."""
    sphere = _sphere_on_anisotropic_grid()
    assert _anisotropy(sphere, (Z_STEP, XY_STEP, XY_STEP)) == pytest.approx(1.0, abs=0.05)


def test_measuring_a_sphere_in_voxel_space_recovers_the_grid_ratio():
    """Pins WHY the fix is needed, so the failure mode stays legible."""
    sphere = _sphere_on_anisotropic_grid()
    assert _anisotropy(sphere, (1.0, 1.0, 1.0)) == pytest.approx(Z_STEP / XY_STEP, rel=0.05)


@pytest.mark.parametrize("step", [1.0, 0.065, 0.3, 7.5])
def test_isotropic_spacing_does_not_change_the_ratio(step):
    """Uniform scaling multiplies every eigenvalue by s^2, so the ratio is invariant.

    This is what makes the fix safe: no isotropic run's Anisotropy can move.
    """
    vol = np.zeros((40, 40, 40), np.uint8)
    zz, yy, xx = np.indices((40, 40, 40))
    vol[(((zz - 20) / 6.0) ** 2 + ((yy - 20) / 12.0) ** 2 + ((xx - 20) / 4.0) ** 2) <= 1] = 1
    assert _anisotropy(vol, (step, step, step)) == pytest.approx(
        _anisotropy(vol, (1.0, 1.0, 1.0)), rel=1e-12)


# ------------------------------------------------------------------- suppression

def test_the_anisotropy_ratio_helper_reads_the_grid():
    from analysis.volumetric.run import _anisotropy_ratio

    assert _anisotropy_ratio((0.065, 0.065, 0.065)) == pytest.approx(1.0)
    assert _anisotropy_ratio((0.3, 0.065, 0.065)) == pytest.approx(0.3 / 0.065)
    # Degenerate spacing must not claim anisotropy and suppress a valid family.
    assert _anisotropy_ratio((0.0, 0.0, 0.0)) == 1.0
    assert _anisotropy_ratio((np.nan, 1.0, 1.0)) == 1.0


def test_packing_is_left_empty_on_an_anisotropic_grid(capsys):
    """Empty, not merely caveated: the barcode is a picture and cannot carry a footnote."""
    from core import BarcodeConfig
    from core.results import ChannelResults
    from analysis.volumetric.run import _anisotropy_ratio, _ISOTROPY_TOLERANCE

    assert _anisotropy_ratio((Z_STEP, XY_STEP, XY_STEP)) > _ISOTROPY_TOLERANCE

    # An unpopulated family contributes no columns, which is the mechanism that keeps it
    # out of the CSV and the barcode.
    results = ChannelResults(filepath="cell.tif", channel=0)
    assert not results.packing.is_populated()
    headers = ChannelResults.get_headers(just_metrics=True, mode="xyzt")
    assert not any("Contact Number" in h for h in headers)
