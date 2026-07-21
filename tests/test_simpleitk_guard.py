"""A missing SimpleITK must fail with a clear, actionable message, not a raw crash.

SimpleITK is a declared dependency but a lazy import. It used to be reached only by runs
that supplied a segmentation; once no-mask runs also resample to an isotropic grid, every
volumetric run needs it -- so an environment that never installed it (set up before the
3D work, or never `pip install -r requirements.txt`'d) suddenly fails mid-run. The guard
turns the opaque ModuleNotFoundError into a message that says what to install.
"""
import numpy as np
import pytest


def test_require_simpleitk_is_a_noop_when_installed():
    from analysis.volumetric.resample import require_simpleitk, sitk
    if sitk is None:
        pytest.skip("SimpleITK genuinely absent in this env")
    require_simpleitk()  # must not raise


def test_require_simpleitk_message_is_actionable(monkeypatch):
    import analysis.volumetric.resample as R
    monkeypatch.setattr(R, "sitk", None)
    monkeypatch.setattr(R, "_SIMPLEITK_IMPORT_ERROR",
                        ImportError("No module named 'SimpleITK'"), raising=False)
    with pytest.raises(ImportError) as excinfo:
        R.require_simpleitk()
    msg = str(excinfo.value)
    assert "pip install SimpleITK" in msg
    assert "requirements.txt" in msg
    assert "2D (xyt)" in msg, "should say 2D does not need it"
    # It must be honest that this is optional, not "required for every run":
    assert "Resample to Isotropic Voxels" in msg
    assert "untick" in msg.lower()


def test_every_resample_entry_point_guards(monkeypatch):
    """Each public path that touches sitk must raise the friendly error, not AttributeError
    on `sitk.sitkLinear` or similar."""
    import analysis.volumetric.resample as R
    monkeypatch.setattr(R, "sitk", None)
    monkeypatch.setattr(R, "_SIMPLEITK_IMPORT_ERROR", ImportError("x"), raising=False)

    imgs = {"t0": np.zeros((4, 5, 5), np.uint16)}
    spac = {"t0": (0.065, 0.065, 0.3)}
    with pytest.raises(ImportError, match="pip install SimpleITK"):
        R.resample_images_to_isotropic(imgs, spac)
    with pytest.raises(ImportError, match="pip install SimpleITK"):
        R.prepare_volume(imgs, spac, np.zeros((4, 5, 5), np.uint8), (0.065, 0.065, 0.065))
    with pytest.raises(ImportError, match="pip install SimpleITK"):
        R._resample_array_to_reference(
            np.zeros((4, 5, 5)), (1, 1, 1), (4, 5, 5), (1, 1, 1), None)


def test_a_no_mask_volumetric_run_reports_the_missing_dep_clearly(monkeypatch, tmp_path):
    import tifffile
    import analysis.volumetric.resample as R
    from core.config import BarcodeConfig
    from analysis.volumetric.run import run_volumetric_analysis

    monkeypatch.setattr(R, "sitk", None)
    monkeypatch.setattr(R, "_SIMPLEITK_IMPORT_ERROR", ImportError("x"), raising=False)

    vol = np.zeros((6, 20, 20), np.uint16)
    vol[2:4, 5:15, 5:15] = 800
    path = str(tmp_path / "Cell.tif")
    tifffile.imwrite(path, vol, imagej=True, metadata={"axes": "ZYX", "spacing": 0.3})

    config = BarcodeConfig()
    config.modules.image_binarization = True
    config.volumetric.analysis_mode = "xyzt"        # anisotropic -> triggers resampling
    with pytest.raises(ImportError, match="pip install SimpleITK"):
        run_volumetric_analysis(path, config)
