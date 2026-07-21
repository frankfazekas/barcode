"""Geometry canonicalization + isotropic resampling.

PORTED VERBATIM from ``chromatin-analysis`` at
``src/chromatin_analysis/resample.py`` — do not diverge from it without porting the
change back. That module is in turn adapted from nuclear-rotation-3D
(``nuclear_rotation/registration/preprocess.py``); the provenance chain is kept
intact here deliberately.

Two call-site gotchas worth repeating: spacing arguments are ``(x, y, z)`` while the
arrays are ``(Z, Y, X)``; and the upstream version cropped to the mask bounding box
unconditionally, which BARCODE must not do. ``prepare_volume`` therefore takes
``crop_to_mask`` and defaults it to False, because every "fraction of field of view"
metric is a fraction of whatever box is analysed. This function is also not
nucleus-specific despite its provenance -- it prepares any labelled volume.

Original module docstring follows.
---
Adapted from nuclear-rotation-3D (``nuclear_rotation/registration/preprocess.py``).
The raw fluorescence channels are anisotropic (xy ~0.065 um, z ~0.3 um) while the
segmentation mask is isotropic (~0.065 um in all axes). We bring every channel
onto the mask's isotropic grid (linear interpolation for intensity, nearest for
the mask), then crop to the nuclear bounding box -- exactly what the rotation
pipeline does for an image/mask pair, generalized here to one mask + N channels.

SimpleITK resamples by physical coordinates (identity transform, origin 0), so
channels and mask must share an origin; the per-cell crops here do.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import SimpleITK as sitk


def _ensure_3d_volume(array, label, allow_redundant_channels=False):
    """Coerce a loaded array to (Z, Y, X).

    Copied from nuclear-rotation-3D (``preprocess._ensure_3d_volume``): squeeze
    singleton axes, and collapse an RGB(A) channel axis to a single channel when
    all colour channels are equal -- the segmentation masks here are saved RGBA
    but are grayscale-identical per channel (R == G == B, values 0/255).
    """
    array = np.asarray(array)
    if array.ndim == 3:
        return array
    squeezed = np.squeeze(array)
    if squeezed.ndim == 3:
        return squeezed
    if allow_redundant_channels and array.ndim == 4 and array.shape[-1] in (3, 4):
        rgb = array[..., :3]
        if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2]):
            return rgb[..., 0]
    raise RuntimeError(f"{label}_not_3d_shape_{tuple(int(v) for v in array.shape)}")


def _to_sitk(array, spacing_xyz_um, origin_xyz_um=None):
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(tuple(float(v) for v in spacing_xyz_um))
    if origin_xyz_um is not None:
        image.SetOrigin(tuple(float(v) for v in origin_xyz_um))
    return image


def _resample_array_to_reference(
    array,
    source_spacing_xyz_um,
    reference_shape_zyx,
    reference_spacing_xyz_um,
    interpolator,
    reference_origin_xyz_um=None,
):
    """Resample ``array`` onto a reference grid, by physical coordinates.

    ``reference_origin_xyz_um`` names where that grid starts. It defaults to the volume
    origin, which is right whenever the reference spans the whole field -- but not when
    the reference is a *crop*. ``run._prepare_geometry`` resamples the per-frame masks
    onto ``union_iso.shape``; with ``crop_to_mask`` on that shape is the cropped box, so
    without an origin the masks were sampled from (0,0,0) and came out offset from the
    image by the whole crop. The shape check downstream cannot see a pure offset.
    """
    source = _to_sitk(array, source_spacing_xyz_um)
    reference = _to_sitk(
        np.zeros(reference_shape_zyx, dtype=np.float32), reference_spacing_xyz_um,
        reference_origin_xyz_um,
    )
    transform = sitk.Transform(3, sitk.sitkIdentity)

    # Clamp to the edge value outside the source extent rather than filling with the
    # default 0. The two grids disagree about their last sample by design: a mask staged
    # by `scripts/_staging.mask_z_to_isotropic` has round(n * z/xy) planes, spanning
    # n*z um, while an image of n planes at spacing z is only interpolatable over
    # (n-1)*z um (the convention `_reference_shape_for_spacing` below uses). The mask grid
    # therefore overhangs the image by about one acquired slice, and zero-fill wrote that
    # overhang into the data as real black voxels -- 2 of 249 planes on 54x0.3/0.065
    # Jurkat volumes, 8% of the depth on a 5-slice stack -- while the mask still declared
    # foreground there. Those planes fed the mean-relative threshold, the intensity
    # histogram and every "fraction of analysed volume" denominator.
    #
    # Nearest-neighbour extrapolation repeats the last acquired plane instead, which is
    # the honest reading of a sub-slice overhang and is also what fixes masks already
    # staged on disk -- changing the staging formula could not. Interior voxels are
    # untouched: extrapolation only applies outside the source extent.
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetOutputPixelType(source.GetPixelID())
    resampler.SetUseNearestNeighborExtrapolator(True)
    return sitk.GetArrayFromImage(resampler.Execute(source))


def _canonicalize_geometry(image, image_spacing_xyz_um, mask, mask_spacing_xyz_um):
    """Resample ``image`` onto the mask grid if their shape/spacing differ."""
    if image.shape == mask.shape and tuple(image_spacing_xyz_um) == tuple(
        mask_spacing_xyz_um
    ):
        return image, "none"
    canonical = _resample_array_to_reference(
        image, image_spacing_xyz_um, mask.shape, mask_spacing_xyz_um, sitk.sitkLinear
    )
    return canonical, "resampled_to_mask_geometry"


def _isotropic_xy_spacing_xyz(spacing_xyz_um):
    xy = float(spacing_xyz_um[0])
    return (xy, xy, xy)


def _reference_shape_for_spacing(
    source_shape_zyx, source_spacing_xyz_um, target_spacing_xyz_um
):
    size_xyz = np.asarray(source_shape_zyx[::-1], dtype=np.float64)
    src = np.asarray(source_spacing_xyz_um, dtype=np.float64)
    tgt = np.asarray(target_spacing_xyz_um, dtype=np.float64)
    physical = np.maximum(size_xyz - 1.0, 0.0) * np.maximum(src, 1e-6)
    target_size = np.maximum(
        1, np.floor(physical / np.maximum(tgt, 1e-6) + 1.0).astype(np.int64)
    )
    return tuple(int(v) for v in target_size[::-1])


def _crop_to_mask_bbox(images: Dict[str, np.ndarray], mask, padding: int):
    coords = np.where(mask > 0)
    if coords[0].size == 0:
        return images, mask, None
    zc, yc, xc = coords
    z0 = max(0, int(zc.min()) - padding)
    z1 = min(mask.shape[0], int(zc.max()) + 1 + padding)
    y0 = max(0, int(yc.min()) - padding)
    y1 = min(mask.shape[1], int(yc.max()) + 1 + padding)
    x0 = max(0, int(xc.min()) - padding)
    x1 = min(mask.shape[2], int(xc.max()) + 1 + padding)
    bbox = {"z": [z0, z1], "y": [y0, y1], "x": [x0, x1]}
    cropped = {k: v[z0:z1, y0:y1, x0:x1] for k, v in images.items()}
    return cropped, mask[z0:z1, y0:y1, x0:x1], bbox


def prepare_volume(
    images: Dict[str, np.ndarray],
    image_spacings: Dict[str, Tuple[float, float, float]],
    mask: np.ndarray,
    mask_spacing: Tuple[float, float, float],
    crop_padding: int = 2,
    crop_to_mask: bool = False,
):
    """Put every channel + the mask on one isotropic grid.

    Returns ``(images_iso, mask_iso, spacing_iso, info)`` where ``images_iso`` is
    a dict of (Z, Y, X) arrays (same dtype as input), ``mask_iso`` is uint8 0/1,
    ``spacing_iso`` is the isotropic (x, y, z) spacing, and ``info`` records the
    actions taken (for run provenance).

    ``crop_to_mask`` crops to the mask's bounding box. It defaults to **False**: the
    analysed field must be the acquired field of view, because every "fraction of field
    of view" metric is a fraction OF THAT BOX. Cropping per file gives each file its own
    denominator, so the same object in a tighter box reports a larger fraction and a
    size trend cannot be distinguished from the box tracking the object. The only
    intended way to analyse less than the full field is an explicit z range.

    The upstream ``chromatin-analysis`` version cropped unconditionally; that suited a
    single-nucleus rotation pipeline where every quantity was absolute. It is wrong for
    BARCODE, whose normalised metrics are all relative to the analysed volume.
    """
    info: Dict[str, object] = {}

    # Coerce loaded arrays to (Z, Y, X) -- the masks are saved RGBA.
    mask = _ensure_3d_volume(np.asarray(mask), "mask", allow_redundant_channels=True)
    images = {name: _ensure_3d_volume(np.asarray(img), name) for name, img in images.items()}

    # 1) Canonicalize each channel onto the mask's grid (upsamples z to isotropic).
    canon: Dict[str, np.ndarray] = {}
    for name, img in images.items():
        c, action = _canonicalize_geometry(
            np.asarray(img), image_spacings[name], np.asarray(mask), mask_spacing
        )
        canon[name] = c
        info[f"{name}_canon"] = action
    working_spacing = tuple(float(v) for v in mask_spacing)
    work_mask = np.asarray(mask)

    # 2) Resample to isotropic-xy spacing (no-op when the mask is already isotropic).
    target = _isotropic_xy_spacing_xyz(working_spacing)
    if not np.allclose(working_spacing, target, atol=1e-9):
        ref = _reference_shape_for_spacing(work_mask.shape, working_spacing, target)
        canon = {
            k: _resample_array_to_reference(
                v, working_spacing, ref, target, sitk.sitkLinear
            )
            for k, v in canon.items()
        }
        work_mask = _resample_array_to_reference(
            work_mask.astype(np.uint8), working_spacing, ref, target, sitk.sitkNearestNeighbor
        )
        working_spacing = target
        info["isotropic"] = "resampled"
    else:
        info["isotropic"] = "already_isotropic"
    work_mask = (work_mask > 0).astype(np.uint8)

    # 3) Optionally crop to the mask's bounding box. Off by default -- see the docstring;
    # the reported bbox then spans the whole field so callers index identically either way.
    if crop_to_mask:
        canon, work_mask, bbox = _crop_to_mask_bbox(canon, work_mask, crop_padding)
        info["cropped"] = "mask_bbox"
    else:
        bbox = {"z": [0, int(work_mask.shape[0])],
                "y": [0, int(work_mask.shape[1])],
                "x": [0, int(work_mask.shape[2])]}
        info["cropped"] = "full_field"
    info["crop_bbox"] = bbox
    canon = {k: np.ascontiguousarray(v) for k, v in canon.items()}
    return canon, np.ascontiguousarray(work_mask), working_spacing, info
