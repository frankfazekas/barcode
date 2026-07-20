"""Locate, load and geometry-check segmentation masks.

A mask that is silently misaligned with its image produces confidently wrong metrics,
which is worse than no mask at all. Everything here therefore fails loudly: an
unresolvable template, an XY mismatch, or an implausible Z extent raises rather than
warning-and-continuing.

Pairing follows BARCODE's existing YAML-driven style — a regex that names capture
groups from the image filename, plus a path template that consumes them. For the
Jurkat nucleus dataset that is::

    segmentation_root     = .../prog_live_cells
    segmentation_regex    = Cell(?P<cell>\\d+)_(?P<frame>\\d+)
    segmentation_template = Cell{cell}/frame{frame}/nucleus/3D_seg/Cell_{cell}_SegMask_origFOV.tif

Note the ``_origFOV`` suffix: the sibling ``Cell_N_SegMask.tif`` is cropped in XY to
the cell and will not align with the original field, so it is rejected by the XY check.
"""
from __future__ import annotations

import os
import re
from typing import Optional, Tuple

import numpy as np
import tifffile

from core import VolumetricConfig

# The mask's Z extent should match the image's to within about one slice once voxel
# spacing is accounted for. Anything further off means the two are not the same view.
_Z_EXTENT_TOLERANCE_UM = 1.0


def _mask_spacing_um(config: VolumetricConfig, xy_step_um: float) -> float:
    """Masks carry no spacing metadata; 0 in config means 'isotropic at xy'."""
    if config.mask_spacing_um and config.mask_spacing_um > 0:
        return float(config.mask_spacing_um)
    return float(xy_step_um)


def resolve_segmentation_path(image_path: str, config: VolumetricConfig) -> Optional[str]:
    """Resolve the mask path for ``image_path``, or None if segmentation is off."""
    if not config.segmentation_enabled:
        return None

    stem = os.path.splitext(os.path.basename(image_path))[0]
    match = re.search(config.segmentation_regex, stem)
    if match is None:
        raise ValueError(
            f"Segmentation regex {config.segmentation_regex!r} does not match "
            f"image name {stem!r}; cannot locate a mask."
        )

    tokens = dict(match.groupdict())
    tokens.setdefault("stem", stem)
    try:
        relative = config.segmentation_template.format(**tokens)
    except KeyError as exc:
        raise ValueError(
            f"Segmentation template {config.segmentation_template!r} refers to "
            f"{exc} which the regex {config.segmentation_regex!r} does not capture. "
            f"Captured: {sorted(tokens)}."
        ) from exc

    root = config.segmentation_root or os.path.dirname(image_path)
    path = os.path.normpath(os.path.join(root, relative))
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No segmentation mask for {os.path.basename(image_path)} at {path} "
            f"(root={root!r}, template={config.segmentation_template!r})."
        )
    return path


def coerce_to_zyx(array: np.ndarray, label: str) -> np.ndarray:
    """Squeeze a loaded mask down to (Z, Y, X).

    Masks are frequently saved as RGB(A) with the colour channels identical; collapse
    those rather than failing, mirroring ``resample._ensure_3d_volume``.
    """
    array = np.asarray(array)
    if array.ndim == 3:
        return array
    squeezed = np.squeeze(array)
    if squeezed.ndim == 3:
        return squeezed
    if array.ndim == 4 and array.shape[-1] in (3, 4):
        rgb = array[..., :3]
        if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2]):
            return rgb[..., 0]
    raise ValueError(f"{label}: mask is not 3-D, shape {tuple(array.shape)}.")


def load_segmentation(
    image_path: str,
    image_shape_zyx: Tuple[int, int, int],
    z_step_um: float,
    xy_step_um: float,
    config: VolumetricConfig,
) -> Optional[Tuple[np.ndarray, str, float]]:
    """Load and validate the mask for ``image_path``.

    Returns ``(mask_bool_zyx, mask_path, mask_spacing_um)``, or None when
    segmentation is disabled. Raises if a mask is expected but unusable.
    """
    path = resolve_segmentation_path(image_path, config)
    if path is None:
        return None

    mask = coerce_to_zyx(tifffile.imread(path), os.path.basename(path))
    mask_spacing = _mask_spacing_um(config, xy_step_um)

    img_z, img_y, img_x = image_shape_zyx
    mask_z, mask_y, mask_x = mask.shape

    if (mask_y, mask_x) != (img_y, img_x):
        raise ValueError(
            f"Segmentation {os.path.basename(path)} has XY {mask_y}x{mask_x} but the "
            f"image has {img_y}x{img_x}. These are different fields of view — if this "
            f"is a cell-cropped mask, point the template at the original-FOV export "
            f"instead."
        )

    image_extent = img_z * z_step_um
    mask_extent = mask_z * mask_spacing
    if abs(image_extent - mask_extent) > _Z_EXTENT_TOLERANCE_UM:
        raise ValueError(
            f"Segmentation {os.path.basename(path)} spans {mask_extent:.3f} um in z "
            f"({mask_z} slices @ {mask_spacing:g} um) but the image spans "
            f"{image_extent:.3f} um ({img_z} @ {z_step_um:g} um). Check "
            f"mask_spacing_um and z_step_um."
        )

    mask = mask > 0
    if not mask.any():
        raise ValueError(f"Segmentation {os.path.basename(path)} is empty.")
    if config.invert_binarization:
        mask = ~mask

    return mask, path, mask_spacing
