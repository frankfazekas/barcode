"""Shared helpers for staging third-party datasets into BARCODE's conventions.

Open datasets arrive in whatever the publisher's tooling emitted. BARCODE's volumetric
reader deliberately refuses to guess (``analysis/volumetric/reader.py``), so rather than
loosening the reader or overriding it at every call site, staging rewrites the data into
files that state their own geometry honestly. These are the three operations that every
such stager needs.

Used by ``fetch_ctc.py`` (Cell Tracking Challenge) and ``stage_allen_fov.py``
(Allen Institute FOVs).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import tifffile


def read_tiff_any(path: str) -> np.ndarray:
    """Read a TIFF, including LZW-compressed ones, on the pinned environment.

    ``tifffile`` can only decode LZW through ``imagecodecs``, which is not in
    requirements.txt and which pip cannot install without dragging numpy past the
    ``numpy==2.0.1`` pin -- breaking scipy 1.14. Pillow is already present (via
    scikit-image) and decodes LZW natively.

    The page order is recovered from ``tifffile``'s own series description rather than
    assumed, because Pillow only ever hands back a flat page sequence: a 65x7 ZCYX stack
    and a 7x65 CZYX one both arrive as 455 pages, and guessing between them silently
    transposes the data.
    """
    try:
        return np.asarray(tifffile.imread(path))
    except ValueError as error:
        if "imagecodecs" not in str(error):
            raise

    with tifffile.TiffFile(path) as handle:
        series = handle.series[0]
        shape = tuple(int(v) for v in series.shape)

    from PIL import Image, ImageSequence

    with Image.open(path) as image:
        pages = [np.asarray(page) for page in ImageSequence.Iterator(image)]
    if not pages:
        raise ValueError(f"{path}: no readable pages")

    stacked = np.stack(pages) if len(pages) > 1 else pages[0]
    if stacked.shape != shape and stacked.size == int(np.prod(shape)):
        stacked = stacked.reshape(shape)
    return stacked


def read_tiff_bytes(payload: bytes, name: str = "<memory>") -> np.ndarray:
    """Decode a TIFF held in memory, LZW included.

    Ground-truth tracking volumes are read straight out of the cached CTC zips rather
    than extracted: a dataset's TRA folder is one volume per frame, and unpacking tens
    of gigabytes to read centroids would cost more disk than the analysis itself.
    """
    import io

    buffer = io.BytesIO(payload)
    try:
        return np.asarray(tifffile.imread(buffer))
    except ValueError as error:
        if "imagecodecs" not in str(error):
            raise

    buffer.seek(0)
    with tifffile.TiffFile(buffer) as handle:
        shape = tuple(int(v) for v in handle.series[0].shape)

    from PIL import Image, ImageSequence

    buffer.seek(0)
    with Image.open(buffer) as image:
        pages = [np.asarray(page) for page in ImageSequence.Iterator(image)]
    if not pages:
        raise ValueError(f"{name}: no readable pages")
    stacked = np.stack(pages) if len(pages) > 1 else pages[0]
    if stacked.shape != shape and stacked.size == int(np.prod(shape)):
        stacked = stacked.reshape(shape)
    return stacked


def write_volume(path: str, volume: np.ndarray, xy_um: float, z_um: float,
                 dt_s: Optional[float] = None) -> None:
    """Write a ZYX volume that declares its own axis order and voxel spacing.

    This is the point of staging. The reader trusts the file and refuses to infer, so
    the staged file is made worth trusting -- no ``--axes`` / ``--xy-step`` / ``--z-step``
    override needed downstream, and the data sits on the same footing as the Jurkat
    stacks the volumetric branch was developed against.
    """
    metadata = {"axes": "ZYX", "spacing": z_um, "unit": "um"}
    if dt_s is not None:
        metadata["finterval"] = dt_s
    tifffile.imwrite(
        path, volume, imagej=True,
        resolution=(1.0 / xy_um, 1.0 / xy_um), metadata=metadata,
    )


def mask_z_to_isotropic(mask: np.ndarray, z_um: float, xy_um: float) -> np.ndarray:
    """Resample a mask's Z onto the isotropic grid at the xy step.

    Masks that ship on the image's own ANISOTROPIC grid cannot be described to BARCODE:
    ``mask_spacing_um`` is a single scalar meaning "the mask is isotropic at this
    spacing", and ``prepare_volume`` resamples the *image* onto whatever grid it names.
    Setting it to the z step therefore does not describe the mask -- it resamples the
    whole dataset to z-sized cubes and discards the xy resolution (0.202 um -> 1.0 um on
    CTC's CHO, a 5x loss that still produces entirely plausible numbers).

    Converting the mask here instead puts it on the same footing as the Jurkat masks --
    isotropic at the xy step, which is exactly what ``mask_spacing_um: 0`` means -- so
    the default describes the file correctly and nothing is overridden.

    Nearest-neighbour index mapping, never interpolation: these carry instance labels,
    and averaging label 7 with label 8 invents an object that was never segmented.
    """
    if z_um <= 0 or xy_um <= 0:
        raise ValueError(f"spacing must be positive, got z={z_um} xy={xy_um}")
    # Node-aligned, matching `resample._reference_shape_for_spacing`: a stack of n planes
    # spans (n-1) steps, not n. This used to be `round(n * z/xy)`, which is larger by
    # about (z/xy - 1) planes while `linspace` below still spans the same source range --
    # so the staged mask was declared isotropic at xy_um while physically standing
    # (z/xy - 1)/n taller. Small at low anisotropy, but this helper exists for exactly the
    # datasets where that is not the case: a 10-plane stack at 11x came out 110 planes
    # instead of 100, i.e. 10% too tall in z, carrying straight into mesh volume, height
    # and sphericity.
    slices = max(1, int(np.floor((mask.shape[0] - 1) * (z_um / xy_um) + 1)))
    index = np.clip(
        np.round(np.linspace(0, mask.shape[0] - 1, slices)).astype(int),
        0, mask.shape[0] - 1,
    )
    return mask[index]
