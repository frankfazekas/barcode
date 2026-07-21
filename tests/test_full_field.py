"""The analysed volume is the acquired field of view, not the mask's bounding box.

Every "fraction of volume" metric is a fraction OF THE ANALYSED BOX. Cropping to the
mask meant each file got its own denominator: on the real Jurkat series all 15 masks
have different bounding boxes (z extent swinging 177 -> 124 -> 132), so an object
shrinking and the crop box tightening around it were indistinguishable, and a column
of the barcode was not a time course at all.

Only an explicit z range may reduce the analysed field.

Run: python -m pytest tests/test_full_field.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from analysis.volumetric.resample import prepare_volume


def volume_and_mask(shape=(20, 40, 40), box=(slice(8, 12), slice(18, 22), slice(18, 22))):
    """A small object sitting well inside a much larger field."""
    image = np.full(shape, 100, np.uint16)
    image[box] = 900
    mask = np.zeros(shape, np.uint8)
    mask[box] = 1
    return image, mask


def run(crop, spacing=(1.0, 1.0, 1.0)):
    image, mask = volume_and_mask()
    return prepare_volume(
        images={"image": image}, image_spacings={"image": spacing},
        mask=mask, mask_spacing=spacing, crop_padding=2, crop_to_mask=crop,
    )


def test_the_default_keeps_the_whole_field():
    images, mask, _, info = run(crop=False)
    assert images["image"].shape == (20, 40, 40), "the acquired field, untouched"
    assert mask.shape == (20, 40, 40)
    assert info["cropped"] == "full_field"


def test_cropping_is_still_available_and_really_crops():
    images, mask, _, info = run(crop=True)
    assert images["image"].shape < (20, 40, 40)
    assert info["cropped"] == "mask_bbox"


def test_the_reported_bbox_spans_the_field_when_not_cropping():
    """Callers slice the per-frame masks with this box, so it must stay valid."""
    _, mask, _, info = run(crop=False)
    bbox = info["crop_bbox"]
    assert bbox == {"z": [0, 20], "y": [0, 40], "x": [0, 40]}
    z, y, x = bbox["z"], bbox["y"], bbox["x"]
    assert mask[z[0]:z[1], y[0]:y[1], x[0]:x[1]].shape == mask.shape


def test_the_denominator_no_longer_depends_on_the_object():
    """The defect, stated directly: two different objects in one field must share a box.

    Cropped, the same-sized object reports a LARGER fraction when its box is tighter.
    """
    fractions, shapes = [], []
    for box in ((slice(8, 12), slice(18, 22), slice(18, 22)),
                (slice(4, 16), slice(10, 30), slice(10, 30))):
        image = np.full((20, 40, 40), 100, np.uint16)
        mask = np.zeros((20, 40, 40), np.uint8)
        image[box] = 900
        mask[box] = 1
        small = np.zeros((20, 40, 40), np.uint8)
        small[8:12, 18:22, 18:22] = 1          # the SAME object in both cases
        images, mask_iso, _, _ = prepare_volume(
            images={"image": small}, image_spacings={"image": (1.0, 1.0, 1.0)},
            mask=mask, mask_spacing=(1.0, 1.0, 1.0), crop_to_mask=False,
        )
        shapes.append(images["image"].shape)
        fractions.append(float(images["image"].sum()) / images["image"].size)

    assert shapes[0] == shapes[1], "same field -> same denominator"
    assert fractions[0] == pytest.approx(fractions[1]), \
        "the same object must report the same fraction regardless of mask extent"


def test_config_default_is_full_field():
    from core import BarcodeConfig
    assert BarcodeConfig().volumetric.crop_to_mask is False
