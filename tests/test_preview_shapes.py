"""The GUI preview stack must always come out as (frames, Y, X, channels).

The three process tabs index it as ``frames[n, :, :, channel]`` with no shape checks, so
anything that is not 4D crashes the preview and anything with the wrong axis order shows
the wrong picture. The cases below are the ones the previous inline
``min(file.shape)`` heuristic got wrong.
"""
import numpy as np
import pytest

from utils.preview_shapes import as_preview_stack

Y, X = 312, 303


def test_single_plane_image_becomes_one_frame():
    # Used to raise IndexError on file.shape[3]: expand_dims only fired for 3D input,
    # so a flat (Y, X) TIFF reached the shape[3] test still 2D.
    assert as_preview_stack(np.zeros((Y, X))).shape == (1, Y, X, 1)


@pytest.mark.parametrize("n", [3, 15, 54])
def test_a_plain_stack_scrubs_along_its_leading_axis(n):
    """(T, Y, X) and (Z, Y, X) are indistinguishable and both scrub the same way."""
    assert as_preview_stack(np.zeros((n, Y, X))).shape == (n, Y, X, 1)


def test_rgb_is_one_frame_with_three_channels_not_three_frames():
    assert as_preview_stack(np.zeros((Y, X, 3))).shape == (1, Y, X, 3)


@pytest.mark.parametrize("c", [1, 2, 3, 4])
def test_trailing_channel_axis_is_left_alone(c):
    assert as_preview_stack(np.zeros((15, Y, X, c))).shape == (15, Y, X, c)


def test_channel_first_volumes_move_channels_last():
    assert as_preview_stack(np.zeros((15, 2, Y, X))).shape == (15, Y, X, 2)


def test_volumetric_timelapse_flattens_time_and_z_into_frames():
    """The regression that made 3D previews look broken.

    (T, Z, Y, X) has no small axis, so the old code's swap dropped Z into the channel
    slot: the channel selector scrolled z-planes and channel 0 was the empty top of the
    stack -- a blank preview. Every plane should be reachable from the frame slider.
    """
    out = as_preview_stack(np.zeros((15, 54, Y, X)))
    assert out.shape == (15 * 54, Y, X, 1)


def test_frames_keep_their_pixels_when_flattened():
    """Flattening must not transpose Y and X or interleave the planes.

    Z is 7 here, not 3: an axis of 3 or fewer is a legitimate channel count, so a toy
    (2, 3, 4, 5) array is genuinely ambiguous and takes the (T, C, Y, X) branch instead.
    """
    arr = np.arange(2 * 7 * 4 * 5).reshape(2, 7, 4, 5)
    out = as_preview_stack(arr)
    assert out.shape == (14, 4, 5, 1)
    np.testing.assert_array_equal(out[0, :, :, 0], arr[0, 0])
    np.testing.assert_array_equal(out[7, :, :, 0], arr[1, 0])


def test_a_1d_array_is_rejected_rather_than_reshaped_into_nonsense():
    with pytest.raises(ValueError):
        as_preview_stack(np.zeros(10))
