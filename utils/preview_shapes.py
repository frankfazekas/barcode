"""Shape normalisation for the GUI's live previews.

Every preview tab indexes the sample array as ``frames[n, :, :, channel]``, so whatever
comes off disk has to be coerced to ``(frames, Y, X, channels)`` first. That coercion
used to live inline in ``gui/configuration_properties/preview_config.txt`` as a single
``min(file.shape)`` test, which mis-handled three real cases:

* a single-plane ``(Y, X)`` TIFF raised ``IndexError`` on ``file.shape[3]``;
* a ``(T, Z, Y, X)`` volumetric time-lapse had Z swapped into the channel axis, so the
  channel selector scrolled through z-slices and channel 0 was the empty top plane --
  a blank preview for exactly the 3D data the volumetric work uses;
* an RGB ``(Y, X, 3)`` TIFF was read as 3 frames of ``(X, 3)``.

This is preview-only. Nothing here is on the analysis path -- ``sample_preview`` is read
by the three process tabs and ``utils.gui``'s preview writers and by nothing else -- so
it cannot move a published metric.
"""
import numpy as np

# A trailing axis this small is a channel axis, not image data. Real microscopy channel
# counts here are 1-4; the smallest sensible Y/X is far larger.
MAX_CHANNELS = 4


def as_preview_stack(arr: np.ndarray) -> np.ndarray:
    """Coerce ``arr`` to ``(frames, Y, X, channels)``.

    The last two axes are always taken as Y and X. Anything in front of them is a
    progression axis (time, z, or both) and gets flattened into ``frames``; a small
    leading or trailing axis is read as channels.
    """
    arr = np.asarray(arr)

    if arr.ndim < 2:
        raise ValueError(f"preview needs at least a 2D image, got shape {arr.shape}")

    if arr.ndim == 2:                                       # (Y, X)
        return arr[np.newaxis, :, :, np.newaxis]

    if arr.ndim == 3:
        if arr.shape[-1] <= MAX_CHANNELS < arr.shape[-2]:   # (Y, X, C), e.g. RGB
            return arr[np.newaxis, ...]
        return arr[..., np.newaxis]                         # (N, Y, X) -- time or z

    # 4D and up: find the channel axis, flatten everything before Y/X into frames.
    if arr.shape[-1] <= MAX_CHANNELS:                       # (..., Y, X, C)
        return arr.reshape(-1, *arr.shape[-3:])

    if arr.ndim == 4 and arr.shape[1] <= MAX_CHANNELS:      # (T, C, Y, X)
        return np.moveaxis(arr, 1, -1)

    # No axis is plausibly channels, so every leading axis is a progression: (T, Z, Y, X)
    # collapses to T*Z scrubbable planes rather than pretending Z is a channel. The frame
    # slider then walks the whole volume series instead of one z-plane per "channel".
    return arr.reshape(-1, *arr.shape[-2:])[..., np.newaxis]


def nd2_as_preview_stack(ndfile) -> np.ndarray:
    """``as_preview_stack`` for an open ``nd2.ND2File``, using its declared axis names.

    ND2 files label their axes, so guessing is unnecessary and wrong. The previous code
    only handled files with a T axis, no Z axis, and fewer than 5 dimensions, and left
    ``file`` unassigned for anything else -- an ``UnboundLocalError`` on every volumetric
    ND2, and on any ND2 with 5 or fewer timepoints.
    """
    arr = ndfile.asarray()
    axes = list(ndfile.sizes)                  # e.g. ['T', 'Z', 'Y', 'X', 'C']

    # Move Y, X (in that order) and then C to the end; T/Z stay in front to be flattened.
    trailing = [a for a in ("Y", "X", "C") if a in axes]
    leading = [a for a in axes if a not in trailing]
    arr = np.transpose(arr, [axes.index(a) for a in leading + trailing])

    if "C" not in axes:
        arr = arr[..., np.newaxis]
    if arr.ndim == 3:                          # single plane, single channel
        arr = arr[np.newaxis, ...]
    return arr.reshape(-1, *arr.shape[-3:])    # collapse any T/Z into one frame axis
