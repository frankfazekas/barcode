"""Reading segmentation masks, whatever the segmenter chose to write.

BARCODE's mask pairing (``segmentation.py``) answers *which file*; this answers *how to
read it*. They are separated because the set of formats is open-ended -- Cellpose writes
``_seg.npy``, other tools write TIFF, npz, or a labelled PNG -- while the pairing rules
are stable.

Adding a format is one ``@mask_loader`` registration and nothing else: the error
messages, the extension list in the config help, and the "unsupported format" text all
read from the registry, so they cannot fall out of step with what is actually supported.

**Labels are preserved.** Every loader returns the array as stored, without casting to
bool. An instance segmentation carries one integer per object, and for a confluent
tissue that distinction is the entire measurement -- collapsing it to a boolean makes
every touching cell one connected component. Deciding whether to *use* the labels is
``segmentation.load_segmentation``'s job (see ``object_partition``); this layer's job is
not to lose them first.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, Tuple

import numpy as np

# extension (with dot, lowercase) -> loader
_LOADERS: Dict[str, Callable[[str], np.ndarray]] = {}


def mask_loader(*extensions: str):
    """Register a loader for one or more file extensions."""
    def register(function: Callable[[str], np.ndarray]):
        for extension in extensions:
            _LOADERS[extension.lower()] = function
        return function
    return register


def supported_mask_extensions() -> Tuple[str, ...]:
    return tuple(sorted(_LOADERS))


@mask_loader(".tif", ".tiff")
def _load_tiff(path: str) -> np.ndarray:
    import tifffile

    return tifffile.imread(path)


@mask_loader(".png", ".bmp")
def _load_image(path: str) -> np.ndarray:
    import imageio.v3 as iio

    return iio.imread(path)


def _masks_from_mapping(mapping, path: str, keys) -> np.ndarray:
    """Pull the label array out of a segmenter's bundled result."""
    for key in ("masks", "mask", "labels"):
        if key in keys:
            return np.asarray(mapping[key])
    raise ValueError(
        f"{os.path.basename(path)} holds {sorted(keys)!r} but none of "
        f"'masks'/'mask'/'labels'. Save the label array itself, or extract it first."
    )


@mask_loader(".npy")
def _load_npy(path: str) -> np.ndarray:
    """A bare label array, or a Cellpose ``_seg.npy`` result bundle.

    Cellpose ``np.save``s a *dict*, which numpy returns as a 0-d object array that only
    unpickles with ``allow_pickle``. That is a real caveat -- unpickling executes code,
    so this must only ever be pointed at masks the user produced themselves -- and it is
    why the plain-array path is tried first and pickling is enabled only when numpy says
    the file needs it.
    """
    try:
        array = np.load(path, allow_pickle=False)
    except ValueError:
        array = np.load(path, allow_pickle=True)

    if array.dtype == object:
        payload = array.item() if array.ndim == 0 else array
        if isinstance(payload, dict):
            return _masks_from_mapping(payload, path, payload.keys())
        raise ValueError(
            f"{os.path.basename(path)} contains a {type(payload).__name__}, not a "
            f"label array or a dict holding one."
        )
    return array


@mask_loader(".npz")
def _load_npz(path: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as bundle:
        names = list(bundle.files)
        if len(names) == 1:
            return np.asarray(bundle[names[0]])
        return _masks_from_mapping(bundle, path, names)


def load_mask_array(path: str) -> np.ndarray:
    """Read a mask file as an array, labels intact.

    Raises with the registry's own list of formats rather than a hard-coded one, so a
    newly registered loader is advertised automatically.
    """
    extension = os.path.splitext(path)[1].lower()
    loader = _LOADERS.get(extension)
    if loader is None:
        described = repr(extension) if extension else "a file with no extension"
        supported = ", ".join(supported_mask_extensions())
        raise ValueError(
            f"{os.path.basename(path)}: no mask loader for {described}. "
            f"Supported: {supported}."
        )
    array = loader(path)
    if array is None:
        raise ValueError(f"{os.path.basename(path)}: loader returned nothing.")
    return np.asarray(array)
