"""The volumetric reader is TIFF-only, and should say so rather than fail obscurely.

``utils.setup.find_files`` also accepts .nd2/.mp4/.avi, so those files reach the
volumetric modes and used to die inside tifffile with "not a TIFF file" -- accurate but
silent about the fact that xyz/xyzt never supported them.
"""
import pytest

from analysis.volumetric.reader import require_tiff


@pytest.mark.parametrize("name", ["Cell1_1.tif", "Cell1_1.tiff", "Cell1_1.TIF"])
def test_tiffs_are_accepted(name):
    require_tiff(name)  # must not raise


def test_nd2_is_rejected_and_says_where_nd2_does_work():
    with pytest.raises(ValueError) as excinfo:
        require_tiff(r"F:\data\Cell1_1.nd2")
    message = str(excinfo.value)
    assert "TIFF only" in message
    assert "Cell1_1.nd2" in message
    # The point of the message: name the mode that *does* read ND2, and the way out.
    assert "xyt" in message and "TIFF" in message


@pytest.mark.parametrize("name", ["movie.avi", "movie.mp4", "noextension"])
def test_other_discoverable_formats_are_rejected(name):
    with pytest.raises(ValueError):
        require_tiff(name)
