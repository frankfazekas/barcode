"""The volumetric reader accepts what it can actually read, and says so otherwise.

``utils.setup.find_files`` also accepts .mp4/.avi, so those files reach the volumetric
modes and used to die inside tifffile with "not a TIFF file" -- accurate but silent about
which formats the volumetric path supports.

ND2 was TIFF-only when these tests were first written; it is now read directly by
``analysis.volumetric.reader._read_nd2``, so the rejection test became a test of removed
behaviour and is inverted here rather than deleted -- ND2 being accepted is the thing
worth pinning now.
"""
import pytest

from analysis.volumetric.reader import (
    SUPPORTED_SUFFIXES,
    require_supported,
    require_tiff,
)


@pytest.mark.parametrize("name", ["Cell1_1.tif", "Cell1_1.tiff", "Cell1_1.TIF"])
def test_tiffs_are_accepted(name):
    require_supported(name)  # must not raise


@pytest.mark.parametrize("name", [r"F:\data\Cell1_1.nd2", "Cell1_1.ND2"])
def test_nd2_is_accepted_now_that_the_reader_handles_it(name):
    require_supported(name)


def test_nd2_is_declared_supported():
    assert ".nd2" in SUPPORTED_SUFFIXES


@pytest.mark.parametrize("name", ["movie.avi", "movie.mp4", "noextension"])
def test_formats_the_reader_cannot_open_are_rejected(name):
    with pytest.raises(ValueError) as excinfo:
        require_supported(name)
    message = str(excinfo.value)
    assert "TIFF and ND2" in message
    # The message has to name the file and the way out, not just the failure.
    assert name.split("\\")[-1] in message
    assert "Tiff" in message


def test_require_tiff_alias_still_works_for_older_callers():
    require_tiff("Cell1_1.tif")
    with pytest.raises(ValueError):
        require_tiff("movie.avi")
