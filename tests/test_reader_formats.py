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


# --------------------------------------------------------------- ND2 reading

def _fake_nd2(monkeypatch, sizes, *, voxel=(0.1625, 0.1625, 0.3), experiment=()):
    """Stand in for nd2.ND2File so the ND2 path is testable without a fixture file.

    ND2 is a proprietary container that the `nd2` package can read but not write, so
    there is no way to build one in a test. The reader's own logic -- axis handling,
    channel selection, spacing, the time loop -- is what is under test here, and it sees
    the same interface either way.
    """
    import sys
    import types

    import numpy as np

    class _Voxel(tuple):
        x = property(lambda self: self[0])
        y = property(lambda self: self[1])
        z = property(lambda self: self[2])

    class _File:
        path = "fake.nd2"

        def __init__(self, *_a, **_k):
            self.sizes = dict(sizes)
            self.experiment = list(experiment)

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def asarray(self):
            shape = tuple(self.sizes.values())
            return np.arange(int(np.prod(shape)), dtype=np.uint16).reshape(shape)

        def voxel_size(self, channel=0):
            return _Voxel(voxel)

    module = types.ModuleType("nd2")
    module.ND2File = _File
    monkeypatch.setitem(sys.modules, "nd2", module)


def test_an_nd2_zstack_reads_as_one_timepoint(monkeypatch, tmp_path):
    from analysis.volumetric.reader import read_volume

    _fake_nd2(monkeypatch, {"Z": 6, "C": 2, "Y": 5, "X": 4})
    path = str(tmp_path / "stack.nd2")

    stack = read_volume(path, channel=1)
    assert stack.data.shape == (1, 6, 5, 4), "ZCYX must pad to (T, Z, Y, X)"
    assert stack.axes == "ZCYX"
    assert stack.z_step_um == 0.3 and stack.xy_step_um == 0.1625
    # No time loop means no timing, which must NOT be dressed up as a real interval --
    # that is the confusion resolve_frame_interval exists to prevent.
    assert stack.timing_from_file is False


def test_an_nd2_time_series_takes_its_interval_from_the_time_loop(monkeypatch, tmp_path):
    """An ND2 states the programmed period; a TIFF's finterval is often the z dwell."""
    from nd2 import structures  # real structures, only ND2File is faked

    from analysis.volumetric.reader import read_volume

    loop = structures.TimeLoop(
        count=3, nestingLevel=0, parameters=structures.TimeLoopParams(
            startMs=0.0, periodMs=30000.0, durationMs=90000.0,
            periodDiff=structures.PeriodDiff(avg=0.0, max=0.0, min=0.0)),
        type="TimeLoop")
    _fake_nd2(monkeypatch, {"T": 3, "Z": 4, "C": 1, "Y": 5, "X": 4}, experiment=[loop])

    stack = read_volume(str(tmp_path / "series.nd2"))
    assert stack.data.shape == (3, 4, 5, 4)
    assert stack.timing_from_file is True
    assert stack.exposure_time_s == 30.0
    assert "ND2" in stack.timing_source


def test_a_multi_position_nd2_is_refused_rather_than_pooled(monkeypatch, tmp_path):
    """Each P is a different field of view; collapsing them would pool unrelated cells."""
    from analysis.volumetric.reader import read_volume

    _fake_nd2(monkeypatch, {"T": 2, "P": 4, "C": 2, "Y": 5, "X": 4})
    with pytest.raises(ValueError, match=r"\['P'\]"):
        read_volume(str(tmp_path / "multipoint.nd2"))


def test_an_nd2_with_no_z_axis_is_sent_to_the_2d_pipeline(monkeypatch, tmp_path):
    from analysis.volumetric.reader import read_volume

    _fake_nd2(monkeypatch, {"T": 8, "C": 2, "Y": 5, "X": 4})
    with pytest.raises(ValueError, match="no Z axis"):
        read_volume(str(tmp_path / "planar.nd2"))
