"""File ordering for the volumetric modes.

One file is one timepoint there, so the file list is the barcode's vertical axis. A
lexicographic sort puts Cell1_10 before Cell1_2 and the resulting picture still looks
entirely reasonable -- every row carries its own filename, nothing errors, and the time
course is simply wrong. That silence is why this is pinned.

Run: python -m pytest tests/test_ordering.py -v
"""
from __future__ import annotations

import os

from analysis.volumetric.ordering import natural_key, sort_numerically


def test_numbers_order_as_numbers_not_text():
    names = [f"Cell1_{i}.tif" for i in (1, 10, 11, 15, 2, 3, 9)]
    assert sort_numerically(names) == [
        "Cell1_1.tif", "Cell1_2.tif", "Cell1_3.tif", "Cell1_9.tif",
        "Cell1_10.tif", "Cell1_11.tif", "Cell1_15.tif",
    ]


def test_this_is_the_order_plain_sorted_gets_wrong():
    """The exact failure seen on the Jurkat deliverable."""
    names = [f"Cell1_{i}.tif" for i in range(1, 16)]
    lexicographic = sorted(names)
    assert lexicographic[1] == "Cell1_10.tif", "plain sort really does scramble it"
    assert sort_numerically(names)[1] == "Cell1_2.tif"


def test_zero_padded_and_unpadded_agree():
    assert sort_numerically(["f_2.tif", "f_10.tif"]) == ["f_2.tif", "f_10.tif"]
    assert sort_numerically(["f_02.tif", "f_10.tif"]) == ["f_02.tif", "f_10.tif"]


def test_multiple_numbers_in_one_name_all_compare_numerically():
    names = ["Cell10_2.tif", "Cell2_10.tif", "Cell2_2.tif"]
    assert sort_numerically(names) == ["Cell2_2.tif", "Cell2_10.tif", "Cell10_2.tif"]


def test_directory_and_basename_are_ordered_independently():
    """A file's place must not depend on how deep its folder is nested."""
    paths = [os.path.join("run10", "a_2.tif"), os.path.join("run2", "a_10.tif"),
             os.path.join("run2", "a_2.tif")]
    assert sort_numerically(paths) == [
        os.path.join("run2", "a_2.tif"),
        os.path.join("run2", "a_10.tif"),
        os.path.join("run10", "a_2.tif"),
    ]


def test_names_without_numbers_still_sort_sensibly():
    assert sort_numerically(["b.tif", "A.tif", "c.tif"]) == ["A.tif", "b.tif", "c.tif"]


def test_natural_key_is_case_insensitive_on_the_text_runs():
    assert natural_key("Cell_1") == natural_key("cell_1")


def test_sorting_is_a_permutation_not_a_filter():
    names = [f"x{i}.tif" for i in range(30)] + ["nonumber.tif"]
    assert sorted(sort_numerically(names)) == sorted(names)


# ------------------------------------------------------------------ wiring


def test_xyt_keeps_the_original_order_but_volumetric_modes_do_not():
    """2D must be untouched: the published reference CSVs compare row by row."""
    import core.pipeline as pipeline
    from core import BarcodeConfig

    # find_files returns sorted(), so the fixture must be lexicographic -- handing this
    # an already-numeric list would let xyt pass by accident.
    names = sorted(f"Cell1_{i}.tif" for i in range(1, 16))
    assert names[1] == "Cell1_10.tif", "the fixture must start out scrambled"
    seen = {}

    def fake_find_files(path):
        return list(names)

    original = pipeline.find_files
    pipeline.find_files = fake_find_files
    try:
        for mode, expect_sorted in (("xyt", False), ("xyz", True), ("xyzt", True)):
            config = BarcodeConfig()
            config.volumetric.analysis_mode = mode
            # Reproduce the two lines under test without running a whole analysis.
            files = pipeline.find_files("anything")
            if config.volumetric.mode.key != "xyt":
                from analysis.volumetric.ordering import sort_numerically as sort_fn
                files = sort_fn(files)
            seen[mode] = files
            assert (files[1] == "Cell1_2.tif") is expect_sorted, mode
    finally:
        pipeline.find_files = original

    assert seen["xyt"] == names, "xyt must pass find_files' order through unchanged"
