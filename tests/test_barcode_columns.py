"""The barcode must show exactly the columns the CSV carries.

This has now failed three times in this codebase, each time silently: the CSV gains a
family's columns and the picture does not. The latest was the worst kind -- a
``metrics_to_visualize`` mask built from the mode alone was 37 long against 53 metrics,
and ``itertools.compress`` stops at the shorter sequence, so 16 columns vanished from
the barcode with no error, no warning, and a perfectly plausible-looking image.

Run: python -m pytest tests/test_barcode_columns.py -v
"""
from __future__ import annotations

import pytest

from core.metrics import selection_mask
from core.results import OPTIONAL_FAMILIES, ChannelResults


def populated_xyzt_results(n=3):
    """Results carrying the optional families a real masked xyzt run produces."""
    from core.results import (
        ComponentResults, CurvatureRangeResults, FlowResults, MaskIntensityResults,
        MeshResults, SliceProfileResults,
    )

    out = []
    for i in range(n):
        r = ChannelResults(filepath=f"Cell1_{i + 1}.tif", channel=0)
        # A flow-bearing run: these tests are about family detection and mask length, so
        # flow must be populated or it is (correctly) dropped as a static z-stack and the
        # column counts below shift by seven. Flow suppression is exercised separately.
        r.flow = FlowResults(mean_speed=1.0, delta_speed=0.0, mean_theta=0.1,
                             mean_sigma_theta=0.2, velocity_correlation_length=3.0,
                             divergence=0.0, curl=0.1)
        r.mesh = MeshResults(mesh_volume=500.0 + i)
        r.components = ComponentResults(count=1.0)
        r.curvature_range = CurvatureRangeResults(min_curvature=-0.05, max_curvature=0.4)
        r.slice_profile = SliceProfileResults(max_area_index=80.0)
        r.mask_intensity = MaskIntensityResults(mfi=350.0)
        out.append(r)
    return out


def switches_for(results):
    return {f.switch: any(getattr(r, f.attribute, None) is not None
                          and getattr(r, f.attribute).is_populated() for r in results)
            for f in OPTIONAL_FAMILIES}


def test_the_families_are_detected_as_populated():
    switches = switches_for(populated_xyzt_results())
    for name in ("include_mesh", "include_components", "include_curvature_range",
                 "include_slice_profile", "include_mask_intensity"):
        assert switches[name], name


def test_a_mask_built_without_the_family_switches_is_the_wrong_length():
    """The exact defect: a base mask against the fuller family-aware header set."""
    results = populated_xyzt_results()
    switches = switches_for(results)

    without = ChannelResults.get_headers(just_metrics=True, mode="xyzt")
    with_families = ChannelResults.get_headers(just_metrics=True, mode="xyzt", **switches)

    assert len(without) == 36
    assert len(with_families) == 52
    assert len(without) != len(with_families), "the mismatch compress silently swallowed"


def test_the_renderer_refuses_a_mismatched_mask(tmp_path):
    """Refusing beats truncating: a short mask used to just drop trailing columns."""
    import matplotlib
    matplotlib.use("Agg")
    from visualization.barcode import generate_combined_barcode

    results = populated_xyzt_results()
    short = [True] * 37                      # built from the mode alone

    with pytest.raises(ValueError, match="37 entries but mode"):
        generate_combined_barcode(results, str(tmp_path / "b"), mode="xyzt",
                                  metrics_to_visualize=short)


def test_a_correctly_built_mask_is_accepted(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from visualization.barcode import generate_combined_barcode

    results = populated_xyzt_results()
    switches = switches_for(results)
    mask = selection_mask(
        ChannelResults.get_headers(just_metrics=True, mode="xyzt", **switches), [])
    assert len(mask) == 52

    generate_combined_barcode(results, str(tmp_path / "b"), mode="xyzt",
                              metrics_to_visualize=mask)
    assert list(tmp_path.glob("b*.png")), "a barcode should have been written"


def test_hiding_metrics_still_yields_a_full_length_mask():
    """--hide-metric trims what is SHOWN; the mask must still span every column."""
    results = populated_xyzt_results()
    switches = switches_for(results)
    headers = ChannelResults.get_headers(just_metrics=True, mode="xyzt", **switches)

    mask = selection_mask(headers, ["Connectivity", "Curl"])
    assert len(mask) == len(headers), "length must not shrink when metrics are hidden"
    # Connectivity + Curl by name, plus Mesh Volume Ratio which is always QC-hidden.
    assert sum(mask) == len(headers) - 3


def test_a_static_zstack_drops_the_flow_columns():
    """A volumetric run whose flow branch produced only NaN must not emit flow columns.

    A single-timepoint z-stack (or any series too short for the flow window) cannot be
    analysed for motion, so painting seven all-NaN Speed/Divergence/Curl columns onto the
    barcode is noise. The writer, reader and barcode drop them; the 2D modes are untouched.
    """
    from core.results import flow_is_populated

    from core.results import FlowResults

    static = populated_xyzt_results()
    for r in static:
        r.flow = FlowResults()                       # all NaN -- the static case

    assert not flow_is_populated(static, "xyzt")
    headers = ChannelResults.get_headers(
        just_metrics=True, mode="xyzt", include_flow=False, **switches_for(static))
    for col in ("Speed", "Speed Change", "Mean Flow Direction", "Directional Spread",
                "Velocity Correlation Length", "Divergence", "Curl"):
        assert col not in headers

    # ...but a 2D run keeps its flow columns whatever is passed.
    assert "Speed" in ChannelResults.get_headers(
        just_metrics=True, mode="xyt", include_flow=False)
