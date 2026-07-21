"""Per-object rows: one barcode row per segmented object.

Nothing here measures anything new — every value already existed per object on the run's
detail objects. The risk is therefore not arithmetic but **joining**: the packing and
in-mask families enumerate objects independently, and a positional join would silently
attribute one cell's intensity to another whenever the two lists differ. They do differ,
because in-mask skips objects below `min_voxels` while packing keeps them.

Run: python -m pytest tests/test_object_rows.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from analysis.volumetric.objects import ObjectResults, extract_objects, objects_to_csv

SPACING = (0.235, 0.195, 0.195)
VOXEL = SPACING[0] * SPACING[1] * SPACING[2]


def labelled(n=4, shape=(10, 40, 40)):
    """n cuboids of deliberately different sizes, with non-contiguous label ids."""
    labels = np.zeros(shape, np.int32)
    ids = [5, 11, 12, 40][:n]
    for k, object_id in enumerate(ids):
        y = 4 + 9 * k
        labels[2:2 + (k + 3), y:y + 6, 4:4 + 6] = object_id
    return labels, ids


class Packing:
    def __init__(self, ids, degrees):
        self.object_ids = list(ids)
        self.contact_numbers = list(degrees)


class MaskIntensity:
    def __init__(self, ids, **columns):
        self.object_ids = list(ids)
        for name, values in columns.items():
            setattr(self, name, list(values))


class Detail:
    def __init__(self, packing=None, mask_intensity=None):
        self.packing = [packing] if packing else []
        self.mask_intensity = [mask_intensity] if mask_intensity else []


# ------------------------------------------------------------------ volumes


def test_one_row_per_object_with_the_real_ids():
    labels, ids = labelled(4)
    rows = extract_objects(labels, SPACING)
    assert [r.object_id for r in rows] == sorted(ids)


def test_volumes_match_the_voxel_count():
    labels, ids = labelled(4)
    counts = np.bincount(labels.ravel())
    rows = extract_objects(labels, SPACING)
    for row in rows:
        assert row.volume == pytest.approx(counts[row.object_id] * VOXEL)


def test_anisotropy_is_a_default_object_column():
    """Every object row carries a principal-axis anisotropy, with no mesh required."""
    labels, _ = labelled(2)
    rows = extract_objects(labels, SPACING)
    assert rows, "expected object rows"
    for row in rows:
        assert np.isfinite(row.anisotropy)
        assert row.anisotropy >= 1.0 - 1e-9      # major/minor ratio


def test_anisotropy_flags_an_elongated_object():
    """A long thin column reads more anisotropic than a near-cube."""
    shape = (30, 30, 30)
    labels = np.zeros(shape, np.int32)
    labels[4:8, 4:8, 4:8] = 1                    # ~cube
    labels[2:28, 14:18, 14:18] = 2               # long column in z
    rows = {r.object_id: r.anisotropy for r in extract_objects(labels, (0.2, 0.2, 0.2))}
    assert rows[2] > rows[1]
    assert rows[2] > 2.0


def test_equivalent_diameter_is_no_longer_emitted():
    """Diameter was dropped as redundant with volume; it must not reappear as a column."""
    assert "Equivalent Diameter" not in ObjectResults.get_headers()
    assert "Anisotropy" in ObjectResults.get_headers()


def test_an_empty_label_volume_yields_no_rows():
    assert extract_objects(np.zeros((4, 8, 8), np.int32), SPACING) == []


# ------------------------------------------------------------------ the join


def test_contact_numbers_join_by_id_not_by_position():
    """The failure this guards: packing listing objects in a different order."""
    labels, ids = labelled(4)                      # ids 5, 11, 12, 40
    reversed_ids = list(reversed(sorted(ids)))     # 40, 12, 11, 5
    degrees = [9, 8, 7, 6]                         # so 40->9, 12->8, 11->7, 5->6
    rows = extract_objects(labels, SPACING,
                           Detail(packing=Packing(reversed_ids, degrees)))
    got = {r.object_id: r.contact_number for r in rows}
    assert got == {40: 9, 12: 8, 11: 7, 5: 6}


def test_an_object_missing_from_a_family_is_nan_not_a_neighbours_value():
    """in-mask skips small objects; the rest must not shift up into the gap."""
    labels, ids = labelled(4)
    partial = [i for i in sorted(ids) if i != 11]            # 11 was skipped
    intensity = MaskIntensity(partial, mfi=[100.0, 200.0, 300.0],
                              cv=[0.1, 0.2, 0.3], entropy=[1.0, 2.0, 3.0],
                              sd=[10.0, 20.0, 30.0], skewness=[0.5, 0.6, 0.7],
                              entropy_normalized=[0.8, 0.85, 0.9],
                              bright_fraction=[0.01, 0.02, 0.03])
    rows = {r.object_id: r for r in extract_objects(
        labels, SPACING, Detail(mask_intensity=intensity))}

    assert np.isnan(rows[11].mfi), "a skipped object must be NaN, not a neighbour's value"
    assert rows[5].mfi == 100.0 and rows[12].mfi == 200.0 and rows[40].mfi == 300.0


def test_a_mismatched_family_is_ignored_rather_than_zipped_wrongly():
    labels, ids = labelled(4)
    broken = Packing(sorted(ids), [1, 2])           # lengths disagree
    rows = extract_objects(labels, SPACING, Detail(packing=broken))
    assert all(np.isnan(r.contact_number) for r in rows)


def test_all_seven_in_mask_columns_come_through():
    labels, ids = labelled(2)
    order = sorted(ids)
    intensity = MaskIntensity(order, mfi=[1.0, 2.0], cv=[0.1, 0.2], entropy=[3.0, 4.0],
                              sd=[5.0, 6.0], skewness=[0.7, 0.8],
                              entropy_normalized=[0.9, 0.95], bright_fraction=[0.01, 0.02])
    row = extract_objects(labels, SPACING, Detail(mask_intensity=intensity))[0]
    assert (row.mfi, row.intensity_cv, row.entropy, row.intensity_sd,
            row.intensity_skew, row.entropy_normalized, row.bright_fraction) == \
        (1.0, 0.1, 3.0, 5.0, 0.7, 0.9, 0.01)


# ------------------------------------------------------------------ schema


def test_the_column_set_is_per_object_only():
    """Field-level metrics are omitted, not repeated down every row."""
    headers = ObjectResults.get_headers(just_metrics=True)
    assert len(headers) == 16
    assert "Object Volume" in headers and "Contact Number" in headers
    # Shape columns come from each object's OWN mesh (object_mesh.mesh_objects); before
    # that existed they could only have carried the largest object's numbers.
    for shape in ("Sphericity", "Solidity", "Concavity", "Aspect Ratio",
                  "Mesh Surface Area", "Mean Curvature <H>"):
        assert shape in headers
    for field_level in ("Connectivity", "Structural Correlation Length",
                        "Maximum Kurtosis", "Speed", "Mean Contact Number"):
        assert field_level not in headers, f"{field_level} is not a per-object metric"


def test_identity_columns_lead_the_csv(tmp_path):
    labels, _ = labelled(3)
    rows = extract_objects(labels, SPACING, filepath="a.tif", fov="a")
    path = objects_to_csv(rows, str(tmp_path / "objects.csv"))

    import csv
    written = list(csv.reader(open(path)))
    assert written[0][:3] == ["File", "FOV", "Object"]
    assert len(written) == 4
    assert written[1][1] == "a"


def test_units_align_with_metrics():
    assert len(ObjectResults.get_units()) == len(ObjectResults.get_metrics())


def test_rows_render_through_the_shared_barcode_renderer(tmp_path):
    """The renderer is parameterised rather than duplicated; prove object rows work."""
    import matplotlib
    matplotlib.use("Agg")
    from visualization.barcode import generate_combined_barcode

    labels, _ = labelled(4)
    rows = extract_objects(labels, SPACING, filepath="a.tif", fov="a")
    generate_combined_barcode(rows, str(tmp_path / "objects"), separate_channels=False,
                              results_cls=ObjectResults)
    assert list(tmp_path.glob("objects*.png"))


def test_a_very_tall_barcode_still_renders(tmp_path):
    """4176 rows exceeded matplotlib's 2^16 pixel limit and wrote nothing at all.

    That bit any run of roughly 960+ rows, objects or not.
    """
    import matplotlib
    matplotlib.use("Agg")
    from visualization.barcode import generate_combined_barcode

    rows = [ObjectResults(object_id=i, volume=float(i), anisotropy=1.0 + i / 1500,
                          contact_number=6.0, mfi=float(i)) for i in range(1, 1500)]
    generate_combined_barcode(rows, str(tmp_path / "tall"), separate_channels=False,
                              results_cls=ObjectResults)
    assert list(tmp_path.glob("tall*.png"))
