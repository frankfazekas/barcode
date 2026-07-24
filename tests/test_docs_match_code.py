"""The two reference documents must not drift from the code, or from each other.

There are two hand-maintained documents describing the same metrics:

* ``docs/volumetric_reference.md`` -- the working manual (scope, column index, every
  metric, all 78 configuration settings, output guidance, known limitations);
* ``docs/_assets/volumetric_metrics.html`` -- the source of ``docs/BARCODE Volumetric
  Metrics.pdf``, the formal view with typeset equations.

Neither is generated from the other, and every documentation defect found so far has been
a *sync* failure rather than a writing error:

* the code renamed the unit ``% of FOV`` to ``fraction of FOV``; both documents kept the
  old label, so a barcode colourbar and its own reference disagreed;
* ``Concavity`` and ``Aspect Ratio`` were removed/renamed in the code and the markdown,
  and stayed live in the PDF, which documented two columns that no run can emit;
* the PDF merged two subsections and so numbered everything after them one lower, meaning
  "section 1.8" named *Mean Island Anisotropy* in one document and *Mean island volume* in
  the other;
* the depth-profile columns were documented under names (``Broadest Slice ...``) that the
  code never emitted, so searching a CSV for them found nothing.

Each of those is mechanically detectable, which is what this module does. ``docs/`` is
gitignored, so every test skips when the documents are absent rather than failing a fresh
clone.

Run: python -m pytest tests/test_docs_match_code.py -v
"""
from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parent.parent / "docs"
MARKDOWN = DOCS / "volumetric_reference.md"
PDF_SOURCE = DOCS / "_assets" / "volumetric_metrics.html"


def _read(path: Path) -> str:
    if not path.exists():
        pytest.skip(f"{path.name} is not present (docs/ is gitignored)")
    # Authored on Windows; normalise so a literal search is not defeated by \r.
    return path.read_text(encoding="utf-8").replace("\r", "")


def emitted_columns() -> set:
    """Every column name any run can produce, across modes and both unit systems."""
    from analysis.volumetric.objects import ObjectResults
    from core.modes import MODES
    from core.results import OPTIONAL_FAMILIES, ChannelResults

    families = {family.switch: True for family in OPTIONAL_FAMILIES}
    names = set()
    for key in ("xyt", "xyz", "xyzt"):
        for physical in (False, True):
            try:
                names |= set(ChannelResults.get_headers(
                    mode=MODES[key], physical_units=physical, **families))
            except TypeError:            # older signature without physical_units
                names |= set(ChannelResults.get_headers(mode=MODES[key], **families))
    return names | {metric.value for metric in ObjectResults.get_metrics()}


# --------------------------------------------------------------- code vs docs


@pytest.mark.parametrize("doc", ["markdown", "pdf"])
def test_every_emitted_column_is_documented(doc):
    """A column nobody can look up is undocumented, whatever the prose says.

    This is the check that catches a renamed column: the docs described
    ``Broadest Slice Area`` for a run that emits ``Maximal Area Slice Area``.
    """
    text = _read(MARKDOWN if doc == "markdown" else PDF_SOURCE)
    if doc == "pdf":                     # the PDF source escapes markup characters
        text = text.replace("&lt;", "<").replace("&gt;", ">")

    missing = sorted(name for name in emitted_columns() if name not in text)
    assert not missing, f"{doc} does not document emitted column(s): {missing}"


def test_every_configuration_field_is_documented():
    """A setting absent from the reference cannot be found by someone reading it.

    Spelled-out names only: an abbreviated entry (``mesh_smoothing_alpha`` / ``_beta``)
    is invisible to anyone searching for the field name the YAML actually uses.
    """
    from core.config import VolumetricConfig

    text = _read(MARKDOWN)
    missing = sorted(f.name for f in dataclasses.fields(VolumetricConfig)
                     if f.name not in text)
    assert not missing, f"undocumented VolumetricConfig field(s): {missing}"


@pytest.mark.parametrize("doc", ["markdown", "pdf"])
def test_no_superseded_unit_labels(doc):
    """The unit labels were corrected in the code; the docs must not resurrect them.

    ``core/metrics.py`` emits ``fraction of FOV``, ``fraction of frames`` and
    ``ratio to initial``. The three strings below are what those columns used to be
    called, and describe the data wrongly -- a value of 0.94 is not "0.94 %".
    """
    text = _read(MARKDOWN if doc == "markdown" else PDF_SOURCE)
    for stale in ("% of FOV", "% of Frames", "Fractional Change"):
        assert stale not in text, (
            f"{doc} uses the superseded unit label {stale!r}; "
            f"see core/metrics.py Units for the current one"
        )


def test_documented_units_match_the_code():
    """The unit a column is documented with must be the unit it is emitted with."""
    from core.modes import MODES
    from core.results import OPTIONAL_FAMILIES, ChannelResults

    text = _read(MARKDOWN)
    families = {family.switch: True for family in OPTIONAL_FAMILIES}
    headers = ChannelResults.get_headers(just_metrics=True, mode=MODES["xyzt"], **families)
    units = ChannelResults.get_units(just_metrics=True, mode=MODES["xyzt"], **families)

    wrong = []
    for column, unit in zip(headers, units):
        if not unit or unit in ("a.u.", ""):
            continue                     # nothing distinctive to look for
        # The column index lists "| `Column` | modes | unit | section |".
        row = re.search(rf"^\|\s*`{re.escape(column)}`\s*\|[^|]*\|\s*([^|]*?)\s*\|",
                        text, re.M)
        if row and unit not in row.group(1):
            wrong.append(f"{column}: code={unit!r} doc={row.group(1)!r}")
    assert not wrong, "column index disagrees with the emitted units: " + "; ".join(wrong)


# ------------------------------------------------------- markdown vs the PDF


def _markdown_subsections(text: str) -> dict:
    return {m.group(1): m.group(2).strip()
            for m in re.finditer(r"^#{3,4} (\d+\.\d+) (.*)$", text, re.M)}


def _pdf_subsections(text: str) -> dict:
    return {m.group(1): m.group(2).strip()
            for m in re.finditer(r"<h[34][^>]*>([\d.–]+) &nbsp; (.*?)</h[34]>", text)}


def _normalise(title: str) -> str:
    title = re.sub(r"<[^>]+>|&nbsp;", "", title)
    title = re.sub(r"\$[^$]*\$", "", title)      # drop inline maths
    title = re.sub(r"\([^)]*\)", "", title)      # drop trailing symbols
    return re.sub(r"[^a-z ]", "", title.lower()).strip()


def test_shared_section_numbers_name_the_same_metric():
    """A citation must resolve to one metric, whichever document the reader has.

    The PDF merges some metrics into one derivation and labels those headings with a
    range ("4.29-4.37"); ranges are skipped here. Every heading carrying a single number
    has to agree, which is what stops the off-by-one that made section 1.8 ambiguous.
    """
    md = _markdown_subsections(_read(MARKDOWN))
    pdf = _pdf_subsections(_read(PDF_SOURCE))

    disagreements = []
    for number, pdf_title in pdf.items():
        if "–" in number or number.startswith("0."):
            continue                     # a range, or PDF-only front matter
        if number not in md:
            disagreements.append(f"§{number} is in the PDF only ({pdf_title!r})")
            continue
        a, b = _normalise(md[number]), _normalise(pdf_title)
        if not (a.startswith(b[:12]) or b.startswith(a[:12])):
            disagreements.append(f"§{number}: markdown={a!r} pdf={b!r}")
    assert not disagreements, "the two documents disagree: " + "; ".join(disagreements)


def test_every_internal_link_resolves():
    """A broken anchor is invisible until a reader clicks it."""
    text = _read(MARKDOWN)
    anchors = {re.sub(r"[^a-z0-9 -]", "", h.lower()).replace(" ", "-")
               for h in re.findall(r"^#{2,4} (.*)$", text, re.M)}
    broken = sorted({t for t in re.findall(r"\]\(#([a-z0-9-]+)\)", text)
                     if t not in anchors})
    assert not broken, f"broken internal link(s): {broken}"


def test_cross_references_resolve_to_real_subsections():
    """Every "§4.31" must point at a subsection that exists."""
    text = _read(MARKDOWN)
    present = set(_markdown_subsections(text))
    dangling = sorted({ref for ref in re.findall(r"§(\d+\.\d+)", text)
                       if ref not in present})
    assert not dangling, f"dangling cross-reference(s): {dangling}"


def test_sibling_documents_point_at_files_that_exist():
    """The reference is a hub; a renamed file must not orphan its companions."""
    for doc in sorted(DOCS.glob("*.md")) if DOCS.exists() else []:
        text = doc.read_text(encoding="utf-8").replace("\r", "")
        for target in re.findall(r"\]\(([a-z_]+\.md)\)", text):
            assert (DOCS / target).exists(), f"{doc.name} links to missing {target}"
