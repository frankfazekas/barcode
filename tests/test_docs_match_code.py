"""The two reference documents must not drift from the code, or from each other.

There are two hand-maintained documents describing the same metrics:

* ``docs/volumetric_manual.md`` -- the working manual (scope, column index, every
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
  code never emitted, so searching a CSV for them found nothing;
* the PDF's column index pointed one section too high for everything between
  ``Initial 2nd Maximum Island ...`` and ``Mean Island Separation``, so looking up
  ``Mean Island Anisotropy`` led the reader to *Mean island volume*. Every column NAME
  still matched, which is all the checks above compare;
* both documents opened by saying 25 of 63 metrics carry over from 2D and 39 do not --
  a sum of 64. Every column was documented; only the sentence was wrong;
* three TeX macros in the PDF source were written as the control character they escape
  to (``\nabla`` as a newline, ``\times`` as a tab, ``\rangle`` as a carriage return),
  so the printed PDF showed "abla" and "imes" where operators belonged.

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
MARKDOWN = DOCS / "volumetric_manual.md"
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


def test_no_documented_setting_that_does_not_exist():
    """The reverse of the check below: a named setting must be a real config field.

    ``test_every_configuration_field_is_documented`` proves nothing is missing, and is
    blind to the opposite defect -- the manual told readers that ``flow_win_size`` and
    ``flow_downsample`` are measured in voxels, and ``flow_win_size`` has never existed
    anywhere in the codebase. Searching a Settings.yaml for it finds nothing, and the
    advice it carries cannot be acted on.

    Restricted to the prefixes that are unambiguously VolumetricConfig namespaces, so a
    backticked local variable or a function from another module is not swept up.
    """
    from core.config import VolumetricConfig

    fields = {f.name for f in dataclasses.fields(VolumetricConfig)}
    prefixes = ("flow_", "mesh_", "packing_", "curvature_", "segmentation_", "mask_",
                "crop_", "timelapse_", "object_mesh", "z_range", "t_range",
                "enable_", "record_range", "make_isotropic", "invert_binarization",
                "threshold_offset", "minimum_island_size", "neighbor_island",
                "frame_step", "frame_interval", "percentage_frames", "bin_size",
                "noise_threshold", "intensity_use_mask", "write_fingerprint",
                "fingerprint_")
    text = _read(MARKDOWN)

    invented = set()
    for name in re.findall(r"`([a-z][a-z0-9]*(?:_[a-z0-9]+)+)`", text):
        if name in fields:
            continue
        if any(name.startswith(p) for p in prefixes):
            invented.add(name)
    assert not invented, (
        f"the manual names setting(s) that are not VolumetricConfig fields: "
        f"{sorted(invented)}; a reader cannot find them in a Settings.yaml"
    )


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


def _all_docs():
    """Every hand-maintained doc file, wherever it now lives under docs/.

    A superseded label can hide in any of them, not just the two the metric entries
    live in -- an orphaned ``_column_index.md`` kept the old ``% of FOV`` long after
    the barcode colorbars were corrected. Recurses, so filing the working documents
    under ``docs/internal/`` does not quietly drop them from this check;
    ``docs/reference/`` is third-party material and is not ours to police.
    """
    if not DOCS.exists():
        return []
    files = [p for p in sorted(DOCS.rglob("*.md"))
             if "reference" not in p.relative_to(DOCS).parts]
    files.append(PDF_SOURCE)
    return [f for f in files if f.exists()]


def test_no_superseded_unit_labels():
    """The unit labels were corrected in the code; no doc may resurrect them.

    ``core/metrics.py`` emits ``fraction of FOV``, ``fraction of frames`` and
    ``ratio to initial``. The three strings below are what those columns used to be
    called, and describe the data wrongly -- a value of 0.94 is not "0.94 %".
    """
    docs = _all_docs()
    if not docs:
        pytest.skip("docs/ is not present (gitignored)")
    bad = []
    for path in docs:
        text = path.read_text(encoding="utf-8").replace("\r", "")
        for stale in ("% of FOV", "% of Frames", "Fractional Change"):
            if stale in text:
                bad.append(f"{path.name}: {stale!r}")
    assert not bad, (
        "superseded unit label(s) present; see core/metrics.py Units for the "
        "current ones: " + "; ".join(bad)
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
    """The manual is a hub; a renamed or relocated file must not orphan its companions.

    Resolves each link against the *linking* document's own directory, so the working
    documents under ``docs/internal/`` are held to the same standard -- their links up
    to ``../volumetric_manual.md`` have to resolve too, which a docs/-relative check
    would have silently skipped.
    """
    broken = []
    for doc in _all_docs():
        if doc.suffix != ".md":
            continue
        text = doc.read_text(encoding="utf-8").replace("\r", "")
        for target in re.findall(r"\]\((\.{0,2}[/a-z_]*[a-z_]+\.md)\)", text):
            if not (doc.parent / target).resolve().exists():
                broken.append(f"{doc.relative_to(DOCS)} -> {target}")
    assert not broken, f"link(s) to a missing document: {broken}"


# --------------------------------------------------- counts, indexes, renumbering
#
# These catch defects the name-presence checks above are structurally blind to: a
# metric total that is off by one (every column is still individually documented,
# only the SUM is wrong), a cross-reference left dangling by the families-5-to-10
# fold into section 4, and the two documents' column indexes drifting apart.


def _all_families():
    from core.results import OPTIONAL_FAMILIES
    return {family.switch: True for family in OPTIONAL_FAMILIES}


def _metric_count(mode_key):
    """xyzt-style metric count (no identity columns), all optional families on."""
    from core.modes import MODES
    from core.results import ChannelResults
    return len(ChannelResults.get_metrics(
        just_metrics=True, mode=MODES[mode_key], **_all_families()))


@pytest.mark.parametrize("doc", ["markdown", "pdf"])
def test_documented_metric_total_matches_code(doc):
    """The 'emits up to N' claim must equal the code's xyzt metric count.

    An off-by-one here passes every column-name check -- each of the 63 columns is
    still documented -- while the headline count reads 64. This is the check that
    would have caught it.
    """
    text = _read(MARKDOWN if doc == "markdown" else PDF_SOURCE)
    m = re.search(r"emits up to (?:\*\*|<b>)?(\d+)", text)
    assert m, f"{doc}: could not find the 'emits up to N' claim"
    n = _metric_count("xyzt")
    assert int(m.group(1)) == n, (
        f"{doc} says the pipeline emits up to {m.group(1)} metrics; "
        f"code emits {n} (xyzt, all families on)"
    )


@pytest.mark.parametrize("doc", ["markdown", "pdf"])
def test_front_matter_arithmetic_adds_up(doc):
    """"25 carry over ... the remaining N" must equal the total minus the 25.

    Both documents opened with 63 total, 25 restated and 39 remaining -- which sums to
    64. Every individual column was still documented and every other count was right, so
    nothing else in this module noticed. Only the sentence was wrong, and it is the first
    quantitative claim a reader meets.
    """
    text = _read(MARKDOWN if doc == "markdown" else PDF_SOURCE)
    total = int(re.search(r"emits up to (?:\*\*|<b>)?(\d+)", text).group(1))
    # Anchor on the claim itself. Matching "N metrics" instead picks up any later
    # sentence that happens to quote a count -- an admonition saying the two documents
    # "cover the same 63 metrics" once made this read 63 of 63.
    restated = re.search(r"all (\d+) carry over", text)
    assert restated, f"{doc}: could not find the 'all N carry over' claim"
    remaining = re.search(r"remaining (?:\*\*|<b>)?(\d+)", text)
    assert remaining, f"{doc}: could not find the 'remaining N' claim"
    want = total - int(restated.group(1))
    assert int(remaining.group(1)) == want, (
        f"{doc}: says {restated.group(1)} of {total} carry over and {remaining.group(1)} "
        f"remain, but {total} - {restated.group(1)} = {want}"
    )


@pytest.mark.parametrize("doc", ["markdown", "pdf"])
def test_documented_resampled_slice_count_matches_the_resampler(doc):
    """The worked example's isotropic slice count must be what the resampler computes.

    Both documents quoted "54 -> ~249" for the 54-plane 0.3 um Jurkat geometry resampled
    to 0.065 um. That is the naive n*dz/dxy estimate; the resampler actually computes
    ``floor((n-1)*src/tgt + 1)`` = 245, which is also what the GUI walkthrough measured
    end to end. The figure appears wherever the manual explains why a slice INDEX is not
    comparable across grids, so being wrong about it undermines the very point it makes.

    A staged *mask* uses ``round(n*z/xy)`` and so genuinely differs -- that number is
    labelled as such in the text and is not what this checks.
    """
    from analysis.volumetric.resample import _reference_shape_for_spacing

    text = _read(MARKDOWN if doc == "markdown" else PDF_SOURCE)
    n_acquired, dz, dxy = 54, 0.3, 0.065
    expected = _reference_shape_for_spacing(
        (n_acquired, 312, 303), (dxy, dxy, dz), (dxy, dxy, dxy))[0]

    # Guard both ends against decimals: "6.54 -> 1.46" (the anisotropy figure in the run
    # recipes) otherwise reads as "54 -> 1".
    quoted = {int(n) for n in re.findall(rf"(?<![\d.]){n_acquired}\s*(?:→|versus|vs\.?)\s*"
                                         rf"\$?\\?~?≈?\s*(?:approx)?(\d+)(?![\d.])", text)}
    assert quoted, f"{doc}: could not find a '{n_acquired} -> N' slice-count claim"
    wrong = sorted(n for n in quoted if n != expected)
    assert not wrong, (
        f"{doc} says {n_acquired} acquired planes resample to {wrong}; "
        f"_reference_shape_for_spacing gives {expected}"
    )


def test_pdf_maths_has_no_stray_control_characters():
    r"""A TeX macro must not have been written as the character it escapes to.

    Three spans reached the printed PDF as raw control characters -- ``\nabla`` as a
    newline plus "abla", ``\times`` as a TAB plus "imes", ``\rangle`` as a carriage
    return plus "angle" -- so the PDF showed "abla" and "imes" where operators belonged.
    A text-mode read hides this, since universal newlines fold CR into LF; read bytes.

    Also enforces the document's own rule that a literal ``<`` inside maths is escaped:
    it renders, but it defeats every HTML-extraction regex in this module.
    """
    if not PDF_SOURCE.exists():
        pytest.skip("the PDF source is not present (docs/ is gitignored)")
    raw = PDF_SOURCE.read_bytes().decode("utf-8")

    stray = sorted({repr(c) for c in raw if c in "\t\r\x07\x08\x0b\x0c"})
    assert not stray, (
        f"control character(s) {stray} in the PDF source -- almost certainly a TeX macro "
        r"written as its escape (\t for \times, \r for \rangle, ...)"
    )

    # A newline immediately followed by the tail of a common macro is the \n case, which
    # the check above cannot see because newlines are legitimate everywhere else.
    tails = ("abla", "angle", "ewline", "umber", "amma", "otin")
    broken = sorted({m.group(1) for m in
                     re.finditer(r"\n(" + "|".join(tails) + r")\b", raw)})
    assert not broken, (
        f"line(s) beginning {broken} -- a macro whose backslash-escape was interpreted "
        r"(\nabla, \rangle, ...); write the backslash literally"
    )

    # Display maths only. Inline spans are interleaved with real markup, so scanning them
    # for '<' reports every "$x$ <span>y</span> $z$"; a display block is pure TeX, which
    # is where the binarization rule (1.2) once carried a raw '<' against the document's
    # own stated rule. It renders -- HTML5 emits a lone '<' as text -- but it defeats the
    # extraction regexes in this module.
    unescaped = [b[:60] for b in re.findall(r"\$\$.*?\$\$", raw, re.S) if "<" in b]
    assert not unescaped, (
        f"literal '<' in display maths: {unescaped}; escape it as &lt;"
    )


def test_scope_table_counts_match_code():
    """'Columns, all families on' must equal metrics + 3 identity, per mode."""
    text = _read(MARKDOWN)
    row = re.search(r"\|\s*Columns, all families on\s*\|(.+)", text)
    assert row, "could not find the scope table's column-count row"
    got = [int(n) for n in re.findall(r"\d+", row.group(1))]
    want = [_metric_count("xyt") + 3, _metric_count("xyz") + 3, _metric_count("xyzt") + 3]
    assert got == want, (
        f"scope table says {got}; code gives {want} (metrics + 3 identity columns)"
    )


# Legitimate [5-9].N NUMBERS that are not section references (a maxrad of 5.0, a
# kurtosis of 9.93, ...). No section has a leading digit >= 5 since families 5-10
# folded into section 4, so any *other* such bare token is a stale cross-reference.
# A new genuine value gets added here; a new stale ref fails the test.
_KNOWN_NUMERIC_VALUES = {
    "5.0", "6.0", "7.434", "8.8", "9.93",
    # From the run recipes and the validation summary folded into the manual so it can
    # be shared without its companion documents: anisotropy before and after a mask
    # loads, run times in seconds, and coefficients of variation in percent.
    "5.89", "6.5", "6.54", "6.5410", "7.54", "8.5", "8.9", "9.3",
}


def test_no_stale_family_cross_references():
    """A bare 5.x/6.x/.../9.x is a reference to a section number that no longer exists."""
    text = _read(MARKDOWN)
    nofence = re.sub(r"```.*?```", "", text, flags=re.S)   # drop code blocks
    nocode = re.sub(r"`[^`]*`", "", nofence)               # drop inline code
    stale = sorted({t for t in re.findall(r"(?<![\w.])[5-9]\.\d+(?![\w.])", nocode)
                    if t not in _KNOWN_NUMERIC_VALUES})
    assert not stale, (
        f"markdown has bare [5-9].N token(s) {stale}; families 5-10 folded into "
        f"section 4, so these read as stale cross-references. If one is a genuine "
        f"numeric value, add it to _KNOWN_NUMERIC_VALUES."
    )


def _markdown_index_columns(text):
    section = text.split("## Column index")[1].split("## Measurement preconditions")[0]
    return set(re.findall(r"^\|\s*`([^`]+)`\s*\|", section, re.M))


def _pdf_index_columns(text):
    # Column-index rows are a mono span in a <td> immediately followed by a modes
    # cell -- 'xyt, ...' or 'object rows'. Other mono spans on the page are not.
    # Capture the (escaped) name first, THEN unescape -- unescaping earlier would
    # turn '&lt;H&gt;' into '<H>' and defeat the [^<]+ capture on that very column.
    section = text.split("Column index")[-1].split("Reading the outputs")[0]
    names = re.findall(r'<td><span class="mono">([^<]+)</span></td><td>(?:xy|object)',
                       section)
    return {n.replace("&lt;", "<").replace("&gt;", ">") for n in names}


def test_column_indexes_agree():
    """The markdown and PDF column indexes must list exactly the same columns."""
    md = _markdown_index_columns(_read(MARKDOWN))
    pdf = _pdf_index_columns(_read(PDF_SOURCE))
    assert md == pdf, (
        f"column-index tables disagree -- only in markdown: {sorted(md - pdf)}; "
        f"only in PDF: {sorted(pdf - md)}"
    )


def _markdown_index_sections(text):
    section = text.split("## Column index")[1].split("## Measurement preconditions")[0]
    return {m.group(1): m.group(2)
            for m in re.finditer(r"^\|\s*`([^`]+)`\s*\|[^|]*\|[^|]*\|\s*§?([^|]*?)\s*\|",
                                 section, re.M)}


def _pdf_index_sections(text):
    section = text.split("Column index")[-1].split("Reading the outputs")[0]
    rows = re.findall(
        r'<td><span class="mono">([^<]+)</span></td><td>(?:xy|object)[^<]*</td>'
        r'<td>[^<]*</td><td>§?([^<]*)</td>',
        section)
    return {n.replace("&lt;", "<").replace("&gt;", ">"): s.strip() for n, s in rows}


def test_column_indexes_agree_on_the_defining_section():
    """A column's section pointer must name the same entry in both documents.

    ``test_column_indexes_agree`` compares column NAMES, which is blind to the defect it
    was written to catch one level down: the PDF merges 1.6-1.7 into a single heading, and
    its column index was once numbered off the merged heading, so every row from
    ``Initial 2nd Maximum Island ...`` to ``Mean Island Separation`` pointed one section
    too high -- ``Mean Island Anisotropy`` sent the reader to *Mean island volume*. Every
    name still matched; only the pointers were wrong.

    The PDF legitimately cites a merged entry by its range ("1.6-1.7", "4.29-4.37"), so a
    pointer agrees when the markdown's number falls inside the PDF's range.
    """
    md = _markdown_index_sections(_read(MARKDOWN))
    pdf = _pdf_index_sections(_read(PDF_SOURCE))

    def covers(pdf_ref: str, md_ref: str) -> bool:
        pdf_ref = pdf_ref.replace("–", "-").replace("—", "-")
        if pdf_ref == md_ref:
            return True
        if "-" not in pdf_ref:
            return False
        low, high = (part.strip() for part in pdf_ref.split("-", 1))
        if "." not in high:                       # "4.29-4.37" and also "4.19-22"
            high = f"{low.split('.')[0]}.{high}"
        try:
            as_pair = lambda s: tuple(int(p) for p in s.split("."))
            return as_pair(low) <= as_pair(md_ref) <= as_pair(high)
        except ValueError:
            return False

    wrong = sorted(f"{name}: markdown=§{md[name]} pdf=§{pdf[name]}"
                   for name in md if name in pdf and not covers(pdf[name], md[name]))
    assert not wrong, (
        "column-index section pointers disagree between the documents; a citation would "
        "resolve to a different metric: " + "; ".join(wrong)
    )


def test_object_only_columns_are_tagged_object_rows():
    """A column that exists ONLY on an object row must be labelled so in the index.

    ``Sphericity`` also exists as a field-mesh column, so looking it up is ambiguous
    -- but ``Object Volume``, ``Anisotropy`` and ``Contact Number`` exist *only* per
    object, and a reader holding an Objects.csv must find them tagged accordingly.
    """
    from analysis.volumetric.objects import ObjectResults
    from core.modes import MODES
    from core.results import ChannelResults

    text = _read(MARKDOWN)
    fam = _all_families()
    field = set()
    for key in ("xyt", "xyz", "xyzt"):
        field |= set(ChannelResults.get_headers(just_metrics=True, mode=MODES[key], **fam))
        field |= {m.value for m in
                  ChannelResults.get_physical_metrics(just_metrics=True, mode=MODES[key], **fam)}
    object_only = [m.value for m in ObjectResults.get_metrics() if m.value not in field]

    section = text.split("## Column index")[1].split("## Measurement preconditions")[0]
    bad = []
    for name in object_only:
        row = re.search(rf"^\|\s*`{re.escape(name)}`\s*\|\s*([^|]*)\|", section, re.M)
        if not row:
            bad.append(f"{name}: absent from the column index")
        elif "object rows" not in row.group(1):
            bad.append(f"{name}: not tagged 'object rows' (modes={row.group(1).strip()!r})")
    assert not bad, "object-only column(s) mis-indexed: " + "; ".join(bad)
