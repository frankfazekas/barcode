# BARCODE Volumetric Manual

Companion to the published 2D reference, [BARCODE 2.0 Metrics](https://www.livingbam.org/barcode-2-metrics),
and written to the same shape: numbered branches, one entry per metric, each stating what
it measures, exactly how it is computed, its units, and the conditions under which it is
NaN or raises a flag.

> **Not to be confused with `BARCODE Volumetric Metrics.pdf`.** That is the formal
> specification — notation, display equations, sign conventions. This is the manual: how to
> run the thing, what every setting does, and how to read what comes out. The two cover the
> same 63 metrics and are kept in step by `tests/test_docs_match_code.py`.

The 2D page defines **25** metrics in three branches — 24 numbered entries, since entry 1.6
covers two. This pipeline is an extension of that work rather than a replacement. It
emits up to **63** metrics in ten families, of which all 25 carry over — 17 under identical
names, 8 renamed Area → Volume — and the remaining **38** are measurements a volume affords
that a plane does not. Restating a metric for a volume sometimes changes what it measures;
where the two are *not numerically comparable*, the entry says so in a **vs 2D** note. The 2D
branch is unchanged and remains the reference for 2D data.

That count is of **distinct measurements**. Six of them — the island and void sizes — are
*written* under two different headers depending on the unit system requested: a bare name
holding a fraction of the analyzed field, and a `… Quantity` name holding µm³. A CSV carries
one set or the other, so a single file has at most 63 metric columns while 69 distinct headers
exist across both. The [column index](#column-index--look-up-what-you-have) lists every one.

> ## ★ New capability: BARCODE can now be given a segmentation
>
> Published BARCODE defines "material" by thresholding intensity at `mean × (1 + offset)` —
> a self-contained rule that needs nothing but the image. **This pipeline adds an optional
> second route: an external segmentation mask, which when it resolves *replaces* the
> threshold.** Objects are then whatever your segmenter says they are. Both routes are
> supported; the mask is an addition, not a correction.
>
> It is the largest behavioral difference from the 2D reference. Two metric families are
> only possible with a mask, and two more change what they describe:
>
> | Family | Behavior without a mask |
> |---|---|
> | **In-Mask Intensity** (§4.29–4.37) | **not produced** — there is no object to look inside |
> | **Packing Topology** (§4.38–4.44) | **not produced**; also needs integer instance labels, and an isotropic grid |
> | **Surface Mesh** (§4.1–4.11) | produced, but meshes the **binarized volume** — a threshold surface, not the object |
> | **Curvature** (§4.12–4.18) | likewise, being computed on that surface |
>
> It also changes metrics that do exist without one: the whole binarization branch (§1)
> measures the segmented object rather than a threshold, and flow and the intensity
> histogram can optionally be restricted to in-mask voxels (`flow_use_mask` — **on by
> default**; `intensity_use_mask` — off).
>
> **Formats.** `.tif`/`.tiff`, `.png`/`.bmp`, `.npy`, `.npz` — including **Cellpose
> `_seg.npy` result bundles**, which are unwrapped by looking for a `masks`/`mask`/`labels`
> key. Dispatch is on the file suffix. (A Cellpose bundle must be unpickled, which executes
> code, so point it only at masks you produced.)
>
> **Instance labels are preserved**, not collapsed to a boolean. That is what makes a
> confluent field measurable: deriving objects by connectivity would fuse every touching
> cell into one component.
>
> **Pairing is by rule, not by hand.** A regex over the image stem feeds a path template, so
> a whole batch resolves automatically — by default `Cell1_1.tif` → `Cell1_1_SegMask.tif`.
> Every check then **fails loudly** rather than degrading, because a silently misaligned
> mask produces confidently wrong numbers, which is worse than no mask at all. See §P.3 for
> the resolution rules and the validation, and §P.4 for what resampling a mask onto the
> image grid does to the numbers.

Everything here describes `analysis/volumetric/`, reached from
`core/pipeline.py::process_single_file` when `analysis_mode` is `xyzt`, or
`analysis/volumetric/slicewise.py` when it is `xyz`. The 2D branches are untouched — see
[How the numbers were checked](#how-the-numbers-were-checked) for the isolation evidence.

> **There is also a typeset PDF: `docs/BARCODE Volumetric Metrics.pdf`** (37 pages), built
> from `docs/_assets/volumetric_metrics.html`. It states each metric as a numbered display
> equation in the style of the published reference. The two are complements, not duplicates:
>
> - **The PDF** is the formal specification — notation, equations, sign conventions, and the
>   known limitations.
> - **This document** is the operational reference — the full configuration table with every
>   default, GUI control names, CLI flags, per-mode column counts, and the practical caveats.
>
> **Section numbers agree from §1.1 to §4.46 and nowhere else.** Cite those freely; for
> everything around them the two documents number differently:
>
> | This document | The PDF |
> |---|---|
> | §P.1–§P.5 | P.1–P.5 (same) |
> | §P.6 *Which timepoints are analyzed* | 0.1 *Timepoint selection* |
> | §P.7 *Time-lapse assembly* | *(not covered — operational)* |
> | *Description* (the three conventions) | 0.2–0.3 *Notation and common reductions* |
> | *Warnings and flags* (unnumbered) | **section 5** |
> | *Known limitations* (unnumbered) | **section 6**, open issues labeled **L1–L8** |
>
> The PDF also merges some entries under a range heading (§1.6–1.7, §4.15–4.16, §4.19–4.22 and
> the other section-4 families), so a single number cited from here may land inside a merged
> entry there. It never lands on a *different* metric.
>
> To rebuild the PDF after editing the HTML:
> ```
> chrome --headless --disable-gpu --virtual-time-budget=60000 \
>   --print-to-pdf="docs/BARCODE Volumetric Metrics.pdf" --no-pdf-header-footer \
>   docs/_assets/volumetric_metrics.html
> ```
> MathJax is vendored at `docs/_assets/mathjax-tex-svg.js`, so rendering needs no network.
> Note that `\boldsymbol` is **not** available in that build and fails silently-but-fatally
> (the page renders with all maths as raw TeX); use `\mathbf` instead, and escape any literal
> `<` in maths as `&lt;` or the HTML parser will eat the rest of the document.

---

## Contents

- [Description](#description)
- [Scope — which mode emits which metrics](#scope--which-mode-emits-which-metrics)
- [Column index — look up what you have](#column-index--look-up-what-you-have)
- [Measurement preconditions](#measurement-preconditions)
  — including [P.3 Segmentation input](#p3-segmentation-replaces-thresholding--and-fails-loudly)
- [1. Image Binarization (3D) branch](#1-image-binarization-3d-branch)
- [2. Intensity Distribution branch](#2-intensity-distribution-branch)
- [3. Optical Flow (3D) branch](#3-optical-flow-3d-branch)
- [4. Volumetric branch](#4-volumetric-branch)
  - [Runs without a segmentation](#runs-without-a-segmentation) — surface mesh (4.1–4.11),
    curvature (4.12–4.18), object size distribution (4.19–4.22), depth profile (4.23–4.28)
  - [Requires a segmentation](#requires-a-segmentation) — in-mask intensity (4.29–4.37),
    packing topology (4.38–4.44)
  - [Per object](#per-object) — object rows (4.45)
  - [Recording what was analyzed](#recording-what-was-analyzed) — range provenance (4.46)
- [`xyz` mode — reading a stack down its depth axis](#xyz-mode--reading-a-stack-down-its-depth-axis)
- [Row axes — what one row is](#row-axes--what-one-row-is)
- [Reading the outputs](#reading-the-outputs)
- [Warnings and flags](#warnings-and-flags)
- [Configuration reference](#configuration-reference)
- [Running a volumetric analysis](#running-a-volumetric-analysis)
- [How the numbers were checked](#how-the-numbers-were-checked)
- [Known limitations and caveats](#known-limitations-and-caveats)

---

## Description

The volumetric pipeline analyzes a `(Z, Y, X)` volume as the unit of measurement, and a
time series of them as the progression axis.

It exists because the 2D pipeline was designed for 2D data and, reasonably, assumes it. Hand
it a Z-stack and nothing announces the mismatch: `iio.imread` returns three axes, the 2D
reader appends a channel axis, and 54 focal planes are analyzed as 54 timepoints — optical
flow between focal planes reported in µm/s, "island growth over time" that is really growth
along depth. The numbers look ordinary, which is what makes the case worth handling
explicitly. The volumetric reader therefore declines to guess an axis order (§P.1) rather
than inferring one.

**The pipeline also accepts a segmentation**, alongside the intensity thresholding it
inherits. When a mask resolves it takes over as the definition of "material", and four metric
families (mesh, curvature, in-mask intensity, packing topology) are possible because of it.
See the summary at the top of this document and §P.3.

Three conventions run through every branch and are worth internalizing before reading any
individual entry:

**One value per analyzed timepoint, then an unweighted finite-only mean.** Every family
computes a scalar per analyzed timepoint and reduces with a mean that drops non-finite
entries rather than propagating them. A timepoint that produced NaN silently leaves the
average rather than poisoning it — which also means a metric can be finite while most
timepoints failed.

**Fractions are labeled `fraction of FOV` and valued 0–1.** No column in this document is a
percentage in the 0–100 sense.

**Unit labels are configurable; this document writes the defaults.** `length_units`
(nm / µm / mm) and `time_units` (s / min / hr) re-derive every label — length, area, volume,
speed, curvature, rate and the two intensity densities. Every "µm" below therefore means
*the configured length unit*. The numbers are unaffected; only the header changes.

**Single-timepoint runs report NaN for every Change metric, not 0.** The 2D code compares a
frame with itself and returns 0 (islands) or 1.0 (ratios); the volumetric branches return
NaN. A single volume is the *normal* case here, so this matters constantly. Everything
else stays live.

---

## Scope — which mode emits which metrics

`core/modes.py` names three modes. They are not an option list but two orthogonal
properties: **spatial_dims** (is the analyzed unit a plane or a volume?) and
**progression** (along which axis do things change?).

| | `xyt` | `xyz` | `xyzt` |
|---|---|---|---|
| Label | xy over time (2D) | xy over z (2D slice-wise) | xyz over time (3D volumetric) |
| Spatial dims | 2 | 2 | **3** |
| Progression axis | time | **depth** | time |
| One step is a | frame | slice | timepoint |
| Required axis | `T` | `Z` | `Z` |
| Optical flow | yes | **disabled** | yes |
| Meshing | no | no | yes |
| Columns, all families on | 36 | 29 | **66** |

Counts include the three identity columns (`File`, `Channel`, `Flags`).

**These are ceilings, not the width of a typical run.** A column set only reaches 66 with every
optional family turned on *and* carrying data. **Any family that produces no data is dropped
from the CSV, not written as a block of NaN**, so the actual width is a property of the run:
a `xyzt` run with no meshing and no segmentation emits far fewer columns, and a static z-stack
also drops its seven flow columns (see below). The barcode is drawn from whatever columns the
CSV actually carries — so if a family you expected is missing, it is because nothing populated
it, not because the schema forgot it.

**`xyz` disables flow deliberately.** Displacement between adjacent focal planes is µm of
structural shift per µm of depth — it is not motion, so a velocity would be meaningless.
The seven flow columns are **omitted from the schema entirely**, not written as NaN.

> **[`xyz` has its own section](#xyz-mode--reading-a-stack-down-its-depth-axis) below.** It covers what changes when
> the third axis is depth rather than time — the reductions, what `(over Z)` actually compares,
> masked behavior, per-slice barcodes, and what these in-plane numbers cannot tell you. This
> document remains the definition of each metric.

**`xyz` renames five metrics to `… (over Z)`.** Maximum Island/Void Area Change, and the
three Skewness/Kurtosis Change metrics. For an **unmasked** run the arithmetic is
bit-identical to the `xyt` versions — only the axis the array index runs along changes, so
"the deepest 5% of slices minus the shallowest 5%" replaces "last 5% of frames minus first
5%". The rename *is* the safety mechanism: a depth trend read as a time trend is a silent
scientific error.

> **With a mask the progression axis is narrower.** Slices the segmentation does not occupy
> are dropped, so the change metrics compare the first and last *occupied* slices rather than
> the first and last of the stack. This is a correction, not a refinement: an empty slice has
> mean 0, hence threshold 0, hence the whole field marked as a single island — and empty
> slices are the norm above and below a nuclear mask. An entirely empty segmentation raises
> rather than producing those numbers.

> **Three different counts of "families" appear in this document, and all three are right.**
> **Ten** is the metric total's denominator: the three branches of §1–§3, plus the seven
> column-producing families of §4 (mesh, curvature, object size distribution, depth profile,
> in-mask intensity, packing topology, range provenance). **Eight** is the number that are
> *optional* — the seven in `OPTIONAL_FAMILIES` that produce columns, plus intensity
> magnitude, which lives inside branch 2. **Six** is how many of those are gated to a
> volumetric mode. Object rows (§4.45) are counted in none of them, being a separate schema
> rather than a block of columns.

**Only `xyzt` computes the six volumetric-only families.** Mesh, curvature range, packing,
slice profile, in-mask intensity and component statistics are hard-gated to
`spatial_dims == 3`, because only `analysis/volumetric/run.py` computes them — the `xyz`
slice-wise path analyzes planes with the unmodified 2D branches and produces none of them.
The schema cannot promise columns nothing fills. Intensity magnitude and range provenance
are exempt, being post-processing of values every branch already has.

**Default-on vs opt-in.** **Every optional family is off by default**, including mesh —
`mesh_enabled` is `False` unconditionally. What is special about mesh is that the *schema*
advertises its columns for `xyzt` (the mode declares `supports_mesh`), so the family is
listed as available where the other seven are not even offered. Whether the columns are
filled still depends on the switch and on a segmentation resolving. A family is emitted
only when at least one row actually carries data for it, so a run that skipped meshing does
not produce a block of empty columns.

> **"Parse All Channels" does nothing in the volumetric modes.** They analyze
> `channels.selected_channel` alone, because `core/pipeline.py` routes them before channel
> enumeration ever runs. Ticking the box yields a single row for one channel with no
> indication the rest were dropped; a notice is printed. Run the other channels
> separately by changing *Choose Channel*.

Run `python scripts/run_barcode.py <path> --mode xyz --list-metrics` to print exactly what a
given configuration will emit, and which of those the barcode will render.

### A separate output shape: the per-slice barcode

`scripts/run_xyz_slice_barcodes.py` writes one barcode **per timepoint** whose **rows are
z-slices** and whose columns are the `xyz` metric set. Reading down a column shows how a
metric varies with depth. This is the counterpart to `xyz` mode, which collapses depth to a
single row; here depth is preserved. Change columns are constant-NaN in this layout — a row
is one slice, and a change needs two points on the progression axis — so the depth trend
appears as the gradient of the *non*-change columns.

---

## Column index — look up what you have

Every column BARCODE can emit, in every mode and both unit systems, with the section that
defines it. Use this when you are holding a CSV rather than reading front to back.

Two things this makes visible that the per-metric entries do not. **The same measurement
changes name between modes** — `Maximum Island Area` in `xyt`/`xyz` is `Maximum Island
Volume` in `xyzt`, and the five Change metrics gain an `(over Z)` suffix in `xyz` because
their progression axis is depth, not time. And **every size metric appears twice**: a bare
name holding a fraction of the analyzed field, and a `… Quantity` name holding a physical
size in µm² or µm³. A given CSV carries one set or the other, never both.

The table lists **metrics** only, ordered by the section that defines them — *not* by their
order in the CSV, which follows the schema. Every CSV also opens with three identity columns
— `File`, `Channel` and `Flags` — described under
[Warnings and flags](#warnings-and-flags).

> **A dash in the Unit column is the emitted unit, and it is not always the same thing as
> "dimensionless".** `Island Volume SD`, `Island Volume Skewness` and `Median Island Volume`
> (§4.20–§4.22) are fractions of the analyzed field, exactly like `Mean Island Volume`, but
> are emitted with no unit label where that one carries `fraction of FOV`. The values are
> directly comparable; what differs is the **barcode color scale**, since that is chosen by
> unit — a `fraction of FOV` column gets a fixed `[0, 1]` range and a dimensionless one is
> scaled to the data (see [Reading the barcode PNG](#reading-the-barcode-png)). Read those
> three off the CSV rather than the picture when comparing across figures.

Rows marked **object rows** belong to the separate per-object schema (§4.45), written to
`<name> Objects.csv` when the row axis resolves to `object`. Besides `Object Volume`,
`Anisotropy` and `Contact Number` (tagged **object rows** below), that schema **reuses**
several field-row headers for a *different, per-object* quantity: five mesh/curvature headers
(`Mesh Surface Area`, `Sphericity`, `Solidity`, `Lateral/Axial Ratio`, `Mean Curvature <H>`) —
always in the object schema but `NaN` unless `object_mesh` is on, each meshed from the object
alone rather than the largest component — and, **only when `--mask-intensity` is on**, the seven
`In-Mask …` statistics. So if you hold an `Objects.csv`, look those up in §4.45, **not** the
field entries the section links below point to. Its identity columns are `File`, `FOV` and
`Object` rather than `File`, `Channel`, `Flags`.

| Column in the CSV | Modes | Unit | Section |
|---|---|---|---|
| `Connectivity` | xyt, xyz, xyzt | fraction of frames | §1.1 |
| `Maximum Island Area` | xyt, xyz | fraction of FOV | §1.2 |
| `Maximum Island Area Quantity` | xyt, xyz | μm^2 | §1.2 |
| `Maximum Island Volume` | xyzt | fraction of FOV | §1.2 |
| `Maximum Island Volume Quantity` | xyzt | μm^3 | §1.2 |
| `Maximum Void Area` | xyt, xyz | fraction of FOV | §1.3 |
| `Maximum Void Area Quantity` | xyt, xyz | μm^2 | §1.3 |
| `Maximum Void Volume` | xyzt | fraction of FOV | §1.3 |
| `Maximum Void Volume Quantity` | xyzt | μm^3 | §1.3 |
| `Maximum Island Area Change` | xyt | ratio to initial | §1.4 |
| `Maximum Island Area Change (over Z)` | xyz | ratio to initial | §1.4 |
| `Maximum Island Volume Change` | xyzt | ratio to initial | §1.4 |
| `Maximum Void Area Change` | xyt | ratio to initial | §1.5 |
| `Maximum Void Area Change (over Z)` | xyz | ratio to initial | §1.5 |
| `Maximum Void Volume Change` | xyzt | ratio to initial | §1.5 |
| `Initial Maximum Island Area` | xyt, xyz | fraction of FOV | §1.6 |
| `Initial Maximum Island Area Quantity` | xyt, xyz | μm^2 | §1.6 |
| `Initial Maximum Island Volume` | xyzt | fraction of FOV | §1.6 |
| `Initial Maximum Island Volume Quantity` | xyzt | μm^3 | §1.6 |
| `Initial 2nd Maximum Island Area` | xyt, xyz | fraction of FOV | §1.7 |
| `Initial 2nd Maximum Island Area Quantity` | xyt, xyz | μm^2 | §1.7 |
| `Initial 2nd Maximum Island Volume` | xyzt | fraction of FOV | §1.7 |
| `Initial 2nd Maximum Island Volume Quantity` | xyzt | μm^3 | §1.7 |
| `Mean Island Anisotropy` | xyt, xyz, xyzt | — | §1.8 |
| `Mean Island Area` | xyt, xyz | fraction of FOV | §1.9 |
| `Mean Island Area Quantity` | xyt, xyz | μm^2 | §1.9 |
| `Mean Island Volume` | xyzt | fraction of FOV | §1.9 |
| `Mean Island Volume Quantity` | xyzt | μm^3 | §1.9 |
| `Total Island Area` | xyt, xyz | fraction of FOV | §1.10 |
| `Total Island Area Quantity` | xyt, xyz | μm^2 | §1.10 |
| `Total Island Volume` | xyzt | fraction of FOV | §1.10 |
| `Total Island Volume Quantity` | xyzt | μm^3 | §1.10 |
| `Mean Island Separation` | xyt, xyz, xyzt | μm | §1.11 |
| `Structural Correlation Length` | xyt, xyz, xyzt | μm | §1.12 |
| `Maximum Kurtosis` | xyt, xyz, xyzt | — | §2.1 |
| `Maximum Median Skewness` | xyt, xyz, xyzt | — | §2.2 |
| `Maximum Mode Skewness` | xyt, xyz, xyzt | — | §2.3 |
| `Kurtosis Change` | xyt, xyzt | — | §2.4 |
| `Kurtosis Change (over Z)` | xyz | — | §2.4 |
| `Median Skewness Change` | xyt, xyzt | — | §2.5 |
| `Median Skewness Change (over Z)` | xyz | — | §2.5 |
| `Mode Skewness Change` | xyt, xyzt | — | §2.6 |
| `Mode Skewness Change (over Z)` | xyz | — | §2.6 |
| `Total Intensity` | xyt, xyz, xyzt | a.u. | §2.7 |
| `Mean Intensity` | xyt, xyz, xyzt | a.u. | §2.8 |
| `Intensity SD` | xyt, xyz, xyzt | a.u. | §2.9 |
| `Intensity Density (per area)` | xyt, xyz | a.u./μm^2 | §2.10 |
| `Intensity Density (per volume)` | xyzt | a.u./μm^3 | §2.10 |
| `Speed` | xyt, xyzt | μm/s | §3.3 |
| `Speed Change` | xyt, xyzt | μm/s | §3.4 |
| `Mean Flow Direction` | xyt, xyzt | rads | §3.5 |
| `Directional Spread` | xyt, xyzt | rads | §3.6 |
| `Velocity Correlation Length` | xyt, xyzt | μm | §3.7 |
| `Divergence` | xyt, xyzt | 1/s | §3.8 |
| `Curl` | xyt, xyzt | 1/s | §3.9 |
| `Mesh Volume` | xyzt | μm^3 | §4.2 |
| `Mesh Surface Area` | xyzt | μm^2 | §4.3 |
| `Sphericity` | xyzt | — | §4.4 |
| `Equivalent Sphere Radius` | xyzt | μm | §4.5 |
| `Mesh Height` | xyzt | μm | §4.6 |
| `Lateral/Axial Ratio` | xyzt | — | §4.7 |
| `Mesh Volume Ratio` | xyzt | — | §4.8 |
| `Solidity` | xyzt | — | §4.9 |
| `Mean Curvature <H>` | xyzt | 1/μm | §4.14 |
| `Invagination Ratio` | xyzt | — | §4.15 |
| `Concave Area Fraction` | xyzt | — | §4.16 |
| `Minimum Curvature` | xyzt | 1/μm | §4.17 |
| `Maximum Curvature` | xyzt | 1/μm | §4.18 |
| `Island Count` | xyzt | — | §4.19 |
| `Island Volume SD` | xyzt | — | §4.20 |
| `Island Volume Skewness` | xyzt | — | §4.21 |
| `Median Island Volume` | xyzt | — | §4.22 |
| `Maximal Area Slice Index` | xyzt | slice | §4.24 |
| `Maximal Area Slice Depth` | xyzt | μm | §4.25 |
| `Maximal Area Slice Area` | xyzt | fraction of FOV | §4.26 |
| `In-Mask MFI` | xyzt | a.u. | §4.31 |
| `In-Mask Intensity SD` | xyzt | a.u. | §4.32 |
| `In-Mask Intensity CV` | xyzt | — | §4.33 |
| `In-Mask Intensity Skewness` | xyzt | — | §4.34 |
| `In-Mask Intensity Entropy` | xyzt | — | §4.35 |
| `In-Mask Normalized Entropy` | xyzt | — | §4.36 |
| `In-Mask Fraction Above 2x Median` | xyzt | — | §4.37 |
| `Mean Contact Number` | xyzt | — | §4.41 |
| `Contact Number SD` | xyzt | — | §4.42 |
| `Hexagonal Fraction` | xyzt | — | §4.43 |
| `Z Range Start` | xyt, xyz, xyzt | slice | §4.46 |
| `Z Range End` | xyt, xyz, xyzt | slice | §4.46 |
| `T Range Start` | xyt, xyz, xyzt | slice | §4.46 |
| `T Range End` | xyt, xyz, xyzt | slice | §4.46 |
| `Object Volume` | object rows | μm^3 | §4.45 |
| `Anisotropy` | object rows | — | §4.45 |
| `Contact Number` | object rows | — | §4.45 |

---

## Measurement preconditions

Every number in this document is conditioned on the steps below. A reader who does not know
which of them fired cannot interpret any metric, so they are documented here as part of the
measurement rather than as implementation detail.

### P.1 Axes are declared, never inferred

`analysis/volumetric/reader.py` accepts **`.tif` / `.tiff` and `.nd2`**, rejecting anything
else with a clear message rather than failing obscurely later. For ND2 it reads the axes,
voxel size and acquisition time loop directly — lazily, via dask — and rejects multi-point
`P` files; that time loop is the one timing source it trusts (P.2). Note the 2D reader skips
Z-stack ND2s, so a volumetric ND2 is readable only by this path.

For TIFF it takes `tifffile`'s declared axis order and **raises** on
anything it cannot interpret: the placeholder axes `I`, `Q`, `S`; anything outside `TZCYX`;
a missing `Y` or `X`; and a missing `Z` (with a message pointing at the 2D pipeline). It
never falls back on shape heuristics.

`axes_override` accepts being *told* the true order, one letter per data dimension, and
replaces the declared order outright. The motivating case is acquisition software writing a
time series into ImageJ's `channels` field, so the file declares `ZCYX` for data that is
really `TZYX`. The reinterpretation is printed and the original retained as
`declared_axes` for provenance. **BARCODE never infers this — it only accepts being told.**

### P.2 Voxel size decides every physical unit

Precedence is **explicit setting → file metadata → warn and default to 1.0 µm**:

- z step ← `z_step_um`, else ImageJ `spacing` **converted through the file's stated unit**;
- xy step ← `xy_step_um`, else `XResolution` — a *pixels-per-unit* tag, so the step is
  `µm_per_unit / px_per_unit`.

Both print a warning when they fall through to 1.0, because silently assuming a voxel size
makes every physical metric wrong by an unknown factor.

> **The resolution tags are read through a unit name, not assumed to be microns.** The unit
> comes from ImageJ's `unit` field, else the TIFF `ResolutionUnit`. A tag naming a unit
> BARCODE does not recognize, or one explicitly declaring the image *uncalibrated*
> (`pixel`, `px`, `none`, `a.u.`), is **discarded** — it warns and falls back to 1.0 rather
> than treating a pixel count as microns. Only a wholly *absent* unit falls back to the old
> "assume microns" reading.

**`frame_interval_s` is different and must be set by hand.** It defaults to 0, and the
fallback is narrower than it looks: the file's value is used **only** where the reader
positively trusts the file's timing. An **ND2 acquisition time loop** qualifies; an ImageJ
`finterval` on a per-timepoint TIFF export does not — there it frequently describes the *z*
acquisition or is simply left at 1, and nothing in the file distinguishes the two. Otherwise
it returns **1.0 outright**, and says so.

Most importantly, **a grouped time-lapse never uses the file's timing at all.** Assembling
per-file volumes into a series marks the timing untrusted, so unless you set Frame Interval
the whole series runs at 1.0 s — in precisely the workflow where per-file exports make
`finterval` meaningless. Speed and Speed Change scale inversely with it and nothing else
changes, so the error is invisible in the output.

### P.3 Segmentation replaces thresholding — and fails loudly

**This is new in the volumetric pipeline; published BARCODE has no segmentation input at
all.** Set `segmentation_enabled` and point the resolver at your masks. Supported formats are
`.tif`/`.tiff`, `.png`/`.bmp`, `.npy` and `.npz`, dispatched on the suffix
(`mask_format="auto"`), and a `.npy` may be either a bare label array or a **Cellpose
`_seg.npy` bundle**, unwrapped via its `masks`/`mask`/`labels` key.

A second, independent mask can be supplied alongside the first
(`segmentation_secondary_root` / `_template`) so that, for example, a nucleus and its cell
are both available for the same image.

A silently misaligned mask produces confidently wrong metrics, which is worse than no mask.
So every check raises rather than degrading:

- The mask path is built by applying `segmentation_regex` to the image stem and formatting
  `segmentation_template` with its named groups. A non-matching regex, a missing capture, or
  a missing file each raise, naming what *was* captured.
- **XY dimensions must match exactly** — a different XY means a different field of view.
- **Z extent must agree within 1.0 µm**: `image_z × z_step` vs `mask_z × mask_spacing`. This
  is the check that catches a wrong `mask_spacing_um` or `z_step_um`.
- An empty mask raises.

`mask_spacing_um = 0` means **"assume isotropic at the image's xy step"** — the common case,
since masks generally carry no spacing metadata.

**Label preservation.** With `object_partition = "auto"` (the default), a mask carrying more
than one distinct positive value keeps its integer labels as `int32`. Collapsing an instance
segmentation to a boolean would re-derive objects by connectivity and **merge every touching
instance** — invisible for a single nucleus, silently wrong for a confluent field.
`invert_binarization` on a label mask raises: inverting instance labels describes nothing.

**When a mask resolves, it *is* the binarization.** `threshold_offset`,
`minimum_island_size` and intensity thresholding are bypassed entirely for the structural
branch. Intensity and autocorrelation still use raw voxels.

### P.4 Isotropic resampling and the crop box — the largest single effect on any number

> **Resampling happens with or without a segmentation.** With `make_isotropic` on (the
> default) and no mask, the image is still resampled to isotropic-at-xy from its own acquired
> spacing — so a maskless run **does** change slice count (54 → 245 on the Jurkat geometry),
> which moves `Maximal Area Slice Index`, every fraction denominator, and the spacing that
> `Intensity Density` divides by. Only `make_isotropic = False` leaves the acquired grid
> untouched. What a mask adds is the *cropping* option below, and the mask-gated families.

With `make_isotropic` on (**default**) **and a mask present**, `prepare_volume` does two
things:

1. **Canonicalize onto the mask grid** — the image is resampled to the mask's shape and
   spacing with **linear** interpolation. On the working Jurkat data this upsamples z from
   0.3 µm to 0.065 µm, a factor of ~4.6.
2. **Resample to isotropic xy spacing** — images with **linear** interpolation, masks with
   **nearest-neighbor**. The split matters: nearest-neighbor maps each output voxel to
   exactly one input voxel and so carries instance labels across unchanged, where any
   interpolating kernel would average neighboring labels into new, meaningless ones.

   Read `resample.py` alone and this looks moot, because the mask it handles is the *union*
   mask and gets binarized anyway. The label preservation happens one level up, where the
   per-timepoint masks are resampled separately as `int32` — not `uint8`, since a confluent
   field routinely has more than 255 objects and `uint8` would wrap them, merging unrelated
   cells into one label.

   Two properties of this step matter for correctness. **Per-frame masks are resampled about
   the crop origin**, not `(0,0,0)`, so with cropping on a mask stays registered to its image
   rather than being offset by the whole crop. And where the mask grid **overhangs** the
   image, the last acquired plane is repeated rather than zero-filled, so no background voxel
   is injected into a region the mask still declares foreground.

It does **not** crop. `crop_to_mask` defaults to **False**, so the analyzed volume is the
acquired field of view.

> **Why cropping is off.** Every "fraction of volume" metric is a fraction *of the analyzed
> box*. Cropping to the mask gives each file its own denominator, so the same object in a
> tighter box reports a larger fraction and a real size trend cannot be told apart from the
> box tracking the object — on the 15-timepoint Jurkat series all 15 bounding boxes differ
> (z extent swinging 177 → 124 → 132). On Cell1 the full field reads ~8.8% where the crop
> read ~39%. **The µm³ Quantity columns are unaffected either way** — compare those when in
> doubt.

Turning `crop_to_mask` on (padded by `crop_padding_vox`, default 2) restores the old
object-relative denominator. A time series is then cropped to the **union** of the per-frame
mask bounding boxes, never each to its own: per-frame cropping would give each
timepoint a different array shape *and* a different denominator, so the fraction metrics
would not be comparable across the very time axis they are meant to describe. **The intended
way to analyze less than the full field is an explicit z range.**

Meshing additionally **requires** an isotropic grid and raises otherwise.

**Which metrics depend on an isotropic grid — and what to do about each.** Every column falls
into exactly one of three groups.

#### Group 1 — will not be produced at all without a cubic grid (16 columns)

| Family | Columns | What you see if the grid is not cubic |
|---|---|---|
| Surface mesh | `Mesh Volume`, `Mesh Surface Area`, `Sphericity`, `Equivalent Sphere Radius`, `Mesh Height`, `Lateral/Axial Ratio`, `Mesh Volume Ratio`, `Solidity` | columns absent; log prints `Meshing failed: …` |
| Curvature | `Mean Curvature <H>`, `Invagination Ratio`, `Concave Area Fraction`, `Minimum Curvature`, `Maximum Curvature` | columns absent — curvature rides on the mesh |
| Packing | `Mean Contact Number`, `Contact Number SD`, `Hexagonal Fraction`, and per-object `Contact Number` | columns absent; log prints `packing topology is not defined on a N.Nx anisotropic grid` |

> **The fix: leave *Resample to Isotropic Voxels* (`make_isotropic`) ticked — it is on by
> default.** These sixteen columns are the entire reason it exists. If they are missing from
> your CSV, this is the first thing to check.

#### Group 2 — correct on any grid; nothing to do

`Mean Island Anisotropy` (inertia tensor built with physical spacing), `Mean Island
Separation` (centroids scaled before distances), **both** correlation lengths (`Structural`
and `Velocity` — each radially binned on *physical* distance, not voxel index), `Speed`,
`Divergence`, `Curl`, `Intensity Density`, `Maximal Area Slice Depth`, and **every volume**
(a voxel count times the true voxel volume). These read the real spacing and need no
resampling to be right.

#### Group 3 — they run either way, but the number moves with the grid

The ones to watch, because nothing warns you:

| Column | Why it moves | What to do |
|---|---|---|
| `Connectivity` | labeled on the **index lattice** with no spacing, so resampling creates and destroys connections | keep `make_isotropic` identical across every run you intend to compare |
| `Maximal Area Slice Index` | a slice **index** — 54 acquired planes versus 245 isotropic ones | read `Maximal Area Slice Depth` (µm) instead; it is grid-independent |
| `Speed`, `Divergence`, `Curl` | the output is physically scaled, but `flow_xyz_sigma`, `flow_w_sigma` and `flow_downsample` are in **voxels**, so the window spans 0.3 µm per step in z and 0.065 µm in xy | resample, so the window is physically isotropic too |
| every intensity statistic — `Maximum Kurtosis`, both skewnesses, `Total`/`Mean Intensity`, `Intensity SD`, all seven `In-Mask` columns | resampling interpolates **linearly**, which smooths voxel values | keep `make_isotropic` fixed across a study; never compare a resampled run against an unresampled one |

> **The last row is the trap.** Intensity is not a geometric quantity, so it looks like
> resampling cannot possibly touch it — but interpolation changes the voxel values the
> histogram is built from. Same data, same settings, different numbers. This is exactly why
> the dim-data flag is judged on the **acquired** volume, before any resampling.

With `make_isotropic` on (the default), every family sees a cubic grid and Groups 1 and 3
stop mattering. They matter when you turn it off, or work deliberately on an anisotropic grid.

### P.5 Range restriction, and the order it is applied in

`z_start`/`z_end` and `t_start`/`t_end` restrict what is analyzed. `end = 0` always means
"to the end"; negatives count back from the end as in Python slicing; **an empty or reversed
range raises** rather than producing a zero-length stack that surfaces later as an
unexplained NaN.

> **The range includes both ends.** `z_start = 12`, `z_end = 46` analyzes slices 12 through
> **46** — 35 slices, with slice 46 included. `z_start = z_end` is therefore a single slice,
> not an empty range.
>
> `end = 0` still means "to the end", so the `(0, 0)` default is the whole axis and existing
> configuration files keep their meaning. Negative values count back from the end, with `-1`
> the last slice.
>

`z_range_units` matters because "slice 46" is ambiguous on anisotropic data — the acquired
stack and the isotropic grid a mask lives on have very different slice counts (54 vs 245 on
0.3/0.065 µm data). The three units are `acquired` (default), `isotropic`, and `microns` of
depth from the bottom. `t_range_units` is `index` or `seconds`.

**Both volumetric paths take the t range first** — it is the cheapest cut, since excluded
volumes then get no mask or geometry work — but they differ in how they keep the mask
aligned, to the same end:

- **`xyz` and the per-slice path** load the mask against the **full acquired stack**, then
  apply the z range, then slice the mask by the same indices. Validating a whole-depth mask
  against an already-restricted image would reject a good mask.
- **`xyzt`** applies the z range first, validates the mask against the recorded *acquired*
  depth, then maps the range onto the mask's own grid by `round(i·(m_z−1)/(n_z−1))` when the
  two differ in depth and slices the mask to match. This keeps a finer-grid mask (e.g. 250
  planes against a 54-plane image) registered to the restricted sub-range rather than to the
  full stack.

Any restriction raises **flag digit 5**.

### P.6 Which timepoints are analyzed

`select_frame_indices(n, frame_step)`: `[0]` when `n ≤ 1`; otherwise the step is integer-
divided by 5 until it is below `n`, indices are `range(0, n, step)`, and the last index is
always appended. Default `frame_step` **10**.

This is a separate selector from the 2D helper rather than a shared one, because the two face
different inputs: a single volume is the ordinary case here, and `frame_step ≥ frame count`
is routine when one file is one timepoint, where in a 2D movie it rarely arises. So this
version divides by *integer* division and handles `n == 1` explicitly. (On the 2D path those
two inputs give a float step and a `TypeError` from `range()`, which is caught per-branch and
leaves the affected rows blank — worth knowing if you run a Z-stack through it, and the
reason `frame_step` is worth setting deliberately in either pipeline.)

**The flow branch does not use this list the way the others do** — see [3.0](#30-the-flow-window-and-why-it-costs-you-the-first-and-last-three-timepoints).

### P.7 Time-lapse assembly, and file ordering

Volumetric time series are frequently exported one file per timepoint. Analyzed
individually, each gives an independent row with every Change metric NaN. `timelapse_enabled`
groups them with `timelapse_regex`, which needs a `series` group (files sharing it belong
together) and a numeric `frame` group (ordering within the series). A series is keyed by
**directory *and* matched name**, so identically-numbered files in two condition folders are
never merged into one series — `find_files` walks recursively, and the same naming convention
appearing in sibling folders is the normal case rather than an exotic one. **Grouping applies to
`xyzt` only** — it is ignored in the other two modes.

All files in a series must share both shape and spacing — a mismatch raises rather than
padding or cropping. Duplicate frame numbers raise, because that means the regex is matching
more files than intended. **Gaps in the frame numbering also raise**: the time axis is assumed
evenly sampled, so a missing timepoint would silently mis-scale Speed and shift the flow
window. Non-matching files are **reported, never silently dropped**.

**File ordering is natural-sorted in the volumetric modes.** In these modes one file is one
timepoint, so the file list *is* the barcode's vertical axis; lexicographic sorting would
give `Cell1_1, Cell1_10, Cell1_11, … Cell1_2` and produce a barcode that looks entirely
plausible while showing the time course out of order. `analysis/volumetric/ordering.py`
sorts digit runs numerically. It is kept out of `utils/setup.py` deliberately: changing the
sort there would reorder the rows of the published 2D reference CSVs.

---

## 1. Image Binarization (3D) branch

The structural branch. Twelve columns, mirroring the published 1.1–1.11 plus the
initial-second-island split, but measuring **volumes with 26-connectivity** rather than
areas with 8-connectivity.

**Implementation:** `analysis/volumetric/binarization.py` · **Enabled by:** the *Image
Binarization* branch checkbox.

Throughout this section, a **"frame" means a whole timepoint** — an entire Z-stack — not a
z-slice.

### 1.0 Producing the binary volume

**With a segmentation** the mask *is* the binary volume (`mask.astype(bool)`), and if it is
an integer instance labeling with more than one positive value, that labeling is
authoritative and is used directly as the object partition. `threshold_offset`,
`minimum_island_size` and `invert_binarization` are not applied at this stage.

**Without one**, per timepoint:

0. **A constant volume returns an all-empty binary.** A blank or saturated-flat acquisition
   has `ptp = 0`, and the mean-relative rule would then give a threshold of 0 and mark
   *every* voxel as foreground — reporting one island filling the field and `Connectivity`
   = 1, indistinguishable in the CSV from a genuinely percolating sample.
1. `threshold = mean(volume) × (1 + threshold_offset)`, default offset **0.1**.
2. `binary = volume >= threshold` (inclusive).
3. `remove_small_objects(binary, minimum_island_size + 1, connectivity=3)` then
   `remove_small_holes(...)` with the same arguments. `connectivity=3` in 3D is
   **26-connectivity**. With the default `minimum_island_size = 1`, objects and holes
   **smaller than 2 voxels** are removed — i.e. isolated single voxels.
4. Optionally inverted if `invert_binarization` is set.

**There is no `bin_factor` in 3D.** The 2D branch block-averages by `bin_factor` (default 2)
and divides its FOV denominator by `bin_factor²`; the 3D structural branch never bins. (A
`group_avg_3d` helper exists in this module but is used only by the flow branch.)

### 1.1 Connectivity (C)

**Column:** `Connectivity` · **Unit:** `fraction of frames`, valued 0–1 ·
**Field:** `binarization.connectivity`

Fraction of analyzed timepoints in which a single connected object percolates the volume.

1. Label the binary volume with a full 3×3×3 structuring element (**26-connectivity**).
2. For **each of the three axes** independently, take face 0 and face −1, drop label 0, and
   check whether any single label appears on both.
3. Score 1 if any axis percolates, else 0. Zero objects scores 0.
4. Mean over analyzed timepoints.

**vs 2D:** the 2D metric uses 8-connectivity and checks only two axes (rows and columns).
The 3D version adds the z axis, so an object spanning depth but not the lateral field now
scores 1 where the 2D metric would score 0. Never NaN in either.

### 1.2 Maximum Island Volume (I)

**Column:** `Maximum Island Volume` · **Unit:** `fraction of FOV`, valued 0–1 ·
**Field:** `binarization.max_island_size`

Typical size of the single largest object, as a fraction of the analyzed volume.

1. Per timepoint, label with 26-connectivity and take the largest region's **voxel count**.
2. Drop non-finite values, then sort descending and average the top `⌈0.1 × n⌉` (at least
   one). The 10% fraction is **hardcoded**, not a setting.
3. Divide by the total voxel count of the analyzed field.

**Physical variant:** `Maximum Island Volume Quantity`, unit `μm^3` — the same top-10% mean
of voxel counts multiplied by the **full voxel volume** `dz·dy·dx`.

**NaN:** when every analyzed timepoint had zero objects.

**vs 2D — the physical variant is not the same quantity.** The 2D "Area Quantity" multiplies
a pixel count by `um_pixel_ratio` **once**, so it carries units of µm·px; the 3D version
multiplies by the full voxel volume and is in µm³. The two therefore cannot be pooled or
compared, whatever the column names suggest. Objects here are also 26- rather than
8-connected, and the denominator is the analyzed field, which is the acquired field unless
`crop_to_mask` is on.

### 1.3 Maximum Void Volume (V)

**Column:** `Maximum Void Volume` · **Unit:** `fraction of FOV`, valued 0–1 ·
**Field:** `binarization.max_void_size`

Mirror of 1.2 on the inverted volume: label the background with 26-connectivity, take the
largest region's voxel count, top-10% mean, divide by the field.

**Physical variant:** `Maximum Void Volume Quantity`, `μm^3`.

**Edge case, where the two branches differ.** If the foreground fills the volume so there is
no background region at all, the 3D branch reports the largest void as **0**, reading the
edge case as "no void exists"; the 2D branch returns the whole frame. Both are defensible
readings of an empty set, but they sit at opposite ends of the range, so the 2D and 3D
columns are not comparable on a filled field.

### 1.4 Maximum Island Volume Change (ΔI)

**Column:** `Maximum Island Volume Change` · **Unit:** `ratio to initial` ·
**Field:** `binarization.max_island_percent_change`

Ratio of the largest-island size at the end of the series to that at the start; 1.0 means no
change. Computed on **raw voxel counts on both sides**, so the ratio is scale-free.

1. `n_eval = max(⌈percentage_frames_evaluated × n⌉, 1)`, default fraction **0.05**.
2. `mean(island[−n_eval:]) / mean(island[:n_eval])`.

**No physical variant** — the same column appears in both CSVs.

**NaN when `n < 2`,** deliberately: with one timepoint the comparison is a value against
itself. **Also NaN when the initial window is zero or non-finite** — something that grew from
nothing has no ratio.

**vs 2D:** two differences at the edges of the parameter range. The 3D window is floored at
one frame, so `percentage_frames_evaluated = 0` still evaluates a frame here where 2D
evaluates an empty window and returns NaN; and on a single frame 2D returns exactly 1.0 —
the honest answer to "how much did it change" for a series compared with itself — where 3D
returns NaN, because a single volume is the *usual* input here rather than a degenerate one.

### 1.5 Maximum Void Volume Change (ΔV)

**Column:** `Maximum Void Volume Change` · **Unit:** `ratio to initial` ·
**Field:** `binarization.max_void_percent_change`

Identical to 1.4 applied to the void series, including the single-timepoint NaN rule.

### 1.6 Initial Maximum Island Volume (I₀,₁)

**Column:** `Initial Maximum Island Volume` · **Unit:** `fraction of FOV`, valued 0–1 ·
**Field:** `binarization.island_size_initial`

Largest-island size averaged over the **first** `n_eval` analyzed timepoints, then divided by
the field volume. **Physical variant:** `Initial Maximum Island Volume Quantity`, `μm^3`.

Note this is **not NaN for a single timepoint** — it is simply that timepoint's value.

### 1.7 Initial 2nd Maximum Island Volume (I₀,₂)

**Column:** `Initial 2nd Maximum Island Volume` · **Unit:** `fraction of FOV`, valued 0–1 ·
**Field:** `binarization.island_size_initial2`

Second-largest object's size over the same initial window. With 1.6 it says whether the field
is dominated by one object.

**When only one object exists this is `0.0`, not NaN** — a meaningful statement, not a missing
value. A single-nucleus run therefore reports 0 here. **Physical variant:**
`Initial 2nd Maximum Island Volume Quantity`, `μm^3`.

### 1.8 Mean Island Anisotropy (A_I)

**Column:** `Mean Island Anisotropy` · **Unit:** dimensionless, ≥ 1 ·
**Field:** `binarization.island_anisotropy`

Mean elongation of objects: the major/minor axis ratio of the equivalent ellipsoid, averaged
over all objects in a timepoint and then over timepoints.

> **Measured in physical coordinates, so it describes the object and not the sampling
> grid.** The inertia tensor is built via a dedicated `regionprops_table` pass with
> `spacing=` — its own pass because `spacing` rescales every property in the call, and
> `area` and `centroid` are already converted downstream, so folding it in would
> double-count both. A physically spherical object therefore reads ≈1 on any grid,
> isotropic or not, where an index-space tensor would report the grid's own aspect ratio.

skimage's `axis_major_length` / `axis_minor_length` **raise `ValueError: math domain error`**
on flat or thin 3D regions — on a real Jurkat nucleus, **135 of 167** islands were
degenerate. So the branch reimplements them from the inertia tensor eigenvalues with a
clamp:

```
ev    = eigenvalues sorted descending
major = sqrt(10 · max(ev0 + ev1 − ev2, 0))
minor = sqrt(10 · max(−ev0 + ev1 + ev2, 0))
ratio = major / minor   where minor > 1e-6, else NaN
```

Degenerate regions are **excluded** from the mean, not zeroed.

**NaN:** when every region in every timepoint is degenerate, or there are no objects.

**vs 2D — not directly comparable.** This is a 3D ellipsoid major/minor ratio from clamped
inertia eigenvalues; the 2D metric is a 2D ellipse ratio from skimage's own properties.

On the barcode this metric is scaled `[1, max]` rather than `[0, max]`.

### 1.9 Mean Island Volume (Ī)

**Column:** `Mean Island Volume` · **Unit:** `fraction of FOV`, valued 0–1 ·
**Field:** `binarization.mean_island_size`

Mean voxel count over all objects in a timepoint, averaged over timepoints, divided by the
field volume. **Physical variant:** `Mean Island Volume Quantity`, `μm^3`.

### 1.10 Total Island Volume (I_tot)

**Column:** `Total Island Volume` · **Unit:** `fraction of FOV`, valued 0–1 ·
**Field:** `binarization.total_island_size`

The **foreground volume fraction** of the analyzed field: sum of all object voxel counts per
timepoint, averaged, divided by the field.

**Physical variant:** `Total Island Volume Quantity`, `μm^3`. **With a segmentation this is
effectively the segmented volume in µm³** — the closest thing the branch has to a direct
nuclear volume, and the natural cross-check against `Mesh Volume` (§4.2).

### 1.11 Mean Island Separation (D_I)

**Column:** `Mean Island Separation` · **Unit:** `μm` ·
**Field:** `binarization.mean_island_separation`

Mean distance from each object's centroid to its nearest neighbors.

1. Scale centroids by `spacing_zyx`, so the tree lives in **physical** coordinates and
   anisotropy is handled correctly.
2. `k = clip(⌈neighbor_island_fraction × n⌉ − 1, 1, n−1)`, default fraction **0.1**.
3. KD-tree query for `k+1` neighbors, **drop the self-distance**, take the mean.
4. Mean over timepoints. **Already in µm** — there is no post-multiplication.

**NaN whenever a timepoint has fewer than 2 objects**, which is the *normal* case for a
single segmented nucleus. A KD-tree is used rather than the 2D branch's dense N×N distance
matrix — an entirely sound approach at 2D island counts, which 3D counts outgrow.

> **vs 2D — the two are not numerically comparable even on identical data.** The 2D
> implementation takes the `k+1` smallest values of each row of the distance matrix, which
> includes the diagonal zero, so the two branches average over a different set: a fixed
> factor of `k/(k+1)` separates them before any other difference. The 3D version excludes the
> self-distance. The 2D version also works in isotropic pixel units and multiplies by
> `um_pixel_ratio` at the end, where 3D scales each axis by its own spacing before querying —
> the anisotropy of a Z-stack leaves no equivalent single ratio to apply afterwards.

### 1.12 Structural Correlation Length (ξ_I)

**Column:** `Structural Correlation Length` · **Unit:** `μm` ·
**Field:** `binarization.island_correlation_length`

The length scale over which image structure stays correlated.

**Computed on the raw voxels, never the binary or the mask** — this holds even when a
segmentation replaced thresholding.

1. **Normalize:** `(volume − mean) / std`, zero-mean and unit-variance. A constant volume
   returns an empty profile.
2. **Autocorrelate** by Wiener–Khinchin: `real(fftshift(ifftn(F · conj(F)))) / field.size`,
   using the full 3D FFT with **no zero padding** — so this is the **circular** (periodic)
   autocorrelation, wrap-around included, exactly as in 2D.
3. **Radially average on physical distance.** Three departures from 2D, each deliberate:
   - Bins are **µm-wide**, of width `min(dz, dy, dx)`, not pixel-index-wide. Anisotropic
     voxels would otherwise distort the profile.
   - `r_max = min(nz/2·dz, ny/2·dy, nx/2·dx)` — the **smallest physical half-extent across
     all three axes**, which on anisotropic data is usually z. The 2D code's cap is a
     y-index assumption that does not carry over.
   - **Each shell is reported at its own mean radius**, not at the bin's left edge. A
     shell's population grows as r², so its mean radius sits well above the left edge;
     anchoring to the edge makes the profile look like it decays faster than it does
     (measured bias ~0.07 against the analytic Gaussian case). **Consequence: reported
     lengths are not exact multiples of the voxel size**, unlike 2D.
4. **Threshold at `1/e ≈ 0.3679`**, and select by the rule below.
5. Mean over analyzed timepoints.

#### The threshold-crossing rule — and its known divergence from the reference

```
for i in range(len(radial) - 1):
    if radial[i] > threshold and radial[i+1] <= threshold:
        return radii[i] if |radial[i] − threshold| < |radial[i+1] − threshold| else radii[i+1]
return NaN
```

Find the first downward crossing, then snap to whichever adjacent shell's correlation *value*
is closer to the threshold (ties go to `i+1`). The 2D and 3D branches use the identical rule,
so their correlation lengths are directly comparable.

> **The divergence: this is not the definition that produced the published reference CSVs.**
> Snapping to the nearer point can return a radius at which the correlation is *still above*
> the threshold. The BARCODE 2.0 release took the first radius at which `radial ≤ threshold` —
> always the outer point of the pair — so **lengths reported here are systematically smaller
> than the published ones.** This matches the current 2D branch verbatim, by design: the two
> branches agree with each other rather than with the 2.0 release. Do not pool a correlation
> length from this pipeline with one read off a published reference CSV.

**NaN when:** the volume is constant; `r_max ≤ step` (a volume too thin in its shortest
physical axis to hold two shells); or no downward crossing exists within `r_max` — the
"correlation length exceeds the field of view" case, which raises **flag digit 3**.

**Subtlety:** an empty radial bin produces NaN in the profile, and NaN comparisons are
`False`, so a gap **silently blocks** a crossing at that index and can push the reported
length outward or to NaN. This is a second, undocumented route to flag 3 beyond
"length exceeds the field of view".

---

## 2. Intensity Distribution branch

The 3D branch is the 2D branch applied to a whole volume rather than a plane: a "frame" is
one `(Z, Y, X)` timepoint, and `np.histogram` flattens it, so on the same voxel set the
arithmetic is identical. Six shape metrics (2.1–2.6) mirror the published branch. Four
further metrics (2.7–2.10) have no 2D counterpart: they are the branch's first **extensive**
quantities, scaling with the amount of material rather than describing the shape of the
histogram.

**Implementation:** `analysis/volumetric/intensity.py` · **Enabled by:** the *Intensity
Distribution* branch checkbox.

> **You can restrict this branch to a mask ROI.** With `intensity_use_mask` on, the histogram
> is built only from voxels **inside the segmentation** — so every metric below describes the
> object rather than the whole field, and the background peak that otherwise dominates a
> volumetric histogram is gone. It is **off by default**; a masked and an unmasked run measure
> different quantities and must not be pooled (§2.0). This is the intensity counterpart of the
> per-object in-mask family (§4.29–4.37); the difference is that this pools all in-mask voxels into one
> field histogram, while §4.29–4.37 reports one object at a time.

### 2.0 The histogram every shape metric is built on

Applied per analyzed timepoint, before any metric below:

1. `np.histogram(volume, bins=bin_size)` — `bin_size` default **300**. The range is the
   **data's own [min, max]**; there is no fixed range and no background exclusion.
2. Bin **centers** are taken as the abscissa.
3. Counts are normalized to probabilities, bins with `p ≤ noise_threshold` are **discarded**
   (default `5e-4`), and the survivors renormalized to sum 1. This is a *bin-population*
   filter, not an intensity filter — it drops sparsely populated bins wherever they sit.

All moments are probability-weighted sums over the surviving bin centers: mean `Σ v·p`;
standard deviation `√(Σ (v−mean)²·p)`; mode `= argmax p`; median = the first `v` whose
cumulative probability reaches 0.5.

**Optional mask restriction.** With `intensity_use_mask` on (default **off**) the histogram
is built from in-mask voxels only. For an instance mask this means *any nonzero label*, so
every object pools together. This changes the quantity rather than refining it: on Cell1_1 it
moved Maximum Kurtosis from 9.93 to −0.11. **Masked and unmasked runs must not be pooled.**

### 2.1 Maximum Kurtosis (K)

**Column:** `Maximum Kurtosis` · **Unit:** dimensionless · **Field:** `intensity.max_kurtosis`

How heavy-tailed the intensity histogram is, at the timepoints where it is most so.

1. Per timepoint, excess kurtosis `Σ(v−mean)⁴p / σ⁴ − 3`. The `−3` makes a Gaussian read 0.
2. **Drop non-finite values**, then sort descending, take the top `⌈0.1 × n⌉` (at least
   one), and report their mean. Dropping non-finite values first matters: `sorted` compares
   with `<`, and every comparison against NaN is false, so a NaN would never move from where
   it started — leaving one NaN kurtosis able to fix the reported maximum at NaN even with
   finite values available. The 10% fraction is **hardcoded**.

Division by `σ⁴` is unguarded, so a constant volume yields inf or NaN.

### 2.2 Maximum Median Skewness (S₁)

**Column:** `Maximum Median Skewness` · **Unit:** dimensionless ·
**Field:** `intensity.max_median_skew`

Pearson's second skewness coefficient, `3(mean − median)/σ`, reduced by the same top-10%
mean. Positive means the bulk of the distribution sits below its mean — a dim body with a
bright tail.

### 2.3 Maximum Mode Skewness (S₂)

**Column:** `Maximum Mode Skewness` · **Unit:** dimensionless ·
**Field:** `intensity.max_mode_skew`

Pearson's first coefficient, `(mean − mode)/σ`, reduced by the top-10% mean. Referenced to
the histogram's peak rather than its median, so it responds to the background peak where 2.2
responds to the body of the distribution.

### 2.4 Kurtosis Change (ΔK)

**Column:** `Kurtosis Change`, or **`Kurtosis Change (over Z)`** in `xyz` mode ·
**Unit:** dimensionless · **Field:** `intensity.kurtosis_diff`

1. `n_eval = max(⌈percentage_frames_evaluated × n⌉, 1)`, default fraction **0.05**.
2. `mean(K[−n_eval:]) − mean(K[:n_eval])` — a **difference**, not a ratio.
3. **`n < 2` returns NaN.**

**vs 2D:** as in §1.4 — the 3D window is floored at one frame, and a single frame returns NaN
here where 2D returns 0 from comparing a frame with itself. In `xyz` mode the progression
axis is **depth, not time**, and the column is renamed accordingly.

### 2.5 Median Skewness Change (ΔS₁)

**Column:** `Median Skewness Change` / `Median Skewness Change (over Z)` ·
**Unit:** dimensionless · **Field:** `intensity.median_skew_diff`

As 2.4, applied to S₁.

### 2.6 Mode Skewness Change (ΔS₂)

**Column:** `Mode Skewness Change` / `Mode Skewness Change (over Z)` ·
**Unit:** dimensionless · **Field:** `intensity.mode_skew_diff`

As 2.4, applied to S₂.

### 2.7 Total Intensity

**Column:** `Total Intensity` · **Unit:** `a.u.` · **Field:** `intensity_magnitude.total`

Opt-in (`enable_intensity_magnitude`, default **off**). Sum of every finite voxel value in
the analyzed region, averaged over analyzed timepoints.

The branch's only genuinely extensive quantity: it doubles when the object doubles, which no
shape metric does. Two caveats, both material:

- It is a **raw sum including background** unless `intensity_use_mask` is on. On a cropped
  stack the background can dominate.
- It is **meaningless if the detector clipped** — check flag digit 2 first.

### 2.8 Mean Intensity

**Column:** `Mean Intensity` · **Unit:** `a.u.` · **Field:** `intensity_magnitude.mean`

Arithmetic mean of the finite voxel values, averaged over timepoints. Per-sample, so unlike
2.10 it is invariant to voxel size.

### 2.9 Intensity SD

**Column:** `Intensity SD` · **Unit:** `a.u.` · **Field:** `intensity_magnitude.sd`

**Population** standard deviation (`ddof = 0`) of the finite voxel values, averaged over
timepoints.

### 2.10 Intensity Density

**Column:** `Intensity Density (per volume)` in a 3D mode, `Intensity Density (per area)` in a
2D mode · **Unit:** `a.u./μm^3` or `a.u./μm^2` · **Field:** `intensity_magnitude.density`

`total / (n_voxels × voxel volume)`, where the voxel volume is `∏ spacing_zyx` on the
**analyzed** grid — after isotropic resampling and cropping, not the acquired grid. Signal
per unit physical volume, so unlike 2.8 it *does* change if you change the voxel size. NaN
when the spacing is degenerate.

In `xyz` mode the sample size is the in-plane **pixel area** (the z factor is 1.0), because
that mode measures planes.

---

## 3. Optical Flow (3D) branch

Seven columns matching the published 3.1–3.7, but computed by a fundamentally different
solver on a fundamentally different temporal unit.

> ### ⚠ Provisional — treat as work in progress
> The 3D flow branch is functional and wired in, but its **absolute values are not yet
> validated for publication**. The solver systematically under-reports speed (§3.3: the gain
> is below 1 and depends on feature size), and it assumes **small displacements** (§3.1). Use
> it for *relative* comparisons within a consistent dataset; do not read an absolute µm/s off
> it yet. **Two of the seven — Divergence and Curl — measure something different from their 2D
> namesakes** and cannot be pooled with them at all; Speed and Direction are the same
> statistic over a different population, so they are comparable in relative terms only. See
> the *vs 2D* notes on each.

**Implementation:** `analysis/volumetric/flow.py`, wrapping a Lucas-Kanade solver vendored
from aicjanelia/OpticalFlow3D · **Enabled by:** the *Optical Flow* branch checkbox ·
**Modes:** `xyt` and `xyzt` only.

### 3.0 The flow window, and why it costs you the first and last three timepoints

Unlike every other branch, flow reads a timepoint's **neighbors** rather than that timepoint
alone. Each analyzed timepoint needs a **contiguous** window of `6 × flow_t_sigma + 1`
volumes centered on it — **7** at the default `flow_t_sigma = 1`.

Consequences, all of them routinely surprising:

- A requested center is kept only if it is at least `⌊size/2⌋ = 3` from either end. Since the
  frame selector always includes indices 0 and `n−1`, **the first and last three timepoints
  of any series always report nothing.**
- **`frame_step` selects which window *centers* are requested**, exactly as it selects frames
  elsewhere — but it cannot thin the window itself, whose seven volumes are contiguous by
  construction. So raising it reduces the number of flow measurements without reducing the
  number of volumes each one reads.
- **A series shorter than 7 volumes skips the branch entirely**, printing the shortfall and
  returning all-NaN results. Structural and intensity metrics are unaffected.
- Because the default `frame_step = 10` plus the edge exclusion often leaves exactly one
  surviving center, **NaN Speed Change is common and expected** on short series.

Both early-return paths return `velocity_correlation_flag = 0`, so **a skipped flow branch
produces NaNs *without* flag 4.**

> ### The flow columns are dropped entirely for a static z-stack
> When the flow branch produces nothing — a single-timepoint stack, or any series shorter than
> the window — the seven flow columns are **not emitted at all**, in the CSV or on the barcode,
> rather than written as seven all-NaN (black) columns. This applies only to the volumetric
> modes; the 2D modes always emit their flow columns, so the published reference schema is
> unchanged. Such a CSV reads back correctly (the reader recognizes the shortened header), and
> an aggregate that mixes flow-bearing and static runs falls back to showing all columns rather
> than misaligning them.

### 3.1 The solver

It is **Lucas-Kanade** — a **differential** method: velocity is inferred from spatial and
temporal image gradients under a linearized brightness-constancy fit. That carries an inherent
assumption of **small displacements between consecutive volumes**: a feature that moves much
more than its own blurred width per timepoint violates the linearization, and there is **no
coarse-to-fine pyramid** here to rescue large motion (spatial pre-smoothing and optional
`flow_downsample` are the only multiscale controls). If your objects move far between frames,
sample more finely in time or expect the speed to be under-read.

> **Why Lucas-Kanade rather than a heavier optical-flow network.** It matches BARCODE's
> philosophy: it is CPU-only, needs no GPU and no training, and runs anywhere the rest of the
> pipeline does — the same reason the 2D branch uses Farnebäck. Accuracy is traded for being
> runnable by any user on any machine.

The solver is not Farnebäck (2D-only; OpenCV has no 3D equivalent) and not a plain
structure-tensor orientation estimate — it uses **Gaussian-derivative gradients and a
Gaussian-weighted structure tensor**:

1. **Temporal derivative** across the window at `flow_t_sigma` (default **1**), after which
   **only the central timepoint is kept**. This is the *only* step that reads the window;
   everything below operates on that single central volume, which is why 7 volumes in yield
   one flow field out.
2. **Spatial derivatives** of the central volume at `flow_xyz_sigma` (default **3.0**
   voxels) along the differentiated axis, smoothed at `σ/4 = 0.75` along the other two.
3. **Structure tensor** components smoothed by an isotropic Gaussian window of
   `flow_w_sigma` (default **4.0** voxels).
4. **Solve** the 3×3 system in closed form, with machine epsilon added to the determinant as
   a divide-by-zero guard (not a scaled regularization). Result: `vx, vy, vz` in
   voxels/frame at **every** voxel — no sparsity, no confidence gating inside the solver.
5. **Reliability** = the **smallest eigenvalue** of the weighted structure tensor.

**Unit conversion and sign:** `v_axis × spacing_axis / Δt`, giving µm/s. The velocity field
is kept in the **solver's own frame**, with no component negated.

> **Where the y-up convention lives.** BARCODE reports directions y-up while the array
> indexes y downward. The 2D branch reconciles this by doing **both** `np.flipud` and a
> negation — reversing the axis as well as the component, so the field stays
> self-consistent. The 3D branch instead leaves the field alone and negates the `y`
> component of the **mean unit vector only**, at the point the azimuth is reported (§3.5).
>
> This is the single place a sign convention is observable, and confining it there keeps the
> spatial operators (§3.8, §3.9) correct: they see the field in one consistent frame.

**Downsampling.** `flow_downsample` (default **1**) block-averages the *volumes* before
solving, and the physical spacing is scaled to compensate. Note the 2D branch's `downsample`
(default 8) block-averages the *flow field* instead — they are not the same setting.

> `flow_downsample` is a **fixed dataset-wide choice, not a performance dial.** Measured on
> Cell1 going from 1 to 2: Speed unchanged (×1.03), but **Curl ×0.41**, **Velocity
> Correlation Length ×2.25**, **Divergence ×1.51 and unstable (0.99–3.44)**. Runtime drops
> ~4×. Runs at different values may be compared on Speed but **not** on curl, divergence, or
> correlation length.

### 3.2 The validity mask — and where it deliberately does not apply

Two filters build a per-voxel validity mask:

- **Reliability:** voxels below `flow_reliability_percentile` of the min-eigenvalue field are
  dropped. Default **50.0** — so **half the voxels are discarded by default**. Set 0 to keep
  all.
- **Segmentation:** with `flow_use_mask` (default **on**), only in-mask voxels are kept. On by
  default because a cropped nucleus is mostly background, which would otherwise dominate mean
  speed. Flow is still *solved* on the whole volume so boundary gradients stay correct.

The mask is then applied differently by metric type, which is the crux of the design:

- **Pointwise** metrics (Speed, Direction, Spread) use the **masked** field.
- **Spatial-derivative** metrics (Divergence, Curl) are computed on the **full** field and
  masked only at the reduction — punching scattered holes before differentiating would make
  the derivative describe the hole pattern rather than the flow.
- **Velocity Correlation Length ignores both filters entirely**, being computed on the full,
  unmasked, unfiltered field.

### 3.3 Speed (v)

**Column:** `Speed` · **Unit:** `μm/s` · **Field:** `flow.mean_speed`

Mean magnitude `√(vx² + vy² + vz²)` over valid voxels, averaged over analyzed windows.
Genuinely 3D — out-of-plane motion contributes.

A fully masked window yields NaN and is **excluded** from the average rather than counted as
zero.

> **Speeds are systematically under-reported.** The solver is gradient-based, so its recovered
> speed is a fixed fraction of the true speed when the tracked features are comparable in size
> to `flow_xyz_sigma`: measured gain **~0.55 at a 2-voxel feature scale, ~0.92 at 6 voxels**,
> both at the default σ = 3. The gain is **linear in speed**, so *relative* comparisons across
> a dataset are sound; **absolute speeds are a lower bound.** At very low contrast the
> structure tensor approaches the solver's regularization and velocities collapse towards
> zero — the reliability mask is what keeps those voxels out of the mean.

**vs 2D:** the 2D branch divides by `exposure_time × (stop − start)`, explicitly accounting
for the frame *stride* of the pair; 3D divides by `dt` alone because the window is
contiguous. 2D applies no reliability weighting and no segmentation mask.

### 3.4 Speed Change (Δv)

**Column:** `Speed Change` · **Unit:** `μm/s` · **Field:** `flow.delta_speed`

`mean(v[−n_eval:]) − mean(v[:n_eval])` over **windows**, with
`n_eval = max(⌈0.05 × n_windows⌉, 1)`.

**A single window yields NaN, not 0**, where the 2D branch returns 0.0 — the same choice as
§1.4 and for the same reason: a lone window is common enough here (§3.0) that "no change"
would be read as a measurement. On the barcode this metric is scaled about 0, since it is
signed.

### 3.5 Mean Flow Direction (θ)

**Column:** `Mean Flow Direction` · **Unit:** `rads`, range (−π, π] ·
**Field:** `flow.mean_theta`

> **This is the XY azimuth only.** It is *not* a full 3D direction — elevation is discarded
> from this metric (it is folded into 3.6 instead), deliberately, so the column stays
> comparable with 2D runs. **There is no elevation or polar-angle column.**

Two-stage vector averaging, never an angle average:

1. Per window, normalize the velocity field to **3D unit vectors** and average them
   componentwise over valid voxels. The result is a mean unit vector with magnitude ≤ 1; it
   is **not** renormalized.
2. Across windows, `atan2(mean(vy component), mean(vx component))` — the azimuth of the grand
   mean unit vector projected onto XY.

**vs 2D:** mathematically the same two-stage average, with two differences that follow from
the extra dimension and the validity mask. The per-window unit vectors are **3D**, so a voxel
moving largely out of plane contributes a *shorter* in-plane projection and its azimuth is
damped, where 2D unit vectors always have in-plane length exactly 1. And 2D averages over
**every** pixel, including motionless ones, where `atan2(0, 0) = 0` contributes `(1, 0)` and
pulls the result toward +x; 3D averages over valid voxels only, having a reliability mask to
define them with. Same statistic, different populations — read them separately.

### 3.6 Directional Spread (σ_θ)

**Column:** `Directional Spread` · **Unit:** `rads` · **Field:** `flow.mean_sigma_theta`

Spherical dispersion from the **resultant length of the full 3D unit vectors** — so
out-of-plane scatter widens it, unlike 3.5 which is purely azimuthal.

1. Per window, `R = ‖mean unit vector‖`, using a plain sum so that an all-NaN (fully masked)
   window **propagates NaN**. This is deliberate: a NaN-tolerant sum would give `R = 0`, which
   reads as "perfectly isotropic flow" and would produce a large *finite* spread for a window
   containing no data at all.
2. `σ = √(−2 ln R)`, per window, then averaged. `R` is clipped to `[1e-12, 1]`, bounding σ at
   ≈ 7.434 rad and making a perfectly coherent window read exactly 0.

`R → 1` means coherent flow (σ → 0); `R → 0` means isotropic scatter.

**vs 2D:** same formula on a 2-component resultant, and unclipped — so at the limits `R = 0`
gives `−ln 0 = +inf`, and an `R` a hair above 1 from floating point gives NaN. Those limits
are far rarer in a plane of real 2D data than in a volume where a mask can empty a whole
window, which is why the 3D branch clips and propagates NaN instead. 2D also includes
motionless pixels in the average (see 3.5), which pulls `R` **upward**, toward the coherent
end; the 3D failure mode the NaN-propagating sum guards against pulls the other way.

The barcode scales this `[0, π]`, so values above π are clipped **in the picture only**.

### 3.7 Velocity Correlation Length (ξ_v)

**Column:** `Velocity Correlation Length` · **Unit:** `μm` ·
**Field:** `flow.velocity_correlation_length`

The physical distance over which velocity vectors stay correlated — the size of coherently
moving domains.

1. Computed on the **full unmasked field** (see 3.2).
2. Wiener–Khinchin per component, summed over the three components: **circular**
   autocorrelation via FFT, where the 2D branch brute-forces every shift at O(N²).
3. Normalized so `C(0) = 1`, i.e. `⟨v(x)·v(x+r)⟩ / ⟨|v|²⟩`.
4. **Radially averaged on physical distance**, using exactly the scheme of section 1.12 —
   µm-wide shells, `r_max` bounded by the smallest physical half-extent, each shell reported
   at its own mean radius.
5. **Threshold 0.5**, with the **same snap-to-nearer-point selection rule** as 1.12 (and the
   same divergence from the 2.0 reference).
6. Mean over windows.

**NaN per window when:** the field is identically zero; `r_max ≤ step`, i.e. the grid cannot
hold even one shell; or `C` never crosses 0.5 within `r_max` — the "coherence exceeds the
field of view" case.

**Any** of those three raises **flag digit 4**, not just the last: the test is
NaN-propagating over all windows. (Section 1.12 has the same property for digit 3.)

### 3.8 Divergence (∇·v)

**Column:** `Divergence` · **Unit:** `1/s` in a volumetric mode (blank in 2D) ·
**Field:** `flow.divergence`

`∂vx/∂x + ∂vy/∂y + ∂vz/∂z` by second-order central differences on the **physical** grid
(one-sided at borders), computed on the full field, masked at the reduction, meaned over
windows. Positive = local expansion, negative = compaction.

> Verified against closed forms: `div(k·r) = 3k` exactly, zero for uniform flow and zero for
> solid-body rotation, and invariant under proper rotations.

> **vs 2D — these are different quantities that share a column name, and must not be pooled.**
> The 2D metric is the spatial mean divergence of the **cumulative sum of *unit* vector
> fields**, taken at the **last frame pair only**: it describes accumulated *direction* over
> the series, carries units of 1/µm, and depends on how many pairs preceded it. That is a
> coherent measurement of its own — it just is not this one, which is a window-mean `∇·v` of
> the physical velocity field in 1/s. The name is the only thing they have in common.

### 3.9 Curl (Ω)

**Column:** `Curl` · **Unit:** `1/s` in a volumetric mode (blank in 2D) ·
**Field:** `flow.curl`

> **Curl means something different in 3D.** In a plane the curl is a signed scalar, which is
> what the 2D branch reports, and opposite vortices cancel in its mean. In three dimensions
> curl is a *vector* and there is one CSV column, so this branch reports the mean of the
> **magnitude ‖∇×v‖** instead. It is **always positive and carries no handedness**: a
> clockwise and a counter-clockwise vortex give the same value, and counter-rotating regions
> **add** rather than canceling. Neither convention converts to the other.

Computed componentwise by central differences on the physical grid, then `√(Ωx² + Ωy² + Ωz²)`,
on the full field, masked at the reduction, meaned over windows.

> Verified against closed forms: `‖∇×v‖ = 2ω` everywhere for solid-body rotation at rate ω,
> zero for uniform flow, and invariant under proper rotations. (A plain transpose is a
> *reflection*, which flips the curl vector's sign, so it is not a valid invariance check.)

---

## 4. Volumetric branch

Sections 1–3 are the **three branches** of published BARCODE, restated for a volume. This is the
fourth branch: everything the volume makes possible that a flat image cannot give you. It is one
branch, not seven — the families inside it are grouped by **what they require**, because that is
the only thing you need to decide before a run.

Almost every family is **opt-in**, off unless you switch it on, because the barcode is already
wide and a column nobody asked for still takes space and still gets normalized.

| Family | §  | Default | Needs a segmentation? | Also needs |
|---|---|---|---|---|
| Surface mesh | 4.1–4.11 | off (`mesh_enabled`) | **no** — falls back to the thresholded volume | isotropic grid |
| Curvature | 4.12–4.16 | on *within* meshing (`mesh_curvature`) | no (rides on the mesh) | the mesh |
| Curvature extremes | 4.17–4.18 | off (`enable_curvature_range`) | no | the mesh |
| Object size distribution | 4.19–4.22 | off (`enable_component_stats`) | no — describes the thresholded objects | — |
| Depth profile | 4.23–4.28 | off (`enable_slice_profile`) | no — runs on the binary | — |
| In-mask intensity | 4.29–4.37 | off (`enable_mask_intensity`) | **yes** | — |
| Packing topology | 4.38–4.44 | off (`enable_packing_topology`) | **instance** mask | isotropic grid |
| Object rows | 4.45 | resolved by `row_axis` | **instance** mask | — |
| Range provenance | 4.46 | off (`record_range_columns`) | no — records what was analyzed | — |

**Sixteen of these columns need cubic voxels and are simply absent without them** — the whole
mesh and curvature set, and all of packing. See [P.4](#p4-isotropic-resampling-and-the-crop-box--the-largest-single-effect-on-any-number)
for which metrics depend on the grid and what to do about each.

Two families live in the branches above but only appear with a mask: the intensity histogram can
be restricted to in-mask voxels (`intensity_use_mask`, §2.0), and optical flow restricts its
averages to the mask by default when one is present (`flow_use_mask`, §3.2).

**A single volume (one timepoint) is a normal input.** Every *Change* metric is then NaN, because
a change needs two points on the progression axis. Everything else is live. See the
[NaN triage table](#my-column-is-all-nan--what-to-check).

**Optical flow needs a time axis, so a static z-stack gets no flow at all.** Rather than paint
seven all-NaN columns, the volumetric pipeline **omits the flow columns entirely** when the branch
produced nothing. The 2D modes are unaffected. See §3.

---

### Runs without a segmentation

**Surface mesh** — triangulated surface of the segmented object (or, with no mask, the
binarized volume), and the geometry measured on it. Eight columns.

**Implementation:** `analysis/volumetric/mesh.py` · **Switch:** `mesh_enabled`, default
**off**; requires an isotropic grid, but **not** a segmentation — without one it meshes the
binarized volume (§4.1) · **GUI:** *Mesh the Segmented Surface* · **Modes:** `xyzt` only ·
**Cost:** roughly 8 s per analyzed timepoint (empirical, not pinned by test).

#### 4.1 How the surface is built

**There is no marching cubes anywhere.** Surface extraction is CGAL's `cgalsurf` via
pyiso2mesh's `v2s` — a Delaunay-refinement restricted surface mesher — at
`mesh_isovalue` (default **0.5**) on a 0/1 volume, with `radbound = mesh_maxrad` (default
**5.0**, in `mesh_maxrad_units`). A marching-cubes fallback is deliberately absent: it would
silently produce meshes that are not comparable to the MATLAB pipeline's.

> ### The isovalue is the boundary a binary mask means
> `mesh_isovalue` defaults to **0.5** — the boundary between the last foreground voxel and
> the first background one, which is what a 0/1 mask represents, and the most accurate value
> on rounded objects (measured against closed forms by `scripts/validate_phantoms.py`: within
> ~0.1% of the exact sphere volume down to a 16-voxel radius). Setting it higher pulls the
> surface inward onto foreground-voxel centers and shrinks small objects; setting
> `mesh_isovalue = 0.99` (with `mesh_matlab_compat`) reproduces the MATLAB pipeline.
>
> One caveat, easy to rediscover as a mystery: at 0.5 on a 0/1 field the level passes exactly
> through voxel-face midpoints, so on shapes whose faces lie *on* those planes — axis-aligned
> cuboids, which real specimens are not — the surface is geometrically ambiguous and congruent
> objects can mesh to visibly different volumes. Nudging to 0.52 removes it at ~0.3% accuracy.

> ### `mesh_maxrad` is the single biggest control on mesh accuracy
> What matters is its size **relative to the object**, and `mesh_maxrad_units` decides how the
> number is read — the three options differ in what stays constant across datasets:
>
> | Value | `mesh_maxrad` means | Constant across acquisitions? |
> |---|---|---|
> | `voxels` (default) | isotropic voxels, as stored | **No** — the same number is a different physical size on every dataset |
> | `um` | microns, converted at meshing time using the voxel size | **Yes**, physically |
> | `relative` | a fraction of the object's equivalent-sphere radius, floored at 0.25 voxels | **Yes**, relative to the object — the thing that actually governs accuracy |
>
> `relative` needs the object's size, so it is only available where that is known at meshing
> time. Neither `voxels` nor `um` adapts to object size; if you are meshing objects of very
> different scales in one study, that is the setting to reach for. **`Mesh Volume Ratio`
> (§4.8) is how you tell whether the value you chose is working.**

Pipeline per timepoint:

1. **Largest connected component** of the mask (26-connectivity). **No Gaussian smoothing, no
   binary closing, no hole filling** is applied to the mask — smoothing happens on the mesh.
2. **One voxel of background padded on every side.** `v2s` does not close the surface at the
   array boundary, so an object touching a face would come out **open** — and with
   `crop_to_mask` off that is the normal case, not an exception. The pad is unconditional
   (an extra background layer where background already exists cannot move the isosurface, so
   already-closed meshes are bit-identical) and is subtracted back off the vertices, leaving
   the caller's coordinate frame unchanged. §4.9's voxel-count volumes are unaffected.
3. **`v2s`** surface extraction.
4. **Decimation:** faces with area above `mesh_area_frac × largest face area` (default
   **0.2**) are counted, and that ratio drives `cgalsimp2`.
5. **Laplacian-HC smoothing** (Vollmer/Mencl/Müller), `mesh_smoothing_iterations` = **10**,
   `alpha` = **0.1**, `beta` = **0.5**. HC costs < 1% of enclosed volume where plain Laplacian
   shrinks more than 5× as much.
6. **Scale to µm** by a single scalar — valid only because meshing requires isotropy.

> **An open surface raises flag digit 7.** Step 2 closes the ordinary boundary-touching case,
> but a mesh can still come out with a boundary, and the row then says so. It matters because
> `Mesh Volume` is a signed tetrahedron sum from the coordinate origin: a hole at height *z*
> contributes roughly ±`A_hole·z/3`, which on full-field vertices is of the same order as the
> object itself. `Mesh Surface Area` is short by the missing cap, `Sphericity` and
> `Equivalent Sphere Radius` inherit both, and the winding test (§4.12) becomes unreliable —
> so **every curvature sign** on that row can be inverted. Treat a flag-7 row's mesh and
> curvature columns as unusable rather than merely approximate.

**Prerequisites and failure modes.** A **missing segmentation is not a failure** — meshing
falls back to the binarized volume and prints that it did so. A missing `pyiso2mesh`, an
anisotropic grid, an empty binary, a volume so small `v2s` produces no faces, or an area filter
that keeps nothing each raise `MeshingError`, which is **caught, printed and skipped** so one
misconfigured file does not abort a batch. Where meshing fails the mesh columns stay NaN.
Transient pyiso2mesh failures get three attempts (two retries).

**Reproducibility.** `cgalsurf` is **not bit-reproducible across processes** even with a fixed
seed: repeated runs land on one of two outcomes differing by 0.004% in volume and 0.02% in
area. Bit-parity with MATLAB is unachievable by construction (the two implementations seed
their bounding sphere from different interior points).

> **To reproduce the MATLAB pipeline**, set `mesh_isovalue = 0.99`: at nucleus scale that
> agrees to within 0.11% in volume, 0.5% in area and sphericity, and 0.4% in height (0.03% /
> 0.05% / 0.05% / 0.25% with `mesh_matlab_compat`). At the default 0.5 the geometry is more
> accurate and so differs from MATLAB by roughly the size of MATLAB's inward bias — about 12%
> in volume on a 16-voxel-radius object, more on smaller ones.

Curvature, by contrast, reproduces MATLAB bit for bit — see §4.12 for the conditions.

#### 4.2 Mesh Volume

**Column:** `Mesh Volume` · **Unit:** `μm^3` · **Field:** `mesh.mesh_volume`

Enclosed volume by the **divergence theorem**: `|⅙ Σ v₀ · (v₁ × v₂)|` over triangles — the
sum of signed tetrahedron volumes from the origin. The sign encodes surface orientation and
is retained separately as a diagnostic.

Deliberately distinct from `Total Island Volume Quantity` (§1.10): meshing smooths the surface,
so the two differ by a few percent, and §4.8 reports exactly that discrepancy.

#### 4.3 Mesh Surface Area

**Column:** `Mesh Surface Area` · **Unit:** `μm^2` · **Field:** `mesh.surface_area`

Plain sum of triangle areas, `½‖(v₁−v₀) × (v₂−v₀)‖`. (A second, quad-fan area formula exists
in the module, used only for the decimation ratio under `mesh_matlab_compat` — it reproduces
how GIBBON's `patch_area` treats a triangle, which is what the MATLAB pipeline's decimation
ratio was computed from. It is off by default because the plain sum is the area you want;
turn it on only to match those older numbers.)

#### 4.4 Sphericity

**Column:** `Sphericity` · **Unit:** dimensionless · **Field:** `mesh.sphericity`

Wadell sphericity, `π^(1/3) (6V)^(2/3) / A` — the surface area of a sphere of equal volume
divided by the actual area. Exactly 1 for a perfect sphere, below 1 otherwise. NaN when the
area is not positive.

#### 4.5 Equivalent Sphere Radius

**Column:** `Equivalent Sphere Radius` · **Unit:** `μm` · **Field:** `mesh.equivalent_radius`

`(3V / 4π)^(1/3)` — the radius of the sphere with the same mesh volume.

#### 4.6 Mesh Height

**Column:** `Mesh Height` · **Unit:** `μm` · **Field:** `mesh.height`

The z extent **of the face centroids**, not of the vertices. This is deliberate MATLAB parity
and is systematically **shorter** than the true z extent, because centroids lie inside the
surface. It is the sole reason a perfect sphere reports a Lateral/Axial Ratio of ≈1.08 rather than
1.00 (see §4.7).

#### 4.7 Lateral/Axial Ratio

**Column:** `Lateral/Axial Ratio` · **Unit:** dimensionless · **Field:** `mesh.aspect_ratio`

This is a **lateral-to-axial** aspect ratio: how wide the object is across the imaging plane
relative to how tall it is along the optical axis. Read it that way — a value above 1 is a
flat/oblate object spread out in xy, ≈1 is round, below 1 is a column standing along z. The
column is named `Lateral/Axial Ratio` for exactly this reason: a bare `Aspect Ratio` reads as
an in-plane width/height, which this is not.

`(MIP major axis + MIP minor axis) / (2 × height)` — lateral size over axial size. Above 1 is
a flat/oblate object, ≈1 is round; dimensionless, so size-independent.

The MIP axes come from the **mask**, not the mesh — and specifically from the largest
connected component of it, since that is what was meshed. Project along z, label the
projection with 8-connectivity, take the largest projected region, and read the major and
minor axes of the ellipse with the same normalized second central moments.

A sphere gives ≈1.08, not 1.00, because the numerator is the full mask silhouette while the
denominator (§4.6) is the inset face-centroid height; the offset grows as the mesh coarsens.
This is pinned by test as a documented property rather than treated as an error.

#### 4.8 Mesh Volume Ratio

**Column:** `Mesh Volume Ratio` · **Unit:** dimensionless · **Field:** `mesh.volume_ratio`

Mesh volume (§4.2) divided by the **raw voxel-counted volume of the same mask**. A **fidelity
check, not a shape descriptor** — ≈1 means the mesh faithfully represents the mask. It is
**never drawn on the barcode** (`ALWAYS_HIDDEN_BARCODE_METRICS` in `core/metrics.py`): per-column
normalization would stretch a flat "≈1 = trustworthy" check into an apparent signal. It stays in
the CSV, which is where a fidelity value is actually read.

> **Check this on any dataset that is not a nucleus.** It is the built-in detector for a
> `mesh_maxrad` mismatched to the object (§4.1): on closed-form spheres the default of 5
> recovers 0.997 of the exact surface area at a 65-voxel radius but only 0.926 at a 16-voxel
> one, and on a thin object it loses roughly half the volume. A ratio well below 1 is that,
> nearly always.

#### 4.9 Solidity

**Column:** `Solidity` · **Unit:** dimensionless · **Field:** `mesh.solidity`

Object volume divided by convex-hull volume, **both as voxel counts** — deliberately, so
meshing shrinkage does not contaminate what should be a pure convexity measure. 1 means
convex; lower means more lobed. It is the same quantity MATLAB `regionprops3` calls
Solidity, computed by the hull convention of chromatin-analysis' `morph3d_solidity`, with
which it agrees exactly — and therefore *not* numerically identical to MATLAB's (see below).

The hull is built over **voxel centers**, from surface voxels only, with a bounding-box-
limited membership test. Two distinct degenerate paths: fewer than 4 *object* voxels reports
the object as fully solid; fewer than 4 *surface* voxels falls back to using all object
voxels. A qhull failure on planar or collinear input also reports fully solid.

> **Which hull convention, and what it costs.** A voxel is a box, so "the hull of the object"
> has more than one reasonable meaning. The module records that against 187 stored MATLAB
> values, the voxel-center convention used here runs **~1.5% high** and the voxel-corner
> convention ~1.2% low, with MATLAB between them — none of the three is the wrong answer, they
> are three readings of the same shape. Centers were chosen for exact agreement with
> chromatin-analysis, so solidity is directly comparable across the two packages and offset by
> ~1.5% against MATLAB. This is a docstring claim, not pinned by any test in this repo.

#### 4.10 Concavity — removed

There is no `Concavity` column. It was `1 − Solidity` and carried **no information Solidity
does not**, so it was dropped from every schema (field mesh family and per-object rows alike)
rather than kept and flagged as hideable. Read lobedness directly off `Solidity` (§4.9): lower
= more lobed. For a genuine surface-concavity measure use `Concave Area Fraction` (§4.16), which
is a different quantity entirely — the area fraction of concave faces, not a hull-fill ratio.

#### 4.11 How multiple objects and timepoints are combined

**Across timepoints (live):** one object is meshed per analyzed timepoint — **the largest
connected component of the mask** — and every mesh field is reduced by the arithmetic mean
over analyzed timepoints, dropping non-finite values. Note this means **extensive quantities
(volume, surface area) are averaged, not summed.**

**Across objects in one frame (implemented but not wired):** `analysis/volumetric/mesh_field.py`
meshes every label in an instance volume, with per-object border and size rejection and
per-object error isolation. It has **no call site in the pipeline** and provides no scalar
reduction.

> `mesh_aggregation`'s `"mean"` and `"total"` settings are **not implemented**: meshing
> **raises** on any value it cannot honor, so largest-component behavior must be requested
> explicitly with `"largest"`.

---

**Curvature** — principal curvatures over the mesh surface, and the invagination metrics built on them. Three
columns in the mesh family plus two opt-in extremes.

**Implementation:** `analysis/volumetric/curvature.py` · **Switch:** `mesh_curvature`,
default **on** (cheap relative to meshing itself) · **Requires:** the mesh family (hence an
isotropic grid), but not a segmentation — it rides on whatever surface meshing produced.

#### 4.12 How curvature is estimated

**Rusinkiewicz (2004)** — finite differences of vertex normals along triangle edges,
least-squares solved per face, projected into per-vertex tangent frames and accumulated with
mixed-Voronoi (Meyer et al. 2002) area weights, then a Jacobi rotation for the principal
curvatures. **Not** a quadric fit and **not** a cotangent Laplacian. Ported from Ben Shabat's
MATLAB toolbox; verified bit-for-bit against it on real Jurkat meshes (cells 1, 11 and 12,
~24,000 faces in total) — every scalar to nine decimals, per-face mean curvature to
1e-15 1/µm.

> **That parity is conditional.** It holds for meshes with no degenerate faces and at least
> three z bins. Two behaviors deliberately depart from the MATLAB source: the `UNCLASSIFIED`
> class for non-finite principal curvatures, which is excluded from the ratio denominators
> (§4.15), and the suspension of bottom/top exclusion for very flat objects (§4.13). Both handle
> cases MATLAB does not, so a mesh hitting either will not reproduce MATLAB exactly.

Per-vertex `k₁, k₂` are averaged to their three corners to give per-face values, from which:

- `H = (k_min + k_max)/2` — pointwise mean curvature;
- `K = k_min · k_max` — Gaussian curvature (computed, never a column).

**Winding is checked and corrected first.** An inward-wound mesh would invert every sign, so
the faces are flipped rather than the result being silently reported as its complement.

**Sign convention: with outward normals a sphere is positive.** Concave is negative. Units
**1/µm**.

#### 4.13 Which faces are excluded — by default, none

**By default, no faces are excluded — curvature is measured over the whole surface, and top
and bottom caps are *not* clipped.** Two exclusion rules exist, both **off by default**, and
both must be switched on deliberately (a user who wants clipping opts in and supplies the
threshold; it is heavily gated precisely because silently discarding surface changes every
curvature number):

| Setting | Default | Excludes |
|---|---|---|
| `curvature_exclude_caps` | `False` | Faces whose centroid falls in the lowest or highest 0.1 µm z bin. |
| `curvature_outlier_limit` | `0.0` (off) | Faces with \|H\| above the limit, in µm⁻¹. MATLAB uses `2.0`, i.e. a radius of curvature below 0.5 µm. |

The two answer different questions and are worth enabling in different situations.

**`curvature_exclude_caps` is anatomical.** Where a segmentation is clipped by the top or
bottom of the imaged stack, the resulting flat cap is an artifact of the acquisition rather
than a surface of the object, and its curvature is meaningless. Turn it on when flag digit 6
is raised, or whenever the object visibly runs out of the stack. For an object sitting
entirely inside the imaged volume it discards real surface, which is why it is off unless
you request it.

**`curvature_outlier_limit` is numerical.** A face reporting a radius of curvature well below
the mesh's own edge length is a badly conditioned triangle, not fine structure. Setting a
limit keeps such faces out of the area-weighted means. With the default of 0 every face
counts, so a mesh with degenerate triangles can pull ⟨H⟩ around — check `Mesh Volume Ratio`
(§4.8) if a curvature value looks implausible.

When a limit is set, the outlier decision is made **once, on `H`**, and reused for the
minimum and maximum, so all three means always run over an identical face set.

> **Setting `curvature_exclude_caps = True` and `curvature_outlier_limit = 2.0` reproduces
> the MATLAB behavior** and the bit-for-bit parity recorded in §4.12. The defaults are a
> deliberate departure from it: measuring the whole surface is the more defensible starting
> point for an object fully inside the stack, and discarding surface should be something you
> ask for rather than something that happens silently.
>
With both exclusion rules off, the two ratios (§4.15, §4.16) and the three means share one
denominator: the whole surface, less any faces whose principal curvatures were non-finite.

#### 4.14 Mean Curvature ⟨H⟩

**Column:** `Mean Curvature <H>` · **Unit:** `1/μm` · **Field:** `mesh.mean_curvature`

Area-weighted mean of the per-face `H` over the usable faces:
`Σ A_f H_f / Σ A_f`.

Two averages are stacked — "mean curvature" is already a *local* quantity (the mean of the two
principal curvatures), and this then averages it over the surface — which is why the column
carries the ⟨H⟩ notation rather than the bare name. Positive means convex overall; a sphere of
radius R reads +1/R.

NaN when the usable face area sums to zero.

On the barcode, curvature columns are scaled about **0** rather than `[0, max]`, because a
concave surface gives a genuinely negative mean and a `[0, max]` scale would clip half the
range.

#### 4.15 Invagination Ratio

**Column:** `Invagination Ratio` · **Unit:** dimensionless, 0–1 ·
**Field:** `mesh.invagination_ratio`

Area fraction of faces that are **concave or saddle**.

Faces are classified from their principal curvatures, in this precedence:

- either curvature non-finite → **unclassified** (this test wins);
- `k_max ≤ 0` → **concave**;
- else `k_min ≤ 0` → **hyperboloid / saddle**;
- else → **convex**.

Both zero classifies as concave. The **unclassified** class exists because NaN principal
curvatures would otherwise fall through to *convex*, counting a face BARCODE could not
measure as evidence against invagination. Unclassified faces are removed from the
**denominator** as well as the numerator, so the ratios describe the surface that was
actually classifiable — not the whole non-bottom/top surface.

#### 4.16 Concave Area Fraction

**Column:** `Concave Area Fraction` · **Unit:** dimensionless, 0–1 ·
**Field:** `mesh.concave_ratio`

Area fraction of **concave faces only** — a strict subset of §4.15, so always ≤ Invagination
Ratio. Same denominator.

#### 4.17 Minimum Curvature

**Column:** `Minimum Curvature` · **Unit:** `1/μm` ·
**Field:** `curvature_range.min_curvature` · **Opt-in:** `enable_curvature_range`

The **area-weighted mean of the per-face minimum principal curvature** over the same usable
faces as §4.14. **Not a global minimum** — it is the "most concave direction", averaged over the
surface.

#### 4.18 Maximum Curvature

**Column:** `Maximum Curvature` · **Unit:** `1/μm` ·
**Field:** `curvature_range.max_curvature` · **Opt-in:** `enable_curvature_range`

Likewise for the maximum principal curvature.

**Why §4.17 and §4.18 exist:** ⟨H⟩ averages `k₁` and `k₂` together, so a **saddle — sharply curved
both ways — averages towards zero and reads as flat.** A surface can be highly structured and
still report a near-zero mean. These keep the two principal directions apart and say how
structured it really is.

---

**Object size distribution** — the binarization family reports the largest, mean and total object size. What it cannot say is
whether those objects are **uniform or wildly unequal** — one dominant object plus debris
gives the same mean as a handful of even ones. This family describes the *shape* of that
distribution.

**Implementation:** `analysis/volumetric/binarization.py`, reduced in `run.py` ·
**Opt-in:** `enable_component_stats`, default **off** · **GUI:** *Per-Object Size Statistics* ·
**CLI:** `--component-stats` · **Modes:** volumetric only.

Off by default because the barcode is already wide, and a metric nobody reads is worse than
absent — it still takes a column and still gets normalized.

> **This is a field-level summary, not per-object rows.** It describes the *shape* of the
> object-size distribution in one field (count, spread, skew, median) as columns on the field
> row. If you want one row **per object** — each cell's own volume, contacts and intensity —
> that is the object row axis (§4.45), a separate output.

Sizes are expressed as a **fraction of the analyzed field**, matching the binarization family,
so a run is comparable with another of a different crop size. All four are therefore
dimensionless.

#### 4.19 Island Count

**Column:** `Island Count` · **Unit:** dimensionless · **Field:** `components.count`

Number of objects per timepoint, averaged. From the label partition if an instance mask
supplied one, else from 26-connected labeling.

#### 4.20 Island Volume SD

**Column:** `Island Volume SD` · **Unit:** dimensionless · **Field:** `components.size_sd`

**Population** SD (`ddof = 0`) of the per-object voxel counts within a timepoint, averaged over
timepoints, then divided by the field voxel count. **`0.0`, not NaN, when a timepoint has one
object.**

#### 4.21 Island Volume Skewness

**Column:** `Island Volume Skewness` · **Unit:** dimensionless · **Field:** `components.size_skew`

Third standardized moment of the per-object size distribution, `mean(((a − μ)/σ)³)` with a
population σ and no bias correction. **NaN unless the timepoint has more than 2 objects and
non-zero variance.** Positive means a few objects much larger than typical.

#### 4.22 Median Island Volume

**Column:** `Median Island Volume` · **Unit:** dimensionless · **Field:** `components.size_median`

Median per-object voxel count, averaged over timepoints, divided by the field voxel count.
Read against Mean Island Volume (1.9): a median well below the mean signals a few dominant
objects.

> **All four are means of per-timepoint statistics, not statistics pooled over timepoints.**
> The reported SD is the average of the per-timepoint SDs, not the SD of all objects across
> all timepoints.

---

**Depth profile** — every other metric in the branch reduces a stack to one number per timepoint and so cannot say
**where** in depth anything happened. For a stack through a curved surface or a rounded object
the maximal-area slice locates the equator, and it moves when the object flattens, tilts, or drifts
through the focal range.

**Implementation:** `analysis/volumetric/slice_profile.py` · **Opt-in:** `enable_slice_profile`,
default **off** · **GUI:** *Maximal Area Slice & Clipping Flag* · **CLI:** `--slice-profile` ·
**Modes:** volumetric only.

> **A secondary output, gated for a reason.** These are diagnostics — where in depth the object
> is widest, and whether it is clipped — not primary shape metrics. Leave them off unless you
> are specifically checking depth placement or field-of-view clipping; they are not meant to sit
> in a default barcode.

#### 4.23 Which binary volume is measured

If a segmentation resolved, the mask (all instance labels pooled). Otherwise the binarization
branch's own rule is re-applied to that timepoint — threshold at
`mean × (1 + threshold_offset)`, then removal of both small objects **and small holes** at
`minimum_island_size + 1` with 26-connectivity, then inversion if configured. Recomputed one
frame at a time rather than cached, so an opt-in family does not multiply peak memory across
the series.

#### 4.24 Maximal Area Slice Index

**Column:** `Maximal Area Slice Index` · **Unit:** `slice` ·
**Field:** `slice_profile.max_area_index`

1. Per z slice, the foreground **fraction** of that slice's pixels.
2. `argmax` over z — **the first maximum wins on a tie**.
3. Averaged over analyzed timepoints, so the reported value can be fractional.

The index is into the **analyzed** stack, which after isotropic resampling has a very different
slice count from the acquired stack (245 vs 54 on 0.3/0.065 µm data).

#### 4.25 Maximal Area Slice Depth

**Column:** `Maximal Area Slice Depth` · **Unit:** `μm` ·
**Field:** `slice_profile.max_area_depth`

`index × z step` on the analyzed grid. Measured from the **first analyzed slice**, so a z-range
restriction moves the origin with it — this is a shape descriptor, **not an absolute stage
position**.

#### 4.26 Maximal Area Slice Area

**Column:** `Maximal Area Slice Area` · **Unit:** labeled `fraction of FOV`, valued as a fraction 0–1 ·
**Field:** `slice_profile.max_area_area`

The foreground fraction of that slice.

#### 4.27 The clipping flag (digit 6)

This family also raises **flag digit 6**. Per timepoint, foreground is checked against all six
faces of the analyzed volume — the four xy borders and the first and last z slice. The flag is
the **union over analyzed timepoints**, because clipping anywhere in the series taints the
time-averaged metrics.

When it fires, the object continues outside the analyzed field and **every size, shape and
curvature metric describes a truncated object** — which is not recoverable from the numbers
themselves. It is deliberately a separate digit from 5: **digit 5 means *you* restricted the
range, digit 6 means the *data* is cut off.**

#### 4.28 NaN rules

A volume with no foreground at all gives all three as NaN (logged as "empty volume") rather
than 0.

---

### Requires a segmentation

**In-mask intensity** — the clustering readout: how signal is distributed **inside** each
segmented object. The
intensity branch (section 2) describes whatever voxels it is handed, which normally means the
background peak dominates. A uniformly filled nucleus and one with bright foci have the same
mean and very different CV and entropy.

**Implementation:** `analysis/volumetric/mask_intensity.py` · **Opt-in:**
`enable_mask_intensity`, default **off** · **GUI:** *In-Mask Intensity Statistics* ·
**CLI:** `--mask-intensity` · **Modes:** volumetric only. **Requires a segmentation** — skipped
with a printed reason if none resolved.

Distinct from `intensity_use_mask`, which merely restricts section 2's *existing* histogram to
in-mask voxels instead of adding per-object statistics.

#### 4.29 What counts as an object, and how objects are combined

- A **boolean** mask is one object.
- An **integer** mask gives one object per distinct positive label.
- Objects smaller than `mask_intensity_min_voxels` (default **8**) are skipped — entropy from a
  handful of voxels is noise, and averaging it in biases the run. So is any object whose values
  are constant.

Every statistic is the **unweighted mean over objects**, and those per-timepoint values are
averaged over timepoints the same way. Unweighted is deliberate: **one large object does not
outvote the rest.**

#### 4.30 Rescaling — which metrics use it, and which deliberately do not

Only **entropy** (§4.35, and hence §4.36) is computed on a per-object rescaling to [0, 1]; §4.31,
§4.32, §4.33, §4.34 and §4.37 are computed on **raw** voxel values.

This departs from the source MATLAB (`clustering_inside_nuc.m`), which rescales uniformly
before every statistic — the simpler and more consistent rule, and harmless for the
statistics that motivated it. The split here is for the ones it is not harmless for. Entropy
genuinely needs rescaling, since objects must be binned over a common range to be comparable.
CV and skewness are already scale-invariant, and the `−min` shift of an affine map moves
them; the bright fraction degenerates outright, because a punctate object's rescaled median
is 0 — exactly the case the metric exists for. **Values here therefore differ from MATLAB's
for CV, skewness and the bright fraction.**

> The `--mask-intensity` CLI help text describes this same rule: only entropy is rescaled.

#### 4.31 In-Mask MFI

**Column:** `In-Mask MFI` · **Unit:** `a.u.` · **Field:** `mask_intensity.mfi`

Mean raw intensity inside an object, averaged over objects then timepoints.

#### 4.32 In-Mask Intensity SD

**Column:** `In-Mask Intensity SD` · **Unit:** `a.u.` · **Field:** `mask_intensity.sd`

**Population** SD (`ddof = 0`) of the raw in-object intensities.

#### 4.33 In-Mask Intensity CV

**Column:** `In-Mask Intensity CV` · **Unit:** dimensionless · **Field:** `mask_intensity.cv`

`sd / mfi` per object. Scale-invariant, so it compares objects of different brightness: the
primary clustering measure. NaN when the object's mean is not positive.

#### 4.34 In-Mask Intensity Skewness

**Column:** `In-Mask Intensity Skewness` · **Unit:** dimensionless ·
**Field:** `mask_intensity.skewness`

Fisher–Pearson third standardized moment, `mean(((x − μ)/σ)³)`, on **raw** values with a
population σ and **no sample-size bias correction**. Positive means a bright tail — foci against
a dimmer body. NaN for fewer than three voxels or zero variance.

#### 4.35 In-Mask Intensity Entropy

**Column:** `In-Mask Intensity Entropy` · **Unit:** dimensionless (bits) ·
**Field:** `mask_intensity.entropy`

1. Rescale the object's voxels to [0, 1].
2. Histogram with `mask_intensity_bins` bins (default **64**) over a **fixed range of [0, 1]** —
   fixed, not data-derived, so every object shares bin widths.
3. Shannon entropy `−Σ p log₂ p` over the non-empty bins. **Base 2, so the unit is bits**, with a
   maximum of `log₂(64) = 6`.

#### 4.36 In-Mask Normalized Entropy

**Column:** `In-Mask Normalized Entropy` · **Unit:** dimensionless, 0–1 ·
**Field:** `mask_intensity.entropy_normalized`

`entropy / log₂(mask_intensity_bins)`; the denominator is 6.0 at the default 64 bins. 1.0 means
a uniformly filled object; it falls towards 0 as signal concentrates into foci.

**Runs are only comparable to each other at the same `mask_intensity_bins`**, since the bin
count sets both the achievable entropy and the denominator.

#### 4.37 In-Mask Fraction Above 2× Median

**Column:** `In-Mask Fraction Above 2x Median` · **Unit:** dimensionless, 0–1 ·
**Field:** `mask_intensity.bright_fraction`

Fraction of the object's raw voxels with `I > 2 × median(I)` — strictly greater. A
threshold-free punctateness measure: it needs no absolute intensity cutoff, so it survives
differences in illumination between datasets. NaN when the object's median is not positive.

---

**Packing topology** — how objects are arranged relative to one another, i.e. who touches
whom. BARCODE otherwise
describes objects individually (volume, sphericity, curvature) and describes their spacing with
a single scalar; nothing described the *topology* of a packing. In a space-filling monolayer
sizes and separations are near-uniform and it is the neighbor-number distribution that
changes, which is the standard epithelial readout.

**Implementation:** `analysis/volumetric/packing.py` · **Opt-in:** `enable_packing_topology`,
default **off** · **GUI:** *Packing Topology* · **CLI:** `--packing` · **Modes:** volumetric
only.

#### 4.38 Preconditions, and why they are refusals rather than fallbacks

Each is reported and skipped, never raised — one misconfigured file must not abort a batch:

- **No segmentation** → skipped.
- **A non-cubic grid** → skipped; see the box below.
- **A boolean mask, or fewer than two distinct positive labels** → skipped, with a message
  telling you to supply an instance segmentation and set `object_partition = "labels"`.

The refusal is the point. In a confluent field every cell touches its neighbors, so deriving
objects by connectivity labeling fuses the whole tissue into one component and the honest
contact number would be reported as 0 — which reads as a *measurement* rather than a
misconfiguration.

> ### ⚠ This family requires an isotropic grid, and is left empty without one
> Contact is counted in **voxel faces**, and gaps are bridged by a **voxel dilation**. On a
> non-cubic grid a z-normal face spans a different physical area from an xy-normal one, so
> the same physical interface passes or fails `packing_min_contact_voxels` **on its
> orientation alone** — making both contact number and hexagonal fraction properties of the
> sampling grid rather than of the tissue.
>
> Unlike Mean Island Anisotropy, which is measured in physical coordinates and so is correct
> on any grid (§1.8), no rescaling repairs this: a single voxel distance cannot mean one
> physical distance on a non-cubic grid.
>
> Above a **1% anisotropy ratio** the family is therefore **left empty, and its columns never
> reach the CSV or the barcode**, with a printed reason. Enable *Resample to Isotropic Voxels*
> (`make_isotropic`) for comparable numbers. (Packing itself still needs an instance mask — see
> §4.38 — but the resampling that makes the grid cubic no longer does.)
>
> The general principle, stated once: **a metric defined only on a cubic grid must not reach
> the output.** Where a fix exists it is fixed; where none does, suppression is the honest
> outcome.

#### 4.39 The contact graph

Shared by all three metrics.

1. **Gap bridging.** If `packing_contact_dilation_vox > 0` (default **1**), every label is
   grown outward by that distance, so objects separated by a thin background gap — a one-voxel
   segmented membrane — still register as neighbors. Done on a copy.
2. **Adjacency test.** Along each of the three axes, adjacent voxels are compared; a contact is
   any position where both are nonzero and different. This is **6-connectivity — shared faces
   only.** Edge- and corner-touching objects are deliberately *not* neighbors; 26-connectivity
   would make every diagonal a contact.
3. **Deduplication**, so `(a,b)` and `(b,a)` unify. The count is the number of shared voxel
   faces.
4. **Noise filter:** pairs sharing fewer than `packing_min_contact_voxels` faces (default **5**)
   are discarded as segmentation noise. The threshold is floored at 1, so setting it to 0
   does **not** disable the filter. Note the face counts are measured on the *dilated*
   volume when gap bridging is active, so they are not raw contact areas.

> **The face-only guarantee holds only at `packing_contact_dilation_vox = 0`.** At the default
> of 1, two labels meeting along an **edge** each grow into the shared diagonal and afterwards
> share a genuine 6-connected face running the whole length of that edge — around 30 voxels for
> a 30-voxel-tall cell, far above the noise threshold, which cannot filter it because it
> measures the *dilated* interface. So with bridging on, edge-touching objects **do** become
> neighbors. Set the dilation to 0 if you need strict face adjacency.

A **contact number** is then the number of *distinct neighboring objects* per label — a degree
in that graph, **not a contact area**.

#### 4.40 Border exclusion

An object at the edge of the array has neighbors outside it, so its degree is an undercount.
With `packing_exclude_border_objects` on (default **true**), a label reaching the edge is
dropped **from the reported set**. Which faces count is set by `packing_border_mode`, default
**`"xy"`** — the four lateral faces only. Ignoring the two z faces is deliberate: in a thin
slab every object touches the top and bottom, so `"all"` (the six-face behavior) would
exclude the entire field. `"none"` reports every object. Its neighbors are still counted
for the interior objects it touches — only the reporting is restricted, not the graph.

Border labels are read from the **undilated** labels, so bridging cannot push an interior
object onto the border.

#### 4.41 Mean Contact Number

**Column:** `Mean Contact Number` · **Unit:** dimensionless ·
**Field:** `packing.contact_number_mean`

Mean degree over interior objects, then averaged over analyzed timepoints. For an ideal 2D
epithelial monolayer this tends to 6.

#### 4.42 Contact Number SD

**Column:** `Contact Number SD` · **Unit:** dimensionless ·
**Field:** `packing.contact_number_sd`

**Population** SD (`ddof = 0`) of the interior degrees — packing disorder. A perfectly ordered
honeycomb gives 0.

#### 4.43 Hexagonal Fraction

**Column:** `Hexagonal Fraction` · **Unit:** dimensionless, 0–1 ·
**Field:** `packing.hexagonal_fraction`

Fraction of interior objects with **exactly six** neighbors. The canonical epithelial packing
readout, and more diagnostic than §4.41, because the mean is pinned near 6 by topology in any
space-filling 2D tiling while the fraction is not.

#### 4.44 NaN rules

All three are NaN, with a stated reason, when fewer than two objects are present ("a packing
needs at least two"), or when every object touches the array border so no interior set remains.

---

### Per object

#### 4.45 Object rows

Everything above describes a **field row**: one row per file, timepoint or z-slice, carrying
every metric the analysis mode produces. When the row axis resolves to `object`, BARCODE emits
a different and much smaller schema instead — one row per segmented object, pooled across
fields, written to `<name> Objects.csv`.

See [Row axes](#row-axes--what-one-row-is) for what a row axis is and how one is chosen. This
section defines the columns.

**The columns are a join, not a new measurement.** Object volume, contact number and the
in-mask statistics already exist per object elsewhere in the run — object volume from a
`bincount` of the label array, the rest from the packing graph (§4.38–4.44) and the in-mask family
(§4.29–4.37) — and the object row joins them by object id. The set is small because most metrics are
field-level *by definition*: there is no per-object connectivity, correlation length, kurtosis
or optical flow, and a column that cannot mean anything is omitted rather than filled with NaN.

The schema has **three tiers** (`analysis/volumetric/objects.py::ObjectResults.get_metrics`).
First, the **three base columns** — always present and always valued (no mesh, no mask flag needed):

> Object rows carry no equivalent-sphere diameter: it is a monotonic function of `Object
> Volume` and adds no information. `Anisotropy` occupies that role instead — a genuinely
> independent shape descriptor.


| Column | Unit | Definition |
|---|---|---|
| `Object Volume` | `μm^3` | This object's **voxel count × voxel volume** — not a mesh volume, and not a fraction of the field. |
| `Anisotropy` | dimensionless, ≥ 1 | Principal-axis elongation: the major/minor ratio of the inertia ellipsoid, from the same clamped-eigenvalue formula as the field's Mean Island Anisotropy (§1.8). **Needs no mesh**, so it is a base object column — the per-object counterpart of the mesh Lateral/Axial Ratio (§4.7), which does. |
| `Contact Number` | dimensionless | This object's degree in the contact graph of §4.39: how many distinct objects it shares a face with. Instance mask + isotropic grid, like §4.38–4.44. |

Second, **five mesh-shape columns — always in the schema, but `NaN` unless `object_mesh` is
on**. With it on, each object is meshed **independently** (`analysis/volumetric/object_mesh.py`),
which is where per-object shape comes from, distinct from the field mesh family (§4.1–4.11) that
meshes only the largest component:

| Column | Unit | Definition |
|---|---|---|
| `Mesh Surface Area` | `μm^2` | §4.3, of this object's own mesh |
| `Sphericity` | dimensionless | §4.4 |
| `Solidity` | dimensionless | §4.9 |
| `Lateral/Axial Ratio` | dimensionless | lateral / axial (§4.7) |
| `Mean Curvature <H>` | `1/μm` | §4.14 |

Third, **seven in-mask intensity columns — present only when `enable_mask_intensity`
(`--mask-intensity`) is on**, and omitted from the CSV entirely otherwise (not written as NaN):

| Column | Unit | Definition |
|---|---|---|
| `In-Mask MFI` | `a.u.` | §4.31, for this object alone |
| `In-Mask Intensity SD` | `a.u.` | §4.32 |
| `In-Mask Intensity CV` | dimensionless | §4.33 |
| `In-Mask Intensity Skewness` | dimensionless | §4.34 |
| `In-Mask Intensity Entropy` | dimensionless (bits) | §4.35 |
| `In-Mask Normalized Entropy` | dimensionless | §4.36 |
| `In-Mask Fraction Above 2x Median` | dimensionless | §4.37 |

So a default `Objects.csv` carries **8 metric columns** (3 base + 5 mesh); adding
`--mask-intensity` brings it to **15**. `object_mesh` is **off by default**; with it off the five
mesh columns are present but NaN. Its `object_mesh_maxrad` is a
**fraction of each object's own radius** (default 0.1), so one setting means the same thing for
objects of different sizes — the `relative` idea of §4.8, applied per object. Objects too small
(`object_mesh_min_voxels`, default 64) or whose meshed volume disagrees with their voxel count
are left without a mesh, and their shape columns are NaN rather than borrowing the field's.

Identity columns are `File`, `FOV` and `Object` — not the `File` / `Channel` / `Flags` triple
a field row carries.

Two consequences of object rows being physical by construction:

- **There is no fraction/quantity split.** A field row reports sizes twice, as a fraction of
  the analyzed field and as a `… Quantity` in µm³ (see the [column index](#column-index--look-up-what-you-have)). An object row reports µm³ directly,
  so the physical-units variant is the same set of columns.
- **Objects carry no flags.** The `Flags` column reads `0`. Flags describe the run that
  produced the objects and live on the field row.

The seven in-mask statistics are the same quantities §4.29–4.37 defines, before the unweighted mean
over objects that a field row applies. An object row is therefore the level at which they were
originally computed; the field row is their summary.

---

### Recording what was analyzed

#### 4.46 Range provenance

Flag digit 5 records *that* the analysis covered only part of the acquired data; these four
columns record *which* part. They also make per-file ranges representable, which the global z/t
settings in `Settings.yaml` cannot express — and they mean a CSV separated from its
Settings.yaml still describes itself.

**Implementation:** `analysis/volumetric/provenance.py` · **Opt-in:** `record_range_columns`,
default **off** · **Modes:** any.

> **You usually do not need these as columns.** The z/t range you set is already in the
> `Settings.yaml` beside the CSV, and flag digit 5 marks *that* a range was restricted. Turn
> this on only when a CSV must be self-describing away from its Settings.yaml, or when different
> files in one run were analyzed over different ranges and you need that recorded per row.

| Column | Unit | Field | Meaning |
|---|---|---|---|
| `Z Range Start` | `slice` | `ranges.z_start` | First analyzed slice, **inclusive** |
| `Z Range End` | `slice` | `ranges.z_end` | Last analyzed index, **inclusive** |
| `T Range Start` | `slice` | `ranges.t_start` | First analyzed timepoint, inclusive |
| `T Range End` | `slice` | `ranges.t_end` | Last analyzed index, inclusive |

Three properties worth stating explicitly:

- **An unrestricted axis reports its full extent, not NaN.** On a 54-plane stack, "0 to 53"
  and "no range set" are the same statement, so the columns are always readable as a range.
  Both ends are inclusive, so the last index is `length − 1`, not `length`.
- **Indices are into the acquired data, before any isotropic resampling.** A mask on a
  250-plane 0.065 µm grid (a staged mask uses `round(n·dz/dxy)`, where the resampled
  image uses `floor((n−1)·dz/dxy + 1)` = 245) and an image on a 54-plane 0.3 µm grid
  disagree about what "slice 46"
  means; these columns always mean the acquired image grid. The range may have been *specified*
  in `acquired`, `isotropic` or `microns` units — it is converted before it reaches these
  columns.
- The two time columns carry the unit label `slice`, which is a mild misnomer: they are
  timepoint indices.

---

## `xyz` mode — reading a stack down its depth axis

Sections 1–4 define what each metric measures. This section is about the mode: what changes
when the third axis is **depth rather than time**, and what the numbers mean once it does.
Read it if you have a Z-stack and you want per-slice structure as a function of depth, rather
than one number for the whole volume.

### The one idea

The 2D branches take a **stack of 2D images** and reduce it to one row: some metrics average
over the stack, some compare its start against its end, some count how many members satisfy a
condition. Published BARCODE assumes that stack is a **time series**. `xyz` feeds it a
**Z-stack** instead. The arithmetic is the same; what changes is what the axis *means*.

| | `xyt` (published) | `xyz` |
|---|---|---|
| One member of the stack is | a timepoint | a **z-slice** |
| "First 5% / last 5%" means | earliest and latest frames | **shallowest and deepest slices** |
| A Change metric measures | evolution over time | **variation with depth** |
| `Connectivity` counts | frames that percolate | **slices** that percolate |
| Optical flow measures | motion | *nothing meaningful — disabled* |

That last row is why the mode exists as a named thing rather than as "just point the 2D
pipeline at a stack". A Z-stack in `xyt` does not fail; it reports displacement between focal
planes as a velocity in µm/s and growth along depth as growth over time. `xyz` makes the
choice explicit and renames the affected columns so the misreading cannot survive into a
figure.

### Which of the three modes you want

| You have | You want | Mode |
|---|---|---|
| A 2D movie | the original behavior | `xyt` |
| A Z-stack, and you care how structure varies **with depth** | 2D metrics per slice, reduced over z | **`xyz`** |
| A Z-stack, and you care about the **3D object** | true volumes, 3D connectivity, meshing, 3D flow | `xyzt` |

`xyz` and `xyzt` answer different questions about the same file. `xyzt` asks "what is this
object" — one volume, one row, 3D quantities. `xyz` asks "how does the cross-section change as
I go down" — many slices, reduced to one row per timepoint, every quantity in-plane. A stack
of **one** timepoint is a perfectly normal `xyz` input; you get one row.

### What `xyz` emits — 29 columns

Three identity columns (`File`, `Channel`, `Flags`) plus 26 metrics:

- **Binarization (12)** — `Connectivity`, `Maximum Island Area`, `Maximum Void Area`,
  `Maximum Island Area Change (over Z)`, `Maximum Void Area Change (over Z)`,
  `Initial Maximum Island Area`, `Initial 2nd Maximum Island Area`, `Mean Island Anisotropy`,
  `Mean Island Area`, `Total Island Area`, `Mean Island Separation`,
  `Structural Correlation Length`
- **Intensity (6)** — `Maximum Kurtosis`, `Maximum Median Skewness`, `Maximum Mode Skewness`,
  `Kurtosis Change (over Z)`, `Median Skewness Change (over Z)`, `Mode Skewness Change (over Z)`
- **Intensity magnitude (4, opt-in)** — `Total Intensity`, `Mean Intensity`, `Intensity SD`,
  `Intensity Density (per area)`
- **Range provenance (4, opt-in)** — `Z Range Start/End`, `T Range Start/End`

**What it does not emit, and why.** The seven optical-flow columns are omitted from the schema
entirely, not written as NaN: displacement between adjacent focal planes is µm of structural
shift per µm of depth — a real quantity, but not a velocity, and there is no honest way to
label it µm/s. Mesh, curvature, packing, in-mask intensity, component statistics and the depth
profile are properties of a 3D object, which `xyz` never assembles; they are hard-gated to
`xyzt`. And everything is an **area** in µm², never a volume, because every measurement is
made inside a single plane.

### How a run works, step by step

Per input file, `analysis/volumetric/slicewise.py`:

1. **Read the volume**, with the axes declared rather than inferred (§P.1).
2. **Apply the t range first** — excluded timepoints then get no mask or geometry work.
3. **Load the mask, if any, against the full acquired stack**, then apply the z range and
   slice the mask by the same indices. Validating a whole-depth mask against an
   already-restricted image would reject a good mask.
4. **Set the spatial scale to the XY pixel size.** Every `xyz` metric is in-plane, so the
   scale is `xy_step_um` read from the file — never the z step, and never whatever the 2D tab
   holds.
5. **For each timepoint**, hand the `(Z, Y, X)` array to the unmodified 2D branches, which
   iterate over its first axis. That axis is depth. **One row per timepoint.**

There is no isotropic resampling and no cropping in this mode. A mask is matched to the
acquired slice grid by nearest-neighbor index mapping, so mask slice *i* lines up with image
slice *i* — the mask comes to the data rather than the other way round.

**How depth is reduced to one number** — whichever reduction the 2D branch already used, now
running over slices:

| Reduction | Metrics | Meaning in `xyz` |
|---|---|---|
| Mean over members | Mean Island Area, Total Island Area, Anisotropy, Separation, Structural Correlation Length | average over analyzed slices |
| Top-decile mean | Maximum Island/Void Area, Maximum Kurtosis, both Max Skewnesses | mean of the ~10% of **slices** with the largest value |
| Fraction of members | Connectivity | fraction of **slices** containing a percolating path |
| First *X* % | Initial Maximum Island Area, Initial 2nd Maximum | mean over the **shallowest** slices |
| Last minus/over first *X* % | the five `(over Z)` metrics | **deepest versus shallowest** |

> **`Maximum Island Area` is a top-decile mean, not a maximum.** It is the mean of the largest
> ~10% of per-slice values, so a single unusual slice cannot set it. The same is true of
> `Maximum Kurtosis` and both maximum skewnesses. This is inherited from published BARCODE and
> is the commonest misreading of any BARCODE CSV.

### What "Change (over Z)" actually measures

Five columns carry the suffix: `Maximum Island Area Change (over Z)`,
`Maximum Void Area Change (over Z)`, `Kurtosis Change (over Z)`,
`Median Skewness Change (over Z)`, `Mode Skewness Change (over Z)`.

With `n_eval = ⌈percentage_frames_evaluated × n_analyzed⌉` — default fraction **0.05**, over
the count of **analyzed** slices (after `frame_step`), not the acquired total — each compares
the **deepest** `n_eval` analyzed slices against the **shallowest** `n_eval`. The two
binarization metrics form a **ratio** (deep ÷ shallow; 1.0 means no change); the three
intensity metrics form a **difference** (deep − shallow).

So a value above 1 for `Maximum Island Area Change (over Z)` means the largest island is
bigger deep in the stack than near the top. **It says nothing about time.** With several
timepoints you get one such value per timepoint, and a trend over time is read *down the
rows*, not out of this column. These are NaN when only one slice is analyzed.

The rename is the entire safety mechanism: a depth trend read as a time trend is a silent
scientific error, because the numbers look completely ordinary.

### Working with a segmentation

A mask changes two things beyond replacing the intensity threshold.

**Empty slices are dropped.** Above and below an object the mask is empty, and an empty slice
has mean 0, hence a threshold of 0, hence the *entire field* marked as one island — and empty
slices are the norm around a nuclear mask. Only slices the segmentation actually occupies are
analyzed, and a segmentation empty on every slice raises rather than producing numbers.

**This narrows the progression axis.** The `(over Z)` metrics therefore compare the first and
last **occupied** slices, not the first and last of the stack. That is the more meaningful
comparison, but it means a masked and an unmasked run of the same file are not measuring
across the same depth span.

**Masked intensity is built directly from in-mask pixels per slice.** The obvious approach —
blanking out-of-mask voxels with NaN and handing the volume to the 2D branch — fails, because
`np.histogram` derives its range from the data's min/max and raises on all-NaN input. The
masked path histograms the in-mask pixels of each occupied slice directly.

### Per-slice barcodes — one row per slice

`xyz` reduces depth to a single row. To *see* the depth profile instead:

```
python scripts/run_xyz_slice_barcodes.py <folder-or-file> --z-start 12 --z-end 46
```

This writes **one barcode per timepoint**, with rows = z-slices (shallowest at the top),
columns = the same 2D metrics, so reading **down a column** shows how that metric varies with
depth. Each row is labeled with its absolute slice index and physical depth, e.g.
`Cell1_1.tif z=23 (6.90um)`. Z indices refer to **acquired** slices and the range **includes
both ends**, so on 0.3 µm data `--z-start 12 --z-end 46` is 35 slices covering 10.5 µm. The
five Change columns are constant-NaN in this layout — a row is one slice — so the depth trend
appears as the gradient of the *other* columns. Outputs go to a `results/xyz_per_slice` folder
beside the source data, never inside it.

### What you must not read into these numbers

Everything in `xyz` is measured **inside a plane**:

- **`Structural Correlation Length` is an in-plane length.** It says nothing about correlation
  along z. Two slices could be completely unrelated and the metric would not notice.
- **`Connectivity` is per-slice percolation**, spanning x or y within one plane. An object
  continuous through depth but not across any single slice scores 0. In `xyzt` the same metric
  tests all three axes with 26-connectivity and would score 1.
- **`Mean Island Anisotropy` is a 2D ellipse ratio** of a cross-section. A cigar standing on
  end reads as circular in every slice.
- **Islands are 2D regions.** One object cut by three slices contributes three islands, so
  island counts and `Mean Island Separation` describe cross-sections, not objects.
- **There is no object.** No volume, no surface, no sphericity, no curvature — those need the
  3D assembly `xyzt` does and this mode deliberately does not.

If any of those is the question you actually have, you want `xyzt`.

### Gotchas

**`frame_step` subsamples slices, and the default is 10.** In `xyz` it strides the depth axis,
so a 54-slice stack at the default is analyzed at **7 slices** — indices 0, 10, 20, 30, 40, 50
and the final slice 53, which is always appended. That is usually not what you want when the
whole point is a depth profile; set it to 1 for a full profile, at proportionate cost. (The
masked paths use the volumetric selector, which handles small counts safely; the unmasked
paths still route through the 2D helper — see §P.6.)

**`percentage_frames_evaluated` is a fraction of the *analyzed* slices.** At the default 0.05
on 7 analyzed slices, `n_eval` rounds up to 1, so the `(over Z)` metrics compare exactly one
deep slice against one shallow one. Widen it, or lower `frame_step`, for a more robust
comparison.

**The z range is in acquired slice indices by default.** `z_range_units` also accepts
`isotropic` and `microns`; in `xyz` there is no isotropic grid, so `acquired` and `microns`
are the meaningful choices.

**Flag digit 1 never fires in `xyz`.** A dim channel is unflagged in this mode — it is raised
in `xyt` and in `xyzt`, but nothing sets it on the slice-wise path.

**"Parse All Channels" does nothing.** Like all volumetric modes, `xyz` analyzes
`selected_channel` alone. Run other channels separately.

**Rows are timepoints, not files, when a series is grouped** — but time-lapse grouping is
`xyzt`-only (§P.7), so in `xyz` a multi-timepoint file gives one row per timepoint and
separate files stay separate.

---

## Row axes — what one row is

`analysis_mode` names what BARCODE *measures*. The **row axis** names what it *compares*.

### Why it matters

**The barcode normalizes each column across its rows.** The rows *are* the comparison. Get
them wrong and the picture is either empty of information — one row is a single flat stripe,
with nothing to normalize against — or quietly misleading, because two figures normalized over
different sets invite a comparison their colors do not support.

The right axis is a property of the data, not a matter of taste. A *Drosophila* embryo is
~840 cells in one field, so the interesting comparison is between **objects**; comparing
fields would be comparing one number against nothing. A Jurkat nucleus is one object per field
imaged over time, so the only available comparison is between **timepoints**. Both are `xyzt`
runs on the same pipeline — nothing about the *mode* distinguishes them, only the row axis
does. So when you have not chosen, it is resolved from the data, and the choice is **printed
and recorded** rather than assumed.

### The axes

Set with `row_axis` (YAML) or `--rows` (scripts).

| Axis | One row is | Use when |
|---|---|---|
| `file` | one input file / field of view | the comparison is between acquisitions — BARCODE's original behavior |
| `timepoint` | one timepoint | a field holds a single object and you want its time course |
| `slice` | one z-slice | you want a depth profile within one timepoint (`xyz` only) |
| `object` | one segmented object, pooled across fields | the field holds many cells and the comparison is between them |
| `auto` | *resolved from the data* | the default |

Reading down a column means something different for each: a time course for `timepoint`, a
depth profile for `slice`, a population distribution for `object`, a between-acquisition
comparison for `file`.

### `auto`, and what it guarantees

The resolution order is deliberately short:

1. **an instance segmentation resolved AND more than one object** → `object`
2. else **more than one timepoint** → `timepoint`
3. else → `file`

Many objects beats many timepoints because a field of cells is almost always asking a per-cell
question, while a single object over time is asking a temporal one.

> **`auto` can never reach `object` without a segmentation.** A run with no mask resolves to
> `file` exactly as BARCODE has always behaved, so the published 2D reference outputs are
> unaffected by this feature existing.

The resolved axis is printed along with what the colors were normalized over — e.g.
*normalized across 840 objects from 1 field* — because a barcode's color scale is meaningless
without it, and two figures built over different sets are not comparable.

### Field scope vs object scope

Each axis carries a **scope**, and the scope decides the column set. `file`, `timepoint` and
`slice` are **field scope**: one row describes a whole field and carries every metric the
analysis mode produces. `object` is **object scope**: one row describes a single object and
carries only the metrics *defined* for a single object — the schema of §4.45.

Most metrics are field-level by definition. There is no per-object connectivity, no per-object
correlation length, no per-object kurtosis and no per-object optical flow; those describe a
field or a whole volume. Following the rule the analysis modes already use, **a column that
cannot mean anything is omitted rather than filled with NaN** — or, worse, filled with the
field's value repeated down every row, which would look like data.

That is why an object barcode is narrower than a field barcode. It is not a reduced version of
the same picture; it is a different comparison.

### When it refuses

An explicitly chosen axis the data cannot support is an **error, not a cue to fall back**.
Silently choosing a different axis would change what the figure compares without saying so.

| You asked for | With | It says |
|---|---|---|
| `object` | no instance segmentation | there are no objects to be rows — supply a mask or choose another axis |
| `object` | exactly one object | a single object is one row, which a barcode cannot normalize — use `timepoint` or `file` |
| `slice` | a mode other than `xyz` | that axis is only available in `xyz` |

### When a barcode is the wrong picture

A barcode needs a population. With one row every column is a flat, uninformative stripe — and
that is the common case for volumetric work: a single stack, or a single object imaged over
time.

For that, use the **fingerprint** (`write_fingerprint`, or `scripts/run_fingerprint.py`): a
one-page report on a single analyzed volume, carrying what a barcode cannot — orthogonal
projections of the volume that was actually analyzed, after any z range and resampling, so the
numbers can be checked against what they describe, plus the distributions behind the scalars.
It is off by default because it is a per-volume *report*: comparing across fields, objects and
timepoints is what the barcode is for, and a document per volume does not help you find
structure across a hundred of them.

---
## Reading the outputs

### What a volumetric run writes

A dataset-level `<name> Summary.csv`, a `<name> Summary Barcode.png`, and a
`<name> Settings.yaml` recording the configuration that produced them, all in the folder that
was processed. Several harnesses additionally write a `… (physical).csv` carrying the µm²/µm³
Quantity columns beside the normalized one.

When objects were extracted, a **`<name> Objects.csv`** is written too — one row per segmented
object, in the smaller schema of §4.45. It is written whenever objects exist, even if you asked
to compare something else; the object *barcode* is drawn only when the row axis resolved to
`object`. See [Row axes](#row-axes--what-one-row-is).

With `write_fingerprint` on, each analyzed volume also gets a **fingerprint card**: a one-page
report carrying orthogonal projections of the volume that was actually analyzed — after any z
range and resampling — alongside the metric table and the distributions behind the scalars. It
exists for the case a barcode cannot serve, and is off by default because it is a per-volume
document rather than a comparison. `fingerprint_dpi` (default 110) sets its resolution.

**`xyzt` writes no per-file outputs.** That path returns before the 2D code that creates
`<file> BARCODE Output/`, so there are no reduced data structures, per-frame graphs or
per-file figures. **`xyz` is different**: an *unmasked* run with *Save Graphs* on does write
`<file> BARCODE Output/Channel N/` from the 2D branches, while a *masked* one takes a
different path and writes none — so whether per-file output appears depends on whether a mask
resolved.

Optional per-file artifacts: *Export Mesh as .OBJ* writes `<stem> BARCODE Meshes/` beside the
input, and `write_fingerprint` writes `<stem> Fingerprint.png` there too.

Keep the `Settings.yaml`. Several metrics are only interpretable against the settings that
produced them — normalized entropy depends on the bin count, correlation lengths on the voxel
size, Speed on the frame interval — and the CSV does not carry those values. Turning on
`record_range_columns` embeds at least the analyzed z/t ranges into the CSV itself (§4.46).

### Reading the barcode PNG

**Rows are whatever the row axis says they are** — files, timepoints, z-slices or objects
(see [Row axes](#row-axes--what-one-row-is)). Columns are metrics; color is the value, on the **plasma**
colormap. Because normalization is per column *across rows*, the row axis decides what the
picture actually compares, and the resolved axis is printed with the run. **NaN renders black**, which is worth knowing before reading a black cell as "low".

Color scaling is per column, but *how* each column is scaled depends on its unit, and only
some columns are normalized against the data:

| Column type | Color range |
|---|---|
| `fraction of FOV`, `fraction of frames` | **fixed** `[0, 1]` — comparable between runs |
| `Mean Flow Direction` | **fixed** `[−π, π]` |
| `Directional Spread` | **fixed** `[0, π]` — values above π are clipped in the image only |
| `Mean Island Anisotropy` | `[1, max]` |
| ratio to initial | data range, always including 1 (no change) |
| Curvature, Divergence, Curl, Speed Change | data range, always including 0 (they are signed) |
| Other physical units (µm, µm², µm³, a.u., slice) | `[0, max]` |
| Remaining dimensionless | data range, always including 0 |

So a column with a **fixed** range can be compared between two barcodes; a column scaled to
the data cannot, because its extremes are set by whatever rows happen to be in that figure.
The dynamic ranges are taken across every row rendered, so **adding or removing files
recolors those columns.** Read the CSV when comparing across figures.

### My column is all NaN — what to check

Every NaN condition is documented in its metric entry; this is the same information indexed by
the symptom. NaN is almost always a refusal to report a number that would be wrong, not a
crash.

| Symptom | Most likely cause |
|---|---|
| **Every Change column** (`… Change`, `… Change (over Z)`) | A single analyzed timepoint. Change needs two points on the progression axis; the branch returns NaN rather than comparing a volume with itself. Group per-timepoint files with `timelapse_enabled` (P.7). |
| **The seven flow columns are absent** (dropped, not NaN) | A static z-stack — a single timepoint, or a series shorter than the 7-volume flow window. The columns are omitted entirely rather than emitted as NaN (§3.0). |
| **Speed Change only** | Exactly one window survived the edge exclusion. Common at the default `frame_step`. |
| **All mesh and curvature columns** | `mesh_enabled` off, a non-isotropic grid, an empty binary at every timepoint, or `MeshingError` — each prints a reason (§4.1). A missing segmentation does not cause this: meshing falls back to the binarized volume. |
| **All three packing columns** | In order of likelihood: a **non-cubic grid** (the family is suppressed entirely there), a boolean rather than instance mask, fewer than two objects, or every object touching the border. Each prints which (§4.38, §4.44). |
| **Curvature columns only, mesh columns fine** | `mesh_curvature` off, or every face was excluded by the bottom/top and outlier rules (§4.13). |
| **`Mean Island Separation`** | Fewer than two objects in a timepoint — the normal case for a single segmented nucleus (§1.11). |
| **`Structural Correlation Length`** + flag 3, or **`Velocity Correlation Length`** + flag 4 | The correlation never fell below its threshold within the field of view, so the length exceeds what the data can measure (§1.12, §3.7). |
| **All seven in-mask intensity columns** | No segmentation, or every object below `mask_intensity_min_voxels` (§4.29). |
| **All six intensity columns in `xyz`** | Possibly the unguarded histogram error in the masked `xyz` path (Known limitations) — check the log for `IndexError`. |
| **`Island Volume Skewness`** | Fewer than three objects, or zero variance among their sizes (§4.21). |
| **Every physical `… Quantity` column absent** | That CSV is the normalized set; the µm³ columns go to a separate `… (physical).csv`. |

### Where to go for other things

- **How to run it** — [Running a volumetric analysis](#running-a-volumetric-analysis) walks
  the GUI tab by tab and gives the equivalent CLI; the `scripts/` harnesses take the same
  settings as command-line flags. `--list-metrics` prints exactly which columns a given
  configuration will emit before you commit to a batch.
- **Why to believe the numbers** — [How the numbers were checked](#how-the-numbers-were-checked)
  records the analytic ground truths, cross-implementation checks and invariance properties
  each branch was tested against.
- **A stack read down its depth axis** — [`xyz` mode](#xyz-mode--reading-a-stack-down-its-depth-axis).
- **What one row of the barcode is** — [Row axes](#row-axes--what-one-row-is).
- **What is provisional** — [Known limitations and caveats](#known-limitations-and-caveats).
  Read it before publishing; the optical-flow branch in particular is provisional (§3).

---

## Warnings and flags

The `Flags` column extends the published 0–4 scheme with three volumetric digits.

> **Flags are emitted semicolon-joined, not as a single digit.** A row with a restricted z
> range whose structural correlation failed reads **`3;5`**. `"0"` means no flag fired. Any
> parser that reads this column as an integer will be wrong on multi-flag rows.

| Digit | Meaning | Raised by |
|---|---|---|
| **0** | No flag fired. | — |
| **1** | **Dim channel.** `2·e⁻¹·mean ≤ min` — the darkest voxel is already a sizeable fraction of the average, so there is barely a foreground to separate. Judged on the first frame in 2D and on the **first analyzed volume** in `xyzt`, so a t-range restriction is respected. **Still never fires in `xyz`.** | reader (2D), `run.py` (`xyzt`) |
| **2** | **Saturation.** The histogram's mode sits in the highest surviving bin, in **every** analyzed timepoint. A timepoint whose histogram has no surviving bin counts as *not* saturated, and a run with no analyzed timepoints scores 0 rather than 1. Clipped intensity prevents accurate structural analysis and makes Total Intensity (2.7) meaningless. | section 2 |
| **3** | **Structural correlation failure.** ξ_I could not be determined for at least one analyzed timepoint. | section 1.12 |
| **4** | **Velocity correlation failure.** ξ_v could not be determined for at least one analyzed window. | section 3.7 |
| **5** | **Range restricted.** A z or t range was applied, so the numbers describe a subset of the acquired data. *You* narrowed the analysis. | preconditions P.5 |
| **6** | **Field-of-view clipping.** Foreground reaches an edge of the analyzed field, so the object continues outside it. The *data* is cut off. Every size, shape and curvature metric describes a truncated object. | §4.27 |
| **7** | **Open mesh surface.** A mesh came out with a boundary rather than closed. `Mesh Volume`, `Sphericity` and the curvature **signs** are unreliable for that row — an unclosed surface has no well-defined inside. | §4.1 |

Three subtleties in the correlation flags:

- **Digits 3 and 4 fire if *any single* analyzed timepoint failed**, not only if all did — the
  test is NaN-propagating. The metric itself is a NaN-dropping mean, so it can be finite while
  its flag is raised. That combination means "most timepoints worked, at least one did not".
- Digit 3's documented meaning is "correlation length exceeds the field of view", which is the
  usual cause but not the only one: an empty radial bin also blocks a threshold crossing
  (section 1.12).
- **A skipped flow branch produces NaNs without flag 4.** If the series is shorter than the flow
  window, or no window center survives the edge exclusion, the branch returns default results
  with the flag at 0. NaN flow columns with no digit 4 mean "flow never ran", not "flow ran and
  could not resolve a length".

---

## Configuration reference

Complete `VolumetricConfig`. Defaults are what a run uses unless a `Settings.yaml` or a CLI flag
overrides them. GUI labels are from the *Volumetric Settings* tab.

### Mode and geometry

| Field | Default | Meaning |
|---|---|---|
| `analysis_mode` | `"xyt"` | `xyt` / `xyz` / `xyzt`. |
| `row_axis` | `"auto"` | What **one row** of the barcode is: `auto` / `file` / `timepoint` / `slice` / `object`. Decides what the per-column normalization compares — see [Row axes](#row-axes--what-one-row-is). |
| `object_mesh` | `False` | Mesh each object independently, adding five shape columns to the object rows (§4.45). |
| `object_mesh_maxrad` | `0.1` | Radbound as a **fraction of each object's own radius**, so one value fits all sizes. |
| `object_mesh_min_voxels` | `64` | Objects smaller than this are not meshed; their shape columns stay NaN. |
| `object_mesh_limit` | `0` | 0 = mesh every object; N = the first N, for iterating on a large field. |
| `enabled` | `False` | Superseded by `analysis_mode`; kept so pre-mode YAMLs load. `enabled: true` with no `analysis_mode` migrates to `xyzt`. |
| `z_step_um` | `0.0` | Z voxel size; 0 = read ImageJ `spacing`. |
| `xy_step_um` | `0.0` | XY pixel size; 0 = read `XResolution`. |
| `axes_override` | `""` | True axis order for a file whose header is wrong. Empty = trust the file. |
| `make_isotropic` | `True` | Resample image + mask onto one isotropic grid. **Required for meshing.** |
| `crop_to_mask` | **`False`** | Crop the analyzed field to the mask bounding box. Off by default — it gives each file its own fraction-of-volume denominator (P.4). |
| `crop_padding_vox` | `2` | Padding around the mask bounding box, when cropping is on. |

### Ranges

| Field | Default | Meaning |
|---|---|---|
| `z_start` / `z_end` | `0` / `0` | Analyzed z range, **inclusive of both ends** (12→46 gives slices 12–46). `end = 0` means "to the end"; negatives count back, `-1` = last. |
| `z_range_units` | `"acquired"` | `acquired` indices, `isotropic` indices, or `microns` of depth. |
| `t_start` / `t_end` | `0` / `0` | Analyzed timepoint range. |
| `t_range_units` | `"index"` | `index`, or `seconds` **via the configured `frame_interval_s`**. It falls back to the file's `finterval` only when the file actually stated timing, and otherwise **raises** rather than silently assuming 1 s. |
| `record_range_columns` | `False` | Emit the four provenance columns (§4.46). |

### Segmentation

| Field | Default | Meaning |
|---|---|---|
| `segmentation_enabled` | `False` | Use masks. A resolved mask **replaces** intensity thresholding. |
| `segmentation_root` | `""` | Root folder; empty = the image's own directory. |
| `segmentation_regex` | `(?P<stem>.+)` | Named groups extracted from the image stem. |
| `segmentation_template` | `{stem}_SegMask.tif` | Path template built from those groups. |
| `mask_spacing_um` | `0.0` | Mask voxel spacing; **0 = "assume isotropic at the image's xy step"**. |
| `mask_format` | `"auto"` | Dispatch on suffix. Supported: `.tif`/`.tiff`, `.png`/`.bmp`, `.npy`, `.npz`. |
| `segmentation_label_mode` | `"binary"` | **Deprecated**, kept for old YAMLs. `labels` forces instance-label preservation. `object_partition` **wins outright**; this flag is consulted only when `object_partition = "auto"` (§P.3). |
| `segmentation_secondary_root` / `segmentation_secondary_template` | `""` / `""` | Optional **second** mask for the same image — e.g. a nucleus *and* a cell outline. Same root/template resolution as the primary. Empty = no secondary mask. |
| `object_partition` | `"auto"` | `labels` / `connectivity` / `auto` (labels when the mask has >1 positive value, else `segmentation_label_mode`). The current control over label preservation; overrides `segmentation_label_mode`. |
| `per_object_rows` | `False` | Legacy boolean; superseded by `row_axis = "object"` (§4.45, [Row axes](#row-axes--what-one-row-is)), which is the mechanism that actually emits `<name> Objects.csv`. |

### Time-lapse

| Field | Default | Meaning |
|---|---|---|
| `timelapse_enabled` | `False` | Group per-timepoint files into one series, restoring the Change metrics. |
| `timelapse_regex` | `^(?P<series>.+?)_(?P<frame>\d+)$` | Needs a `series` group and a numeric `frame` group. |

### Structural branch (section 1)

| Field | Default | Meaning |
|---|---|---|
| `threshold_offset` | `0.1` | Threshold = `mean × (1 + offset)`. |
| `minimum_island_size` | `1` | Objects and holes ≤ this many voxels are removed. |
| `neighbor_island_fraction` | `0.1` | Fraction of objects used as k for Mean Island Separation. |
| `frame_step` | `10` | Stride between analyzed timepoints. |
| `percentage_frames_evaluated` | `0.05` | Head/tail window for every Change metric. |
| `invert_binarization` | `False` | Swap islands and voids. **Raises on a label mask.** |
| `enable_component_stats` | `False` | §4.19–4.22. |

### Intensity branch (section 2)

| Field | Default | Meaning |
|---|---|---|
| `bin_size` | `300` | Histogram bins. |
| `noise_threshold` | `5e-4` | Bins below this probability are discarded. |
| `intensity_use_mask` | `False` | Restrict the histogram **and** the magnitude family to in-mask voxels. |
| `enable_intensity_magnitude` | `False` | Metrics 2.7–2.10. |

### Flow branch (section 3)

| Field | Default | Meaning |
|---|---|---|
| `flow_xyz_sigma` | `3.0` | Spatial Gaussian-derivative σ, in voxels. |
| `flow_t_sigma` | `1` | Temporal σ. **Sets the window to `6σ+1` = 7 volumes.** |
| `flow_w_sigma` | `4.0` | Lucas-Kanade neighborhood σ, in voxels. |
| `flow_reliability_percentile` | `50.0` | Drop voxels below this percentile of reliability. **0 keeps all.** |
| `flow_use_mask` | `True` | Restrict pointwise metrics to in-mask voxels. |
| `flow_downsample` | `1` | Block-average volumes before solving. **Changes Curl, Divergence and ξ_v — see 3.1.** |
| `frame_interval_s` | `0.0` | **Set this.** Seconds per timepoint. 0 uses the file's timing only where the reader trusts it (ND2 time loop); a grouped time-lapse never does, and falls to 1.0 s. Also governs `t_range_units="seconds"` (see *Ranges*). |

### Mesh and curvature (section 4)

| Field | Default | Meaning |
|---|---|---|
| `mesh_enabled` | `False`¹ | Mesh the segmented surface. |
| `mesh_maxrad` | `5.0` | cgalsurf radbound — the single biggest control on mesh accuracy (§4.8). |
| `mesh_maxrad_units` | `"voxels"` | How to read it: `voxels`, `um` (constant physical size), or `relative` (a fraction of this object's equivalent radius). `relative` is config/CLI only — the GUI offers the first two. |
| `mesh_isovalue` | `0.5` | Where the isosurface sits in a 0/1 mask — the foreground/background boundary. Set 0.99 to match the MATLAB pipeline (§4.1). |
| `mesh_area_frac` | `0.2` | Face-area filter driving the decimation ratio. |
| `mesh_smoothing_iterations` | `10` | Laplacian-HC iterations; 0 disables. |
| `mesh_smoothing_alpha` / `mesh_smoothing_beta` | `0.1` / `0.5` | HC parameters (MATLAB pipeline defaults). |
| `mesh_matlab_compat` | `False` | Use GIBBON's quad-fan area convention for the decimation ratio, as the MATLAB pipeline did (§4.3). On only to reproduce those numbers. |
| `mesh_curvature` | `True` | Compute principal curvatures and the invagination metrics. |
| `curvature_exclude_caps` | `False` | Exclude faces in the lowest/highest 0.1 µm z bin (§4.13). Turn on when the object is clipped by the stack ends. |
| `curvature_outlier_limit` | `0.0` | Exclude faces with \|H\| above this, in µm⁻¹; 0 keeps all. MATLAB uses 2.0 (§4.13). |
| `mesh_iso2mesh_bin` | `""` | iso2mesh `bin/` to stage from; empty = let pyiso2mesh find or download. |
| `enable_curvature_range` | `False` | §4.17–4.18. |
| `mesh_aggregation` | `"largest"` | Only `largest` is implemented; anything else **raises** rather than silently behaving as `largest`. |
| `mesh_export_obj` | `False` | Writes one OBJ per analyzed timepoint into `<stem> BARCODE Meshes/` beside the input. |

¹ The config field defaults to `False`, but the family registry marks mesh as *supported by
default* for `xyzt`, so the schema advertises the columns; whether they are filled depends on
this switch and on a segmentation resolving.

### Optional families

| Field | Default | Section |
|---|---|---|
| `enable_packing_topology` | `False` | §4.38–4.44 |
| `packing_contact_dilation_vox` | `1` | §4.39 |
| `packing_min_contact_voxels` | `5` | §4.39 |
| `packing_exclude_border_objects` | `True` | §4.40 |
| `packing_border_mode` | `"xy"` | §4.40 — which faces count as the edge: `xy` (lateral only), `all` (six faces), `none`. |
| `enable_slice_profile` | `False` | §4.23–4.28 |
| `enable_mask_intensity` | `False` | §4.29–4.37 |
| `write_fingerprint` | `False` | Write a one-page per-volume report beside the input (see *Reading the outputs*). |
| `fingerprint_dpi` | `110` | Resolution of that report. |
| `mask_intensity_bins` | `64` | §4.35–4.36 |
| `mask_intensity_min_voxels` | `8` | §4.29 |

---

## Running a volumetric analysis

### From the GUI

```
mamba activate barcode
python main.py
```

Home → **Process Data**. Five tabs; the volumetric one is **Volumetric Settings**. The on/off
switch is **not** on that tab — it is **Volumetric Analysis** on *Execution Settings*, so the
2D-or-3D decision is made once, before anything else. Everything on the Volumetric tab is
inert until it is ticked, so the 2D workflow is unchanged when it is off.

### Four recipes, each adding one thing

Work down the list — a failure then tells you which piece broke. Every value below was
verified end to end on `Cell1_1.tif` / the Cell1 series; if your run reproduces them, the
whole path is working.

**A. Single file, no mask** — proves the reader and the 3D metrics. *Execution Settings*:
**Process File** → Browse File; Choose Channel `0`; tick Image Binarization + Intensity
Distribution; Verbose on; tick **Volumetric Analysis**. Nothing else needed on the Volumetric
tab. The log line that proves it read a stack rather than a movie:

```
Volumetric: (54, 312, 303) @ (0.3, 0.065, 0.065) um
```

`Cell1_1 Summary.csv`, **21 columns** (12 binarization + 6 intensity + 3 identity):
Maximum Island Volume 0.1374 · Mean Island Anisotropy 6.5410 · Mean Island Separation 2.2494 ·
Structural Correlation Length 3.9969 · Maximum Kurtosis 9.4707. About 3 s. Change metrics are
NaN and the seven flow columns are **absent** — both correct for one volume (§3.0). The column
is `Maximum Island Volume`, not `… Area`: `xyzt` measures volumes. The analyzed stack is
resampled to isotropic voxels (54 → 245 z-slices), so these describe the resampled grid, not
the acquired 54-plane one (§P.4).

**B. Add the mask** — proves pairing and resampling. Tick **Use Segmentation Masks** and set
**Segmentation Root Folder**. Leave Filename Pattern and Mask Path Template at their defaults
if the masks are named `<stem>_SegMask.tif`. Still **21 columns** — a mask replaces the
binarization, it does not add columns: Maximum Island Volume 0.0893 · **Mean Island Anisotropy
1.4623** (was 6.5410) · Mean Island Separation NaN · Structural Correlation Length 3.9969.
About 4.5 s. **Anisotropy falling 6.54 → 1.46 is the signal that the mask loaded** and
isotropic resampling engaged; if it stays near 6.5 the mask did not load, so check the log.
`Maximum Island Volume` is a fraction of the **full** field (≈8.9%, since cropping is off by
default), and `Mean Island Separation` is NaN because a single nucleus is one object (§1.11).

**C. Add meshing** — 11 more columns. Tick **Mesh the Segmented Surface** and **Measure
Surface Curvature**. The CSV grows to **32 columns**: Mesh Volume 576.9 µm³ · Mesh Surface
Area 402.6 µm² · Sphericity 0.8324 · Mesh Volume Ratio 0.9951 · Mean Curvature 0.1852 ·
Invagination Ratio 0.4768. About 8.5 s. Meshing is not bit-reproducible across processes
(§4.1), so expect these to two or three digits, not exactly. `Mesh Volume Ratio` ≈0.99 is
healthy — smoothing shrinks the surface by a fraction of a percent; far from 1.0 would mean
the surface is not tracking the mask (§4.8).

**D. The whole series** — the only run with live Change metrics. *Execution Settings*: switch
to **Process Directory**. *Volumetric tab*: tick **Group Files Into Time Series** (§P.7). The
log prints `Series 1 of 1 -- Cell1: 15 timepoints (frames 1..15)`. At the default **Frame Step
10** only 3 of the 15 are analyzed — frames 1, 11 and 15. The Summary CSV has **1 row** (one
per series, not per file) and the same 32 columns; grouping fills the Change columns rather
than adding any: Mesh Volume ≈591 µm³ · Sphericity 0.843 · Maximum Island Volume Change 1.0355
· Kurtosis Change −0.087. About 29 s.

### From the command line

```
python scripts/run_volumetric_single.py path/to/Cell1_1.tif
python scripts/run_volumetric_batch.py path/to/folder --seg-root ... --seg-template ...
python scripts/run_volumetric_timelapse_barcode.py <data folder> --seg-root <masks folder> --mesh
```

`python scripts/run_barcode.py <path> --mode xyz --list-metrics` prints exactly which columns
a given configuration will emit, and which of those the barcode will render, before you commit
to a batch.

### Things that will catch you out

- **`frame_step` subsamples, and its default is 10.** A 15-timepoint series analyzes frames 1,
  11 and 15 — three, not fifteen. Set **Frame Step** to 1 to analyze every timepoint (§P.6).
- **Meshing without a mask meshes the *threshold* volume.** It does not skip: it falls back to
  the binarized volume and says so in the log. Where thresholding yields scattered specks, the
  mesh columns come out populated but describe halo rather than object (§4.1).
- **Meshing keeps only the largest connected component.** Satellite fragments are discarded
  silently — correct for single-cell crops, but in a multi-object field the mesh columns
  describe one object while the voxel columns describe all of them (§4.11).
- **Use the Browse buttons rather than typing a path.** `browse_file()` clears the directory
  field; typing does not, and the run uses `dir_path if dir_path else file_path`, so a stale
  directory path silently wins over the file you typed.
- **Time-lapse grouping needs Process Directory.** It happens across the file list, so it does
  nothing on a single file.
- **These Execution Settings do nothing in volumetric mode:** Micron to Pixel Ratio and
  Exposure Time (spacing comes from the Volumetric tab or the file's metadata, §P.2), and Save
  Graphs / Save Reduced Data Structures (no per-file 3D outputs exist).
- **Output lands beside the input**, not in a `results/` folder — move it afterwards.

If something looks wrong, the log is the first place to look: most failures print a specific,
actionable line rather than raising, and mask problems in particular are designed to fail
loudly — an XY-size mismatch, a missing file or an implausible Z extent each name the file and
the resolved path (§P.3).

---

## How the numbers were checked

Summarized here so this document stands alone. The full record, including the per-frame
numbers and the mutation script, is kept with the source tree.

**Isolation.** The 2D branches, all of `utils/`, `core/results.py` and `core/metrics.py` are
unchanged — `git diff --stat` over them is empty. New code lives in `analysis/volumetric/`,
the GUI's volumetric tab, `scripts/` and `tests/`. The only hand-off into existing code is one
config-gated branch in `core/pipeline.py::process_single_file`, off by default. That is why a
2D run's numbers cannot move because this exists.

**Analytic ground truth.** Where a closed form exists, the code is checked against it rather
than against itself: sphere volume within 1% (within 2% on anisotropic voxels), sphere
anisotropy within 0.02, ellipsoid anisotropy within 3% at semi-axis ratios 2:1, 3:1 and 1:1,
the autocorrelation of a Gaussian-smoothed field against `exp(−r²/4s²)` to a maximum deviation
below 0.05, its 1/e correlation length against `2s` to within one voxel, and the largest void
with one object present exactly. The Gaussian case is load-bearing, and it found a real defect:
the radial average originally reported each shell at the bin's left edge, where a shell's
population grows as r² so its mean radius sits well above that. Shells are now reported at
their own mean radius (§1.12).

**Cross-implementation checks.** Each computes the same quantity a second, independent way:
`group_avg_3d` against the 2D `groupAvg` (exact), `_average_largest` against
`utils.average_largest` (exact), island counts and volumes by `regionprops` against
`scipy.ndimage.label` + `bincount`, the clamped anisotropy formula against skimage's own axis
lengths wherever skimage can compute them (`rtol=1e-9`), the FFT autocorrelation against an
explicit shift-and-correlate, and the threshold-crossing rule fuzzed over 300 random quantized
profiles against an independent restatement.

**Invariance properties.** Resolution independence (the same physical field sampled
isotropically and subsampled 5× in z gives the same correlation length — this is what catches
binning on voxel index instead of physical distance), unit scaling (doubling the voxel size
doubles lengths and separations and leaves voxel counts and anisotropy untouched), rotation
invariance, per-axis span detection, and determinism.

**The suite has teeth.** Seven deliberate mutations were injected into the binarization module
and **all seven were caught** — binning on voxel index, swapped major/minor axis formulas,
6-connectivity instead of 26, dropping the degenerate-region clamp, ignoring voxel spacing in
island separation, an off-by-one in the threshold crossing, and forgetting to normalize the
autocorrelation. The off-by-one initially escaped, which is why the crossing-semantics and
fuzz tests exist.

**Real data.** Across 15 timepoints of a Jurkat nucleus, the pipeline's largest-island voxel
count matches the on-disk mask exactly on every frame. Nuclear volume has a CV of 2.11%
(mean 585.62 µm³), island anisotropy 7.54%, structural correlation length 5.89%. Over the
series the nucleus flattens axially while spreading laterally at near-conserved volume —
Z extent varies by 9.3% while volume varies by 2.1% — which is what a Jurkat T cell does
spreading on an anti-CD3 surface. An error in voxel volume, spacing or resampling would break
that conservation; it holds.

**Curvature** reproduces its MATLAB source bit-for-bit under the conditions in §4.12, and the
mesh geometry agrees with it to within 0.11% in volume at `mesh_isovalue = 0.99` (§4.1).

---

## Known limitations and caveats

The current version has the following limitations. None changes how a metric is defined; they
are things to know when setting up a run or reading a result.

### Open issues

- **`perslice._intensity_for_slice` can raise an unguarded `values[-1]` IndexError** when no
  histogram bin clears `noise_threshold` — routine for a sparse in-mask selection, and the
  masked `xyz` intensity path routes through it. The caller swallows the exception, so the
  symptom is all six intensity columns NaN while the run reports success.
- **The "Parse All Channels" notice fires once per process, not per run** — a module-level flag
  that is never reset. In the GUI, the second and later runs drop channels with no message.
- **No test exercises the pipeline-level flow sign convention.** The flow tests call the
  divergence and curl operators directly on synthetic fields, so a regression in the field's
  sign handling would not be caught by CI.
- **The isotropic resampler edge-clamps the overhang** where the mask grid extends past the
  image (rather than zero-filling, which would inject background into the data). Nothing in the
  CSV or provenance records which resampling convention produced a file.
- **`packing_border_mode` and `row_axis` have config fields and GUI variables but no GUI
  control** — both are reachable only from a YAML or the CLI.
- **The z-range/mask mapping assumes node-aligned grids** (`round(i·(m_z−1)/(n_z−1))`), so a
  restricted range on a finer mask grid can be misregistered by up to about one acquired slice
  at the deep end.
- **The constant-volume guard uses `np.ptp`, which is NaN-blind.** A float volume containing
  NaN falls through to the mean-relative threshold; the all-False comparison happens to give
  the right binary, but an `np.isfinite` check would make the intent explicit.
- **A very flat object (fewer than three z bins) reports an excluded-cap fraction of 0.0**,
  indistinguishable from a genuinely unclipped object; curvature is then measured over the
  whole surface, caps included.

### Where 2D and 3D numbers must not be pooled

Four columns share a name across the 2D and 3D branches while measuring something
*definitionally* different, beyond the connectivity and dimensionality differences. Each 2D
definition is coherent on its own terms; the point here is only that the numbers do not go
in the same column of a spreadsheet:

- **Mean Island Separation** — 2D averages over the `k+1` nearest entries including the
  self-distance (zero), 3D over the `k` excluding it, a fixed `k/(k+1)` apart.
- **Divergence** — 2D is the divergence of a *cumulative unit-vector* field at the last frame
  pair (1/µm); 3D is the window-mean divergence of the *velocity* field (1/s).
- **Curl** — 2D is a signed scalar (opposite vortices cancel); 3D is an unsigned magnitude
  (they add).
- **Area / Volume Quantity** — the 2D "Area Quantity" multiplies a pixel count by the
  micron-per-pixel ratio once, so it carries µm·px; the 3D Volume Quantity multiplies by the
  voxel volume and is in µm³.

### Measurement caveats

- **Flow speeds are a lower bound** (gain 0.55–0.92 depending on feature scale); relative
  comparisons within a dataset are sound, absolute values are not (§3.3).
- **`flow_downsample` is a fixed dataset-wide choice, not a performance dial** — it changes
  Curl, Divergence and ξ_v by factors of 0.4–2.3 (§3.1).
- **The velocity correlation length ignores the reliability filter and the segmentation mask**,
  unlike every other flow metric (§3.2).
- **`mesh_maxrad` must be scaled to the object** — the default of 5 loses roughly half the
  volume of a thin object; `Mesh Volume Ratio` (§4.8) is the built-in detector.
- **Solidity sits ~1.5% above MATLAB `regionprops3`** by choice of hull convention (§4.9).
- **`crop_to_mask` (off by default) changes the fraction-of-volume denominator** to the cropped
  object; compare the µm³ Quantity columns across differently-cropped runs (P.4).
- **Masked and unmasked intensity runs are different quantities**, not refinements (§2.0).
- **Meshing is not bit-reproducible across processes** (§4.1).

### Scope of the volumetric outputs

- **No per-file outputs in `xyzt`.** Only the dataset-level Summary CSV, barcode PNG and
  Settings YAML are written; there are no reduced data structures or per-frame graphs.
- **Flow reports nothing for the first and last three timepoints** of any series, and nothing
  at all for a series shorter than the flow window (§3.0).
- **Time-lapse memory scales linearly** with series length — a series is held in RAM at once.
