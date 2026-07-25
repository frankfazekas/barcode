# Parallel work plan: closing the 4D gaps

Five gaps remain in the 4D BARCODE work, excluding optical flow and testing on new data.
This plan splits them across two agents working simultaneously **without touching the
same files**.

## Why the split needs a contract landed first

Two collisions already happened this session, both in the schema layer:

* another agent added `Solidity` and `Concavity` to the mesh family mid-run, taking xyzt
  from 37 to 39 columns and failing a test that was correct when it started;
* the mode work and the mesh work both edited `core/results.py` within minutes.

`core/metrics.py`, `core/results.py` and `utils/reader.py` are where every feature wants
to add a column, so they are where two agents will always meet. **Phase 0 lands every
new column, config field and call site as an inert stub. After that, neither stream edits
the schema layer at all** — they only fill in behind it.

## The five gaps

| # | gap | stream |
|---|---|---|
| 1 | No way to select a specific T or T range (Z has one, T does not) | **B** |
| 2 | Intensity branch reports only distribution *shape* — no extensive quantity | **A** |
| 3 | Mesh describes only the largest object; "avg vs total" is neither | **B** |
| 4 | Segmentation collapses label masks to binary; one mask per image | **B** |
| 5 | Range flagging says *that* a range was restricted, not *which* | **A** |
| 6 | **Supplied instance masks are re-segmented by connectivity** | **B** |

---

# Phase 0 — the contract (one agent, before forking)

Everything here is additive and inert. Nothing computes a new number yet; the point is
that both streams then have somewhere to put their results without editing shared files.

### Config fields (`core/config.py`, `VolumetricConfig`)

```python
# --- stream B: timepoint selection, mirroring the z range ---
t_start: float = 0
t_end: float = 0
t_range_units: str = "index"          # "index" | "seconds"

# --- stream B: segmentation ---
segmentation_label_mode: str = "binary"   # "binary" | "labels"
segmentation_secondary_root: str = ""     # a second mask, e.g. cell as well as nucleus
segmentation_secondary_template: str = ""

# --- stream B: mesh aggregation ---
mesh_aggregation: str = "largest"     # "largest" | "mean" | "total"

# --- stream A ---
enable_intensity_magnitude: bool = False
record_range_columns: bool = False
```

### Metrics (`core/metrics.py`)

```python
# Stream A, gap 2 -- the first EXTENSIVE quantities in the intensity branch.
INTENSITY_TOTAL   = "Total Intensity"
INTENSITY_MEAN    = "Mean Intensity"
INTENSITY_SD      = "Intensity SD"
INTENSITY_DENSITY_AREA   = "Intensity Density (per area)"
INTENSITY_DENSITY_VOLUME = "Intensity Density (per volume)"

# Stream A, gap 5 -- provenance.
RANGE_Z_START = "Z Range Start"
RANGE_Z_END   = "Z Range End"
RANGE_T_START = "T Range Start"
RANGE_T_END   = "T Range End"
```

Add `Units.INTENSITY = "a.u."`, `Units.INTENSITY_PER_AREA`, `Units.INTENSITY_PER_VOLUME`
and include them in `get_data_limits`' membership test — that function `raise`s on an
unrecognised unit, which is how a missing unit surfaces as a crash rather than a warning.

### Results families (`core/results.py`)

Two new families following the `MeshResults` / `ComponentResults` pattern exactly:

* `IntensityMagnitudeResults` — 4 fields, gated by `include_intensity_magnitude`
* `RangeResults` — 4 fields, gated by `include_ranges`

Extend `_resolve` to carry both switches. It currently returns
`(mode, include_mesh, with_flow, include_components)`; at six optional families it is
past the point where positional flags pay for themselves — **replace it with a registry**
of `(name, results class, capability predicate)` as part of Phase 0, so streams A and B
are not both rewriting the same function later.

### Reader / writer / barcode

`utils/reader.py` builds its accepted header sets from the mode registry already; extend
the loop over `(mesh, components)` to cover the two new families. `utils/writer.py` and
`visualization/barcode.py` auto-detect families from populated data — extend both
identically. **They must agree**: the last bug here was a CSV gaining columns the barcode
did not render.

### Phase 0 exit criteria

* `python -m pytest tests -q` green.
* Column counts pinned in `tests/test_modes.py` for every family combination.
* `xyt` still **28 columns**, byte-identical — the published reference set depends on it.
* GUI wrappers regenerated: `PYTHONIOENCODING=utf-8 python core/config.py`.
* Both new families return all-NaN and appear in no CSV, because nothing populates them.

---

# Stream A — measurement

Fills in the two families Phase 0 created. Touches no selection logic and no
segmentation.

### A1. Intensity magnitude (gap 2)

The intensity branch is currently **entirely intensive** — kurtosis and skewness are
dimensionless descriptors of histogram shape. Nothing scales with the amount of material,
so "is the intensity branch volume based?" has no answer today. Add the extensive
quantities:

* `Total Intensity` — summed voxel values over the analysed region
* `Mean Intensity` — total / voxel count
* `Intensity SD` — spread of voxel values (not of the histogram bins)
* `Intensity Density` — total per µm³ (3D) or per µm² (2D)

Two decisions to make explicitly and document:

* **Background subtraction.** A raw sum includes background; on a 34-slice crop that can
  dominate. Either subtract a background estimate or state clearly that it is a raw sum.
  Whichever is chosen, `intensity_use_mask` becomes far more meaningful here than it is
  for shape metrics — in-mask total intensity is the quantity people actually want.
* **Saturation.** Total intensity is meaningless if the detector clipped. The saturation
  flag (digit 2) already exists; make sure it is set when magnitude metrics are on.

**Files owned:** `analysis/volumetric/intensity.py`, and the per-slice intensity helper
in `analysis/volumetric/perslice.py` (`_intensity_for_slice` only — see the boundary note).

### A2. Range provenance (gap 5)

Flag digit `5` already marks *that* a range was restricted. Populate `RangeResults` so
the CSV also records *which*: `Z Range Start/End` and `T Range Start/End`, in acquired
indices, per row. That also makes per-file ranges representable, which the global
`z_start`/`z_end` settings cannot express today.

Emit only when `record_range_columns` is on; default it on for `xyz`/`xyzt` and off for
`xyt` so the 2D schema does not move.

**Files owned:** a new `analysis/volumetric/provenance.py`.

### Stream A verification

* Analytic: a synthetic volume of known constant value and known object size must give
  `Total = value x voxels` and `Density = value / voxel volume` exactly.
* Scaling: doubling voxel size leaves `Total` and `Mean` unchanged and divides `Density`
  by 8 — the check that intensive and extensive quantities are not confused.
* In-mask vs whole-volume totals differ by exactly the out-of-mask sum.
* Ranges round-trip: a run with `z[12:46]` reports 12 and 46, and reloads through
  `read_csv_to_channel_results`.

---

# Stream B — selection and objects

Everything about *which data* is analysed and *what counts as an object*. Touches no
metric definitions.

### B1. Timepoint selection (gap 1)

Mirror the z range exactly. `VolumeStack.restrict_t` alongside `restrict_z`, a
`resolve_t_range` supporting `index` and `seconds` (using `exposure_time_s`), and one
shared `apply_t_range` helper so all three pipelines cannot diverge — the same structure
that already prevents divergence for z.

Reuse rather than re-derive: `restrict_z` already handles negative indices, `0` meaning
"to the end", and raises on an empty range. `restrict_t` should behave identically.

Note the interaction: in `xyzt` with time-lapse grouping, a T range selects *files* from
the series, not slices within one file. Decide and document which happens first.

### B2. Mesh aggregation over multiple objects (gap 3)

`mesh.py:740` calls `largest_component(mask_zyx)`, so all 11 mesh and curvature columns
describe one object with nothing in the output saying so. Implement `mesh_aggregation`:

* `largest` — current behaviour, kept as default so existing runs do not change
* `mean` — mesh every object, average the scalars
* `total` — mesh every object, sum the extensive ones (volume, surface area) and average
  the intensive ones (sphericity, curvature). **Summing sphericity is meaningless**;
  the split between extensive and intensive has to be explicit per metric.

This closes the "avg vs total" question, which for sizes is already answered (both
`Mean Island Volume` and `Total Island Volume` exist) but for mesh is currently neither.

### B3. Mask-defined objects (gaps 4 and 6)

**The governing principle: BARCODE does not segment. When a mask is supplied, its
objects are authoritative.**

Today they are not. `binarization.py:129` runs `label(binary, connectivity=3)`, so a
supplied instance mask is flattened to binary and then *re-segmented* by connectivity.
Two touching cells that Cellpose correctly separated silently merge:

```
cellpose says   : 2 instances, 1500 voxels each
BARCODE reports : 1 object, largest 3000, separation nan
```

Count, separation, SD, skewness and median are all wrong, with nothing in the output
saying so. Touching instances are the normal case in confluent fields — precisely when
instance segmentation is used at all. This is a correctness bug, not a missing feature,
and it is the highest priority item in stream B.

**(a) A general mask loader.** One registry keyed by suffix, so a new format is one
entry rather than a new code path:

```python
MASK_READERS = {".tif": _read_tiff, ".tiff": _read_tiff,
                ".npy": _read_npy, ".npz": _read_npz}
```

Every reader returns an **integer label volume** plus provenance. An unknown suffix
raises listing what is supported, rather than guessing.

Cellpose's `_seg.npy` is a pickled dict (`np.load(..., allow_pickle=True).item()`) whose
`masks` key holds the label array; it may also be a bare array. Handle both, and do not
special-case Cellpose beyond that — other tools write plain label TIFFs and `.npz`, and
the registry should treat them all the same.

**(b) Labels define objects.** Add `object_partition`:

* `labels` — every distinct positive integer is one object; touching instances stay
  separate
* `connectivity` — re-derive from the binary volume, today's behaviour, correct when only
  a binary mask or an intensity threshold is available

Default to `labels` when the loaded mask carries more than one distinct positive value,
`connectivity` otherwise; an explicit setting always wins. `find_island_properties_3d`
gains an optional `labelled` argument and skips `label()` entirely when it is supplied.

**(c) Consequences to handle deliberately**

* The word *island* assumes connectivity defines objecthood. Under `labels` an island is
  an instance. Say so in the docs rather than quietly changing what a column means.
* `check_span` and `find_largest_void` operate on the binary field and are unaffected.
* `mesh.py`'s `largest_component` must become per-label when partitioning by labels;
  `mesh_aggregation` (B2) then aggregates over instances, which is what makes
  "avg vs total" meaningful.
* A second mask (`segmentation_secondary_*`) still applies: nucleus *and* cell.

**(d) Per-object rows** stay a separable step, but the contract now carries an
`Object ID` column so adding them later is not a schema change. With `per_object_rows`
on, emit one row per instance with that column filled; otherwise one aggregate row with
it blank.

### B3 verification

* The touching-instances case above must report **2** objects, 1500 voxels each.
* A Cellpose-style `_seg.npy` (pickled dict) and a plain label TIFF must load
  identically.
* Aggregates over N labelled instances match hand-computed values.
* A binary mask still partitions by connectivity, unchanged.

**Files owned:** `analysis/volumetric/reader.py`, `analysis/volumetric/segmentation.py`,
`analysis/volumetric/mesh.py`, `analysis/volumetric/slicewise.py`,
`analysis/volumetric/perslice.py` (except `_intensity_for_slice`),
`analysis/volumetric/run.py`.

### Stream B verification

* All three t-unit forms select the same timepoints, checked through the CLI, exactly as
  was done for the z units.
* A restricted t range on a 15-timepoint series yields the expected count and sets the
  provenance columns stream A populates.
* Mesh aggregation: a synthetic volume with three known spheres gives `total` volume =
  sum of the three, `mean` = the average, `largest` = the biggest — and sphericity is
  averaged, never summed, under all three.
* A label mask with N objects survives loading with N distinct labels.

---

# Boundaries

**Neither stream edits, after Phase 0:**

```
core/metrics.py      core/results.py      core/config.py
utils/reader.py      utils/writer.py      visualization/barcode.py
core/pipeline.py     core/modes.py        gui/**            scripts/**
```

If either stream finds it *needs* a change there, that is a signal Phase 0's contract was
wrong. Stop and amend the contract jointly rather than editing in parallel.

**The one shared file** is `analysis/volumetric/perslice.py`: stream A owns
`_intensity_for_slice`, stream B owns everything else in it. If that proves awkward,
split the intensity helper into its own module during Phase 0.

**Still on the do-not-modify list for both:** `analysis/binarization.py`,
`analysis/optical_flow.py`, `analysis/intensity_distribution.py`, `analysis/run.py`,
`utils/__init__.py`, `utils/binarization.py`, `utils/setup.py`. The one exception taken so
far — the single-island guard in `analysis/binarization.py` — was verified byte-identical
on a 2D reference run and should not be treated as precedent.

**Third-party coordination:** another agent has been active in `analysis/volumetric/mesh.py`
and `curvature.py`. B2 collides with that directly. Confirm they have stopped before B2
starts, or hand B2 to them.

---

# Integration

1. Phase 0 lands and is green before either stream starts.
2. Streams A and B run in parallel, each on its own branch or worktree.
3. Merge A first (it only adds numbers to existing stubs), then B.
4. Post-merge: full suite, `git diff --stat` over the do-not-modify list must be empty,
   and one run per mode on the Cell1 series with column counts asserted.
5. Regenerate the REVIEW set so the barcodes reflect the new columns.

## Sequencing note

Gap 2 is the highest-value item. The metric count has gone 25 to 39, but on single-nucleus
data only ~24 columns are informative and three of those collapse to the same number when
there is one object. The additions that earned their place measured genuinely new physics
— surface area, sphericity, curvature, height. Integrated intensity is another such
quantity; most of the rest re-express size. If only one stream can run, run A.

---

# Stream A — delivered

Both families are implemented and tested. **The computation is done; the wiring is not**,
because the three lines that populate the results live in `analysis/volumetric/run.py`
and `slicewise.py`, which stream B owns and was actively editing. Integration applies
them.

### What exists

* `analysis/volumetric/intensity.py`
  * `compute_intensity_magnitude(values, sample_size)` — pure, takes a flat array and the
    physical size of one sample (um^3 per voxel, um^2 per pixel)
  * `analyze_intensity_magnitude(volumes, spacing_zyx, frame_indices, masks=None)`
  * `is_saturated(values, bins, noise_threshold)` — magnitude callers that skip the
    shape branch still need to ask, because a clipped detector makes `total` meaningless
* `analysis/volumetric/provenance.py`
  * `build_range_results(stack, n_timepoints=None)`
  * `was_restricted(stack)`, `describe_range(results)`

### The three lines integration adds

In `run_volumetric_analysis`, after the intensity branch:

```python
if vcfg.enable_intensity_magnitude:
    results.intensity_magnitude = analyze_intensity_magnitude(
        volumes, spacing_zyx, frame_indices,
        masks if vcfg.intensity_use_mask else None)

if vcfg.record_range_columns:
    results.ranges = build_range_results(stack)
```

and the same two in `run_slicewise_analysis` / `run_per_slice_analysis`, using each
one's own stack. The writer and barcode already detect populated families, so nothing
else changes.

### Decisions taken, and why

* **`total` is a raw sum including background.** On Cell1_1 the mask holds 8.80% of the
  voxels but 23.11% of the signal, so ~77% of a whole-volume total is background. Rather
  than pick a background model, the number is honest and `intensity_use_mask` is the
  lever — which is why that switch matters far more here than for the shape metrics.
* **Density is named per mode** (`per volume` / `per area`), following the Area/Volume
  precedent, because its unit genuinely differs rather than its meaning.
* **An unrestricted axis reports its full extent**, not NaN. "0 to 54" and "no range was
  set" are the same statement about the data, and a reader should not have to know which
  produced the row.
* **An empty mask yields NaN, not 0** — no voxels is "not measured", not "measured as
  zero".

### Validated on Cell1_1

```
                    whole volume     in-mask only
total                6.90764e+08      1.59647e+08
mean                     135.313          355.481
density                   106756           280458

mask = 8.80% of voxels, holds 23.11% of the signal -> 2.63x enrichment
saturated: False, so total is interpretable
```

In-mask mean of 355.5 matches the value measured independently when the mask was first
characterised, which is the cross-check that the masking is being applied to the voxels
it claims.

### Note for stream B

`analysis/volumetric/slicewise.py` calls `apply_t_range(stack, vcfg)` twice, at lines 84
and 91. The second call re-restricts an already-restricted stack: with `t_start=2,
t_end=9` the first gives 7 timepoints and the second cuts those to 5. Silently wrong
rather than an error. Not fixed here — that file belongs to stream B.
