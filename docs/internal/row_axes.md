# Row axes — what one row of a barcode *is*

Companion to [`volumetric_manual.md`](../volumetric_manual.md), which defines the metrics.
This document is about **what is being compared**.

`analysis_mode` names what BARCODE *measures*. The row axis names what it *compares*.

---

## Contents

- [Why this exists](#why-this-exists)
- [The row axes](#the-row-axes)
- [`auto`, and what it guarantees](#auto-and-what-it-guarantees)
- [Field scope vs object scope](#field-scope-vs-object-scope)
- [The object-row schema](#the-object-row-schema)
- [When a barcode is the wrong picture](#when-a-barcode-is-the-wrong-picture)
- [When it refuses](#when-it-refuses)

---

## Why this exists

**The barcode normalises each column across its rows.** The rows *are* the comparison. Get
them wrong and the picture is either empty of information — one row is a single flat stripe,
with nothing to normalise against — or quietly misleading, because two figures normalised
over different sets invite a comparison their colours do not support.

The right axis is a property of the data, not a matter of taste:

- A **Drosophila embryo** is ~840 cells in one field. The interesting comparison is between
  **objects**; comparing fields would be comparing one number against nothing.
- A **Jurkat nucleus** is one object per field imaged over time. There are no other objects
  to compare it with, so the only available comparison is between **timepoints**.

Both are `xyzt` runs on the same pipeline. Nothing about the *mode* distinguishes them — only
the row axis does. So when you have not chosen, it is resolved from the data, and the choice
is **printed and recorded** rather than assumed.

---

## The row axes

Set with `row_axis` (GUI, YAML) or `--rows` (scripts).

| Axis | One row is | Use when |
|---|---|---|
| `file` | one input file / field of view | the comparison is between acquisitions — BARCODE's original behaviour |
| `timepoint` | one timepoint | a field holds a single object and you want its time course |
| `slice` | one z-slice | you want a depth profile within one timepoint (`xyz` only) |
| `object` | one segmented object, pooled across fields | the field holds many cells and the comparison is between them |
| `auto` | *resolved from the data* | the default |

`slice` is restricted to `xyz` mode — it is the axis that mode's per-slice barcodes already
used, now a first-class setting. See [`xyz_mode.md`](xyz_mode.md).

Reading down a column means something different for each: a time course for `timepoint`, a
depth profile for `slice`, a population distribution for `object`, a between-acquisition
comparison for `file`.

---

## `auto`, and what it guarantees

The resolution order is deliberately short:

1. **an instance segmentation resolved AND more than one object** → `object`
2. else **more than one timepoint** → `timepoint`
3. else → `file`

Many objects beats many timepoints because a field of cells is almost always asking a per-cell
question, while a single object over time is asking a temporal one.

> **`auto` can never reach `object` without a segmentation.** A run with no mask resolves to
> `file` exactly as BARCODE has always behaved, so the published 2D reference outputs are
> unaffected by this feature existing.

The resolved axis is printed along with what the colours were normalised over — e.g.
*normalised across 840 objects from 1 field* — because a barcode's colour scale is
meaningless without it, and two figures built over different sets are not comparable.

---

## Field scope vs object scope

Each axis carries a **scope**, and the scope decides the column set.

- `file`, `timepoint` and `slice` are **field scope**: one row describes a whole field, and
  carries every metric the analysis mode produces.
- `object` is **object scope**: one row describes a single object, and carries only the
  metrics that are *defined* for a single object.

Most metrics are field-level by definition. There is no per-object connectivity, no
per-object correlation length, no per-object kurtosis and no per-object optical flow — those
describe a field or a whole volume. Following the rule the analysis modes already use, **a
column that cannot mean anything is omitted rather than filled with NaN** — or, worse, filled
with the field's value repeated down every row, which would look like data.

That is why an object barcode is narrower than a field barcode. It is not a reduced version
of the same picture; it is a different comparison.

---

## The object-row schema

**The columns are a join, not a new measurement.** Every value already exists per object
somewhere in the run — contact numbers from the packing graph, the in-mask statistics from the
in-mask family, and volume from a `bincount` of the label array — and the object row joins them
by object id. Anisotropy is computed here from the label region's inertia eigenvalues, the same
formula the field metric uses, and needs no mesh.

Written to `<name> Objects.csv`, with identity columns `File`, `FOV`, `Object`. The **three
base columns**, always present and always valued:

| Column | Unit | Same as |
|---|---|---|
| `Object Volume` | µm³ | voxel count × voxel volume, for this object alone |
| `Anisotropy` | dimensionless, ≥ 1 | principal-axis major/minor ratio (§1.8), from the inertia ellipsoid — no mesh needed |
| `Contact Number` | dimensionless | this object's degree in the contact graph (§4.39) |

Then **five mesh-shape columns, always in the schema but `NaN` unless `object_mesh` is on**
(off by default) — with it on, each object is meshed **independently**, distinct from the field
mesh family which meshes only the largest component:

| Column | Unit | Same as |
|---|---|---|
| `Mesh Surface Area` | µm² | §4.3, of this object's own mesh |
| `Sphericity` | dimensionless | §4.4 |
| `Solidity` | dimensionless | §4.9 |
| `Lateral/Axial Ratio` | dimensionless | lateral / axial (§4.7) |
| `Mean Curvature <H>` | 1/µm | §4.14 |

Finally **seven in-mask intensity columns, present only when `--mask-intensity`
(`enable_mask_intensity`) is on** — omitted from the CSV entirely otherwise, not written as NaN:

| Column | Unit | Same as |
|---|---|---|
| `In-Mask MFI` | a.u. | §4.31, for this object |
| `In-Mask Intensity SD` | a.u. | §4.32 |
| `In-Mask Intensity CV` | dimensionless | §4.33 |
| `In-Mask Intensity Skewness` | dimensionless | §4.34 |
| `In-Mask Intensity Entropy` | dimensionless (bits) | §4.35 |
| `In-Mask Normalized Entropy` | dimensionless | §4.36 |
| `In-Mask Fraction Above 2x Median` | dimensionless | §4.37 |

So a default `Objects.csv` carries **8 metric columns** (the 3 base + 5 mesh); adding
`--mask-intensity` brings it to 15. Section references are to
[`volumetric_manual.md`](../volumetric_manual.md), where each is defined in full.

Two properties worth knowing:

- **Object rows are physical to begin with** — a volume in µm³, not a fraction of the field —
  so the "physical units" variant is the same set of columns. There is no fraction/quantity
  split as there is for field rows.
- **Objects carry no flags of their own.** The `Flags` column reads `0`; the field's flags
  describe the run that produced them, and live in the field-row CSV.

`Objects.csv` is written whenever objects were extracted, even if you asked to compare
something else — it costs nothing and is the only per-object record. The object *barcode* is
drawn only when the row axis actually resolved to `object`.

---

## When a barcode is the wrong picture

A barcode needs a population. With one row, every column is a flat, uninformative stripe —
and that is the common case for volumetric work: a single stack, or a single object imaged
over time.

For that, use the **fingerprint** (`write_fingerprint`, or `scripts/run_fingerprint.py`): a
one-page report on a single analysed volume, carrying things a barcode cannot — orthogonal
projections of the volume that was actually analysed (after any z range and resampling), so
the numbers can be checked against what they describe, plus the distributions behind the
scalars.

It is off by default. It is a per-volume *report*, and comparing across fields, objects and
timepoints is what the barcode is for; a document per volume does not help you find structure
across a hundred of them.

---

## When it refuses

An explicitly chosen axis that the data cannot support is an **error, not a cue to fall
back**. Silently choosing a different axis would change what the figure compares without
saying so, which is the whole failure this module exists to prevent.

| You asked for | With | It says |
|---|---|---|
| `object` | no instance segmentation | there are no objects to be rows — supply a mask or choose another axis |
| `object` | exactly one object | a single object is one row, which a barcode cannot normalise — use `timepoint` or `file` |
| `slice` | a mode other than `xyz` | that axis is only available in `xyz` |

---

## See also

- [`volumetric_manual.md`](../volumetric_manual.md) — every metric defined, including the
  in-mask statistics an object row carries.
- [`xyz_mode.md`](xyz_mode.md) — the `slice` axis in its natural habitat.
