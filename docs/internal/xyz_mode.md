# `xyz` mode — reading a stack down its depth axis

Companion to [`volumetric_manual.md`](../volumetric_manual.md), which defines every metric.
This document is about the mode: what changes when the third axis is **depth rather than
time**, and what the numbers mean once it does.

Read this if you have a Z-stack and you want per-slice structure as a function of depth,
rather than one number for the whole volume.

---

## Contents

- [The one idea](#the-one-idea)
- [Which of the three modes you want](#which-of-the-three-modes-you-want)
- [What `xyz` emits — 29 columns](#what-xyz-emits--29-columns)
- [How a run works, step by step](#how-a-run-works-step-by-step)
- [What "Change (over Z)" actually measures](#what-change-over-z-actually-measures)
- [Working with a segmentation](#working-with-a-segmentation)
- [Per-slice barcodes — one row per slice](#per-slice-barcodes--one-row-per-slice)
- [What you must not read into these numbers](#what-you-must-not-read-into-these-numbers)
- [Gotchas](#gotchas)

---

## The one idea

BARCODE's 2D branches take a **stack of 2D images** and reduce it to one row: some metrics
average over the stack, some compare its start against its end, some count how many members
satisfy a condition.

Published BARCODE assumes that stack is a **time series**. `xyz` mode feeds it a **Z-stack**
instead. The arithmetic is the same; what changes is what the axis *means*.

| | `xyt` (published) | `xyz` |
|---|---|---|
| One member of the stack is | a timepoint | a **z-slice** |
| "First 5% / last 5%" means | earliest and latest frames | **shallowest and deepest slices** |
| A Change metric measures | evolution over time | **variation with depth** |
| `Connectivity` counts | frames that percolate | **slices** that percolate |
| Optical flow measures | motion | *nothing meaningful — disabled* |

That last row is the reason the mode exists as a named thing rather than as "just point the 2D
pipeline at a stack". Feeding a Z-stack to `xyt` does not fail — it silently reports
displacement between focal planes as a velocity in µm/s, and reports growth along depth as
growth over time. Both are plausible-looking and wrong. `xyz` makes the choice explicit and
renames the affected columns so the mistake cannot survive into a figure.

---

## Which of the three modes you want

| You have | You want | Mode |
|---|---|---|
| A 2D movie | the original behaviour | `xyt` |
| A Z-stack, and you care how structure varies **with depth** | 2D metrics per slice, reduced over z | **`xyz`** |
| A Z-stack, and you care about the **3D object** | true volumes, 3D connectivity, meshing, 3D flow | `xyzt` |

`xyz` and `xyzt` answer different questions about the same file. `xyzt` asks "what is this
object" — one volume, one row, 3D quantities. `xyz` asks "how does the cross-section change as
I go down" — many slices, reduced to one row per timepoint, all quantities in-plane.

A stack of **one** timepoint is a perfectly normal `xyz` input; you get one row.

---

## What `xyz` emits — 29 columns

Three identity columns (`File`, `Channel`, `Flags`) plus 26 metrics:

**Binarization (12)** — `Connectivity`, `Maximum Island Area`, `Maximum Void Area`,
`Maximum Island Area Change (over Z)`, `Maximum Void Area Change (over Z)`,
`Initial Maximum Island Area`, `Initial 2nd Maximum Island Area`, `Mean Island Anisotropy`,
`Mean Island Area`, `Total Island Area`, `Mean Island Separation`,
`Structural Correlation Length`

**Intensity (6)** — `Maximum Kurtosis`, `Maximum Median Skewness`, `Maximum Mode Skewness`,
`Kurtosis Change (over Z)`, `Median Skewness Change (over Z)`, `Mode Skewness Change (over Z)`

**Intensity magnitude (4, opt-in)** — `Total Intensity`, `Mean Intensity`, `Intensity SD`,
`Intensity Density (per area)`

**Range provenance (4, opt-in)** — `Z Range Start/End`, `T Range Start/End`

### What it does *not* emit, and why

- **The seven optical-flow columns.** Omitted from the schema entirely, not written as NaN.
  Displacement between adjacent focal planes is µm of structural shift per µm of depth — a
  real quantity, but not a velocity, and there is no honest way to label it µm/s.
- **Mesh, curvature, packing, in-mask intensity, per-object stats, slice profile.** These are
  properties of a 3D object; `xyz` never assembles one. They are hard-gated to `xyzt`, so the
  schema does not advertise columns nothing would fill.
- **Volumes.** Everything is an **area**, in µm², because every measurement is made inside a
  single plane.

---

## How a run works, step by step

Per input file, `analysis/volumetric/slicewise.py`:

1. **Read the volume**, with the axes declared rather than inferred.
2. **Apply the t range** first — excluded timepoints then get no mask or geometry work.
3. **Load the mask, if any, against the full acquired stack**, then apply the z range and slice
   the mask by the same indices. Validating a whole-depth mask against an already-restricted
   image would reject a good mask.
4. **Set the spatial scale to the XY pixel size.** Every `xyz` metric is in-plane, so the scale
   is `xy_step_um` read from the file — never the z step, and never whatever the 2D tab holds.
5. **For each timepoint**, hand the `(Z, Y, X)` array to the unmodified 2D branches, which
   iterate over its first axis. That axis is depth. **One `ChannelResults` row per timepoint.**

There is no isotropic resampling and no cropping in this mode. A mask is matched to the
acquired slice grid by nearest-neighbour index mapping, so mask slice *i* lines up with image
slice *i* — the mask comes to the data rather than the other way round.

### How depth is reduced to one number

Whichever reduction the 2D branch already used, now running over slices:

| Reduction | Metrics | Meaning in `xyz` |
|---|---|---|
| Mean over members | Mean Island Area, Total Island Area, Anisotropy, Separation, Structural Correlation Length | average over analysed slices |
| Top-decile mean | Maximum Island/Void Area, Maximum Kurtosis, both Max Skewnesses | mean of the ~10% of **slices** with the largest value |
| Fraction of members | Connectivity | fraction of **slices** containing a percolating path |
| First *X* % | Initial Maximum Island Area, Initial 2nd Maximum | mean over the **shallowest** slices |
| Last minus/over first *X* % | the five `(over Z)` metrics | **deepest versus shallowest** |

> **`Maximum Island Area` is a top-decile mean, not a maximum.** It is the mean of the largest
> ~10% of per-slice values, so a single unusual slice cannot set it. The same is true of
> `Maximum Kurtosis` and both maximum skewnesses. This is inherited from published BARCODE and
> is the commonest misreading of any BARCODE CSV.

---

## What "Change (over Z)" actually measures

Five columns carry the `(over Z)` suffix:

`Maximum Island Area Change (over Z)`, `Maximum Void Area Change (over Z)`,
`Kurtosis Change (over Z)`, `Median Skewness Change (over Z)`, `Mode Skewness Change (over Z)`

With `n_eval = ⌈percentage_frames_evaluated × n_analysed⌉` — default fraction **0.05**, over the
count of **analysed** slices (after `frame_step`), not the acquired total — each compares the
**deepest** `n_eval` analysed slices against the **shallowest** `n_eval`:

- the two binarization metrics form a **ratio** (deep ÷ shallow; 1.0 means no change);
- the three intensity metrics form a **difference** (deep − shallow).

So a value above 1 for `Maximum Island Area Change (over Z)` means the largest island is bigger
deep in the stack than near the top. **It says nothing about time.** If your series has several
timepoints you get one such value per timepoint, and a trend over time is something you read
*down the rows*, not out of this column.

The rename is the entire safety mechanism. A depth trend read as a time trend is a silent
scientific error — the numbers look completely ordinary — so the column name is the only thing
standing between you and it.

**These are NaN when only one slice is analysed.** A change needs two points on the axis.

---

## Working with a segmentation

`xyz` accepts a mask, and it changes two things beyond replacing the intensity threshold.

**Empty slices are dropped.** Above and below an object the mask is empty, and an empty slice
has mean 0, hence a threshold of 0, hence the *entire field* would be marked as one island —
and empty slices are the norm around a nuclear mask. Only slices the segmentation actually
occupies are analysed, and a segmentation empty on every slice raises rather than producing
numbers.

**This narrows the progression axis.** The `(over Z)` metrics therefore compare the first and
last **occupied** slices, not the first and last of the stack. That is the more meaningful
comparison, but it means a masked and an unmasked run of the same file are not measuring across
the same depth span.

**Masked intensity is built directly from in-mask pixels per slice.** The obvious approach —
blanking out-of-mask voxels with NaN and handing the volume to the 2D branch — fails, because
`np.histogram` derives its range from the data's min/max and raises on all-NaN input. The
masked path avoids that by histogramming the in-mask pixels of each occupied slice directly.

---

## Per-slice barcodes — one row per slice

`xyz` mode reduces depth to a single row. If you want to *see* the depth profile instead, use:

```
python scripts/run_xyz_slice_barcodes.py <folder-or-file> --z-start 12 --z-end 46
```

This writes **one barcode per timepoint**, with:

- **rows** = z-slices, shallowest at the top;
- **columns** = the same 2D metrics;
- reading **down a column** = how that metric varies with depth.

Each row is labelled with its absolute slice index and physical depth, e.g.
`Cell1_1.tif z=23 (6.90um)`. Z indices refer to **acquired** slices, and the range **includes
both ends**, so on 0.3 µm data `--z-start 12 --z-end 46` is 35 slices covering 10.5 µm.

The five Change columns are constant-NaN in this layout — a row is one slice, and a change needs
two points — so the depth trend appears as the gradient of the *other* columns.

Outputs go to a `results/xyz_per_slice` folder beside the source data, never inside it.

---

## What you must not read into these numbers

Everything in `xyz` is measured **inside a plane**. Specifically:

- **`Structural Correlation Length` is an in-plane length.** It says nothing about correlation
  along z. Two slices could be completely unrelated and the metric would not notice.
- **`Connectivity` is per-slice percolation**, spanning x or y within one plane. An object
  continuous through depth but not across any single slice scores 0. (In `xyzt` the same metric
  tests all three axes with 26-connectivity and would score 1.)
- **`Mean Island Anisotropy` is a 2D ellipse ratio** of a cross-section. A cigar standing on end
  reads as circular in every slice.
- **Islands are 2D regions.** One object cut by three slices contributes three islands, so
  `Island Count`-like readings and `Mean Island Separation` describe cross-sections, not
  objects.
- **There is no object.** No volume, no surface, no sphericity, no curvature — those need the
  3D assembly that `xyzt` does and this mode deliberately does not.

If any of those is the question you actually have, you want `xyzt`.

---

## Gotchas

**`frame_step` subsamples slices, and the default is 10.** In `xyz` it strides the depth axis,
so a 54-slice stack at the default is analysed at **7 slices** — indices `0, 10, 20, 30, 40, 50`
and the final slice `53`, which is always appended. That is usually not what you want when the
whole point is a depth profile — set it to 1 for a full profile, at proportionate cost. (The
masked paths use the volumetric selector, which handles small counts safely; the unmasked paths
still route through the 2D helper.)

**`percentage_frames_evaluated` is a fraction of the *analysed* slices.** At the default 0.05 on
7 analysed slices, `n_eval` rounds up to 1 — the `(over Z)` metrics compare exactly one deep
slice against one shallow one. Widen it, or lower `frame_step`, if you want a more robust
comparison.

**The z range is in acquired slice indices by default.** `z_range_units` also accepts
`isotropic` and `microns`; in `xyz` there is no isotropic grid, so `acquired` and `microns` are
the meaningful choices.

**Flag digit 1 never fires in `xyz`.** A dim channel is silently unflagged in this mode — it is
raised in `xyt`, and now in `xyzt`, but nothing sets it on the slice-wise path.

**"Parse All Channels" does nothing.** Like all volumetric modes, `xyz` analyses
`selected_channel` alone. Run other channels separately.

**Rows are timepoints, not files, when a series is grouped** — but time-lapse grouping is
`xyzt`-only, so in `xyz` a multi-timepoint file gives one row per timepoint and separate files
stay separate.

---

## See also

- [`volumetric_manual.md`](../volumetric_manual.md) — the definition of every metric, the
  measurement preconditions, the flag scheme, and the known limitations.
- [`gui_volumetric_howto.md`](gui_volumetric_howto.md) — driving this from the GUI.
- [`volumetric_validation.md`](volumetric_validation.md) — how the branches were checked.
