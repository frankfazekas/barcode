# Volumetric BARCODE — what was built and how it was checked

Companion to `analysis/volumetric/`. The question this document answers is not "does it
run" but "why should anyone believe the numbers".

## How to run it

Single file, no mask:

```bash
python scripts/run_volumetric_single.py path/to/Cell1_1.tif
```

Single file, with segmentation:

```bash
python scripts/run_volumetric_single.py path/to/Cell1_1.tif \
  --seg-root ".../prog_live_cells" \
  --seg-regex "Cell(?P<cell>\d+)_(?P<frame>\d+)" \
  --seg-template "Cell{cell}/frame{frame}/nucleus/3D_seg/Cell_{cell}_SegMask_origFOV.tif"
```

Whole folder, writing a BARCODE Summary CSV plus a cross-frame consistency report:

```bash
python scripts/run_volumetric_batch.py path/to/folder --seg-root ... --seg-template ...
```

In the GUI: *Execution Settings* → tick **Volumetric Analysis**. Everything on the
**Volumetric Settings** tab is inert while that box is unticked.

## Why the 2D pipeline needed this

A `(Z, Y, X)` stack does not fail in the 2D reader — it is silently misread. `iio.imread`
returns three axes, `utils/reader.py:36` appends a channel axis giving `(54,312,303,1)`,
and `shape[3] == 1 == min(shape)` so no transpose fires. BARCODE then analyses **54
Z-slices as 54 timepoints**: optical flow between adjacent focal planes, "island growth
over time" that is really growth along depth. The output is plausible and meaningless.
The volumetric reader refuses to guess instead.

## Isolation

`git diff --stat` over the 2D branches (`analysis/binarization.py`, `optical_flow.py`,
`intensity_distribution.py`, `run.py`), all of `utils/`, `core/results.py` and
`core/metrics.py` is **empty**. Across every shared file the change is 163 insertions and
2 deletions — one blank line and one `requirements.txt` line that was re-added in place.

New code lives in `analysis/volumetric/`, `gui/frames/process/volumetric_tab.py`,
`scripts/`, `tests/`. The only hand-off into existing code is one config-gated branch in
`core/pipeline.py::process_single_file`, off by default. `gui/config.py` is generated —
regenerated with `PYTHONIOENCODING=utf-8 python core/config.py`, purely additive.

## Evidence

### 1. Analytic ground truth

Where a closed-form answer exists, the code is checked against it, not against itself.

| Check | Expected | Result |
|---|---|---|
| Sphere volume, r=20 | `4/3 pi r^3` | within 1% |
| Sphere volume, anisotropic voxels (0.3, 0.065, 0.065) | ellipsoid volume | within 2% |
| Sphere anisotropy | 1.0 | within 0.02 |
| Ellipsoid anisotropy, semi-axes 6:12, 5:15, 10:10 | 2.0, 3.0, 1.0 | within 3% |
| `g(r)` of a Gaussian-smoothed field | `exp(-r^2/4s^2)` | max deviation < 0.05 |
| 1/e correlation length of that field | `2s` | within 1 voxel |
| Largest void with one object present | `size - object` | exact |

The Gaussian-field case is the load-bearing one: the autocorrelation of white noise
smoothed by a Gaussian of width `s` is analytically `exp(-r^2/(4 s^2))`, so the 1/e
correlation length must be exactly `2s`. **This test found a real bug.** The radial
average originally reported each shell's correlation at the bin's *left edge*, but a
shell spans a range of radii and its population grows as `r^2`, so its mean radius sits
well above the left edge. The profile therefore appeared to decay faster than theory
(peak error 0.069). Shells are now reported at their own mean radius.

### 2. Cross-implementation checks

Each of these computes the same quantity a second, independent way:

- `group_avg_3d(vol, (1,N,N))` per slice vs the existing 2D `utils.groupAvg` — exact.
- `_average_largest` vs `utils.average_largest` — exact.
- Island counts and volumes: skimage `regionprops` vs `scipy.ndimage.label` + `bincount`.
- Anisotropy: our clamped formula vs skimage's own `axis_major_length/axis_minor_length`
  on regions where skimage can compute them — agreement to `rtol=1e-9`.
- Autocorrelation: FFT result vs an explicit shift-and-correlate.
- Threshold crossing: fuzzed over 300 random quantised profiles against an independent
  restatement of the 2D rule.
- Frame selection vs `utils.find_analysis_frames` — identical wherever the 2D helper
  works (see the defect note below).

### 3. Invariance properties

Properties that must hold whatever the implementation:

- **Resolution independence.** The same physical field sampled isotropically, and
  subsampled 5x in z with `z_step=5`, gives the same correlation length. This is what
  catches binning on voxel index instead of physical distance.
- **Unit scaling.** Doubling the voxel size doubles lengths and separations, leaves voxel
  counts and anisotropy untouched.
- **Rotation invariance.** Scalar metrics are unchanged under a 90-degree rotation.
- **Span detection** fires on each of the three axes independently.
- **Determinism.** Identical input, identical output.

### 4. The suite has teeth

Passing tests prove nothing unless a wrong implementation fails them. Seven deliberate
mutations were injected into `analysis/volumetric/binarization.py`; **all seven were
caught**:

| Mutation | Caught by |
|---|---|
| Bin on voxel index rather than physical distance | Gaussian-theory, resolution-independence |
| Swap the major/minor axis formulas | 6 tests incl. all ellipsoid cases |
| Face (6) connectivity instead of full 26 | independent-labelling cross-check |
| Drop the degenerate-region clamp | 3 degenerate-shape tests |
| Ignore voxel spacing in island separation | unit-scaling |
| Off-by-one in the threshold crossing | crossing-oracle fuzz |
| Forget to normalise the autocorrelation | 3 autocorrelation tests |

The off-by-one initially escaped, which is why the crossing-semantics and fuzz tests
exist. Reproduce with the mutation script kept in the session scratchpad.

### 5. Real data — 15 timepoints of Cell1

Mask preservation, checked against the on-disk mask for every frame:

```
pipeline largest-island voxels vs on-disk mask foreground
worst relative difference: 0.000e+00   (all 15 frames exact)
```

Cross-frame consistency, and the reason to believe the geometry is right:

```
nuclear volume (um^3)         mean 585.62   sd 12.37   CV 2.11%
island anisotropy             mean  1.895   sd  0.143  CV 7.54%
structural correlation (um)   mean  2.255   sd  0.133  CV 5.89%

shape trend over the 15 frames:
  Z extent   slope -2.00 /frame   r -0.660
  Y extent   slope +2.51 /frame   r +0.885
  volume     slope +2.59 /frame   r +0.905   (CV 2.11%)
```

The nucleus **flattens axially while spreading laterally at near-conserved volume** —
Z extent varies by 9.3% while volume varies by 2.1%. That is what a Jurkat T cell does
spreading on an anti-CD3 surface, which is what this dataset is. An error in voxel
volume, spacing, or the resampling step would break that conservation; it holds.

#### Flow on the same series

Run at full resolution with masks (`results\flow_run1\Cell1 Timepoints (flow).csv`). Nine
of fifteen timepoints get a window; the first and last three cannot (t=1-3, 13-15).
Reliability keeps the top 50% of voxels, intersected with the mask, leaving ~24% used.

```
  t   vol um^3   d(lnV)/dt  divergence    speed     curl    VCorr
  4     571.09    +0.01247    -0.01300   0.1526   0.3688   1.1359
  6     578.97    +0.00138    -0.02758   0.1405   0.3255   1.2008
  8     585.95    +0.00064    -0.01586   0.1344   0.2340   1.0707
 10     594.35    +0.01722    +0.00963   0.1198   0.1852   1.5910
 12     599.37    -0.00140    -0.00177   0.1099   0.1696   1.3963
```

Speed above is per frame, because this run predates `frame_interval_s`. This series is
60 s per timepoint, so the true speeds are these divided by 60 (0.1526 um/frame ->
2.54e-3 um/s). Re-run with `--frame-interval 60` for the corrected table.

The internally consistent story across four independently computed metrics: as the cell
settles, motion slows (speed 0.153 -> 0.110), becomes more coherent (`Directional Spread`
1.38 -> 0.69, `VCorr` 0.62 -> 1.40 um), and less rotational (`Curl` 0.37 -> 0.17). Nothing
in the implementation couples these, so their agreeing on a single monotone trend is
weak but real evidence the field is physical rather than noise.

**One thing that does not close, stated plainly.** Divergence should track the rate of
volume change, and the two are only weakly related: Pearson r = +0.38 over the nine
windows (n=9, p~0.3 — not significant), and divergence is systematically *negative*
(mean -0.0102 /frame) while the mask volume *grows* (mean d(lnV)/dt +0.0034 /frame). The
sign of the trend is right and the largest expansion (t=10) does carry the only strongly
positive divergence, but this is **not** a passed cross-check. Two candidate explanations,
untested: the two quantities measure different things — divergence comes from the
intensity field inside the nucleus (chromatin can compact while the envelope expands),
volume from the mask — and the gradient-based solver's known low bias would depress
divergence magnitudes unevenly. Do not cite divergence as validated.

### 6. Negative tests

Each of these must fail, loudly, with an actionable message — a silently misaligned mask
is worse than no mask:

- XY-cropped mask (`Cell_1_SegMask.tif`, 229x210 vs image 312x303) — rejected.
- Missing mask — `FileNotFoundError` naming the resolved path and template.
- Wrong `mask_spacing_um` (0.3) — rejected, reporting 75.0 um vs 16.2 um of z extent.
- Regex that cannot match the filename — rejected, listing what it captured.
- Mask TIFF passed to `read_volume` (axes `IYX`) — rejected rather than guessed.

### 7. End-to-end

- 2D pipeline on a synthetic planar movie: all three branches, Summary CSV, barcode PNG,
  Settings YAML — unaffected.
- Volumetric CSV round-trips through BARCODE's own `read_csv_to_channel_results`.
- GUI widget tree builds; all five tabs render and select.
- Driving the GUI's own config objects through `create_processing_worker` over all 15
  files: 47 s, flow branch skipped with an explanation, results **identical** to the CLI
  batch run.
- `Settings.yaml` round-trips: an old YAML with no `volumetric:` section still loads.

### 8. Which shape metrics survive our imaging grid — a real 4 nm Jurkat nucleus

Synthetic phantoms bound the discretisation error for spheres, ellipsoids and tori. A real
nucleus is none of those, and the error depends on the shape. Janelia's OpenOrganelle
[`jrc_jurkat-1`](https://openorganelle.janelia.org/) supplies the missing case: real Jurkat
nuclei — our own cell type — segmented at 3.44 x 4 x 4 nm, near-isotropic.

`scripts/stage_openorganelle.py` stages a single interior nucleus (object 10, 289 um^3,
verified not to touch the field edge) and `scripts/validate_openorganelle.py` resamples it
down a ladder of coarser and more anisotropic grids, re-running the pipeline at each rung.
The mask is taken through a two-step **acquisition simulation** — decimated onto the
anisotropic acquired grid, then upsampled back to isotropic, as a real segmentation of live
data would be. Going straight to isotropic instead makes every rung sharing an xy step
produce a byte-identical mask, so the ladder reports anisotropy as free; it is not.

Drift at our live acquisition geometry (0.065 um xy, 0.3 um z, 4.6x anisotropic), against
the 32 nm rung as truth:

| Metric | Drift | Verdict |
|---|---|---|
| Maximum Island Volume, Mesh Volume, Equivalent Sphere Radius | -0.0% to -0.5% | measured |
| Lateral/Axial Ratio, Solidity, Mean Curvature `<H>`, Sphericity, Mesh Surface Area, Mesh Height | +0.2% to -1.6% | measured |
| **Invagination Ratio** | **-69%** | grid-dominated |
| **Concave Area Fraction** | **-69%** | grid-dominated |
| **Structural Correlation Length** | **+54%** | unstable |
| **Mean Island Separation** | **-54%** (NaN at one rung) | unstable |

Two things worth carrying forward. The bulk shape metrics are trustworthy at our
resolution — better than expected, within ~2%. But the fine-concavity pair (invagination
ratio, concave area fraction) loses roughly 70% of its signal, so those numbers describe the
imaging grid more than the nucleus and are only ever comparable between equally-sampled
datasets. And the damage is dominated by the **xy** step, not by anisotropy: the isotropic
0.128 um rung is worse on most metrics than the 4.6x-anisotropic 0.065/0.3 one.

Caveats: FIB-SEM contrast is inverted relative to fluorescence, so this validates the
mask-driven geometry path, not thresholding or the intensity metrics; EM preparation shrinks
cells, so absolute um^3 is not comparable across modalities; and the volume has no time
axis, so the flow branch gains nothing from it. Packing is also **not** validated here —
Jurkat are suspension cells (0 contacts across 11 nuclei), and OpenOrganelle's `pm_seg` is a
membrane prediction rather than filled cell bodies (on `jrc_mus-liver`, one 27,602 um^3
connected network plus ~230 fragments), so contact-number ground truth still needs a
watershed fill or another source.

## Time-lapse assembly

Volumetric time-lapses are often exported one file per timepoint. Analysed individually
they give N rows with every change metric NaN — the dynamics BARCODE exists to measure.
`analysis/volumetric/timelapse.py` groups them back into one `(T, Z, Y, X)` series.

Enable **Group Files Into Time Series** on the Volumetric tab (needs Process Directory),
or `timelapse_enabled: true` in the YAML. Output becomes one row per *series* rather than
one per file.

Grouping uses a regex with two named groups — `series` (files sharing it belong together)
and `frame` (numeric ordering). The default `^(?P<series>.+?)_(?P<frame>\d+)$` splits on
the final underscore. Ordering is numeric, so `Cell1_2` precedes `Cell1_10`; the
lexicographic sort in `utils/setup.py::find_files` would not. Files that do not match are
reported, never silently dropped.

### The shared crop box, and why it matters

Each timepoint's mask has its own bounding box — across Cell1 the Z extent runs from 181
down to 128 voxels as the cell flattens. Cropping each timepoint to its own box would
give arrays that cannot be stacked, and a **different denominator for every "fraction of
volume" metric** — so those columns would drift from the crop rather than from the
object. Every timepoint is therefore cropped to the *union* of the per-frame boxes.

Measured on Cell1, 15 timepoints:

```
                       per-frame crop      union crop
physical volume CV        2.11%              2.11%
fraction-of-volume CV     4.63%              2.11%   <- now tracks real volume change
shared voxel count          --            9,001,692 (identical every timepoint)
```

The two CVs now agree to 9 decimal places, which is the arithmetic signature of a
constant denominator.

### Checks

- Per-timepoint volumes from the assembled series are **identical** to the 15 independent
  single-frame runs (579.7, 566.3, 565.1 ... 600.3 µm³) — the union crop changes the
  framing, not the physics.
- Change metrics become finite: island change 1.0355, void change 0.9891, kurtosis change
  −0.2527.
- `tests/test_timelapse.py` (11 tests) pins the grouping and geometry, including two
  collision cases that would silently corrupt a series: `Cell1` must not absorb `Cell11`,
  and `Cell1_centrin_3` must form its own series rather than becoming frame 3 of `Cell1`.
  Shape and spacing mismatches raise rather than pad or crop.

Note that `frame_step` still subsamples timepoints as it does in 2D — at the default of
10, a 15-timepoint series analyses frames 0, 10 and 14. Set it to 1 to analyse every
timepoint.

## Analysis modes

BARCODE can analyse the same file three ways. The choice is explicit and validated -- no
guessing from file axes, because guessing is what produced the original silent bug.

| mode | analysed unit | progression | flow | mesh | components | columns |
|---|---|---|---|---|---|---|
| `xyt` | 2D plane | time | velocity um/s | - | - | 28 |
| `xyz` | 2D plane | **depth** | omitted | - | - | 21 |
| `xyzt` | 3D volume | time | 3D velocity | yes | yes | 37 (+4) |

Modes are two orthogonal properties -- spatial dimensionality and progression axis -- and
every applicability rule is derived from them in `core/modes.py` rather than kept in a
hand-maintained table. Spatial dimensionality decides Area (um^2) vs Volume (um^3) and
whether meshing is possible; the progression axis decides whether Change metrics mean
anything; progression being *time* decides whether flow is physical.

`xyz` needed no new analysis mathematics: a `(Z, Y, X)` array is exactly what the 2D
branch functions accept, so `analysis/volumetric/slicewise.py` is a reader and a loop
calling them read-only. **Verified 18 of 18 metrics bit-identical** to what the 2D
pipeline already produced on the same Z-stack, once the same voxel spacing is used --
xyz reads the file's true XResolution (0.06500000162500004) where the baseline had a
hand-typed 0.065, a 2.5e-8 difference in xyz's favour.

Selecting `xyt` on a file that declares `Z` and no `T` is refused with a message naming
both alternatives. The guard is deliberately narrow: plain planar TIFFs declare `QYX`
(no axis at all), so requiring a `T` axis would reject ordinary 2D data including the
published reference set.

### Z range

`z_start` / `z_end` restrict the analysed depth in `xyz` and `xyzt`. The full stack is
often the wrong range -- slices past the object are background. On `Cell1_1.tif`,
restricting to `z[15:45]` moved max kurtosis from **4.87 to 36.58**, because the empty
slices had been diluting the intensity distribution.

**State the unit.** "Slice 46" is ambiguous on anisotropic data: the acquired stack and
the isotropic grid a segmentation lives on differ by the anisotropy factor -- 54 slices
versus ~249 for the same 16.2 um of depth. `z_range_units` therefore takes:

| unit | meaning | example (0.3 um acquired, 0.065 um isotropic) |
|---|---|---|
| `acquired` (default) | indices into the stack as acquired | `12 .. 46` |
| `isotropic` | indices on the isotropic grid | `55 .. 212` |
| `microns` | physical depth from the bottom | `3.6 .. 13.8` |

All three select the same 34 slices / 10.2 um -- verified through the CLI, not just in
unit tests. `microns` is the one that cannot be misread. `0` as the end always means "to
the end" in every unit; for the two index units, negatives count back from the end.

All three pipelines resolve the range through one helper
(`analysis.volumetric.reader.apply_z_range`) so they cannot interpret the same setting
differently.

### Per-slice barcodes: one barcode per timepoint, one row per z-slice

`slicewise.py` reduces a depth profile to a single row, which is right for comparing
timepoints but collapses the z structure. `analysis/volumetric/perslice.py` keeps it:
every analysed slice becomes a row, so one barcode per timepoint reads top-to-bottom as
depth. Run with `scripts/run_xyz_slice_barcodes.py`.

Rows are labelled with index and physical depth (`Cell1_1.tif z=24 (7.20um)`), so a row
always traces back to a plane. Two consequences of a row being one slice, both intended:
Change metrics are NaN (a change needs two points; the trend is the gradient *between*
rows), and Connectivity is 0/1 rather than a fraction.

Like `slicewise`, no new mathematics -- it calls the same per-frame primitives
`analysis/binarization.py` uses internally, read-only.

### Per-object statistics (opt-in)

`Island Count`, `Island Volume SD`, `Island Volume Skewness`, `Median Island Volume`
describe the *spread* of the object-size distribution, which the mean and total cannot:
one dominant object plus debris gives the same mean as several even ones. Off by default
so the barcode stays readable; volumetric modes only, where BARCODE owns the labelling.

They quantify the threshold-vs-mask difference that was previously only describable in
words:

```
threshold:  167 objects, skewness 12.81, median size 3.9e-07   <- one nucleus + 166 specks
with mask:    1 object,  skewness  nan,  median size 0.334     <- just the nucleus
```

Mean and total size are deliberately *not* in this family -- they already exist as
`Mean Island Volume` and `Total Island Volume` in the binarization family.

### Turning metrics off

`hidden_barcode_metrics` trims the **barcode image only**; the CSV always carries the
full set for the mode, so a trimmed run stays comparable with an untrimmed one and still
round-trips. Selection is by metric *name* (`core.metrics.selection_mask`), not by
position -- a positional boolean list silently hides the wrong column the moment a mode
adds or drops a family.

In the GUI: a checklist on Execution Settings that rebuilds when the mode changes.
From the CLI: `--hide-metric NAME` (repeatable), and `--list-metrics` to see what a
configuration would produce before running it.

## Known gaps and caveats

- **3D optical flow needs a contiguous 7-frame window per timepoint.** The branch is
  implemented (`analysis/volumetric/flow.py`, wrapping the Lucas-Kanade solver vendored
  from aicjanelia/OpticalFlow3D in `flow_lucas_kanade.py`), but unlike the structural and
  intensity branches it reads a timepoint's neighbours rather than that timepoint alone.
  It needs `6*flow_t_sigma+1` consecutive volumes — 7 at the default — centred on each
  analysed frame, so series shorter than that are skipped entirely and the first and last
  three timepoints of any series report NaN. This also means `frame_step` does not
  subsample it the way it does the other branches.
- **Flow velocities are systematically under-reported.** The solver is gradient-based, so
  its recovered speed is a fixed fraction of the true speed when the tracked features are
  comparable in size to `flow_xyz_sigma`: measured gain is ~0.55 at a 2-voxel feature
  scale and ~0.92 at 6 voxels, both at the default `flow_xyz_sigma = 3`. The gain is
  linear in speed, so *relative* comparisons across a dataset are sound; absolute speeds
  are a lower bound. At very low contrast the structure tensor approaches the solver's
  eps regularisation and velocities collapse towards zero — the reliability mask
  (`flow_reliability_percentile`, default 50) is what keeps those voxels out of the mean.
- **`flow_downsample` changes the gradient-derived flow metrics.** Block-averaging smooths
  the velocity field before the derivatives are taken. Measured on Cell1 (Jurkat nucleus,
  15 timepoints, isotropic grid 196x243x189 at 0.065 um) across all nine analysed windows,
  going from 1 to 2 (ratio ds2/ds1, mean over windows):

  | metric | ratio | range | reading |
  |---|---|---|---|
  | `Speed` | 1.03 | 0.97-1.06 | unchanged — the physical unit conversion tracks the coarser grid, which is the point of the check |
  | `Curl` | 0.41 | 0.39-0.44 | ~halves; curl is a spatial derivative and smoothing lowers it |
  | `Velocity Correlation Length` | 2.25 | 1.66-3.66 | ~doubles; block-averaging correlates neighbours by construction |
  | `Divergence` | 1.51 | 1.01-3.44 | unstable — it is a small difference of larger terms |

  Runtime drops from ~500 s to ~125 s for the 15-timepoint series. So it is a fixed
  dataset-wide setting, not a per-run performance dial: runs at different values may be
  compared on speed but not on curl, divergence, or correlation length.
- **`Curl` means something different in 3D.** Curl is a vector in three dimensions, and
  there is one CSV column, so the 3D branch reports the mean of `||curl v||`. It is
  always positive and carries no handedness, unlike the signed 2D metric. `Mean Flow
  Direction` stays the XY azimuth so the column remains comparable with 2D runs, while
  `Directional Spread` uses the resultant length of the full 3D unit vectors.
- **No per-file outputs in 3D.** The volumetric path returns before the 2D code that
  creates `<file> BARCODE Output/`, so there are no RDS CSVs, per-frame graphs, or
  summary figures — only the dataset-level Summary CSV, barcode PNG and Settings YAML.
- **The denominator is the acquired field.** `prepare_volume` (formerly `prepare_nucleus`)
  no longer crops to the mask bounding box: `crop_to_mask` defaults to **False**, so
  "fraction of volume" is relative to the whole analysed field and one denominator is
  shared across files and timepoints. On Cell1 that reads ~8.8%. Turning `crop_to_mask`
  on restores the old behaviour and the ~39% figure, but gives each file its own
  denominator — an object shrinking and the box tightening around it are then
  indistinguishable. Physical volumes in um^3 are unaffected either way.
- **Masked intensity histograms are opt-in and unvalidated.** `intensity_use_mask`
  defaults off. Turning it on changed Maximum Kurtosis from 9.93 to -0.11 on Cell1_1 —
  a different quantity, not a refinement, so masked and unmasked runs must not be pooled.
- **Batch file order is numeric in the volumetric modes.** `utils/setup.py::find_files`
  sorts lexicographically (`Cell1_1, Cell1_10, ..., Cell1_2`), which scrambles a numbered
  time course. `core/pipeline.py` re-sorts with `analysis/volumetric/ordering.py` for
  every mode except `xyt`, and `group_timelapse` orders series the same way, so barcode
  rows follow the numbering. `xyt` deliberately keeps `find_files`' exact order so the
  published 2D reference CSVs stay byte-identical.
- **Denominators are shared, because cropping is off by default.** `crop_to_mask` defaults
  to **False**, so the analysed field is the whole acquired field of view and every
  fraction-of-volume column has one denominator across files and timepoints. Turning
  `crop_to_mask` on restores the old per-file bounding box and reintroduces the mismatch:
  an object shrinking and the crop tightening around it are then indistinguishable, and
  the barcode PNG normalises per column across exactly those rows. Compare the µm³
  columns if you do.
- **Time-lapse memory.** A series is held in RAM at once: 15 timepoints of Cell1 at the
  union crop is ~9.0M voxels x 15, fine as uint16, but a longer series or a larger union
  box will grow linearly.
- **The hackathon 2D reference datasets are not on this machine** (CLAUDE.md points at
  `C:\Users\frank\...`), so the reference-CSV regression documented in the plan was
  replaced by the synthetic end-to-end 2D run plus the empty diff over the 2D files.

## A 2D defect found and fixed

`find_island_properties` in `analysis/binarization.py` computed
`np.partition(distances, k_num + 1)` over an N-by-N matrix of island centroids. With a
single island that is a 1x1 matrix and kth=1, which numpy rejects. `analysis/run.py`
swallowed the exception and wrote a **blank binarization row**, so the failure was
invisible.

It is not a rare shape. The top and bottom slices of a z-stack through a single cell
routinely contain exactly one island, and it took out one file in fifteen of the real
Jurkat series in `xyz` mode -- with no config workaround, since the crash happens for any
`neighbor_island_fraction`.

Fixed with a two-line guard returning NaN, which is what the metric means when there is
nothing to be separated from, and what the 3D branch already returned. The change can
only affect a path that previously *raised*, and a 2D reference run before and after is
byte-identical across all 28 columns.

## Another pre-existing 2D defect (not fixed)

`utils/find_analysis_frames` divides its step by 5 until it drops below the frame count,
which makes the step a **float**, and `range()` then raises
`TypeError: 'float' object cannot be interpreted as an integer`. It triggers whenever
`frame_step >= frame count` — e.g. a 6-frame movie at the default step of 10.

In the 2D pipeline the exception is swallowed by `analysis/run.py`'s per-branch
`try/except` and logged, so the affected rows are written **blank** rather than failing
visibly. Any dataset with fewer frames than `frame_step` is silently losing metrics.

Not fixed here: `utils/` is on the do-not-modify list for this work. The volumetric path
uses its own integer-safe `select_frame_indices`, which is pinned by test to agree with
the 2D helper everywhere the 2D helper functions. Worth raising separately.
