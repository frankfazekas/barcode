# Running volumetric analysis from the GUI

Every value below was verified end to end on `Cell1_1.tif` / the Cell1 series. If your
run produces these numbers, the whole path is working.

```
mamba activate barcode
cd C:\Users\Upadhyaya_Lab\Code\barcode
python main.py
```

Home → **Process Data**. Five tabs; the new one is **Volumetric Settings**.

The on/off switch is **not** on that tab — it is **Volumetric Analysis** on
*Execution Settings*, so the 2D-or-3D decision is made once, before anything else.
Everything on the Volumetric tab is inert until it is ticked, so the 2D workflow is
unchanged when it is off.

## The four recipes

Work down this list — each adds one thing, so a failure tells you which piece broke.

### A. Single file, no mask — proves the reader and 3D metrics

*Execution Settings*: **Process File** → Browse File → `Cell1_1.tif`; Choose Channel `0`;
tick Image Binarization + Intensity Distribution; Verbose on; tick **Volumetric Analysis**.
*Volumetric tab*: nothing else needed.

Log should contain the line that proves it read a stack, not a movie:

```
Volumetric: (54, 312, 303) @ (0.3, 0.065, 0.065) um
```

`Cell1_1 Summary.csv`, **21 columns** (12 binarization + 6 intensity + 3 identity):

| metric | expect |
|---|---|
| Maximum Island Volume | 0.1374 |
| Mean Island Anisotropy | 6.5410 |
| Mean Island Separation | 2.2494 |
| Structural Correlation Length | 3.9969 |
| Maximum Kurtosis | 9.4707 |

~3 s. Change metrics are NaN and the seven flow columns are **absent** (dropped, not written
NaN, for a single timepoint) — both correct for one volume. The column is `Maximum Island
Volume`, not `… Area`: `xyzt` measures volumes. Note the analysed stack is resampled to
isotropic voxels (54 → 245 z-slices), so these numbers describe the resampled grid, not the
acquired 54-plane one.

### B. Add the mask — proves pairing and resampling

*Volumetric tab*: tick **Use Segmentation Masks**, set **Segmentation Root Folder** to
`...\all_cells_together\BARCODE\masks`. Leave Filename Pattern and Mask Path Template at
their defaults — the masks were named `Cell1_7_SegMask.tif` precisely so the defaults
`(?P<stem>.+)` and `{stem}_SegMask.tif` resolve them with nothing to type.

Still **21 columns** — a mask replaces the binarization, it does not add columns:

| metric | expect |
|---|---|
| Maximum Island Volume | 0.0893 |
| Mean Island Anisotropy | **1.4623** (was 6.5410) |
| Mean Island Separation | NaN |
| Structural Correlation Length | 3.9969 |

~4.5 s. **Anisotropy falling 6.54 → 1.46 is the signal the mask loaded** and isotropic
resampling engaged. If it stays near 6.5 the mask did not load — check the log. `Maximum Island
Volume` is a fraction of the **full** field (0.089 ≈ 8.9% — cropping is off by default, so the
box is not tightened to the nucleus); `Mean Island Separation` is NaN because a single nucleus
is one object.

### C. Add meshing — 11 more columns

*Volumetric tab*: tick **Mesh the Segmented Surface** and **Measure Surface Curvature**.

CSV grows to **32 columns**:

| metric | expect |
|---|---|
| Mesh Volume | 576.9 µm³ |
| Mesh Surface Area | 402.6 µm² |
| Sphericity | 0.8324 |
| Mesh Volume Ratio | 0.9951 |
| Mean Curvature | 0.1852 |
| Invagination Ratio | 0.4768 |

~8.5 s. Meshing is not bit-reproducible across processes, so the mesh numbers vary by a
fraction of a percent run to run — expect these to the first two or three digits, not exactly.
`Mesh Volume Ratio` is the fidelity check: the mesh volume over the voxel-counted volume. ~0.99
is healthy — smoothing shrinks the surface by a fraction of a percent. Far from 1.0 would mean
the surface is not tracking the mask.

### D. The whole series — the only run with live change metrics

*Execution Settings*: switch to **Process Directory** → `...\BARCODE\data`.
*Volumetric tab*: tick **Group Files Into Time Series**.

Log:

```
Series 1 of 1 -- Cell1: 15 timepoints (frames 1..15)
```

At the default **Frame Step 10** only 3 of the 15 are analysed — frames 1, 11 and 15 (see
*Things that will catch you out*) — and cropping is off by default, so each is the full
resampled field, not a tightened crop box. The values below are that 3-timepoint reduction; set
Frame Step 1 to analyse all 15.

`data Summary.csv` — **1 row** (one per series, not per file), **32 columns** (same schema as
recipe C; grouping fills the Change columns rather than adding any):

| metric | expect |
|---|---|
| Mesh Volume | ≈ 591 µm³ |
| Sphericity | 0.843 |
| Maximum Island Volume Change | 1.0355 |
| Kurtosis Change | −0.087 |

~29 s. The 7 flow columns are **absent** (dropped — flow was not run), so the only NaN column is
`Mean Island Separation` (a single nucleus is one object). `Maximum Island Volume Change` — not
`… Area Change`; the change ratio (deep/shallow ≈ 1.04) is scale-free and reproduces exactly,
while the mesh geometry varies by a fraction of a percent run to run.

## Things that will catch you out

**`frame_step` subsamples.** At its default of 10 a 15-timepoint series analyses frames
1, 11 and 15 (0-indexed positions 0, 10, 14) — three, not fifteen. That is why recipe D
takes 29 s rather than two minutes. Set **Frame Step** to 1 on the Volumetric tab to analyse
every timepoint.

**Meshing without a mask meshes the *threshold* volume — and for this data that is garbage.**
With Mesh ticked but Use Segmentation Masks unticked, meshing does **not** skip: it falls back
to the binarized (intensity-threshold) volume and says so in the log (*"meshing the BINARIZED
volume (no segmentation supplied)"*). Thresholding this data yields 167 disconnected specks, so
the mesh columns come out populated but describe halo, not nucleus. Supply a mask for a
meaningful nuclear mesh.

**Meshing keeps only the largest connected object.** Satellite fragments are discarded
silently. Correct for single-cell crops; in a genuinely multi-object field the mesh
columns would describe one object while the voxel columns describe all of them.

**Use the Browse buttons, never type a path.** `browse_file()` clears the directory
field; typing does not. The run uses `dir_path if dir_path else file_path`, so a stale
directory path silently wins over the file you typed and processes the whole folder.

**Time-lapse needs Process Directory.** Grouping happens across the file list, so it does
nothing on a single file.

**These Execution Settings do nothing in volumetric mode:** Micron to Pixel Ratio and
Exposure Time (spacing comes from the Volumetric tab or the file's ImageJ metadata), and
Save Graphs / Save Reduced Data Structures (no per-file 3D outputs exist yet).

**Output lands beside the input.** A run over `data\` writes `data Summary.csv`,
`data Summary Barcode (Channel 0).png`, `data Settings.yaml` and `Time.txt` into `data\`.
Move them into `results\` afterwards; see `dataset_layout.md`.

## Reading the barcode

Colour is normalised **per column across rows**, so compare down a column, never across.

Two things make columns look flat that are not broken:

* Six binarization columns carry unit `fraction of FOV` and are pinned to a **fixed
  [0, 1]** colour scale (`core/metrics.py`, `get_data_limits`); a seventh,
  `Maximal Area Slice Area`, joins them when the depth profile is on. With cropping off
  (the default) Cell1's `Maximum Island Volume` varies over ~0.087–0.093, i.e. **under 1% of
  the scale** — visually identical shades despite a real few-percent change. The **Change**
  columns are dynamically scaled and show the same trend at full contrast; read those.
* `Connectivity` is a genuine constant 0 — it measures percolation across the field, and
  an isolated nucleus never percolates.

A single-row barcode (recipe D) has no contrast by construction. For a barcode with one
row per timepoint use the CLI:

```
python scripts/run_volumetric_timelapse_barcode.py <data folder> --seg-root <masks folder> --mesh
```

## If something looks wrong

The log is the first place to look — most failures print a specific, actionable line
rather than raising. Mask problems in particular are designed to fail loudly: an
XY-size mismatch, a missing file, or an implausible Z extent each name the file and the
resolved path.
