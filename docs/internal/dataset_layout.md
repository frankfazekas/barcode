# Dataset layout for volumetric runs

The layout used for `Control_3D_CD3\all_cells_together\BARCODE`:

```
BARCODE/
  data/     Cell1_1.tif  ... Cell1_15.tif            input images, nothing else
  masks/    Cell1_1_SegMask.tif ... Cell1_15_SegMask.tif
  results/
    timepoints_with_masks/    one row per timepoint, mask as binarization
    timepoints_no_masks/      one row per timepoint, intensity threshold
    per_frame/                run_volumetric_batch.py output
    gui_run1_single_file/     a GUI single-file run, kept as a reference
    _superseded/              stale output + WHY_SUPERSEDED.txt
```

`masks/` is a **copy**; the originals stay in
`prog_live_cells/Cell{n}/frame{m}/nucleus/3D_seg/Cell_{n}_SegMask_origFOV.tif` and are
untouched. Each copy was sha256-verified against its source.

## Why the mask filenames end in `_SegMask.tif`

They match BARCODE's *default* segmentation settings, so the only field that needs
filling in is the root folder:

| setting | value |
|---|---|
| Segmentation Root Folder | `...\BARCODE\masks` |
| Filename Pattern (regex) | leave default `(?P<stem>.+)` |
| Mask Path Template | leave default `{stem}_SegMask.tif` |

`Cell1_7.tif` resolves to `masks\Cell1_7_SegMask.tif`. Nothing to type, nothing to typo.

## The caveat: GUI runs write into the folder they process

`utils/setup.py::setup_paths` puts the Summary CSV, barcode PNG, Settings YAML and
Time.txt in the directory being processed. So a GUI run over `data/` drops its output
back into `data/`, and `data/` stops being images-only.

The CLI scripts (`run_volumetric_timelapse_barcode.py`, `run_volumetric_batch.py`)
default their output into `results/` instead, so they leave `data/` alone. After a GUI
run, move the four generated files into a `results/` subfolder by hand.

Changing this properly means touching `utils/setup.py`, which is on the do-not-modify
list for the volumetric work — it is shared with the 2D pipeline.

## Naming outputs so they stay readable

The stale pile that prompted this cleanup had two failure modes worth avoiding:

* **Orphaned pairs.** A barcode PNG and the CSV it came from had different timestamps,
  because a later run overwrote one and not the other. Keep a run's CSV and PNG in the
  same folder and write them together.
* **Undated results that outlive a bug fix.** One CSV predated the radial-average
  bin-centre correction, so its correlation lengths were quietly wrong, and nothing in
  the filename said so. If a result matters, note which code produced it.
