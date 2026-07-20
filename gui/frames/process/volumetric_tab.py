import tkinter as tk
from tkinter import ttk, filedialog

from utils.gui import create_option_section, create_popup

from gui.config import BarcodeConfigGUI, InputConfigGUI


def create_volumetric_frame(parent, config: BarcodeConfigGUI, input_config: InputConfigGUI):
    """Create the volumetric (3D) settings tab.

    A separate tab rather than additions to the existing branch tabs, so the 2D
    preview code paths are untouched. Every control here is inert unless "Enable
    Volumetric (3D) Analysis" is ticked.
    """
    frame = ttk.Frame(parent)

    cv = config.volumetric

    row_idx = 0
    header = ("TkDefaultFont", 15, "bold")
    frame.option_add("*font", "TkDefaultFont 13")

    tk.Label(frame, text="Volumetric Analysis", font=header).grid(
        row=row_idx, column=0, columnspan=3, sticky="w", padx=(5, 5), pady=(10, 5)
    )
    row_idx += 1

    create_option_section(
        frame,
        row_idx,
        cv.enabled,
        "Enable Volumetric (3D) Analysis",
        "Treat inputs as Z-stacks and compute true 3D metrics (volumes, 26-connectivity "
        "islands, 3D correlation lengths) instead of analysing them as 2D time series. "
        "When off, BARCODE behaves exactly as before. Note that a Z-stack read by the "
        "standard 2D pipeline is silently interpreted as a time series.",
    )
    row_idx += 2

    tk.Label(frame, text="Voxel Size", font=header).grid(
        row=row_idx, column=0, columnspan=3, sticky="w", padx=(5, 5), pady=(10, 5)
    )
    row_idx += 1

    z_label = tk.Label(frame, text="Z Step [microns] (0 = read from file)")
    z_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    ttk.Spinbox(
        frame, from_=0.0, to=10.0, increment=0.01, textvariable=cv.z_step_um, width=9
    ).grid(row=row_idx, column=1, padx=5, pady=5)
    create_popup(
        frame,
        "Spacing between Z slices. Left at 0 this is read from the ImageJ 'spacing' "
        "metadata field. Z spacing is usually much coarser than XY, and getting it "
        "wrong distorts every 3D shape metric.",
        row_idx, z_label,
    )
    row_idx += 1

    xy_label = tk.Label(frame, text="XY Step [microns] (0 = read from file)")
    xy_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    ttk.Spinbox(
        frame, from_=0.0, to=10.0, increment=0.001, textvariable=cv.xy_step_um, width=9
    ).grid(row=row_idx, column=1, padx=5, pady=5)
    create_popup(
        frame,
        "Microns per pixel in XY. Left at 0 this is read from the TIFF XResolution tag.",
        row_idx, xy_label,
    )
    row_idx += 1

    tk.Label(frame, text="Time-Lapse", font=header).grid(
        row=row_idx, column=0, columnspan=3, sticky="w", padx=(5, 5), pady=(10, 5)
    )
    row_idx += 1

    create_option_section(
        frame,
        row_idx,
        cv.timelapse_enabled,
        "Group Files Into Time Series",
        "For time-lapses exported one file per timepoint, group them back into a single "
        "series so the change metrics can be computed. Without this each volume is "
        "analysed alone and every change metric is NaN. Produces one row per series "
        "rather than one per file. Requires Process Directory.",
    )
    row_idx += 2

    tl_regex_label = tk.Label(frame, text="Series Pattern (regex):")
    tl_regex_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=2)
    tk.Entry(frame, textvariable=cv.timelapse_regex, width=35).grid(
        row=row_idx, column=1, columnspan=2, sticky="w", padx=5, pady=2
    )
    create_popup(
        frame,
        "Needs two named groups: 'series' (files sharing it form one time-lapse) and "
        "'frame' (numeric ordering). The default splits on the final underscore, so "
        "Cell1_1..Cell1_15 become one series ordered 1..15, and Cell1_centrin_3 "
        "correctly forms its own 'Cell1_centrin' series rather than joining Cell1.",
        row_idx, tl_regex_label,
    )
    row_idx += 1

    tk.Label(frame, text="Segmentation", font=header).grid(
        row=row_idx, column=0, columnspan=3, sticky="w", padx=(5, 5), pady=(10, 5)
    )
    row_idx += 1

    create_option_section(
        frame,
        row_idx,
        cv.segmentation_enabled,
        "Use Segmentation Masks",
        "Use a pre-computed mask as the binarization instead of intensity thresholding. "
        "The mask is located per image via the pattern and template below.",
    )
    row_idx += 2

    seg_root_label = tk.Label(frame, text="Segmentation Root Folder:")
    seg_root_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=2)
    tk.Entry(frame, textvariable=cv.segmentation_root, width=35).grid(
        row=row_idx, column=1, padx=5, pady=2
    )

    def browse_seg_root():
        chosen = filedialog.askdirectory(title="Select the Segmentation Root Folder")
        if chosen:
            cv.segmentation_root.set(chosen)

    tk.Button(frame, text="Browse Folder…", command=browse_seg_root).grid(
        row=row_idx, column=2, sticky="w", padx=5
    )
    create_popup(
        frame,
        "Folder the mask template is resolved against. Leave blank to look beside each "
        "image file.",
        row_idx, seg_root_label,
    )
    row_idx += 1

    regex_label = tk.Label(frame, text="Filename Pattern (regex):")
    regex_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=2)
    tk.Entry(frame, textvariable=cv.segmentation_regex, width=35).grid(
        row=row_idx, column=1, columnspan=2, sticky="w", padx=5, pady=2
    )
    create_popup(
        frame,
        "Named capture groups pulled out of the image filename, for use in the template "
        r"below. Example: Cell(?P<cell>\d+)_(?P<frame>\d+) captures cell and frame from "
        "Cell1_7.tif. {stem} is always available.",
        row_idx, regex_label,
    )
    row_idx += 1

    template_label = tk.Label(frame, text="Mask Path Template:")
    template_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=2)
    tk.Entry(frame, textvariable=cv.segmentation_template, width=35).grid(
        row=row_idx, column=1, columnspan=2, sticky="w", padx=5, pady=2
    )
    create_popup(
        frame,
        "Path to the mask, relative to the root folder, using the captured groups. "
        "Example: Cell{cell}/frame{frame}/nucleus/3D_seg/Cell_{cell}_SegMask_origFOV.tif. "
        "Use the original-field-of-view mask, not a cell-cropped one; a cropped mask will "
        "be rejected because its XY size does not match the image.",
        row_idx, template_label,
    )
    row_idx += 1

    mask_spacing_label = tk.Label(frame, text="Mask Voxel Size [microns] (0 = XY step)")
    mask_spacing_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    ttk.Spinbox(
        frame, from_=0.0, to=10.0, increment=0.001, textvariable=cv.mask_spacing_um, width=9
    ).grid(row=row_idx, column=1, padx=5, pady=5)
    create_popup(
        frame,
        "Masks usually carry no spacing metadata. They are often saved on an isotropic "
        "grid finer than the image's Z spacing, so this defaults to the XY step.",
        row_idx, mask_spacing_label,
    )
    row_idx += 1

    tk.Label(frame, text="Geometry", font=header).grid(
        row=row_idx, column=0, columnspan=3, sticky="w", padx=(5, 5), pady=(10, 5)
    )
    row_idx += 1

    create_option_section(
        frame,
        row_idx,
        cv.make_isotropic,
        "Resample to Isotropic Voxels",
        "Put the image and mask on one isotropic grid, then crop to the mask's bounding "
        "box. Recommended: 3D connectivity and shape metrics assume equal spacing on "
        "every axis. Note that cropping makes the 'fraction of volume' metrics relative "
        "to the cropped region rather than the original field. In a time series the crop "
        "is the union of all timepoints' boxes, so every timepoint shares one "
        "denominator. Requires a mask.",
    )
    row_idx += 2

    padding_label = tk.Label(frame, text="Crop Padding [voxels]")
    padding_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    ttk.Spinbox(
        frame, from_=0, to=50, increment=1, textvariable=cv.crop_padding_vox, width=7
    ).grid(row=row_idx, column=1, padx=5, pady=5)
    create_popup(
        frame, "Voxels of margin kept around the mask bounding box when cropping.",
        row_idx, padding_label,
    )
    row_idx += 1

    tk.Label(frame, text="Branch Parameters", font=header).grid(
        row=row_idx, column=0, columnspan=3, sticky="w", padx=(5, 5), pady=(10, 5)
    )
    row_idx += 1

    threshold_label = tk.Label(frame, text="Threshold Offset")
    threshold_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    ttk.Spinbox(
        frame, from_=-1.0, to=5.0, increment=0.01, textvariable=cv.threshold_offset, width=7
    ).grid(row=row_idx, column=1, padx=5, pady=5)
    create_popup(
        frame,
        "Binarization threshold as a fraction above the volume mean. Ignored when a "
        "segmentation mask is used, since the mask replaces thresholding entirely.",
        row_idx, threshold_label,
    )
    row_idx += 1

    step_label = tk.Label(frame, text="Frame Step")
    step_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    ttk.Spinbox(
        frame, from_=1, to=1000, increment=1, textvariable=cv.frame_step, width=7
    ).grid(row=row_idx, column=1, padx=5, pady=5)
    create_popup(
        frame,
        "Timepoints between analysed volumes. A single-timepoint stack analyses its one "
        "volume and reports change metrics as NaN.",
        row_idx, step_label,
    )
    row_idx += 1

    bins_label = tk.Label(frame, text="Intensity Histogram Bins")
    bins_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    ttk.Spinbox(
        frame, from_=1, to=10000, increment=1, textvariable=cv.bin_size, width=7
    ).grid(row=row_idx, column=1, padx=5, pady=5)
    row_idx += 1

    create_option_section(
        frame,
        row_idx,
        cv.intensity_use_mask,
        "Intensity Histogram Inside Mask Only",
        "Build the intensity distribution from voxels inside the segmentation only. This "
        "removes the background peak and changes the metrics substantially, so masked and "
        "unmasked runs are not comparable with each other.",
    )
    row_idx += 2

    create_option_section(
        frame,
        row_idx,
        cv.invert_binarization,
        "Invert Binarization",
        "Swap foreground and background before computing structural metrics.",
    )
    row_idx += 2

    note = tk.Label(
        frame,
        text=("Note: 3D optical flow is not implemented. With volumetric analysis enabled "
              "the flow branch is skipped and its metrics are reported as NaN."),
        wraplength=560,
        justify="left",
        fg="#666666",
    )
    note.grid(row=row_idx, column=0, columnspan=3, sticky="w", padx=5, pady=(15, 5))

    return frame
