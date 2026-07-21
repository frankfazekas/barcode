import os
import re
import tkinter as tk
from tkinter import ttk, filedialog

from utils.gui import create_option_section, create_popup, volumetric_submode_var

from gui.config import BarcodeConfigGUI, InputConfigGUI


def create_volumetric_frame(parent, config: BarcodeConfigGUI, input_config: InputConfigGUI):
    """Create the Volumetric settings tab.

    A separate tab rather than additions to the existing branch tabs, so the 2D preview
    code paths are untouched. Every control here is inert unless "Volumetric Analysis"
    is ticked on the Execution Settings tab.
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

    # No "Enable Volumetric" checkbox here: that switch lives on Execution Settings, so
    # the 2D-or-3D decision is made once, before anything else. (There used to be one,
    # bound to cv.enabled, back when core/pipeline.py gated on that flag. Commit 7b972d9
    # replaced the gate with mode.key and the checkbox stayed behind doing nothing.)
    #
    # What belongs here is the choice the Execution checkbox cannot express: given that
    # the third axis IS depth, do we measure the stack as one solid object or as a pile
    # of separate 2D slices? That is the xyzt/xyz split, and it is a z-stack question.
    submode = volumetric_submode_var(config)

    tk.Label(frame, text="Measure each z-stack as:").grid(
        row=row_idx, column=0, columnspan=3, sticky="w", padx=5, pady=(0, 2)
    )
    row_idx += 1

    SUBMODES = [
        # States what this choice alone gives you. Meshing, curvature and 3D flow are
        # also only available here, but each needs its own setting below (and meshing a
        # segmentation), so listing them as though they came for free overpromised.
        ("xyzt", "One 3D object",
         "Volumes, sphericity and 3D connectivity, measured on the whole stack at once. "
         "Surface meshing, curvature and 3D flow are possible only here, each switched "
         "on separately below."),
        ("xyz", "Separate 2D slices through depth",
         "Each slice measured as a flat image; Change metrics describe variation with "
         "depth. No flow — displacement between focal planes is not motion."),
    ]
    submode_widgets = []
    for key, title, blurb in SUBMODES:
        radio = tk.Radiobutton(frame, variable=submode, value=key)
        radio.grid(row=row_idx, column=0, sticky="w", padx=(20, 0))
        title_label = tk.Label(frame, text=title, font=("TkDefaultFont", 13))
        title_label.grid(row=row_idx, column=0, sticky="w", padx=(45, 5))
        blurb_label = tk.Label(frame, text=blurb, fg="gray40", wraplength=560,
                               justify="left", font=("TkDefaultFont", 11))
        blurb_label.grid(row=row_idx + 1, column=0, columnspan=3, sticky="w", padx=(45, 5))
        submode_widgets.append((radio, title_label, blurb_label))
        row_idx += 2

    mode_state = tk.Label(frame, wraplength=620, justify="left")
    mode_state.grid(row=row_idx, column=0, columnspan=3, sticky="w", padx=5, pady=(4, 5))

    def _show_mode(*_args):
        """Grey the whole section out when Volumetric is off, rather than let it look live."""
        key = cv.analysis_mode.get()
        active = key != "xyt"
        for radio, title_label, blurb_label in submode_widgets:
            state = "normal" if active else "disabled"
            radio.config(state=state)
            title_label.config(fg="black" if active else "gray60")
            blurb_label.config(fg="gray40" if active else "gray70")
        if active:
            mode_state.config(
                text=f"Volumetric analysis is ON (mode {key}) — these settings apply.",
                fg="#15803d")
        else:
            mode_state.config(
                text="Volumetric analysis is OFF — nothing on this tab is used. Tick "
                     "“Volumetric Analysis” on the Execution Settings tab to turn it on.",
                fg="#b45309")

    cv.analysis_mode.trace_add("write", _show_mode)
    _show_mode()
    row_idx += 1

    # Where each piece of "how to read this file" is set. The single most common
    # confusion is the channel: there is no channel control on this tab, because it lives
    # on Execution Settings and is shared with the 2D path -- and "Parse All Channels" is
    # silently a no-op in the volumetric modes (analysis/volumetric/run.py analyses the
    # one selected channel and warns). Said here, next to the geometry settings that DO
    # live on this tab, so the division is visible without reading the log after a run.
    tk.Label(frame, text="Reading the file", font=header).grid(
        row=row_idx, column=0, columnspan=3, sticky="w", padx=(5, 5), pady=(10, 5)
    )
    row_idx += 1

    channel_note = tk.Label(frame, wraplength=640, justify="left", fg="gray25")
    channel_note.grid(row=row_idx, column=0, columnspan=3, sticky="w", padx=5, pady=(0, 2))

    def _describe_channel(*_args):
        # Reads the SAME variables the Execution tab drives, so it always mirrors what is
        # set there rather than keeping a second copy.
        if config.channels.parse_all_channels.get():
            channel_note.config(
                text="Channel: “Parse All Channels” is set on Execution Settings, but the "
                     "volumetric modes analyse ONE channel per run — the run will use "
                     "“Choose Channel” and warn. Untick Parse All Channels, and run the "
                     "other channels separately by changing Choose Channel.",
                fg="#b45309")
            return
        try:
            ch = config.channels.selected_channel.get()
        except tk.TclError:
            return
        where = f"channel {ch}" + (" (counting back from the last)" if ch < 0 else "")
        channel_note.config(
            text=f"Channel: set on the Execution Settings tab (“Choose Channel”) — "
                 f"currently {where}. There is no channel control here; one channel is "
                 f"analysed per run.",
            fg="gray25")

    config.channels.selected_channel.trace_add("write", _describe_channel)
    config.channels.parse_all_channels.trace_add("write", _describe_channel)
    _describe_channel()
    row_idx += 1

    tk.Label(
        frame,
        text="Everything else about reading the stack is set below on this tab: the "
             "voxel size, the axis order (including which axis is the channel), and the "
             "z / time ranges. Left at their defaults, voxel size and axis order are "
             "read from the file’s own metadata.",
        wraplength=640, justify="left", fg="gray25",
    ).grid(row=row_idx, column=0, columnspan=3, sticky="w", padx=5, pady=(0, 4))
    row_idx += 1

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

    xy_label = tk.Label(frame, text="XY Pixel Size [microns] (0 = read from file)")
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

    tk.Label(frame, text="File Layout", font=header).grid(
        row=row_idx, column=0, columnspan=3, sticky="w", padx=(5, 5), pady=(10, 5)
    )
    row_idx += 1

    axes_label = tk.Label(frame, text="Axis Order Override (blank = read from file)")
    axes_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    ttk.Entry(frame, textvariable=cv.axes_override, width=9).grid(
        row=row_idx, column=1, padx=5, pady=5
    )
    create_popup(
        frame,
        "The true axis order, one letter per dimension, for a file whose header is "
        "wrong -- for example 'TZYX' for a stack that declares 'ZCYX' because the "
        "microscope wrote its timepoints into ImageJ's 'channels' field. Leave blank "
        "to trust the file. BARCODE never guesses an axis order, but it will accept "
        "being told: a wrong entry here silently reinterprets every axis, so check it "
        "against the shape the log prints.",
        row_idx, axes_label,
    )
    row_idx += 1

    tk.Label(frame, text="Z Range", font=header).grid(
        row=row_idx, column=0, columnspan=3, sticky="w", padx=(5, 5), pady=(10, 5)
    )
    row_idx += 1

    units_label = tk.Label(frame, text="Z Range Units")
    units_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    ttk.Combobox(
        frame, textvariable=cv.z_range_units,
        values=["acquired", "isotropic", "microns"], width=10, state="readonly",
    ).grid(row=row_idx, column=1, sticky="w", padx=5, pady=5)
    create_popup(
        frame,
        "How the two values below are read. 'acquired' = slice indices in the stack as "
        "acquired. 'isotropic' = indices on the isotropic grid a segmentation lives on, "
        "which has far more slices (54 vs ~249 on 0.3/0.065 um data). 'microns' = "
        "physical depth from the bottom, which cannot be misread either way.",
        row_idx, units_label,
    )
    row_idx += 1

    z_start_label = tk.Label(frame, text="Z Range Start")
    z_start_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    ttk.Spinbox(
        frame, from_=0, to=10000, increment=1, textvariable=cv.z_start, width=7
    ).grid(row=row_idx, column=1, padx=5, pady=5)
    create_popup(
        frame,
        "Start of the range, in the units chosen above. The full stack is often not the "
        "right range: slices past the object are background, and including them drags "
        "every metric toward it.",
        row_idx, z_start_label,
    )
    row_idx += 1

    z_end_label = tk.Label(frame, text="Z Range End (0 = to the end)")
    z_end_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    ttk.Spinbox(
        frame, from_=-10000, to=10000, increment=1, textvariable=cv.z_end, width=7
    ).grid(row=row_idx, column=1, padx=5, pady=5)
    create_popup(
        frame,
        "End of the range, in the units chosen above. 0 always means to the end of the "
        "stack; for the two index units, negatives count back from the end.",
        row_idx, z_end_label,
    )
    row_idx += 1

    tk.Label(frame, text="Time Range", font=header).grid(
        row=row_idx, column=0, columnspan=3, sticky="w", padx=(5, 5), pady=(10, 5)
    )
    row_idx += 1

    t_units_label = tk.Label(frame, text="Time Range Units")
    t_units_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    ttk.Combobox(
        frame, textvariable=cv.t_range_units, values=["index", "seconds"],
        width=10, state="readonly",
    ).grid(row=row_idx, column=1, sticky="w", padx=5, pady=5)
    create_popup(
        frame,
        "How the two values below are read. 'index' counts timepoints from 0. "
        "'seconds' converts through the exposure time, so a range can be stated in the "
        "units the experiment was designed in rather than frame numbers that change if "
        "the acquisition rate does.",
        row_idx, t_units_label,
    )
    row_idx += 1

    t_start_label = tk.Label(frame, text="Time Range Start")
    t_start_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    ttk.Spinbox(
        frame, from_=0, to=100000, increment=1, textvariable=cv.t_start, width=7
    ).grid(row=row_idx, column=1, padx=5, pady=5)
    create_popup(
        frame,
        "First timepoint to analyse, in the units chosen above. Useful when the start of "
        "an acquisition is out of focus or pre-stimulus.",
        row_idx, t_start_label,
    )
    row_idx += 1

    t_end_label = tk.Label(frame, text="Time Range End (0 = to the end)")
    t_end_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    ttk.Spinbox(
        frame, from_=-100000, to=100000, increment=1, textvariable=cv.t_end, width=7
    ).grid(row=row_idx, column=1, padx=5, pady=5)
    create_popup(
        frame,
        "One past the last timepoint. 0 always means to the end of the series; for "
        "'index', negatives count back from the end.",
        row_idx, t_end_label,
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

    # Nobody should have to read a regex to find out whether their files will group
    # correctly. Show the actual answer for the folder that is actually selected, and
    # re-run it whenever the pattern or the folder changes.
    tl_match_label = tk.Label(frame, fg="gray25", wraplength=640, justify="left")
    tl_match_label.grid(row=row_idx, column=0, columnspan=3, sticky="w", padx=5, pady=(0, 6))

    def _describe_series(*_args):
        from analysis.volumetric.timelapse import group_timelapse

        directory = input_config.dir_path.get()
        if not directory or not os.path.isdir(directory):
            tl_match_label.config(
                text="Select a folder under Process Directory to preview the grouping.",
                fg="gray45")
            return
        try:
            pattern = cv.timelapse_regex.get()
        except tk.TclError:
            return
        paths = [
            os.path.join(root_dir, f)
            for root_dir, _, files in os.walk(directory)
            for f in files
            if f.lower().endswith((".tif", ".tiff", ".nd2")) and not f.startswith("._")
        ]
        if not paths:
            tl_match_label.config(text="No image files found in that folder.", fg="gray45")
            return
        try:
            groups, unmatched = group_timelapse(paths, pattern)
        except re.error as exc:
            tl_match_label.config(text=f"Not a valid pattern: {exc}", fg="#b91c1c")
            return

        if not groups:
            tl_match_label.config(
                text=f"This pattern matches none of the {len(paths)} files — every file "
                     f"would be analysed on its own, and change metrics would be NaN.",
                fg="#b91c1c")
            return

        shown = ", ".join(
            f"{g.series} ({len(g.paths)} frames)" for g in groups[:3]
        )
        if len(groups) > 3:
            shown += f", +{len(groups) - 3} more"
        text = f"Groups {len(paths)} files into {len(groups)} series: {shown}."
        colour = "#15803d"
        if unmatched:
            text += f"  {len(unmatched)} file(s) match nothing and would be skipped."
            colour = "#b45309"
        tl_match_label.config(text=text, fg=colour)

    cv.timelapse_regex.trace_add("write", _describe_series)
    input_config.dir_path.trace_add("write", _describe_series)
    input_config.mode.trace_add("write", _describe_series)
    _describe_series()
    row_idx += 1

    tk.Label(frame, text="Segmentation", font=header).grid(
        row=row_idx, column=0, columnspan=3, sticky="w", padx=(5, 5), pady=(10, 5)
    )
    row_idx += 1

    create_option_section(
        frame,
        row_idx,
        cv.segmentation_enabled,
        "Use Existing Segmentation Masks",
        "Use a pre-computed mask (e.g. from Cellpose) as the binarization instead of "
        "intensity thresholding. BARCODE does not create the mask; it locates an existing "
        "one per image via the pattern and template below.",
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

    mask_spacing_label = tk.Label(frame, text="Mask Voxel Size [microns] (0 = XY pixel size)")
    mask_spacing_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    ttk.Spinbox(
        frame, from_=0.0, to=10.0, increment=0.001, textvariable=cv.mask_spacing_um, width=9
    ).grid(row=row_idx, column=1, padx=5, pady=5)
    create_popup(
        frame,
        "Masks usually carry no spacing metadata. They are often saved on an isotropic "
        "grid finer than the image's Z spacing, so this defaults to the XY pixel size.",
        row_idx, mask_spacing_label,
    )
    row_idx += 1

    # Sits here, under the setting that controls it, rather than beside any one of the
    # options it explains -- they are spread across three sections further down.
    mask_note = tk.Label(frame, wraplength=620, justify="left", fg="#b45309",
                         font=("TkDefaultFont", 11))
    mask_note.grid(row=row_idx, column=0, columnspan=3, sticky="w", padx=5, pady=(2, 5))
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
        "Put the image and mask on one isotropic grid. Recommended: 3D connectivity and "
        "shape metrics assume equal spacing on every axis, and this data is ~4.6x "
        "anisotropic. The analysed field stays the whole acquired field of view, so "
        "every 'fraction of volume' metric keeps one denominator across files and "
        "timepoints. Requires a mask; without one there is no isotropic grid to resample "
        "onto and the acquired voxels are analysed as they are.",
    )
    row_idx += 2

    _w_crop = create_option_section(
        frame,
        row_idx,
        cv.crop_to_mask,
        "Crop to the Mask's Bounding Box",
        "Off, and it should normally stay off. Cropping gives each file its own "
        "denominator for every 'fraction of volume' metric, so an object shrinking and "
        "the crop box tightening around it become indistinguishable. Use the Z Range to "
        "analyse less than the full field. Kept for reproducing older cropped runs.",
    )
    row_idx += 2

    padding_label = tk.Label(frame, text="Crop Padding [voxels]")
    padding_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    padding_spin = ttk.Spinbox(
        frame, from_=0, to=50, increment=1, textvariable=cv.crop_padding_vox, width=7
    )
    padding_spin.grid(row=row_idx, column=1, padx=5, pady=5)
    create_popup(
        frame, "Voxels of margin kept around the mask bounding box. Only has an effect "
        "when 'Crop to the Mask's Bounding Box' is on.",
        row_idx, padding_label,
    )

    # Padding is only read inside _crop_to_mask_bbox, which runs only when crop_to_mask
    # is on (analysis/volumetric/resample.py). Grey it when cropping is off so it does
    # not look like a live setting.
    def _apply_crop_dependency(*_args):
        on = bool(cv.crop_to_mask.get())
        padding_spin.config(state="normal" if on else "disabled")
        padding_label.config(fg="black" if on else "gray60")

    cv.crop_to_mask.trace_add("write", _apply_crop_dependency)
    _apply_crop_dependency()
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
    create_popup(
        frame,
        "Number of bins in the voxel-intensity histogram the distribution metrics are "
        "measured from. More bins resolve finer structure but make each bin noisier; "
        "300 matches the 2D branch.",
        row_idx, bins_label,
    )
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


    min_island_label = tk.Label(frame, text="Minimum Island Size [voxels]")
    min_island_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    ttk.Spinbox(
        frame, from_=0, to=1000, increment=1,
        textvariable=cv.minimum_island_size, width=7
    ).grid(row=row_idx, column=1, padx=5, pady=5)
    create_popup(
        frame,
        "Objects and holes smaller than this are removed after thresholding. Ignored "
        "when a segmentation mask supplies the binarization.",
        row_idx, min_island_label,
    )
    row_idx += 1

    neighbour_label = tk.Label(frame, text="Neighbour Island Fraction")
    neighbour_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    ttk.Spinbox(
        frame, from_=0.01, to=1.0, increment=0.01,
        textvariable=cv.neighbor_island_fraction, width=7
    ).grid(row=row_idx, column=1, padx=5, pady=5)
    create_popup(
        frame,
        "Fraction of the other objects averaged over when computing Mean Island "
        "Separation. With a clean single-object mask there is nothing to separate from "
        "and that metric is reported as NaN regardless.",
        row_idx, neighbour_label,
    )
    row_idx += 1

    # Was "Percentage of Frames Evaluated" showing 0.05, which is a fraction, not a
    # percentage -- and the name suggested it decided how much of the series gets
    # analysed. It does not: Frame Step above does that. This picks how many frames at
    # each END of the analysed series are averaged to form the Change metrics, so it is
    # a window size, and the name now says so.
    pct_frames_label = tk.Label(frame, text="Change Window (fraction at each end)")
    pct_frames_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    ttk.Spinbox(
        frame, from_=0.05, to=0.5, increment=0.05,
        textvariable=cv.percentage_frames_evaluated, width=7
    ).grid(row=row_idx, column=1, padx=5, pady=5)
    create_popup(
        frame,
        "Every Change metric compares the average of the LAST few analysed frames with "
        "the average of the FIRST few. This sets how many that is, as a fraction of the "
        "analysed frames: ceil(fraction x frames), never fewer than 1.\n\n"
        "It does not decide how much of the series is analysed -- Frame Step does that. "
        "This only sets the two windows being compared.\n\n"
        "On short series it saturates: any value up to 1/N gives a 1-frame window, so "
        "for a 15-timepoint run everything below about 0.07 behaves identically.\n\n"
        "A single-timepoint run reports every Change metric as NaN regardless.",
        row_idx, pct_frames_label,
    )
    row_idx += 1

    # The fraction is unintuitive on its own -- 0.05 of what, and how many frames is
    # that? Spell out the arithmetic for a few series lengths, live.
    pct_example = tk.Label(frame, fg="gray40", font=("TkDefaultFont", 11),
                           wraplength=620, justify="left")
    pct_example.grid(row=row_idx, column=0, columnspan=3, sticky="w", padx=(30, 5),
                     pady=(0, 4))

    def _describe_change_window(*_args):
        import math

        try:
            fraction = float(cv.percentage_frames_evaluated.get())
        except (tk.TclError, ValueError):
            return
        parts = [f"{n} → {max(math.ceil(fraction * n), 1)}" for n in (10, 15, 50, 200)]
        pct_example.config(
            text="Frames at each end (analysed frames → window): " + ",  ".join(parts))

    cv.percentage_frames_evaluated.trace_add("write", _describe_change_window)
    _describe_change_window()
    row_idx += 1

    noise_label = tk.Label(frame, text="Intensity Noise Threshold")
    noise_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    ttk.Spinbox(
        frame, from_=0.0, to=1.0, increment=0.0001, format="%.4f",
        textvariable=cv.noise_threshold, width=9
    ).grid(row=row_idx, column=1, padx=5, pady=5)
    create_popup(
        frame,
        "Histogram bins holding less than this fraction of voxels are discarded before "
        "the intensity statistics are computed.",
        row_idx, noise_label,
    )
    row_idx += 1

    tk.Label(frame, text="Optional Metrics", font=header).grid(
        row=row_idx, column=0, columnspan=3, sticky="w", padx=(5, 5), pady=(10, 5)
    )
    row_idx += 1

    create_option_section(
        frame,
        row_idx,
        cv.enable_component_stats,
        "Per-Object Size Statistics",
        "Add object count, size SD, size skewness and median size. These describe the "
        "spread of the object-size distribution, which the mean and total cannot: one "
        "dominant object plus debris gives the same mean as several even ones. "
        "Volumetric modes only. Off by default so the barcode stays readable.",
    )
    row_idx += 2

    _w_packing = create_option_section(
        frame,
        row_idx,
        cv.enable_packing_topology,
        "Packing Topology",
        "Add mean contact number, contact number SD and hexagonal fraction: how objects "
        "are arranged relative to each other, which no other metric describes. In a "
        "space-filling monolayer sizes and separations are near-uniform and it is the "
        "neighbour-number distribution that changes. Needs an instance segmentation "
        "with integer labels -- in a confluent field, deriving objects by connectivity "
        "merges the whole tissue into one.",
    )
    row_idx += 2

    contact_label = tk.Label(frame, text="Minimum Contact [voxels]")
    contact_label.grid(row=row_idx, column=0, sticky="w", padx=(30, 5), pady=5)
    _sb_contact = ttk.Spinbox(
        frame, from_=1, to=1000, increment=1,
        textvariable=cv.packing_min_contact_voxels, width=7
    )
    _sb_contact.grid(row=row_idx, column=1, padx=5, pady=5)
    create_popup(
        frame,
        "Shared face area, in voxels, below which a touch is treated as a segmentation "
        "nick rather than a real interface between two objects.",
        row_idx, contact_label, indent=25,
    )
    row_idx += 1

    dilation_label = tk.Label(frame, text="Contact Gap Bridging [voxels]")
    dilation_label.grid(row=row_idx, column=0, sticky="w", padx=(30, 5), pady=5)
    _sb_dilation = ttk.Spinbox(
        frame, from_=0, to=20, increment=1,
        textvariable=cv.packing_contact_dilation_vox, width=7
    )
    _sb_dilation.grid(row=row_idx, column=1, padx=5, pady=5)
    create_popup(
        frame,
        "Grow labels by this many voxels before looking for contacts, so objects "
        "separated by a thin background gap -- a segmented membrane between cells -- "
        "still count as neighbours. 0 requires them to touch exactly.",
        row_idx, dilation_label, indent=25,
    )
    row_idx += 1

    _w_border = create_option_section(
        frame,
        row_idx,
        cv.packing_exclude_border_objects,
        "Exclude Border Objects From Packing",
        "Objects touching the edge of the field have an artificially low contact number. "
        "They stay in the graph so their interior neighbours keep full degree; only the "
        "reported statistics exclude them.",
        indent=25,
    )
    row_idx += 2

    create_option_section(
        frame,
        row_idx,
        cv.enable_slice_profile,
        "Broadest Slice & Clipping Flag",
        "Add the index, depth and area of the widest z slice -- the only metrics that "
        "say WHERE in depth something is, rather than reducing the stack to one number. "
        "For a stack through a curved surface or a rounded object it locates the "
        "equator, and it moves when the object flattens or tilts. Also raises flag "
        "digit 6 when foreground reaches an edge of the analysed field, meaning the "
        "object continues outside it and every size and shape metric describes a "
        "truncated object. Digit 6 is separate from digit 5 on purpose: 5 means YOU "
        "restricted the range, 6 means the DATA is cut off.",
    )
    row_idx += 2

    create_option_section(
        frame,
        row_idx,
        cv.enable_curvature_range,
        "Minimum & Maximum Curvature",
        "Add the extremes of the curvature field. Mean Curvature <H> averages the two "
        "principal curvatures together, so a saddle -- sharply curved both ways -- "
        "averages towards zero and reads as flat. These keep the most concave and most "
        "convex parts of the surface apart. Needs the mesh family, so it also needs a "
        "segmentation.",
    )
    row_idx += 2

    _w_maskint = create_option_section(
        frame,
        row_idx,
        cv.enable_mask_intensity,
        "In-Mask Intensity Statistics",
        "Add per-object MFI, SD, CV, skewness, entropy, normalized entropy and the "
        "fraction of voxels above twice the median. These describe how signal is "
        "distributed INSIDE each object -- the clustering readout -- whereas the "
        "intensity branch describes whatever voxels it is given, usually including "
        "background. A uniformly-filled object and one with bright foci have the same "
        "mean and very different CV and entropy. Needs a segmentation; with an instance "
        "mask each object is measured separately and the results averaged unweighted, "
        "so one large object does not outvote the rest.",
    )
    row_idx += 2

    create_option_section(
        frame,
        row_idx,
        cv.enable_intensity_magnitude,
        "Total & Mean Intensity",
        "Add total intensity, mean intensity, intensity SD and intensity per unit volume. "
        "These say HOW MUCH signal there is, where the intensity branch describes the "
        "SHAPE of the distribution and is blind to a uniform brightening. Sensitive to "
        "exposure and gain, so only compare acquisitions taken with identical settings.",
    )
    row_idx += 2

    create_option_section(
        frame,
        row_idx,
        cv.record_range_columns,
        "Record Analysed Range",
        "Add the z and t extents this row's numbers were computed over. Flag digit 5 "
        "already says THAT a range was restricted; these say WHICH, so a CSV separated "
        "from its Settings.yaml still describes itself. Indices are into the acquired "
        "data, before any isotropic resampling.",
    )
    row_idx += 2


    tk.Label(frame, text="3D Optical Flow", font=header).grid(
        row=row_idx, column=0, columnspan=3, sticky="w", padx=(5, 5), pady=(10, 5)
    )
    row_idx += 1

    xyz_sigma_label = tk.Label(frame, text="Spatial Smoothing Sigma")
    xyz_sigma_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    ttk.Spinbox(
        frame, from_=0.5, to=20.0, increment=0.5, textvariable=cv.flow_xyz_sigma, width=7
    ).grid(row=row_idx, column=1, padx=5, pady=5)
    create_popup(
        frame,
        "Gaussian smoothing applied in Z, Y and X before the velocity is solved. Larger "
        "values suppress noise but blur out small features. This is a gradient-based "
        "method, so it under-reports motion when the structures being tracked are the "
        "same size as this sigma or smaller; keep it below the feature scale.",
        row_idx, xyz_sigma_label,
    )
    row_idx += 1

    t_sigma_label = tk.Label(frame, text="Temporal Smoothing Sigma")
    t_sigma_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    ttk.Spinbox(
        frame, from_=1, to=10, increment=1, textvariable=cv.flow_t_sigma, width=7
    ).grid(row=row_idx, column=1, padx=5, pady=5)
    create_popup(
        frame,
        "Smoothing along time. Each analysed timepoint needs a CONTIGUOUS window of "
        "6*sigma+1 volumes centred on it — 7 at the default of 1. Series shorter than "
        "that are skipped entirely, and timepoints within half a window of either end "
        "of a series are skipped too. Raise this only for long, noisy time-lapses.",
        row_idx, t_sigma_label,
    )
    row_idx += 1

    w_sigma_label = tk.Label(frame, text="Lucas-Kanade Neighborhood Sigma")
    w_sigma_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    ttk.Spinbox(
        frame, from_=0.5, to=20.0, increment=0.5, textvariable=cv.flow_w_sigma, width=7
    ).grid(row=row_idx, column=1, padx=5, pady=5)
    create_popup(
        frame,
        "Size of the neighbourhood over which the flow equation is solved. Larger values "
        "give a better-conditioned solution but smooth over small-scale motion.",
        row_idx, w_sigma_label,
    )
    row_idx += 1

    rel_label = tk.Label(frame, text="Reliability Percentile (0 = keep all)")
    rel_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    ttk.Spinbox(
        frame, from_=0.0, to=99.0, increment=5.0,
        textvariable=cv.flow_reliability_percentile, width=7,
    ).grid(row=row_idx, column=1, padx=5, pady=5)
    create_popup(
        frame,
        "Voxels below this percentile of the solver's reliability score are dropped from "
        "every flow metric. In flat, textureless regions the flow equation is "
        "ill-conditioned and the velocity it returns is noise, so the default of 50 keeps "
        "only the better-conditioned half of the volume. Set to 0 to use every voxel.",
        row_idx, rel_label,
    )
    row_idx += 1

    interval_label = tk.Label(frame, text="Frame Interval [seconds] (0 = read from file)")
    interval_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    ttk.Spinbox(
        frame, from_=0.0, to=86400.0, increment=1.0,
        textvariable=cv.frame_interval_s, width=9,
    ).grid(row=row_idx, column=1, padx=5, pady=5)
    create_popup(
        frame,
        "Seconds between consecutive TIMEPOINTS — not the camera exposure, and not the "
        "z-step dwell. Speed is a distance divided by this, so it scales Speed and Speed "
        "Change directly and nothing else. SET IT: at 0 this falls back to the file's "
        "ImageJ 'finterval' tag, which on per-timepoint stacks often describes the z "
        "acquisition or is just left at 1, silently turning um/s into um/frame. The file "
        "cannot tell you which it is.",
        row_idx, interval_label,
    )
    row_idx += 1

    flow_down_label = tk.Label(frame, text="Flow Downsampling")
    flow_down_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    ttk.Spinbox(
        frame, from_=1, to=16, increment=1, textvariable=cv.flow_downsample, width=7
    ).grid(row=row_idx, column=1, padx=5, pady=5)
    create_popup(
        frame,
        "Block-average the volumes by this factor on every axis before solving. The "
        "reliability score needs a 3x3 eigendecomposition at every voxel, so large fields "
        "of view get slow; on a 196x243x189 nucleus, 2 cuts a window from ~54 s to ~8 s. "
        "Speed is barely affected, but the gradient-derived metrics are: on that same "
        "nucleus Curl roughly halved and Velocity Correlation Length nearly doubled going "
        "from 1 to 2, because block-averaging smooths the velocity field. Keep this value "
        "FIXED across a dataset — runs at different settings are not comparable.",
        row_idx, flow_down_label,
    )
    row_idx += 1

    create_option_section(
        frame,
        row_idx,
        cv.flow_use_mask,
        "Flow Metrics Inside Mask Only",
        "Restrict the flow metrics to voxels inside the segmentation. On by default "
        "because the analysed field is the whole acquired volume and the object occupies "
        "a small part of it, so background noise velocities would otherwise dominate the "
        "average speed. The flow is still solved on the whole "
        "volume, so gradients at the mask boundary use real neighbours. Has no effect "
        "when no segmentation is in use.",
    )
    row_idx += 2

    note = tk.Label(
        frame,
        text=("Note: in 3D the curl is a vector, so the Curl metric reports the mean "
              "magnitude ||curl v||. It is therefore always positive and does not carry "
              "the handedness the 2D Curl metric does. Mean Flow Direction remains the "
              "XY azimuth so it stays comparable with 2D runs, while Flow Directional "
              "Spread accounts for out-of-plane scatter as well."),
        wraplength=560,
        justify="left",
        fg="#666666",
    )
    note.grid(row=row_idx, column=0, columnspan=3, sticky="w", padx=5, pady=(15, 5))
    row_idx += 1

    tk.Label(frame, text="Surface Meshing", font=header).grid(
        row=row_idx, column=0, columnspan=3, sticky="w", padx=(5, 5), pady=(10, 5)
    )
    row_idx += 1

    create_option_section(
        frame,
        row_idx,
        cv.mesh_enabled,
        "Build Surface Mesh",
        "OFF by default; meshing runs only when this is ticked. It builds a triangulated "
        "surface and measures it: volume, surface area, sphericity, equivalent sphere "
        "radius and height (nine columns). With Use Existing Segmentation Masks on it "
        "meshes that mask; without one it meshes the intensity-thresholded volume, and "
        "the log says which was used. Needs an isotropic grid, so keep Resample to "
        "Isotropic Voxels on. Costs roughly 8 s per analysed timepoint.",
    )
    row_idx += 2

    # Meshing is the one branch users most often expect to happen "because it's 3D", so
    # state plainly whether it will run and what it will mesh -- source depends on whether
    # a segmentation is configured, decided in analysis/volumetric/run.py.
    mesh_status = tk.Label(frame, wraplength=640, justify="left", font=("TkDefaultFont", 11))
    mesh_status.grid(row=row_idx, column=0, columnspan=3, sticky="w", padx=(25, 5), pady=(0, 4))

    def _describe_meshing(*_args):
        if not cv.mesh_enabled.get():
            mesh_status.config(
                text="Meshing is OFF — no mesh columns are produced. Tick “Build Surface "
                     "Mesh” above to turn it on.",
                fg="gray45")
        elif cv.segmentation_enabled.get():
            mesh_status.config(
                text="Meshing is ON — it will mesh the segmentation mask.", fg="#15803d")
        else:
            mesh_status.config(
                text="Meshing is ON — with no segmentation it will mesh the "
                     "intensity-thresholded volume (set by Threshold Offset).",
                fg="#15803d")

    cv.mesh_enabled.trace_add("write", _describe_meshing)
    cv.segmentation_enabled.trace_add("write", _describe_meshing)
    _describe_meshing()
    row_idx += 1

    # Everything from here to the end of the meshing section only does anything when
    # meshing is on -- curvature rides on the mesh, and the geometry/smoothing settings
    # shape it. Rather than name each of ~15 widgets, the whole block is greyed by grid
    # row when "Build Surface Mesh" is off, so a dead control never looks live.
    mesh_first_row = row_idx

    create_option_section(
        frame,
        row_idx,
        cv.mesh_curvature,
        "Measure Surface Curvature",
        "Also compute per-face curvature and reduce it to Mean Curvature, Invagination "
        "Ratio and Concave Area Fraction. These describe how folded the surface is, "
        "which the volume and sphericity metrics cannot capture on their own.",
    )
    row_idx += 2

    # curvature_exclude_caps is intentionally not exposed here. It drops faces in the
    # lowest and highest z bin, which only makes sense for a segmentation genuinely
    # clipped by the ends of the stack -- a niche acquisition-artefact correction that
    # discards real surface on any object sitting inside the volume. It stays in
    # VolumetricConfig (default False) and is settable from a Settings.yaml or script for
    # that rare case; it does not belong beside the everyday curvature controls.
    outlier_label = tk.Label(frame, text="Curvature Outlier Limit (0 = keep all)")
    outlier_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    ttk.Spinbox(
        frame, from_=0.0, to=100.0, increment=0.5,
        textvariable=cv.curvature_outlier_limit, width=7
    ).grid(row=row_idx, column=1, padx=5, pady=5)
    create_popup(
        frame,
        "Drop faces whose mean curvature exceeds this in magnitude. Guards the averages "
        "against a handful of degenerate triangles, whose curvature can be enormous. 0 "
        "keeps every face, which is the default -- curvature is measured over the whole "
        "surface unless you ask otherwise.",
        row_idx, outlier_label,
    )
    row_idx += 1

    create_option_section(
        frame,
        row_idx,
        cv.mesh_export_obj,
        "Export Mesh as .OBJ",
        "Write the triangulated surface alongside the results so it can be opened in "
        "MeshLab, Blender or ParaView. Does not affect any metric.",
    )
    row_idx += 2

    maxrad_label = tk.Label(frame, text="Mesh Max Radius [voxels]")
    maxrad_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    ttk.Spinbox(
        frame, from_=0.5, to=50.0, increment=0.5, textvariable=cv.mesh_maxrad, width=7
    ).grid(row=row_idx, column=1, padx=5, pady=5)
    create_popup(
        frame,
        "Target triangle size, as a radius. Smaller gives a finer surface and a slower "
        "run. This is one radius for all three axes, which is why the mesh needs an "
        "isotropic grid.",
        row_idx, maxrad_label,
    )
    row_idx += 1

    maxrad_units_label = tk.Label(frame, text="Mesh Max Radius Units")
    maxrad_units_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    ttk.Combobox(
        frame, textvariable=cv.mesh_maxrad_units, values=["voxels", "um"],
        width=8, state="readonly",
    ).grid(row=row_idx, column=1, sticky="w", padx=5, pady=5)
    create_popup(
        frame,
        "How to read Mesh Max Radius. 'voxels' is the historical meaning, so the same "
        "number is a different physical size on every dataset; 'um' converts using the "
        "isotropic voxel size at meshing time, so one setting means the same physical "
        "thing across acquisitions. Neither adapts to object size.",
        row_idx, maxrad_units_label,
    )

    # The radius caption used to hardcode "[voxels]" while mesh_maxrad_units was
    # settable from YAML, so a run configured in microns showed a label saying voxels.
    def _label_maxrad_units(*_args):
        unit = cv.mesh_maxrad_units.get()
        maxrad_label.config(
            text=f"Mesh Max Radius [{'microns' if unit == 'um' else 'voxels'}]")

    cv.mesh_maxrad_units.trace_add("write", _label_maxrad_units)
    _label_maxrad_units()
    row_idx += 1

    area_frac_label = tk.Label(frame, text="Mesh Area Fraction")
    area_frac_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    ttk.Spinbox(
        frame, from_=0.01, to=1.0, increment=0.01, textvariable=cv.mesh_area_frac, width=7
    ).grid(row=row_idx, column=1, padx=5, pady=5)
    create_popup(
        frame, "Fraction of the target area kept when simplifying the surface.",
        row_idx, area_frac_label,
    )
    row_idx += 1

    isovalue_label = tk.Label(frame, text="Mesh Isovalue")
    isovalue_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    ttk.Spinbox(
        frame, from_=0.01, to=0.99, increment=0.01, textvariable=cv.mesh_isovalue, width=7
    ).grid(row=row_idx, column=1, padx=5, pady=5)
    create_popup(
        frame,
        "Where the isosurface sits inside a 0/1 mask. 0.5 is the boundary between the "
        "last foreground voxel and the first background one -- what a binary mask means, "
        "and the most accurate value on rounded objects. The old default 0.99 (from the "
        "MATLAB original) pulled the surface onto the outermost voxel CENTRES and shrank "
        "every object by ~0.7 voxels per side: -9% of the volume of a 32-voxel sphere, "
        "-67% of a 4-voxel one. Set 0.99 only to reproduce pre-change or MATLAB numbers.",
        row_idx, isovalue_label,
    )
    row_idx += 1

    smooth_iter_label = tk.Label(frame, text="Smoothing Iterations")
    smooth_iter_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    ttk.Spinbox(
        frame, from_=0, to=200, increment=1,
        textvariable=cv.mesh_smoothing_iterations, width=7
    ).grid(row=row_idx, column=1, padx=5, pady=5)
    create_popup(
        frame,
        "Surface smoothing passes. Smoothing shrinks the object slightly, so the mesh "
        "volume comes out a few percent below the voxel-counted volume. The Mesh Volume "
        "Ratio column reports exactly that: near 1.0 means the surface is faithful.",
        row_idx, smooth_iter_label,
    )
    row_idx += 1

    alpha_label = tk.Label(frame, text="Smoothing Alpha")
    alpha_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    ttk.Spinbox(
        frame, from_=0.0, to=1.0, increment=0.01,
        textvariable=cv.mesh_smoothing_alpha, width=7
    ).grid(row=row_idx, column=1, padx=5, pady=5)
    create_popup(
        frame,
        "Laplacian smoothing weight alpha: how far each vertex moves toward its "
        "neighbours on every pass. Higher smooths faster but shrinks the surface more. "
        "See Smoothing Beta for the paired term.",
        row_idx, alpha_label,
    )
    row_idx += 1

    beta_label = tk.Label(frame, text="Smoothing Beta")
    beta_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    ttk.Spinbox(
        frame, from_=0.0, to=1.0, increment=0.01,
        textvariable=cv.mesh_smoothing_beta, width=7
    ).grid(row=row_idx, column=1, padx=5, pady=5)
    create_popup(
        frame, "Laplacian smoothing weights (alpha, beta) passed to the mesh smoother.",
        row_idx, beta_label,
    )
    row_idx += 1

    mesh_last_row = row_idx

    def _apply_mesh_dependency(*_args):
        on = bool(cv.mesh_enabled.get())
        for r in range(mesh_first_row, mesh_last_row):
            for w in frame.grid_slaves(row=r):
                cls = w.winfo_class()
                if cls == "TCombobox":
                    # readonly comboboxes: re-enabling with "normal" would make them
                    # freely editable, which they are not meant to be.
                    w.config(state="readonly" if on else "disabled")
                elif cls in ("TSpinbox", "Checkbutton", "Radiobutton", "TButton", "Entry"):
                    w.config(state="normal" if on else "disabled")
                elif cls == "Label" and str(w.cget("text")).strip() not in ("ℹ️", "ℹ"):
                    w.config(fg="black" if on else "gray60")

    cv.mesh_enabled.trace_add("write", _apply_mesh_dependency)
    _apply_mesh_dependency()

    # Two controls deliberately absent from this section:
    #
    # mesh_matlab_compat reproduces the MATLAB original's quad-fan misreading of v2s'
    # face array when computing the decimation ratio. core/config.py calls it "a bug, not
    # a design choice"; it exists only to reproduce pre-port numbers, which is a scripted
    # comparison, not something to offer beside the real settings.
    #
    # mesh_iso2mesh_bin points at a staged copy of the CGAL executables. Left blank --
    # the default -- pyiso2mesh looks in ~/iso2mesh-tools/, downloads them if missing,
    # then falls back to PATH, so the binaries install themselves on first use and there
    # is nothing for a user to find. Setting it is an install-time concern (an offline
    # or air-gapped machine staging from an existing iso2mesh tree), which belongs in
    # setup rather than in the per-run settings.
    #
    # Both fields remain in VolumetricConfig and are still settable from a Settings.yaml
    # or a script; only the GUI controls are gone.

    # Several options do nothing at all without a segmentation. Each one's runner reports
    # it and carries on -- "Packing topology needs a segmentation; skipping." and the
    # like -- so before this the only way to discover that four ticked boxes had produced
    # nothing was to read the log after the batch finished. The tooltips said so, but
    # only on hover, and nothing stopped you. Now they are unavailable until a mask is
    # configured, with one line saying why.
    #
    # Curvature Range is included because it needs the mesh, and the mesh needs a mask.
    # Only what genuinely cannot run without a segmentation.
    #
    # Meshing, curvature and isotropic resampling used to be in this list and are not any
    # more: meshing now falls back to the binarized volume, curvature rides on that mesh,
    # and the isotropic grid is computed from the acquired spacing rather than taken from
    # the mask. Greying them would now be the misleading thing.
    MASK_DEPENDENT = (
        # Needs an INSTANCE label volume -- distinct objects to build a contact graph
        # between. A threshold gives one binary blob, not labelled objects.
        (_w_packing, "Packing Topology"),
        # Describes the inside of each segmented object; there is nothing to describe.
        (_w_maskint, "In-Mask Intensity Statistics"),
        # Crops to the mask's bounding box, which requires a mask to have a box.
        (_w_crop, "Crop to the Mask's Bounding Box"),
    )

    def _apply_mask_dependency(*_args):
        have_mask = bool(cv.segmentation_enabled.get())
        for (checkbox, title_label), _name in MASK_DEPENDENT:
            checkbox.config(state="normal" if have_mask else "disabled")
            title_label.config(fg="black" if have_mask else "gray60")

        # The three packing parameters follow Packing Topology itself, not the mask:
        # they are meaningless when the metric they configure is off, whatever the
        # segmentation says. (Packing Topology in turn needs the mask, so this is
        # stricter than the rule above, not a competing one.)
        packing_live = have_mask and bool(cv.enable_packing_topology.get())
        state = "normal" if packing_live else "disabled"
        colour = "black" if packing_live else "gray60"
        for widget in (contact_label, dilation_label):
            widget.config(fg=colour)
        for widget in (_sb_contact, _sb_dilation):
            widget.config(state=state)
        _w_border[0].config(state=state)
        _w_border[1].config(fg=colour)

        if have_mask:
            mask_note.config(text="")
        else:
            # Says what still works, not only what does not. Meshing without a mask is
            # the common case now, and a bare list of disabled things read as though the
            # whole 3D branch needed a segmentation.
            mask_note.config(
                text="Greyed out above because they need a segmentation: "
                     + ", ".join(name for _w, name in MASK_DEPENDENT)
                     + ".  Everything else still runs: the volume is binarized by "
                       "Threshold Offset, resampled to isotropic voxels, and meshed from "
                       "that surface. Supply a segmentation for a boundary you control "
                       "rather than one that follows the intensity threshold.")

    cv.segmentation_enabled.trace_add("write", _apply_mask_dependency)
    cv.enable_packing_topology.trace_add("write", _apply_mask_dependency)
    _apply_mask_dependency()

    return frame
