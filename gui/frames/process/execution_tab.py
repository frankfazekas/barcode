import tkinter as tk
from tkinter import ttk, filedialog
from utils.gui import create_option_section, create_popup, volumetric_submode_var

# from core import BarcodeConfig, InputConfig
from gui.config import BarcodeConfigGUI, InputConfigGUI

def create_execution_frame(parent, config: BarcodeConfigGUI, input_config: InputConfigGUI):
    """Create the execution settings tab"""
    frame = ttk.Frame(parent)

    # Access config sections directly  
    ci = input_config
    cc = config.channels
    cr = config.reader
    co = config.writer
    cm = config.modules

    row_idx = 0
    header = ("TkDefaultFont", 15, "bold")
    frame.option_add("*font", "TkDefaultFont 13")


    # File/Directory Selection
    def browse_file():
        chosen = filedialog.askopenfilename(
            filetypes=[("TIFF Image", "*.tif *.tiff"), ("ND2 Document", "*.nd2"), ("MP4 File", "*.mp4"), ("AVI File", "*.avi")],
            title="Select a File",
        )
        if chosen:
            ci.file_path.set(chosen)
            ci.dir_path.set("")

    def browse_folder():
        chosen = filedialog.askdirectory(title="Select a Folder")
        if chosen:
            ci.dir_path.set(chosen)
            ci.file_path.set("")

    def on_mode_change():
        m = ci.mode.get()
        file_state = "normal" if m == "file" else "disabled"
        dir_state = "normal" if m == "dir" else "disabled"
        file_entry.config(state=file_state)
        browse_file_btn.config(state=file_state)
        dir_entry.config(state=dir_state)
        browse_folder_btn.config(state=dir_state)

    # Analysis mode governs the whole run -- which pipeline executes and which metrics
    # the output can carry -- so it sits above the data selection. See core/modes.py.
    from core.modes import MODES

    cvol = config.volumetric

    tk.Label(frame, text="Analysis Mode", font=header).grid(
        row=row_idx, column=0, columnspan=3, sticky="w", padx=(5, 5), pady=(10, 5)
    )
    row_idx += 1

    # One checkbox, not a three-way dropdown. "xyt / xyz / xyzt" named the axes rather
    # than the question, and the real question here is binary: is the third axis of this
    # file depth, or time? Which of the two volumetric modes is meant -- one 3D object,
    # or a stack of 2D slices -- is a z-stack question, so it lives on the Volumetric tab
    # next to everything else about z. analysis_mode still holds the bare key that
    # core/modes.py, the YAML and the CLI expect; this is purely how it is presented.
    submode = volumetric_submode_var(config)

    volumetric_on = tk.BooleanVar(value=cvol.analysis_mode.get() != "xyt")

    tk.Checkbutton(frame, variable=volumetric_on).grid(row=row_idx, column=0, sticky="w", padx=5)
    volumetric_caption = tk.Label(frame, text="Volumetric Analysis", font=("TkDefaultFont", 13))
    volumetric_caption.grid(row=row_idx, column=0, sticky="w", padx=(25, 5))
    # Everything this control needs to say lives in the popup, not on the page. The
    # wording states only what the switch itself decides -- which axis is which -- and
    # deliberately does NOT list metrics: meshing needs its own checkbox AND a
    # segmentation, curvature rides on the mesh, and 3D flow needs the flow branch and
    # more than one timepoint, so naming them here would promise output the run may not
    # produce. It also says nothing about "single vs time series", which is not a choice:
    # per-timepoint FILES are grouped by "Group Files Into Time Series", while a single
    # 4D file already carries its timepoints and needs no setting at all.
    create_popup(
        frame,
        (chr(10) * 2).join([
            "Which axis is the third one. Tick this when it is DEPTH (a z-stack); "
            "leave it unticked when it is TIME (an ordinary 2D movie).",
            "Unticked is the original BARCODE behaviour and the reference-validated "
            "path: each frame measured as a flat image, optical flow as a true velocity.",
            "Ticked, the Volumetric tab decides how the stack is measured: as one 3D "
            "object, or as separate 2D slices through depth.",
            "A z-stack left unticked is silently analysed as a time series, which "
            "reports flow between focal planes as though it were motion.",
        ]),
        row_idx, volumetric_caption,
    )
    row_idx += 1

    def describe_mode(*_args):
        # Keep the checkbox honest when the key is set from elsewhere (loading a YAML).
        should_be_on = cvol.analysis_mode.get() != "xyt"
        if volumetric_on.get() != should_be_on:
            volumetric_on.set(should_be_on)

    def _sync_from_widgets(*_args):
        """Checkbox + the Volumetric tab's sub-choice -> the one key everything reads."""
        wanted = submode.get() if volumetric_on.get() else "xyt"
        if cvol.analysis_mode.get() != wanted:
            cvol.analysis_mode.set(wanted)
        describe_mode()

    volumetric_on.trace_add("write", _sync_from_widgets)
    submode.trace_add("write", _sync_from_widgets)
    cvol.analysis_mode.trace_add("write", describe_mode)
    describe_mode()
    row_idx += 1

    tk.Label(frame, text="Select Data", font=header).grid(
        row=row_idx, column=0, columnspan=3, sticky="w", padx=(5, 5), pady=(10, 5)
    )
    row_idx += 1
    
    # Process File
    tk.Radiobutton(
        frame,
        text="Process File",
        variable=ci.mode,
        value="file",
        command=on_mode_change,
    ).grid(row=row_idx, column=0, sticky="w", padx=5, pady=2)

    file_entry = tk.Entry(frame, textvariable=ci.file_path, width=35)
    file_entry.grid(row=row_idx, column=1, padx=5, pady=2)

    browse_file_btn = tk.Button(frame, text="Browse File…", command=browse_file)
    browse_file_btn.grid(row=row_idx, column=2, sticky="w", padx=5)
    row_idx += 1

    # Process Directory
    tk.Radiobutton(
        frame,
        text="Process Directory",
        variable=ci.mode,
        value="dir",
        command=on_mode_change,
    ).grid(row=row_idx, column=0, sticky="w", padx=5, pady=2)

    dir_entry = tk.Entry(frame, textvariable=ci.dir_path, width=35)
    dir_entry.grid(row=row_idx, column=1, padx=5, pady=2)

    browse_folder_btn = tk.Button(frame, text="Browse Folder…", command=browse_folder)
    browse_folder_btn.grid(row=row_idx, sticky="w", column=2, padx=5)
    row_idx += 1

    tk.Label(frame, text="Select Channels", font=header).grid(
        row=row_idx, column=0, columnspan=3, sticky="w", padx=(5, 5), pady=(10, 5)
    )
    row_idx += 1

    # Channel selection
    channel_label = tk.Label(frame, text="Choose Channel:")
    channel_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    channel_spin = tk.Spinbox(
        frame, from_=-4, to=4, textvariable=cc.selected_channel, width=5
    )
    channel_spin.grid(row=row_idx, column=1, padx=(50, 5), pady=2)
    create_popup(
        frame,
        "Which channel of a multi-channel file to analyse. 0 is the first channel, 1 the "
        "second, and so on. A negative number counts back from the last channel: -1 is "
        "the last, -2 the second-to-last. A single-channel file is always channel 0.",
        row_idx, channel_label,
    )
    row_idx += 1

    create_option_section(
       frame,
       row_idx,
       cc.parse_all_channels,
       "Parse All Channels",
       "Analyse every channel of each file and write one row per channel, instead of the "
       "single channel chosen above. Not supported by Volumetric analysis, which reads "
       "one channel per run -- see the note below.",
    )
    row_idx += 1

    # The Parse-All-Channels / Volumetric conflict is surfaced HERE, where both controls
    # live, not only on the Volumetric tab. Volumetric is one checkbox up on this same
    # tab, so a user can tick both without ever opening the Volumetric tab; the run would
    # then quietly analyse channel 0 alone and only say so in the log.
    channel_conflict = tk.Label(frame, wraplength=560, justify="left", fg="#b45309",
                                font=("TkDefaultFont", 11))
    channel_conflict.grid(row=row_idx, column=0, columnspan=3, sticky="w", padx=25, pady=(0, 2))

    def _channel_conflict(*_args):
        both = cc.parse_all_channels.get() and cvol.analysis_mode.get() != "xyt"
        if not both:
            channel_conflict.config(text="")
            return
        # Name the channel the run will ACTUALLY use, not a guessed 0: it uses
        # selected_channel (see analysis/volumetric/run.py's channel-dropped warning),
        # which keeps whatever value it held when Parse All was ticked.
        try:
            ch = cc.selected_channel.get()
        except tk.TclError:
            ch = 0
        # "Choose Channel" is greyed while Parse All is ticked (the 2D mutual exclusion),
        # so the actionable instruction is to untick Parse All -- which re-enables it --
        # rather than "just change Choose Channel", which the user cannot reach.
        channel_conflict.config(
            text=f"Volumetric analysis reads ONE channel per run, so “Parse All Channels” "
                 f"does nothing here — the run will analyse channel {ch}. Untick “Parse "
                 f"All Channels” to enable “Choose Channel”, and run other channels one at "
                 f"a time.")

    cc.parse_all_channels.trace_add("write", _channel_conflict)
    cvol.analysis_mode.trace_add("write", _channel_conflict)
    cc.selected_channel.trace_add("write", _channel_conflict)
    _channel_conflict()
    row_idx += 1

    # Channel selection mutual exclusion
    def on_channels_toggled(*args):
        if cc.parse_all_channels.get():
            channel_spin.config(state="disabled")
        else:
            channel_spin.config(state="normal")

    cc.parse_all_channels.trace_add("write", on_channels_toggled)

    def on_channel_selection_changed(*args):
        if cc.selected_channel.get() is not None:
            cc.parse_all_channels.set(False)

    cc.selected_channel.trace_add("write", on_channel_selection_changed)

    tk.Label(frame, text="Specify Metadata", font=header).grid(
        row=row_idx, column=0, columnspan=3, sticky="w", padx=(5, 5), pady=(10, 5)
    )
    row_idx += 1

    micron_pixel_label = tk.Label(frame, text="Micron to Pixel Ratio (1 nm – 1 mm)")
    micron_pixel_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    um_pixel_spin = ttk.Spinbox(
        frame, from_=10**-3, to=10**3,
        increment=10**-3,
        textvariable=cr.um_pixel_ratio,
        width=9
    )
    um_pixel_spin.grid(row=row_idx, column=1, padx=5, pady=5)
    create_popup(frame, "Set ratio of physical units (in microns) to pixels in image. Automatically read for ND2 files.", row_idx, micron_pixel_label)
    row_idx += 1

    exp_time_label = tk.Label(frame, text="Exposure Time [seconds] (1 ms - 1 hour)")
    exp_time_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    frame_interval_spin = ttk.Spinbox(
        frame, from_=10**-3, to=3.6 * 10**3,
        increment=10**-3,
        textvariable=cr.exposure_time,
        width=7
    )
    frame_interval_spin.grid(row=row_idx, column=1, padx=5, pady=5)
    create_popup(frame, "Control interval (in seconds) between frames. Automatically read for ND2 files.", row_idx, exp_time_label)
    row_idx += 1

    length_units_label = tk.Label(frame, text="BARCODE Output Length Units")
    length_units_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    length_units_menu = ttk.Combobox(
        frame,
        textvariable=ci.length_units,
        values=["nm", "μm", "mm"],
        width=5,
        state="readonly"  # force selection from list
    )
    length_units_menu.grid(row=row_idx, column=1, sticky="w", padx=5, pady=5)
    row_idx += 1

    time_units_label = tk.Label(frame, text="BARCODE Output Time Units")
    time_units_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
    time_units_menu = ttk.Combobox(
        frame,
        textvariable=ci.time_units,
        values=["s", "min", "hr"],
        width=5,
        state="readonly"  # force selection from list
    )
    time_units_menu.grid(row=row_idx, column=1, sticky="w", padx=5, pady=5)
    row_idx += 1

    tk.Label(frame, text="Select Branches", font=header).grid(
        row=row_idx, column=0, columnspan=3, sticky="w", padx=(5, 5), pady=(10, 5)
    )
    row_idx += 1
    
    # Analysis modules
    create_option_section(
        frame,
        row_idx,
        cm.image_binarization,
        "Image Binarization",
        "Evaluate file(s) using Binarization branch (will generate a .CSV reduced data structure (RDS) for further analysis).",
    )
    row_idx += 2

    create_option_section(
        frame,
        row_idx,
        cm.optical_flow,
        "Optical Flow",
        "Evaluate file(s) using Optical Flow branch (will generate a .CSV reduced data structure (RDS) for further analysis).",
    )
    row_idx += 2

    create_option_section(
        frame,
        row_idx,
        cm.intensity_distribution,
        "Intensity Distribution",
        "Evaluate file(s) using Intensity Distribution branch (will generate a .CSV reduced data structure (RDS) for further analysis).",
    )
    row_idx += 2

    tk.Label(frame, text="Handling Dim Data", font=header).grid(
        row=row_idx, column=0, columnspan=3, sticky="w", padx=(5, 5), pady=(10, 5)
    )
    row_idx += 1

    # Options
    create_option_section(
        frame,
        row_idx,
        cr.accept_dim_images,
        "Scan dim files",
        "Include files that may be too dim to accurately profile (e.g. low light conditions, poor contrast).",
    )
    row_idx += 2

    create_option_section(
        frame,
        row_idx,
        cr.accept_dim_channels,
        "Scan dim channels",
        "Include channels that may be too dim to accurately profile (e.g. one channel is dim while others are better defined).",
    )
    row_idx += 2

    tk.Label(frame, text="Output Settings", font=header).grid(
        row=row_idx, column=0, columnspan=3, sticky="w", padx=(5, 5), pady=(10, 5)
    )
    row_idx += 1

    create_option_section(
        frame,
        row_idx,
        cr.verbose,
        "Verbose Output",
        "Provide additional information in the run-time Processing Log while the data is being processed (e.g. time step updates, total processing time, image dimness).",
    )
    row_idx += 2

    create_option_section(
        frame,
        row_idx,
        co.save_visualizations,
        "Save Graphs",
        "Save .PNG graphs representing chosen data structures (binarized images, optical flow fields, intensity distributions).",
    )
    row_idx += 2

    create_option_section(
        frame,
        row_idx,
        co.save_rds,
        "Save Reduced Data Structures",
        "Save .CSV reduced data structures for chosen branches (binarized images, optical flow fields, intensity distributions) for further analysis.",
    )
    row_idx += 2

    create_option_section(
        frame,
        row_idx,
        co.generate_barcode,
        "Generate Dataset Barcode",
        "Save an .PNG BARCODE matrix for the dataset, plotting the 23 BARCODE metrics for each channel in the dataset on a color-coded scale.",
    )
    row_idx += 2

    # Which metrics appear on the barcode image. The CSV always carries the full set for
    # the mode; this only trims the picture, so a trimmed run stays comparable with an
    # untrimmed one.
    tk.Label(frame, text="Barcode Metrics", font=header).grid(
        row=row_idx, column=0, columnspan=3, sticky="w", padx=(5, 5), pady=(10, 5)
    )
    row_idx += 1

    metric_menu = tk.Menubutton(frame, text="Choose Metrics to Show on the Barcode",
                                relief="raised")
    metric_menu.grid(row=row_idx, column=0, columnspan=2, sticky="w", padx=5, pady=2)
    metric_menu.menu = tk.Menu(metric_menu, tearoff=0)
    metric_menu["menu"] = metric_menu.menu
    metric_vars = {}

    def sync_hidden_metrics(*_args):
        co.hidden_barcode_metrics.clear()
        co.hidden_barcode_metrics.extend(
            name for name, var in metric_vars.items() if not var.get()
        )
        shown = len(metric_vars) - len(co.hidden_barcode_metrics)
        metric_menu.config(text=f"Barcode Metrics ({shown} of {len(metric_vars)} shown)")

    def rebuild_metric_menu(*_args):
        """Rebuild when the mode changes -- each mode produces a different metric set."""
        from core.results import ChannelResults

        try:
            mode_key = cvol.analysis_mode.get()
            names = ChannelResults.get_headers(
                just_metrics=True, mode=mode_key,
                include_components=bool(cvol.enable_component_stats.get()),
            )
        except Exception:
            return

        previously_hidden = set(co.hidden_barcode_metrics)
        metric_menu.menu.delete(0, "end")
        metric_vars.clear()
        for name in names:
            var = tk.IntVar(value=0 if name in previously_hidden else 1)
            var.trace_add("write", sync_hidden_metrics)
            metric_vars[name] = var
            metric_menu.menu.add_checkbutton(label=name, variable=var, onvalue=1, offvalue=0)
        sync_hidden_metrics()

    cvol.analysis_mode.trace_add("write", rebuild_metric_menu)
    cvol.enable_component_stats.trace_add("write", rebuild_metric_menu)
    rebuild_metric_menu()
    row_idx += 1

    tk.Label(frame, text="Configuration Settings", font=header).grid(
        row=row_idx, column=0, columnspan=3, sticky="w", padx=(5, 5), pady=(10, 5)
    )
    row_idx += 1

    # Configuration file
    config_label = tk.Label(frame, text="Configuration YAML File:")
    config_label.grid(
        row=row_idx, column=0, sticky="w", padx=5, pady=2
    )
    config_entry = tk.Entry(frame, textvariable=ci.configuration_file, width=35)
    config_entry.grid(row=row_idx, column=1, padx=5, pady=2)

    def browse_config_file():
        chosen = filedialog.askopenfilename(
            filetypes=[("YAML Files", "*.yaml"), ("YAML Files", "*.yml")],
            title="Select a Configuration YAML",
        )
        if chosen:
            ci.configuration_file.set(chosen)

    tk.Button(frame, text="Browse YAML...", command=browse_config_file).grid(
        row=row_idx, column=2, sticky="w", padx=5
    )

    create_popup(frame, "If desired, choose branch settings from a prior .YAML file.", row_idx, config_label)

    return frame


def _create_analysis_section(parent, row, title, var, description):
    """Helper to create analysis module sections"""
    tk.Label(parent, text=title, font=("TkDefaultFont", 10, "bold")).grid(
        row=row, column=0, columnspan=3, sticky="w", padx=5, pady=(10, 0)
    )

    tk.Checkbutton(parent, variable=var).grid(row=row + 1, column=0, sticky="w", padx=5)

    tk.Label(parent, text=description).grid(
        row=row + 1, column=0, sticky="w", padx=(25, 5), pady=(0, 0)
    )