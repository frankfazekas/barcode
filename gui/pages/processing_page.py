import tkinter as tk
from tkinter import ttk, messagebox

import threading, os
import traceback

# import tab frame creators directly
from gui.frames.process.execution_tab import create_execution_frame
from gui.frames.process.binarization_tab import create_binarization_frame
from gui.frames.process.flow_tab import create_flow_frame
from gui.frames.process.intensity_tab import create_intensity_frame
from gui.frames.process.volumetric_tab import create_volumetric_frame

from core import BarcodeConfig, InputConfig, PreviewConfig, AggregationConfig

# import configs directly
from gui.config import (
    BarcodeConfigGUI,
    InputConfigGUI,
    PreviewConfigGUI,
    AggregationConfigGUI,
)

from gui.window import setup_log_window

def create_tabs(parent, config, input_config, preview_config):
    core_config = BarcodeConfig()
    notebook = ttk.Notebook(parent, takefocus=0)
    notebook.pack(fill="both", expand=True)

    execution_frame = create_execution_frame(notebook, config, input_config)
    binarization_frame = create_binarization_frame(notebook, config, preview_config, input_config)
    flow_frame = create_flow_frame(notebook, config, preview_config, input_config)
    intensity_frame = create_intensity_frame(notebook, config, preview_config, input_config)
    volumetric_frame = create_volumetric_frame(notebook, config, input_config)

    notebook.add(execution_frame, text="Execution Settings")
    notebook.add(binarization_frame, text="Binarization Settings")
    notebook.add(flow_frame, text="Optical Flow Settings")
    notebook.add(intensity_frame, text="Intensity Distribution Settings")
    notebook.add(volumetric_frame, text="Volumetric Settings")

    return notebook

def create_processing_worker(
    config: BarcodeConfig,
    input_config: InputConfig,
    on_finished=None,
):
    """Create the worker function for processing in background thread.

    ``on_finished(outcome)`` is called on the way out with "done", "failed" or
    "cancelled" (the validation early-returns), whatever happened. It is what puts the
    Process Data button back and settles the log window's banner, so it has to run on
    every path -- hence the try/finally around the whole body.
    """

    if input_config.configuration_file:
        try:
            config = BarcodeConfig.load_from_yaml(input_config.configuration_file)
        except Exception as e:
            messagebox.showerror("Error reading config file", str(e))
            return

    def worker():
        outcome = "cancelled"
        try:
            mode = input_config.mode
            from core.pipeline import run_analysis

            # Handle file/directory processing
            file_path = input_config.file_path
            dir_path = input_config.dir_path

            if not (dir_path or file_path):
                messagebox.showerror(
                    "Error", "No file or directory has been selected."
                )
                return

            channels = config.channels.parse_all_channels
            channel_selection = config.channels.selected_channel
            length_conversion = input_config.length
            time_conversion = input_config.time
            config.reader.um_pixel_ratio /= length_conversion
            config.reader.exposure_time /= time_conversion

            if not (channels or (channel_selection is not None)):
                messagebox.showerror("Error", "No channel has been specified.")
                return

            dir_name = dir_path if dir_path else file_path
            run_analysis(dir_name, config, input_config)

        except Exception as e:
            outcome = "failed"
            print(f"Error during processing: {e}")
            print(traceback.format_exc())
            messagebox.showerror("Error during processing", str(e))

        else:
            # `else`, not `finally`. In a `finally` this fired unconditionally: right
            # after the error dialog on a crash, and after "No file or directory has been
            # selected" on the early returns above -- so a run that did nothing at all
            # still reported that it "finished successfully". `else` skips both, because
            # it runs only when the try block completes without raising *and* without
            # returning early.
            outcome = "done"
            messagebox.showinfo(
                "Processing Complete", "Analysis has finished successfully."
            )

        finally:
            if on_finished is not None:
                on_finished(outcome)

    return worker

def create_process_page(parent, switch_page):
    frame = tk.Frame(parent)
    frame.pack(fill="both", expand=True)

    gui_config = BarcodeConfigGUI()
    gui_input_config = InputConfigGUI()
    gui_preview_config = PreviewConfigGUI()
    gui_aggregation_config = AggregationConfigGUI()

    def on_run():
        log_win = setup_log_window(parent)

        # (Removed a leftover `gui_input_config.mode.set("dir")` debug line here. The
        # worker picks the path with `dir_path if dir_path else file_path` and never
        # reads `mode`, so forcing it changed no behaviour -- it only flipped the radio
        # button to "Process Directory" and greyed the file box after every run, which
        # read as a bug.)

        # Convert GUI configs to pure data configs
        config = gui_config.config
        input_config = gui_input_config.config

        # The button stayed live for the whole run, so a second click started a second
        # analysis over the same files, writing the same outputs from two threads.
        run_button.config(state="disabled", text="Processing…  (see Processing Log)")

        def finished(outcome):
            """Runs on the worker thread -- bounce onto the Tk thread before touching Tk."""
            def restore():
                run_button.config(state="normal", text="Process Data")
                banner = {
                    "done": ("Finished", "#15803d"),
                    "failed": ("Failed — see the error above", "#b91c1c"),
                    "cancelled": ("Stopped before starting", "#b45309"),
                }[outcome]
                try:
                    log_win.set_status(*banner)
                except tk.TclError:
                    pass  # the user closed the log window

            parent.after(0, restore)

        worker = create_processing_worker(config, input_config, on_finished=finished)
        threading.Thread(target=worker, daemon=True).start()

    back_button = ttk.Button(frame, text="← Back", command=lambda: switch_page("home"))
    back_button.pack(anchor="w", padx=20, pady=10)

    create_tabs(
        frame,
        gui_config,
        gui_input_config,
        gui_preview_config,
    )

    run_button = ttk.Button(frame, text="Process Data", command=on_run)
    run_button.pack(pady=10)

    return frame