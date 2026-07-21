#!/usr/bin/env python3
"""
Pure dataclass configurations - no tkinter dependencies.
Generate GUI wrappers by running: python _config.py
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List
import yaml, csv, os
from abc import ABC
import numpy as np
import imageio.v3 as iio
import av, nd2

"""
DEVELOPING GUIDE:

1. Define your configuration options here.

2. Add them to the __init__.py file in the core module to export them.

3. Add them to the `GUI_CONFIG_CLASSES` list at the bottom.

4. Run this file to generate GUI wrappers in the `gui` module.

    python core/_config.py
    
5. Use the generated GUI classes in your application.

    `from gui.config import BarcodeConfigGUI as BarcodeGUI`

"""

@dataclass
class BaseConfig(ABC):
    """Base class for all configuration sections."""

    @classmethod
    def from_dict(cls, data: dict) -> "BaseConfig":
        """Create config instance from dictionary."""
        return cls(**data)

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }
    
@dataclass
class InputConfig(BaseConfig):
    """File and data input configuration."""

    file_path: str = ""
    dir_path: str = ""
    mode: str = "file"  # "file", "dir", "agg", "comp"
    configuration_file: str = ""
    length_units: str = "μm"
    time_units: str = "s"

    def _apply_units(self):
        """Re-derive every unit label from the chosen length and time units.

        This used to set only LENGTH, AREA and SPEED. The volumetric work added six more
        unit strings, all with "μm" or "s" baked in, and none of them were updated -- so a
        run with length_units="nm" produced a CSV declaring "Maximum Island Area" in nm^2
        alongside "Mesh Volume" in μm^3 and "Mean Curvature <H>" in 1/μm, and a run with
        time_units="min" labelled Speed μm/min while Divergence still said 1/s. The
        numbers were right and the labels were not, which for metrics whose whole point is
        being physical is the same class of error the mode system exists to prevent.
        """
        from core.metrics import Units

        length, time = self.length_units, self.time_units
        Units.LENGTH = length
        Units.AREA = f"{length}^2"
        Units.VOLUME = f"{length}^3"
        Units.SPEED = f"{length}/{time}"
        Units.CURVATURE = f"1/{length}"
        Units.RATE = f"1/{time}"
        Units.INTENSITY_PER_AREA = f"a.u./{length}^2"
        Units.INTENSITY_PER_VOLUME = f"a.u./{length}^3"

    @property
    def length(self):
        self._apply_units()
        _length = {"nm": 1e-3, "μm": 1, "mm": 1e3}
        return _length[self.length_units]
    
    @property
    def time(self):
        self._apply_units()
        _time = {"s": 1, "min": 60, "hr":3600}
        return _time[self.time_units]

@dataclass
class ChannelConfig(BaseConfig):
    """Channel selection and processing configuration."""
    parse_all_channels: bool = False
    selected_channel: int = 0  # -3 to 4 range

@dataclass
class PreviewConfig(BaseConfig):
    """GUI preview and visualization settings."""

    sample_file: str = ""
    enable_live_preview: bool = True
    _sample_file: str = ""
    _sample_preview: np.ndarray = None
    preview_frame_number: int = 0

@dataclass
class AggregationConfig(BaseConfig):
    """CSV aggregation and post-processing configuration."""

    output_location: str = ""
    generate_single_barcode: bool = False
    generate_comparison_barcodes: bool = False
    separate_channels: bool = False
    sort_parameter: str = "Default"  # One of the metric headers
    csv_paths_list: List[str] = field(default_factory=list)
    metrics_list: List[bool] = field(default_factory=list)

@dataclass
class ComparisonConfig(BaseConfig):
    """BARCODE CSV post-processing parameter comparison configuration"""

    csv_location: str = ""
    first_comparison_metric: str = "Connectivity"
    second_comparison_metric: str = "Connectivity"
    output_location: str = ""

@dataclass
class VisualizationConfig(BaseConfig):
    file_path: str = ""
    rds_type: str = ""
    scale: float = 1.0
    preview_frame_number: int = 0
    video_framerate: float = 1.0
    _file_path: str = ""
    _frames: List[np.ndarray] = field(default_factory=list)
    _indices: List[int] = field(default_factory=list)

@dataclass
class ModuleConfig(BaseConfig):
    """Analysis module selection and coordination"""
    
    image_binarization: bool = False
    optical_flow: bool = False
    intensity_distribution: bool = False
    
@dataclass
class ReaderConfig(BaseConfig):
    accept_dim_channels: bool = False
    accept_dim_images: bool = False
    exposure_time: float = 1.0
    um_pixel_ratio: float = 1.0
    verbose: bool = False

@dataclass
class WriterConfig(BaseConfig):
    generate_barcode: bool = False
    save_rds: bool = False
    save_visualizations: bool = False

    # Metrics to leave OFF the barcode image, by name. The CSV always carries the full
    # set for the analysis mode -- hiding a column there would break the round-trip and
    # make two runs of the same mode incomparable -- so this is purely about what the
    # picture shows. Empty means show everything the mode produced.
    hidden_barcode_metrics: List[str] = field(default_factory=list)

@dataclass
class BinarizationConfig(BaseConfig):
    threshold_offset: float = 0.1
    frame_step: int = 10
    percentage_frames_evaluated: float = 0.05
    bin_factor: int = 2
    neighbor_island_fraction: float = 0.1
    minimum_island_size: int = 1
    enable_physical_units: bool = False
    invert_binarization: bool = False


@dataclass
class OpticalFlowConfig(BaseConfig):
    frame_step: int = 10
    win_size: int = 32
    downsample: int = 8
    percentage_frames_evaluated: float = 0.05

@dataclass
class IntensityDistributionConfig(BaseConfig):
    bin_size: int = 300
    frame_step: int = 10
    noise_threshold: float = 5e-4
    percentage_frames_evaluated: float = 0.05

@dataclass
class VolumetricConfig(BaseConfig):
    """Volumetric (3D) side-car pipeline.

    Inert unless ``enabled`` is True: with the default settings BARCODE behaves
    exactly as it did before this config existed. See ``analysis/volumetric/``.
    """

    # What ONE ROW of the barcode is -- see core/row_axis.py. The barcode normalises per
    # column across rows, so this decides what is being compared, and it depends on the
    # data: a field of 840 cells asks a per-object question, a single nucleus over time
    # asks a temporal one. "auto" resolves from the data and prints what it chose; it
    # can only reach "object" when an instance segmentation resolved, so a run without
    # one behaves exactly as before.
    row_axis: str = "auto"

    # Which of the three analyses to run -- see core/modes.py.
    #   xyt  : planar images over time (the original BARCODE behaviour)
    #   xyz  : for each timepoint, 2D metrics over z; flow disabled
    #   xyzt : volumes over time; true 3D metrics, meshing, 3D flow
    analysis_mode: str = "xyt"

    # Superseded by analysis_mode. Kept so Settings.yaml files written before modes
    # existed still load: enabled=True with no analysis_mode is migrated to xyzt.
    enabled: bool = False

    # Physical voxel size. Left at 0 these are read from the file's ImageJ metadata
    # (``spacing`` for z, ``XResolution`` for xy); set them to override.
    z_step_um: float = 0.0
    xy_step_um: float = 0.0

    # True axis order, one letter per data dimension, for a file whose header is wrong.
    # Acquisition software writing a time series into ImageJ's `channels` field is
    # common: the file then declares ZCYX for data that is really TZYX. Empty means
    # trust the file. BARCODE never infers this -- it only accepts being told.
    axes_override: str = ""

    # Segmentation masks. When a mask resolves it *replaces* intensity thresholding
    # as the binarization; intensity and autocorrelation still use raw voxels.
    segmentation_enabled: bool = False
    segmentation_root: str = ""
    segmentation_regex: str = r"(?P<stem>.+)"
    segmentation_template: str = "{stem}_SegMask.tif"
    # Masks generally carry no spacing metadata; 0 means "assume isotropic at xy".
    mask_spacing_um: float = 0.0

    # Z range to analyse. The full stack is often not the right range: slices beyond the
    # object are background, and including them drags every metric toward it.
    #
    # z_range_units decides how z_start/z_end are read, because "slice 46" is ambiguous
    # on anisotropic data -- the acquired stack and the isotropic grid a mask lives on
    # have very different slice counts (54 vs ~249 on 0.3/0.065 um data):
    #   "acquired"  indices into the stack as acquired      (default)
    #   "isotropic" indices into the isotropic grid, at the xy spacing
    #   "microns"   physical depth from the bottom of the stack
    # z_end = 0 always means "to the end". For the two index units, negatives count back
    # from the end as in Python slicing.
    z_start: float = 0
    z_end: float = 0
    z_range_units: str = "acquired"

    # Resample image + mask onto one isotropic grid.
    make_isotropic: bool = True

    # Crop the analysed volume to the mask's bounding box. OFF, and it should stay off:
    # every "fraction of volume" metric is a fraction of the analysed box, so cropping
    # per file gives each file its own denominator and an object shrinking becomes
    # indistinguishable from the box tightening around it. Use z_start/z_end to analyse
    # less than the full field. Kept only for reproducing older cropped runs.
    crop_to_mask: bool = False
    crop_padding_vox: int = 2

    # One summary card per analysed volume, beside the Summary CSV: projections, a
    # grouped metric table and the distributions behind the scalars.
    #
    # OFF by default. It is a per-volume *report*, and the work here is exploratory --
    # comparing FOVs, objects and timepoints -- which is what the barcode is for. A
    # document per volume does not help you find structure across a hundred of them.
    # Useful for inspecting one run closely; not the thing to generate by default.
    write_fingerprint: bool = False
    fingerprint_dpi: int = 110

    # Mesh EVERY object separately instead of only the largest, so sphericity, solidity,
    # concavity and curvature become per-object quantities. Off by default: it is one
    # iso2mesh call per object (~2.5 s), so a 839-cell field is ~35 minutes.
    object_mesh: bool = False
    object_mesh_maxrad: float = 0.1        # fraction of each object's own radius
    object_mesh_min_voxels: int = 64
    object_mesh_limit: int = 0     # 0 = every object; N = the first N, for iterating

    # Per-connected-component size distribution (count, SD, skewness, median). Off by
    # default so the barcode does not widen unless the spread of object sizes is
    # actually of interest. Volumetric modes only -- see AnalysisMode.supports_component_stats.
    enable_component_stats: bool = False

    # --- Phase 0 contract: fields the parallel work streams fill in. -------------
    # Timepoint selection, mirroring the z range above. "index" counts timepoints;
    # "seconds" uses the exposure time to convert. 0 as the end means "to the last".
    t_start: float = 0
    t_end: float = 0
    t_range_units: str = "index"

    # What counts as an object. BARCODE does not segment: when a mask supplies
    # instances, those instances are authoritative.
    #   "labels"       every distinct positive integer in the mask is one object, so
    #                  touching instances stay separate
    #   "connectivity" derive objects from the binary volume (correct for a binary mask
    #                  or an intensity threshold, wrong for supplied instances)
    #   "auto"         labels when the mask carries more than one positive value
    object_partition: str = "auto"

    # One row per object rather than one aggregate row. Needs object_partition to
    # resolve to "labels"; the Object ID column exists either way.
    per_object_rows: bool = False

    # Mask file format. "auto" dispatches on the file suffix through the reader
    # registry; name one explicitly only to override that.
    mask_format: str = "auto"

    # Segmentation. "labels" preserves the integer labels of a multi-object mask
    # instead of collapsing it to a boolean, so objects stay distinguishable. A
    # secondary mask lets a nucleus and a cell be supplied for the same image.
    segmentation_label_mode: str = "binary"
    segmentation_secondary_root: str = ""
    segmentation_secondary_template: str = ""

    # How the mesh family reports a field containing several objects. "largest" is the
    # current behaviour; "mean"/"total" mesh every object, averaging intensive metrics
    # (sphericity, curvature) and summing extensive ones (volume, surface area).
    mesh_aggregation: str = "largest"

    # Packing topology (analysis/volumetric/packing.py). Needs an integer label volume:
    # in a confluent field every cell touches its neighbours, so connectivity labelling
    # collapses the whole tissue to one object and there is no graph to build.
    enable_packing_topology: bool = False
    # Both of these count VOXELS, and the shared-face threshold weights a z-normal face
    # the same as an xy-normal one, so both assume an isotropic grid. Keep make_isotropic
    # on for this family: at 4.6x anisotropy one voxel of dilation bridges 0.3 um in z and
    # 0.065 um in xy, and whether a contact clears the threshold then depends on its
    # orientation rather than its area.
    packing_contact_dilation_vox: int = 1     # bridge a background gap between labels
    packing_min_contact_voxels: int = 5       # reject spurious one-voxel touches
    packing_exclude_border_objects: bool = True
    # Which faces make an object a "border" object: "xy" ignores the z faces, "all" uses
    # all six, "none" excludes nothing. Defaults to "xy", matching mesh_field: a packing
    # is a monolayer, so its segmentation spans the full acquired depth and every object
    # touches both z faces -- with "all" that excluded the entire field and every packing
    # metric came back NaN.
    packing_border_mode: str = "xy"

    # Extensive intensity metrics -- total, mean, SD, density. The intensity branch is
    # otherwise entirely intensive (histogram shape), so nothing in it scales with the
    # amount of material.
    enable_intensity_magnitude: bool = False

    # Minimum and maximum of the curvature field (analysis/volumetric/curvature.py).
    # Already computed there; these expose them as columns. <H> averages the two
    # principal curvatures together, so a saddle reads as flat -- the extremes do not.
    # Only meaningful alongside the mesh family.
    enable_curvature_range: bool = False

    # Broadest-slice depth and the field-of-view clipping flag
    # (analysis/volumetric/slice_profile.py). The only metrics in the branch that say
    # *where* in depth something is, rather than reducing the stack to one number.
    enable_slice_profile: bool = False

    # Intensity distribution inside the segmentation
    # (analysis/volumetric/mask_intensity.py): MFI, SD, CV, skewness, entropy,
    # normalized entropy and the fraction above twice the median. Needs a segmentation.
    # Distinct from intensity_use_mask below, which restricts the *existing* intensity
    # branch to in-mask voxels rather than adding per-object statistics.
    enable_mask_intensity: bool = False
    # Histogram bins for the in-mask entropy. Normalized entropy divides by log2 of
    # this, so runs are only comparable with each other at the same value.
    mask_intensity_bins: int = 64
    # Objects below this many voxels are skipped: entropy from a handful of voxels is
    # noise, and averaging it in biases the run.
    mask_intensity_min_voxels: int = 8

    # Record the analysed z/t ranges as columns. Flag digit 5 already marks *that* a
    # range was restricted; these say which, and make per-file ranges representable.
    record_range_columns: bool = False

    # Time-lapse assembly. Volumetric time series are often exported one file per
    # timepoint; grouping them restores the change metrics, which are NaN when each
    # volume is analysed on its own. The regex needs a 'series' group (files sharing it
    # belong together) and a numeric 'frame' group (ordering within the series).
    timelapse_enabled: bool = False
    timelapse_regex: str = r"^(?P<series>.+?)_(?P<frame>\d+)$"

    # Structural branch.
    threshold_offset: float = 0.1
    minimum_island_size: int = 1
    neighbor_island_fraction: float = 0.1
    frame_step: int = 10
    percentage_frames_evaluated: float = 0.05
    invert_binarization: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "VolumetricConfig":
        """Load, migrating pre-mode configs.

        Before modes existed the volumetric pipeline was gated by ``enabled``. A YAML
        carrying ``enabled: true`` but no ``analysis_mode`` therefore means xyzt; without
        this it would silently load as xyt and analyse a Z-stack as a time series, which
        is the exact failure modes were introduced to prevent.
        """
        data = dict(data)
        if "analysis_mode" not in data and data.get("enabled"):
            data["analysis_mode"] = "xyzt"
        return cls(**data)

    @property
    def mode(self):
        """The resolved AnalysisMode for this config."""
        from core.modes import get_mode

        return get_mode(self.analysis_mode)

    # Intensity branch. By default the histogram uses every voxel in the analysed
    # volume, matching the 2D branch; set intensity_use_mask to restrict it to voxels
    # inside the segmentation, which removes background from the distribution but
    # makes the numbers incomparable with unmasked runs.
    bin_size: int = 300
    noise_threshold: float = 5e-4
    intensity_use_mask: bool = False

    # Flow branch (analysis/volumetric/flow.py), a Lucas-Kanade solver vendored from
    # aicjanelia/OpticalFlow3D. Each analysed timepoint needs a *contiguous* window of
    # 6*flow_t_sigma+1 volumes centred on it, so unlike the other branches this one
    # cannot work from strided frames. Series shorter than that window are skipped, not
    # approximated.
    flow_xyz_sigma: float = 3.0
    flow_t_sigma: int = 1
    flow_w_sigma: float = 4.0
    # Reliability is the smallest eigenvalue of the structure tensor; where it is small
    # the solved velocity is noise. Voxels below this percentile of it are dropped from
    # every metric. 0 keeps all of them.
    flow_reliability_percentile: float = 50.0
    # Restrict the metrics to voxels inside the segmentation. On by default because the
    # analysed field is the whole acquired volume, most of which is background whose noise
    # velocities would otherwise dominate mean speed.
    # Flow is still solved on the whole volume so gradients at the boundary stay correct.
    flow_use_mask: bool = True
    # Block-average the volumes before solving. Reliability needs a 3x3 eigendecomposition
    # at every voxel, so large fields of view get expensive; 1 means no downsampling.
    flow_downsample: int = 1

    # Seconds between consecutive TIMEPOINTS. Every speed is a distance divided by this,
    # so it scales Speed and Speed Change directly.
    #
    # Set it. 0 means "fall back to the file's ImageJ 'finterval' tag, then to 1.0", and
    # that fallback is a trap: on stacks exported one timepoint per file, finterval often
    # describes the z acquisition or is simply left at 1, so it silently produces a number
    # that looks like seconds but is really per-frame. There is no way to tell the two
    # apart from the file alone, which is why this is an input rather than a guess.
    frame_interval_s: float = 0.0

    # Surface meshing of the segmented nucleus (analysis/volumetric/mesh.py), ported
    # from TCell-3D-Morphodynamics. Needs pyiso2mesh and a segmentation mask; the
    # defaults are the MATLAB pipeline's (config/defaults/nucleus_defaults.m).
    mesh_enabled: bool = False
    # Triangle-size bound for meshing (cgalsurf's radbound). This is the single biggest
    # control on mesh accuracy, and the value that matters is maxrad RELATIVE to the
    # object: on closed-form spheres at 0.108 um, maxrad 5 gives 0.997 of the exact
    # surface area on a 65-voxel radius and only 0.926 on a 16-voxel one.
    #
    # mesh_maxrad_units says how to read mesh_maxrad:
    #   "voxels"  as stored, in isotropic voxels -- the historical meaning, so the same
    #             number means a different physical size on every dataset
    #   "um"      microns, converted using the isotropic voxel size at meshing time,
    #             so one setting means the same physical thing across acquisitions
    # Neither adapts to object size; see analysis/volumetric/mesh.py:resolve_maxrad.
    mesh_maxrad: float = 5.0
    mesh_maxrad_units: str = "voxels"
    # Where the meshing isosurface sits inside a 0/1 mask. The boundary between the last
    # foreground voxel and the first background one is 0.5 -- what a binary mask means,
    # and the most accurate value measured on rounded objects. See
    # analysis/volumetric/mesh.py:DEFAULT_ISOVALUE for the one caveat (axis-aligned
    # boxes). Was 0.99 (from the MATLAB original), which pulled the surface onto the
    # outermost foreground voxel CENTRES and shrank every object by ~0.7 voxels on each
    # side: -9% of the volume of a 32-voxel sphere, -67% of a 4-voxel one. Set it to
    # 0.99 to reproduce mesh numbers from before this default changed, or MATLAB's.
    mesh_isovalue: float = 0.5
    mesh_area_frac: float = 0.2         # face-area filter -> decimation ratio
    mesh_smoothing_iterations: int = 10
    mesh_smoothing_alpha: float = 0.1
    mesh_smoothing_beta: float = 0.5
    # Reproduce MATLAB's quad-fan reading of v2s' 4-column face array when computing
    # the decimation ratio. Off by default because it is a bug, not a design choice.
    mesh_matlab_compat: bool = False
    # Principal curvatures over the mesh (analysis/volumetric/curvature.py) and the
    # invagination metrics built on them. Cheap relative to the meshing itself.
    mesh_curvature: bool = True
    # Face exclusions in the curvature branch, both OFF by default -- curvature is
    # measured over the whole surface unless you ask otherwise.
    #
    # curvature_exclude_caps drops faces in the lowest and highest z bin. Worth turning
    # on only when the segmentation is genuinely clipped by the ends of the stack, where
    # that surface is an artefact of the acquisition rather than the object; for an
    # object sitting entirely inside the imaged volume it discards real surface.
    #
    # curvature_outlier_limit drops faces whose mean curvature exceeds it in magnitude,
    # in 1/um -- a radius of curvature below 1/limit. MATLAB uses 2.0, i.e. 0.5 um, which
    # on a normally-sampled mesh means a degenerate triangle. 0 keeps every face.
    #
    # Setting both (True, 2.0) reproduces the MATLAB behaviour and the bit-for-bit parity
    # recorded in analysis/volumetric/curvature.py.
    curvature_exclude_caps: bool = False
    curvature_outlier_limit: float = 0.0
    # An iso2mesh bin/ directory to stage the CGAL executables from. Empty means let
    # pyiso2mesh find or download them.
    mesh_iso2mesh_bin: str = ""
    mesh_export_obj: bool = False


@dataclass
class AnalysisConfig(BaseConfig):
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    comparison: ComparisonConfig = field(default_factory=ComparisonConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

@dataclass
class BarcodeConfig(BaseConfig):
    channels: ChannelConfig = field(default_factory=ChannelConfig)
    image_binarization_parameters: BinarizationConfig = field(default_factory=BinarizationConfig)
    intensity_distribution_parameters: IntensityDistributionConfig = field(default_factory=IntensityDistributionConfig)
    modules: ModuleConfig = field(default_factory=ModuleConfig)
    optical_flow_parameters: OpticalFlowConfig = field(default_factory=OpticalFlowConfig)
    reader: ReaderConfig = field(default_factory=ReaderConfig)
    writer: WriterConfig = field(default_factory=WriterConfig)
    volumetric: VolumetricConfig = field(default_factory=VolumetricConfig)

    def save_to_yaml(self, filepath: str) -> None:
        """Save configuration to YAML file."""
        config_data = {}
        for field_name in self.__dataclass_fields__:
            subconfig = getattr(self, field_name)
            config_data[field_name] = subconfig.to_dict()

        with open(filepath, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    @classmethod
    def load_from_yaml(cls, filepath: str) -> "BarcodeConfig":
        """Load configuration from YAML file."""
        with open(filepath, "r") as f:
            config_data = yaml.safe_load(f)

        if not isinstance(config_data, dict):
            raise ValueError("Error loading YAML: expected a dictionary structure")

        try:
            return cls._load_from_yaml(config_data)
        except (KeyError, AssertionError) as e:
            print(f"Error loading YAML: {e}")
            pass

        print(f"Attempting to load legacy YAML format from {filepath}")
        try:
            return cls._load_from_legacy_yaml(config_data)
        except (KeyError, AssertionError) as e:
            print(f"Error loading legacy YAML: {e}")
            pass

        raise ValueError(f"Unknown YAML format in {filepath}")

    @classmethod
    def _load_from_yaml(cls, config_data: Dict[str, Any]) -> "BarcodeConfig":
        """Load configuration from YAML data."""

        kwargs = {}
        for subconfig_class_name, subconfig_data in config_data.items():

            assert (
                subconfig_class_name in cls.__dataclass_fields__
            ), f"Unknown configuration section: {subconfig_class_name}"

            field_info = cls.__dataclass_fields__[subconfig_class_name]
            subconfig_class = field_info.default_factory

            assert callable(
                subconfig_class
            ), f"Expected {subconfig_class_name} to be a callable class, got {subconfig_class}"
            assert issubclass(
                subconfig_class, BaseConfig
            ), f"Expected {subconfig_class_name} to be a subclass of BaseConfig"

            # Get the config class and create new instance from dict
            kwargs[subconfig_class_name] = subconfig_class.from_dict(subconfig_data)

        return cls(**kwargs)

    @classmethod
    def _load_from_legacy_yaml(cls, config_data: Dict[str, Any]) -> "BarcodeConfig":
        """Load configuration from legacy YAML format."""

        read: dict = config_data["reader"]
        write: dict = config_data["writer"]

        int_params: dict = config_data["intensity_distribution_parameters"]
        flow_params: dict = config_data["optical_flow_parameters"]
        bin_params: dict = config_data["image_binarization_parameters"]

        return BarcodeConfig(
            channels=ChannelConfig(
                parse_all_channels=read["channel_select"] == "All",
                selected_channel=(
                    read["channel_select"] if read["channel_select"] != "All" else 0
                ),
            ),
            modules=ModuleConfig(
                image_binarization=read["binarization"],
                optical_flow=read["flow"],
                intensity_distribution=read["intensity_distribution"],
            ),
            reader=ReaderConfig(
                accept_dim_images=read["accept_dim_images"],
                accept_dim_channels=read["accept_dim_channels"],
                exposure_time=flow_params["exposure_time"],
                um_pixel_ratio=flow_params["um_pixel_ratio"],
                verbose=read["verbose"],
            ),
            writer=WriterConfig(
                save_visualizations=write["save_visualizations"],
                save_rds=write["save_rds"],
                generate_barcode=write["generate_barcode"],
            ),
            image_binarization_parameters=BinarizationConfig(
                frame_step=bin_params["frame_step"],
                percentage_frame_evaluated=bin_params["percentage_frames_evaluated"],
                threshold_offset=bin_params["threshold_offset"],
            ),
            optical_flow_parameters=OpticalFlowConfig(
                downsample=flow_params["downsample"],
                frame_step=flow_params["frame_step"],
                percentage_frames_evaluated=flow_params["percentage_frames_evaluated"],
                win_size=flow_params["win_size"],
            ),
            intensity_distribution_parameters=IntensityDistributionConfig(
                bin_size=int_params["bin_size"],
                frame_step=int_params["frame_step"],
                noise_threshold=int_params["noise_threshold"],
                percentage_frames_evaluated=int_params["percentage_frames_evaluated"],            
            ),
        )

# === CONFIG GENERATION SETUP ===
# Define which configs should get GUI wrappers (edit this list as needed)
GUI_CONFIG_CLASSES = [
    InputConfig,
    ReaderConfig,
    WriterConfig,
    ChannelConfig,
    BinarizationConfig,
    OpticalFlowConfig,
    IntensityDistributionConfig,
    PreviewConfig,
    AggregationConfig,
    ComparisonConfig,
    ModuleConfig,
    VisualizationConfig,
    VolumetricConfig,
]


if __name__ == "__main__":
    import sys, os

    # Add the parent directory to sys.path to import core.config
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from gui.core import create_gui_configs

    # Generate GUI configs
    num_generated = create_gui_configs(GUI_CONFIG_CLASSES)

    print("\n📋 Usage:")
    print("  from gui import BarcodeConfigGUI")
    print("  from gui import BinarizationConfigGUI as BinGUI  # Optional short names")
    print("  gui_config = BarcodeConfigGUI(core_config)")
    print(
        "  threshold_slider = ttk.Scale(textvariable=gui_config.binarization.threshold_offset)"
    )