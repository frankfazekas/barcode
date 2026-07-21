from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from core import Metrics, Units
from core.config import BinarizationConfig
from core.modes import AnalysisMode, get_mode


@dataclass
class ResultsBase(ABC):
    """Base class for all analysis results."""

    @classmethod
    @abstractmethod
    def get_metrics(cls, **kwargs) -> List[Metrics]:
        """Get the metrics associated with this results class."""
        pass

    @classmethod
    @abstractmethod
    def get_units(cls, **kwargs) -> List[Units]:
        """Get the units associated with this results class."""
        pass

    @classmethod
    def get_headers(cls, **kwargs) -> List[str]:
        """Get headers for CSV output."""
        return [metric.value for metric in cls.get_metrics(**kwargs)]

    @abstractmethod
    def get_data(self, **kwargs) -> List[float]:
        """Return the results as a list for CSV writing."""
        pass

    def to_array(self, **kwargs) -> np.ndarray:
        """Convert results to a NumPy array for easier manipulation."""
        return np.array(self.get_data(**kwargs), dtype=float)
    
    def get_dict_data(self, **kwargs) -> np.ndarray:
        """Convert results to dictionary"""
        pass


@dataclass
class BinarizationResults(ResultsBase):
    """Results from binarization analysis."""

    connectivity: float = np.nan
    max_island_size: float = np.nan
    max_void_size: float = np.nan
    max_island_percent_change: float = np.nan
    max_void_percent_change: float = np.nan
    island_size_initial: float = np.nan
    island_size_initial2: float = np.nan
    island_anisotropy: float = np.nan
    mean_island_size: float = np.nan
    total_island_size: float = np.nan
    mean_island_separation: float = np.nan
    island_correlation_length: float = np.nan

    max_island_size_quantity: float = np.nan
    max_void_size_quantity: float = np.nan
    island_size_initial_quantity: float = np.nan
    island_size_initial2_quantity: float = np.nan
    mean_island_size_quantity: float = np.nan
    total_island_size_quantity: float = np.nan
    
    structural_correlation_flag: int = 0

    @classmethod
    def get_metrics(cls, mode: AnalysisMode = None) -> List[Metrics]:
        """Metric names for this mode.

        A 3D mode measures volumes, not areas, so those eight slots take different
        names; in xyz the two Change slots describe variation with depth. Connectivity,
        anisotropy, separation and correlation length are dimension-neutral and keep one
        name everywhere. ``mode=None`` gives the original 2D names.
        """
        volumetric = mode is not None and mode.is_volumetric
        depth = mode is not None and mode.progression == "depth"
        return [
            Metrics.CONNECTIVITY,
            Metrics.ISLAND_MAX_VOLUME if volumetric else Metrics.ISLAND_MAX_AREA,
            Metrics.VOID_MAX_VOLUME if volumetric else Metrics.VOID_MAX_AREA,
            Metrics.MAX_ISLAND_VOLUME_CHANGE if volumetric
            else (Metrics.MAX_ISLAND_AREA_CHANGE_Z if depth else Metrics.MAX_ISLAND_AREA_CHANGE),
            Metrics.MAX_VOID_VOLUME_CHANGE if volumetric
            else (Metrics.MAX_VOID_AREA_CHANGE_Z if depth else Metrics.MAX_VOID_AREA_CHANGE),
            Metrics.ISLAND_MAX_VOLUME_INITIAL if volumetric else Metrics.ISLAND_MAX_AREA_INITIAL,
            Metrics.ISLAND_MAX_VOLUME_INITIAL2 if volumetric else Metrics.ISLAND_MAX_AREA_INITIAL2,
            Metrics.ISLAND_ANISOTROPY,
            Metrics.ISLAND_MEAN_VOLUME if volumetric else Metrics.ISLAND_MEAN_AREA,
            Metrics.ISLAND_TOTAL_VOLUME if volumetric else Metrics.ISLAND_TOTAL_AREA,
            Metrics.ISLAND_DISTANCE,
            Metrics.ISLAND_CORRELATION,
        ]
    
    @classmethod
    def get_physical_metrics(cls, mode: AnalysisMode = None) -> List[Metrics]:
        volumetric = mode is not None and mode.is_volumetric
        depth = mode is not None and mode.progression == "depth"
        return [
            Metrics.CONNECTIVITY,
            Metrics.ISLAND_MAX_VOLUME_QUANTITY if volumetric else Metrics.ISLAND_MAX_AREA_QUANTITY,
            Metrics.VOID_MAX_VOLUME_QUANTITY if volumetric else Metrics.VOID_MAX_AREA_QUANTITY,
            Metrics.MAX_ISLAND_VOLUME_CHANGE if volumetric
            else (Metrics.MAX_ISLAND_AREA_CHANGE_Z if depth else Metrics.MAX_ISLAND_AREA_CHANGE),
            Metrics.MAX_VOID_VOLUME_CHANGE if volumetric
            else (Metrics.MAX_VOID_AREA_CHANGE_Z if depth else Metrics.MAX_VOID_AREA_CHANGE),
            Metrics.ISLAND_MAX_VOLUME_INITIAL_QUANTITY if volumetric else Metrics.ISLAND_MAX_AREA_INITIAL_QUANTITY,
            Metrics.ISLAND_MAX_VOLUME_INITIAL2_QUANTITY if volumetric else Metrics.ISLAND_MAX_AREA_INITIAL2_QUANTITY,
            Metrics.ISLAND_ANISOTROPY,
            Metrics.ISLAND_MEAN_VOLUME_QUANTITY if volumetric else Metrics.ISLAND_MEAN_AREA_QUANTITY,
            Metrics.ISLAND_TOTAL_VOLUME_QUANTITY if volumetric else Metrics.ISLAND_TOTAL_AREA_QUANTITY,
            Metrics.ISLAND_DISTANCE,
            Metrics.ISLAND_CORRELATION,
        ]

    @classmethod
    def get_units(cls, mode: AnalysisMode = None) -> List[Units]:
        return [
            Units.FRACTION_FRAMES,
            Units.FRACTION_FOV,
            Units.FRACTION_FOV,
            Units.RATIO_TO_INITIAL,
            Units.RATIO_TO_INITIAL,
            Units.FRACTION_FOV,
            Units.FRACTION_FOV,
            Units.NONE,
            Units.FRACTION_FOV,
            Units.FRACTION_FOV,
            Units.LENGTH,
            Units.LENGTH,
        ]
    
    @classmethod
    def get_physical_units(cls, mode: AnalysisMode = None) -> List[Units]:
        size = Units.VOLUME if (mode is not None and mode.is_volumetric) else Units.AREA
        return [
            Units.FRACTION_FRAMES,
            size,
            size,
            Units.RATIO_TO_INITIAL,
            Units.RATIO_TO_INITIAL,
            size,
            size,
            Units.NONE,
            size,
            size,
            Units.LENGTH,
            Units.LENGTH,
        ]

    def get_data(self) -> List[float]:
        return [
            self.connectivity,
            self.max_island_size,
            self.max_void_size,
            self.max_island_percent_change,
            self.max_void_percent_change,
            self.island_size_initial,
            self.island_size_initial2,
            self.island_anisotropy,
            self.mean_island_size,
            self.total_island_size,
            self.mean_island_separation,
            self.island_correlation_length,
        ]
    
    def get_physical_data(self) -> List[float]:
        return [
            self.connectivity,
            self.max_island_size_quantity,
            self.max_void_size_quantity,
            self.max_island_percent_change,
            self.max_void_percent_change,
            self.island_size_initial_quantity,
            self.island_size_initial2_quantity,
            self.island_anisotropy,
            self.mean_island_size_quantity,
            self.total_island_size_quantity,
            self.mean_island_separation,
            self.island_correlation_length,
        ]
    
    def get_dict_data(self) -> dict:
        return dict(zip(self.get_metrics(), self.get_data()))
    
    def get_physical_dict_data(self) -> dict:
        return dict(zip(self.get_physical_metrics(), self.get_physical_data()))


@dataclass
class FlowResults(ResultsBase):
    """Results from optical flow analysis."""

    mean_speed: float = np.nan
    delta_speed: float = np.nan
    mean_theta: float = np.nan
    mean_sigma_theta: float = np.nan
    velocity_correlation_length: float = np.nan
    divergence: float = np.nan
    curl: float = np.nan
    velocity_correlation_flag: int = 0

    @classmethod
    def get_metrics(cls) -> List[Metrics]:
        return [
            Metrics.SPEED,
            Metrics.DELTA_SPEED,
            Metrics.MEAN_THETA,
            Metrics.MEAN_SIGMA_THETA,
            Metrics.VELOCITY_CORRELATION,
            Metrics.DIVERGENCE,
            Metrics.CURL,
        ]

    @classmethod
    def get_units(cls, mode: AnalysisMode = None) -> List[Units]:
        """Units for this branch.

        Divergence and curl are the only mode-dependent entries. In 3D they are spatial
        derivatives of a velocity field and carry 1/s; the 2D branch differentiates a
        cumulative *unit-vector* field instead, giving 1/um for a different quantity, so
        it keeps the blank label it has always had. ``mode=None`` is the legacy 2D layout
        and must stay byte-identical.
        """
        rate = Units.RATE if (mode is not None and mode.is_volumetric) else Units.NONE
        return [
            Units.SPEED,
            Units.SPEED,
            Units.DIRECTION,
            Units.DIRECTION,
            Units.LENGTH,
            rate,
            rate,
        ]

    def get_data(self) -> List[float]:
        return [
            self.mean_speed,
            self.delta_speed,
            self.mean_theta,
            self.mean_sigma_theta,
            self.velocity_correlation_length,
            self.divergence,
            self.curl,
        ]
    
    def get_dict_data(self) -> dict:
        return dict(zip(self.get_metrics(), self.get_data()))

    def is_populated(self) -> bool:
        """True if the flow branch actually produced a value.

        A static z-stack (one timepoint, or fewer than the flow window needs) cannot be
        analysed for motion, so every field here is NaN. The writer and the barcode use
        this to drop the seven flow columns rather than paint a row of meaningless black
        cells -- see the ``include_flow`` handling in ``_resolve``.
        """
        return bool(np.any(np.isfinite(np.array(self.get_data(), dtype=float))))


@dataclass
class IntensityResults(ResultsBase):
    """Results from intensity distribution analysis."""

    max_kurtosis: float = np.nan
    max_median_skew: float = np.nan
    max_mode_skew: float = np.nan
    kurtosis_diff: float = np.nan
    median_skew_diff: float = np.nan
    mode_skew_diff: float = np.nan
    saturation_flag: int = 0

    @classmethod
    def get_metrics(cls, mode: AnalysisMode = None) -> List[Metrics]:
        depth = mode is not None and mode.progression == "depth"
        return [
            Metrics.MAX_KURTOSIS,
            Metrics.MAX_MEDIAN_SKEW,
            Metrics.MAX_MODE_SKEW,
            Metrics.KURTOSIS_DIFF_Z if depth else Metrics.KURTOSIS_DIFF,
            Metrics.MEDIAN_SKEW_DIFF_Z if depth else Metrics.MEDIAN_SKEW_DIFF,
            Metrics.MODE_SKEW_DIFF_Z if depth else Metrics.MODE_SKEW_DIFF,
        ]

    @classmethod
    def get_units(cls, mode: AnalysisMode = None) -> List[Units]:
        return [Units.NONE] * 6

    def get_data(self) -> List[float]:
        return [
            self.max_kurtosis,
            self.max_median_skew,
            self.max_mode_skew,
            self.kurtosis_diff,
            self.median_skew_diff,
            self.mode_skew_diff,
        ]
    
    def get_dict_data(self) -> dict:
        return dict(zip(self.get_metrics(), self.get_data()))


@dataclass
class MeshResults(ResultsBase):
    """Results from surface meshing and curvature of a segmented object.

    Only emitted when mesh data exists (see ``ChannelResults.get_metrics``), so adding
    this family leaves the 2D schema byte-identical.
    """

    mesh_volume: float = np.nan
    surface_area: float = np.nan
    sphericity: float = np.nan
    equivalent_radius: float = np.nan
    height: float = np.nan
    aspect_ratio: float = np.nan
    volume_ratio: float = np.nan
    solidity: float = np.nan

    mean_curvature: float = np.nan
    invagination_ratio: float = np.nan
    concave_ratio: float = np.nan

    @classmethod
    def get_metrics(cls, mode: AnalysisMode = None) -> List[Metrics]:
        return [
            Metrics.MESH_VOLUME,
            Metrics.MESH_SURFACE_AREA,
            Metrics.MESH_SPHERICITY,
            Metrics.MESH_EQUIVALENT_RADIUS,
            Metrics.MESH_HEIGHT,
            Metrics.MESH_ASPECT_RATIO,
            Metrics.MESH_VOLUME_RATIO,
            Metrics.MESH_SOLIDITY,
            Metrics.MESH_CONCAVITY,
            Metrics.CURVATURE_MEAN,
            Metrics.CURVATURE_INVAGINATION,
            Metrics.CURVATURE_CONCAVE,
        ]

    @classmethod
    def get_units(cls, mode: AnalysisMode = None) -> List[Units]:
        return [
            Units.VOLUME,
            Units.AREA,
            Units.NONE,
            Units.LENGTH,
            Units.LENGTH,
            Units.NONE,
            Units.NONE,
            Units.NONE,
            Units.NONE,
            Units.CURVATURE,
            Units.NONE,
            Units.NONE,
        ]

    # The single source of truth for the CSV column order of this family: get_data
    # writes it and from_values reads it back. They used to be two hand-kept lists of
    # positional indices in different files, which silently mis-assigns every column
    # after an insertion. None marks a derived column that is written but not read back.
    _CSV_FIELDS = (
        "mesh_volume",
        "surface_area",
        "sphericity",
        "equivalent_radius",
        "height",
        "aspect_ratio",
        "volume_ratio",
        "solidity",
        None,                             # Concavity = 1 - Solidity
        "mean_curvature",
        "invagination_ratio",
        "concave_ratio",
    )

    def get_data(self) -> List[float]:
        return [
            getattr(self, name) if name else 1.0 - self.solidity
            for name in self._CSV_FIELDS
        ]

    @classmethod
    def from_values(cls, values) -> "MeshResults":
        """Rebuild from one CSV row's mesh block, in ``_CSV_FIELDS`` order."""
        return cls(**{
            name: float(value)
            for name, value in zip(cls._CSV_FIELDS, values)
            if name
        })

    def get_dict_data(self, mode: AnalysisMode = None) -> dict:
        return dict(zip(self.get_metrics(mode), self.get_data()))

    def is_populated(self) -> bool:
        """True if any value was actually measured."""
        return bool(np.any(np.isfinite(np.array(self.get_data(), dtype=float))))



@dataclass
class ComponentResults(ResultsBase):
    """Spread of the per-connected-component size distribution.

    The binarization family already reports the largest, mean and total object size.
    What it cannot say is whether those objects are uniform or wildly unequal -- one
    dominant object plus debris gives the same mean as a handful of even ones. These
    describe the *shape* of that distribution.

    Off by default: the barcode is already wide, and a metric nobody reads is worse than
    absent because it still takes a column and still gets normalised.
    """

    count: float = np.nan
    size_sd: float = np.nan
    size_skew: float = np.nan
    size_median: float = np.nan

    @classmethod
    def get_metrics(cls, mode: AnalysisMode = None) -> List[Metrics]:
        volumetric = mode is not None and mode.is_volumetric
        return [
            Metrics.ISLAND_COUNT,
            Metrics.ISLAND_VOLUME_SD if volumetric else Metrics.ISLAND_AREA_SD,
            Metrics.ISLAND_VOLUME_SKEW if volumetric else Metrics.ISLAND_AREA_SKEW,
            Metrics.ISLAND_VOLUME_MEDIAN if volumetric else Metrics.ISLAND_AREA_MEDIAN,
        ]

    @classmethod
    def get_units(cls, mode: AnalysisMode = None) -> List[Units]:
        # Sizes are reported as a fraction of the analysed field, like the binarization
        # family, so SD and median are dimensionless; skewness always is.
        return [Units.NONE, Units.NONE, Units.NONE, Units.NONE]

    def get_data(self) -> List[float]:
        return [self.count, self.size_sd, self.size_skew, self.size_median]

    def get_dict_data(self, mode: AnalysisMode = None) -> dict:
        return dict(zip(self.get_metrics(mode), self.get_data()))

    def is_populated(self) -> bool:
        return bool(np.any(np.isfinite(np.array(self.get_data(), dtype=float))))


@dataclass
class CurvatureRangeResults(ResultsBase):
    """The extremes of the curvature field, which <H> averages away.

    ``curvature.py`` has always computed these -- they are printed by ``describe()``
    -- but they never reached the CSV. A saddle point is sharply curved in both
    principal directions and averages to nearly zero in <H>, so a surface can be
    highly structured and still report a flat mean. These say how structured.

    Off by default: they are only meaningful alongside the mesh family, and the barcode
    is already wide.
    """

    min_curvature: float = np.nan
    max_curvature: float = np.nan

    @classmethod
    def get_metrics(cls, mode: AnalysisMode = None) -> List[Metrics]:
        return [Metrics.CURVATURE_MIN, Metrics.CURVATURE_MAX]

    @classmethod
    def get_units(cls, mode: AnalysisMode = None) -> List[Units]:
        return [Units.CURVATURE, Units.CURVATURE]

    def get_data(self) -> List[float]:
        return [self.min_curvature, self.max_curvature]

    def get_dict_data(self, mode: AnalysisMode = None) -> dict:
        return dict(zip(self.get_metrics(mode), self.get_data()))

    def is_populated(self) -> bool:
        return bool(np.any(np.isfinite(np.array(self.get_data(), dtype=float))))


@dataclass
class SliceProfileResults(ResultsBase):
    """Where through the stack the object is widest.

    Everything else in the volumetric branch reduces a stack to one number per
    timepoint and so cannot say *where* in depth anything happened. For a stack through
    a curved surface or a rounded object the broadest slice locates the equator, and it
    moves when the object flattens, tilts, or drifts through the focal range.

    The depth is measured from the first *analysed* slice, so it is unaffected by a
    z-range restriction -- but that also means it is not an absolute stage position.
    """

    broadest_index: float = np.nan
    broadest_depth: float = np.nan
    broadest_area: float = np.nan

    @classmethod
    def get_metrics(cls, mode: AnalysisMode = None) -> List[Metrics]:
        return [
            Metrics.BROADEST_SLICE_INDEX,
            Metrics.BROADEST_SLICE_DEPTH,
            Metrics.BROADEST_SLICE_AREA,
        ]

    @classmethod
    def get_units(cls, mode: AnalysisMode = None) -> List[Units]:
        return [Units.SLICE_INDEX, Units.LENGTH, Units.FRACTION_FOV]

    def get_data(self) -> List[float]:
        return [self.broadest_index, self.broadest_depth, self.broadest_area]

    def get_dict_data(self, mode: AnalysisMode = None) -> dict:
        return dict(zip(self.get_metrics(mode), self.get_data()))

    def is_populated(self) -> bool:
        return bool(np.any(np.isfinite(np.array(self.get_data(), dtype=float))))


@dataclass
class MaskIntensityResults(ResultsBase):
    """How signal is distributed *inside* the segmented objects.

    The intensity branch describes whatever voxels it is handed, which normally means
    the background peak dominates. These describe the inside of each object and then
    average over objects, which is the clustering readout: a uniformly-filled nucleus
    and one with bright foci have the same mean and very different CV and entropy.

    Only ``entropy`` is computed on a per-object [0, 1] rescaling, which it needs so that
    every object is binned over the same range. The rest are computed on raw voxels: CV
    and skewness are already scale-invariant, and the bright fraction is undefined on
    rescaled values for punctate objects. See ``analysis/volumetric/mask_intensity.py``,
    which sets out where this departs from the source MATLAB and why.
    """

    mfi: float = np.nan
    sd: float = np.nan
    cv: float = np.nan
    skewness: float = np.nan
    entropy: float = np.nan
    entropy_normalized: float = np.nan
    bright_fraction: float = np.nan

    @classmethod
    def get_metrics(cls, mode: AnalysisMode = None) -> List[Metrics]:
        return [
            Metrics.MASK_INTENSITY_MFI,
            Metrics.MASK_INTENSITY_SD,
            Metrics.MASK_INTENSITY_CV,
            Metrics.MASK_INTENSITY_SKEW,
            Metrics.MASK_INTENSITY_ENTROPY,
            Metrics.MASK_INTENSITY_ENTROPY_NORM,
            Metrics.MASK_INTENSITY_BRIGHT_FRACTION,
        ]

    @classmethod
    def get_units(cls, mode: AnalysisMode = None) -> List[Units]:
        # MFI and SD are raw voxel statistics and carry detector units; CV, skewness,
        # entropy and the two fractions are dimensionless by construction.
        return [Units.INTENSITY, Units.INTENSITY, Units.NONE, Units.NONE,
                Units.NONE, Units.NONE, Units.NONE]

    def get_data(self) -> List[float]:
        return [self.mfi, self.sd, self.cv, self.skewness,
                self.entropy, self.entropy_normalized, self.bright_fraction]

    def get_dict_data(self, mode: AnalysisMode = None) -> dict:
        return dict(zip(self.get_metrics(mode), self.get_data()))

    def is_populated(self) -> bool:
        return bool(np.any(np.isfinite(np.array(self.get_data(), dtype=float))))


@dataclass
class IntensityMagnitudeResults(ResultsBase):
    """Extensive intensity quantities -- how much signal, not what shape.

    Every other intensity metric is intensive: kurtosis and skewness describe the shape
    of the histogram and are unchanged if the object doubles in size. Nothing in the
    branch scaled with the amount of material until this family existed, which is why
    "is the intensity branch volume based?" had no answer.

    ``total`` is a raw sum over the analysed region and therefore includes background;
    on a cropped stack that can dominate, so it is most meaningful with
    ``intensity_use_mask`` on. It is also meaningless if the detector clipped -- check
    the saturation flag (digit 2) before reading it.

    Populated by stream A; see docs/parallel_work_plan.md.
    """

    total: float = np.nan
    mean: float = np.nan
    sd: float = np.nan
    density: float = np.nan

    @classmethod
    def get_metrics(cls, mode: AnalysisMode = None) -> List[Metrics]:
        volumetric = mode is not None and mode.is_volumetric
        return [
            Metrics.INTENSITY_TOTAL,
            Metrics.INTENSITY_MEAN,
            Metrics.INTENSITY_SD,
            Metrics.INTENSITY_DENSITY_VOLUME if volumetric
            else Metrics.INTENSITY_DENSITY_AREA,
        ]

    @classmethod
    def get_units(cls, mode: AnalysisMode = None) -> List[Units]:
        volumetric = mode is not None and mode.is_volumetric
        return [
            Units.INTENSITY,
            Units.INTENSITY,
            Units.INTENSITY,
            Units.INTENSITY_PER_VOLUME if volumetric else Units.INTENSITY_PER_AREA,
        ]

    def get_data(self) -> List[float]:
        return [self.total, self.mean, self.sd, self.density]

    def get_dict_data(self, mode: AnalysisMode = None) -> dict:
        return dict(zip(self.get_metrics(mode), self.get_data()))

    def is_populated(self) -> bool:
        return bool(np.any(np.isfinite(np.array(self.get_data(), dtype=float))))


@dataclass
class RangeResults(ResultsBase):
    """Which slices and timepoints this row's numbers were computed over.

    Flag digit 5 marks *that* the analysis covered part of the data; this says which
    part. It also makes per-file ranges representable, which the global z_start/z_end
    settings cannot express -- and it means a CSV separated from its Settings.yaml still
    describes itself.

    Indices are into the acquired data, before any isotropic resampling.
    Populated by stream A; see docs/parallel_work_plan.md.
    """

    z_start: float = np.nan
    z_end: float = np.nan
    t_start: float = np.nan
    t_end: float = np.nan

    @classmethod
    def get_metrics(cls, mode: AnalysisMode = None) -> List[Metrics]:
        return [Metrics.RANGE_Z_START, Metrics.RANGE_Z_END,
                Metrics.RANGE_T_START, Metrics.RANGE_T_END]

    @classmethod
    def get_units(cls, mode: AnalysisMode = None) -> List[Units]:
        return [Units.SLICE_INDEX] * 4

    def get_data(self) -> List[float]:
        return [self.z_start, self.z_end, self.t_start, self.t_end]

    def get_dict_data(self, mode: AnalysisMode = None) -> dict:
        return dict(zip(self.get_metrics(mode), self.get_data()))

    def is_populated(self) -> bool:
        return bool(np.any(np.isfinite(np.array(self.get_data(), dtype=float))))


@dataclass
class PackingResults(ResultsBase):
    """How objects are arranged relative to each other -- who touches whom.

    BARCODE describes objects individually (volume, sphericity, curvature) and describes
    their spacing with one scalar. Nothing described the *topology* of a packing. In a
    space-filling monolayer sizes and separations are near-uniform and what changes is
    the neighbour-number distribution, which is the standard epithelial readout.

    Computable only from an integer label volume: in a confluent field every cell touches
    its neighbours, so connectivity labelling yields a single object and no graph.
    """

    contact_number_mean: float = np.nan
    contact_number_sd: float = np.nan
    hexagonal_fraction: float = np.nan

    @classmethod
    def get_metrics(cls, mode: AnalysisMode = None) -> List[Metrics]:
        return [Metrics.CONTACT_NUMBER_MEAN, Metrics.CONTACT_NUMBER_SD,
                Metrics.HEXAGONAL_FRACTION]

    @classmethod
    def get_units(cls, mode: AnalysisMode = None) -> List[Units]:
        return [Units.NONE, Units.NONE, Units.NONE]

    def get_data(self) -> List[float]:
        return [self.contact_number_mean, self.contact_number_sd,
                self.hexagonal_fraction]

    def get_dict_data(self, mode: AnalysisMode = None) -> dict:
        return dict(zip(self.get_metrics(mode), self.get_data()))

    def is_populated(self) -> bool:
        return bool(np.any(np.isfinite(np.array(self.get_data(), dtype=float))))


@dataclass(frozen=True)
class OptionalFamily:
    """A metric family that only some modes or runs produce.

    Replaces the positional include_* flags that _resolve used to juggle. With six
    families those flags were about to be rewritten by two work streams at once, and a
    registry means adding a seventh touches one tuple rather than eight signatures.

    Three separate questions, which were previously conflated into one predicate:

    * ``allowed``   -- *can* this mode produce the family at all? A mode that cannot
      must not advertise the columns, however the switch is set. This is what keeps
      ``--list-metrics`` honest: it used to report per-object columns for xyz while the
      runner printed "volumetric-only; not added for xyz" and computed nothing.
    * ``supported`` -- is it on by DEFAULT for this mode (mesh is, for xyzt).
    * the per-call switch -- did this run actually produce it, which is how the writer
      avoids emitting a family's columns full of NaN.

    ``allowed=None`` means every mode, which is the right default for families that are
    pure post-processing of values the branches already computed.
    """

    switch: str          # keyword argument name, e.g. "include_mesh"
    attribute: str       # ChannelResults attribute holding the values
    results_cls: type
    supported: object    # callable(mode) -> bool: on by default for this mode
    allowed: object = None   # callable(mode) -> bool, or None for "any mode"

    def is_allowed(self, mode) -> bool:
        """Whether ``mode`` can produce this family at all.

        An unknown mode (``None``, the legacy 2D layout) is permitted everything: there
        is no mode object to ask, and refusing would change the pre-mode behaviour.
        """
        if self.allowed is None or mode is None:
            return True
        return bool(self.allowed(mode))


# Only ``analysis/volumetric/run.py`` (the xyzt path) computes these; the slice-wise xyz
# path analyses planes with the 2D branches and produces none of them. `_volumetric`
# therefore mirrors what the runner actually does, so the schema cannot promise columns
# that no code fills.
def _volumetric(mode) -> bool:
    return bool(mode is not None and mode.spatial_dims == 3)


OPTIONAL_FAMILIES = (
    OptionalFamily("include_mesh", "mesh", MeshResults,
                   lambda mode: bool(mode is not None and mode.supports_mesh),
                   allowed=lambda mode: bool(mode is not None and mode.supports_mesh)),
    OptionalFamily("include_components", "components", ComponentResults,
                   lambda mode: False, allowed=_volumetric),
    # Extensive intensity and the range provenance columns are post-processing of values
    # every branch already has, so they are not restricted to a mode.
    OptionalFamily("include_intensity_magnitude", "intensity_magnitude",
                   IntensityMagnitudeResults, lambda mode: False),
    OptionalFamily("include_ranges", "ranges", RangeResults,
                   lambda mode: False),
    OptionalFamily("include_packing", "packing", PackingResults,
                   lambda mode: False, allowed=_volumetric),
    OptionalFamily("include_curvature_range", "curvature_range", CurvatureRangeResults,
                   lambda mode: False, allowed=_volumetric),
    OptionalFamily("include_slice_profile", "slice_profile", SliceProfileResults,
                   lambda mode: False, allowed=_volumetric),
    OptionalFamily("include_mask_intensity", "mask_intensity", MaskIntensityResults,
                   lambda mode: False, allowed=_volumetric),
)


def flow_is_populated(results, mode) -> bool:
    """Whether the optical-flow branch carries data across ``results``.

    Only ever answers False in a **volumetric** mode: the 2D modes always emit flow, so
    the reference schema stays byte-identical whatever a given 2D run's flow happens to
    be. In a volumetric mode a static z-stack -- one timepoint, or too few for the flow
    window -- produces an all-NaN flow branch, and this returns False so the seven flow
    columns are dropped rather than painted black.
    """
    if mode is not None and not isinstance(mode, AnalysisMode):
        mode = get_mode(mode)
    if not _volumetric(mode):
        return True
    return any(
        getattr(r, "flow", None) is not None and r.flow.is_populated()
        for r in results
    )


def _resolve(mode, **switches):
    """Normalise the mode and the optional-family switches.

    ``mode`` may be an AnalysisMode, a key string, or None for the legacy 2D layout.
    Each family defaults to what the mode supports; passing the switch explicitly forces
    it either way. Returns ``(mode, with_flow, enabled)`` where ``enabled`` maps each
    family's switch name to a bool.

    ``include_flow`` is accepted alongside the family switches but is not one of them: it
    can only *suppress* the flow branch a mode already supports (for a static z-stack),
    never force it on where the mode has no time axis.
    """
    include_flow = switches.pop("include_flow", None)

    known = {family.switch for family in OPTIONAL_FAMILIES}
    unknown = set(switches) - known
    if unknown:
        raise TypeError(
            f"Unknown optional-family switch(es) {sorted(unknown)}; "
            f"expected any of {sorted(known)}."
        )

    if mode is not None and not isinstance(mode, AnalysisMode):
        mode = get_mode(mode)
    base_flow = True if mode is None else mode.supports_flow
    # include_flow can only suppress flow in a VOLUMETRIC mode. The 2D modes always emit
    # their flow columns whatever a caller passes, so the published reference schema stays
    # byte-identical -- the suppression exists only for static z-stacks (xyzt).
    if include_flow is None or not _volumetric(mode):
        with_flow = base_flow
    else:
        with_flow = base_flow and bool(include_flow)

    enabled = {}
    for family in OPTIONAL_FAMILIES:
        value = switches.get(family.switch)
        wanted = family.supported(mode) if value is None else bool(value)
        # A mode that cannot produce the family must not advertise its columns, however
        # the switch was set. Silently off rather than raising: the CLI already prints a
        # note for the modes it declines, and one over-broad flag in a batch should not
        # abort the run.
        enabled[family.switch] = wanted and family.is_allowed(mode)
    return mode, with_flow, enabled


def _family_parts(mode, enabled, getter):
    """Concatenate ``getter(family)`` for every enabled family, in registry order."""
    parts = []
    for family in OPTIONAL_FAMILIES:
        if enabled[family.switch]:
            parts.extend(getter(family))
    return parts


@dataclass
class ChannelResults(ResultsBase):
    """Complete analysis results for a single channel."""

    filepath: str
    channel: int
    total_flags: str = "0"
    dim_channel_flag: int = 0  # 0=normal, 1=dim channel
    # 1 when the analysis covered only part of the acquired z stack. Without this a CSV
    # separated from its Settings.yaml gives no hint that its numbers describe a subset
    # of the data -- the same way a stale CSV once gave no hint that its correlation
    # lengths predated a bug fix.
    z_range_flag: int = 0
    # 1 when foreground reaches an edge of the analysed field, so the object continues
    # outside it. Deliberately a separate digit from 5: digit 5 says the *user* narrowed
    # the analysis, this says the *data* is cut off. Every size, shape and curvature
    # metric describes a truncated object when this fires, which is not recoverable
    # from the numbers themselves.
    fov_clip_flag: int = 0
    # 1 when a mesh came back with an open boundary. `mesh_has_holes` has always been
    # computed, but it only ever reached `MeshGeometry.describe()`, which nothing in the
    # pipeline prints -- so an open surface, which makes mesh volume, sphericity and the
    # sign of every curvature unreliable, was completely invisible in a GUI or batch run.
    # A flag rather than a column: it describes whether the mesh row can be trusted, not
    # a property of the object, and it costs no schema change.
    mesh_open_surface_flag: int = 0
    # The AnalysisMode this row was read back from, when it came from a CSV. Aggregation
    # and comparison re-write results they did not compute, and they had no way to know
    # which mode produced them, so they fell back to the 2D layout: a volumetric
    # aggregate came out headed "Maximum Island Area" (um^2) over values that are um^3
    # volumes. The reader already identifies the layout to parse the row; recording it
    # costs nothing and lets the writers ask. None means "not read from a CSV", which is
    # every freshly computed result -- those callers pass the mode explicitly.
    source_mode: Optional[AnalysisMode] = None

    binarization: BinarizationResults = field(default_factory=BinarizationResults)
    intensity: IntensityResults = field(default_factory=IntensityResults)
    flow: FlowResults = field(default_factory=FlowResults)
    mesh: MeshResults = field(default_factory=MeshResults)
    components: ComponentResults = field(default_factory=ComponentResults)
    intensity_magnitude: IntensityMagnitudeResults = field(
        default_factory=IntensityMagnitudeResults)
    ranges: RangeResults = field(default_factory=RangeResults)
    packing: PackingResults = field(default_factory=PackingResults)
    curvature_range: CurvatureRangeResults = field(default_factory=CurvatureRangeResults)
    slice_profile: SliceProfileResults = field(default_factory=SliceProfileResults)
    mask_intensity: MaskIntensityResults = field(default_factory=MaskIntensityResults)

    @classmethod
    def _get_base_headers(cls) -> List[str]:
        return ["Filepath", "Channel", "Flags"]

    @classmethod
    def get_metrics(cls, just_metrics: bool = False, mode=None, **switches) -> List[Metrics]:
        mode, with_flow, enabled = _resolve(mode, **switches)
        return (
            ([Metrics.FILEPATH, Metrics.CHANNEL, Metrics.FLAGS] if not just_metrics else [])
            + BinarizationResults.get_metrics(mode)
            + IntensityResults.get_metrics(mode)
            + (FlowResults.get_metrics() if with_flow else [])
            + _family_parts(mode, enabled, lambda f: f.results_cls.get_metrics(mode))
        )

    @classmethod
    def get_physical_metrics(cls, just_metrics: bool = False, mode=None,
                             **switches) -> List[Metrics]:
        mode, with_flow, enabled = _resolve(mode, **switches)
        return (
            ([Metrics.FILEPATH, Metrics.CHANNEL, Metrics.FLAGS] if not just_metrics else [])
            + BinarizationResults.get_physical_metrics(mode)
            + IntensityResults.get_metrics(mode)
            + (FlowResults.get_metrics() if with_flow else [])
            + _family_parts(mode, enabled, lambda f: f.results_cls.get_metrics(mode))
        )

    @classmethod
    def get_physical_headers(cls, just_metrics: bool = False, mode=None,
                             **switches) -> List[str]:
        """Get headers for CSV output."""
        return [m.value for m in cls.get_physical_metrics(
            just_metrics, mode, **switches)]

    @classmethod
    def get_units(cls, just_metrics: bool = False, mode=None, **switches) -> List[Units]:
        mode, with_flow, enabled = _resolve(mode, **switches)
        return (
            ([Units.NONE, Units.NONE, Units.NONE] if not just_metrics else [])
            + BinarizationResults.get_units(mode)
            + IntensityResults.get_units(mode)
            + (FlowResults.get_units(mode) if with_flow else [])
            + _family_parts(mode, enabled, lambda f: f.results_cls.get_units(mode))
        )

    @classmethod
    def get_physical_units(cls, just_metrics: bool = False, mode=None,
                           **switches) -> List[Units]:
        mode, with_flow, enabled = _resolve(mode, **switches)
        return (
            ([Units.NONE, Units.NONE, Units.NONE] if not just_metrics else [])
            + BinarizationResults.get_physical_units(mode)
            + IntensityResults.get_units(mode)
            + (FlowResults.get_units(mode) if with_flow else [])
            + _family_parts(mode, enabled, lambda f: f.results_cls.get_units(mode))
        )

    def convert_flags(self) -> str:
        flag_lst = []
        if self.dim_channel_flag == 1:
            flag_lst.append("1")
        if self.intensity.saturation_flag == 1:
            flag_lst.append("2")
        if self.binarization.structural_correlation_flag == 1:
            flag_lst.append("3")
        if self.flow.velocity_correlation_flag == 1:
            flag_lst.append("4")
        if self.z_range_flag == 1:
            flag_lst.append("5")
        if self.fov_clip_flag == 1:
            flag_lst.append("6")
        if self.mesh_open_surface_flag == 1:
            flag_lst.append("7")
        return ";".join(flag_lst) if flag_lst else "0"

    def _rows(self, just_metrics, mode, enabled, with_flow, physical):
        data = []
        self.total_flags = self.convert_flags()
        if not just_metrics:
            data = [self.filepath, self.channel, self.total_flags]
        data.extend(self.binarization.get_physical_data() if physical
                    else self.binarization.get_data())
        data.extend(self.intensity.get_data())
        if with_flow:
            data.extend(self.flow.get_data())
        for family in OPTIONAL_FAMILIES:
            if enabled[family.switch]:
                data.extend(getattr(self, family.attribute).get_data())
        return data

    def get_data(self, just_metrics: bool = False, mode=None, **switches) -> List[float]:
        mode, with_flow, enabled = _resolve(mode, **switches)
        return self._rows(just_metrics, mode, enabled, with_flow, physical=False)

    def get_physical_data(self, just_metrics: bool = False, mode=None,
                          **switches) -> List[float]:
        mode, with_flow, enabled = _resolve(mode, **switches)
        return self._rows(just_metrics, mode, enabled, with_flow, physical=True)

    def _dict(self, just_metrics, mode, enabled, with_flow, physical):
        # Keys come from the mode-aware metric lists so a dict and a CSV row built from
        # the same results always agree on names and membership.
        data = dict(zip(
            BinarizationResults.get_physical_metrics(mode) if physical
            else BinarizationResults.get_metrics(mode),
            self.binarization.get_physical_data() if physical
            else self.binarization.get_data()))
        data |= dict(zip(IntensityResults.get_metrics(mode), self.intensity.get_data()))
        if with_flow:
            data |= self.flow.get_dict_data()
        for family in OPTIONAL_FAMILIES:
            if enabled[family.switch]:
                values = getattr(self, family.attribute)
                try:
                    data |= values.get_dict_data(mode)
                except TypeError:      # families whose dict takes no mode
                    data |= values.get_dict_data()
        self.total_flags = self.convert_flags()
        if just_metrics:
            return data
        return {Metrics.FILEPATH: self.filepath,
                 Metrics.CHANNEL: self.channel,
                 Metrics.FLAGS: self.total_flags} | data

    def get_dict_data(self, just_metrics: bool = False, mode=None, **switches) -> dict:
        mode, with_flow, enabled = _resolve(mode, **switches)
        return self._dict(just_metrics, mode, enabled, with_flow, physical=False)

    def get_physical_dict_data(self, just_metrics: bool = False, mode=None,
                               **switches) -> dict:
        mode, with_flow, enabled = _resolve(mode, **switches)
        return self._dict(just_metrics, mode, enabled, with_flow, physical=True)

    def to_physical_array(self, **kwargs) -> np.ndarray:
        """Convert results to a NumPy array for easier manipulation."""
        return np.array(self.get_physical_data(**kwargs), dtype=float)

def sort_channel_results_by_metric(
    results: List[ChannelResults], sort_metric: str
) -> None:
    def get_metric_value(result: ChannelResults, metric_name: str) -> float:
        """Get metric value by header name."""
        headers = ChannelResults.get_headers(just_metrics=False)
        data = result.get_data(just_metrics=False)

        try:
            idx = headers.index(metric_name)
            return data[idx]
        except (ValueError, IndexError):
            return 0.0  # Default for sorting if metric not found

    results.sort(key=lambda r: get_metric_value(r, sort_metric))


