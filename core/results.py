from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

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
            Units.PERCENT_FRAMES,
            Units.PERCENT_FOV,
            Units.PERCENT_FOV,
            Units.PERCENT_CHANGE,
            Units.PERCENT_CHANGE,
            Units.PERCENT_FOV,
            Units.PERCENT_FOV,
            Units.NONE,
            Units.PERCENT_FOV,
            Units.PERCENT_FOV,
            Units.LENGTH,
            Units.LENGTH,
        ]
    
    @classmethod
    def get_physical_units(cls, mode: AnalysisMode = None) -> List[Units]:
        size = Units.VOLUME if (mode is not None and mode.is_volumetric) else Units.AREA
        return [
            Units.PERCENT_FRAMES,
            size,
            size,
            Units.PERCENT_CHANGE,
            Units.PERCENT_CHANGE,
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
    def get_units(cls) -> List[Units]:
        return [
            Units.SPEED,
            Units.SPEED,
            Units.DIRECTION,
            Units.DIRECTION,
            Units.LENGTH,
            Units.NONE,
            Units.NONE,
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
    volume_ratio: float = np.nan

    mean_curvature: float = np.nan
    invagination_ratio: float = np.nan
    concave_ratio: float = np.nan

    @classmethod
    def get_metrics(cls) -> List[Metrics]:
        return [
            Metrics.MESH_VOLUME,
            Metrics.MESH_SURFACE_AREA,
            Metrics.MESH_SPHERICITY,
            Metrics.MESH_EQUIVALENT_RADIUS,
            Metrics.MESH_HEIGHT,
            Metrics.MESH_VOLUME_RATIO,
            Metrics.CURVATURE_MEAN,
            Metrics.CURVATURE_INVAGINATION,
            Metrics.CURVATURE_CONCAVE,
        ]

    @classmethod
    def get_units(cls) -> List[Units]:
        return [
            Units.VOLUME,
            Units.AREA,
            Units.NONE,
            Units.LENGTH,
            Units.LENGTH,
            Units.NONE,
            Units.CURVATURE,
            Units.NONE,
            Units.NONE,
        ]

    def get_data(self) -> List[float]:
        return [
            self.mesh_volume,
            self.surface_area,
            self.sphericity,
            self.equivalent_radius,
            self.height,
            self.volume_ratio,
            self.mean_curvature,
            self.invagination_ratio,
            self.concave_ratio,
        ]

    def get_dict_data(self) -> dict:
        return dict(zip(self.get_metrics(), self.get_data()))

    def is_populated(self) -> bool:
        """True if any value was actually measured."""
        return bool(np.any(np.isfinite(np.array(self.get_data(), dtype=float))))



def _resolve(mode, include_mesh, include_components=None):
    """Normalise the optional-family switches used across ChannelResults.

    ``mode`` may be an AnalysisMode, a key string, or None for the legacy 2D layout.
    Each optional family defaults to whatever the mode supports but can be forced either
    way; the writer uses that to emit a family's columns only when data is really there,
    so an xyzt run that skipped meshing does not produce nine empty columns.

    If a fourth optional family ever appears, replace these positional flags with a
    registry of (name, results class, capability predicate) -- three is the point at
    which that starts paying for itself.
    """
    if mode is not None and not isinstance(mode, AnalysisMode):
        mode = get_mode(mode)
    if include_mesh is None:
        include_mesh = bool(mode is not None and mode.supports_mesh)
    if include_components is None:
        include_components = False
    with_flow = True if mode is None else mode.supports_flow
    return mode, bool(include_mesh), with_flow, bool(include_components)


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
class ChannelResults(ResultsBase):
    """Complete analysis results for a single channel."""

    filepath: str
    channel: int
    total_flags: str = "0"
    dim_channel_flag: int = 0  # 0=normal, 1=dim channel

    binarization: BinarizationResults = field(default_factory=BinarizationResults)
    intensity: IntensityResults = field(default_factory=IntensityResults)
    flow: FlowResults = field(default_factory=FlowResults)
    mesh: MeshResults = field(default_factory=MeshResults)
    components: ComponentResults = field(default_factory=ComponentResults)

    @classmethod
    def _get_base_headers(cls) -> List[str]:
        return ["Filepath", "Channel", "Flags"]

    @classmethod
    def get_metrics(cls, just_metrics: bool = False, include_mesh: bool = None,
                    mode=None, include_components=None) -> List[Metrics]:
        mode, include_mesh, with_flow, include_components = _resolve(
            mode, include_mesh, include_components)
        return (
            (
                [Metrics.FILEPATH, Metrics.CHANNEL, Metrics.FLAGS]
                if not just_metrics
                else []
            )
            + BinarizationResults.get_metrics(mode)
            + IntensityResults.get_metrics(mode)
            + (FlowResults.get_metrics() if with_flow else [])
            + (MeshResults.get_metrics() if include_mesh else [])
            + (ComponentResults.get_metrics(mode) if include_components else [])
        )

    @classmethod
    def get_physical_metrics(cls, just_metrics: bool = False, include_mesh: bool = None,
                             mode=None, include_components=None) -> List[Metrics]:
        mode, include_mesh, with_flow, include_components = _resolve(
            mode, include_mesh, include_components)
        return (
            (
                [Metrics.FILEPATH, Metrics.CHANNEL, Metrics.FLAGS]
                if not just_metrics
                else []
            )
            + BinarizationResults.get_physical_metrics(mode)
            + IntensityResults.get_metrics(mode)
            + (FlowResults.get_metrics() if with_flow else [])
            + (MeshResults.get_metrics() if include_mesh else [])
            + (ComponentResults.get_metrics(mode) if include_components else [])
        )
    
    @classmethod
    def get_physical_headers(cls, just_metrics: bool = False, include_mesh: bool = None,
                             mode=None, include_components=None) -> List[str]:
        """Get headers for CSV output."""
        return [m.value for m in cls.get_physical_metrics(
            just_metrics, include_mesh, mode, include_components)]

    @classmethod
    def get_units(cls, just_metrics: bool = False, include_mesh: bool = None,
                  mode=None, include_components=None) -> List[Units]:
        mode, include_mesh, with_flow, include_components = _resolve(
            mode, include_mesh, include_components)
        return (
            ([Units.NONE, Units.NONE, Units.NONE] if not just_metrics else [])
            + BinarizationResults.get_units(mode)
            + IntensityResults.get_units(mode)
            + (FlowResults.get_units() if with_flow else [])
            + (MeshResults.get_units() if include_mesh else [])
            + (ComponentResults.get_units(mode) if include_components else [])
        )
    
    @classmethod
    def get_physical_units(cls, just_metrics: bool = False, include_mesh: bool = None,
                           mode=None, include_components=None) -> List[Units]:
        mode, include_mesh, with_flow, include_components = _resolve(
            mode, include_mesh, include_components)
        return (
            ([Units.NONE, Units.NONE, Units.NONE] if not just_metrics else [])
            + BinarizationResults.get_physical_units(mode)
            + IntensityResults.get_units(mode)
            + (FlowResults.get_units() if with_flow else [])
            + (MeshResults.get_units() if include_mesh else [])
            + (ComponentResults.get_units(mode) if include_components else [])
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
        return ";".join(flag_lst) if flag_lst else "0"
            

    def get_data(self, just_metrics: bool = False, include_mesh: bool = None,
                 mode=None, include_components=None) -> List[float]:
        mode, include_mesh, with_flow, include_components = _resolve(
            mode, include_mesh, include_components)
        data = []
        self.total_flags = self.convert_flags()
        if not just_metrics:
            data = [self.filepath, self.channel, self.total_flags]
        data.extend(self.binarization.get_data())
        data.extend(self.intensity.get_data())
        if with_flow:
            data.extend(self.flow.get_data())
        if include_mesh:
            data.extend(self.mesh.get_data())
        if include_components:
            data.extend(self.components.get_data())
        return data
    
    def get_physical_data(self, just_metrics: bool = False, include_mesh: bool = None,
                 mode=None, include_components=None) -> List[float]:
        mode, include_mesh, with_flow, include_components = _resolve(
            mode, include_mesh, include_components)
        data = []
        self.total_flags = self.convert_flags()
        if not just_metrics:
            data = [self.filepath, self.channel, self.total_flags]
        data.extend(self.binarization.get_physical_data())
        data.extend(self.intensity.get_data())
        if with_flow:
            data.extend(self.flow.get_data())
        if include_mesh:
            data.extend(self.mesh.get_data())
        if include_components:
            data.extend(self.components.get_data())
        return data
    
    def get_dict_data(self, just_metrics: bool = False, include_mesh: bool = None,
                      mode=None, include_components=None) -> dict:
        mode, include_mesh, with_flow, include_components = _resolve(
            mode, include_mesh, include_components)
        # Keys come from the mode-aware metric lists so a dict and a CSV row
        # built from the same results always agree on names and membership.
        binarization_data = dict(zip(
            BinarizationResults.get_metrics(mode), self.binarization.get_data()))
        intensity_data = dict(zip(
            IntensityResults.get_metrics(mode), self.intensity.get_data()))
        flow_data = self.flow.get_dict_data() if with_flow else {}
        mesh_data = self.mesh.get_dict_data() if include_mesh else {}
        component_data = (self.components.get_dict_data(mode)
                          if include_components else {})
        self.total_flags = self.convert_flags()
        if just_metrics:
            data = binarization_data | intensity_data | flow_data | mesh_data | component_data
        else:
            data = {Metrics.FILEPATH: self.filepath,
                    Metrics.CHANNEL: self.channel,
                    Metrics.FLAGS: self.total_flags}
            data = data | binarization_data | intensity_data | flow_data | mesh_data
        return data
    
    def get_physical_dict_data(self, just_metrics: bool = False, include_mesh: bool = None,
                      mode=None, include_components=None) -> dict:
        mode, include_mesh, with_flow, include_components = _resolve(
            mode, include_mesh, include_components)
        # Keys come from the mode-aware metric lists so a dict and a CSV row
        # built from the same results always agree on names and membership.
        binarization_data = dict(zip(
            BinarizationResults.get_physical_metrics(mode), self.binarization.get_physical_data()))
        intensity_data = dict(zip(
            IntensityResults.get_metrics(mode), self.intensity.get_data()))
        flow_data = self.flow.get_dict_data() if with_flow else {}
        mesh_data = self.mesh.get_dict_data() if include_mesh else {}
        component_data = (self.components.get_dict_data(mode)
                          if include_components else {})
        self.total_flags = self.convert_flags()
        if just_metrics:
            data = binarization_data | intensity_data | flow_data | mesh_data | component_data
        else:
            data = {Metrics.FILEPATH: self.filepath,
                    Metrics.CHANNEL: self.channel,
                    Metrics.FLAGS: self.total_flags}
            data = data | binarization_data | intensity_data | flow_data | mesh_data
        return data
    
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


