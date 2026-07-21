from abc import ABC
from enum import Enum
from typing import List
from core.config import WriterConfig

import numpy as np

class UnitsNum(ABC):
    def _keys(self):
        return (name for name in dir(self) if not name.startswith('_'))
    __iter__ = _keys
    def _values(self):
        return (getattr(self, name) for name in self)
    def _items(self):
        return ((name, getattr(self, name)) for name in self)
    
    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError
    
    def __setattr__(self, name, val):
        if name in self:
            self.name = val
        raise AttributeError

class Metrics(Enum):
    """Enum for different metrics used in analysis."""

    # Metrics for binarization analysis
    CONNECTIVITY = "Connectivity"
    ISLAND_MAX_AREA = "Maximum Island Area"
    VOID_MAX_AREA = "Maximum Void Area"
    MAX_ISLAND_AREA_CHANGE = "Maximum Island Area Change"
    MAX_VOID_AREA_CHANGE = "Maximum Void Area Change"
    ISLAND_MAX_AREA_INITIAL = "Initial Maximum Island Area"
    ISLAND_MAX_AREA_INITIAL2 = "Initial 2nd Maximum Island Area"
    ISLAND_ANISOTROPY = "Mean Island Anisotropy"
    ISLAND_MEAN_AREA = "Mean Island Area"
    ISLAND_TOTAL_AREA = "Total Island Area"
    ISLAND_DISTANCE = "Mean Island Separation"
    ISLAND_CORRELATION = "Structural Correlation Length"

    # Physical Units for Binarization Void/Island Metrics
    ISLAND_MAX_AREA_QUANTITY = "Maximum Island Area Quantity"
    VOID_MAX_AREA_QUANTITY = "Maximum Void Area Quantity"
    ISLAND_MAX_AREA_INITIAL_QUANTITY = "Initial Maximum Island Area Quantity"
    ISLAND_MAX_AREA_INITIAL2_QUANTITY = "Initial 2nd Maximum Island Area Quantity"
    ISLAND_MEAN_AREA_QUANTITY = "Mean Island Area Quantity"
    ISLAND_TOTAL_AREA_QUANTITY = "Total Island Area Quantity"

    # Volumetric (xyzt) counterparts of the island/void size metrics. Separate members
    # rather than a rename so the 2D modes keep byte-identical headers and existing
    # 2D CSVs -- including the published reference set -- still load.
    ISLAND_MAX_VOLUME = "Maximum Island Volume"
    VOID_MAX_VOLUME = "Maximum Void Volume"
    MAX_ISLAND_VOLUME_CHANGE = "Maximum Island Volume Change"
    MAX_VOID_VOLUME_CHANGE = "Maximum Void Volume Change"
    ISLAND_MAX_VOLUME_INITIAL = "Initial Maximum Island Volume"
    ISLAND_MAX_VOLUME_INITIAL2 = "Initial 2nd Maximum Island Volume"
    ISLAND_MEAN_VOLUME = "Mean Island Volume"
    ISLAND_TOTAL_VOLUME = "Total Island Volume"

    ISLAND_MAX_VOLUME_QUANTITY = "Maximum Island Volume Quantity"
    VOID_MAX_VOLUME_QUANTITY = "Maximum Void Volume Quantity"
    ISLAND_MAX_VOLUME_INITIAL_QUANTITY = "Initial Maximum Island Volume Quantity"
    ISLAND_MAX_VOLUME_INITIAL2_QUANTITY = "Initial 2nd Maximum Island Volume Quantity"
    ISLAND_MEAN_VOLUME_QUANTITY = "Mean Island Volume Quantity"
    ISLAND_TOTAL_VOLUME_QUANTITY = "Total Island Volume Quantity"

    # In xyz the progression axis is depth, so a "Change" is a change with depth, not
    # over time. Named explicitly: a depth trend read as a time trend would be a silent
    # scientific error, which is the whole reason modes exist.
    MAX_ISLAND_AREA_CHANGE_Z = "Maximum Island Area Change (over Z)"
    MAX_VOID_AREA_CHANGE_Z = "Maximum Void Area Change (over Z)"
    KURTOSIS_DIFF_Z = "Kurtosis Change (over Z)"
    MEDIAN_SKEW_DIFF_Z = "Median Skewness Change (over Z)"
    MODE_SKEW_DIFF_Z = "Mode Skewness Change (over Z)"

    # Per-connected-component size distribution. Off by default: the barcode is already
    # wide, and these describe the *spread* of object sizes rather than adding a new
    # physical quantity. Mean and total size are not here -- they are already in the
    # binarization family as Mean/Total Island Area or Volume.
    ISLAND_COUNT = "Island Count"
    ISLAND_AREA_SD = "Island Area SD"
    ISLAND_AREA_SKEW = "Island Area Skewness"
    ISLAND_AREA_MEDIAN = "Median Island Area"
    ISLAND_VOLUME_SD = "Island Volume SD"
    ISLAND_VOLUME_SKEW = "Island Volume Skewness"
    ISLAND_VOLUME_MEDIAN = "Median Island Volume"

    # Extensive intensity quantities. Every other intensity metric is intensive -- a
    # dimensionless descriptor of histogram shape -- so until these exist nothing in the
    # branch scales with the amount of material. Density is named per mode, following
    # the Area/Volume precedent, because its unit genuinely differs.
    INTENSITY_TOTAL = "Total Intensity"
    INTENSITY_MEAN = "Mean Intensity"
    INTENSITY_SD = "Intensity SD"
    INTENSITY_DENSITY_AREA = "Intensity Density (per area)"
    INTENSITY_DENSITY_VOLUME = "Intensity Density (per volume)"

    # Packing topology: how objects are arranged relative to each other, which nothing
    # else in BARCODE describes. Sizes and separations can be near-uniform in a
    # space-filling monolayer while the neighbour-number distribution changes -- that
    # distribution is the canonical epithelial readout.
    CONTACT_NUMBER_MEAN = "Mean Contact Number"
    CONTACT_NUMBER_SD = "Contact Number SD"
    HEXAGONAL_FRACTION = "Hexagonal Fraction"

    # Per-OBJECT metrics, for barcodes whose rows are objects rather than fields. Named
    # distinctly from their field-level aggregates on purpose: "Contact Number" is one
    # object's neighbour count, "Mean Contact Number" is the field's average, and a
    # column headed the same in both would make the two silently interchangeable.
    OBJECT_VOLUME = "Object Volume"
    OBJECT_DIAMETER = "Equivalent Diameter"
    OBJECT_CONTACT_NUMBER = "Contact Number"

    # Identity of the object a row describes, when a supplied instance mask partitions
    # the field into several. Blank on aggregate rows. Present in the contract from the
    # start so per-object rows are later a behaviour change, not a schema change.
    OBJECT_ID = "Object ID"

    # Provenance: which range the numbers in this row were computed over. Flag digit 5
    # says a range was restricted; these say which one, per row.
    RANGE_Z_START = "Z Range Start"
    RANGE_Z_END = "Z Range End"
    RANGE_T_START = "T Range Start"
    RANGE_T_END = "T Range End"

    # Metrics for optical flow analysis
    SPEED = "Speed"
    DELTA_SPEED = "Speed Change"
    MEAN_THETA = "Mean Flow Direction"
    MEAN_SIGMA_THETA = "Directional Spread"
    VELOCITY_CORRELATION = "Velocity Correlation Length"
    DIVERGENCE = "Divergence"
    CURL = "Curl"

    # Metrics for intensity distribution comparison
    MAX_KURTOSIS = "Maximum Kurtosis"
    MAX_MEDIAN_SKEW = "Maximum Median Skewness"
    MAX_MODE_SKEW = "Maximum Mode Skewness"
    KURTOSIS_DIFF = "Kurtosis Change"
    MEDIAN_SKEW_DIFF = "Median Skewness Change"
    MODE_SKEW_DIFF = "Mode Skewness Change"

    # Metrics from surface meshing of a segmented object (analysis/volumetric/mesh.py).
    # Only emitted when mesh data is present, so the 2D schema is unchanged.
    # "Mesh Volume" is deliberately distinct from the island volume: meshing smooths the
    # surface, so the two differ by a few percent, and Mesh Volume Ratio reports exactly
    # that discrepancy as a fidelity check.
    MESH_VOLUME = "Mesh Volume"
    MESH_SURFACE_AREA = "Mesh Surface Area"
    MESH_SPHERICITY = "Sphericity"
    MESH_EQUIVALENT_RADIUS = "Equivalent Sphere Radius"
    MESH_HEIGHT = "Mesh Height"
    # TCell nuc_aspect_ratio: lateral size over axial size, (MIP major + minor)/(2*height).
    # > 1 is a flat/oblate object, ~1 is round. Dimensionless, so it is size-independent.
    MESH_ASPECT_RATIO = "Aspect Ratio"
    MESH_VOLUME_RATIO = "Mesh Volume Ratio"
    # Convexity, as voxel counts (volume / convex-hull volume), matching
    # chromatin-analysis' morph3d_solidity and MATLAB's regionprops3 Solidity. 1 = convex;
    # lower = more lobed. Concavity is its complement and is reported for readers who
    # think in "how much of the hull is empty".
    MESH_SOLIDITY = "Solidity"
    MESH_CONCAVITY = "Concavity"

    # Curvature of that surface (analysis/volumetric/curvature.py).
    # <H> denotes the area-weighted surface average of the pointwise mean curvature
    # H = (k1 + k2)/2. Two averages are stacked -- "mean curvature" is already a
    # local quantity -- and the surface average excludes bottom/top and outlier
    # faces, so the bare name "Mean Curvature" undersells what was computed.
    CURVATURE_MEAN = "Mean Curvature <H>"
    CURVATURE_INVAGINATION = "Invagination Ratio"
    CURVATURE_CONCAVE = "Concave Area Fraction"

    # Extremes of the same area-weighted curvature field. <H> averages k1 and k2
    # together, so a saddle -- sharply curved both ways -- averages towards zero and
    # reads as flat. These keep the two principal directions apart: the most concave
    # and the most convex parts of the surface.
    CURVATURE_MIN = "Minimum Curvature"
    CURVATURE_MAX = "Maximum Curvature"

    # Where a volume is widest (analysis/volumetric/slice_profile.py). For a stack
    # through a curved surface or a rounded object, the slice with the most foreground
    # locates the equator, and its depth is a shape descriptor rather than a
    # bookkeeping detail: it moves when the object flattens or tilts.
    BROADEST_SLICE_INDEX = "Broadest Slice Index"
    BROADEST_SLICE_DEPTH = "Broadest Slice Depth"
    BROADEST_SLICE_AREA = "Broadest Slice Area"

    # Intensity distribution WITHIN the segmentation (analysis/volumetric/mask_intensity.py).
    # Distinct from the intensity branch, which describes whichever voxels it was given
    # -- usually including background. These describe how signal is distributed inside
    # each object, which is the clustering readout.
    MASK_INTENSITY_MFI = "In-Mask MFI"
    MASK_INTENSITY_SD = "In-Mask Intensity SD"
    MASK_INTENSITY_CV = "In-Mask Intensity CV"
    MASK_INTENSITY_SKEW = "In-Mask Intensity Skewness"
    MASK_INTENSITY_ENTROPY = "In-Mask Intensity Entropy"
    MASK_INTENSITY_ENTROPY_NORM = "In-Mask Normalized Entropy"
    MASK_INTENSITY_BRIGHT_FRACTION = "In-Mask Fraction Above 2x Median"

    IGNORE = "Ignore this"
    FILEPATH = "File"
    CHANNEL = "Channel"
    FLAGS = "Flags"


class Units(UnitsNum):
    """Enum for units corresponding to each metric."""

    NONE: str = ""
    # These three hold values in 0-1, NOT percentages, and the labels used to say "%".
    # A void filling 94% of the field was drawn on the barcode as "0.94 % of FOV" -- two
    # orders of magnitude out. The metrics themselves are reference-validated and are NOT
    # rescaled to match the old wording; the wording is corrected to match the data.
    # Named FRACTION_*, not PERCENT_*, so the next person writing a label reads the unit
    # off the name and gets it right.
    FRACTION_FOV: str = "fraction of FOV"
    # final / initial, so 1.0 means NO CHANGE. "Fractional Change" conventionally means
    # (final - initial) / initial, where 0 means no change -- so the old label was wrong
    # about the magnitude AND the zero point. See analysis/binarization.py: the value is
    # `mean(final window) / initial`. The difference-style Change metrics (Speed,
    # Kurtosis, Skewness) are computed as `final - initial` and keep their own units.
    RATIO_TO_INITIAL: str = "ratio to initial"
    SPEED: str = "μm/s"
    DIRECTION: str = "rads"
    FRACTION_FRAMES: str = "fraction of frames"
    LENGTH: str = "μm"
    AREA: str = "μm^2"
    VOLUME: str = "μm^3"
    CURVATURE: str = "1/μm"
    # Divergence and curl are spatial derivatives of a velocity, so (μm/s)/μm = 1/s. Only
    # the volumetric branch earns this label: the 2D branch differentiates a *cumulative
    # unit-vector* field, which is 1/μm and a different quantity entirely, so its columns
    # keep the blank unit they have always had rather than being relabelled wrongly.
    RATE: str = "1/s"
    INTENSITY: str = "a.u."
    INTENSITY_PER_AREA: str = "a.u./μm^2"
    INTENSITY_PER_VOLUME: str = "a.u./μm^3"
    SLICE_INDEX: str = "slice"


def get_data_limits(
    data: np.ndarray, metrics: List[Metrics], units: List[Units]
) -> List[List[float]]:
    """
    Get limits for each metric in the data array based on the provided metrics and units.

    Args:
        data: 2D numpy array with shape (n_samples, n_metrics)
        metrics: List of Metrics to consider
        units: Corresponding list of Units for each metric

    Returns:
        List of limits for each metric
    """
    binarized_static_limits = [0, 1]
    direction_static_limits = [-np.pi, np.pi]
    direction_spread_static_limit = [0, np.pi]

    limits = []

    def _all_nan(_data: np.ndarray) -> bool:
        """True when a column holds no finite value at all.

        A family is emitted whenever ANY of its values is finite, but the colour scale is
        built per COLUMN -- so meshing with `mesh_curvature` off emits the three curvature
        columns entirely NaN. `nanmin`/`nanmax` then return NaN, the `_limits[0] ==
        _limits[1]` repair below never fires (NaN != NaN), and `Normalize(vmin=nan,
        vmax=nan)` reaches the colorbar. That renders as a garbage column with a
        meaningless scale, and because `save_analysis_results` wraps barcode generation in
        a bare except, a hard failure there is swallowed into the fail log.
        """
        return not np.any(np.isfinite(np.asarray(_data, dtype=float)))

    def dynamic_limits(_data: np.ndarray, threshold: float) -> List[float]:
        """Calculate dynamic limits based on the data and a threshold."""
        if _all_nan(_data):
            return [0.0, 1.0]
        _limits = [np.nanmin(_data), np.nanmax(_data)]
        if threshold < _limits[0]:
            _limits[0] = threshold
        elif threshold > _limits[1]:
            _limits[1] = threshold
        
        if _limits[0] == _limits[1]:
            if _limits[0] == 0:
                _limits[1] = 1
            elif _limits[0] > 0:
                _limits[0] = 0
            else:
                _limits[1] = 0

        return _limits

    # Assign limits based on metrics and units
    for i, (metric, unit) in enumerate(zip(metrics, units)):
        if unit == Units.FRACTION_FRAMES or unit == Units.FRACTION_FOV:
            limits.append(binarized_static_limits)
        elif unit == Units.DIRECTION:
            if metric == Metrics.MEAN_SIGMA_THETA:
                limits.append(direction_spread_static_limit)
            else:
                limits.append(direction_static_limits)
        elif unit == Units.RATIO_TO_INITIAL:
            limits.append(dynamic_limits(data[:, i], 1))
        elif unit in [Units.SPEED, Units.LENGTH, Units.AREA, Units.VOLUME,
                      Units.CURVATURE, Units.RATE, Units.INTENSITY,
                      Units.INTENSITY_PER_AREA, Units.INTENSITY_PER_VOLUME,
                      Units.SLICE_INDEX]:
            if metric == Metrics.DELTA_SPEED or unit in (Units.CURVATURE, Units.RATE):
                # Curvature is signed: a concave surface gives a negative mean, so a
                # [0, max] scale would clip half its range. Divergence is signed for the
                # same reason -- compaction is negative -- and scaling it about 0 keeps
                # the barcode identical to when these columns carried a blank unit.
                limits.append(dynamic_limits(data[:, i], 0))
            elif _all_nan(data[:, i]):
                limits.append([0.0, 1.0])
            else:
                limits.append([0, np.nanmax(data[:, i])])
        elif unit == Units.NONE:
            if metric == Metrics.ISLAND_ANISOTROPY:
                limits.append([1.0, 2.0] if _all_nan(data[:, i])
                              else [1, np.nanmax(data[:, i])])
            else:
                limits.append(dynamic_limits(data[:, i], 0))
        else:
            raise ValueError(f"Unsupported unit: {unit}")
    return limits


def selection_mask(headers: List[str], hidden: List[str] = None) -> List[bool]:
    """Boolean mask over ``headers``, False for anything named in ``hidden``.

    Feeds ``visualization.barcode.generate_combined_barcode``'s ``metrics_to_visualize``.
    Matching is by metric *name*, not position, so a mode that adds or drops a family
    cannot silently hide the wrong column -- the failure that positional boolean lists
    invite.

    Unknown names in ``hidden`` are ignored rather than raising: a selection saved for
    one mode is often reused for another where some metrics do not exist.
    """
    if not hidden:
        return [True] * len(headers)
    hidden_set = {h.strip() for h in hidden if h and h.strip()}
    return [header not in hidden_set for header in headers]
