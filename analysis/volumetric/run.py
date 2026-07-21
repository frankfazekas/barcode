"""Orchestration for the volumetric pipeline — the 3D counterpart of ``analysis/run.py``.

Load -> (optionally) pair with a segmentation and put both on one isotropic grid ->
structural + intensity metrics -> a ``ChannelResults`` the existing writer understands.
"""
from __future__ import annotations

import os
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from analysis.volumetric.binarization import (
    VolumetricBinarizationDetail,
    analyze_binarization_3d,
    binarize_volume,
    invert_volume,
)
from analysis.volumetric.mask_intensity import (
    MaskIntensityDetail,
    analyze_mask_intensity,
    summarise_mask_intensity,
)
from analysis.volumetric.slice_profile import (
    SliceProfileDetail,
    slice_profile,
    summarise_slice_profile,
)
from analysis.volumetric.flow import VolumetricFlowDetail, analyze_optical_flow_3d
from analysis.volumetric.intensity import (
    VolumetricIntensityDetail,
    analyze_intensity_3d,
    analyze_intensity_magnitude,
)
from analysis.volumetric.mesh import MeshingError, NucleusMesh, mesh_series
from analysis.volumetric.packing import (
    VolumetricPackingDetail, packing_topology, summarise_packing)
from analysis.volumetric.provenance import build_range_results
from analysis.volumetric.reader import (
    VolumeStack, apply_t_range, apply_z_range, read_volume)
from analysis.volumetric.segmentation import load_segmentation
from core import BarcodeConfig, ChannelResults, VolumetricConfig
from core.results import ComponentResults, CurvatureRangeResults, MeshResults


@dataclass
class VolumetricRunDetail:
    """Everything the harness wants to print but the CSV has no column for."""

    stack: VolumeStack = None
    mask_paths: Optional[List[str]] = None
    spacing_zyx_um: Tuple[float, float, float] = None
    shape_zyx: Tuple[int, int, int] = None
    resample_info: Dict[str, object] = field(default_factory=dict)
    binarization: Optional[VolumetricBinarizationDetail] = None
    intensity: Optional[VolumetricIntensityDetail] = None
    flow: Optional[VolumetricFlowDetail] = None
    meshes: List[NucleusMesh] = field(default_factory=list)
    packing: List[VolumetricPackingDetail] = field(default_factory=list)
    slice_profile: List[SliceProfileDetail] = field(default_factory=list)
    mask_intensity: List[MaskIntensityDetail] = field(default_factory=list)
    frame_indices: List[int] = field(default_factory=list)


def _frame_binary(volumes, masks, frame_idx, vcfg):
    """The binary volume for one timepoint, by the binarization branch's own rule.

    Duplicated deliberately rather than cached on the detail: the binarization branch
    holds only reduced statistics, and keeping every binarized volume alive to serve an
    optional family would multiply peak memory across the whole series. Recomputing one
    frame at a time is cheap; the rule itself must not diverge, hence the same helpers.
    """
    if masks is not None:
        return masks[frame_idx].astype(bool)
    binary = binarize_volume(
        volumes[frame_idx], vcfg.threshold_offset, vcfg.minimum_island_size)
    return invert_volume(binary) if vcfg.invert_binarization else binary


def _add_optional_families(results, detail, volumes, masks, spacing_zyx,
                           frame_indices, vcfg):
    """Populate the opt-in families that need their own pass over the volumes.

    Each reports rather than raises when its prerequisites are missing, matching how
    meshing and packing handle an absent segmentation: one misconfigured file should not
    abort a batch, and an empty column with a printed reason is honest whereas a zero is
    not.
    """
    if vcfg.enable_slice_profile:
        per_frame = []
        for frame_idx in frame_indices:
            binary = _frame_binary(volumes, masks, frame_idx, vcfg)
            frame_result, frame_detail = slice_profile(binary, spacing_zyx[0])
            per_frame.append(frame_result)
            detail.slice_profile.append(frame_detail)
        if per_frame:
            results.slice_profile = summarise_slice_profile(per_frame)
            # Clipping anywhere in the series taints the averaged metrics, so the flag
            # is the union over analysed timepoints rather than the last one's value.
            results.fov_clip_flag = int(any(d.clipped for d in detail.slice_profile))
            print(f"  {detail.slice_profile[0].describe()}", flush=True)

    if vcfg.enable_curvature_range:
        curvatures = [m.curvature for m in detail.meshes if m.curvature is not None]
        if not curvatures:
            print(
                "Curvature range needs the mesh family, which produced no curvature; "
                "skipping. Enable meshing and supply a segmentation.",
                flush=True,
            )
        else:
            results.curvature_range = CurvatureRangeResults(
                min_curvature=_mean_of([c.min_curvature for c in curvatures]),
                max_curvature=_mean_of([c.max_curvature for c in curvatures]),
            )

    if vcfg.enable_mask_intensity:
        if masks is None:
            print(
                "In-mask intensity needs a segmentation; skipping. These metrics "
                "describe the inside of each object, so there is nothing to describe "
                "without one.",
                flush=True,
            )
        else:
            per_frame = []
            for frame_idx in frame_indices:
                frame_result, frame_detail = analyze_mask_intensity(
                    volumes[frame_idx], masks[frame_idx],
                    bins=vcfg.mask_intensity_bins,
                    min_voxels=vcfg.mask_intensity_min_voxels,
                )
                per_frame.append(frame_result)
                detail.mask_intensity.append(frame_detail)
            results.mask_intensity = summarise_mask_intensity(per_frame)
            print(f"  {detail.mask_intensity[0].describe()}", flush=True)


def _mean_of(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else np.nan


_channel_warning_shown = False


def warn_if_channels_dropped(config: BarcodeConfig) -> None:
    """Say so when "Parse All Channels" is set but only one channel will be analysed.

    ``core.pipeline`` routes the volumetric modes before ``determine_channels_to_process``
    ever runs, and every volumetric entry point analyses
    ``config.channels.selected_channel`` alone. Ticking the box therefore produced a
    single row for one channel and no indication that the rest had been dropped. Said
    once per run rather than per file, so a 1000-file batch does not repeat it.
    """
    global _channel_warning_shown
    if _channel_warning_shown or not getattr(config.channels, "parse_all_channels", False):
        return
    _channel_warning_shown = True
    print(
        f"Note: 'Parse All Channels' is not supported by the volumetric modes; analysing "
        f"channel {config.channels.selected_channel} only. Run the other channels "
        f"separately by changing Choose Channel.",
        flush=True,
    )


def select_frame_indices(n_frames: int, frame_step: int) -> List[int]:
    """Timepoints to analyse.

    The 2D helper (``utils.find_analysis_frames``) divides its step by 5 until it is
    below the series length, which at ``n_frames == 1`` yields a float step and makes
    ``range`` raise ``TypeError``. A single-volume series is the normal case here, so
    it is handled explicitly.
    """
    if n_frames <= 1:
        return [0]
    # A step of 0 or less is meaningless, and the loop below cannot correct it: the
    # `step >= n_frames` guard is False for 0, so it fell through to `range(0, n, 0)`,
    # which raises "range() arg 3 must not be zero". A negative step produced an empty
    # list and then IndexError on `indices[-1]`. frame_step is user-editable, so clamp to
    # "every frame" -- the interpretation the value is closest to.
    step = max(int(frame_step), 1)
    while step >= n_frames:
        step = max(step // 5, 1)
        if step == 1:
            break
    indices = list(range(0, n_frames, step))
    if indices[-1] != n_frames - 1:
        indices.append(n_frames - 1)
    return indices


def summarise_meshes(meshes: List[NucleusMesh]) -> MeshResults:
    """Reduce per-timepoint meshes to the one row the CSV holds.

    Averaged over analysed timepoints, matching how every other volumetric metric is
    reduced. Curvature is optional (``mesh_curvature``), so those fields stay NaN when
    it was not computed rather than reporting a mean of nothing.
    """
    def mean_of(values) -> float:
        array = np.asarray([v for v in values], dtype=np.float64)
        finite = array[np.isfinite(array)]
        return float(finite.mean()) if finite.size else np.nan

    if not meshes:
        return MeshResults()

    geometries = [m.geometry for m in meshes]
    curvatures = [m.curvature for m in meshes if m.curvature is not None]

    return MeshResults(
        mesh_volume=mean_of([g.volume_um3 for g in geometries]),
        surface_area=mean_of([g.surface_area_um2 for g in geometries]),
        sphericity=mean_of([g.sphericity for g in geometries]),
        equivalent_radius=mean_of([g.equivalent_sphere_radius_um for g in geometries]),
        height=mean_of([g.height_um for g in geometries]),
        aspect_ratio=mean_of([g.aspect_ratio for g in geometries]),
        volume_ratio=mean_of([g.volume_ratio for g in geometries]),
        solidity=mean_of([g.solidity for g in geometries]),
        mean_curvature=mean_of([c.mean_curvature for c in curvatures]),
        invagination_ratio=mean_of([c.invagination_ratio for c in curvatures]),
        concave_ratio=mean_of([c.concave_ratio for c in curvatures]),
    )


def _dim_flag(volumes: np.ndarray, frame_indices: List[int]) -> int:
    """Flag digit 1 -- too little signal for binarization to mean anything.

    A channel is dim when ``2/e * mean <= min``: the darkest voxel is already a sizeable
    fraction of the average, so there is barely a foreground to separate. ``utils`` holds
    two identical copies of that formula (``check_channel_dim`` and
    ``reader.check_first_frame_dim``, the one ``core.pipeline`` uses); this calls the
    former, so the definitions can still drift apart until they are unified. Kept as a
    call rather than a fourth copy.

    Judged on the volume as **acquired**, before ``_prepare_geometry``. That is
    deliberate: isotropic resampling interpolates, which raises the minimum, and cropping
    to a mask discards dark background -- both push ``min/mean`` upward, i.e. toward
    "dim". Measuring the acquired data keeps the 2D and 3D flags answering the same
    question about the same pixels. The t range is still respected, since ``stack`` has
    already been restricted.

    Flags but does not skip, unlike the 2D path: a volumetric run is one file rather than
    one channel of many, so aborting would discard the whole analysis over a warning the
    flag already carries.
    """
    from utils import check_channel_dim

    if volumes.size == 0 or not frame_indices:
        return 0
    index = min(frame_indices[0], volumes.shape[0] - 1)
    return 1 if bool(check_channel_dim(volumes[index])) else 0


#: Values ``mesh_aggregation`` accepts, mapped to whether the pipeline implements them.
_MESH_AGGREGATION_IMPLEMENTED = {"largest": True, "mean": False, "total": False}


def _check_mesh_aggregation(mode: str) -> None:
    """Reject a mesh_aggregation the pipeline cannot honour.

    ``mean`` and ``total`` are described in the config as meshing every object and
    averaging the intensive metrics while summing the extensive ones. The per-object
    mesher that would supply them (``analysis/volumetric/mesh_field.py``) exists and is
    tested, but nothing calls it, so both settings previously produced ``largest``
    silently -- with no GUI control to expose the discrepancy, and no column that would
    reveal it. Failing here converts a wrong number into a stopped run.
    """
    key = str(mode).strip().lower()
    if _MESH_AGGREGATION_IMPLEMENTED.get(key):
        return
    if key in _MESH_AGGREGATION_IMPLEMENTED:
        raise ValueError(
            f"mesh_aggregation={mode!r} is not implemented: the mesh family would "
            f"silently describe only the largest connected component. Set "
            f"mesh_aggregation='largest' to accept that explicitly. Per-object meshing "
            f"exists in analysis/volumetric/mesh_field.py but has no pipeline call site."
        )
    raise ValueError(
        f"Unknown mesh_aggregation {mode!r}. Valid values: "
        f"{', '.join(sorted(_MESH_AGGREGATION_IMPLEMENTED))}."
    )


def _export_meshes(filepath: str, meshes: List[NucleusMesh]) -> None:
    """Write one OBJ per analysed timepoint, beside the data it came from.

    The GUI's *Export Mesh as .OBJ* switch is a bool with nowhere to put the files, so
    the destination is derived rather than configured: a folder next to the input, named
    after it, mirroring how the 2D pipeline puts ``<file> BARCODE Output/`` beside the
    file it analysed. Deliberately not a fixed path and never the working directory --
    outputs belong with the dataset that produced them.

    Reported rather than raised, matching how the rest of this branch treats an optional
    output: a read-only drive should not lose a run whose metrics already succeeded.
    """
    from analysis.volumetric.mesh import write_obj

    stem = os.path.splitext(os.path.basename(filepath))[0]
    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(filepath)), f"{stem} BARCODE Meshes")
    try:
        written = [
            write_obj(os.path.join(out_dir, f"{stem}_t{mesh.frame_index}.obj"),
                      mesh.vertices_um, mesh.faces)
            for mesh in meshes
        ]
    except Exception as exc:  # noqa: BLE001 - see below
        # Deliberately broad. A read-only drive raises OSError, but a malformed face
        # array raises ValueError or IndexError, and none of them should cost a run whose
        # metrics already succeeded -- the OBJ is an optional side artefact, not a result.
        print(f"  mesh export failed ({type(exc).__name__}): {exc}", flush=True)
        return
    if written:
        print(f"  wrote {len(written)} mesh(es) to {out_dir}", flush=True)


def summarise_components(detail) -> ComponentResults:
    """Reduce the per-timepoint size distributions to one row.

    Sizes are expressed as a fraction of the analysed field, matching the binarization
    family, so a run is comparable with another of a different crop size.
    """
    voxels = float(detail.voxel_count) or np.nan

    def mean_of(values, scale=1.0):
        array = np.asarray(values, dtype=np.float64)
        finite = array[np.isfinite(array)]
        return float(finite.mean() * scale) if finite.size else np.nan

    return ComponentResults(
        count=mean_of(detail.island_counts),
        size_sd=mean_of(detail.size_sds, 1.0 / voxels),
        size_skew=mean_of(detail.size_skews),
        size_median=mean_of(detail.size_medians, 1.0 / voxels),
    )


def resolve_frame_interval(stack: VolumeStack, config: VolumetricConfig) -> float:
    """Seconds between consecutive timepoints, and say out loud where it came from.

    Speed is a distance divided by this number, so getting it wrong rescales two of the
    twenty-five metrics without any other symptom. The configured value wins; falling
    back to the file is announced, because ImageJ's ``finterval`` frequently describes
    the z acquisition rather than the time axis and a silent 1.0 turns "um/s" into
    "um/frame" while still printing a units label that says seconds.
    """
    if config.frame_interval_s > 0:
        return float(config.frame_interval_s)

    # `timing_from_file` distinguishes a real tag from the reader's 1.0 fallback. Testing
    # the value alone could not: the fallback IS 1.0, so a file carrying no timing at all
    # was announced as "1 s from the file" — precisely the claim this message exists to
    # prevent, and Speed was then wrong by exactly the factor it was meant to flag.
    from_file = float(stack.exposure_time_s or 0.0)
    have_file_value = bool(getattr(stack, "timing_from_file", False)) and from_file > 0
    source = "from the file's ImageJ finterval tag" if have_file_value else "(defaulted; the file states no timing)"
    print(
        f"  flow: no frame interval configured; using {from_file or 1.0:g} s {source}. "
        f"If that is not the true spacing between timepoints, set Frame Interval — "
        f"Speed scales inversely with it.",
        flush=True,
    )
    return from_file or 1.0


def _load_masks(
    stack: VolumeStack, config: VolumetricConfig
) -> Optional[Tuple[np.ndarray, List[str], float]]:
    """Load one mask per timepoint, stacked as ``(T, mz, my, mx)``.

    A grouped series carries its constituent file paths in ``metadata_source``; each
    timepoint gets its own mask, since the object moves and changes shape over time.
    """
    # Validate against the acquired depth, not the analysed sub-range; the mask covers
    # the whole acquisition and is cropped to match further down.
    acquired_z = stack.n_slices_acquired or stack.n_slices
    shape_zyx = (acquired_z,) + tuple(stack.data.shape[2:])
    paths = stack.metadata_source.get("paths") or [stack.source_path]

    masks, mask_paths, spacing = [], [], None
    for path in paths:
        loaded = load_segmentation(
            path, shape_zyx, stack.z_step_um, stack.xy_step_um, config
        )
        if loaded is None:
            return None
        mask, mask_path, mask_spacing = loaded
        if masks and mask.shape != masks[0].shape:
            raise ValueError(
                f"Mask {os.path.basename(mask_path)} has shape {mask.shape} but "
                f"{os.path.basename(mask_paths[0])} has {masks[0].shape}; masks in one "
                f"series must share a grid."
            )
        masks.append(mask)
        mask_paths.append(mask_path)
        spacing = mask_spacing

    stacked = np.stack(masks)
    if stack.z_range:
        m0, m1 = _mask_z_slice_for_range(
            stacked.shape[1], acquired_z, stack.z_range[0], stack.z_range[1]
        )
        stacked = stacked[:, m0:m1]
    return stacked, mask_paths, spacing


def _mask_z_slice_for_range(
    mask_slices: int, acquired_slices: int, z0: int, z1: int
) -> Tuple[int, int]:
    """Which mask slices span the same physical depth as acquired slices ``[z0, z1)``.

    The mask is routinely on a *finer* z grid than the acquisition (250 planes at
    0.065 um for a 54-slice stack at 0.3 um), so it cannot be sliced with acquired
    indices. Testing ``mask_slices == acquired_slices`` and skipping the crop when they
    differ -- which is what this used to do -- left the mask spanning the whole
    acquisition while the image had already been cut down. ``prepare_volume`` then
    resamples by physical coordinates from origin 0, so the sub-range was planted at
    z = 0 and the two no longer described the same slab: on a 54/250 pair keeping 20
    slices put the image at isotropic z 0..89 and its own mask at 100..150, with no
    overlap at all and every mask-gated metric measuring background.

    Both grids are node-aligned over the same extent (the convention
    ``resample._reference_shape_for_spacing`` and ``match_mask_to_image_grid`` already
    use), so acquired slice ``i`` sits at mask slice ``round(i * (mz-1)/(nz-1))``.
    """
    if mask_slices == acquired_slices:
        return int(z0), int(z1)
    if acquired_slices < 2 or mask_slices < 2:
        return 0, int(mask_slices)

    scale = (mask_slices - 1) / (acquired_slices - 1)
    start = int(round(z0 * scale))
    stop = int(round((z1 - 1) * scale)) + 1
    return max(0, start), min(int(mask_slices), max(stop, start + 1))


def _prepare_geometry(
    stack: VolumeStack, config: VolumetricConfig
) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[float, float, float], Dict, Optional[List[str]]]:
    """Load any masks and put image + masks on one common grid.

    Returns ``(volumes, masks, spacing_zyx, info, mask_paths)`` where ``volumes`` is
    ``(T, Z, Y, X)`` and ``masks`` is either None or the same shape.

    For a time series every timepoint is cropped to the **union** of the per-frame mask
    bounding boxes, not to its own. Per-frame cropping would give each timepoint a
    different array shape (they cannot then be stacked) and, worse, a different
    denominator for every "fraction of volume" metric -- so those metrics would not be
    comparable across the very time axis they are meant to describe.
    """
    z_um, xy_um = stack.z_step_um, stack.xy_step_um
    n_timepoints = stack.data.shape[0]

    loaded = _load_masks(stack, config)
    if loaded is None:
        # Isotropic resampling is defined against the mask's grid, so with no segmentation
        # there is nothing to resample onto and the data is analysed as acquired. Say which
        # of the two it was: recording "not_requested" while make_isotropic was True (the
        # default) made the provenance assert the user had not asked, and 3D connectivity
        # and shape metrics on a 4.6x anisotropic grid are exactly what that setting exists
        # to prevent.
        reason = "no_segmentation" if config.make_isotropic else "not_requested"
        if config.make_isotropic:
            print(
                "  make_isotropic is on but no segmentation resolved, so there is no "
                "isotropic grid to resample onto; analysing the acquired anisotropic "
                "voxels. 3D shape and connectivity metrics assume equal spacing.",
                flush=True,
            )
        return stack.data, None, (z_um, xy_um, xy_um), {"isotropic": reason}, None

    masks, mask_paths, mask_spacing = loaded
    if len(mask_paths) == 1 and n_timepoints > 1:
        # One mask, many timepoints: correct for a static object, wrong for a moving one,
        # and the two are indistinguishable in the output. Only a grouped per-file series
        # carries one mask per timepoint; a single TZYX hyperstack resolves exactly one.
        print(
            f"  one segmentation resolved for {n_timepoints} timepoints; it is applied to "
            f"all of them. If the object moves or deforms, export one mask per timepoint.",
            flush=True,
        )

    if not config.make_isotropic:
        # Bring the masks down onto the image's own grid by nearest-neighbour index
        # mapping. Exact for a boolean mask and far cheaper than upsampling the image.
        if masks.shape[1] != stack.data.shape[1]:
            idx = np.clip(
                np.round(np.linspace(0, masks.shape[1] - 1, stack.data.shape[1])).astype(int),
                0, masks.shape[1] - 1,
            )
            masks = masks[:, idx]
        if masks.shape[0] == 1 and n_timepoints > 1:
            masks = np.repeat(masks, n_timepoints, axis=0)
        return stack.data, masks, (z_um, xy_um, xy_um), {"isotropic": "skipped"}, mask_paths

    # The analysed field is the ACQUIRED field of view. Cropping to the mask's bounding
    # box would give each file its own denominator for every "fraction of volume"
    # metric, so an object shrinking and the box tightening around it would be
    # indistinguishable -- and on a per-file run all 15 boxes really are different.
    # An explicit z range is the only intended way to analyse less than the full field,
    # and it is applied to the stack before this point.
    # Note prepare_volume takes spacing as (x, y, z) while the arrays are (Z, Y, X).
    from analysis.volumetric.resample import prepare_volume

    union_mask = masks.any(axis=0).astype(np.uint8)
    if not union_mask.any():
        # This used to be caught downstream by `crop_bbox is None`, which `_crop_to_mask_bbox`
        # returned for an all-zero mask. With cropping off by default that branch always
        # builds a full-field box, so the guard could never fire and an entirely empty
        # segmentation flowed through to the metrics, surfacing as an unexplained NaN
        # island count rather than as the configuration error it is.
        raise ValueError(
            "The segmentation is empty at every analysed timepoint. Check the "
            "segmentation template and the z/t ranges."
        )

    # Every timepoint in ONE call. prepare_volume already takes a dict of channels and
    # puts them all on the mask's grid, so calling it per timepoint resampled the same
    # union mask identically T times (15x on this dataset, a 250^3 nearest-neighbour pass
    # each) and threw away all but the last. The results are unchanged.
    keys = [f"t{t}" for t in range(n_timepoints)]
    images_iso, union_iso, spacing_iso, info = prepare_volume(
        images={key: stack.data[t] for t, key in enumerate(keys)},
        image_spacings={key: (xy_um, xy_um, z_um) for key in keys},
        mask=union_mask,
        mask_spacing=(mask_spacing, mask_spacing, mask_spacing),
        crop_padding=config.crop_padding_vox,
        crop_to_mask=getattr(config, "crop_to_mask", False),
    )
    volumes = np.stack([images_iso[key] for key in keys])
    spacing_zyx = (spacing_iso[2], spacing_iso[1], spacing_iso[0])

    # Put the per-frame masks on that same cropped grid. When the masks were already
    # isotropic prepare_volume only resampled, so the bounding box indexes the masks'
    # own grid and slicing is exact. Otherwise they must be resampled the same way the
    # union was, with nearest-neighbour -- which maps each output voxel to exactly one
    # input voxel and so carries instance labels across unchanged, where any
    # interpolating kernel would average neighbouring labels into new, meaningless ones.
    bbox = info.get("crop_bbox")
    if bbox is None:
        raise ValueError("Segmentation is empty; cannot establish a crop box.")

    if info.get("isotropic") == "already_isotropic":
        z0, z1 = bbox["z"]
        y0, y1 = bbox["y"]
        x0, x1 = bbox["x"]
        masks_iso = masks[:, z0:z1, y0:y1, x0:x1]
    else:
        from analysis.volumetric.resample import _resample_array_to_reference
        import SimpleITK as sitk

        target = (spacing_iso[0], spacing_iso[1], spacing_iso[2])
        source = (mask_spacing, mask_spacing, mask_spacing)
        reference_shape = union_iso.shape
        # Where that reference grid starts, in physical coordinates. Zero unless the
        # volume was cropped, in which case the crop's corner -- the resampler works in
        # physical space, so without an origin the masks would be sampled from the volume
        # origin onto a box that begins somewhere else, and the shape check below cannot
        # detect a pure offset.
        origin = (
            bbox["x"][0] * target[0],
            bbox["y"][0] * target[1],
            bbox["z"][0] * target[2],
        )
        # int32, not uint8: an instance segmentation routinely has more than 255
        # objects -- a confluent Cellpose field has thousands -- and uint8 would wrap
        # them silently, merging unrelated cells into one label.
        masks_iso = np.stack([
            _resample_array_to_reference(
                m.astype(np.int32), source, reference_shape, target,
                sitk.sitkNearestNeighbor, origin,
            )
            for m in masks
        ])

    # Preserve the label dtype. Casting to bool here used to undo the loader's label
    # preservation, collapsing every instance into one foreground blob -- which is
    # invisible in the metrics for a single object and silently wrong for a field of
    # them. Consumers that want a binary field say so themselves with .astype(bool).
    masks_iso = np.asarray(masks_iso)
    if masks_iso.dtype != bool and not np.issubdtype(masks_iso.dtype, np.integer):
        masks_iso = masks_iso.astype(bool)
    if masks_iso.shape[0] == 1 and n_timepoints > 1:
        masks_iso = np.repeat(masks_iso, n_timepoints, axis=0)
    if masks_iso.shape[1:] != volumes.shape[1:]:
        raise ValueError(
            f"Mask grid {masks_iso.shape[1:]} does not match the analysed image grid "
            f"{volumes.shape[1:]} after resampling."
        )

    info = dict(info)
    info["common_crop"] = n_timepoints > 1
    return volumes, masks_iso, spacing_zyx, info, mask_paths


def run_volumetric_analysis(
    filepath: str,
    config: BarcodeConfig,
    channel: int = 0,
    stack: Optional[VolumeStack] = None,
) -> Tuple[ChannelResults, VolumetricRunDetail]:
    """Run the volumetric pipeline and return CSV-ready results.

    Pass ``stack`` to analyse an already-assembled time series (see
    ``timelapse.read_series``); otherwise ``filepath`` is read as a single volume.
    """
    vcfg = config.volumetric

    if stack is None:
        stack = read_volume(
            filepath,
            channel=channel,
            z_step_um=vcfg.z_step_um or None,
            xy_step_um=vcfg.xy_step_um or None,
            axes_override=getattr(vcfg, "axes_override", "") or None,
        )
    # Same depth restriction as xyz: a volume padded with empty slices reports a
    # different shape and a diluted intensity distribution.
    stack = apply_t_range(stack, vcfg)
    stack = apply_z_range(stack, vcfg)

    volumes, masks, spacing_zyx, info, mask_paths = _prepare_geometry(stack, vcfg)
    frame_indices = select_frame_indices(volumes.shape[0], vcfg.frame_step)

    detail = VolumetricRunDetail(
        stack=stack,
        mask_paths=mask_paths,
        spacing_zyx_um=spacing_zyx,
        shape_zyx=tuple(int(v) for v in volumes.shape[1:]),
        resample_info=info,
        frame_indices=frame_indices,
    )
    results = ChannelResults(filepath=filepath, channel=channel)
    results.z_range_flag = 1 if (stack.z_range or stack.t_range) else 0
    # Acquired voxels, not the resampled/cropped ones -- see _dim_flag.
    results.dim_channel_flag = _dim_flag(stack.data, frame_indices)

    if config.modules.image_binarization:
        results.binarization, detail.binarization = analyze_binarization_3d(
            volumes, spacing_zyx, vcfg, frame_indices, masks
        )
        if vcfg.enable_component_stats:
            results.components = summarise_components(detail.binarization)

    if config.modules.intensity_distribution:
        results.intensity, detail.intensity = analyze_intensity_3d(
            volumes, vcfg, frame_indices,
            masks if vcfg.intensity_use_mask else None,
        )

    if vcfg.mesh_enabled:
        # A setting that silently does nothing is worse than one that is absent: the
        # mesh family would describe the largest component while the config says it
        # describes every object, and no output distinguishes the two. Raised rather
        # than reported because it changes what the numbers MEAN, unlike a missing
        # segmentation, which merely leaves them empty.
        _check_mesh_aggregation(getattr(vcfg, "mesh_aggregation", "largest"))

        # Meshing describes the segmented object, so it needs a segmentation: a mesh
        # of an intensity threshold would not be the nucleus. Reported, not raised,
        # so one misconfigured file does not abort a batch.
        if masks is None:
            print(
                "Meshing is enabled but no segmentation resolved; skipping the mesh.",
                flush=True,
            )
        else:
            try:
                detail.meshes = mesh_series(
                    masks, spacing_zyx, frame_indices, vcfg
                )
                results.mesh = summarise_meshes(detail.meshes)
                if vcfg.mesh_export_obj:
                    _export_meshes(filepath, detail.meshes)
            except MeshingError as exc:
                print(f"Meshing failed: {exc}", flush=True)

    if vcfg.enable_packing_topology:
        # Needs a label volume. Reported rather than raised, matching how meshing
        # handles a missing segmentation: silently falling back to connectivity would
        # report a contact number of 0 for a confluent field, which reads as a
        # measurement rather than a misconfiguration.
        if masks is None:
            print("Packing topology needs a segmentation; skipping.", flush=True)
        elif masks.dtype == bool or int(np.count_nonzero(np.unique(masks))) < 2:
            print(
                "Packing topology needs an integer label volume with more than one "
                "object; the mask has none, so the family is left empty. Supply an "
                "instance segmentation and set object_partition='labels'.",
                flush=True,
            )
        else:
            per_frame, frame_details = [], []
            for frame_idx in frame_indices:
                frame_result, frame_detail = packing_topology(masks[frame_idx], vcfg)
                per_frame.append(frame_result)
                frame_details.append(frame_detail)
            results.packing = summarise_packing(frame_details, per_frame)
            detail.packing = frame_details
            print(f"  {frame_details[0].describe()}", flush=True)

    # Stream A's families: computed there, populated here, because the orchestration
    # files belong to stream B. The writer and barcode detect them automatically.
    if vcfg.enable_intensity_magnitude:
        results.intensity_magnitude = analyze_intensity_magnitude(
            volumes, spacing_zyx, frame_indices,
            masks if vcfg.intensity_use_mask else None)

    if vcfg.enable_slice_profile or vcfg.enable_curvature_range or vcfg.enable_mask_intensity:
        _add_optional_families(results, detail, volumes, masks, spacing_zyx,
                               frame_indices, vcfg)

    if vcfg.record_range_columns:
        results.ranges = build_range_results(stack)

    if config.modules.optical_flow:
        # Unlike the other branches, flow needs a contiguous window of 6*t_sigma+1
        # volumes centred on each analysed timepoint, so the guard is on the series
        # length rather than on how many frames were selected. analyze_optical_flow_3d
        # reports the shortfall itself and returns empty results.
        results.flow, detail.flow = analyze_optical_flow_3d(
            volumes,
            spacing_zyx,
            resolve_frame_interval(stack, vcfg),
            vcfg,
            frame_indices,
            masks,
        )

    return results, detail


def run_volumetric_timelapse(
    filepaths: List[str],
    config: BarcodeConfig,
    fail_file_loc: str,
) -> List[ChannelResults]:
    """Group per-timepoint files into series and analyse each as one time-lapse.

    Returns one ``ChannelResults`` per series rather than per file, so the change
    metrics that are NaN in single-volume mode become meaningful.
    """
    from analysis.volumetric.timelapse import group_timelapse, read_series

    warn_if_channels_dropped(config)

    vcfg = config.volumetric
    groups, unmatched = group_timelapse(filepaths, vcfg.timelapse_regex)

    if unmatched:
        print(
            f"{len(unmatched)} file(s) did not match the time-lapse pattern "
            f"{vcfg.timelapse_regex!r} and were skipped: "
            f"{', '.join(os.path.basename(p) for p in unmatched[:5])}"
            f"{' ...' if len(unmatched) > 5 else ''}",
            flush=True,
        )
    if not groups:
        print("No time-lapse series found; nothing to analyse.", flush=True)
        return []

    all_results = []
    for index, group in enumerate(groups, start=1):
        print(f"Series {index} of {len(groups)} -- {group.describe()}", flush=True)
        try:
            stack = read_series(
                group,
                channel=config.channels.selected_channel,
                z_step_um=vcfg.z_step_um or None,
                xy_step_um=vcfg.xy_step_um or None,
                axes_override=getattr(vcfg, "axes_override", "") or None,
            )
            results, detail = run_volumetric_analysis(
                group.paths[0], config, config.channels.selected_channel, stack=stack
            )
        except Exception as exc:
            print(f"  failed: {type(exc).__name__}: {exc}", flush=True)
            with open(fail_file_loc, "a", encoding="utf-8") as log_file:
                log_file.write(traceback.format_exc())
                log_file.write(f"Series: {group.series}, Exception: {exc}\n")
            continue

        print(
            f"  {detail.stack.n_timepoints} of {len(group)} timepoints"
            + (f" (t[{detail.stack.t_range[0]}:{detail.stack.t_range[1]}])"
               if detail.stack.t_range else "")
            + f" -> {detail.shape_zyx} @ "
            f"{tuple(round(s, 4) for s in detail.spacing_zyx_um)} um"
            f"{' (common crop box)' if detail.resample_info.get('common_crop') else ''}",
            flush=True,
        )
        all_results.append(results)

    return all_results


def run_volumetric_pipeline(
    filepath: str,
    config: BarcodeConfig,
    in_config,
    fail_file_loc: str,
    count: int,
    total: int,
) -> Tuple[List[ChannelResults], int]:
    """Entry point matching ``core.pipeline.process_single_file``'s contract."""
    if total != 1:
        print(f"File {count} of {total}")
        print(filepath)
        count += 1

    warn_if_channels_dropped(config)

    try:
        results, detail = run_volumetric_analysis(
            filepath, config, channel=config.channels.selected_channel
        )
    except Exception as exc:
        with open(fail_file_loc, "a", encoding="utf-8") as log_file:
            log_file.write(traceback.format_exc())
            log_file.write(f"File: {filepath}, Module: Volumetric, Exception: {exc}\n")
        raise

    print(f"Volumetric: {detail.shape_zyx} @ {tuple(round(s, 4) for s in detail.spacing_zyx_um)} um")
    return [results], count
