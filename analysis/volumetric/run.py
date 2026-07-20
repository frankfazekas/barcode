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
)
from analysis.volumetric.flow import VolumetricFlowDetail, analyze_optical_flow_3d
from analysis.volumetric.intensity import (
    VolumetricIntensityDetail,
    analyze_intensity_3d,
)
from analysis.volumetric.mesh import MeshingError, NucleusMesh, mesh_series
from analysis.volumetric.reader import VolumeStack, apply_z_range, read_volume
from analysis.volumetric.segmentation import load_segmentation
from core import BarcodeConfig, ChannelResults, VolumetricConfig
from core.results import ComponentResults, MeshResults


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
    frame_indices: List[int] = field(default_factory=list)


def select_frame_indices(n_frames: int, frame_step: int) -> List[int]:
    """Timepoints to analyse.

    The 2D helper (``utils.find_analysis_frames``) divides its step by 5 until it is
    below the series length, which at ``n_frames == 1`` yields a float step and makes
    ``range`` raise ``TypeError``. A single-volume series is the normal case here, so
    it is handled explicitly.
    """
    if n_frames <= 1:
        return [0]
    step = int(frame_step)
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
        volume_ratio=mean_of([g.volume_ratio for g in geometries]),
        mean_curvature=mean_of([c.mean_curvature for c in curvatures]),
        invagination_ratio=mean_of([c.invagination_ratio for c in curvatures]),
        concave_ratio=mean_of([c.concave_ratio for c in curvatures]),
    )


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

    from_file = float(stack.exposure_time_s or 0.0)
    print(
        f"  flow: no frame interval configured; using {from_file or 1.0:g} s from the "
        f"file{'' if from_file else ' (defaulted)'}. If that is not the true spacing "
        f"between timepoints, set Frame Interval — Speed scales inversely with it.",
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
    shape_zyx = stack.data.shape[1:]
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

    return np.stack(masks), mask_paths, spacing


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
        return stack.data, None, (z_um, xy_um, xy_um), {"isotropic": "not_requested"}, None

    masks, mask_paths, mask_spacing = loaded

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

    # prepare_nucleus crops to the bounding box of whatever mask it is handed, so give
    # it the union: every timepoint then lands on an identical grid.
    # Note it takes spacing as (x, y, z) while the arrays are (Z, Y, X).
    from analysis.volumetric.resample import prepare_nucleus

    union_mask = masks.any(axis=0).astype(np.uint8)

    volumes, info = [], {}
    for t in range(n_timepoints):
        images_iso, union_iso, spacing_iso, info = prepare_nucleus(
            images={"image": stack.data[t]},
            image_spacings={"image": (xy_um, xy_um, z_um)},
            mask=union_mask,
            mask_spacing=(mask_spacing, mask_spacing, mask_spacing),
            crop_padding=config.crop_padding_vox,
        )
        volumes.append(images_iso["image"])

    volumes = np.stack(volumes)
    spacing_zyx = (spacing_iso[2], spacing_iso[1], spacing_iso[0])

    # Put the per-frame masks on that same cropped grid. When the masks were already
    # isotropic prepare_nucleus only cropped, so the bounding box indexes the masks'
    # own grid and slicing is exact. Otherwise they must be resampled the same way the
    # union was, with nearest-neighbour so the mask stays binary.
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
        masks_iso = np.stack([
            _resample_array_to_reference(
                m.astype(np.uint8), source, reference_shape, target, sitk.sitkNearestNeighbor
            )
            for m in masks
        ])

    masks_iso = np.asarray(masks_iso, dtype=bool)
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
        )
    # Same depth restriction as xyz: a volume padded with empty slices reports a
    # different shape and a diluted intensity distribution.
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
            except MeshingError as exc:
                print(f"Meshing failed: {exc}", flush=True)

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
            f"  {len(group)} timepoints -> {detail.shape_zyx} @ "
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
