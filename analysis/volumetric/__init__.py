"""Volumetric BARCODE — an isolated side-car pipeline.

This package is deliberately self-contained: nothing here imports from ``analysis``
(the 2D branches) or mutates any of the shared 2D helpers. It reuses the dataclasses
in ``core.results`` read-only, so the existing CSV writer, barcode PNG and aggregator
keep working untouched.

The 2D pipeline is reached exactly as before; a single config-gated branch in
``core.pipeline.process_single_file`` is the only hand-off point.
"""

from analysis.volumetric.curvature import CurvatureResults, analyze_curvature
from analysis.volumetric.mesh import (
    MeshGeometry,
    MeshingError,
    ObjectMesh,
    mesh_object,
    write_obj,
)
from analysis.volumetric.reader import VolumeStack, read_volume
from analysis.volumetric.segmentation import load_segmentation, resolve_segmentation_path

__all__ = [
    "VolumeStack",
    "read_volume",
    "load_segmentation",
    "resolve_segmentation_path",
    "CurvatureResults",
    "analyze_curvature",
    "MeshGeometry",
    "MeshingError",
    "ObjectMesh",
    "mesh_object",
    "write_obj",
]
