"""3D optical flow — placeholder for the volumetric flow implementation.

Deliberately unimplemented. The signature mirrors ``analysis/optical_flow.py``'s
``analyze_optical_flow`` so a real implementation can be dropped in without touching
any caller.

Notes for whoever fills this in:

* ``cv.calcOpticalFlowFarneback`` — which the 2D branch uses — is 2D only. There is no
  drop-in 3D equivalent in OpenCV.
* ``utils/optical_flow.py``'s ``divergence`` and ``curl`` are 2D operators. Divergence
  generalises directly (add the ``d/dz`` term), but **curl becomes a 3-vector in 3D**,
  so ``Metrics.CURL`` needs a magnitude convention chosen and documented.
* Velocity correlation length should bin on physical distance and cap at the smallest
  physical half-extent, the same way ``binarization.spatial_volume_autocorrelation``
  does — with anisotropic voxels that bound is usually z, not y.
* At a single timepoint there are no frame pairs at all; callers skip this module
  entirely rather than relying on it to no-op.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from core import FlowResults, VolumetricConfig

FLOW_3D_AVAILABLE = False


def analyze_optical_flow_3d(
    volume_series: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
    exposure_time_s: float,
    config: VolumetricConfig,
    frame_indices: List[int],
) -> FlowResults:
    """Return empty flow results and say so, rather than fabricating numbers."""
    print(
        "3D optical flow is not implemented; skipping the flow branch. "
        "Structural and intensity metrics are unaffected.",
        flush=True,
    )
    return FlowResults()
