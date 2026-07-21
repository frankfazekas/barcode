#!/usr/bin/env python3
"""Measure BARCODE's mesh metrics against shapes whose answers are known exactly.

Every other check in this project compares BARCODE against a file somebody else wrote.
That cannot validate curvature, because no dataset publishes per-object curvature, and
it cannot cleanly validate surface area either. Mathematics can do both: a digitized
sphere, ellipsoid and torus have closed-form volume, surface area, sphericity and mean
curvature, so the error is computable rather than merely comparable.

It is also the check the code itself asks for. ``tests/test_mesh.py`` already meshes a
ball and asserts against 4/3 pi r^3, but only loosely, and says why:

    "The default chain decimates hard, so this is a shape check, not a precision one:
     a closed sphere-like surface within 25% of the analytic values."

A +-25% band on volume and surface area sits under every mesh metric and has never been
narrowed to a number. This narrows it, and says what moves it:

  * object radius in voxels   -- how big must a thing be before it can be measured?
  * anisotropy                -- masks arrive on anisotropic grids and are resampled
  * ``maxrad``                -- the decimation knob the comment above implicates

Nothing in ``analysis/`` is changed on the strength of these results; the point is to
publish the accuracy, not to silently retune the defaults.

    python scripts/validate_phantoms.py --sweep all
    python scripts/validate_phantoms.py --sweep maxrad --radius 16
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from analysis.volumetric.curvature import analyze_curvature
from analysis.volumetric.mesh import MeshingError, mesh_nucleus
from scripts._staging import mask_z_to_isotropic

DEFAULT_OUT = r"L:\FF\Hackathon\full_datasets\_validation"


# ------------------------------------------------------------------ the phantoms

@dataclass
class Phantom:
    """A digitized shape and the exact values it is supposed to reproduce.

    ``truth`` holds only quantities with a closed form. Anything that has to be
    approximated is left out rather than compared against another approximation --
    the whole value of this test is that the reference is not in doubt.
    """

    name: str
    mask: np.ndarray
    spacing_um: float
    truth: Dict[str, float]
    note: str = ""


def _grid(shape: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z, y, x = (np.arange(n) - (n - 1) / 2.0 for n in shape)
    return np.meshgrid(z, y, x, indexing="ij")


def sphere(radius_vox: float, spacing_um: float = 0.1, pad: int = 6) -> Phantom:
    """A ball. Volume 4/3 pi r^3, area 4 pi r^2, sphericity 1, mean curvature 1/r.

    The sphere is the one shape whose curvature is uniform, which makes it the only
    clean curvature reference: BARCODE's mean curvature is an area-weighted mean over
    the surface *excluding* top and bottom faces, and on a sphere that exclusion cannot
    bias the answer because every face carries the same value.
    """
    n = int(2 * radius_vox) + 2 * pad
    zz, yy, xx = _grid((n, n, n))
    mask = (zz ** 2 + yy ** 2 + xx ** 2) <= radius_vox ** 2
    r = radius_vox * spacing_um
    return Phantom(
        name=f"sphere r={radius_vox:g}vox",
        mask=mask.astype(np.uint8),
        spacing_um=spacing_um,
        truth={
            "volume_um3": 4 / 3 * math.pi * r ** 3,
            "surface_area_um2": 4 * math.pi * r ** 2,
            "sphericity": 1.0,
            "equivalent_radius_um": r,
            "mean_curvature_inv_um": 1.0 / r,
            "solidity": 1.0,
        },
        note=f"radius {r:.3f} um",
    )


def ellipsoid(a_vox: float, b_vox: float, c_vox: float,
              spacing_um: float = 0.1, pad: int = 6) -> Phantom:
    """A triaxial ellipsoid. Volume is exact; area uses Thomsen's approximation.

    Thomsen is accurate to about 1% for moderate axis ratios, so ``surface_area_um2``
    is recorded but flagged: a 1%-uncertain reference cannot resolve a 1% error. Volume
    and sphericity-from-volume remain exact.
    """
    n_z, n_y, n_x = (int(2 * v) + 2 * pad for v in (a_vox, b_vox, c_vox))
    zz, yy, xx = _grid((n_z, n_y, n_x))
    mask = (zz / a_vox) ** 2 + (yy / b_vox) ** 2 + (xx / c_vox) ** 2 <= 1.0
    a, b, c = (v * spacing_um for v in (a_vox, b_vox, c_vox))
    volume = 4 / 3 * math.pi * a * b * c
    p = 1.6075
    area = 4 * math.pi * (((a * b) ** p + (a * c) ** p + (b * c) ** p) / 3) ** (1 / p)
    return Phantom(
        name=f"ellipsoid {a_vox:g}x{b_vox:g}x{c_vox:g}vox",
        mask=mask.astype(np.uint8),
        spacing_um=spacing_um,
        truth={
            "volume_um3": volume,
            "surface_area_um2": area,
            # Sphericity is defined from volume and area, so it inherits Thomsen's ~1%.
            "sphericity": (math.pi ** (1 / 3) * (6 * volume) ** (2 / 3)) / area,
            "equivalent_radius_um": (3 * volume / (4 * math.pi)) ** (1 / 3),
            "solidity": 1.0,
        },
        note="area via Thomsen, ~1% uncertain",
    )


def torus(major_vox: float, minor_vox: float,
          spacing_um: float = 0.1, pad: int = 6) -> Phantom:
    """A torus: the shape that tests whether saddles are recognised at all.

    Volume 2 pi^2 R r^2 and area 4 pi^2 R r are exact. Two less obvious exact values
    make this the sharpest test in the file:

    * The area-weighted mean curvature integrates to 1/(2r), independent of R --
      ``int H dA`` = 2 pi^2 R over an area of 4 pi^2 R r.
    * Exactly the inner portion has negative Gaussian curvature, and its area fraction
      is ``0.5 - r/(pi R)``. That is a non-trivial number a mis-signed or mis-classified
      curvature cannot hit by accident, so it tests ``concave_ratio`` far better than a
      convex shape, where the right answer is simply zero.
    """
    n_xy = int(2 * (major_vox + minor_vox)) + 2 * pad
    n_z = int(2 * minor_vox) + 2 * pad
    zz, yy, xx = _grid((n_z, n_xy, n_xy))
    radial = np.sqrt(yy ** 2 + xx ** 2)
    mask = (radial - major_vox) ** 2 + zz ** 2 <= minor_vox ** 2
    R, r = major_vox * spacing_um, minor_vox * spacing_um
    return Phantom(
        name=f"torus R={major_vox:g} r={minor_vox:g}vox",
        mask=mask.astype(np.uint8),
        spacing_um=spacing_um,
        truth={
            "volume_um3": 2 * math.pi ** 2 * R * r ** 2,
            "surface_area_um2": 4 * math.pi ** 2 * R * r,
            "mean_curvature_inv_um": 1.0 / (2 * r),
            "saddle_area_fraction": 0.5 - r / (math.pi * R),
        },
        note=f"saddle fraction {0.5 - r / (math.pi * R):.4f}",
    )


def anisotropic_sphere(radius_vox: float, anisotropy: float,
                       spacing_um: float = 0.1, pad: int = 6) -> Phantom:
    """A sphere sampled on an anisotropic grid, then restored to isotropic.

    This is the shape the real pipeline actually sees. Masks are acquired with a coarse
    z step and staged back onto the isotropic grid by nearest-neighbour index mapping
    (``scripts/_staging.mask_z_to_isotropic``); the truth is unchanged because the
    physical sphere is unchanged, so any error is the round trip's.
    """
    base = sphere(radius_vox, spacing_um, pad)
    mask = base.mask
    step = max(1, int(round(anisotropy)))
    coarse = mask[::step]                                  # acquire every step-th slice
    # Restore through the SAME helper staging uses, not a copy of its index arithmetic.
    # The copy that used to live here carried the endpoint-anchored `linspace` mapping,
    # whose pitch is (m-1)/(c-1) rather than the intended step -- at 4x that inflates the
    # voxel count by ~7.5% on its own, which is most of what this sweep used to report as
    # an anisotropy effect. Calling the real helper means the test cannot drift from it.
    restored = mask_z_to_isotropic(coarse, spacing_um * step, spacing_um)
    return Phantom(
        name=f"sphere r={radius_vox:g}vox @ {step}x anisotropy",
        mask=restored, spacing_um=spacing_um, truth=base.truth,
        note=f"{coarse.shape[0]} acquired slices -> {restored.shape[0]}",
    )


# ------------------------------------------------------------------- measurement

@dataclass
class Measurement:
    phantom: str
    maxrad: float
    voxels: int
    smoothing: Optional[int] = None
    measured: Dict[str, float] = field(default_factory=dict)
    truth: Dict[str, float] = field(default_factory=dict)
    failed: str = ""

    def error(self, key: str) -> float:
        t, m = self.truth.get(key), self.measured.get(key)
        if t is None or m is None or not np.isfinite(m) or not t:
            return np.nan
        return (m - t) / t


def measure(phantom: Phantom, maxrad: float = 5.0, curvature: bool = True,
            smoothing: Optional[int] = None) -> Measurement:
    """``maxrad`` is the triangle-size bound: cgalsurf's radbound, in voxels.

    ``smoothing`` is the other half of the shrinkage. Laplacian-style smoothing pulls a
    convex surface inward, so it biases volume down independently of how coarsely the
    surface was triangulated -- which is why reducing maxrad alone cannot reach zero
    error. Left at None it keeps the pipeline default (10 iterations).
    """
    spacing = (phantom.spacing_um,) * 3
    out = Measurement(phantom=phantom.name, maxrad=maxrad,
                      voxels=int(phantom.mask.sum()), truth=dict(phantom.truth))
    extra = {} if smoothing is None else {"smoothing_iterations": smoothing}
    out.smoothing = smoothing
    try:
        mesh = mesh_nucleus(phantom.mask, spacing, maxrad=maxrad, solidity=True, **extra)
    except (MeshingError, Exception) as error:          # meshing is a native backend
        out.failed = f"{type(error).__name__}: {error}"
        return out

    geometry = mesh.geometry
    out.measured = {
        "volume_um3": geometry.volume_um3,
        "surface_area_um2": geometry.surface_area_um2,
        "sphericity": geometry.sphericity,
        "equivalent_radius_um": geometry.equivalent_sphere_radius_um,
        "solidity": geometry.mesh_solidity,
        # The voxel-counted volume is a SECOND, independent estimate of the same
        # quantity. Reporting both says which of BARCODE's two answers to prefer.
        "voxel_volume_um3": float(phantom.mask.sum()) * phantom.spacing_um ** 3,
    }
    out.truth.setdefault("voxel_volume_um3", phantom.truth.get("volume_um3", np.nan))

    if curvature:
        try:
            c = analyze_curvature(mesh.vertices_um, mesh.faces)
            out.measured["mean_curvature_inv_um"] = c.mean_curvature
            if c.concavity_classes is not None and c.k_gaussian_faces is not None:
                # Area fraction with negative Gaussian curvature, weighted by face area
                # so it is comparable with the analytic surface-area fraction.
                from analysis.volumetric.mesh import face_areas
                areas = face_areas(mesh.vertices_um, mesh.faces)
                negative = c.k_gaussian_faces < 0
                out.measured["saddle_area_fraction"] = float(
                    areas[negative].sum() / areas.sum())
        except Exception as error:
            out.failed = f"curvature: {type(error).__name__}: {error}"
    return out


# ------------------------------------------------------------------------ sweeps

KEYS = ("volume_um3", "voxel_volume_um3", "surface_area_um2", "sphericity",
        "equivalent_radius_um", "mean_curvature_inv_um", "saddle_area_fraction",
        "solidity")


def _report(title: str, results: List[Measurement], keys=KEYS) -> None:
    present = [k for k in keys if any(k in r.truth and k in r.measured for r in results)]
    print(f"\n{title}")
    print(f"{'phantom':<26}{'maxrad':>7}{'smooth':>7}{'voxels':>9}"
          + "".join(f"{k.split('_um')[0][:13]:>14}" for k in present))
    print("-" * (49 + 14 * len(present)))
    for r in results:
        # maxrad in the row, not just the CSV: a sweep over it prints otherwise
        # identical phantom names, and the reader cannot tell the rows apart.
        smooth = "dflt" if r.smoothing is None else str(r.smoothing)
        stem = f"{r.phantom:<26}{r.maxrad:>7g}{smooth:>7}{r.voxels:>9,}"
        if r.failed:
            print(f"{stem}   FAILED {r.failed[:60]}")
            continue
        line = stem
        for k in present:
            e = r.error(k)
            line += f"{e:>13.1%}" + " " if np.isfinite(e) else f"{'-':>14}"
        print(line)


def sweep_resolution(spacing: float, maxrad: float) -> List[Measurement]:
    return [measure(sphere(r, spacing), maxrad)
            for r in (4, 6, 8, 12, 16, 24, 32)]


def sweep_maxrad(radius: float, spacing: float) -> List[Measurement]:
    """Finer triangles, then no smoothing: the two causes of the shrinkage, separated.

    maxrad below 1 buys progressively less, because at that point the surface is
    triangulated finely enough that the remaining bias is the smoothing pass, not the
    triangulation. The last rows turn smoothing off to show what is left.
    """
    out = [measure(sphere(radius, spacing), m) for m in (0.5, 0.75, 1, 2, 3, 5, 8)]
    out += [measure(sphere(radius, spacing), m, smoothing=s)
            for m in (1, 2, 5) for s in (0, 3)]
    return out


def sweep_anisotropy(radius: float, spacing: float, maxrad: float) -> List[Measurement]:
    return [measure(anisotropic_sphere(radius, a, spacing), maxrad)
            for a in (1, 2, 4, 8, 12)]


def sweep_shapes(radius: float, spacing: float, maxrad: float) -> List[Measurement]:
    return [
        measure(sphere(radius, spacing), maxrad),
        measure(ellipsoid(radius, radius, radius * 2, spacing), maxrad),
        measure(ellipsoid(radius / 2, radius, radius * 2, spacing), maxrad),
        measure(torus(radius, radius / 3, spacing), maxrad),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Accuracy of the mesh metrics against closed-form shapes.")
    parser.add_argument("--sweep", default="all",
                        choices=["all", "resolution", "maxrad", "anisotropy", "shapes"])
    parser.add_argument("--radius", type=float, default=16.0,
                        help="sphere radius in voxels for the fixed-radius sweeps")
    parser.add_argument("--spacing", type=float, default=0.1, metavar="UM")
    parser.add_argument("--maxrad", type=float, default=5.0,
                        help="meshing decimation radius (core/config.py default: 5.0)")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    if os.path.abspath(args.out)[:2].upper() == "C:":
        print("Refusing to write results to C: -- that drive holds code, not data.")
        return 1

    sweeps: List[Tuple[str, Callable[[], List[Measurement]]]] = []
    if args.sweep in ("all", "resolution"):
        sweeps.append((
            f"RESOLUTION  sphere, maxrad {args.maxrad} -- how big must an object be?",
            lambda: sweep_resolution(args.spacing, args.maxrad)))
    if args.sweep in ("all", "maxrad"):
        sweeps.append((
            f"MAXRAD  sphere r={args.radius:g}vox -- what does the decimation cost?",
            lambda: sweep_maxrad(args.radius, args.spacing)))
    if args.sweep in ("all", "anisotropy"):
        sweeps.append((
            f"ANISOTROPY  sphere r={args.radius:g}vox, acquired coarsely in z",
            lambda: sweep_anisotropy(args.radius, args.spacing, args.maxrad)))
    if args.sweep in ("all", "shapes"):
        sweeps.append((
            "SHAPES  does it hold away from a sphere?",
            lambda: sweep_shapes(args.radius, args.spacing, args.maxrad)))

    everything: List[Measurement] = []
    for title, run in sweeps:
        results = run()
        _report(title, results)
        everything.extend(results)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "phantom_accuracy.csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["phantom", "maxrad", "smoothing", "voxels", "metric",
                         "truth", "measured", "relative_error", "failed"])
        for r in everything:
            for key in KEYS:
                if key in r.truth and (key in r.measured or r.failed):
                    writer.writerow([r.phantom, r.maxrad, r.smoothing, r.voxels, key,
                                     f"{r.truth.get(key, ''):.6g}" if r.truth.get(key) else "",
                                     f"{r.measured.get(key, float('nan')):.6g}",
                                     f"{r.error(key):.6g}", r.failed])
    print(f"\nwrote {path}")
    print("\nRelative error against the closed form. These are ACCURACY numbers: unlike "
          "the mask-fidelity\nchecks, nothing here reads the answer from the same place "
          "the measurement came from.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
