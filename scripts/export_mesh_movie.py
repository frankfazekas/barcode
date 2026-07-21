#!/usr/bin/env python3
"""Export a mesh curvature movie as .gif and .avi, from the viewer's camera.

The frame is deliberately bare -- mesh, colorbar, scale bar -- so it drops straight onto
a slide without cropping out titles or stats.

The camera comes from the interactive viewer's "copy view" button, which yields the
rotation matrix and zoom currently on screen. That matrix is applied to the vertices
here and the plot is then drawn from one canonical direction, which reproduces the
viewer exactly rather than approximating it with elev/azim.

    python scripts/export_mesh_movie.py <objs-dir> --out <dir> \\
        --view '{"R": [...9 numbers...], "zoom": 1.0}' --mode timelapse

``--mode timelapse`` scrubs the frames; ``--mode rotation`` spins one frame; ``both``
writes both movies.

Encoding uses OpenCV (AVI, MJPG) and Pillow (GIF), both already dependencies. ffmpeg is
*not* required: the only copy on this machine is a Julia artifact that fails to load its
own shared libraries. Pass ``--ffmpeg`` with a working binary to use it instead, which
gives a slightly cleaner GIF via two-pass palettegen.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from analysis.volumetric.curvature import analyze_curvature

# The viewer's named orientations, as (right, depth, up) basis rows -- identical to
# BASIS in the viewer and to setup_mesh_scene.m's view([az 0]) conventions.
BASIS = {
    "xz":     [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
    "xz_rev": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "yz":     [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
    "yz_rev": [[0, -1, 0], [-1, 0, 0], [0, 0, 1]],
}


def read_obj(path):
    v, f = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("v "):
                v.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                f.append([int(t.split("/")[0]) for t in line.split()[1:4]])
    return np.array(v, dtype=np.float64), np.array(f, dtype=np.int64)


def working_ffmpeg(explicit=None):
    """Return a *usable* ffmpeg, or None. Presence on disk is not enough.

    The only copy on this machine ships inside a Julia artifact and exits 127 with
    "error while loading shared libraries", so the binary is probed rather than trusted.
    """
    for candidate in (explicit, shutil.which("ffmpeg")):
        if not candidate:
            continue
        try:
            probe = subprocess.run([candidate, "-hide_banner", "-version"],
                                   capture_output=True, timeout=20)
            if probe.returncode == 0:
                return candidate
            print(f"ignoring ffmpeg at {candidate}: exit {probe.returncode}")
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"ignoring ffmpeg at {candidate}: {type(exc).__name__}")
    return None


def parse_clim(spec, values):
    """Resolve the colour limits.

    ``0.75``        symmetric, fixed
    ``-1.5,0.75``   asymmetric, fixed -- zero stays at the midpoint of the colourmap
    ``p98``         the 98th percentile of |H|, i.e. data-derived

    A *fixed* limit is the default because a percentile is recomputed per dataset, so
    two figures drawn that way cannot be compared by eye, the printed number is
    arbitrary (0.6872), and the value tracks mesh resolution -- per-face curvature
    sharpens as ``maxrad`` falls -- rather than the specimen.
    """
    text = str(spec).strip().lower()
    if text.startswith("p"):
        limit = float(np.percentile(np.abs(values), float(text[1:])))
        return -limit, limit
    if "," in text:
        low, high = (float(v) for v in text.split(","))
        return low, high
    limit = abs(float(text))
    return -limit, limit


def parse_view(spec: str) -> tuple:
    """Accept a named orientation, or the viewer's copied JSON {R:[9], zoom:f}."""
    if not spec:
        return np.array(BASIS["yz"], dtype=np.float64), 1.0
    text = spec.strip()
    if text in BASIS:
        return np.array(BASIS[text], dtype=np.float64), 1.0
    data = json.loads(text)
    matrix = np.asarray(data["R"], dtype=np.float64).reshape(3, 3)
    return matrix, float(data.get("zoom", 1.0))


def natural_frames(objs_dir: str, pattern: str):
    """OBJs ordered by their trailing number, so _2 precedes _10."""
    files = [f for f in os.listdir(objs_dir) if f.lower().endswith(".obj")
             and re.search(pattern, f)]
    def key(name):
        m = re.findall(r"(\d+)", os.path.splitext(name)[0])
        return int(m[-1]) if m else 0
    return [os.path.join(objs_dir, f) for f in sorted(files, key=key)]


def render(V, F, curv, matrix, zoom, centre, span, norm, cmap, args, time_text=""):
    """One frame: mesh, colorbar, scale bar, and an optional time label. Nothing else."""
    # Apply the viewer's rotation to the geometry, then look from one fixed direction.
    # rotated = (right, depth, up); matplotlib elev=0/azim=-90 puts right on screen-x,
    # up on screen-y and depth into the screen -- i.e. exactly the viewer's projection.
    rotated = (V - centre) @ matrix.T
    view_span = span / max(zoom, 1e-6)

    # Lay the canvas out in inches from the font size rather than with a fixed ratio:
    # at 48 pt a hard-coded margin clips the rotated label off the right edge. Widths
    # are generous on purpose -- crop_to_content trims whatever is left over.
    points = args.fontsize / 72.0
    ticks_in = points * 0.70 * 4.4          # widest tick, e.g. "-0.75"
    label_in = points * 1.5                 # rotated label's thickness plus padding
    bar_in = 0.34
    pad_in = points * 0.5
    right_in = pad_in + bar_in + ticks_in + label_in
    fig_w = args.size + right_in

    top_in = points * 2.0 if time_text else 0.0
    bottom_in = points * 1.9                # the scale bar caption sits below the axes
    # The rotated colorbar label runs along the bar, so the canvas has to be at least as
    # tall as the label is long or its ends are sliced off -- at 48 pt "Mean Curvature
    # (1/um)" is about 7 inches of text.
    label_run_in = 0.58 * points * len(args.clabel)
    fig_h = max(args.size + top_in + bottom_in, label_run_in + top_in + 0.8)
    mesh_h = fig_h - top_in - bottom_in

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="black")
    ax = fig.add_axes([0.0, bottom_in / fig_h, args.size / fig_w, mesh_h / fig_h],
                      projection="3d", facecolor="black")
    ax.set_proj_type("ortho")
    ax.add_collection3d(
        Poly3DCollection(
            rotated[F - 1],
            facecolors=cmap(norm(np.clip(curv.k_mean_faces, norm.vmin, norm.vmax))),
            edgecolor="none")
    )
    for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        setter(-view_span, view_span)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=0, azim=-90)
    ax.set_axis_off()

    # Scale bar, in axes fractions so it is correct for any rotation/zoom.
    frac = args.scalebar / (2 * view_span)
    ax.add_line(Line2D([0.06, 0.06 + frac], [0.06, 0.06], transform=ax.transAxes,
                       color="white", lw=5, zorder=100, solid_capstyle="butt"))
    ax.text2D(0.06, 0.085, f"{args.scalebar:g} µm",
              transform=ax.transAxes, color="white", ha="left", va="bottom",
              fontsize=args.fontsize, zorder=100)

    if time_text:
        # Above the object, centred on the mesh panel rather than the whole canvas, so
        # the colorbar column does not pull it off-centre.
        fig.text((args.size / fig_w) / 2, 1.0 - (top_in * 0.62) / fig_h, time_text,
                 color="white", ha="center", va="center", fontsize=args.fontsize)

    # Tall enough for the label to sit alongside it, and centred in the space left below
    # the time stamp.
    cb_h = min(max(label_run_in, mesh_h * 0.7), fig_h - top_in - 0.4)
    cb_left = (args.size + pad_in) / fig_w
    cb_bottom = (fig_h - top_in - cb_h) / 2.0
    cax = fig.add_axes([cb_left, cb_bottom / fig_h, bar_in / fig_w, cb_h / fig_h])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax)
    cb.set_label(args.clabel, color="white", fontsize=args.fontsize, labelpad=8)
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white",
                                labelsize=args.fontsize * 0.8)
    cb.outline.set_edgecolor("white")
    return fig


def crop_to_content(images, margin=12, threshold=12):
    """Trim the dead background so the frame is only nucleus, colorbar and scale bar.

    The bounding box is the **union** over every frame, and the same box is applied to
    all of them. Cropping each frame to its own content would make the object jitter and
    change scale from frame to frame, which would read as motion that is not there.
    """
    boxes = []
    for image in images:
        mask = np.asarray(image).max(axis=2) > threshold
        if not mask.any():
            continue
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        boxes.append((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))
    if not boxes:
        return images

    width, height = images[0].size
    left = max(min(b[0] for b in boxes) - margin, 0)
    top = max(min(b[1] for b in boxes) - margin, 0)
    right = min(max(b[2] for b in boxes) + margin, width)
    bottom = min(max(b[3] for b in boxes) + margin, height)

    # Even dimensions keep every video encoder happy.
    if (right - left) % 2:
        right = min(right + 1, width) if right < width else right - 1
    if (bottom - top) % 2:
        bottom = min(bottom + 1, height) if bottom < height else bottom - 1

    print(f"cropped {width}x{height} -> {right - left}x{bottom - top} "
          f"(dead margin removed)")
    return [im.crop((left, top, right, bottom)) for im in images]


def figures_to_images(figures, dpi):
    """Rasterise the figures to RGB PIL images, closing each as we go."""
    from PIL import Image

    images = []
    for fig in figures:
        fig.set_dpi(dpi)
        fig.canvas.draw()
        images.append(
            Image.frombytes("RGBA", fig.canvas.get_width_height(),
                            bytes(fig.canvas.buffer_rgba())).convert("RGB")
        )
        plt.close(fig)
    return images


def write_gif(images, path, fps):
    """Animated GIF via Pillow, with one palette for the whole clip.

    Quantising each frame independently makes a smooth diverging colourmap shimmer
    between frames; a single adaptive palette keeps it stable.
    """
    from PIL import Image

    reference = images[len(images) // 2].quantize(colors=255, method=Image.MEDIANCUT)
    frames = [im.quantize(palette=reference, dither=Image.FLOYDSTEINBERG)
              for im in images]
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=int(round(1000 / fps)), loop=0, optimize=True)
    return path


def write_avi(images, path, fps):
    """AVI via OpenCV's MJPG encoder -- no external binary, and PowerPoint plays it."""
    import cv2

    width, height = images[0].size
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"OpenCV could not open {path} for writing")
    for image in images:
        writer.write(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
    writer.release()
    return path


def write_with_ffmpeg(images, out_stem, fps, formats, ffmpeg, tmpdir, prefix):
    """Higher-quality path used only when a working ffmpeg was found."""
    for i, image in enumerate(images):
        image.save(os.path.join(tmpdir, f"{prefix}_{i:04d}.png"))
    pattern = os.path.join(tmpdir, f"{prefix}_%04d.png")
    made = []
    if "avi" in formats:
        avi = out_stem + ".avi"
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps),
                        "-i", pattern, "-c:v", "mjpeg", "-q:v", "2",
                        "-pix_fmt", "yuvj420p", avi], check=True)
        made.append(avi)
    if "gif" in formats:
        gif = out_stem + ".gif"
        palette = os.path.join(tmpdir, f"{prefix}_palette.png")
        # Two-pass: a per-clip palette beats the default web palette badly here.
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps),
                        "-i", pattern, "-vf", "palettegen=stats_mode=diff", palette],
                       check=True)
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps),
                        "-i", pattern, "-i", palette,
                        "-lavfi", "paletteuse=dither=bayer:bayer_scale=3", gif],
                       check=True)
        made.append(gif)
    return made


def encode(images, out_stem, fps, formats, ffmpeg, prefix):
    if ffmpeg:
        with tempfile.TemporaryDirectory() as tmpdir:
            return write_with_ffmpeg(images, out_stem, fps, formats, ffmpeg,
                                     tmpdir, prefix)
    made = []
    if "avi" in formats:
        made.append(write_avi(images, out_stem + ".avi", fps))
    if "gif" in formats:
        made.append(write_gif(images, out_stem + ".gif", fps))
    return made


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("objs", help="directory of .obj meshes")
    p.add_argument("--out", required=True, help="output directory (use a data drive)")
    p.add_argument("--stem", default="mesh_movie")
    p.add_argument("--pattern", default=r".", help="regex selecting which OBJs to use")
    p.add_argument("--view", default="yz",
                   help="'xz'/'xz_rev'/'yz'/'yz_rev', or the viewer's copied JSON")
    p.add_argument("--mode", default="timelapse",
                   choices=("timelapse", "rotation", "both"))
    p.add_argument("--rotation-frame", type=int, default=0,
                   help="index of the mesh to spin for --mode rotation")
    p.add_argument("--rotation-step", type=float, default=6.0, help="degrees per frame")
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--rotation-fps", type=int, default=20)
    p.add_argument("--formats", default="gif,avi")
    p.add_argument("--size", type=float, default=7.0, help="figure size in inches")
    p.add_argument("--dpi", type=int, default=140)
    p.add_argument("--fontsize", type=int, default=22,
                   help="colorbar label, scale bar and time label text size")
    p.add_argument("--clabel", default="Mean Curvature (1/µm)",
                   help="colorbar label text")
    p.add_argument("--time-label", default="{t:g} {unit}",
                   help="time label above the object; '' to omit. {t} and {unit} expand")
    p.add_argument("--frame-interval", type=float, default=1.0,
                   help="time between meshes, in --time-unit")
    p.add_argument("--time-unit", default="min")
    p.add_argument("--scalebar", type=float, default=5.0, help="scale bar length in um")
    p.add_argument("--clim", default="0.5",
                   help="colour limits in 1/um: '0.75' symmetric, '-1.5,0.75' "
                        "asymmetric, or 'p98' for the data-derived percentile")
    p.add_argument("--zoom", type=float, default=1.9,
                   help="extra magnification on top of the view's own zoom. matplotlib's "
                        "3-D axes reserve a wide margin, so 1.0 leaves the object small "
                        "in frame; ~1.6 fills a slide without clipping")
    p.add_argument("--no-crop", action="store_true",
                   help="keep the full square frame instead of trimming to content")
    p.add_argument("--ffmpeg", default=None)
    args = p.parse_args()

    ffmpeg = working_ffmpeg(args.ffmpeg)
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    os.makedirs(args.out, exist_ok=True)

    paths = natural_frames(args.objs, args.pattern)
    if not paths:
        raise SystemExit(f"no .obj files matching {args.pattern!r} in {args.objs}")
    print(f"{len(paths)} mesh(es) from {args.objs}")

    meshes = []
    for path in paths:
        V, F = read_obj(path)
        meshes.append((os.path.basename(path), V, F, analyze_curvature(V, F, z_axis=2)))

    # One shared frame and colour scale across the whole movie, so brightness changes
    # mean shape changes rather than rescaling.
    all_v = np.concatenate([m[1] for m in meshes])
    centre = (all_v.min(axis=0) + all_v.max(axis=0)) / 2
    span = float(np.abs(all_v - centre).max()) * 1.06
    all_k = np.concatenate([m[3].k_mean_faces for m in meshes])
    low, high = parse_clim(args.clim, all_k)
    norm = TwoSlopeNorm(vmin=low, vcenter=0.0, vmax=high)
    cmap = plt.get_cmap("RdBu_r")
    matrix, zoom = parse_view(args.view)
    zoom *= args.zoom
    clipped = 100.0 * float(((all_k < low) | (all_k > high)).mean())
    print(f"colour limits [{low:g}, {high:g}] 1/um  (clips {clipped:.1f}% of faces; "
          f"H spans {all_k.min():+.2f} to {all_k.max():+.2f}), zoom {zoom:.2f}")

    written = []
    if args.mode in ("timelapse", "both"):
        figs = []
        for i, (_, V, F, c) in enumerate(meshes):
            stamp = (args.time_label.format(t=i * args.frame_interval,
                                            unit=args.time_unit)
                     if args.time_label else "")
            figs.append(render(V, F, c, matrix, zoom, centre, span, norm, cmap,
                               args, stamp))
        images = figures_to_images(figs, args.dpi)
        if not args.no_crop:
            images = crop_to_content(images)
        written += encode(images, os.path.join(args.out, f"{args.stem}_timelapse"),
                          args.fps, formats, ffmpeg, "t")

    if args.mode in ("rotation", "both"):
        _, V, F, c = meshes[args.rotation_frame]
        figs = []
        for angle in np.arange(0, 360, args.rotation_step):
            radians = np.deg2rad(angle)
            # Spin about the screen-vertical axis, so the object turns in place from
            # whatever view the user handed in.
            spin = np.array([[np.cos(radians), np.sin(radians), 0],
                             [-np.sin(radians), np.cos(radians), 0],
                             [0, 0, 1]])
            stamp = (args.time_label.format(
                t=args.rotation_frame * args.frame_interval, unit=args.time_unit)
                if args.time_label else "")
            figs.append(render(V, F, c, spin @ matrix, zoom, centre, span,
                               norm, cmap, args, stamp))
        images = figures_to_images(figs, args.dpi)
        if not args.no_crop:
            images = crop_to_content(images)
        written += encode(images, os.path.join(args.out, f"{args.stem}_rotation"),
                          args.rotation_fps, formats, ffmpeg, "r")

    for path in written:
        print(f"  {os.path.getsize(path) / 1e6:6.2f} MB  {path}")
