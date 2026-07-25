"""Named-orientation curvature renders + a self-contained rotatable mesh viewer.

Orientations follow TCell-3D-Morphodynamics/src/plotting/mesh_with_channel/scene/
setup_mesh_scene.m:37-52, where MATLAB vertices are (image-Y, image-X, image-Z):

    xz      view([90 0])    camera at +image-Y   right=-X depth=-Y up=+Z
    xz_rev  view([-90 0])   camera at -image-Y   right=+X depth=+Y up=+Z
    yz      view([0 0])     camera at -image-X   right=+Y depth=+X up=+Z
    yz_rev  view([180 0])   camera at +image-X   right=-Y depth=-X up=+Z

Our OBJ is (image-X, image-Y, image-Z), so the matplotlib equivalents are elev=0 with
azim = 90 / -90 / 180 / 0 respectively, orthographic projection.
"""
import base64
import json
import os
import sys

sys.path.insert(0, r"C:\Users\Upadhyaya_Lab\Code\barcode")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from analysis.volumetric.curvature import CONCAVE, CONVEX, HYPERBOLOID, analyze_curvature
from analysis.volumetric.mesh import mesh_geometry

OUT = (r"F:\FF\nucleus_live_cell\jurkat_nucleus_centrosome\NaBu800 Experiments"
       r"\Control_3D_CD3\all_cells_together\BARCODE\results\mesh_timepoints")
SCRATCH = (r"C:\Users\UPADHY~1\AppData\Local\Temp\claude"
           r"\C--Users-Upadhyaya-Lab-Code-barcode"
           r"\9f48b303-899e-4482-ab3b-afb87486e1b4\scratchpad")
LABEL = "Cell1_1"

AZIM = {"xz": 90, "xz_rev": -90, "yz": 180, "yz_rev": 0}        # matplotlib azim
MATLAB_AZ = {"xz": 90, "xz_rev": -90, "yz": 0, "yz_rev": 180}   # MATLAB view([az 0])
# (right, depth, up) basis per orientation, as signed world-axis indices.
BASIS = {
    "xz":     ((0, -1), (1, -1), (2, 1)),
    "xz_rev": ((0, 1), (1, 1), (2, 1)),
    "yz":     ((1, 1), (0, 1), (2, 1)),
    "yz_rev": ((1, -1), (0, -1), (2, 1)),
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


V, F = read_obj(os.path.join(OUT, "objs", f"{LABEL}.obj"))
curv = analyze_curvature(V, F, z_axis=2)
geom = mesh_geometry(V[:, ::-1], F)
k = curv.k_mean_faces
lim = float(np.percentile(np.abs(k), 98))
norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
cmap = plt.get_cmap("RdBu_r")
print(f"{LABEL}: {V.shape[0]} vertices / {F.shape[0]} faces, |k| p98 = {lim:.4f}")


def render_png(orient):
    fig = plt.figure(figsize=(11, 11.5), facecolor="black")
    ax = fig.add_axes([0.02, 0.06, 0.80, 0.86], projection="3d", facecolor="black")
    ax.set_proj_type("ortho")
    ax.add_collection3d(
        Poly3DCollection(V[F - 1], facecolors=cmap(norm(k)), edgecolor="none")
    )
    lo, hi = V.min(axis=0), V.max(axis=0)
    centre, span = (lo + hi) / 2, (hi - lo).max() / 2 * 1.02
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), centre):
        setter(c - span, c + span)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=0, azim=AZIM[orient])
    ax.set_axis_off()

    # Scale bar as a 2-D overlay in axes-fraction coordinates, so it is correct for
    # every orientation. With box_aspect (1,1,1), equal limits and elev=0 at a multiple
    # of 90 degrees, the projected cube is a square of side 2*span fitted to the axes,
    # so 5 um is exactly 5/(2*span) of the axes width.
    bar = 5.0
    frac = bar / (2 * span)
    ax.add_line(Line2D([0.06, 0.06 + frac], [0.055, 0.055], transform=ax.transAxes,
                       color="white", lw=3, zorder=100))
    # text2D, not text: Axes3D.text takes (x, y, z, s) and cannot do a 2-D overlay.
    ax.text2D(0.06 + frac / 2, 0.030, f"{bar:g} µm", transform=ax.transAxes,
              color="white", ha="center", va="top", fontsize=13, zorder=100)

    cax = fig.add_axes([0.86, 0.20, 0.030, 0.58])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax)
    cb.set_label("mean curvature (1/µm)", color="white", fontsize=13)
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    cb.outline.set_edgecolor("white")

    fig.text(0.02, 0.975,
             f"NaBu800 Control_3D_CD3 — Cell1, timepoint 1 — mean curvature, {orient} view",
             color="white", fontsize=15, va="top")
    fig.text(0.02, 0.945,
             f"volume {geom.volume_um3:.1f} µm³   SA {geom.surface_area_um2:.1f} µm²   "
             f"sphericity {geom.sphericity:.3f}   "
             f"invagination {curv.invagination_ratio:.3f}   {F.shape[0]} faces",
             color="0.75", fontsize=11.5, va="top")
    fig.text(0.02, 0.020,
             f"blue = concave (invagination)   red = convex   camera: MATLAB "
             f"view([{MATLAB_AZ[orient]} 0]), orthographic, daspect [1 1 1]",
             color="0.6", fontsize=10, va="bottom")

    path = os.path.join(OUT, f"{LABEL} Mean Curvature {orient}.png")
    fig.savefig(path, dpi=200, facecolor="black")
    plt.close(fig)
    print("wrote", path)
    return path


for _o in ("yz", "xz_rev"):
    render_png(_o)

# --------------------------------------------------------------------------- #
# self-contained rotatable viewer
# --------------------------------------------------------------------------- #
def b64(array):
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


centre = (V.min(axis=0) + V.max(axis=0)) / 2
verts = (V - centre).astype(np.float32)

# 256-entry palette; each face stores an index into it.
palette = (np.array([cmap(i / 255.0)[:3] for i in range(256)]) * 255).astype(np.uint8)

# Clip BEFORE normalising. TwoSlopeNorm maps out-of-range values to +/-inf (here 109
# faces, the 2% most extreme curvature), and casting inf to int silently yields index 0
# -- i.e. the most convex faces would have been painted deep blue. matplotlib's own
# cmap(norm(k)) handles inf via its over/under colours, so the PNGs were never affected.
# With symmetric limits about vcenter=0, TwoSlopeNorm is exactly linear, so clipping
# gives the same colours as the PNG with the tails saturated at the ramp ends.
scaled = (np.clip(k, -lim, lim) + lim) / (2 * lim)
assert np.isfinite(scaled).all(), "non-finite colour index"
face_curv_idx = np.clip(np.rint(scaled * 255.0), 0, 255).astype(np.uint8)

class_rgb = np.array([[192, 57, 43], [224, 176, 64], [74, 128, 168]], dtype=np.uint8)
class_palette = np.zeros((256, 3), dtype=np.uint8)
class_palette[:3] = class_rgb
face_class_idx = curv.concavity_classes.astype(np.uint8)

payload = {
    "verts": b64(verts),
    "faces": b64(F.astype(np.uint16) - 1),
    "nv": int(V.shape[0]),
    "nf": int(F.shape[0]),
    "curvIdx": b64(face_curv_idx),
    "classIdx": b64(face_class_idx),
    "palette": b64(palette),
    "classPalette": b64(class_palette),
    "basis": BASIS,
    "lim": lim,
    "radius": float(np.abs(verts).max()),
    "stats": {
        "volume": geom.volume_um3, "sa": geom.surface_area_um2,
        "sph": geom.sphericity, "invag": curv.invagination_ratio,
        "meanCurv": curv.mean_curvature, "nf": int(F.shape[0]),
        "nConcave": int((curv.concavity_classes == CONCAVE).sum()),
        "nSaddle": int((curv.concavity_classes == HYPERBOLOID).sum()),
        "nConvex": int((curv.concavity_classes == CONVEX).sum()),
    },
}

html = """<meta charset="utf-8">
<title>Cell1 t1 nucleus mesh &mdash; rotatable</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; background: #000; color: #ddd;
         font: 14px/1.5 system-ui, -apple-system, Segoe UI, sans-serif; }
  .wrap { max-width: 1100px; margin: 0 auto; padding: 18px 16px 40px; }
  h1 { font-size: 18px; font-weight: 600; margin: 0 0 4px; color: #fff; }
  .sub { color: #999; font-size: 13px; margin-bottom: 14px; }
  .stage { position: relative; background: #000; border: 1px solid #222;
           border-radius: 8px; overflow: hidden; }
  canvas { display: block; width: 100%; height: auto; cursor: grab; touch-action: none; }
  canvas.drag { cursor: grabbing; }
  .bar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
         margin: 12px 0 4px; }
  button { background: #1b1b1b; color: #ddd; border: 1px solid #333; border-radius: 6px;
           padding: 6px 12px; font: inherit; font-size: 13px; cursor: pointer; }
  button:hover { background: #262626; }
  button.on { background: #2d4a63; border-color: #4a7ba3; color: #fff; }
  .sep { width: 1px; height: 22px; background: #333; margin: 0 4px; }
  .legend { display: flex; align-items: center; gap: 10px; margin-top: 12px;
            flex-wrap: wrap; font-size: 12.5px; color: #aaa; }
  .ramp { width: 220px; height: 12px; border: 1px solid #444; border-radius: 2px; }
  .sw { display: inline-block; width: 12px; height: 12px; border-radius: 2px;
        vertical-align: -1px; margin-right: 5px; }
  table { border-collapse: collapse; margin-top: 16px; font-size: 13px; }
  td { padding: 3px 18px 3px 0; color: #bbb; }
  td.k { color: #888; }
  .hint { color: #777; font-size: 12.5px; margin-top: 10px; }
</style>
<div class="wrap">
  <h1>NaBu800 Control_3D_CD3 &mdash; Cell1, timepoint 1</h1>
  <div class="sub">Nucleus surface mesh coloured by mean curvature. Drag to rotate,
    scroll to zoom, double-click to reset.</div>

  <div class="bar">
    <button data-view="xz">xz</button>
    <button data-view="xz_rev">xz_rev</button>
    <button data-view="yz" class="on">yz</button>
    <button data-view="yz_rev">yz_rev</button>
    <div class="sep"></div>
    <button id="mode" data-mode="curv">colour: mean curvature</button>
    <button id="shade" class="on">shading</button>
    <button id="spin">spin</button>
  </div>

  <div class="stage"><canvas id="c"></canvas></div>

  <div class="legend" id="legend"></div>

  <table>
    <tr><td class="k">volume</td><td id="s-vol"></td>
        <td class="k">surface area</td><td id="s-sa"></td></tr>
    <tr><td class="k">sphericity</td><td id="s-sph"></td>
        <td class="k">invagination ratio</td><td id="s-invag"></td></tr>
    <tr><td class="k">mean curvature</td><td id="s-curv"></td>
        <td class="k">faces</td><td id="s-nf"></td></tr>
  </table>

  <div class="hint">Orientations match the MATLAB pipeline
    (<code>setup_mesh_scene.m</code>): xz = view([90 0]), xz_rev = view([-90 0]),
    yz = view([0 0]), yz_rev = view([180 0]); orthographic, equal aspect.</div>
</div>
<script>
const DATA = __PAYLOAD__;

function unpack(b64, Type) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Type(bytes.buffer);
}

const verts = unpack(DATA.verts, Float32Array);
const faces = unpack(DATA.faces, Uint16Array);
const curvIdx = unpack(DATA.curvIdx, Uint8Array);
const classIdx = unpack(DATA.classIdx, Uint8Array);
const palette = unpack(DATA.palette, Uint8Array);
const classPalette = unpack(DATA.classPalette, Uint8Array);
const NF = DATA.nf;

const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
let W = 0, H = 0, dpr = Math.min(window.devicePixelRatio || 1, 2);

function resize() {
  const cssW = canvas.parentElement.clientWidth;
  const cssH = Math.round(cssW * 0.82);
  canvas.style.height = cssH + 'px';
  W = canvas.width = Math.round(cssW * dpr);
  H = canvas.height = Math.round(cssH * dpr);
  draw();
}

// --- camera -----------------------------------------------------------------
let R = new Float64Array(9);          // rows: right, depth, up
let zoom = 1, spinning = false, mode = 'curv', shading = true;

function setBasis(name) {
  const b = DATA.basis[name];
  R.fill(0);
  for (let r = 0; r < 3; r++) { R[r * 3 + b[r][0]] = b[r][1]; }
}
function rotate(dx, dy) {
  // Rotate about the current screen up (dx) and right (dy) axes.
  const ax = [R[0], R[1], R[2]], ay = [R[3], R[4], R[5]], az = [R[6], R[7], R[8]];
  const rot = (u, v, a) => {
    const c = Math.cos(a), s = Math.sin(a);
    for (let i = 0; i < 3; i++) {
      const t = c * u[i] + s * v[i];
      v[i] = -s * u[i] + c * v[i];
      u[i] = t;
    }
  };
  rot(ax, ay, dx);      // yaw: right/depth about up
  rot(az, ay, dy);      // pitch: up/depth about right
  R.set([...ax, ...ay, ...az]);
}

// --- drawing ----------------------------------------------------------------
const px = new Float32Array(DATA.nv), py = new Float32Array(DATA.nv),
      pz = new Float32Array(DATA.nv);
const order = new Uint32Array(NF), depth = new Float32Array(NF);

function draw() {
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, W, H);
  const scale = Math.min(W, H) / (2.15 * DATA.radius) * zoom;
  const cx = W / 2, cy = H / 2;

  for (let i = 0, n = DATA.nv; i < n; i++) {
    const x = verts[i * 3], y = verts[i * 3 + 1], z = verts[i * 3 + 2];
    px[i] = cx + (R[0] * x + R[1] * y + R[2] * z) * scale;
    py[i] = cy - (R[6] * x + R[7] * y + R[8] * z) * scale;
    pz[i] = R[3] * x + R[4] * y + R[5] * z;
  }
  for (let f = 0; f < NF; f++) {
    const a = faces[f * 3], b = faces[f * 3 + 1], c = faces[f * 3 + 2];
    depth[f] = pz[a] + pz[b] + pz[c];
    order[f] = f;
  }
  // Painter's algorithm: far (large depth) first.
  const idx = Array.prototype.slice.call(order);
  idx.sort((p, q) => depth[q] - depth[p]);

  const pal = (mode === 'curv') ? palette : classPalette;
  const src = (mode === 'curv') ? curvIdx : classIdx;

  for (let n = 0; n < NF; n++) {
    const f = idx[n];
    const a = faces[f * 3], b = faces[f * 3 + 1], c = faces[f * 3 + 2];
    const ci = src[f] * 3;
    let r = pal[ci], g = pal[ci + 1], bl = pal[ci + 2];

    if (shading) {
      // Screen-space normal gives a cheap Lambert term; keeps hue, varies value.
      const ux = px[b] - px[a], uy = py[b] - py[a];
      const vx = px[c] - px[a], vy = py[c] - py[a];
      const area = ux * vy - uy * vx;
      const t = 0.72 + 0.28 * Math.min(1, Math.abs(area) / 900);
      r = r * t; g = g * t; bl = bl * t;
    }
    ctx.fillStyle = 'rgb(' + (r | 0) + ',' + (g | 0) + ',' + (bl | 0) + ')';
    ctx.beginPath();
    ctx.moveTo(px[a], py[a]);
    ctx.lineTo(px[b], py[b]);
    ctx.lineTo(px[c], py[c]);
    ctx.closePath();
    ctx.fill();
  }

  // 5 um scale bar
  const bar = 5 * scale;
  ctx.strokeStyle = '#fff'; ctx.lineWidth = 3 * dpr;
  ctx.beginPath();
  ctx.moveTo(20 * dpr, H - 26 * dpr); ctx.lineTo(20 * dpr + bar, H - 26 * dpr);
  ctx.stroke();
  ctx.fillStyle = '#fff'; ctx.font = (13 * dpr) + 'px system-ui';
  ctx.fillText('5 \\u00b5m', 20 * dpr, H - 34 * dpr);
}

// --- interaction ------------------------------------------------------------
let dragging = false, lx = 0, ly = 0;
canvas.addEventListener('pointerdown', e => {
  dragging = true; lx = e.clientX; ly = e.clientY;
  canvas.classList.add('drag'); canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener('pointermove', e => {
  if (!dragging) return;
  rotate((e.clientX - lx) * 0.01, (e.clientY - ly) * 0.01);
  lx = e.clientX; ly = e.clientY;
  draw();
});
canvas.addEventListener('pointerup', e => {
  dragging = false; canvas.classList.remove('drag');
});
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  zoom *= Math.exp(-e.deltaY * 0.0012);
  zoom = Math.max(0.3, Math.min(6, zoom));
  draw();
}, { passive: false });
canvas.addEventListener('dblclick', () => { setBasis('yz'); zoom = 1; draw(); });

document.querySelectorAll('[data-view]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('[data-view]').forEach(b => b.classList.remove('on'));
    btn.classList.add('on');
    setBasis(btn.dataset.view); draw();
  });
});
const modeBtn = document.getElementById('mode');
modeBtn.addEventListener('click', () => {
  mode = (mode === 'curv') ? 'class' : 'curv';
  modeBtn.textContent = 'colour: ' +
    (mode === 'curv' ? 'mean curvature' : 'concavity class');
  legend(); draw();
});
const shadeBtn = document.getElementById('shade');
shadeBtn.addEventListener('click', () => {
  shading = !shading; shadeBtn.classList.toggle('on', shading); draw();
});
const spinBtn = document.getElementById('spin');
spinBtn.addEventListener('click', () => {
  spinning = !spinning; spinBtn.classList.toggle('on', spinning);
  if (spinning) requestAnimationFrame(tick);
});
function tick() {
  if (!spinning) return;
  rotate(0.012, 0); draw(); requestAnimationFrame(tick);
}

// --- legend + stats ---------------------------------------------------------
function legend() {
  const el = document.getElementById('legend');
  if (mode === 'curv') {
    let stops = [];
    for (let i = 0; i <= 10; i++) {
      const p = palette[Math.round(i * 25.5) * 3];
      const ci = Math.round(i * 25.5) * 3;
      stops.push('rgb(' + palette[ci] + ',' + palette[ci + 1] + ',' +
                 palette[ci + 2] + ') ' + (i * 10) + '%');
    }
    el.innerHTML = '<span>' + (-DATA.lim).toFixed(2) + '</span>' +
      '<div class="ramp" style="background:linear-gradient(90deg,' +
      stops.join(',') + ')"></div>' +
      '<span>' + DATA.lim.toFixed(2) + ' 1/\\u00b5m</span>' +
      '<span style="margin-left:10px">blue = concave (invagination), red = convex</span>';
  } else {
    const s = DATA.stats;
    el.innerHTML =
      '<span><i class="sw" style="background:#c0392b"></i>concave (' + s.nConcave + ')</span>' +
      '<span><i class="sw" style="background:#e0b040"></i>saddle (' + s.nSaddle + ')</span>' +
      '<span><i class="sw" style="background:#4a80a8"></i>convex (' + s.nConvex + ')</span>';
  }
}
const S = DATA.stats;
document.getElementById('s-vol').textContent = S.volume.toFixed(1) + ' \\u00b5m\\u00b3';
document.getElementById('s-sa').textContent = S.sa.toFixed(1) + ' \\u00b5m\\u00b2';
document.getElementById('s-sph').textContent = S.sph.toFixed(3);
document.getElementById('s-invag').textContent = S.invag.toFixed(3);
document.getElementById('s-curv').textContent = S.meanCurv.toFixed(4) + ' 1/\\u00b5m';
document.getElementById('s-nf').textContent = S.nf;

setBasis('yz');
legend();
window.addEventListener('resize', resize);
window.addEventListener('load', resize);
(function boot() {
  if (canvas.parentElement.clientWidth > 0) { resize(); }
  else { requestAnimationFrame(boot); }
})();
</script>
"""

html = html.replace("__PAYLOAD__", json.dumps(payload))
viewer = os.path.join(SCRATCH, "mesh_viewer.html")
with open(viewer, "w", encoding="utf-8") as fh:
    fh.write(html)
print("wrote", viewer, f"({os.path.getsize(viewer) / 1024:.0f} KB)")
