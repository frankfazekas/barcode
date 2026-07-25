"""Rotatable, scrubbable viewer over every timepoint of a meshed series.

All frames share one coordinate frame, one quantisation scale and one curvature colour
limit, so scrubbing shows real motion and shape change rather than per-frame rescaling.

Vertices are quantised to int16 (~0.0004 um at this size) purely to keep the
self-contained page small; that is a display-side choice and does not touch the meshes
or metrics on disk.
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

from analysis.volumetric.curvature import CONCAVE, CONVEX, HYPERBOLOID, analyze_curvature
from analysis.volumetric.mesh import mesh_geometry

OUT = (r"F:\FF\nucleus_live_cell\jurkat_nucleus_centrosome\NaBu800 Experiments"
       r"\Control_3D_CD3\all_cells_together\BARCODE\results\mesh_timepoints")
SERIES, FRAMES = "Cell1", list(range(1, 16))

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


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


meshes = []
for fr in FRAMES:
    V, F = read_obj(os.path.join(OUT, "objs", f"{SERIES}_{fr}.obj"))
    curv = analyze_curvature(V, F, z_axis=2)
    geom = mesh_geometry(V[:, ::-1], F)
    meshes.append((fr, V, F, curv, geom))
    print(f"  {SERIES}_{fr}: {F.shape[0]} faces  vol {geom.volume_um3:.1f}  "
          f"invag {curv.invagination_ratio:.3f}", flush=True)

# One shared frame, scale and colour limit across all timepoints.
all_v = np.concatenate([m[1] for m in meshes])
centre = (all_v.min(axis=0) + all_v.max(axis=0)) / 2
radius = float(np.abs(all_v - centre).max())
quant = radius / 32000.0
# Fixed, round colour limit rather than a percentile of the data. A percentile is
# recomputed per dataset, so two figures cannot be compared by eye, it tracks mesh
# resolution rather than the specimen, and it puts an arbitrary number on the legend.
CLIM = 0.5
all_k = np.concatenate([m[3].k_mean_faces for m in meshes])
lim = CLIM
print("colour limit +/-%.2f 1/um (clips %.1f%% of faces; H spans %+.2f to %+.2f)"
      % (lim, 100 * float(((all_k < -lim) | (all_k > lim)).mean()),
         all_k.min(), all_k.max()))
print(f"shared: radius {radius:.3f} um, colour limit +/-{lim:.4f} 1/um")

cmap = plt.get_cmap("RdBu_r")
palette = (np.array([cmap(i / 255.0)[:3] for i in range(256)]) * 255).astype(np.uint8)
class_palette = np.zeros((256, 3), dtype=np.uint8)
class_palette[:3] = np.array([[192, 57, 43], [224, 176, 64], [74, 128, 168]], dtype=np.uint8)

frames_payload = []
for fr, V, F, curv, geom in meshes:
    k = curv.k_mean_faces
    # Clip before scaling: values beyond the shared limit must saturate, never wrap.
    scaled = (np.clip(k, -lim, lim) + lim) / (2 * lim)
    assert np.isfinite(scaled).all()
    qv = np.rint((V - centre) / quant).astype(np.int16)
    assert F.max() <= 65535 and F.min() >= 1
    frames_payload.append({
        "frame": fr,
        "verts": b64(qv),
        "faces": b64((F.astype(np.uint16) - 1)),
        "nv": int(V.shape[0]),
        "nf": int(F.shape[0]),
        "curvIdx": b64(np.clip(np.rint(scaled * 255.0), 0, 255).astype(np.uint8)),
        "classIdx": b64(curv.concavity_classes.astype(np.uint8)),
        "stats": {
            "volume": round(geom.volume_um3, 3),
            "sa": round(geom.surface_area_um2, 3),
            "sph": round(geom.sphericity, 4),
            "invag": round(curv.invagination_ratio, 4),
            "meanCurv": round(curv.mean_curvature, 5),
            "nConcave": int((curv.concavity_classes == CONCAVE).sum()),
            "nSaddle": int((curv.concavity_classes == HYPERBOLOID).sum()),
            "nConvex": int((curv.concavity_classes == CONVEX).sum()),
        },
    })

payload = {
    "series": SERIES,
    "frames": frames_payload,
    "palette": b64(palette),
    "classPalette": b64(class_palette),
    "basis": BASIS,
    "lim": lim,
    "radius": radius,
    "quant": quant,
}

html = r"""<meta charset="utf-8">
<title>Cell1 nucleus meshes &mdash; all timepoints, rotatable</title>
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
  .bar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 12px 0 4px; }
  button { background: #1b1b1b; color: #ddd; border: 1px solid #333; border-radius: 6px;
           padding: 6px 12px; font: inherit; font-size: 13px; cursor: pointer; }
  button:hover { background: #262626; }
  button.on { background: #2d4a63; border-color: #4a7ba3; color: #fff; }
  .sep { width: 1px; height: 22px; background: #333; margin: 0 4px; }
  .scrub { display: flex; align-items: center; gap: 12px; margin: 14px 0 2px; }
  input[type=range] { flex: 1; accent-color: #4a7ba3; }
  .fnum { font-variant-numeric: tabular-nums; color: #fff; min-width: 118px; }
  .legend { display: flex; align-items: center; gap: 10px; margin-top: 12px;
            flex-wrap: wrap; font-size: 12.5px; color: #aaa; }
  .ramp { width: 220px; height: 12px; border: 1px solid #444; border-radius: 2px; }
  .sw { display: inline-block; width: 12px; height: 12px; border-radius: 2px;
        vertical-align: -1px; margin-right: 5px; }
  table { border-collapse: collapse; margin-top: 14px; font-size: 13px; }
  td { padding: 3px 18px 3px 0; color: #bbb; font-variant-numeric: tabular-nums; }
  td.k { color: #888; }
  .spark { margin-top: 14px; }
  .hint { color: #777; font-size: 12.5px; margin-top: 12px; }
  #framebar { gap: 4px; margin-top: 8px; }
  #framebar button { padding: 4px 9px; font-size: 12px; min-width: 34px; }
  #framebar button.ref { border-color: #7a7a7a; color: #fff; }
  select { background: #1b1b1b; color: #ddd; border: 1px solid #333; border-radius: 6px;
           padding: 5px 8px; font: inherit; font-size: 12.5px; }
  .note { color: #8a8a90; font-size: 12.5px; min-height: 18px; margin: 2px 0 6px; }
  .note.warn { color: #e0a040; }
  button.rec { background: #6a2020; border-color: #b34040; color: #fff; }
</style>
<div class="wrap">
  <h1>NaBu800 Control_3D_CD3 &mdash; <span id="ttl"></span></h1>
  <div class="sub">Nucleus surface meshes coloured by mean curvature, every timepoint.
    Drag to rotate, scroll to zoom, double-click to reset. All frames share one
    coordinate frame and one colour scale, so motion and shape change are real.</div>

  <div class="bar">
    <button data-view="xz">xz</button>
    <button data-view="xz_rev">xz_rev</button>
    <button data-view="yz" class="on">yz</button>
    <button data-view="yz_rev">yz_rev</button>
    <div class="sep"></div>
    <button id="mode">colour: mean curvature</button>
    <button id="shade" class="on">shading</button>
    <button id="spin">spin</button>
    <div class="sep"></div>
    <button id="ghost">ghost: off</button>
    <button id="pin" title="use the current frame as the ghost reference">pin t1</button>
    <div class="sep"></div>
    <button id="record" title="play through the timepoints and download the movie">
      &#9679; record movie</button>
    <button id="copyview" title="copy this camera for scripts/export_mesh_movie.py">
      copy view</button>
  </div>

  <div class="note" id="recnote"></div>

  <div class="stage"><canvas id="c"></canvas></div>

  <div class="scrub">
    <button id="play">&#9654; play</button>
    <input type="range" id="slider" min="0" max="0" value="0" step="1">
    <span class="fnum" id="flabel"></span>
    <select id="speed" title="playback speed">
      <option value="440">0.5x</option>
      <option value="220" selected>1x</option>
      <option value="110">2x</option>
      <option value="55">4x</option>
    </select>
  </div>

  <div class="bar" id="framebar"></div>

  <canvas id="spark" class="spark" height="90"></canvas>

  <div class="legend" id="legend"></div>

  <table>
    <tr><td class="k">volume</td><td id="s-vol"></td>
        <td class="k">surface area</td><td id="s-sa"></td>
        <td class="k">faces</td><td id="s-nf"></td></tr>
    <tr><td class="k">sphericity</td><td id="s-sph"></td>
        <td class="k">invagination ratio</td><td id="s-invag"></td>
        <td class="k">mean curvature</td><td id="s-curv"></td></tr>
  </table>

  <div class="hint">Orientations match the MATLAB pipeline
    (<code>setup_mesh_scene.m</code>): xz = view([90 0]), xz_rev = view([-90 0]),
    yz = view([0 0]), yz_rev = view([180 0]); orthographic, equal aspect.</div>
</div>
<script>
const DATA = __PAYLOAD__;

function unpack(b64, Type) {
  const bin = atob(b64), bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Type(bytes.buffer);
}

const palette = unpack(DATA.palette, Uint8Array);
const classPalette = unpack(DATA.classPalette, Uint8Array);
const FR = DATA.frames.map(f => ({
  frame: f.frame, nv: f.nv, nf: f.nf, stats: f.stats,
  verts: unpack(f.verts, Int16Array),
  faces: unpack(f.faces, Uint16Array),
  curvIdx: unpack(f.curvIdx, Uint8Array),
  classIdx: unpack(f.classIdx, Uint8Array),
}));
const N = FR.length;
let cur = 0, mode = 'curv', shading = true, spinning = false, playing = false;
let ghost = false, refFrame = 0, stepMs = 220;

const canvas = document.getElementById('c'), ctx = canvas.getContext('2d');
let W = 0, H = 0, dpr = Math.min(window.devicePixelRatio || 1, 2);
let R = new Float64Array(9), zoom = 1;

function setBasis(name) {
  const b = DATA.basis[name];
  R.fill(0);
  for (let r = 0; r < 3; r++) R[r * 3 + b[r][0]] = b[r][1];
}
function rotate(dx, dy) {
  const ax = [R[0], R[1], R[2]], ay = [R[3], R[4], R[5]], az = [R[6], R[7], R[8]];
  const rot = (u, v, a) => {
    const c = Math.cos(a), s = Math.sin(a);
    for (let i = 0; i < 3; i++) {
      const t = c * u[i] + s * v[i];
      v[i] = -s * u[i] + c * v[i];
      u[i] = t;
    }
  };
  rot(ax, ay, dx); rot(az, ay, dy);
  R.set([...ax, ...ay, ...az]);
}

function resize() {
  const cssW = canvas.parentElement.clientWidth;
  canvas.style.height = Math.round(cssW * 0.78) + 'px';
  W = canvas.width = Math.round(cssW * dpr);
  H = canvas.height = Math.round(cssW * 0.78 * dpr);
  const sp = document.getElementById('spark');
  sp.width = Math.round(cssW * dpr); sp.style.width = '100%';
  sp.height = Math.round(90 * dpr); sp.style.height = '90px';
  draw(); sparkline();
}

let px = new Float32Array(0), py = new Float32Array(0), pz = new Float32Array(0);
let depth = new Float32Array(0), idx = [];

// Text sizes as a fraction of canvas width, so the widget matches what
// scripts/export_mesh_movie.py writes at --fontsize 48: there the label is 48 pt on a
// 1386 px frame, i.e. ~6% of the width. Ticks are deliberately much smaller than the
// label -- they are reference, the label is the thing being read.
function labelPx() { return Math.max(12 * dpr, Math.round(W * 0.030)); }
function tickPx() { return Math.max(9 * dpr, Math.round(W * 0.019)); }

function draw() {
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, W, H);
  // Ghost first, so the current surface paints over it. Where grey survives, the
  // reference frame occupied space this frame does not -- a direct shape comparison.
  if (ghost && refFrame !== cur) paintMesh(FR[refFrame], true);
  paintMesh(FR[cur], false);

  const big = labelPx();
  const panelW = W - colorbarWidth();
  const panelTop = big * 1.6, panelBottom = H - big * 2.2;
  const scaleBar = 5 * (Math.min(panelW, panelBottom - panelTop)
                        / (1.9 * DATA.radius) * zoom);
  ctx.strokeStyle = '#fff'; ctx.lineWidth = Math.max(3, big * 0.09);
  ctx.beginPath();
  const barY = H - big * 1.5;
  ctx.moveTo(big * 0.5, barY); ctx.lineTo(big * 0.5 + scaleBar, barY);
  ctx.stroke();
  ctx.fillStyle = '#fff';
  ctx.font = big + 'px system-ui';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'bottom';
  ctx.fillText('5 \u00b5m', big * 0.5, barY - big * 0.3);

  // Time stamp above the object, matching the exported movies.
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillStyle = '#fff';
  ctx.fillText((FR[cur].frame - 1) + ' min', meshCentreX(), big * 0.35);

  if (ghost && refFrame !== cur) {
    ctx.fillStyle = '#8a8a90';
    ctx.font = tickPx() + 'px system-ui';
    ctx.textAlign = 'left';
    ctx.fillText('ghost: t' + FR[refFrame].frame, big * 0.5, barY - big * 1.1);
  }
  ctx.textBaseline = 'alphabetic';
  colorbar();
}

// Horizontal centre of the mesh panel -- i.e. excluding the colorbar column, so the
// time stamp sits over the object rather than over the whole canvas.
function meshCentreX() { return (W - colorbarWidth()) / 2; }
function colorbarWidth() { return labelPx() * 1.5 + tickPx() * 3.0; }

// Colorbar drawn on the canvas, so a screenshot of the viewer carries its own scale --
// the same reason the static PNG renders have one.
function colorbar() {
  const big = labelPx(), small = tickPx();
  const bw = Math.max(14 * dpr, big * 0.42);
  const bh = Math.min(H * 0.60, H - big * 3.0);
  // Right-hand column: bar, then tick numbers, then the rotated label hard against the
  // edge. Laid out from the right so the label never runs off, whatever the font size.
  const labelX = W - big * 2.5;
  const bx = labelX - small * 4.6 - bw;
  const by = (H - bh) / 2;
  ctx.textBaseline = 'middle';

  if (mode === 'curv') {
    const grad = ctx.createLinearGradient(0, by + bh, 0, by);   // low at the bottom
    for (let i = 0; i <= 32; i++) {
      const ci = Math.round((i / 32) * 255) * 3;
      grad.addColorStop(i / 32,
        'rgb(' + palette[ci] + ',' + palette[ci + 1] + ',' + palette[ci + 2] + ')');
    }
    ctx.fillStyle = grad;
    ctx.fillRect(bx, by, bw, bh);
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1 * dpr;
    ctx.strokeRect(bx, by, bw, bh);

    ctx.font = small + 'px system-ui';
    const lim = DATA.lim;
    for (let i = 0; i <= 4; i++) {
      const frac = i / 4;
      const y = by + bh * (1 - frac);
      const value = -lim + 2 * lim * frac;
      ctx.strokeStyle = '#ddd';
      ctx.beginPath();
      ctx.moveTo(bx + bw, y); ctx.lineTo(bx + bw + small * 0.35, y); ctx.stroke();
      ctx.fillStyle = '#fff';
      ctx.textAlign = 'left';
      ctx.fillText(value.toFixed(2), bx + bw + small * 0.6, y);
    }
    // The label carries the units, so no separate "1/um" caption is needed.
    ctx.save();
    ctx.translate(labelX, by + bh / 2);
    ctx.rotate(Math.PI / 2);
    ctx.fillStyle = '#fff';
    ctx.font = big + 'px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText('Mean Curvature (1/\u00b5m)', 0, 0);
    ctx.restore();
  } else {
    const labels = [['concave', '#c0392b'], ['saddle', '#e0b040'], ['convex', '#4a80a8']];
    ctx.font = small + 'px system-ui';
    ctx.textAlign = 'left';
    labels.forEach((entry, i) => {
      const y = by + i * small * 2.0;
      ctx.fillStyle = entry[1];
      ctx.fillRect(bx, y - small * 0.5, small, small);
      ctx.fillStyle = '#fff';
      ctx.fillText(entry[0], bx + small * 1.5, y);
    });
  }
  ctx.textBaseline = 'alphabetic';
  ctx.textAlign = 'left';
}

function paintMesh(f, asGhost) {
  const q = DATA.quant;
  // Centre on the mesh panel, not the canvas: the colorbar column takes the right and
  // the time stamp the top, so a naive W/2, H/2 would push the object under both.
  const big = labelPx();
  const panelW = W - colorbarWidth();
  const panelTop = big * 1.6, panelBottom = H - big * 2.2;
  const scale = Math.min(panelW, panelBottom - panelTop) / (1.9 * DATA.radius) * zoom;
  const cx = panelW / 2, cy = (panelTop + panelBottom) / 2;

  if (px.length < f.nv) {
    px = new Float32Array(f.nv); py = new Float32Array(f.nv); pz = new Float32Array(f.nv);
  }
  for (let i = 0; i < f.nv; i++) {
    const x = f.verts[i * 3] * q, y = f.verts[i * 3 + 1] * q, z = f.verts[i * 3 + 2] * q;
    px[i] = cx + (R[0] * x + R[1] * y + R[2] * z) * scale;
    py[i] = cy - (R[6] * x + R[7] * y + R[8] * z) * scale;
    pz[i] = R[3] * x + R[4] * y + R[5] * z;
  }
  if (depth.length < f.nf) { depth = new Float32Array(f.nf); }
  if (idx.length !== f.nf) { idx = new Array(f.nf); }
  for (let t = 0; t < f.nf; t++) {
    const a = f.faces[t * 3], b = f.faces[t * 3 + 1], c = f.faces[t * 3 + 2];
    depth[t] = pz[a] + pz[b] + pz[c];
    idx[t] = t;
  }
  idx.sort((p, q2) => depth[q2] - depth[p]);   // painter's algorithm: far first

  const pal = (mode === 'curv') ? palette : classPalette;
  const src = (mode === 'curv') ? f.curvIdx : f.classIdx;
  for (let n = 0; n < f.nf; n++) {
    const t = idx[n];
    const a = f.faces[t * 3], b = f.faces[t * 3 + 1], c = f.faces[t * 3 + 2];
    const ci = src[t] * 3;
    let r = pal[ci], g = pal[ci + 1], bl = pal[ci + 2];
    if (asGhost) { r = 96; g = 96; bl = 100; }
    if (shading) {
      const ux = px[b] - px[a], uy = py[b] - py[a];
      const vx = px[c] - px[a], vy = py[c] - py[a];
      const sh = 0.72 + 0.28 * Math.min(1, Math.abs(ux * vy - uy * vx) / 900);
      r *= sh; g *= sh; bl *= sh;
    }
    ctx.fillStyle = 'rgb(' + (r | 0) + ',' + (g | 0) + ',' + (bl | 0) + ')';
    ctx.beginPath();
    ctx.moveTo(px[a], py[a]); ctx.lineTo(px[b], py[b]); ctx.lineTo(px[c], py[c]);
    ctx.closePath(); ctx.fill();
  }
}

// --- per-frame readouts -----------------------------------------------------
function stats() {
  const f = FR[cur], s = f.stats;
  document.getElementById('flabel').textContent =
    'frame ' + f.frame + ' / ' + FR[N - 1].frame + '  (' + (f.frame - 1) + ' min)';
  document.getElementById('ttl').textContent = DATA.series + ', timepoint ' + f.frame;
  document.getElementById('s-vol').textContent = s.volume.toFixed(1) + ' \u00b5m\u00b3';
  document.getElementById('s-sa').textContent = s.sa.toFixed(1) + ' \u00b5m\u00b2';
  document.getElementById('s-nf').textContent = f.nf;
  document.getElementById('s-sph').textContent = s.sph.toFixed(3);
  document.getElementById('s-invag').textContent = s.invag.toFixed(3);
  document.getElementById('s-curv').textContent = s.meanCurv.toFixed(4) + ' 1/\u00b5m';
  if (mode !== 'curv') legend();
}

// --- sparkline of the whole series, with the current frame marked -----------
function sparkline() {
  const sp = document.getElementById('spark'), c2 = sp.getContext('2d');
  const w = sp.width, h = sp.height, padL = 46 * dpr, padR = 8 * dpr,
        padT = 12 * dpr, padB = 20 * dpr;
  c2.clearRect(0, 0, w, h);
  const series = [
    { key: 'volume', color: '#4a9ad4', label: 'volume' },
    { key: 'invag', color: '#e08a40', label: 'invagination' },
  ];
  series.forEach((s, si) => {
    const vals = FR.map(f => f.stats[s.key]);
    const lo = Math.min(...vals), hi = Math.max(...vals), rng = (hi - lo) || 1;
    c2.strokeStyle = s.color; c2.lineWidth = 2 * dpr; c2.beginPath();
    FR.forEach((f, i) => {
      const x = padL + (w - padL - padR) * (i / (N - 1));
      const y = padT + (h - padT - padB) * (1 - (vals[i] - lo) / rng) * 0.5 + si * (h - padT - padB) * 0.5;
      i ? c2.lineTo(x, y) : c2.moveTo(x, y);
    });
    c2.stroke();
    c2.fillStyle = s.color; c2.font = (11 * dpr) + 'px system-ui';
    c2.fillText(s.label, 4 * dpr, padT + si * (h - padT - padB) * 0.5 + 10 * dpr);
  });
  const x = padL + (w - padL - padR) * (cur / (N - 1));
  c2.strokeStyle = '#fff'; c2.lineWidth = 1 * dpr;
  c2.beginPath(); c2.moveTo(x, padT * 0.4); c2.lineTo(x, h - padB * 0.6); c2.stroke();
  c2.fillStyle = '#888'; c2.font = (10.5 * dpr) + 'px system-ui';
  c2.fillText('t=1', padL, h - 5 * dpr);
  c2.fillText('t=' + FR[N - 1].frame, w - padR - 26 * dpr, h - 5 * dpr);
}

function show(i) {
  cur = Math.max(0, Math.min(N - 1, i));
  document.getElementById('slider').value = cur;
  document.querySelectorAll('#framebar button').forEach((b, j) => {
    b.classList.toggle('on', j === cur);
    b.classList.toggle('ref', ghost && j === refFrame && j !== cur);
  });
  stats(); draw(); sparkline();
}

// One button per timepoint -- discrete frame switching alongside the slider.
const framebar = document.getElementById('framebar');
FR.forEach((f, i) => {
  const b = document.createElement('button');
  b.textContent = 't' + f.frame;
  b.addEventListener('click', () => { playing = false; syncPlay(); show(i); });
  framebar.appendChild(b);
});

const ghostBtn = document.getElementById('ghost');
const pinBtn = document.getElementById('pin');
ghostBtn.addEventListener('click', () => {
  ghost = !ghost;
  ghostBtn.textContent = 'ghost: ' + (ghost ? 't' + FR[refFrame].frame : 'off');
  ghostBtn.classList.toggle('on', ghost);
  show(cur);
});
pinBtn.addEventListener('click', () => {
  refFrame = cur;
  pinBtn.textContent = 'pin t' + FR[refFrame].frame;
  if (ghost) ghostBtn.textContent = 'ghost: t' + FR[refFrame].frame;
  show(cur);
});
document.getElementById('speed').addEventListener('change', e => {
  stepMs = +e.target.value;
});

// --- interaction ------------------------------------------------------------
let dragging = false, lx = 0, ly = 0;
canvas.addEventListener('pointerdown', e => {
  dragging = true; lx = e.clientX; ly = e.clientY;
  canvas.classList.add('drag'); canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener('pointermove', e => {
  if (!dragging) return;
  rotate((e.clientX - lx) * 0.01, (e.clientY - ly) * 0.01);
  lx = e.clientX; ly = e.clientY; draw();
});
canvas.addEventListener('pointerup', () => { dragging = false; canvas.classList.remove('drag'); });
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  zoom = Math.max(0.3, Math.min(6, zoom * Math.exp(-e.deltaY * 0.0012)));
  draw();
}, { passive: false });
canvas.addEventListener('dblclick', () => { setBasis('yz'); zoom = 1; draw(); });

document.querySelectorAll('[data-view]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('[data-view]').forEach(b => b.classList.remove('on'));
    btn.classList.add('on'); setBasis(btn.dataset.view); draw();
  });
});
const modeBtn = document.getElementById('mode');
modeBtn.addEventListener('click', () => {
  mode = (mode === 'curv') ? 'class' : 'curv';
  modeBtn.textContent = 'colour: ' + (mode === 'curv' ? 'mean curvature' : 'concavity class');
  legend(); draw();
});
const shadeBtn = document.getElementById('shade');
shadeBtn.addEventListener('click', () => {
  shading = !shading; shadeBtn.classList.toggle('on', shading); draw();
});
const spinBtn = document.getElementById('spin');
spinBtn.addEventListener('click', () => {
  spinning = !spinning; spinBtn.classList.toggle('on', spinning);
  if (spinning) requestAnimationFrame(spinTick);
});
function spinTick() { if (!spinning) return; rotate(0.012, 0); draw(); requestAnimationFrame(spinTick); }

document.getElementById('slider').addEventListener('input', e => show(+e.target.value));
const playBtn = document.getElementById('play');
function syncPlay() {
  playBtn.innerHTML = playing ? '&#10073;&#10073; pause' : '&#9654; play';
  playBtn.classList.toggle('on', playing);
}
playBtn.addEventListener('click', () => {
  playing = !playing; syncPlay();
  if (playing) requestAnimationFrame(playTick);
});
let lastStep = 0;
function playTick(ts) {
  if (!playing) return;
  if (!ts || ts - lastStep > stepMs) { lastStep = ts || 0; show((cur + 1) % N); }
  requestAnimationFrame(playTick);
}
window.addEventListener('keydown', e => {
  if (e.key === 'ArrowRight') show(cur + 1);
  else if (e.key === 'ArrowLeft') show(cur - 1);
});

// --- movie export -----------------------------------------------------------
// MediaRecorder over canvas.captureStream(): whatever is on screen -- current camera,
// zoom, colour mode, shading, ghost -- is what gets recorded, which is the point.
// GIF is deliberately not produced here: encoding one in the browser needs an LZW
// encoder and would be worse than the two-pass palette GIF scripts/export_mesh_movie.py
// already writes. "copy view" hands this exact camera to that script.
const recBtn = document.getElementById('record');
const recNote = document.getElementById('recnote');

function pickMimeType() {
  const wanted = [
    'video/mp4;codecs=avc1',            // PowerPoint's happiest case
    'video/webm;codecs=vp9',
    'video/webm;codecs=vp8',
    'video/webm',
  ];
  for (const type of wanted) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(type)) return type;
  }
  return null;
}

let recording = false;
recBtn.addEventListener('click', () => {
  if (recording) return;
  const mime = pickMimeType();
  if (!mime) {
    recNote.className = 'note warn';
    recNote.textContent = 'This browser cannot record canvas video. '
      + 'Use "copy view" and scripts/export_mesh_movie.py instead.';
    return;
  }

  const stream = canvas.captureStream(30);
  const chunks = [];
  let recorder;
  try {
    recorder = new MediaRecorder(stream, {mimeType: mime, videoBitsPerSecond: 12000000});
  } catch (err) {
    recNote.className = 'note warn';
    recNote.textContent = 'Recorder failed to start: ' + err.message;
    return;
  }
  recorder.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
  recorder.onstop = () => {
    const ext = mime.startsWith('video/mp4') ? 'mp4' : 'webm';
    const blob = new Blob(chunks, {type: mime});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = DATA.series + '_timelapse.' + ext;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
    recording = false;
    recBtn.classList.remove('rec');
    recBtn.innerHTML = '&#9679; record movie';
    recNote.className = 'note';
    recNote.textContent = 'saved ' + a.download + '  ('
      + (blob.size / 1e6).toFixed(1) + ' MB, ' + ext.toUpperCase() + ')';
  };

  // Stop any animation that would fight the recording, then walk the frames.
  playing = false; syncPlay();
  spinning = false; spinBtn.classList.remove('on');
  recording = true;
  recBtn.classList.add('rec');
  recBtn.textContent = 'recording...';
  recNote.className = 'note';
  recorder.start();

  let i = 0;
  show(0);
  const hold = Math.max(stepMs, 120);
  const tick = () => {
    if (i >= N) {
      // A short tail so the last frame is not clipped by the muxer.
      setTimeout(() => recorder.stop(), 250);
      return;
    }
    show(i);
    recNote.textContent = 'recording frame ' + (i + 1) + ' / ' + N + '...';
    i += 1;
    setTimeout(tick, hold);
  };
  tick();
});

document.getElementById('copyview').addEventListener('click', () => {
  // Row-major R plus zoom -- exactly what export_mesh_movie.py's --view expects.
  const payload = JSON.stringify({
    R: Array.from(R).map(v => +v.toFixed(6)),
    zoom: +zoom.toFixed(4),
  });
  const done = () => {
    recNote.className = 'note';
    recNote.textContent = 'view copied: ' + payload;
  };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(payload).then(done, done);
  } else {
    done();
  }
});

function legend() {
  const el = document.getElementById('legend');
  if (mode === 'curv') {
    const stops = [];
    for (let i = 0; i <= 10; i++) {
      const ci = Math.round(i * 25.5) * 3;
      stops.push('rgb(' + palette[ci] + ',' + palette[ci + 1] + ',' + palette[ci + 2] +
                 ') ' + (i * 10) + '%');
    }
    el.innerHTML = '<span>' + (-DATA.lim).toFixed(2) + '</span>' +
      '<div class="ramp" style="background:linear-gradient(90deg,' + stops.join(',') + ')"></div>' +
      '<span>' + DATA.lim.toFixed(2) + ' 1/\u00b5m</span>' +
      '<span style="margin-left:10px">blue = concave (invagination), red = convex' +
      ' &mdash; shared across all frames</span>';
  } else {
    const s = FR[cur].stats;
    el.innerHTML =
      '<span><i class="sw" style="background:#c0392b"></i>concave (' + s.nConcave + ')</span>' +
      '<span><i class="sw" style="background:#e0b040"></i>saddle (' + s.nSaddle + ')</span>' +
      '<span><i class="sw" style="background:#4a80a8"></i>convex (' + s.nConvex + ')</span>';
  }
}

document.getElementById('slider').max = N - 1;
setBasis('yz'); legend(); stats();
window.addEventListener('resize', resize);
window.addEventListener('load', resize);
(function boot() {
  if (canvas.parentElement.clientWidth > 0) { resize(); }
  else { requestAnimationFrame(boot); }
})();
</script>
"""

html = html.replace("__PAYLOAD__", json.dumps(payload))
path = os.path.join(OUT, f"{SERIES} Mesh Viewer (all timepoints).html")
with open(path, "w", encoding="utf-8") as fh:
    fh.write(html)
print(f"wrote {path} ({os.path.getsize(path) / 1024:.0f} KB)")
assert not [c for c in set(html) if ord(c) > 127], "non-ASCII leaked into the page"
