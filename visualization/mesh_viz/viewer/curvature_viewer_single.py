"""Rotatable single-object curvature viewer -> one self-contained HTML page.

Meshes one segmented object, computes per-face mean curvature, and writes a standalone
widget: drag to rotate, scroll to zoom, double-click to reset, toggle curvature vs
concavity class, shading, spin, and a spin->movie recorder. In-canvas colorbar + scale bar
so a screenshot carries its own scale. The page is pure ASCII (asserted at build time).

This is the single-mesh sibling of curvature_viewer.py (which scrubs a timepoint series).
Edit the CONFIG block for a different object. Outputs go to a data drive, never C:.
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
import tifffile
from scipy import ndimage as ndi

from analysis.volumetric.mesh import mesh_object, write_obj, largest_component
from analysis.volumetric.curvature import (CONCAVE, CONVEX, HYPERBOLOID,
                                           analyze_curvature)

# ---- CONFIG -----------------------------------------------------------------
SEG = (r"Y:\User_data\Kiet\03102026_coculture_hek_CD19gfp_Hoescht_CAR_HA_APC"
       r"\FMC_30min_\converted\cells\channels\C2-cell1_Simple Segmentation.tiff")
OUT = (r"Y:\User_data\Kiet\03102026_coculture_hek_CD19gfp_Hoescht_CAR_HA_APC"
       r"\FMC_30min_\converted\cells\channels\mesh_test")
LABEL = "C2-cell1"                 # object mask = (segmentation == OBJECT_VALUE)
OBJECT_VALUE = 1
XY_UM, Z_UM = 0.065, 0.3           # NB: file's dCalibration 0.1625 is WRONG (user-corrected)
CLIM = 0.5                         # mean-curvature colour limit, 1/um (fixed, not data-derived)
SCALEBAR_UM = 5.0
TITLE = "Kiet coculture (HEK / CAR) — " + LABEL
SUBTITLE = ("Cell surface mesh coloured by mean curvature. Drag to rotate, scroll to zoom, "
            "double-click to reset. xy 0.065, z 0.3 " + "µm; isotropic-resampled.")
# -----------------------------------------------------------------------------

os.makedirs(OUT, exist_ok=True)


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


# 1. mask -> isotropic -> mesh -> curvature
seg = tifffile.imread(SEG)
mask = largest_component(seg == OBJECT_VALUE)
mask_iso = ndi.zoom(mask.astype(np.uint8), (Z_UM / XY_UM, 1, 1), order=0).astype(bool)
print(f"mask {int(mask.sum())} vox -> iso {mask_iso.shape}", flush=True)

om = mesh_object(mask_iso, (XY_UM, XY_UM, XY_UM), maxrad=5.0, verbose=False)
g = om.geometry
# write_obj reorders (z,y,x) -> (x,y,z); the viewer works in that OBJ frame, so curvature
# and geometry use the same (x,y,z) convention (z is axis 2), matching curvature_viewer.py.
V = om.vertices_um[:, ::-1].astype(np.float64)     # (x, y, z)
F = om.faces.astype(np.int64)
curv = analyze_curvature(V, F, z_axis=2)
write_obj(os.path.join(OUT, f"{LABEL}.obj"), om.vertices_um, om.faces)
print(f"faces {F.shape[0]}  vol {g.volume_um3:.1f}  sph {g.sphericity:.3f}  "
      f"sol {g.solidity:.3f}  AR {g.aspect_ratio:.3f}  invag {curv.invagination_ratio:.3f}",
      flush=True)

# 2. quantise for a small self-contained page
centre = (V.min(axis=0) + V.max(axis=0)) / 2
radius = float(np.abs(V - centre).max())
quant = radius / 32000.0
qv = np.rint((V - centre) / quant).astype(np.int16)
assert F.max() <= 65535 and F.min() >= 1

k = curv.k_mean_faces
scaled = (np.clip(k, -CLIM, CLIM) + CLIM) / (2 * CLIM)
assert np.isfinite(scaled).all()

cmap = plt.get_cmap("RdBu_r")
palette = (np.array([cmap(i / 255.0)[:3] for i in range(256)]) * 255).astype(np.uint8)
class_palette = np.zeros((256, 3), dtype=np.uint8)
class_palette[:3] = np.array([[192, 57, 43], [224, 176, 64], [74, 128, 168]], np.uint8)

payload = {
    "label": LABEL, "title": TITLE, "subtitle": SUBTITLE,
    "verts": b64(qv), "faces": b64(F.astype(np.uint16) - 1),
    "nv": int(V.shape[0]), "nf": int(F.shape[0]),
    "curvIdx": b64(np.clip(np.rint(scaled * 255.0), 0, 255).astype(np.uint8)),
    "classIdx": b64(curv.concavity_classes.astype(np.uint8)),
    "palette": b64(palette), "classPalette": b64(class_palette),
    "basis": {"xz": ((0, -1), (1, -1), (2, 1)), "xz_rev": ((0, 1), (1, 1), (2, 1)),
              "yz": ((1, 1), (0, 1), (2, 1)), "yz_rev": ((1, -1), (0, -1), (2, 1))},
    "lim": CLIM, "radius": radius, "quant": quant, "scalebar": SCALEBAR_UM,
    "stats": {
        "volume": round(g.volume_um3, 2), "sa": round(g.surface_area_um2, 2),
        "sph": round(g.sphericity, 4), "sol": round(g.solidity, 4),
        "ar": round(g.aspect_ratio, 4), "invag": round(curv.invagination_ratio, 4),
        "meanCurv": round(float(curv.mean_curvature), 5), "nf": int(F.shape[0]),
        "nConcave": int((curv.concavity_classes == CONCAVE).sum()),
        "nSaddle": int((curv.concavity_classes == HYPERBOLOID).sum()),
        "nConvex": int((curv.concavity_classes == CONVEX).sum()),
    },
}

HTML = r"""<meta charset="utf-8">
<title>Mesh Viewer</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#000; color:#ddd;
         font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif; }
  .wrap { max-width:1000px; margin:0 auto; padding:18px 16px 40px; }
  h1 { font-size:18px; font-weight:600; margin:0 0 4px; color:#fff; }
  .sub { color:#999; font-size:13px; margin-bottom:14px; }
  .stage { position:relative; background:#000; border:1px solid #222; border-radius:8px;
           overflow:hidden; }
  canvas { display:block; width:100%; height:auto; cursor:grab; touch-action:none; }
  canvas.drag { cursor:grabbing; }
  .bar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:12px 0 4px; }
  button { background:#1b1b1b; color:#ddd; border:1px solid #333; border-radius:6px;
           padding:6px 12px; font:inherit; font-size:13px; cursor:pointer; }
  button:hover { background:#262626; }
  button.on { background:#2d4a63; border-color:#4a7ba3; color:#fff; }
  button.rec { background:#6a2020; border-color:#b34040; color:#fff; }
  .sep { width:1px; height:22px; background:#333; margin:0 4px; }
  .note { color:#8a8a90; font-size:12.5px; min-height:18px; margin:2px 0 6px; }
  .note.warn { color:#e0a040; }
  table { border-collapse:collapse; margin-top:14px; font-size:13px; }
  td { padding:3px 20px 3px 0; color:#bbb; font-variant-numeric:tabular-nums; }
  td.k { color:#888; }
  .hint { color:#777; font-size:12.5px; margin-top:14px; }
</style>
<div class="wrap">
  <h1 id="ttl"></h1>
  <div class="sub" id="sub"></div>
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
    <button id="record" title="record a spin and download the movie">&#9679; record spin</button>
    <button id="copyview" title="copy this camera for scripts/export_mesh_movie.py">copy view</button>
  </div>
  <div class="note" id="note"></div>
  <div class="stage"><canvas id="c"></canvas></div>
  <table>
    <tr><td class="k">volume</td><td id="s-vol"></td>
        <td class="k">surface area</td><td id="s-sa"></td>
        <td class="k">faces</td><td id="s-nf"></td></tr>
    <tr><td class="k">sphericity</td><td id="s-sph"></td>
        <td class="k">solidity</td><td id="s-sol"></td>
        <td class="k">aspect ratio</td><td id="s-ar"></td></tr>
    <tr><td class="k">invagination ratio</td><td id="s-invag"></td>
        <td class="k">mean curvature</td><td id="s-curv"></td>
        <td class="k">concave / saddle / convex</td><td id="s-cls"></td></tr>
  </table>
  <div class="hint">Orientations match the MATLAB pipeline: xz = view([90 0]),
    xz_rev = view([-90 0]), yz = view([0 0]), yz_rev = view([180 0]); orthographic,
    equal aspect. Colour scale is fixed at +/-__LIM__ 1/um.</div>
</div>
<script>
const DATA = __PAYLOAD__;
function unpack(b64, Type){ const bin=atob(b64), a=new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++) a[i]=bin.charCodeAt(i); return new Type(a.buffer); }
const palette=unpack(DATA.palette,Uint8Array), classPalette=unpack(DATA.classPalette,Uint8Array);
const M={ nv:DATA.nv, nf:DATA.nf, verts:unpack(DATA.verts,Int16Array),
  faces:unpack(DATA.faces,Uint16Array), curvIdx:unpack(DATA.curvIdx,Uint8Array),
  classIdx:unpack(DATA.classIdx,Uint8Array) };
let mode='curv', shading=true, spinning=false;
const canvas=document.getElementById('c'), ctx=canvas.getContext('2d');
let W=0,H=0,dpr=Math.min(window.devicePixelRatio||1,2);
let R=new Float64Array(9), zoom=1;

function setBasis(name){ const b=DATA.basis[name]; R.fill(0);
  for(let r=0;r<3;r++) R[r*3+b[r][0]]=b[r][1]; }
function rotate(dx,dy){
  const ax=[R[0],R[1],R[2]],ay=[R[3],R[4],R[5]],az=[R[6],R[7],R[8]];
  const rot=(u,v,a)=>{ const c=Math.cos(a),s=Math.sin(a);
    for(let i=0;i<3;i++){ const t=c*u[i]+s*v[i]; v[i]=-s*u[i]+c*v[i]; u[i]=t; } };
  rot(ax,ay,dx); rot(az,ay,dy); R.set([...ax,...ay,...az]);
}
function resize(){ const cssW=canvas.parentElement.clientWidth;
  canvas.style.height=Math.round(cssW*0.78)+'px';
  W=canvas.width=Math.round(cssW*dpr); H=canvas.height=Math.round(cssW*0.78*dpr); draw(); }

let px=new Float32Array(0),py=new Float32Array(0),pz=new Float32Array(0);
let depth=new Float32Array(0), idx=[];
function labelPx(){ return Math.max(12*dpr, Math.round(W*0.030)); }
function tickPx(){ return Math.max(9*dpr, Math.round(W*0.019)); }
function colorbarWidth(){ return labelPx()*1.5 + tickPx()*3.0; }
function meshCentreX(){ return (W-colorbarWidth())/2; }

function draw(){
  ctx.fillStyle='#000'; ctx.fillRect(0,0,W,H);
  paintMesh(M);
  const big=labelPx();
  const panelW=W-colorbarWidth(), panelTop=big*1.6, panelBottom=H-big*2.2;
  const scaleBar=DATA.scalebar*(Math.min(panelW,panelBottom-panelTop)/(1.9*DATA.radius)*zoom);
  ctx.strokeStyle='#fff'; ctx.lineWidth=Math.max(3,big*0.09);
  const barY=H-big*1.5;
  ctx.beginPath(); ctx.moveTo(big*0.5,barY); ctx.lineTo(big*0.5+scaleBar,barY); ctx.stroke();
  ctx.fillStyle='#fff'; ctx.font=big+'px system-ui'; ctx.textAlign='left'; ctx.textBaseline='bottom';
  ctx.fillText(DATA.scalebar+' µm', big*0.5, barY-big*0.3);
  ctx.textBaseline='alphabetic';
  colorbar();
}
function colorbar(){
  const big=labelPx(), small=tickPx();
  const bw=Math.max(14*dpr,big*0.42), bh=Math.min(H*0.60,H-big*3.0);
  const labelX=W-big*2.5, bx=labelX-small*4.6-bw, by=(H-bh)/2;
  ctx.textBaseline='middle';
  if(mode==='curv'){
    const grad=ctx.createLinearGradient(0,by+bh,0,by);
    for(let i=0;i<=32;i++){ const ci=Math.round((i/32)*255)*3;
      grad.addColorStop(i/32,'rgb('+palette[ci]+','+palette[ci+1]+','+palette[ci+2]+')'); }
    ctx.fillStyle=grad; ctx.fillRect(bx,by,bw,bh);
    ctx.strokeStyle='#fff'; ctx.lineWidth=1*dpr; ctx.strokeRect(bx,by,bw,bh);
    ctx.font=small+'px system-ui'; const lim=DATA.lim;
    for(let i=0;i<=4;i++){ const frac=i/4, y=by+bh*(1-frac), value=-lim+2*lim*frac;
      ctx.strokeStyle='#ddd'; ctx.beginPath();
      ctx.moveTo(bx+bw,y); ctx.lineTo(bx+bw+small*0.35,y); ctx.stroke();
      ctx.fillStyle='#fff'; ctx.textAlign='left'; ctx.fillText(value.toFixed(2),bx+bw+small*0.6,y); }
    ctx.save(); ctx.translate(labelX,by+bh/2); ctx.rotate(Math.PI/2);
    ctx.fillStyle='#fff'; ctx.font=big+'px system-ui'; ctx.textAlign='center';
    ctx.fillText('Mean Curvature (1/µm)',0,0); ctx.restore();
  } else {
    const labels=[['concave','#c0392b'],['saddle','#e0b040'],['convex','#4a80a8']];
    ctx.font=small+'px system-ui'; ctx.textAlign='left';
    labels.forEach((e,i)=>{ const y=by+i*small*2.0;
      ctx.fillStyle=e[1]; ctx.fillRect(bx,y-small*0.5,small,small);
      ctx.fillStyle='#fff'; ctx.fillText(e[0],bx+small*1.5,y); });
  }
  ctx.textBaseline='alphabetic'; ctx.textAlign='left';
}
function paintMesh(f){
  const q=DATA.quant, big=labelPx();
  const panelW=W-colorbarWidth(), panelTop=big*1.6, panelBottom=H-big*2.2;
  const scale=Math.min(panelW,panelBottom-panelTop)/(1.9*DATA.radius)*zoom;
  const cx=panelW/2, cy=(panelTop+panelBottom)/2;
  if(px.length<f.nv){ px=new Float32Array(f.nv); py=new Float32Array(f.nv); pz=new Float32Array(f.nv); }
  for(let i=0;i<f.nv;i++){ const x=f.verts[i*3]*q,y=f.verts[i*3+1]*q,z=f.verts[i*3+2]*q;
    px[i]=cx+(R[0]*x+R[1]*y+R[2]*z)*scale;
    py[i]=cy-(R[6]*x+R[7]*y+R[8]*z)*scale;
    pz[i]=R[3]*x+R[4]*y+R[5]*z; }
  if(depth.length<f.nf){ depth=new Float32Array(f.nf); }
  if(idx.length!==f.nf){ idx=new Array(f.nf); }
  for(let t=0;t<f.nf;t++){ const a=f.faces[t*3],b=f.faces[t*3+1],c=f.faces[t*3+2];
    depth[t]=pz[a]+pz[b]+pz[c]; idx[t]=t; }
  idx.sort((p,q2)=>depth[q2]-depth[p]);
  const pal=(mode==='curv')?palette:classPalette, src=(mode==='curv')?f.curvIdx:f.classIdx;
  for(let n=0;n<f.nf;n++){ const t=idx[n];
    const a=f.faces[t*3],b=f.faces[t*3+1],c=f.faces[t*3+2], ci=src[t]*3;
    let r=pal[ci],g=pal[ci+1],bl=pal[ci+2];
    if(shading){ const ux=px[b]-px[a],uy=py[b]-py[a],vx=px[c]-px[a],vy=py[c]-py[a];
      const sh=0.72+0.28*Math.min(1,Math.abs(ux*vy-uy*vx)/900); r*=sh; g*=sh; bl*=sh; }
    ctx.fillStyle='rgb('+(r|0)+','+(g|0)+','+(bl|0)+')';
    ctx.beginPath(); ctx.moveTo(px[a],py[a]); ctx.lineTo(px[b],py[b]); ctx.lineTo(px[c],py[c]);
    ctx.closePath(); ctx.fill(); }
}
function fillStats(){ const s=DATA.stats;
  document.getElementById('ttl').textContent=DATA.title;
  document.getElementById('sub').textContent=DATA.subtitle;
  document.getElementById('s-vol').textContent=s.volume.toFixed(1)+' µm³';
  document.getElementById('s-sa').textContent=s.sa.toFixed(1)+' µm²';
  document.getElementById('s-nf').textContent=s.nf;
  document.getElementById('s-sph').textContent=s.sph.toFixed(3);
  document.getElementById('s-sol').textContent=s.sol.toFixed(3);
  document.getElementById('s-ar').textContent=s.ar.toFixed(3);
  document.getElementById('s-invag').textContent=s.invag.toFixed(3);
  document.getElementById('s-curv').textContent=s.meanCurv.toFixed(4)+' 1/µm';
  document.getElementById('s-cls').textContent=s.nConcave+' / '+s.nSaddle+' / '+s.nConvex;
}

let dragging=false,lx=0,ly=0;
canvas.addEventListener('pointerdown',e=>{ dragging=true; lx=e.clientX; ly=e.clientY;
  canvas.classList.add('drag'); canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener('pointermove',e=>{ if(!dragging) return;
  rotate((e.clientX-lx)*0.01,(e.clientY-ly)*0.01); lx=e.clientX; ly=e.clientY; draw(); });
canvas.addEventListener('pointerup',()=>{ dragging=false; canvas.classList.remove('drag'); });
canvas.addEventListener('wheel',e=>{ e.preventDefault();
  zoom=Math.max(0.3,Math.min(6,zoom*Math.exp(-e.deltaY*0.0012))); draw(); },{passive:false});
canvas.addEventListener('dblclick',()=>{ setBasis('yz'); zoom=1; draw(); });
document.querySelectorAll('[data-view]').forEach(btn=>{ btn.addEventListener('click',()=>{
  document.querySelectorAll('[data-view]').forEach(b=>b.classList.remove('on'));
  btn.classList.add('on'); setBasis(btn.dataset.view); draw(); }); });
const modeBtn=document.getElementById('mode');
modeBtn.addEventListener('click',()=>{ mode=(mode==='curv')?'class':'curv';
  modeBtn.textContent='colour: '+(mode==='curv'?'mean curvature':'concavity class'); draw(); });
const shadeBtn=document.getElementById('shade');
shadeBtn.addEventListener('click',()=>{ shading=!shading; shadeBtn.classList.toggle('on',shading); draw(); });
const spinBtn=document.getElementById('spin');
spinBtn.addEventListener('click',()=>{ spinning=!spinning; spinBtn.classList.toggle('on',spinning);
  if(spinning) requestAnimationFrame(spinTick); });
function spinTick(){ if(!spinning) return; rotate(0.02,0); draw(); requestAnimationFrame(spinTick); }

// spin -> movie: MediaRecorder over the canvas; whatever is on screen is what records.
const recBtn=document.getElementById('record'), note=document.getElementById('note');
function pickMime(){ for(const t of ['video/mp4;codecs=avc1','video/webm;codecs=vp9',
  'video/webm;codecs=vp8','video/webm']){ if(window.MediaRecorder&&MediaRecorder.isTypeSupported(t)) return t; } return null; }
let recording=false;
recBtn.addEventListener('click',()=>{ if(recording) return;
  const mime=pickMime();
  if(!mime){ note.className='note warn'; note.textContent='This browser cannot record canvas video. Use the .avi/.gif written by export_mesh_movie.py.'; return; }
  const stream=canvas.captureStream(30), chunks=[];
  let rec; try{ rec=new MediaRecorder(stream,{mimeType:mime,videoBitsPerSecond:12000000}); }
  catch(err){ note.className='note warn'; note.textContent='Recorder failed: '+err.message; return; }
  rec.ondataavailable=e=>{ if(e.data.size) chunks.push(e.data); };
  rec.onstop=()=>{ const ext=mime.startsWith('video/mp4')?'mp4':'webm';
    const blob=new Blob(chunks,{type:mime}), url=URL.createObjectURL(blob);
    const a=document.createElement('a'); a.href=url; a.download=DATA.label+'_spin.'+ext; a.click();
    setTimeout(()=>URL.revokeObjectURL(url),10000);
    recording=false; recBtn.classList.remove('rec'); recBtn.innerHTML='&#9679; record spin';
    note.className='note'; note.textContent='saved '+a.download+'  ('+(blob.size/1e6).toFixed(1)+' MB)'; };
  spinning=false; spinBtn.classList.remove('on'); recording=true;
  recBtn.classList.add('rec'); recBtn.textContent='recording...';
  rec.start();
  let angle=0; setBasis('yz'); zoom=1;
  const tick=()=>{ if(angle>=360){ setTimeout(()=>rec.stop(),250); return; }
    rotate(0.10472,0); draw(); angle+=6;
    note.className='note'; note.textContent='recording spin... '+Math.round(angle/3.6)+'%';
    setTimeout(tick,33); };
  tick();
});
document.getElementById('copyview').addEventListener('click',()=>{
  const payload=JSON.stringify({R:Array.from(R).map(v=>+v.toFixed(6)),zoom:+zoom.toFixed(4)});
  note.className='note'; note.textContent='view copied: '+payload;
  if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(payload); });

document.title=DATA.title; setBasis('yz'); fillStats();
window.addEventListener('resize',resize); window.addEventListener('load',resize);
(function boot(){ if(canvas.parentElement.clientWidth>0){ resize(); } else { requestAnimationFrame(boot); } })();
</script>
"""

# json.dumps(ensure_ascii=True) already escapes the title/subtitle/stats, so payload text
# is safe whatever charset serves the page. The only literal non-ASCII left is in the JS
# string bodies (µm, µm³, ...) -- escape those to \uXXXX so the whole page is pure ASCII,
# exactly as the standalone ascii_fix.py does for the series viewer.
html = HTML.replace("__PAYLOAD__", json.dumps(payload)).replace("__LIM__", f"{CLIM:g}")
BS = chr(92)
for cp in (0xB5, 0xB3, 0xB2):                       # µ, ³, ²
    html = html.replace(chr(cp), BS + "u%04x" % cp)
leftover = sorted({hex(ord(c)) for c in html if ord(c) > 127})
assert not leftover, f"non-ASCII leaked into the page: {leftover}"

path = os.path.join(OUT, f"{LABEL} Mesh Viewer (rotatable).html")
with open(path, "w", encoding="ascii") as fh:
    fh.write(html)
print(f"wrote {path} ({os.path.getsize(path) / 1024:.0f} KB)")
