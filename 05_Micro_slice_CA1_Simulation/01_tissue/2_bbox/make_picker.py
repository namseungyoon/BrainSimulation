# -*- coding: utf-8 -*-
"""
01_tissue/2_bbox/make_picker.py  —  Stage 2 보조: 창·전극 배치 인터랙티브 UI 생성 (2D)

slice400 세포를 국소좌표(종축 u · 층관통 r · 두께 w)로 투영 → 자립형 HTML(window_picker.html).
기능:
  - 세포(층별 색) 위에 창(사각형). 창 가로·세로·깊이·층관통중심·각도(회전) 수치 입력.
  - 창 프리셋: 층관통 / SP평면.
  - 전극 배열: 배치(행×열)·간격·직경 입력 → 강체 배열. 중심 이동 + 방향(각도) 회전.
  - 이름 지정 + 설정 내보내기(프레임+창+전극+물리xyz JSON) → config.
  - 실시간: 창 내부 세포수(회전·깊이 반영 3D 박스)·층별, 전극 위치·층.

실행: python 01_tissue/2_bbox/make_picker.py  →  window_picker.html
"""
import os
import glob
import json

import numpy as np
import h5py
import nrrd
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "data")
LAYERS = ["SO", "SP", "SR", "SLM"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
N_SUB = 7000


def find(p):
    return sorted(glob.glob(os.path.join(DATA, "**", p), recursive=True), key=len)[0]


def decode(g, name):
    lib = [s.decode() if isinstance(s, bytes) else s for s in g["@library"][name][:]]
    return np.array(lib, dtype=object)[g[name][:]]


def unit(v):
    n = np.linalg.norm(v); return v / n if n > 1e-9 else v


def best_center(u, r, w, Lu, Lr, Lw):
    """주어진 창 크기에서 세포가 가장 많이 담기는 중심(u,r)을 스캔."""
    ucs = np.linspace(u.min() + Lu / 2, u.max() - Lu / 2, 60)
    rcs = np.linspace(-50, 350, 40)
    win = np.abs(w) <= Lw / 2
    uu, rr = u[win], r[win]
    best = (-1, 0.0, 0.0)
    for uc in ucs:
        rr2 = rr[np.abs(uu - uc) <= Lu / 2]
        for rc in rcs:
            n = int((np.abs(rr2 - rc) <= Lr / 2).sum())
            if n > best[0]:
                best = (n, float(uc), float(rc))
    return best


def main():
    mask, h = nrrd.read(find(os.path.join("slices", "slice400.nrrd")))
    origin = np.asarray(h["space origin"], float); vs = float(h["space directions"][0][0])
    nx, ny, nz = mask.shape
    with h5py.File(find(os.path.join("hippocampus_neurons", "nodes.h5")), "r") as f:
        g = f["nodes/hippocampus_neurons/0"]
        xyz = np.stack([g["x"][:], g["y"][:], g["z"][:]], 1)
        quat = np.stack([g[f"orientation_{c}"][:] for c in "wxyz"], 1)
        layer = decode(g, "layer")

    idx = np.floor((xyz - origin) / vs).astype(int)
    ok = (idx >= 0).all(1) & (idx[:, 0] < nx) & (idx[:, 1] < ny) & (idx[:, 2] < nz)
    inside = np.zeros(len(xyz), bool); ii = idx[ok]
    inside[ok] = mask[ii[:, 0], ii[:, 1], ii[:, 2]] > 0
    sl = np.where(inside)[0]
    Cs, Ls, Qs = xyz[sl], layer[sl], quat[sl]

    c0 = Cs.mean(0)
    _, _, Vt = np.linalg.svd(Cs - c0, full_matrices=False)
    spans = (Cs - c0) @ Vt.T
    long_dir = Vt[int(np.argmax(spans.max(0) - spans.min(0)))]
    u_all = (Cs - c0) @ long_dir; u_c = np.median(u_all)
    band = np.abs(u_all - u_c) <= 400
    sp_band = band & (Ls == "SP")
    radial_dir = unit(Rot.from_quat(Qs[sp_band][:, [1, 2, 3, 0]]).apply([0, 1, 0]).mean(0))
    radial_dir = unit(radial_dir - (radial_dir @ long_dir) * long_dir)
    thick_dir = unit(np.cross(long_dir, radial_dir))
    seed = Cs[sp_band].mean(0)
    d = Cs - seed
    u = d @ long_dir; r = d @ radial_dir; w = d @ thick_dir

    # 가로/세로 기본값 = 각 크기에서 세포 최다 담기는 중심 자동탐색
    g_c = best_center(u, r, w, 800, 500, 400)
    v_c = best_center(u, r, w, 500, 800, 400)
    print(f"[기본값] 가로 800x500 → 중심(u={g_c[1]:.0f}, r={g_c[2]:.0f}) 세포 {g_c[0]:,}")
    print(f"[기본값] 세로 500x800 → 중심(u={v_c[1]:.0f}, r={v_c[2]:.0f}) 세포 {v_c[0]:,}")

    lyr_idx = np.array([LAYERS.index(x) for x in Ls])
    rng = np.random.default_rng(0)
    keep = rng.choice(len(u), size=min(N_SUB, len(u)), replace=False)
    cells = np.stack([u[keep].round(), r[keep].round(), w[keep].round(),
                      lyr_idx[keep]], 1).astype(int).tolist()

    data = {
        "cells": cells, "nTotalSlice": int(len(sl)),
        "bounds": {"umin": float(u.min()), "umax": float(u.max()),
                   "rmin": float(r.min()), "rmax": float(r.max()),
                   "wmin": float(w.min()), "wmax": float(w.max())},
        "layers": LAYERS, "layerColors": COLORS,
        "presets": {
            "가로": {"cu": round(g_c[1]), "cr": round(g_c[2]), "Lu": 800, "Lr": 500, "Lw": 400,
                    "arr": {"rows": 1, "cols": 3, "sp": 200, "ang": 0, "cr": 238}},
            "세로": {"cu": round(v_c[1]), "cr": round(v_c[2]), "Lu": 500, "Lr": 800, "Lw": 400,
                    "arr": {"rows": 3, "cols": 1, "sp": 200, "ang": 0, "cr": round(v_c[2])}}},
        "frame": {"seed": seed.tolist(), "long_dir": long_dir.tolist(),
                  "radial_dir": radial_dir.tolist(), "thick_dir": thick_dir.tolist()},
    }
    # 확정 config 있으면 초기상태로 임베드(새로고침해도 유지 + 불러오기 기준)
    cfg_path = os.path.join(ROOT, "config", "window_layout.json")
    if os.path.exists(cfg_path):
        data["saved"] = json.load(open(cfg_path, encoding="utf-8"))

    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(data, ensure_ascii=False))
    out = os.path.join(HERE, "window_picker.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[UI] 세포 {len(cells):,}개(서브샘플) · 슬랩 {len(sl):,} → {out}")


TEMPLATE = r'''<title>CA1 창·전극 배치기</title>
<style>
  :root{
    --bg:#eef1f5; --panel:#ffffff; --ink:#18212e; --muted:#5c6775; --line:#d7dde5;
    --field:#f4f6f9; --accent:#0d9488; --stim:#e23b3b; --rec:#1f2733;
  }
  @media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
    --bg:#0e131a; --panel:#161d26; --ink:#e7edf4; --muted:#8b97a8; --line:#28313d;
    --field:#0f1620; --accent:#2dd4bf; --stim:#ff5a5a; --rec:#e7edf4;
  }}
  :root[data-theme="dark"]{
    --bg:#0e131a; --panel:#161d26; --ink:#e7edf4; --muted:#8b97a8; --line:#28313d;
    --field:#0f1620; --accent:#2dd4bf; --stim:#ff5a5a; --rec:#e7edf4;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font-family:system-ui,-apple-system,"Malgun Gothic",sans-serif;font-size:13px}
  .wrap{display:flex;gap:14px;padding:14px;min-height:100vh;align-items:stretch}
  @media(max-width:900px){.wrap{flex-direction:column}}
  .stage{flex:1;background:var(--panel);border:1px solid var(--line);border-radius:12px;
         position:relative;overflow:hidden;min-height:460px}
  canvas{display:block;width:100%;height:100%;touch-action:none;cursor:crosshair}
  .panel{width:312px;flex:none;display:flex;flex-direction:column;gap:12px}
  @media(max-width:900px){.panel{width:auto}}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:13px}
  h1{font-size:15px;margin:0 0 2px} .sub{font-size:11.5px;color:var(--muted);margin:0 0 10px}
  .eyebrow{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin:0 0 9px}
  .btns{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:4px}
  button{font:inherit;font-size:12.5px;padding:7px 11px;border-radius:8px;cursor:pointer;
         border:1px solid var(--line);background:transparent;color:var(--ink)}
  button.primary{background:var(--accent);border-color:var(--accent);color:#04211d;font-weight:600}
  button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
  label{display:flex;flex-direction:column;gap:3px;font-size:11.5px;color:var(--muted)}
  input,select{font:inherit;font-size:13px;padding:6px 8px;border:1px solid var(--line);
    border-radius:7px;background:var(--field);color:var(--ink);width:100%;
    font-variant-numeric:tabular-nums}
  input:focus,select:focus{outline:2px solid var(--accent);outline-offset:0}
  textarea{width:100%;margin-top:8px;border:1px solid var(--line);border-radius:7px;
    background:var(--field);color:var(--ink);font-family:ui-monospace,Menlo,monospace;
    font-size:11px;padding:8px;resize:vertical}
  .row{display:flex;justify-content:space-between;align-items:baseline;gap:8px;
       font-variant-numeric:tabular-nums;padding:3px 0}
  .row .k{color:var(--muted)} .row .v{font-weight:600}
  .legend{display:flex;gap:11px;flex-wrap:wrap;font-size:11.5px;margin-top:8px}
  .legend span{display:inline-flex;align-items:center;gap:5px}
  .dot{width:10px;height:10px;border-radius:50%;display:inline-block}
  table{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums;margin-top:4px}
  th,td{text-align:right;padding:3px 2px;border-bottom:1px solid var(--line)}
  th:first-child,td:first-child{text-align:left}
  .hint{font-size:11px;color:var(--muted);line-height:1.5;margin:8px 0 0}
  .range{display:flex;align-items:center;gap:8px}
  .range input[type=range]{padding:0}
</style>

<div class="wrap">
  <div class="stage"><canvas id="cv"></canvas></div>
  <div class="panel">
    <div class="card">
      <h1>CA1 창·전극 배치기</h1>
      <p class="sub">slice400 (400µm 슬랩) · 종축 × 층관통 (2D 투영, 실제는 3D)</p>
      <label>이름 <input id="name" type="text" value="cand_층관통"></label>
      <div class="btns" style="margin-top:8px">
        <button class="primary" onclick="preset('가로')">가로 기본</button>
        <button onclick="preset('세로')">세로 기본</button>
      </div>
      <div class="legend" id="legend"></div>
    </div>

    <div class="card">
      <p class="eyebrow">창 크기 (µm)</p>
      <div class="grid">
        <label>가로 종축 <input id="Lu" type="number" step="10" value="800"></label>
        <label>세로 층관통 <input id="Lr" type="number" step="10" value="500"></label>
        <label>깊이 두께 <input id="Lw" type="number" step="10" value="400"></label>
        <label>층관통 중심 r <input id="cr" type="number" step="10" value="175"></label>
      </div>
      <label style="margin-top:8px">창 각도 (°) 회전
        <span class="range"><input id="wang" type="range" min="-90" max="90" step="1" value="0">
        <input id="wangn" type="number" min="-90" max="90" step="1" value="0" style="width:64px"></span>
      </label>
      <div class="row" style="margin-top:8px"><span class="k">내부 세포수 (3D)</span><span class="v" id="wn">—</span></div>
      <table id="wtab"><tbody></tbody></table>
    </div>

    <div class="card">
      <p class="eyebrow">전극 배열</p>
      <div class="grid">
        <label>행 (rows) <input id="rows" type="number" min="1" max="12" step="1" value="1"></label>
        <label>열 (cols) <input id="cols" type="number" min="1" max="12" step="1" value="3"></label>
        <label>간격 (µm) <input id="sp" type="number" step="10" value="200"></label>
        <label>직경 (µm) <input id="dia" type="number" step="1" value="10"></label>
      </div>
      <label style="margin-top:8px">배열 방향 (°) 강체 회전
        <span class="range"><input id="ang" type="range" min="0" max="180" step="1" value="0">
        <input id="angn" type="number" min="0" max="180" step="1" value="0" style="width:64px"></span>
      </label>
      <div class="grid" style="margin-top:8px;align-items:end">
        <label>자극 전극 <select id="stim"></select></label>
        <button onclick="alignCenter()">창 중심에 정렬</button>
      </div>
      <p class="hint">배열=강체: <b>◆중심 드래그=이동</b> · <b>전극 드래그=방향</b> (슬라이더). 자극=✚(빨강).</p>
      <table id="etab"><tbody></tbody></table>
    </div>

    <div class="card">
      <p class="eyebrow">설정 내보내기 → config</p>
      <div class="btns">
        <button class="primary" onclick="exportCfg()">설정 생성</button>
        <button id="copybtn" onclick="copyCfg()">복사</button>
        <button onclick="importCfg()">불러오기</button>
      </div>
      <textarea id="cfgout" rows="8" placeholder="[설정 생성] 또는 저장한 JSON을 붙여넣고 [불러오기]"></textarea>
      <p class="hint">이 JSON을 <b>config/</b>에 저장하면 파이프라인(5단계 배치·전극)이 국소 프레임으로 <b>3D 재구성</b>합니다.</p>
    </div>
  </div>
</div>

<script>
const D = /*__DATA__*/;
const LC=D.layerColors, LN=D.layers, DEG=Math.PI/180;
const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
let box={Lu:800,Lr:500,Lw:400,cu:0,cr:175,cw:0,ang:0};
let arr={rows:1,cols:3,sp:200,dia:10,ang:0,cu:0,cr:238,stim:1};
let T={s:1,ox:0,oy:0};

document.getElementById('legend').innerHTML=LN.map((n,i)=>
  `<span><i class="dot" style="background:${LC[i]}"></i>${n}</span>`).join('');
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim()||'#888';}
const X=u=>T.s*u+T.ox, Y=r=>-T.s*r+T.oy, uOf=x=>(x-T.ox)/T.s, rOf=y=>-(y-T.oy)/T.s;

function resize(){
  const st=cv.parentElement.getBoundingClientRect();
  if(st.width<80||st.height<80) return;
  const dpr=Math.min(devicePixelRatio||1,2);
  cv.width=st.width*dpr; cv.height=st.height*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
  const b=D.bounds, pad=34, uw=b.umax-b.umin, rh=b.rmax-b.rmin;
  T.s=Math.max(1e-3,Math.min((st.width-2*pad)/uw,(st.height-2*pad)/rh));
  T.ox=pad-T.s*b.umin+((st.width-2*pad)-T.s*uw)/2;
  T.oy=pad+T.s*b.rmax+((st.height-2*pad)-T.s*rh)/2;
  draw();
}

// 창 국소변환/코너 (회전 반영)
function toLocal(u,r){const c=Math.cos(box.ang*DEG),s=Math.sin(box.ang*DEG);
  const du=u-box.cu,dr=r-box.cr;return{a:c*du+s*dr,b:-s*du+c*dr};}
function winCorners(){const c=Math.cos(box.ang*DEG),s=Math.sin(box.ang*DEG),hw=box.Lu/2,hh=box.Lr/2;
  return [[-hw,-hh],[hw,-hh],[hw,hh],[-hw,hh]].map(([a,b])=>({u:box.cu+c*a-s*b,r:box.cr+s*a+c*b}));}
function elecPos(){
  const out=[], ca=Math.cos(arr.ang*DEG), sa=Math.sin(arr.ang*DEG);
  for(let i=0;i<arr.rows;i++)for(let j=0;j<arr.cols;j++){
    const a=(j-(arr.cols-1)/2)*arr.sp, bb=(i-(arr.rows-1)/2)*arr.sp;
    out.push({u:arr.cu+ca*a-sa*bb, r:arr.cr+sa*a+ca*bb, idx:i*arr.cols+j+1});}
  return out;
}
function layerAt(u,r){let best=1e18,bl=-1;
  for(const c of D.cells){const du=c[0]-u,dr=c[1]-r,dd=du*du+dr*dr;if(dd<best){best=dd;bl=c[3];}}
  return best<40000?LN[bl]:'—';}

function draw(){
  const st=cv.parentElement.getBoundingClientRect();
  ctx.clearRect(0,0,st.width,st.height);
  for(const c of D.cells){ctx.fillStyle=LC[c[3]];ctx.globalAlpha=c[3]===1?0.5:0.85;
    ctx.fillRect(X(c[0])-1.1,Y(c[1])-1.1,2.2,2.2);}
  ctx.globalAlpha=1;
  const bx=st.width-150,by=st.height-22;
  ctx.strokeStyle=css('--muted');ctx.fillStyle=css('--muted');ctx.lineWidth=2;
  ctx.beginPath();ctx.moveTo(bx,by);ctx.lineTo(bx+T.s*200,by);ctx.stroke();
  ctx.font='11px system-ui';ctx.fillText('200 µm',bx,by-5);
  // window (회전 폴리곤)
  const cn=winCorners();
  ctx.beginPath();ctx.moveTo(X(cn[0].u),Y(cn[0].r));
  for(let i=1;i<4;i++)ctx.lineTo(X(cn[i].u),Y(cn[i].r));ctx.closePath();
  ctx.fillStyle=css('--accent');ctx.globalAlpha=0.10;ctx.fill();
  ctx.globalAlpha=1;ctx.strokeStyle=css('--accent');ctx.lineWidth=2;ctx.stroke();
  ctx.fillStyle=css('--accent');ctx.beginPath();ctx.arc(X(cn[2].u),Y(cn[2].r),6,0,7);ctx.fill();
  // electrodes
  const eps=elecPos(), rad=Math.max(4,arr.dia/2*T.s);
  const hx=X(arr.cu),hy=Y(arr.cr);
  ctx.save();ctx.translate(hx,hy);ctx.rotate(Math.PI/4);ctx.fillStyle=css('--stim');
  ctx.globalAlpha=0.9;ctx.fillRect(-6,-6,12,12);ctx.restore();ctx.globalAlpha=1;
  for(const e of eps){const ex=X(e.u),ey=Y(e.r),isS=e.idx===arr.stim;
    ctx.strokeStyle=isS?css('--stim'):css('--accent');ctx.globalAlpha=0.22;ctx.lineWidth=1.4;
    ctx.beginPath();ctx.arc(ex,ey,Math.max(0,T.s*100),0,7);ctx.stroke();ctx.globalAlpha=1;
    if(isS){ctx.strokeStyle=css('--stim');ctx.lineWidth=4;ctx.beginPath();
      ctx.moveTo(ex-9,ey);ctx.lineTo(ex+9,ey);ctx.moveTo(ex,ey-9);ctx.lineTo(ex,ey+9);ctx.stroke();}
    else{ctx.fillStyle=css('--rec');ctx.beginPath();ctx.arc(ex,ey,rad,0,7);ctx.fill();
      ctx.strokeStyle='#fff';ctx.lineWidth=1.4;ctx.stroke();}
    ctx.fillStyle=isS?css('--stim'):css('--ink');ctx.font='bold 11px system-ui';ctx.textAlign='center';
    ctx.fillText((isS?'자극 ':'')+'E'+e.idx,ex,ey-rad-6);ctx.textAlign='left';}
  readout(eps);
}

function readout(eps){
  const by=[0,0,0,0];let n=0;const scale=D.nTotalSlice/D.cells.length;
  for(const c of D.cells){const L=toLocal(c[0],c[1]);
    if(Math.abs(L.a)<=box.Lu/2&&Math.abs(L.b)<=box.Lr/2&&Math.abs(c[2]-box.cw)<=box.Lw/2){by[c[3]]++;n++;}}
  const wn=document.getElementById('wn');
  wn.textContent=`≈ ${Math.round(n*scale).toLocaleString()}`;
  wn.title=`표본 ${n}/${D.cells.length}개 기준 추정 (정확값은 5단계 전세포 배치)`;
  document.querySelector('#wtab tbody').innerHTML=LN.map((nm,i)=>
    `<tr><td><i class="dot" style="background:${LC[i]}"></i> ${nm}</td><td>≈ ${Math.round(by[i]*scale).toLocaleString()}</td></tr>`).join('');
  document.querySelector('#etab tbody').innerHTML=eps.map(e=>
    `<tr><td>${e.idx===arr.stim?'⚡ ':''}E${e.idx}</td><td>${Math.round(e.u)}</td><td>${Math.round(e.r)}</td><td>${layerAt(e.u,e.r)}</td></tr>`).join('');
}
document.querySelector('#etab').insertAdjacentHTML('afterbegin',
  '<thead><tr><th>전극</th><th>종축</th><th>층관통</th><th>층</th></tr></thead>');

// ---- inputs ----
function bindNum(id,obj,key){const el=document.getElementById(id);
  el.addEventListener('input',()=>{const v=parseFloat(el.value);if(!isNaN(v)){obj[key]=v;draw();}});}
bindNum('Lu',box,'Lu');bindNum('Lr',box,'Lr');bindNum('Lw',box,'Lw');bindNum('cr',box,'cr');
bindNum('rows',arr,'rows');bindNum('cols',arr,'cols');bindNum('sp',arr,'sp');bindNum('dia',arr,'dia');
function pair(rid,nid,setter){const R=document.getElementById(rid),N=document.getElementById(nid);
  const f=v=>{setter(v);R.value=v;N.value=Math.round(v);draw();};
  R.addEventListener('input',()=>f(parseFloat(R.value)));
  N.addEventListener('input',()=>f(parseFloat(N.value)||0));return f;}
const setAng=pair('ang','angn',v=>arr.ang=v);
const setWang=pair('wang','wangn',v=>box.ang=v);

function preset(k){const p=D.presets[k],a=p.arr;
  box.Lu=p.Lu;box.Lr=p.Lr;box.Lw=p.Lw;box.cr=p.cr;box.cu=p.cu||0;box.ang=0;
  arr.rows=a.rows;arr.cols=a.cols;arr.sp=a.sp;arr.ang=a.ang;arr.cu=box.cu;arr.cr=box.cr;arr.stim=1;
  for(const [id,val] of [['Lu',p.Lu],['Lr',p.Lr],['Lw',p.Lw],['cr',p.cr],
      ['rows',a.rows],['cols',a.cols],['sp',a.sp],['ang',a.ang],['angn',a.ang]])document.getElementById(id).value=val;
  document.getElementById('wang').value=0;document.getElementById('wangn').value=0;
  populateStim();draw();}
window.preset=preset;

// ---- export ----
function toPhys(u,r,w){const f=D.frame;return [0,1,2].map(k=>
  +(f.seed[k]+u*f.long_dir[k]+r*f.radial_dir[k]+w*f.thick_dir[k]).toFixed(1));}
function exportCfg(){
  const eps=elecPos();
  const cfg={
    name:document.getElementById('name').value||'unnamed',
    note:"micro-slice window+electrodes (local frame → physical xyz). Stage5가 읽음.",
    frame_um:{seed:D.frame.seed.map(x=>+x.toFixed(2)),long_dir:D.frame.long_dir.map(x=>+x.toFixed(5)),
      radial_dir:D.frame.radial_dir.map(x=>+x.toFixed(5)),thick_dir:D.frame.thick_dir.map(x=>+x.toFixed(5))},
    window_um:{long:box.Lu,radial:box.Lr,thick:box.Lw,angle_deg:+box.ang.toFixed(1),
      center_local:{u:box.cu,r:box.cr,w:box.cw},center_xyz:toPhys(box.cu,box.cr,box.cw)},
    electrodes:{rows:arr.rows,cols:arr.cols,spacing_um:arr.sp,diameter_um:arr.dia,angle_deg:+arr.ang.toFixed(1),
      stim_id:"E"+arr.stim,mea_face_w_um:box.Lw/2,center_local:{u:+arr.cu.toFixed(1),r:+arr.cr.toFixed(1)},
      list:eps.map(e=>({id:"E"+e.idx,role:e.idx===arr.stim?"stim":"rec",u:Math.round(e.u),r:Math.round(e.r),
        layer:layerAt(e.u,e.r),xyz_um:toPhys(e.u,e.r,box.Lw/2)}))}
  };
  document.getElementById('cfgout').value=JSON.stringify(cfg,null,2);
}
function copyCfg(){const t=document.getElementById('cfgout');if(!t.value)exportCfg();t.select();
  const done=()=>{const b=document.getElementById('copybtn');b.textContent='복사됨 ✓';
    setTimeout(()=>b.textContent='복사',1200);};
  try{navigator.clipboard.writeText(t.value).then(done,()=>{document.execCommand('copy');done();});}
  catch(e){document.execCommand('copy');done();}}
window.exportCfg=exportCfg;window.copyCfg=copyCfg;

// ---- 설정 불러오기(복원) ----
function applyCfg(cfg){
  if(!cfg||!cfg.window_um) return;
  const w=cfg.window_um; box.Lu=w.long;box.Lr=w.radial;box.Lw=w.thick;box.ang=w.angle_deg||0;
  box.cu=w.center_local.u;box.cr=w.center_local.r;box.cw=w.center_local.w||0;
  const e=cfg.electrodes; arr.rows=e.rows;arr.cols=e.cols;arr.sp=e.spacing_um;arr.dia=e.diameter_um;
  arr.ang=e.angle_deg||0;arr.cu=e.center_local.u;arr.cr=e.center_local.r;
  arr.stim=parseInt(String(e.stim_id||'E1').replace('E',''))||1;
  document.getElementById('name').value=cfg.name||'';
  for(const [id,val] of [['Lu',box.Lu],['Lr',box.Lr],['Lw',box.Lw],['cr',box.cr],
    ['rows',arr.rows],['cols',arr.cols],['sp',arr.sp],['dia',arr.dia],
    ['ang',arr.ang],['angn',arr.ang],['wang',box.ang],['wangn',box.ang]]){
    const el=document.getElementById(id); if(el)el.value=Math.round(val);}
  if(typeof populateStim==='function')populateStim();
  draw();
}
function importCfg(){try{applyCfg(JSON.parse(document.getElementById('cfgout').value));}
  catch(e){alert('JSON 파싱 실패: '+e.message);}}
window.applyCfg=applyCfg;window.importCfg=importCfg;

// ---- 자극전극 선택 · 중심정렬 ----
function populateStim(){const N=Math.max(1,arr.rows*arr.cols),sel=document.getElementById('stim');
  sel.innerHTML=Array.from({length:N},(_,i)=>`<option value="${i+1}">E${i+1}</option>`).join('');
  if(arr.stim>N)arr.stim=1; sel.value=arr.stim;}
document.getElementById('stim').addEventListener('change',e=>{arr.stim=parseInt(e.target.value)||1;draw();});
document.getElementById('rows').addEventListener('input',()=>{populateStim();draw();});
document.getElementById('cols').addEventListener('input',()=>{populateStim();draw();});
populateStim();
function alignCenter(){arr.cu=box.cu;arr.cr=box.cr;draw();}
window.alignCenter=alignCenter;

// ---- interaction ----
let drag=null;
function pos(ev){const b=cv.getBoundingClientRect();return{x:ev.clientX-b.left,y:ev.clientY-b.top};}
cv.addEventListener('pointerdown',ev=>{const p=pos(ev);cv.setPointerCapture(ev.pointerId);
  if(Math.hypot(p.x-X(arr.cu),p.y-Y(arr.cr))<13){drag={t:'amove',ox:uOf(p.x)-arr.cu,oy:rOf(p.y)-arr.cr};return;}
  for(const e of elecPos()){if(Math.hypot(p.x-X(e.u),p.y-Y(e.r))<Math.max(8,arr.dia/2*T.s+6)){
    drag={t:'arot',a0:Math.atan2(rOf(p.y)-arr.cr,uOf(p.x)-arr.cu),ang0:arr.ang};return;}}
  const cn=winCorners();
  if(Math.hypot(p.x-X(cn[2].u),p.y-Y(cn[2].r))<12){drag={t:'rz'};return;}
  const L=toLocal(uOf(p.x),rOf(p.y));
  if(Math.abs(L.a)<=box.Lu/2&&Math.abs(L.b)<=box.Lr/2){drag={t:'wmove',ox:uOf(p.x)-box.cu,oy:rOf(p.y)-box.cr};return;}
});
cv.addEventListener('pointermove',ev=>{if(!drag)return;const p=pos(ev);const u=uOf(p.x),r=rOf(p.y);
  if(drag.t==='amove'){arr.cu=u-drag.ox;arr.cr=r-drag.oy;}
  else if(drag.t==='arot'){const a=Math.atan2(r-arr.cr,u-arr.cu);
    setAng(((drag.ang0+(a-drag.a0)/DEG)%360+360)%360);return;}
  else if(drag.t==='rz'){const L=toLocal(u,r);box.Lu=Math.max(40,2*Math.abs(L.a));box.Lr=Math.max(40,2*Math.abs(L.b));
    document.getElementById('Lu').value=Math.round(box.Lu);document.getElementById('Lr').value=Math.round(box.Lr);}
  else if(drag.t==='wmove'){box.cu=u-drag.ox;box.cr=r-drag.oy;document.getElementById('cr').value=Math.round(box.cr);}
  draw();});
cv.addEventListener('pointerup',()=>drag=null);
cv.addEventListener('pointercancel',()=>drag=null);

new ResizeObserver(resize).observe(cv.parentElement);
window.addEventListener('load',resize);resize();
if(D.saved)applyCfg(D.saved);   // 확정 config 로 초기화(새로고침해도 유지)
</script>
'''

if __name__ == "__main__":
    main()
