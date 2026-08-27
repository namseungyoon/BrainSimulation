#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""slice_cells.npz(배치 결과) → 전 슬라이스 위치 3D 미리보기 HTML (시뮬 불필요).

실제 17,647세포 소마 위치를 층별 색으로 3D로 그린다. 슬라이스 형태·층 배치·전극
위치가 맞는지 눈으로 확인하는 용도. 전류 애니메이션은 별개(전규모 런 후 make_fepsp3d).
"""
import os, json
import numpy as np
from scipy.spatial import cKDTree

TEMPLATE = r'''<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>slice positions preview</title>
<style>
*{box-sizing:border-box}html,body{margin:0;height:100%}
body{background:#080b12;color:#e7eef7;font-family:system-ui,sans-serif;font-size:14px;display:flex;flex-direction:column;height:100vh;overflow:hidden}
header{padding:10px 18px;border-bottom:1px solid #243044;background:#0f141f}
header h1{margin:0 0 3px;font-size:15px;font-weight:700}header .sub{color:#8697b0;font-size:11.5px}
.wrap{flex:1;display:flex;min-height:0}.stage{flex:1;position:relative;min-width:0}
canvas#gl{width:100%;height:100%;display:block;cursor:grab}canvas#gl:active{cursor:grabbing}
.side{width:250px;border-left:1px solid #243044;background:#0f141f;padding:14px;display:flex;flex-direction:column;gap:12px}
button{background:#141b28;border:1px solid #243044;color:#e7eef7;border-radius:8px;padding:8px 11px;font-size:12.5px;cursor:pointer;width:100%}
button.on{border-color:#ffc857;color:#ffc857}
.lg{font-size:12px;color:#c7d3e6;line-height:1.9}.sw{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-1px;margin-right:6px}
.k{color:#8697b0;font-size:11px}.b{font-weight:600}
</style>
<header><h1>전 슬라이스 위치 미리보기 — Romani CA1 (<span id="nc"></span> 세포)</h1>
<div class="sub">실제 배치 좌표(05_placement) · 층별 색 · 흰 점 = MEA 전극. 드래그=회전. <b>전류 애니메이션 아님(위치 확인용)</b></div></header>
<div class="wrap"><div class="stage"><canvas id="gl"></canvas></div>
<div class="side">
  <button id="rot" class="on">자동회전 끄기</button>
  <div><div class="k">슬라이스 크기 (um)</div><div id="sz" class="b"></div></div>
  <div class="lg" id="leg"></div>
  <div class="k" style="line-height:1.5">회전해서 얇은 판(두께축) 확인 · 층이 면 안에 띠로 배열</div>
</div></div>
<script>
const D=__DATA__;const gl=document.getElementById('gl'),gx=gl.getContext('2d');
const N=D.n,POS=D.pos,LAY=D.lay,EL=D.elec;
const LC=['#DD8452','#c084fc','#2dd4bf','#ffd43b'],LN=['SO','SP','SR','SLM'];
let yaw=0.6,pitch=0.18,drag=false,lx=0,ly=0,autorot=true,zoom=1,DPR=Math.min(2,devicePixelRatio||1);
document.getElementById('nc').textContent=N.toLocaleString();
document.getElementById('sz').textContent=D.size.u+' (장축) x '+D.size.r+' (층관통) x '+D.size.w+' (두께)';
document.getElementById('leg').innerHTML=LN.map((L,i)=>'<div><span class="sw" style="background:'+LC[i]+'"></span>'+L+' <span class="k">'+D.counts[L].toLocaleString()+'</span></div>').join('')+'<div><span class="sw" style="background:#fff"></span>MEA 전극 24</div>';
const span=(()=>{let m=0;for(const p of POS)m=Math.max(m,Math.abs(p[0]),Math.abs(p[1]),Math.abs(p[2]));return m*2.1;})();
function resize(){DPR=Math.min(2,devicePixelRatio||1);const r=gl.getBoundingClientRect();gl.width=r.width*DPR;gl.height=r.height*DPR;}
function proj(p,W,H,sc){const cY=Math.cos(yaw),sY=Math.sin(yaw),cP=Math.cos(pitch),sP=Math.sin(pitch);let X=p[0],Y=p[1],Z=p[2];let x1=X*cY-Z*sY,z1=X*sY+Z*cY;let y1=Y*cP-z1*sP,z2=Y*sP+z1*cP;const f=900/(900+z2);return [W/2+x1*f*sc,H/2+y1*f*sc,z2,f];}
const LAYFILL=['rgba(221,132,82,.08)','rgba(192,132,252,.09)','rgba(45,212,191,.08)','rgba(255,212,59,.07)'];
function drawScene(W,H,sc){const bu=D.box.u/2,br=D.box.r/2,bw=D.box.w/2;
  D.layers.forEach((L,i)=>{const c=[[-bu,L.r0,bw],[bu,L.r0,bw],[bu,L.r1,bw],[-bu,L.r1,bw]].map(p=>proj(p,W,H,sc));
    gx.fillStyle=LAYFILL[i];gx.beginPath();c.forEach((q,k)=>k?gx.lineTo(q[0],q[1]):gx.moveTo(q[0],q[1]));gx.closePath();gx.fill();
    const lb=proj([-bu,(L.r0+L.r1)/2,bw],W,H,sc);gx.fillStyle=LC[i];gx.font='bold 12px system-ui';gx.textAlign='right';gx.fillText(L.name,lb[0]-5,lb[1]+4);gx.textAlign='left';});
  const cor=[];for(const su of[-bu,bu])for(const srr of[-br,br])for(const sw of[-bw,bw])cor.push([su,srr,sw]);
  const e=[[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]];
  gx.strokeStyle='rgba(150,175,210,.28)';gx.lineWidth=1;for(const [a,b] of e){const pa=proj(cor[a],W,H,sc),pb=proj(cor[b],W,H,sc);gx.beginPath();gx.moveTo(pa[0],pa[1]);gx.lineTo(pb[0],pb[1]);gx.stroke();}}
function draw(){const W=gl.width/DPR,H=gl.height/DPR;gx.setTransform(DPR,0,0,DPR,0,0);gx.clearRect(0,0,W,H);
  const sc=Math.min(W,H)/(span*1.1)*zoom;drawScene(W,H,sc);
  const ord=[];for(let i=0;i<N;i++){const q=proj(POS[i],W,H,sc);ord.push([q[2],i,q]);}ord.sort((a,b)=>a[0]-b[0]);
  for(const [,i,q] of ord){gx.fillStyle=LC[LAY[i]];gx.globalAlpha=0.55;gx.beginPath();gx.arc(q[0],q[1],1.5*(0.8+0.4*q[3]),0,6.28);gx.fill();}
  gx.globalAlpha=1;
  for(const e of EL){const q=proj(e,W,H,sc);gx.fillStyle='#fff';gx.beginPath();gx.arc(q[0],q[1],3.2*q[3],0,6.28);gx.fill();gx.strokeStyle='#0f141f';gx.lineWidth=1;gx.stroke();}}
function loop(){if(autorot&&!drag)yaw+=0.0025;draw();requestAnimationFrame(loop);}
document.getElementById('rot').onclick=function(){autorot=!autorot;this.classList.toggle('on',autorot);this.textContent=autorot?'자동회전 끄기':'자동회전 켜기';};
gl.addEventListener('wheel',e=>{e.preventDefault();zoom*=Math.exp(-e.deltaY*0.0012);zoom=Math.max(0.3,Math.min(8,zoom));},{passive:false});
gl.addEventListener('mousedown',e=>{drag=true;lx=e.clientX;ly=e.clientY;});
window.addEventListener('mouseup',()=>drag=false);
window.addEventListener('mousemove',e=>{if(!drag)return;yaw+=(e.clientX-lx)*0.008;pitch=Math.max(-0.6,Math.min(1.2,pitch+(e.clientY-ly)*0.006));lx=e.clientX;ly=e.clientY;});
window.addEventListener('resize',resize);resize();loop();
</script>'''

CELLS = os.path.join(os.path.dirname(__file__), "..", "05_placement", "slice_cells.npz")
OUT = os.path.join(os.path.dirname(__file__), "figures", "preview_slice.html")
PITCH, R_ON, NCOL, NROW = 200.0, 100.0, 8, 3

c = np.load(CELLS, allow_pickle=True)
xyz = c["xyz"].astype(float); layer = c["layer"].astype(str); mtype = c["mtype"].astype(str)
Ntot = len(xyz)
c0 = xyz.mean(0); Vall = np.linalg.svd(xyz - c0, full_matrices=False)[2]
sp = []
for i in range(3):
    pr = (xyz - c0) @ Vall[i]
    cen = [pr[layer == L].mean() for L in ("SO", "SP", "SR", "SLM") if (layer == L).any()]
    sp.append(float(np.ptp(cen)))
it = int(np.argmin(sp)); iff = [i for i in range(3) if i != it]
face = Vall[iff]; thick = Vall[it]
lc = {L: ((xyz[layer == L] - c0) @ face.T).mean(0) for L in ("SO", "SP", "SR", "SLM") if (layer == L).any()}
ul = lc["SLM"] - lc["SP"]; ul /= np.linalg.norm(ul); ulong = np.array([-ul[1], ul[0]])


def frame(uv2, thk1):
    d2 = uv2 - lc["SP"]
    return np.array([d2 @ ulong, d2 @ ul, thk1])


uv = (xyz - c0) @ face.T
u = (uv - lc["SP"]) @ ulong
r = (uv - lc["SP"]) @ ul
w = (xyz - c0) @ thick
LAYS = ("SO", "SP", "SR", "SLM")
lay_idx = np.array([LAYS.index(L) if L in LAYS else 1 for L in layer])

# 전극 격자 맞추기 (mea_experiment.py 방식) — PC층(SP)에 최대한 얹기
pc = uv[layer == "SP"]
gx = (np.arange(NCOL) - (NCOL - 1) / 2) * PITCH
gy = (np.arange(NROW) - (NROW - 1) / 2) * PITCH
Gx, Gy = np.meshgrid(gx, gy); G0 = np.column_stack([Gx.ravel(), Gy.ravel()])
tree = cKDTree(pc); fc = pc.mean(0); best = (-1, None)
for th in np.deg2rad(np.arange(0, 180, 10)):
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]); Grot = G0 @ Rm.T
    for dxx in np.linspace(-400, 400, 9):
        for dyy in np.linspace(-200, 200, 9):
            E2 = Grot + fc + [dxx, dyy]; on = int((tree.query(E2)[0] < R_ON).sum())
            if on > best[0]:
                best = (on, E2.copy())
E2d = best[1]
el_u = (E2d - lc["SP"]) @ ulong; el_r = (E2d - lc["SP"]) @ ul

# 소마점 재중심화
P = np.column_stack([u, r, w])
lo = np.percentile(P, 1, axis=0); hi = np.percentile(P, 99, axis=0); ctr = (lo + hi) / 2
P -= ctr
wglass = float(np.percentile(P[:, 2], 1))
elec = np.column_stack([el_u - ctr[0], el_r - ctr[1], np.full(len(E2d), wglass)])
cen = np.array([float((lc[L] - lc["SP"]) @ ul) for L in LAYS]) - ctr[1]
ed = np.array([cen[0] - (cen[1] - cen[0]) / 2, (cen[0] + cen[1]) / 2,
               (cen[1] + cen[2]) / 2, (cen[2] + cen[3]) / 2, cen[3] + (cen[3] - cen[2]) / 2])
box = dict(u=float(hi[0] - lo[0]), r=float(hi[1] - lo[1]), w=float(hi[2] - lo[2]))

# 표시 세포수 제한(성능): 전부 그리되 너무 많으면 균일 샘플
D = dict(
    n=Ntot, pos=[[round(float(x), 1) for x in p] for p in P],
    lay=[int(x) for x in lay_idx],
    elec=[[round(float(x), 1) for x in e] for e in elec],
    layers=[dict(name=L, r0=float(a), r1=float(b)) for L, a, b in zip(LAYS, ed[:-1], ed[1:])],
    box=box, counts={L: int((layer == L).sum()) for L in LAYS},
    size=dict(u=round(float(box["u"])), r=round(float(box["r"])), w=round(float(box["w"]))),
)
html = TEMPLATE.replace("__DATA__", json.dumps(D, separators=(",", ":")))
with open(OUT, "w", encoding="utf-8") as f:
    f.write(html)
print("saved:", OUT, "(%.1f MB, %d cells)" % (os.path.getsize(OUT) / 1e6, Ntot))
print("slice size um:", D["size"], "| layer counts:", D["counts"])
