# -*- coding: utf-8 -*-
"""
03_network/1_connectome/make_wiring_html.py  —  3-1(b) 완전체 wiring 인터랙티브 HTML

대상 추체 + 연결된 전시냅스 세포 전체(기본 83개)의 형태를 canvas로 그리고
마우스 휠 확대 / 드래그 이동이 되는 자체완결 HTML을 만든다.
  - 종류=색 · 수상돌기=진한톤 · 축삭=연한톤 · 시냅스=★
결과: scratch/3-1b_wiring.html (자체완결, 브라우저에서 열기)

실행: python 03_network/1_connectome/make_wiring_html.py
"""
import os
import json
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
import matplotlib.colors as mc

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DERIVED = os.path.join(ROOT, "data", "derived")
LIB = os.path.join(ROOT, "data", "morphology_library", "morphology_library")
CFG = os.path.join(ROOT, "config", "window_layout.json")
OUT = os.path.join(ROOT, "scratch", "3-1b_wiring.html")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
DEND_STRIDE, AXON_STRIDE = 2, 18
CMAP = {"SP_PC": "#C44E52", "SP_Ivy": "#8172B3", "SP_PVBC": "#4C72B0", "SP_CCKBC": "#55A868",
        "SO_OLM": "#CCB974", "SP_BS": "#DD8452", "SO_Tri": "#937860", "SR_SCA": "#DA8BC3",
        "SO_BS": "#8C8C8C", "SLM_PPA": "#64B5CD", "SP_AA": "#E377C2", "SO_BP": "#7F7F7F"}


def hx(c, f, lighten):
    r, g, b = mc.to_rgb(c)
    if lighten:
        r, g, b = r+(1-r)*f, g+(1-g)*f, b+(1-b)*f
    else:
        r, g, b = r*f, g*f, b*f
    return "#%02x%02x%02x" % (int(r*255), int(g*255), int(b*255))


def load(path):
    r = np.loadtxt(path, comments="#")
    return r[:, 1].astype(int), r[:, 2:5].astype(np.float64), r[:, 0].astype(int), r[:, 6].astype(int)


def Lc(pts, q, xyz0, seed, M):
    return (xyz0 + Rot.from_quat(q[[1, 2, 3, 0]]).apply(pts) - seed) @ M


def polylines(typ, pts, idx, par, q, xyz0, seed, M, which, stride):
    """부모연결 세그먼트를 flat [x1,y1,x2,y2,...] (µm, r-band 유형만, stride 적용)."""
    id2 = {i: k for k, i in enumerate(idx)}
    loc = Lc(pts, q, xyz0, seed, M)
    seg = []
    keep = [k for k in range(len(idx)) if par[k] in id2 and typ[k] in which]
    for k in keep[::stride]:
        a = id2[par[k]]
        seg += [round(loc[a, 0], 1), round(loc[a, 1], 1), round(loc[k, 0], 1), round(loc[k, 1], 1)]
    return seg, loc


def main():
    wc = np.load(os.path.join(DERIVED, "window_cells.npz"), allow_pickle=True)
    XYZ = wc["xyz"]; Q = wc["orientation_wxyz"]; mt = wc["mtype"].astype(str); morph = wc["morphology"].astype(str)
    syn = np.load(os.path.join(DERIVED, "synapses_internal.npz"))
    pre = syn["pre_gid"]; post = syn["post_gid"]; ns = syn["n_syn"]
    cfg = json.load(open(CFG, encoding="utf-8")); fr = cfg["frame_um"]
    seed = np.array(fr["seed"]); M = np.column_stack([fr["long_dir"], fr["radial_dir"], fr["thick_dir"]])
    e3 = np.array(next(e for e in cfg["electrodes"]["list"] if e["role"] == "stim")["xyz_um"])
    rng = np.random.default_rng(1)
    pc = np.where(mt == "SP_PC")[0]
    tgt = pc[np.argmin(np.linalg.norm(XYZ[pc] - e3, axis=1))]
    inmask = post == tgt; ins = list(pre[inmask]); insyn = dict(zip(pre[inmask], ns[inmask]))

    cells = []
    # 대상 추체
    typ, pts, idx, par = load(os.path.join(LIB, morph[tgt] + ".swc"))
    dseg, dloc = polylines(typ, pts, idx, par, Q[tgt], XYZ[tgt], seed, M, (1, 3, 4), DEND_STRIDE)
    aseg, _ = polylines(typ, pts, idx, par, Q[tgt], XYZ[tgt], seed, M, (2,), AXON_STRIDE)
    cells.append({"mt": "TARGET", "dc": hx(CMAP["SP_PC"], 0.55, False), "ac": hx(CMAP["SP_PC"], 0.7, True),
                  "d": dseg, "a": aseg})
    dmask = (typ == 1) | (typ == 3) | (typ == 4)
    tree = cKDTree(Lc(pts[dmask], Q[tgt], XYZ[tgt], seed, M)); tdl = Lc(pts[dmask], Q[tgt], XYZ[tgt], seed, M)

    syns = []
    for p in ins:
        c = CMAP.get(mt[p], "#888888")
        t2, p2, i2, pr2 = load(os.path.join(LIB, morph[p] + ".swc"))
        ds, _ = polylines(t2, p2, i2, pr2, Q[p], XYZ[p], seed, M, (1, 3, 4), DEND_STRIDE)
        as_, _ = polylines(t2, p2, i2, pr2, Q[p], XYZ[p], seed, M, (2,), AXON_STRIDE)
        cells.append({"mt": mt[p], "dc": hx(c, 0.55, False), "ac": hx(c, 0.6, True), "d": ds, "a": as_})
        axpts = Lc(p2[t2 == 2][::10], Q[p], XYZ[p], seed, M)
        if len(axpts):
            d, ii = tree.query(axpts, distance_upper_bound=4.0)
            hit = np.unique(ii[np.isfinite(d)])
            if len(hit):
                sel = rng.choice(hit, min(int(insyn[p]), len(hit)), replace=len(hit) < int(insyn[p]))
                for s in sel:
                    syns.append([round(tdl[s, 0], 1), round(tdl[s, 1], 1), c])
    tsoma = [round(dloc[0, 0], 1), round(dloc[0, 1], 1)]
    data = {"cells": cells, "syns": syns, "tsoma": tsoma, "n_pre": len(ins), "n_syn": len(syns)}
    html = HTML.replace("__DATA__", json.dumps(data))
    open(OUT, "w", encoding="utf-8").write(html)
    print(f"[3-1b] HTML ({len(ins)}개 전시냅스 전체, {len(syns)} 시냅스) -> {OUT}")
    print(f"       파일크기 {os.path.getsize(OUT)/1e6:.1f} MB")


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>3-1(b) wiring</title>
<style>html,body{margin:0;background:#0d0d12;color:#ddd;font-family:sans-serif;overflow:hidden}
#hud{position:fixed;top:8px;left:8px;font-size:13px;background:#0009;padding:8px 10px;border-radius:6px;line-height:1.5}
#leg{position:fixed;top:8px;right:8px;font-size:12px;background:#0009;padding:8px 10px;border-radius:6px}
canvas{display:block;cursor:grab}</style></head><body>
<div id="hud"><b>3-1(b) 완전체 wiring</b><br>휠=확대 · 드래그=이동 · 더블클릭=리셋<br><span id=info></span></div>
<div id="leg"></div><canvas id=c></canvas>
<script>
const D=__DATA__;const cv=document.getElementById('c'),cx=cv.getContext('2d');
let S=1,ox=0,oy=0;function fit(){cv.width=innerWidth;cv.height=innerHeight;
let xs=[],ys=[];for(const c of D.cells){for(let i=0;i<c.d.length;i+=2){xs.push(c.d[i]);ys.push(c.d[i+1]);}}
let x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
S=0.85*Math.min(cv.width/(x1-x0),cv.height/(y1-y0));ox=cv.width/2-S*(x0+x1)/2;oy=cv.height/2+S*(y0+y1)/2;draw();}
function TX(x){return ox+S*x}function TY(y){return oy-S*y}
function draw(){cx.clearRect(0,0,cv.width,cv.height);
for(const c of D.cells){cx.strokeStyle=c.ac;cx.lineWidth=Math.max(.3,S*0.5);cx.globalAlpha=.5;cx.beginPath();
for(let i=0;i<c.a.length;i+=4){cx.moveTo(TX(c.a[i]),TY(c.a[i+1]));cx.lineTo(TX(c.a[i+2]),TY(c.a[i+3]));}cx.stroke();}
for(const c of D.cells){cx.strokeStyle=c.dc;cx.lineWidth=Math.max(.5,S*1.1);cx.globalAlpha=.92;cx.beginPath();
for(let i=0;i<c.d.length;i+=4){cx.moveTo(TX(c.d[i]),TY(c.d[i+1]));cx.lineTo(TX(c.d[i+2]),TY(c.d[i+3]));}cx.stroke();}
cx.globalAlpha=1;for(const s of D.syns){star(TX(s[0]),TY(s[1]),Math.max(3,S*2.2),s[2]);}
cx.fillStyle='#fff';cx.beginPath();cx.arc(TX(D.tsoma[0]),TY(D.tsoma[1]),Math.max(4,S*3),0,7);cx.fill();
cx.strokeStyle='#000';cx.lineWidth=1.5;cx.stroke();}
function star(x,y,r,col){cx.beginPath();for(let i=0;i<10;i++){let a=Math.PI/5*i-Math.PI/2,rr=i%2?r*.45:r;
cx[i?'lineTo':'moveTo'](x+rr*Math.cos(a),y+rr*Math.sin(a));}cx.closePath();cx.fillStyle=col;cx.fill();
cx.strokeStyle='#000';cx.lineWidth=.6;cx.stroke();}
cv.onwheel=e=>{e.preventDefault();let f=e.deltaY<0?1.12:1/1.12;let mx=e.clientX,my=e.clientY;
ox=mx-(mx-ox)*f;oy=my-(my-oy)*f;S*=f;draw();};
let drag=false,px,py;cv.onmousedown=e=>{drag=true;px=e.clientX;py=e.clientY;cv.style.cursor='grabbing';};
onmouseup=()=>{drag=false;cv.style.cursor='grab';};onmousemove=e=>{if(drag){ox+=e.clientX-px;oy+=e.clientY-py;px=e.clientX;py=e.clientY;draw();}};
cv.ondblclick=fit;onresize=fit;
document.getElementById('info').textContent=`전시냅스 ${D.n_pre}개 · 시냅스 ${D.n_syn}개 · 수상돌기=진함/축삭=연함/★=시냅스`;
const cm={SP_PC:'#C44E52',SP_Ivy:'#8172B3',SP_PVBC:'#4C72B0',SP_CCKBC:'#55A868',SO_OLM:'#CCB974',SP_BS:'#DD8452',SO_Tri:'#937860',SR_SCA:'#DA8BC3',SO_BS:'#8C8C8C',SLM_PPA:'#64B5CD',SP_AA:'#E377C2',SO_BP:'#7F7F7F'};
let seen=new Set(D.cells.map(c=>c.mt));let lh='<b>종류</b><br>';for(const k in cm){if(seen.has(k))lh+=`<span style="color:${cm[k]}">■</span> ${k}<br>`;}
document.getElementById('leg').innerHTML=lh+'<span style="color:#fff">●</span> 대상 추체';
fit();
</script></body></html>"""


if __name__ == "__main__":
    main()
