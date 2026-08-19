# -*- coding: utf-8 -*-
"""
03_network/1_connectome/make_wiring_html.py  —  3-1(b) 완전체 wiring 인터랙티브 HTML

대상 추체 + 연결된 전시냅스 세포 전체(기본 83개)의 형태를 canvas로 그리고
마우스 휠 확대 / 드래그 이동이 되는 자체완결 HTML을 만든다. 흰 배경.
  - 종류=색 · 수상돌기=진한톤(연속) · 축삭=연한톤 · 시냅스=★
  - 형태는 **연속 폴리라인**(점 1회 저장)으로 컴팩트하게.
결과: scratch/3-3_wiring.html

실행: python 03_network/1_connectome/make_wiring_html.py
"""
import os
import json
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
import matplotlib.colors as mc

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DERIVED = os.path.join(ROOT, "data", "derived")
LIB = os.path.join(ROOT, "data", "morphology_library", "morphology_library")
CFG = os.path.join(ROOT, "config", "window_layout.json")
OUT = os.path.join(ROOT, "scratch", "3-3_wiring.html")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
DEND_STRIDE, AXON_STRIDE = 2, 10   # 폴리라인이라 점 줄여도 연속(끊김X) · 16MB 대비
CMAP = {"SP_PC": "#C44E52", "SP_Ivy": "#8172B3", "SP_PVBC": "#4C72B0", "SP_CCKBC": "#55A868",
        "SO_OLM": "#CCB974", "SP_BS": "#DD8452", "SO_Tri": "#937860", "SR_SCA": "#DA8BC3",
        "SO_BS": "#8C8C8C", "SLM_PPA": "#64B5CD", "SP_AA": "#E377C2", "SO_BP": "#7F7F7F"}


def darkhex(c, f=0.6):
    r, g, b = mc.to_rgb(c)
    return "#%02x%02x%02x" % (int(r*f*255), int(g*f*255), int(b*f*255))


def load(path):
    r = np.loadtxt(path, comments="#")
    return r[:, 1].astype(int), r[:, 2:5].astype(np.float64), r[:, 0].astype(int), r[:, 6].astype(int)


def Lc(pts, q, xyz0, seed, M):
    return (xyz0 + Rot.from_quat(q[[1, 2, 3, 0]]).apply(pts) - seed) @ M


def polylines(typ, pts, idx, par, q, xyz0, seed, M, which, stride):
    """연속 폴리라인 리스트: 각 [x0,y0,x1,y1,...] (µm 정수). 미분지 구간 단위."""
    id2 = {i: k for k, i in enumerate(idx)}
    loc = Lc(pts, q, xyz0, seed, M)
    ch = defaultdict(list)
    for k in range(len(idx)):
        if par[k] in id2:
            ch[id2[par[k]]].append(k)
    inw = lambda k: typ[k] in which
    starts = []
    for k in range(len(idx)):
        if not inw(k):
            continue
        p = id2.get(par[k], -1)
        if p < 0 or not inw(p) or len(ch[p]) > 1:
            starts.append(k)
    polys = []
    for s in starts:
        chain = []
        p = id2.get(par[s], -1)
        if p >= 0 and inw(p):
            chain.append(p)
        k = s
        while True:
            chain.append(k)
            nx = [c for c in ch[k] if inw(c)]
            if len(nx) == 1:
                k = nx[0]
            else:
                break
        if stride > 1 and len(chain) > 2:
            keep = chain[::stride]
            if keep[-1] != chain[-1]:
                keep.append(chain[-1])
            chain = keep
        poly = []
        for k in chain:
            poly += [int(round(loc[k, 0])), int(round(loc[k, 1]))]
        if len(poly) >= 4:
            polys.append(poly)
    return polys, loc


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
    typ, pts, idx, par = load(os.path.join(LIB, morph[tgt] + ".swc"))
    dp, dloc = polylines(typ, pts, idx, par, Q[tgt], XYZ[tgt], seed, M, (1, 3, 4), DEND_STRIDE)
    ap, _ = polylines(typ, pts, idx, par, Q[tgt], XYZ[tgt], seed, M, (2,), AXON_STRIDE)
    cells.append({"mt": "TARGET", "dc": darkhex(CMAP["SP_PC"]), "ac": CMAP["SP_PC"], "d": dp, "a": ap})
    dmask = (typ == 1) | (typ == 3) | (typ == 4)
    tdl = Lc(pts[dmask], Q[tgt], XYZ[tgt], seed, M); tree = cKDTree(tdl)

    syns = []
    for p in ins:
        c = CMAP.get(mt[p], "#888888")
        t2, p2, i2, pr2 = load(os.path.join(LIB, morph[p] + ".swc"))
        dp2, _ = polylines(t2, p2, i2, pr2, Q[p], XYZ[p], seed, M, (1, 3, 4), DEND_STRIDE)
        ap2, _ = polylines(t2, p2, i2, pr2, Q[p], XYZ[p], seed, M, (2,), AXON_STRIDE)
        cells.append({"mt": mt[p], "dc": darkhex(c), "ac": c, "d": dp2, "a": ap2})
        axpts = Lc(p2[t2 == 2][::10], Q[p], XYZ[p], seed, M)
        if len(axpts):
            d, ii = tree.query(axpts, distance_upper_bound=4.0)
            hit = np.unique(ii[np.isfinite(d)])
            if len(hit):
                sel = rng.choice(hit, min(int(insyn[p]), len(hit)), replace=len(hit) < int(insyn[p]))
                for s in sel:
                    syns.append([int(round(tdl[s, 0])), int(round(tdl[s, 1])), c, mt[p]])
    data = {"cells": cells, "syns": syns, "tsoma": [int(round(dloc[0, 0])), int(round(dloc[0, 1]))],
            "n_pre": len(ins), "n_syn": len(syns)}
    open(OUT, "w", encoding="utf-8").write(HTML.replace("__DATA__", json.dumps(data)))
    print(f"[3-3] HTML ({len(ins)} 전시냅스 · {len(syns)} 시냅스) -> {OUT}")
    print(f"       파일크기 {os.path.getsize(OUT)/1e6:.1f} MB")


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>3-1(b) wiring</title>
<style>html,body{margin:0;background:#fff;color:#333;font-family:sans-serif;overflow:hidden}
#hud{position:fixed;top:8px;left:8px;font-size:13px;background:#ffffffd0;border:1px solid #ccc;padding:8px 10px;border-radius:6px;line-height:1.5}
#leg{position:fixed;top:8px;right:8px;font-size:12px;background:#ffffffd0;border:1px solid #ccc;padding:8px 10px;border-radius:6px}
canvas{display:block;cursor:grab;background:#fff}</style></head><body>
<div id="hud"><b>3-1(b) 완전체 wiring</b><br>휠=확대 · 드래그=이동 · 더블클릭=리셋<br><span id=info></span></div>
<div id="leg"></div><canvas id=c></canvas>
<script>
const D=__DATA__;const cv=document.getElementById('c'),cx=cv.getContext('2d');
let S=1,ox=0,oy=0;
function fit(){cv.width=innerWidth;cv.height=innerHeight;let x0=1e9,x1=-1e9,y0=1e9,y1=-1e9;
for(const c of D.cells)for(const p of c.d)for(let i=0;i<p.length;i+=2){if(p[i]<x0)x0=p[i];if(p[i]>x1)x1=p[i];if(p[i+1]<y0)y0=p[i+1];if(p[i+1]>y1)y1=p[i+1];}
S=0.85*Math.min(cv.width/(x1-x0),cv.height/(y1-y0));ox=cv.width/2-S*(x0+x1)/2;oy=cv.height/2+S*(y0+y1)/2;draw();}
function TX(x){return ox+S*x}function TY(y){return oy-S*y}
function stroke(polys){cx.beginPath();for(const p of polys){cx.moveTo(TX(p[0]),TY(p[1]));for(let i=2;i<p.length;i+=2)cx.lineTo(TX(p[i]),TY(p[i+1]));}cx.stroke();}
let hidden=new Set();
function vis(m){return m==='TARGET'||!hidden.has(m);}
function draw(){cx.clearRect(0,0,cv.width,cv.height);cx.lineCap='round';cx.lineJoin='round';
for(const c of D.cells){if(!vis(c.mt))continue;cx.strokeStyle=c.ac;cx.lineWidth=Math.max(.25,S*0.35);cx.globalAlpha=.30;stroke(c.a);}
for(const c of D.cells){if(!vis(c.mt))continue;cx.strokeStyle=c.dc;cx.lineWidth=Math.max(.6,S*1.2);cx.globalAlpha=.95;stroke(c.d);}
cx.globalAlpha=1;for(const s of D.syns){if(!vis(s[3]))continue;star(TX(s[0]),TY(s[1]),Math.max(3,S*2.2),s[2]);}
cx.fillStyle='#000';cx.beginPath();cx.arc(TX(D.tsoma[0]),TY(D.tsoma[1]),Math.max(4,S*3),0,7);cx.fill();}
function star(x,y,r,col){cx.beginPath();for(let i=0;i<10;i++){let a=Math.PI/5*i-Math.PI/2,rr=i%2?r*.45:r;cx[i?'lineTo':'moveTo'](x+rr*Math.cos(a),y+rr*Math.sin(a));}cx.closePath();cx.fillStyle=col;cx.fill();cx.strokeStyle='#000';cx.lineWidth=.6;cx.stroke();}
cv.onwheel=e=>{e.preventDefault();let f=e.deltaY<0?1.12:1/1.12;ox=e.clientX-(e.clientX-ox)*f;oy=e.clientY-(e.clientY-oy)*f;S*=f;draw();};
let drag=false,px,py;cv.onmousedown=e=>{drag=true;px=e.clientX;py=e.clientY;cv.style.cursor='grabbing';};
onmouseup=()=>{drag=false;cv.style.cursor='grab';};onmousemove=e=>{if(drag){ox+=e.clientX-px;oy+=e.clientY-py;px=e.clientX;py=e.clientY;draw();}};
cv.ondblclick=fit;onresize=fit;
document.getElementById('info').textContent=`전시냅스 세포 ${D.n_pre}개 · 시냅스 ${D.n_syn}개 · 수상돌기=진함 / 축삭=연함 / ★=시냅스`;
const cm={SP_PC:'#C44E52',SP_Ivy:'#8172B3',SP_PVBC:'#4C72B0',SP_CCKBC:'#55A868',SO_OLM:'#CCB974',SP_BS:'#DD8452',SO_Tri:'#937860',SR_SCA:'#DA8BC3',SO_BS:'#8C8C8C',SLM_PPA:'#64B5CD',SP_AA:'#E377C2',SO_BP:'#7F7F7F'};
let ccnt={},scnt={};for(const c of D.cells)if(c.mt!=='TARGET')ccnt[c.mt]=(ccnt[c.mt]||0)+1;for(const s of D.syns)scnt[s[3]]=(scnt[s[3]]||0)+1;
let lh='<b>종류 (체크=표시)</b><br><label style="display:block"><input type=checkbox id=allck checked> <b>전체</b></label><hr style="margin:3px 0">';
for(const k in cm)if(ccnt[k])lh+=`<label style="display:block;cursor:pointer"><input type=checkbox checked data-mt="${k}"><span style="color:${cm[k]}">■</span> ${k} <span style="color:#888">(${ccnt[k]}세포·${scnt[k]||0}시냅스)</span></label>`;
lh+='<hr style="margin:3px 0"><span style="color:#000">●</span> 대상 추체';
document.getElementById('leg').innerHTML=lh;
document.querySelectorAll('#leg input[data-mt]').forEach(cb=>cb.onchange=()=>{cb.checked?hidden.delete(cb.dataset.mt):hidden.add(cb.dataset.mt);draw();});
document.getElementById('allck').onchange=e=>{document.querySelectorAll('#leg input[data-mt]').forEach(cb=>{cb.checked=e.target.checked;cb.checked?hidden.delete(cb.dataset.mt):hidden.add(cb.dataset.mt);});draw();};
fit();
</script></body></html>"""


if __name__ == "__main__":
    main()
