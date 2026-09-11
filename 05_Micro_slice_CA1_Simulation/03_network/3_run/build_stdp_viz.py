# -*- coding: utf-8 -*-
"""STDP 전파 3D 뷰어 빌더 — scratch/stdp_viz_dt{DT}_ca{CA}.npz → 자립형 Three.js HTML.
세그먼트 Vm(30회 평균)을 색으로, 재생 시 EPSP 전파(시냅스→소마) + bAP 역전파(소마→수상돌기) 관찰.
실행: build_stdp_viz.py --fe scratch/stdp_viz_dt10_ca0.npz --out .../ui/stdp_viz_dt10.html"""
import os, sys, base64, json
import numpy as np

def arg(f, d=None): return sys.argv[sys.argv.index(f)+1] if f in sys.argv else d
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
fe = arg("--fe", os.path.join(ROOT, "scratch", "stdp_viz_dt10_ca0.npz"))
out = arg("--out", os.path.join(ROOT, "04_experiments", "Ex10_STDP_pair", "ui", "stdp_viz.html"))
os.makedirs(os.path.dirname(out), exist_ok=True)

d = np.load(fe, allow_pickle=True)
pos = d["segpos"].astype(float); V = d["Vavg"].astype(float)   # (nseg, nframes)
twin = d["twin"].astype(float); synpos = d["synpos"].astype(float); soma_pos = d["soma_pos"].astype(float)
DT = int(d["dt"]); CA = int(d["ca_stp"]); NP = int(d["npair"])
nseg, nfr = V.shape

# 중심·스케일 정규화 (뷰 좌표)
c = pos.mean(axis=0); P = pos - c
scale = 1.0 / (np.abs(P).max() + 1e-6)
P *= scale; syn = (synpos - c) * scale; som = (soma_pos - c) * scale

# Vm 색양자화 (uint8): [vlo, vhi] -> 0..255
vlo, vhi = -75.0, 40.0
Vq = np.clip((V - vlo) / (vhi - vlo), 0, 1)
Vq = (Vq * 255).astype(np.uint8)                    # (nseg, nfr)
Vb64 = base64.b64encode(Vq.tobytes(order="C")).decode()

data = dict(nseg=nseg, nfr=nfr, pos=[round(float(x), 4) for x in P.flatten()],
            syn=[round(float(x), 4) for x in syn], som=[round(float(x), 4) for x in som],
            twin=[round(float(x), 2) for x in twin], dt=DT, ca=CA, npair=NP,
            vlo=vlo, vhi=vhi, Vb64=Vb64)

HTML = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>STDP 전파 (__DT__ms)</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/examples/js/controls/OrbitControls.js"></script>
<style>
:root{--bg:#0e1016;--panel:#171a24;--ink:#e8eaf2;--muted:#9aa0b4;--acc:#7a9cff;--line:#2a2f3e}
*{box-sizing:border-box}html,body{margin:0;height:100%;background:var(--bg);color:var(--ink);
 font-family:"Malgun Gothic",system-ui,sans-serif;overflow:hidden}
#c{position:fixed;inset:0}
.hud{position:fixed;left:16px;top:14px;background:rgba(23,26,36,.86);border:1px solid var(--line);
 border-radius:12px;padding:12px 16px;max-width:320px;backdrop-filter:blur(6px)}
.hud h1{margin:0 0 4px;font-size:1.05rem}.hud .s{font-size:.8rem;color:var(--muted);line-height:1.5}
.phase{display:inline-block;margin-top:6px;padding:2px 10px;border-radius:20px;font-size:.82rem;font-weight:600}
.bar{position:fixed;left:50%;bottom:18px;transform:translateX(-50%);width:min(680px,92vw);
 background:rgba(23,26,36,.9);border:1px solid var(--line);border-radius:14px;padding:12px 18px}
.row{display:flex;align-items:center;gap:12px}
button{font:inherit;font-size:.85rem;background:#222738;color:var(--ink);border:1px solid var(--line);
 border-radius:9px;padding:7px 14px;cursor:pointer}button:hover{border-color:var(--acc)}
input[type=range]{flex:1;accent-color:var(--acc)}
.t{font-family:"Consolas",monospace;font-size:.85rem;color:var(--acc);min-width:92px;text-align:right}
.cb{position:fixed;right:16px;top:14px;background:rgba(23,26,36,.86);border:1px solid var(--line);
 border-radius:10px;padding:10px 12px;font-size:.75rem;color:var(--muted);text-align:center}
.cbbar{width:16px;height:120px;margin:6px auto;border-radius:4px;
 background:linear-gradient(to top,#2b3a8f,#3f6fd0,#8a8f98,#d97a2a,#d63a3a)}
.lg{position:fixed;right:16px;bottom:18px;font-size:.78rem;color:var(--muted)}
.lg b{display:inline-block;width:10px;height:10px;border-radius:50%;vertical-align:-1px;margin-right:4px}
</style></head><body>
<canvas id="c"></canvas>
<div class="hud"><h1>STDP 신호 전파 · 30회 평균</h1>
 <div class="s">SC→PC 단일세포 · Δt = <b id="dt"></b> ms · ca_stp <b id="ca"></b><br>
 EPSP가 <span style="color:#d97a2a">시냅스</span>에서 소마로 전파 → 소마 스파이크의 <b>bAP</b>가 수상돌기로 역전파해 시냅스 칼슘 형성</div>
 <div><span class="phase" id="phase"></span></div></div>
<div class="cb">막전위<div class="cbbar"></div><span>+40 / −75 mV</span></div>
<div class="lg"><span><b style="background:#d97a2a"></b>시냅스(SC)</span> &nbsp; <span><b style="background:#5fe0a0"></b>소마</span></div>
<div class="bar"><div class="row">
 <button id="play">⏸ 일시정지</button>
 <input type="range" id="sl" min="0" max="100" value="0" step="1">
 <span class="t" id="tt">0.0 ms</span>
</div></div>
<script>
const D = __DATA__;
const V = Uint8Array.from(atob(D.Vb64), c=>c.charCodeAt(0));   // nseg*nfr
document.getElementById('dt').textContent=(D.dt>=0?'+':'')+D.dt;
document.getElementById('ca').textContent=D.ca;

const cv=document.getElementById('c');
const rn=new THREE.WebGLRenderer({canvas:cv,antialias:true});rn.setPixelRatio(devicePixelRatio);rn.setSize(innerWidth,innerHeight);
const sc=new THREE.Scene();sc.background=new THREE.Color(0x0e1016);
const cam=new THREE.PerspectiveCamera(55,innerWidth/innerHeight,0.01,100);cam.position.set(0,0,3.0);
let ctl=null; if(THREE.OrbitControls){ctl=new THREE.OrbitControls(cam,rn.domElement);ctl.enableDamping=true;}
sc.add(new THREE.AmbientLight(0xffffff,0.9));
// 색 램프 (파랑→회색→빨강)
function ramp(u){ // u 0..1
 const stops=[[43,58,143],[63,111,208],[138,143,152],[217,122,42],[214,58,58]];
 const x=u*(stops.length-1),i=Math.min(stops.length-2,Math.floor(x)),f=x-i;
 const a=stops[i],b=stops[i+1];return [(a[0]+(b[0]-a[0])*f)/255,(a[1]+(b[1]-a[1])*f)/255,(a[2]+(b[2]-a[2])*f)/255];}
// 세그먼트 포인트
const g=new THREE.BufferGeometry();
const posArr=new Float32Array(D.pos);g.setAttribute('position',new THREE.BufferAttribute(posArr,3));
const col=new Float32Array(D.nseg*3);g.setAttribute('color',new THREE.BufferAttribute(col,3));
const pts=new THREE.Points(g,new THREE.PointsMaterial({size:0.028,vertexColors:true}));sc.add(pts);
// 시냅스·소마 마커
function marker(p,color,r){const m=new THREE.Mesh(new THREE.SphereGeometry(r,20,20),new THREE.MeshBasicMaterial({color}));m.position.set(p[0],p[1],p[2]);sc.add(m);return m;}
marker(D.syn,0xd97a2a,0.05); marker(D.som,0x5fe0a0,0.055);

let fr=0, playing=true, speed=1;
const sl=document.getElementById('sl'),tt=document.getElementById('tt'),phase=document.getElementById('phase');
sl.max=D.nfr-1;
function paint(f){
 for(let s=0;s<D.nseg;s++){const u=V[s*D.nfr+f]/255;const rgb=ramp(u);col[s*3]=rgb[0];col[s*3+1]=rgb[1];col[s*3+2]=rgb[2];}
 g.attributes.color.needsUpdate=true;
 const tv=D.twin[f];tt.textContent=tv.toFixed(1)+' ms';sl.value=f;
 // 위상 라벨
 let lab,cl;
 if(tv<0){lab='자극 전(기저)';cl='#6b7080';}
 else if(tv<D.dt){lab='① pre EPSP → 소마 전파';cl='#d97a2a';}
 else if(tv<D.dt+40){lab='② post 버스트 → bAP 역전파';cl='#d63a3a';}
 else {lab='③ 칼슘 감쇠·회복';cl='#3f6fd0';}
 phase.textContent=lab;phase.style.background=cl+'33';phase.style.color=cl;
}
sl.oninput=()=>{fr=+sl.value;playing=false;setPlay();paint(fr);};
const pb=document.getElementById('play');
function setPlay(){pb.textContent=playing?'⏸ 일시정지':'▶ 재생';}
pb.onclick=()=>{playing=!playing;setPlay();};
paint(0);
let acc=0,last=performance.now();
function loop(now){const dt=now-last;last=now;
 if(playing){acc+=dt*speed;if(acc>33){fr=(fr+1)%D.nfr;paint(fr);acc=0;}}
 if(ctl)ctl.update();else sc.rotation.y+=0.003;rn.render(sc,cam);requestAnimationFrame(loop);}
requestAnimationFrame(loop);
addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();rn.setSize(innerWidth,innerHeight);});
</script></body></html>"""

open(out, "w", encoding="utf-8").write(HTML.replace("__DATA__", json.dumps(data)).replace("__DT__", str(DT)))
print(f"[stdp-viz-html] {out} · 세그 {nseg} · 프레임 {nfr} · Δt{DT:+d} ca{CA} · 평균{NP}회")
