#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""_mea_<TAG>.npz(--save_cellcur 저장본) → 전 슬라이스 fEPSP 3D 시각화 HTML.

예시 fepsp_3d_full.html을 전 슬라이스판으로 확장. 세포당 2점(소마/수상돌기)·세그먼트
막전류 축약·전극 fEPSP를 2D 캔버스 손수 3D 투영으로 그린다.

사용:  python make_fepsp3d.py <npz경로> [출력html]
"""
import sys, os, json
import numpy as np

HTML_TEMPLATE = r'''<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>fEPSP 3D — 전 슬라이스</title>
<style>
:root{--bg:#080b12;--panel:#0f141f;--panel2:#141b28;--edge:#243044;--ink:#e7eef7;--muted:#8697b0;--dim:#5a6a84;--gold:#ffc857;--sink:#4a90ff;--source:#ff5555;--soma:#ffb454;--dend:#5fd08a}
*{box-sizing:border-box}html,body{margin:0;height:100%}
body{background:var(--bg);color:var(--ink);font-family:system-ui,"Segoe UI",sans-serif;font-size:14px;display:flex;flex-direction:column;height:100vh;overflow:hidden}
.mono{font-variant-numeric:tabular-nums}
header{padding:10px 18px;border-bottom:1px solid var(--edge);background:var(--panel)}
header h1{margin:0 0 3px;font-size:15px;font-weight:700}
header .sub{color:var(--muted);font-size:11.5px}
.wrap{flex:1;display:flex;min-height:0}
.stage{flex:1;position:relative;min-width:0}
canvas#gl{width:100%;height:100%;display:block;cursor:grab}canvas#gl:active{cursor:grabbing}
.side{width:320px;border-left:1px solid var(--edge);background:var(--panel);padding:12px;overflow-y:auto;display:flex;flex-direction:column;gap:11px}
.lab{font-size:10px;text-transform:uppercase;letter-spacing:.6px;color:var(--dim);margin-bottom:5px}
.ctrl{display:flex;gap:7px;align-items:center}
button,select{background:var(--panel2);border:1px solid var(--edge);color:var(--ink);border-radius:8px;padding:7px 10px;font-size:12.5px;cursor:pointer}
button:hover{border-color:var(--gold)}
input[type=range]{width:100%;accent-color:var(--gold)}
.seg{display:flex;gap:5px}.seg button{flex:1;padding:6px 0;font-size:11.5px}
.seg button.on{border-color:var(--gold);color:var(--gold)}
.lays{display:flex;gap:5px}.lays button{flex:1;padding:6px 0;font-size:11px;opacity:.4}.lays button.on{opacity:1}
.chart{border:1px solid var(--edge);border-radius:10px;background:var(--panel2);padding:7px}
.chart canvas{width:100%;height:104px;display:block}
.tiles{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px}
.tile{background:var(--panel2);border:1px solid var(--edge);border-radius:9px;padding:7px 5px;text-align:center}
.tile .t{font-size:9.5px;color:var(--muted)}.tile .v{font-size:13.5px;font-weight:600;margin-top:2px}
.lg{font-size:10.5px;color:var(--muted);line-height:1.6}.lg .sw{display:inline-block;width:11px;height:11px;border-radius:3px;vertical-align:-1px;margin-right:5px}
.note{font-size:10px;color:var(--dim);line-height:1.5}
select{width:100%}
</style>
<header>
  <h1>fEPSP 발생 3D — 전 슬라이스 (like-slice · 세포 <b id="nc" class="mono"></b>)</h1>
  <div class="sub">SC volley(t=0) → 세포 <b>막전류</b>(소마/수상돌기) → 전극 <b>fEPSP</b>. <span style="color:var(--sink)">파랑=sink</span> · <span style="color:var(--source)">빨강=source</span> · <b id="lvl"></b></div>
</header>
<div class="wrap">
  <div class="stage"><canvas id="gl"></canvas></div>
  <div class="side">
    <div>
      <div class="lab">재생 · 시간 <span id="tnow" class="mono" style="float:right;color:var(--gold)">0.0 ms</span></div>
      <div class="ctrl"><button id="play" style="flex:1">일시정지</button><button id="rot" class="on" style="flex:1;border-color:var(--gold);color:var(--gold)">자동회전</button></div>
      <input type="range" id="seek" min="0" max="100" value="0" style="margin-top:8px">
    </div>
    <div><div class="lab">속도</div><div class="seg" id="spd"><button data-s="0.25">0.25x</button><button data-s="0.5" class="on">0.5x</button><button data-s="1">1x</button><button data-s="2">2x</button></div></div>
    <div><div class="lab">전극 층 표시</div><div class="lays" id="lays"><button data-l="SO" class="on">SO</button><button data-l="SP" class="on">SP</button><button data-l="SR" class="on">SR</button><button data-l="SLM" class="on">SLM</button></div></div>
    <div><div class="lab">자극 입력 (SC volley · t=0)</div><div class="chart"><canvas id="stw" style="height:50px"></canvas></div></div>
    <div><div class="lab">fEPSP 차트 전극</div><select id="esel"></select><div class="chart" style="margin-top:6px"><canvas id="fe"></canvas></div></div>
    <div><div class="lab">막전류 합 (수상돌기 vs 소마)</div><div class="chart"><canvas id="cu"></canvas></div></div>
    <div class="ctrl"><button id="syntog" class="on" style="flex:1">SC 시냅스</button><button id="catch" style="flex:1;opacity:.45">전극 관할</button></div>
    <div class="tiles"><div class="tile"><div class="t">선택전극 Ve</div><div class="v mono" id="vsel">0</div></div><div class="tile"><div class="t">최대 |Ve|</div><div class="v mono" id="vmax">0</div></div><div class="tile"><div class="t">발화 세포</div><div class="v mono" id="nspk">0</div></div></div>
    <div class="lg">
      <div class="k" style="margin-bottom:3px">뉴런 색 = 자극에 의한 막전류 변화 (ΔI)</div>
      <div style="display:flex;align-items:center;gap:6px;margin:2px 0 7px">
        <span style="color:#4a90ff">sink</span>
        <span style="flex:1;height:11px;border-radius:3px;background:linear-gradient(90deg,#4a90ff,#7a8496,#ff5555)"></span>
        <span style="color:#ff5555">source</span>
      </div>
      <div><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:#9aa7bd;vertical-align:-1px;margin-right:4px"></span>큰 점 = 소마 &nbsp; <span style="display:inline-block;width:5px;height:5px;border-radius:50%;background:#9aa7bd;vertical-align:1px;margin-right:4px"></span>작은 점 = 수상돌기</div>
      <div><span style="display:inline-block;width:14px;border-top:1px solid #9aa7bd;vertical-align:3px;margin-right:5px"></span>가는 선 = 같은 세포(소마↔수상돌기 쌍극자)</div>
      <div><span class="sw" style="background:#ffc857"></span>SC 시냅스 (자극 입력 지점)</div>
      <div><span style="display:inline-block;width:10px;height:10px;border:1.5px solid #fff;border-radius:2px;vertical-align:-1px;margin-right:5px"></span>MEA 전극 24</div>
      <div><span style="display:inline-block;width:11px;height:11px;border:2px solid #ffc857;border-radius:50%;vertical-align:-1px;margin-right:5px"></span>SC 자극 위치</div>
    </div>
    <div class="note">전규모 시뮬 실측 데이터 · 드래그=회전 · 세포당 2점(소마/수상돌기)</div>
  </div>
</div>
<script>
const D=__DATA__;
const gl=document.getElementById('gl'),gx=gl.getContext('2d');
const fe=document.getElementById('fe'),fx=fe.getContext('2d');
const cu=document.getElementById('cu'),cx=cu.getContext('2d');
const stw=document.getElementById('stw'),sw=stw.getContext('2d');
const N=D.n,NF=D.nf,POS=D.pos,SOMA=D.soma,CUR=D.cur,T=D.t,EL=D.elec,V=D.V,SYN=D.syn||[];
const ISOMA=D.Isoma,IDEND=D.Idend,ENAMES=D.enames;
const LAYC={SO:'#DD8452',SP:'#c084fc',SR:'#2dd4bf',SLM:'#ffd43b'};
function elayer(j){const m=ENAMES[j].match(/\((\w+)\)/);return m?m[1]:'SR';}
const ECOL=EL.map((_,j)=>LAYC[elayer(j)]||'#8aa');
const CMAX=D.cmax||1;
let frame=0,facc=0,speed=0.5,playing=true,yaw=0.62,pitch=0.16,drag=false,lx=0,ly=0,DPR=Math.min(2,devicePixelRatio||1);
let synOn=true,autorot=true,catchment=false,sel=D.rec_j||0,zoom=1;
let layOn={SO:true,SP:true,SR:true,SLM:true};
function eon(j){return layOn[elayer(j)];}
function hexA(h,a){return 'rgba('+parseInt(h.slice(1,3),16)+','+parseInt(h.slice(3,5),16)+','+parseInt(h.slice(5,7),16)+','+a.toFixed(2)+')';}
document.getElementById('nc').textContent=D.ncell.toLocaleString();
document.getElementById('lvl').textContent='자극 '+(D.level*100).toFixed(1)+'% (섬유 '+Math.round(D.level*200)+'/200)';
const seek=document.getElementById('seek');seek.max=NF-1;
const esel=document.getElementById('esel');
ENAMES.forEach((nm,j)=>{const o=document.createElement('option');o.value=j;o.textContent=nm;if(j===sel)o.selected=true;esel.appendChild(o);});
const span=(()=>{let mx=0;for(const p of POS)mx=Math.max(mx,Math.abs(p[0]),Math.abs(p[1]),Math.abs(p[2]));for(const e of EL)mx=Math.max(mx,Math.abs(e[0]),Math.abs(e[1]),Math.abs(e[2]));return mx*2.1;})();
function resize(){DPR=Math.min(2,devicePixelRatio||1);let r=gl.getBoundingClientRect();gl.width=r.width*DPR;gl.height=r.height*DPR;for(const cv of [fe,cu,stw]){let q=cv.getBoundingClientRect();cv.width=q.width*DPR;cv.height=q.height*DPR;}}
function proj(p,W,H,sc){const cY=Math.cos(yaw),sY=Math.sin(yaw),cP=Math.cos(pitch),sP=Math.sin(pitch);let X=p[0],Y=p[1],Z=p[2];let x1=X*cY-Z*sY,z1=X*sY+Z*cY;let y1=Y*cP-z1*sP,z2=Y*sP+z1*cP;const f=900/(900+z2);return [W/2+x1*f*sc,H/2+y1*f*sc,z2,f];}
function curColor(v){const n=v/CMAX,a=Math.min(1,Math.abs(n));if(n<-0.04)return 'rgba(74,144,255,'+(0.2+0.8*a).toFixed(2)+')';if(n>0.04)return 'rgba(255,85,85,'+(0.2+0.8*a).toFixed(2)+')';return 'rgba(120,130,150,0.26)';}
const LAYFILL={SO:'rgba(221,132,82,.09)',SP:'rgba(192,132,252,.10)',SR:'rgba(45,212,191,.09)',SLM:'rgba(255,212,59,.08)'};
function drawScene(W,H,sc){if(!D.box)return;const bu=D.box.u/2,br=D.box.r/2,bw=D.box.w/2;
  for(const L of (D.layers||[])){const c=[[-bu,L.r0,bw],[bu,L.r0,bw],[bu,L.r1,bw],[-bu,L.r1,bw]].map(p=>proj(p,W,H,sc));
    gx.fillStyle=LAYFILL[L.name]||'rgba(120,130,150,.05)';gx.beginPath();c.forEach((q,i)=>i?gx.lineTo(q[0],q[1]):gx.moveTo(q[0],q[1]));gx.closePath();gx.fill();
    const lb=proj([-bu,(L.r0+L.r1)/2,bw],W,H,sc);gx.fillStyle=LAYC[L.name]||'#888';gx.font='bold 12px system-ui';gx.textAlign='right';gx.fillText(L.name,lb[0]-5,lb[1]+4);gx.textAlign='left';}
  const cor=[];for(const su of[-bu,bu])for(const sr of[-br,br])for(const sw of[-bw,bw])cor.push([su,sr,sw]);
  const ed=[[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]];
  gx.strokeStyle='rgba(150,175,210,.28)';gx.lineWidth=1;for(const [a,b] of ed){const pa=proj(cor[a],W,H,sc),pb=proj(cor[b],W,H,sc);gx.beginPath();gx.moveTo(pa[0],pa[1]);gx.lineTo(pb[0],pb[1]);gx.stroke();}
  if(D.stim_locus){const q=proj(D.stim_locus,W,H,sc);gx.strokeStyle='#ffc857';gx.lineWidth=2;gx.beginPath();gx.arc(q[0],q[1],7,0,6.28);gx.stroke();gx.fillStyle='#ffc857';gx.font='bold 11px system-ui';gx.fillText('SC 자극',q[0]+10,q[1]+4);}}
function draw3d(){const W=gl.width/DPR,H=gl.height/DPR;gx.setTransform(DPR,0,0,DPR,0,0);gx.clearRect(0,0,W,H);
  const sc=Math.min(W,H)/(span*1.15)*zoom;const cf=CUR[frame];drawScene(W,H,sc);
  gx.strokeStyle='rgba(150,162,185,0.16)';gx.lineWidth=0.6;
  for(let c2=0;c2<N/2;c2++){const a=proj(POS[2*c2],W,H,sc),b=proj(POS[2*c2+1],W,H,sc);gx.beginPath();gx.moveTo(a[0],a[1]);gx.lineTo(b[0],b[1]);gx.stroke();}
  const ord=[];for(let i=0;i<N;i++){const q=proj(POS[i],W,H,sc);ord.push([q[2],i,q]);}ord.sort((a,b)=>a[0]-b[0]);
  for(const [,i,q] of ord){const v=cf[i];const r=(SOMA[i]?3.8:1.5)*(0.85+0.35*q[3]);let col;
    if(catchment){let bi=-1,bd=1e18;for(let j=0;j<EL.length;j++){if(!eon(j))continue;const dx=POS[i][0]-EL[j][0],dy=POS[i][1]-EL[j][1],dz=POS[i][2]-EL[j][2];const dd=dx*dx+dy*dy+dz*dz;if(dd<bd){bd=dd;bi=j;}}col=bi<0?'rgba(120,130,150,0.15)':hexA(ECOL[bi],0.18+0.45*Math.min(1,Math.abs(v)/CMAX));}
    else col=curColor(v);
    gx.fillStyle=col;gx.beginPath();gx.arc(q[0],q[1],r,0,6.28);gx.fill();}
  const stimOn=Math.abs(T[frame])<2.5;
  if(synOn){const col=stimOn?'rgba(255,231,140,0.95)':'rgba(214,168,60,0.4)';for(const s of SYN){const q=proj(s,W,H,sc);gx.fillStyle=col;gx.beginPath();gx.arc(q[0],q[1],1.4*(0.85+0.35*q[3]),0,6.28);gx.fill();}}
  if(stimOn){gx.fillStyle='rgba(255,200,87,0.95)';gx.font='bold 15px system-ui';gx.textAlign='center';gx.fillText('SC 자극 (t=0)',W/2,26);gx.textAlign='left';}
  for(let j=0;j<EL.length;j++){if(!eon(j))continue;const q=proj(EL[j],W,H,sc);const vv=V[j][frame];
    const rr=Math.max(3.5,(D.elec_diam||10)/2*sc*q[3]);
    const a=Math.min(0.85,Math.abs(vv)/((D.vglow||1)*10)*0.85);   // 신호 세기 → 채우기 진하기(크기 고정)
    if(a>0.02){gx.fillStyle=(vv<0?'rgba(74,144,255,':'rgba(255,85,85,')+a.toFixed(2)+')';gx.fillRect(q[0]-rr,q[1]-rr,2*rr,2*rr);}
    gx.strokeStyle=j===sel?'#fff':'rgba(255,255,255,.72)';gx.lineWidth=j===sel?2:1;gx.strokeRect(q[0]-rr,q[1]-rr,2*rr,2*rr);
    gx.textAlign='center';
    if(j===sel){gx.fillStyle='#fff';gx.font='bold 11px system-ui';gx.fillText(ENAMES[j],q[0],q[1]-rr-6);}
    else{gx.fillStyle='rgba(255,255,255,.62)';gx.font='9px system-ui';gx.fillText('#'+j,q[0],q[1]-rr-4);}
    gx.textAlign='left';}}
function lineChart(ctx,cv,series,cursorFrame){const W=cv.width/DPR,H=cv.height/DPR;ctx.setTransform(DPR,0,0,DPR,0,0);ctx.clearRect(0,0,W,H);
  const P={l:34,r:6,t:6,b:14};let mx=1e-9;for(const s of series){if(!s.on)continue;for(const y of s.data)mx=Math.max(mx,Math.abs(y));}
  const px=f=>P.l+f/(NF-1)*(W-P.l-P.r);const py=v=>H-P.b-((v+mx)/(2*mx))*(H-P.t-P.b);
  ctx.strokeStyle='rgba(120,150,190,.14)';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(P.l,py(0));ctx.lineTo(W-P.r,py(0));ctx.stroke();
  let f0=0;for(let f=0;f<NF;f++){if(T[f]>=0){f0=f;break;}}
  ctx.strokeStyle='rgba(255,200,87,.5)';ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(px(f0),P.t);ctx.lineTo(px(f0),H-P.b);ctx.stroke();ctx.setLineDash([]);
  for(const s of series){if(!s.on)continue;ctx.strokeStyle=s.col;ctx.lineWidth=1.6;ctx.beginPath();for(let f=0;f<NF;f++){const X=px(f),Y=py(s.data[f]);f?ctx.lineTo(X,Y):ctx.moveTo(X,Y);}ctx.stroke();}
  ctx.strokeStyle='rgba(255,255,255,.5)';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(px(cursorFrame),P.t);ctx.lineTo(px(cursorFrame),H-P.b);ctx.stroke();
  ctx.fillStyle='#5a6a84';ctx.font='8px system-ui';ctx.fillText(mx.toPrecision(2),1,P.t+7);ctx.fillText('t=0',px(f0)+2,H-4);}
function updateReadout(){document.getElementById('tnow').textContent=T[frame].toFixed(1)+' ms';
  const vv=V[sel][frame];const vs=document.getElementById('vsel');vs.textContent=(vv>0?'+':'')+vv.toFixed(1);vs.style.color=vv<0?'#4a90ff':vv>0?'#ff5555':'#8697b0';
  let vm=0;for(let j=0;j<EL.length;j++)if(eon(j))vm=Math.max(vm,Math.abs(V[j][frame]));document.getElementById('vmax').textContent=vm.toFixed(0);
  seek.value=frame;}
function loop(){if(playing){facc+=speed;while(facc>=1){frame=(frame+1)%NF;facc-=1;}}if(autorot&&!drag)yaw+=0.0022;draw3d();
  lineChart(sw,stw,[{data:D.stim_wave,col:'#ffc857',on:true}],frame);
  lineChart(fx,fe,[{data:V[sel],col:ECOL[sel],on:true}],frame);
  lineChart(cx,cu,[{data:IDEND,col:'#5fd08a',on:true},{data:ISOMA,col:'#ffb454',on:true}],frame);
  updateReadout();requestAnimationFrame(loop);}
document.getElementById('play').onclick=function(){playing=!playing;this.textContent=playing?'일시정지':'재생';};
seek.oninput=function(){frame=+this.value;playing=false;document.getElementById('play').textContent='재생';};
document.getElementById('spd').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;speed=+b.dataset.s;[...e.currentTarget.children].forEach(x=>x.classList.toggle('on',x===b));});
document.getElementById('lays').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;const l=b.dataset.l;layOn[l]=!layOn[l];b.classList.toggle('on',layOn[l]);});
esel.addEventListener('change',()=>{sel=+esel.value;});
document.getElementById('syntog').onclick=function(){synOn=!synOn;this.classList.toggle('on',synOn);this.style.opacity=synOn?1:.45;};
document.getElementById('rot').onclick=function(){autorot=!autorot;this.classList.toggle('on',autorot);};
document.getElementById('catch').onclick=function(){catchment=!catchment;this.classList.toggle('on',catchment);this.style.opacity=catchment?1:.45;};
gl.addEventListener('wheel',e=>{e.preventDefault();zoom*=Math.exp(-e.deltaY*0.0012);zoom=Math.max(0.3,Math.min(8,zoom));},{passive:false});
gl.addEventListener('mousedown',e=>{drag=true;lx=e.clientX;ly=e.clientY;});
window.addEventListener('mouseup',()=>drag=false);
window.addEventListener('mousemove',e=>{if(!drag)return;yaw+=(e.clientX-lx)*0.008;pitch=Math.max(-0.5,Math.min(1.0,pitch+(e.clientY-ly)*0.006));lx=e.clientX;ly=e.clientY;});
window.addEventListener('resize',resize);resize();loop();
</script>'''

npz = sys.argv[1] if len(sys.argv) > 1 else "figures/_mea_Sviz_smoke.npz"
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(npz) or ".",
                                                         "fepsp3d_" + os.path.basename(npz)[5:-4] + ".html")
d = np.load(npz, allow_pickle=True)

need = ["cell_pos", "cell_soma", "cell_cur", "Isoma", "Idend", "syn_xyz",
        "viz_V", "viz_elec", "viz_lay_name", "viz_lay_r0", "viz_lay_r1",
        "viz_box", "viz_stim_locus", "viz_t"]
miss = [k for k in need if k not in d.files]
if miss:
    raise SystemExit(f"npz에 시각화 필드 없음: {miss}\n→ --save_cellcur 로 다시 저장했는지 확인")

stim_t = float(d["stim_t"])
vt = np.asarray(d["viz_t"], float)
# 자극 전후 창으로 잘라 프레임 다운샘플(웹 로드용)
lo, hi = stim_t - 5.0, stim_t + 40.0
idx = np.where((vt >= lo) & (vt <= hi))[0]
if idx.size == 0:
    idx = np.arange(len(vt))
stride = max(1, idx.size // 90)
idx = idx[::stride]

cur = np.asarray(d["cell_cur"], float)[:, idx]       # (2N, nf)
V = np.asarray(d["viz_V"], float)[:, idx]            # (nelec, nf)
Isoma = np.asarray(d["Isoma"], float)[idx]
Idend = np.asarray(d["Idend"], float)[idx]
t = vt[idx] - stim_t                                  # volley = 0

# ★기저선 차감: 자극 전 평균을 빼서 '자극에 의한 변화(ΔI)'만 색으로. 자극 전=0(회색).
_pre = t < 0
if _pre.sum() >= 2:
    cur = cur - cur[:, _pre].mean(1, keepdims=True)
    V = V - V[:, _pre].mean(1, keepdims=True)
    Isoma = Isoma - Isoma[_pre].mean()
    Idend = Idend - Idend[_pre].mean()
pos = np.asarray(d["cell_pos"], float)
soma = np.asarray(d["cell_soma"], int)
elec = np.asarray(d["viz_elec"], float)
ell = [str(x) for x in d["el_layer"]]
enames = [f"#{i}({ell[i]})" for i in range(len(elec))]
syn = np.asarray(d["syn_xyz"], float)
stim = np.asarray(d["viz_stim_locus"], float)
lay_r0 = np.asarray(d["viz_lay_r0"], float)
lay_r1 = np.asarray(d["viz_lay_r1"], float)

# ── 재중심화: 세포 구름 중심을 원점으로(예시 뷰는 원점 대칭 박스를 가정) ──
lo = np.percentile(pos, 1, axis=0)
hi = np.percentile(pos, 99, axis=0)
ctr = (lo + hi) / 2.0
pos = pos - ctr
if syn.size:
    syn = syn - ctr
elec = elec - ctr
stim = stim - ctr
lay_r0 = lay_r0 - ctr[1]
lay_r1 = lay_r1 - ctr[1]
box = dict(u=float(hi[0] - lo[0]), r=float(hi[1] - lo[1]), w=float(hi[2] - lo[2]))
layers = [dict(name=str(n), r0=float(a), r1=float(b))
          for n, a, b in zip(d["viz_lay_name"], lay_r0, lay_r1)]

acur = np.abs(cur)
_nz = acur[acur > 0]
cmax = float(np.percentile(_nz, 99.5)) if _nz.size else 1.0
if cmax <= 0:
    cmax = 1.0
vglow = max(float(np.abs(V).max()) / 10.0, 1e-6)   # 전극 fEPSP 글로우 적응 스케일
stim_wave = np.exp(-((t) / 0.6) ** 2)               # 자극(SC volley) 입력 파형: t=0 펄스

D = dict(
    n=int(pos.shape[0]), nf=int(idx.size), ncell=int(pos.shape[0] // 2), cmax=round(cmax, 5),
    vglow=round(vglow, 5),
    pos=[[round(float(x), 1) for x in p] for p in pos],
    soma=[int(x) for x in soma],
    cur=[[round(float(cur[i, f]), 2) for i in range(cur.shape[0])] for f in range(cur.shape[1])],
    t=[round(float(x), 2) for x in t],
    elec=[[round(float(x), 1) for x in e] for e in elec],
    enames=enames,
    V=[[round(float(V[j, f]), 2) for f in range(V.shape[1])] for j in range(V.shape[0])],
    Isoma=[round(float(x), 2) for x in Isoma],
    Idend=[round(float(x), 2) for x in Idend],
    stim_wave=[round(float(x), 3) for x in stim_wave],
    syn=[[round(float(x), 1) for x in s] for s in syn],
    box=box, layers=layers,
    stim_locus=[round(float(x), 1) for x in stim],
    elec_diam=10, rec_j=int(d["rec_j"]),
    level=float(d["io_test"]) if "io_test" in d.files else 0.02,
)

html = HTML_TEMPLATE.replace("__DATA__", json.dumps(D, separators=(",", ":")))
with open(out, "w", encoding="utf-8") as f:
    f.write(html)
sz = os.path.getsize(out) / 1e6
print(f"saved: {out}  ({sz:.1f} MB · 점 {D['n']} · 프레임 {D['nf']} · 전극 {len(elec)} · SC시냅스 {len(syn)})")
