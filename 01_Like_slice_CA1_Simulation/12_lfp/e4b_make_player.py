# -*- coding: utf-8 -*-
"""12_lfp/e4b_make_player.py  —  단독 실행 HTML 플레이어 생성

_e4b_anim.json(실제 시뮬 시계열)을 임베드한 자체완결 HTML(e4b_player.html)을 만든다.
브라우저에서 더블클릭으로 열어 재생·클릭 제어(느린 배속)로 fEPSP 생성 과정을 확인.
실행: <ca1sim>/python.exe 12_lfp/e4b_make_player.py
"""
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "figures", "_e4b_anim.json"), encoding="utf-8"))
DJ = json.dumps({k: data[k] for k in ("t", "vm_soma", "im_soma", "im_syn", "ve")}, separators=(",", ":"))

HTML = """<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>한 전극에서 fEPSP가 만들어지는 과정 (재생)</title>
<style>
 body{font-family:'Malgun Gothic',system-ui,sans-serif;background:#faf9f6;color:#2c2c2a;margin:0;padding:20px;}
 .wrap{max-width:900px;margin:0 auto;}
 h1{font-size:20px;font-weight:600;margin:0 0 4px;}
 .sub{color:#5f5e5a;font-size:13px;margin:0 0 16px;}
 .cards{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:16px;}
 .card{background:#f1efe8;border-radius:8px;padding:10px 12px;}
 .card .l{font-size:12px;color:#5f5e5a;} .card .v{font-size:22px;font-weight:600;}
 .main{display:grid;grid-template-columns:230px 1fr;gap:16px;align-items:start;}
 svg,canvas{border:1px solid #e2e0d8;border-radius:8px;background:#f1efe8;}
 .ctrl{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:14px;}
 button{font:inherit;font-size:14px;padding:6px 12px;border:1px solid #cfcdc4;background:#fff;border-radius:8px;cursor:pointer;}
 button:hover{background:#f1efe8;} button:active{transform:scale(.97);}
 #spd{display:inline-flex;border:1px solid #cfcdc4;border-radius:8px;overflow:hidden;}
 #spd button{border:0;border-radius:0;}
 input[type=range]{flex:1;min-width:180px;}
</style></head>
<body><div class="wrap">
<h1>한 전극에서 fEPSP가 만들어지는 과정</h1>
<p class="sub">뉴런 발화(①막전위) → ②막전류 → ③거리·MEA 영상법 공식 → ⑤전극 세포외 fEPSP. 실제 NEURON 시뮬 데이터. ▶ 재생·배속·스크럽으로 느리게 확인.</p>
<div class="cards">
 <div class="card"><div class="l">시간</div><div class="v"><span id="rT">0.0</span> ms</div></div>
 <div class="card"><div class="l">① 소마 막전위</div><div class="v" style="color:#c0392b"><span id="rVm">-65</span> mV</div></div>
 <div class="card"><div class="l">② 소마 막전류</div><div class="v" style="color:#7d3c98"><span id="rIm">0.00</span> nA</div></div>
 <div class="card"><div class="l">⑤ 전극 fEPSP</div><div class="v" style="color:#185fa5"><span id="rVe">0.00</span> µV</div></div>
</div>
<div class="main">
 <svg id="scene" viewBox="0 0 230 330" style="width:100%;height:auto;">
  <rect x="0" y="0" width="230" height="330" fill="#f1efe8"/>
  <rect x="0" y="300" width="230" height="30" fill="#5f5e5a"/>
  <text x="8" y="319" fill="#f1efe8" style="font-size:11px;">유리 MEA 전극면 (z=0)</text>
  <text x="120" y="24" fill="#5f5e5a" style="font-size:11px;">조직 슬라이스</text>
  <line id="contrib" x1="96" y1="248" x2="150" y2="300" stroke="#378ADD" stroke-width="1.5" stroke-dasharray="4 3" opacity="0.3"/>
  <line x1="96" y1="150" x2="96" y2="238" stroke="#888780" stroke-width="6" stroke-linecap="round"/>
  <line x1="96" y1="60" x2="96" y2="150" stroke="#888780" stroke-width="4" stroke-linecap="round"/>
  <circle id="glow" cx="96" cy="250" r="26" fill="#E24B4A" opacity="0"/>
  <circle id="soma" cx="96" cy="250" r="13" fill="#B4B2A9" stroke="#5f5e5a" stroke-width="1"/>
  <polygon id="syn" points="88,150 104,150 96,138" fill="#85B7EB"/>
  <text x="110" y="146" fill="#5f5e5a" style="font-size:10px;">SR 시냅스</text>
  <text x="58" y="270" fill="#5f5e5a" style="font-size:10px;">소마</text>
  <rect x="144" y="294" width="12" height="12" rx="2" fill="#E24B4A"/>
  <circle id="ring" cx="150" cy="300" r="8" fill="none" stroke="#185FA5" stroke-width="0" opacity="0.85"/>
  <text x="150" y="292" fill="#185fa5" text-anchor="middle" style="font-size:10px;">전극 j</text>
  <text x="126" y="284" fill="#1f618d" style="font-size:10px;">r</text>
 </svg>
 <canvas id="cv" style="width:100%;height:330px;"></canvas>
</div>
<div class="ctrl">
 <button id="play" style="width:46px;">▶</button>
 <button id="reset">↺ 처음</button>
 <div id="spd"><button data-s="0.25">0.25×</button><button data-s="0.5">0.5×</button><button data-s="1">1×</button><button data-s="2">2×</button></div>
 <input id="scrub" type="range" min="0" max="__MAX__" step="1" value="0"/>
</div>
</div>
<script>
const D=__DATA__;const N=D.t.length;
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
function setup(){const r=cv.getBoundingClientRect(),dp=window.devicePixelRatio||1;cv.width=r.width*dp;cv.height=330*dp;ctx.setTransform(dp,0,0,dp,0,0);}
setup();
const panels=[{k:'vm_soma',lab:'① 소마 막전위 V_m (mV)',col:'#E24B4A',min:-72,max:42},
{k:'im_soma',lab:'② 소마 막전류 I_m (nA) · 아래=sink',col:'#7F77DD',min:-1.3,max:0.95},
{k:'ve',lab:'⑤ 전극 세포외 fEPSP V_e (µV)',col:'#185FA5',min:-2.6,max:2.7}];
function draw(f){const W=cv.getBoundingClientRect().width,H=330,ph=H/3,padL=46,padR=10,padT=16,padB=10;
 ctx.clearRect(0,0,W,H);panels.forEach((p,pi)=>{const y0=pi*ph+padT,y1=(pi+1)*ph-padB,pw=W-padL-padR;
  const yv=v=>y1-(v-p.min)/(p.max-p.min)*(y1-y0),xi=i=>padL+i/(N-1)*pw;
  ctx.strokeStyle='rgba(120,120,120,.3)';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(padL,yv(0));ctx.lineTo(padL+pw,yv(0));ctx.stroke();
  ctx.fillStyle='#5f5e5a';ctx.font='11px sans-serif';ctx.textAlign='left';ctx.fillText(p.lab,padL,y0-4);
  ctx.strokeStyle=p.col;ctx.lineWidth=2;ctx.beginPath();
  for(let i=0;i<=f;i++){const X=xi(i),Y=yv(D[p.k][i]);i?ctx.lineTo(X,Y):ctx.moveTo(X,Y);}ctx.stroke();
  const cx=xi(f),cy=yv(D[p.k][f]);ctx.strokeStyle='rgba(120,120,120,.5)';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(cx,y0);ctx.lineTo(cx,y1);ctx.stroke();
  ctx.fillStyle=p.col;ctx.beginPath();ctx.arc(cx,cy,3.5,0,7);ctx.fill();});}
const g=id=>document.getElementById(id);
function lerp(a,b,t){const A=[1,3,5].map(i=>parseInt(a.substr(i,2),16)),B=[1,3,5].map(i=>parseInt(b.substr(i,2),16));
 return 'rgb('+A.map((v,i)=>Math.round(v+(B[i]-v)*t)).join(',')+')';}
function scene(f){const vm=D.vm_soma[f],im=D.im_soma[f],iy=D.im_syn[f],ve=D.ve[f];
 const sp=Math.max(0,Math.min(1,(vm+66)/104));g('glow').setAttribute('opacity',(sp*.85).toFixed(2));g('soma').setAttribute('fill',lerp('#B4B2A9','#E24B4A',sp));
 const sk=Math.max(0,Math.min(1,Math.abs(iy)/.1));g('syn').setAttribute('fill',lerp('#D3D1C7',iy<0?'#378ADD':'#D85A30',sk));
 const vn=Math.min(1,Math.abs(ve)/2.4);const cc=ve<0?'#378ADD':'#D85A30';
 g('contrib').setAttribute('opacity',(.2+vn*.8).toFixed(2));g('contrib').setAttribute('stroke',cc);g('contrib').setAttribute('stroke-width',(1+vn*3).toFixed(1));
 g('ring').setAttribute('stroke-width',(vn*6).toFixed(1));g('ring').setAttribute('stroke',cc);
 g('rT').textContent=D.t[f].toFixed(1);g('rVm').textContent=Math.round(vm);g('rIm').textContent=im.toFixed(2);g('rVe').textContent=ve.toFixed(2);}
let frame=0,playing=false,speed=0.5,last=0,raf=null;const FULL=18000;
function render(){draw(frame|0);scene(frame|0);g('scrub').value=frame|0;}
function loop(ts){if(!playing)return;if(!last)last=ts;const dt=ts-last;last=ts;frame+=dt/FULL*(N-1)*speed;
 if(frame>=N-1){frame=N-1;render();stop();return;}render();raf=requestAnimationFrame(loop);}
function play(){if(frame>=N-1)frame=0;playing=true;last=0;g('play').textContent='⏸';raf=requestAnimationFrame(loop);}
function stop(){playing=false;g('play').textContent='▶';if(raf)cancelAnimationFrame(raf);}
g('play').onclick=()=>playing?stop():play();
g('reset').onclick=()=>{stop();frame=0;render();};
g('scrub').oninput=e=>{stop();frame=+e.target.value;render();};
document.querySelectorAll('#spd button').forEach(b=>b.onclick=()=>{speed=+b.dataset.s;document.querySelectorAll('#spd button').forEach(x=>x.style.background='#fff');b.style.background='#e8e6dd';});
document.querySelector('#spd button[data-s="0.5"]').style.background='#e8e6dd';
window.addEventListener('resize',()=>{setup();render();});
render();
</script></body></html>
"""

html = HTML.replace("__DATA__", DJ).replace("__MAX__", str(len(data["t"]) - 1))
out = os.path.join(HERE, "e4b_player.html")
open(out, "w", encoding="utf-8").write(html)
print("saved:", out, "|", len(html), "bytes")
