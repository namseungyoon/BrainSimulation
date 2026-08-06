# -*- coding: utf-8 -*-
"""13_net_fepsp/_chunk_verify.py — 시간 청크 누적의 수치 동일성 검증 (전규모 실행의 전제조건)

검증 질문: **record 중인 Vector에 시뮬레이션 도중 resize(0)을 호출하면 이후 기록이 정상적으로
이어지는가?** 이어진다면 청크마다 M@I를 계산·누적하고 버퍼를 비워 메모리를 상수로 유지할 수 있다.

판정 기준(셋 다 통과해야 전규모 실행 허용):
  A. 시간축 동일   — 청크 이어붙인 t == 통째 t (원소별)
  B. 막전류 동일   — 청크 이어붙인 i_membrane_ == 통째 i_membrane_ (원소별, rtol=0)
  C. 전극전위 동일 — (M @ I) 를 청크별로 계산해 이어붙인 것 == 통째 계산 (rtol=1e-12)

C가 핵심이다. 실제 코드는 청크마다 M@I를 곱해 누적하므로, 행렬곱의 시간축 분할이
이어붙이기와 같음을 확인한다(분배법칙 — 수학적으로는 자명하지만 부동소수 순서까지 확인).

실행: <ca1sim>/python.exe 13_net_fepsp/_chunk_verify.py
     (MPI 불필요 — 단일 랭크로 API 동작만 확인. 랭크별 M@I는 독립이므로 결론이 그대로 적용됨)
"""
import sys
import numpy as np
from neuron import h

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

h.load_file("stdrun.hoc")
DT = 0.025
REC_DT = 0.4
TSTOP = 100.0
CHUNK = 50.0

# ── 최소 모델: 소마+수상돌기(세그먼트 여러 개 = 실제와 같은 조건) + 시냅스 자극 ──
soma = h.Section(name="soma")
soma.L = soma.diam = 20.0
dend = h.Section(name="dend")
dend.L, dend.diam, dend.nseg = 200.0, 2.0, 9
dend.connect(soma(1))
for sec in (soma, dend):
    sec.insert("hh")
    sec.Ra = 150.0

syn = h.Exp2Syn(dend(0.5))
syn.tau1, syn.tau2, syn.e = 0.2, 3.0, 0.0
ns = h.NetStim()
ns.number, ns.start, ns.interval = 4, 10.0, 20.0     # 청크 경계(50ms)를 걸치도록 배치
nc = h.NetCon(ns, syn)
nc.weight[0] = 0.004
nc.delay = 1.0

cv = h.CVode()
cv.use_fast_imem(1)
h.cvode_active(0)
h.dt = DT
h.celsius = 34.0

segs = [seg for sec in (soma, dend) for seg in sec]
NSEG = len(segs)
rng = np.random.default_rng(0)
M = rng.normal(size=(24, NSEG)) * 1e-3               # 전달행렬 대역(실제 MoI 행렬 대신 임의 — 동일성만 검증)

vt = h.Vector(); vt.record(h._ref_t, REC_DT)
vecs = []
for seg in segs:
    v = h.Vector(); v.record(seg._ref_i_membrane_, REC_DT)
    vecs.append(v)

pc = h.ParallelContext()
pc.set_maxstep(10)


def snap():
    return np.array([np.asarray(v) for v in vecs])


# ── 실행 1: 통째 ──
h.finitialize(-65.0)
pc.psolve(TSTOP)
t_whole = np.asarray(vt).copy()
I_whole = snap()
Ve_whole = (M @ I_whole) * 1e3
print(f"[통째]  t {t_whole.shape} {t_whole[0]:.4f}~{t_whole[-1]:.4f} · I {I_whole.shape} · Ve {Ve_whole.shape}")

# ── 실행 2: 청크(도중 resize(0)) ──
vt.resize(0)
for v in vecs:
    v.resize(0)
h.finitialize(-65.0)

t_parts, Ve_parts, I_parts, bounds = [], [], [], []
t_next = 0.0
while t_next < TSTOP - 1e-9:
    t_next = min(t_next + CHUNK, TSTOP)
    pc.psolve(t_next)
    tt = np.asarray(vt).copy()
    Ic = snap()
    t_parts.append(tt)
    I_parts.append(Ic)
    Ve_parts.append((M @ Ic) * 1e3)
    bounds.append((float(tt[0]) if tt.size else np.nan, float(tt[-1]) if tt.size else np.nan, tt.size))
    vt.resize(0)
    for v in vecs:
        v.resize(0)

t_chunk = np.concatenate(t_parts)
I_chunk = np.concatenate(I_parts, axis=1)
Ve_chunk = np.concatenate(Ve_parts, axis=1)
print(f"[청크]  {len(t_parts)}조각 · t {t_chunk.shape} · I {I_chunk.shape} · Ve {Ve_chunk.shape}")
for k, (a, b, n) in enumerate(bounds):
    print(f"        조각{k+1}: t {a:.4f}~{b:.4f} ({n}점)")

# ── 판정 ──
print()
ok = True

if t_chunk.shape != t_whole.shape:
    print(f"❌ A 시간축 길이 불일치: 통째 {t_whole.shape[0]} vs 청크 {t_chunk.shape[0]}")
    n = min(t_chunk.size, t_whole.size)
    d = np.nonzero(t_chunk[:n] != t_whole[:n])[0]
    print(f"   첫 불일치 인덱스: {d[0] if d.size else '(길이만 다름)'}")
    print(f"   통째 앞 8점: {np.round(t_whole[:8], 4)}")
    print(f"   청크 앞 8점: {np.round(t_chunk[:8], 4)}")
    ok = False
else:
    dmax = float(np.max(np.abs(t_chunk - t_whole)))
    print(f"{'✅' if dmax == 0 else '❌'} A 시간축: 최대차 {dmax:.3e}  (길이 {t_whole.size})")
    ok &= dmax == 0

if I_chunk.shape == I_whole.shape:
    dmax = float(np.max(np.abs(I_chunk - I_whole)))
    print(f"{'✅' if dmax == 0 else '❌'} B 막전류: 최대차 {dmax:.3e}  (완전 동일해야 함)")
    ok &= dmax == 0
else:
    print(f"❌ B 막전류 shape 불일치: {I_whole.shape} vs {I_chunk.shape}")
    ok = False

if Ve_chunk.shape == Ve_whole.shape:
    dmax = float(np.max(np.abs(Ve_chunk - Ve_whole)))
    rel = dmax / max(float(np.max(np.abs(Ve_whole))), 1e-30)
    print(f"{'✅' if rel < 1e-12 else '❌'} C 전극전위: 최대차 {dmax:.3e} (상대 {rel:.3e}) · 진폭 {np.max(np.abs(Ve_whole)):.3f}")
    ok &= rel < 1e-12
else:
    print(f"❌ C 전극전위 shape 불일치: {Ve_whole.shape} vs {Ve_chunk.shape}")
    ok = False

print()
print("=" * 70)
if ok:
    print("✅ 청크 누적이 통째 계산과 수치적으로 동일 → 전규모 실행 허용")
else:
    print("❌ 불일치 → 전규모 실행 금지. resize(0) 대안(record 재등록) 필요")
print("=" * 70)
sys.exit(0 if ok else 1)
