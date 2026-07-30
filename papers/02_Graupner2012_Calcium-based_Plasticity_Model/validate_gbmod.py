# -*- coding: utf-8 -*-
"""
E8.1 검증: NMODL GBPlasticitySyn 의 칼슘 c(t)·효능 rho(t) 가
papers/02 Python 레퍼런스(calcium_trace + integrate_rho, noise=False)와 일치하는가.

동일 pre/post 스파이크 프로토콜을 NEURON mod 와 Python 오프라인 모델에 각각 넣고
c(t)·rho(t) 궤적을 대조. 결정론(σ=0)이라 두 결과가 수치적으로 일치해야 함.

실행(WSL, plain NEURON): mod 폴더(x86_64 포함)에서
  python <이 파일>            # 파라미터 = Wittenberg2006, rho0=0.5
"""
import os
import sys
import numpy as np
from neuron import h

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plasticity_model as PM

h.load_file("stdrun.hoc")

# ── 프로토콜: 50Hz pre-post 쌍(Δt=+10ms, 인과) 1초 → 칼슘 문턱 초과·rho 이동 유도 ──
pre_times = [20.0 + 20.0 * k for k in range(50)]          # 20,40,...,1000 ms
post_times = [t + 10.0 for t in pre_times]                 # 각 pre +10ms
TSTOP = 1100.0
DT = 0.025
RHO0 = 0.5                                                 # 불안정점 근처(이동 뚜렷)
SET = "hippo_slice_Wittenberg2006"
p = PM.PARAM_SETS[SET]

# ── NEURON mod 실행 ─────────────────────────────────────────────────────────
soma = h.Section(name="soma")
soma.L = soma.diam = 20.0
soma.insert("pas"); soma.g_pas = 1e-4; soma.e_pas = -65.0

syn = h.GBPlasticitySyn(soma(0.5))
# 파라미터를 Python 세트와 동일하게 주입
syn.tau_ca = p.tau_ca; syn.C_pre = p.C_pre; syn.C_post = p.C_post; syn.D = p.D
syn.theta_d = p.theta_d; syn.theta_p = p.theta_p
syn.gamma_d = p.gamma_d; syn.gamma_p = p.gamma_p
syn.tau = p.tau; syn.rho_star = p.rho_star; syn.b = p.b
syn.rho0 = RHO0

vpre = h.Vector(pre_times); vspre = h.VecStim(); vspre.play(vpre)
ncpre = h.NetCon(vspre, syn); ncpre.weight[0] = 1.0; ncpre.delay = 0.0    # 전달 강도(임의)·지연 0
vpost = h.Vector(post_times); vspost = h.VecStim(); vspost.play(vpost)
ncpost = h.NetCon(vspost, syn); ncpost.weight[0] = -1.0; ncpost.delay = 0.0  # weight<0 = POST sentinel

trec = h.Vector(); trec.record(h._ref_t)
crec = h.Vector(); crec.record(syn._ref_c)
rrec = h.Vector(); rrec.record(syn._ref_rho)

h.dt = DT; h.celsius = 34.0; h.finitialize(-65.0)
h.continuerun(TSTOP)

t = np.array(trec); c_mod = np.array(crec); rho_mod = np.array(rrec)

# ── Python 레퍼런스 (동일 시간격자) ─────────────────────────────────────────
c_py = PM.calcium_trace(t, pre_times, post_times, p)
rho_py = PM.integrate_rho(t, c_py, p, rho0=RHO0, noise=False)

# ── 대조 ────────────────────────────────────────────────────────────────────
dc = np.max(np.abs(c_mod - c_py))
drho = np.max(np.abs(rho_mod - rho_py))
print(f"[E8.1 검증] set={SET} rho0={RHO0} protocol=50Hz pre-post(dt+10) 1s")
print(f"  칼슘  c   : mod max={c_mod.max():.4f}  py max={c_py.max():.4f}  |최대차|={dc:.4e}")
print(f"  효능 rho  : mod final={rho_mod[-1]:.5f}  py final={rho_py[-1]:.5f}  |최대차|={drho:.4e}")
ok = (dc < 5e-2) and (drho < 2e-2)
print(f"  판정: {'PASS' if ok else 'FAIL'}  (기준 c<5e-2, rho<2e-2)")
