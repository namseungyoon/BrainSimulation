# -*- coding: utf-8 -*-
"""13_net_fepsp/stdp_verify.py  —  STDP 곡선 검증 (NEURON mod ↔ Python 참조 ↔ 문헌 현상)

우리 LTP 엔진(`GBPlasticitySyn.mod`, Graupner & Brunel 2012)이 **스파이크 타이밍 의존
가소성(STDP)** 현상을 재현하는지 면밀히 검증한다.

프로토콜(표준 STDP): pre 스파이크와 post 스파이크를 Δt = t_post − t_pre 만큼 벌려
N회 짝지어 반복 → 최종 효능 ρ → 시냅스 세기비 w_after/w_before.
  · Δt > 0 : pre가 먼저(인과적) → 고전 STDP는 LTP
  · Δt < 0 : post가 먼저(역인과) → 고전 STDP는 LTD

3중 검증:
  ① **NEURON mod** GBPlasticitySyn 실측 ρ
  ② **Python 참조** plasticity_model.integrate_rho (동일 파라미터·σ=0 결정론)
  ③ **문헌 현상**(Wittenberg & Wang 2006, 해마 CA3→CA1 슬라이스):
       - **단일** pre-post 짝 → 거의 **LTD 전용**(Δt 무관하게 LTP 안 남)
       - post **버스트(doublet)** → **양방향** STDP(Δt>0 LTP · Δt<0 LTD)
     ⇒ 같은 시냅스가 프로토콜에 따라 다른 STDP를 보이는 'malleability'가 핵심.

실행: <ca1sim>/python.exe 13_net_fepsp/stdp_verify.py            (기본: 단일+doublet)
      <ca1sim>/python.exe 13_net_fepsp/stdp_verify.py --npair 60 --freq 5
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BRAIN = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(BRAIN, "papers", "02_Graupner2012_Calcium-based_Plasticity_Model"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from neuron import h
h.load_file("stdrun.hoc")
import plasticity_model as PM

FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
SET = "hippo_slice_Wittenberg2006"          # 해마 슬라이스 피팅(우리 mod 기본값과 동일)
P = PM.PARAM_SETS[SET]
DT = 0.025


def argval(flag, d):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else d


NPAIR = int(argval("--npair", "60"))        # 짝 반복 횟수
FREQ = float(argval("--freq", "1.0"))       # 짝 반복 주기(Hz)
T0 = 100.0                                  # 첫 짝 시각(안정화 후)
RHO0 = float(argval("--rho0", "0.5"))       # 불안정점 근처(양방향 이동이 보이게)


def spike_times(dt_ms, mode):
    """Δt(ms)와 post 모드에 따라 pre/post 스파이크 시각 목록 생성.
    mode='single' : post 1발  ·  mode='doublet' : post 2발(10ms 간격, 50Hz 버스트 근사)"""
    ivl = 1000.0 / FREQ
    pre, post = [], []
    for i in range(NPAIR):
        tp = T0 + i * ivl
        pre.append(tp)
        if mode == "single":
            post.append(tp + dt_ms)
        else:                                # doublet: Δt 기준 첫 post + 10ms 뒤 두 번째
            post += [tp + dt_ms, tp + dt_ms + 10.0]
    return sorted(pre), sorted(post)


def run_mod(dt_ms, mode):
    """NEURON GBPlasticitySyn 실측: 최종 ρ."""
    soma = h.Section(name="s")
    soma.L = soma.diam = 20.0
    soma.insert("pas")
    syn = h.GBPlasticitySyn(soma(0.5))
    # 파라미터를 Python 세트와 동일하게 주입(mod 기본값과 같지만 명시)
    syn.tau_ca = P.tau_ca; syn.C_pre = P.C_pre; syn.C_post = P.C_post; syn.D = P.D
    syn.theta_d = P.theta_d; syn.theta_p = P.theta_p
    syn.gamma_d = P.gamma_d; syn.gamma_p = P.gamma_p
    syn.tau = P.tau; syn.rho_star = P.rho_star; syn.b = P.b
    syn.rho0 = RHO0
    pre, post = spike_times(dt_ms, mode)
    vpre = h.Vector(pre); vspre = h.VecStim(); vspre.play(vpre)
    ncpre = h.NetCon(vspre, syn); ncpre.weight[0] = 1.0; ncpre.delay = 0.0
    vpost = h.Vector(post); vspost = h.VecStim(); vspost.play(vpost)
    ncpost = h.NetCon(vspost, syn); ncpost.weight[0] = -1.0; ncpost.delay = 0.0   # weight<0 = POST
    tstop = max(pre[-1], post[-1]) + 200.0
    h.celsius = 34.0; h.cvode_active(0); h.dt = DT
    h.finitialize(-65.0); h.continuerun(tstop)
    return float(syn.rho), tstop, pre, post


def run_py(dt_ms, mode, tstop, pre, post):
    """Python 참조: 동일 스파이크열로 칼슘·ρ 적분(노이즈 없음)."""
    t = np.arange(0.0, tstop + DT, DT)
    c = PM.calcium_trace(t, np.array(pre), np.array(post), P)
    rho = PM.integrate_rho(t, c, P, rho0=RHO0, noise=False)
    return float(rho[-1])


def sweep(mode, dts):
    rows = []
    for d in dts:
        r_mod, tstop, pre, post = run_mod(d, mode)
        r_py = run_py(d, mode, tstop, pre, post)
        w_mod = PM.strength_change_ratio(RHO0, r_mod, P)
        w_py = PM.strength_change_ratio(RHO0, r_py, P)
        rows.append((d, r_mod, r_py, w_mod, w_py))
        print(f"  Δt={d:+7.1f}ms  ρ_mod={r_mod:.4f}  ρ_py={r_py:.4f}  |차|={abs(r_mod-r_py):.2e}  "
              f"세기비 mod={w_mod:.3f} py={w_py:.3f}", flush=True)
    return np.array(rows)


def main():
    dts = [float(x) for x in argval("--dts", "-100,-50,-30,-20,-10,-5,5,10,20,30,50,100").split(",")]
    print(f"[STDP 검증] set={SET} · ρ0={RHO0} · 짝 {NPAIR}회 @ {FREQ}Hz · Δt {len(dts)}점", flush=True)
    print(f"  파라미터: C_pre={P.C_pre} C_post={P.C_post} τ_ca={P.tau_ca}ms D={P.D}ms "
          f"θ_d={P.theta_d} θ_p={P.theta_p} γ_d={P.gamma_d} γ_p={P.gamma_p} τ={P.tau/1000:.1f}s b={P.b}", flush=True)
    out = {}
    for mode in ("single", "doublet"):
        print(f"\n=== [{mode}] post 스파이크 {'1발' if mode=='single' else '2발(doublet, 10ms)'} ===", flush=True)
        out[mode] = sweep(mode, dts)
    # 검증 판정
    print("\n=== 판정 ===", flush=True)
    for mode in ("single", "doublet"):
        R = out[mode]
        dmax = np.abs(R[:, 1] - R[:, 2]).max()
        pos = R[R[:, 0] > 0]; neg = R[R[:, 0] < 0]
        print(f"[{mode}] mod↔py 최대 |Δρ| = {dmax:.2e} ({'일치' if dmax < 1e-3 else '불일치'})", flush=True)
        print(f"        Δt>0 세기비 평균 {pos[:, 3].mean():.3f} · Δt<0 평균 {neg[:, 3].mean():.3f}", flush=True)
    np.savez(os.path.join(FIG, "_stdp_verify.npz"),
             dts=np.array(dts), single=out["single"], doublet=out["doublet"],
             npair=NPAIR, freq=FREQ, rho0=RHO0, pset=SET,
             params=np.array([P.C_pre, P.C_post, P.tau_ca, P.D, P.theta_d, P.theta_p,
                              P.gamma_d, P.gamma_p, P.tau, P.b]))
    print("saved: figures/_stdp_verify.npz", flush=True)


if __name__ == "__main__":
    main()
