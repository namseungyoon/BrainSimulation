"""
3_dendritic_attenuation.py — 뉴런 검증: 수상돌기 EPSP 감쇠 (PC)
============================================================================
Source: Ecker(2020) §3.3; Magee & Cook (2000).

PC 첨단수상돌기(apical)의 여러 거리에 동일 EPSP(Exp2Syn)를 넣고,
**국소(시냅스 자리) vs 소마**에서 EPSP 진폭을 재서 "멀수록 소마 EPSP 가 작아진다(감쇠)"를 검증.
시냅스 PSP 는 소마에서 측정되므로 이 거리감쇠가 정확해야 한다.

실행:
    conda activate ca1sim
    python papers/01_Ecker2020_CA1_synaptic/02_neurons/3_dendritic_attenuation.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(THIS)))
SHARED = os.path.join(ROOT, "shared")
sys.path.insert(0, SHARED)
sys.path.insert(0, THIS)
from scipy.optimize import curve_fit       # noqa: E402
from common.nrn_env import h               # noqa: E402
from common.cell_loader import load_cell   # noqa: E402
from common.plotstyle import set_korean_font   # noqa: E402
import experimental_refs as REFS           # noqa: E402

set_korean_font()
h.load_file("stdrun.hoc")
OUT = os.path.join(THIS, "figures")

W = 0.0008      # Exp2Syn weight (µS) — EPSP 크기
T_SYN = 5.0
V_HOLD = -70.0


def main():
    os.makedirs(OUT, exist_ok=True)
    pyr = os.path.join(SHARED, "models", "pyramidal")
    pc_dir = os.path.join(pyr, sorted(os.listdir(pyr))[0])
    cell, tname = load_cell(pc_dir)
    soma = cell.soma[0]
    h.distance(0, soma(0.5))

    # 첨단수상돌기 구획을 소마 경로거리로 정렬 후 ~18개 균등 샘플
    apics = sorted([(h.distance(s(0.5)), s) for s in cell.apic], key=lambda x: x[0])
    apics = [(d, s) for d, s in apics if d > 5]
    idxs = np.linspace(0, len(apics) - 1, 18).astype(int)
    chosen = [apics[i] for i in idxs]

    vsoma = h.Vector().record(soma(0.5)._ref_v)
    tvec = h.Vector().record(h._ref_t)
    h.celsius = 34.0
    h.cvode_active(1)

    dists, loc_amp, soma_amp = [], [], []
    keep = []
    near_trace = far_trace = None
    for k, (dist, sec) in enumerate(chosen):
        syn = h.Exp2Syn(sec(0.5)); syn.tau1 = 0.5; syn.tau2 = 3.0; syn.e = 0.0
        ns = h.NetStim(); ns.number = 1; ns.start = T_SYN; ns.noise = 0
        nc = h.NetCon(ns, syn); nc.weight[0] = W; nc.delay = 0.0
        vloc = h.Vector().record(sec(0.5)._ref_v)
        h.finitialize(V_HOLD)
        h.continuerun(60.0)
        t = np.array(tvec); vs = np.array(vsoma); vl = np.array(vloc)
        i0 = np.searchsorted(t, T_SYN - 1)
        la = vl[i0:].max() - vl[i0]; sa = vs[i0:].max() - vs[i0]
        dists.append(dist); loc_amp.append(la); soma_amp.append(sa)
        if k == 0:
            near_trace = (t.copy(), vs.copy() - vs[i0], vl.copy() - vl[i0], dist)
        far_trace = (t.copy(), vs.copy() - vs[i0], vl.copy() - vl[i0], dist)
        nc.weight[0] = 0.0; ns.number = 0          # 다음 반복 위해 비활성
        keep += [syn, ns, nc, vloc]

    dists = np.array(dists); loc_amp = np.array(loc_amp); soma_amp = np.array(soma_amp)
    atten = soma_amp / np.maximum(loc_amp, 1e-9)

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16.5, 4.9))
    fig.suptitle(f"뉴런 검증 — 수상돌기 EPSP 감쇠 (PC: {tname})", fontsize=13, fontweight="bold")

    # (A) 예시 트레이스: 가까운 vs 먼 시냅스의 국소·소마 EPSP
    for tr, lab, c in [(near_trace, f"가까운 ({near_trace[3]:.0f}µm)", "tab:green"),
                       (far_trace, f"먼 ({far_trace[3]:.0f}µm)", "tab:red")]:
        t, vs, vl, _ = tr
        axA.plot(t, vl, color=c, lw=1.8, label=f"{lab} 국소")
        axA.plot(t, vs, color=c, lw=1.4, ls="--", label=f"{lab} 소마")
    axA.set_xlim(0, 40); axA.set_title("(A) 국소(실선) vs 소마(점선) EPSP", fontsize=10)
    axA.set_xlabel("시간 t (ms)"); axA.set_ylabel("EPSP (mV)"); axA.legend(fontsize=7)

    # (B) 거리별 진폭
    axB.plot(dists, loc_amp, "o-", color="tab:purple", lw=1.8, ms=4, label="국소 EPSP")
    axB.plot(dists, soma_amp, "s-", color="tab:blue", lw=1.8, ms=4, label="소마 EPSP")
    axB.set_title("(B) 거리별 EPSP 진폭", fontsize=10)
    axB.set_xlabel("시냅스 거리 (µm, 소마 기준)"); axB.set_ylabel("EPSP 진폭 (mV)"); axB.legend(fontsize=8)

    # (C) 감쇠비 + 지수 적합 → 공간상수 λ
    def expdecay(d, lam, A):
        return A * np.exp(-d / lam)
    lam, A = None, 1.0
    try:
        popt, _ = curve_fit(expdecay, dists, atten, p0=[100.0, 1.0],
                            bounds=([10, 0.2], [1000, 1.5]), maxfev=10000)
        lam, A = float(popt[0]), float(popt[1])
    except Exception as e:
        print(f"[경고] 지수 적합 실패: {e}")

    axC.plot(dists, atten, "o", color="tab:red", ms=5, label="모델(소마/국소)")
    if lam is not None:
        dd = np.linspace(dists.min(), dists.max(), 100)
        axC.plot(dd, expdecay(dd, lam, A), "-", color="k", lw=1.8,
                 label=f"지수적합 λ={lam:.0f}µm")
    rng = REFS.ATTENUATION["psp_lambda_um"]["approx_range"]
    axC.axvspan(rng[0], rng[1], color="tab:gray", alpha=0.12,
                label=f"실험 λ 근사 {rng[0]}–{rng[1]}µm")
    axC.axhline(np.exp(-1), color="0.5", ls=":", lw=1)
    axC.text(dists.max() * 0.62, np.exp(-1) + 0.02, "1/e (λ 지점)", fontsize=7, color="0.4")
    axC.set_title("(C) 감쇠비 = 소마/국소 + 공간상수 λ", fontsize=10)
    axC.set_xlabel("시냅스 거리 (µm)"); axC.set_ylabel("소마/국소 비"); axC.set_ylim(0, 1.05)
    axC.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUT, "3_dendritic_attenuation.png")
    plt.savefig(out, dpi=120)
    print(f"[그림] {out}")
    print(f"[검증] 소마 EPSP: 근위 {soma_amp[0]:.2f}mV → 원위 {soma_amp[-1]:.2f}mV, "
          f"감쇠비 {atten[0]:.2f}→{atten[-1]:.2f} (거리↑ 감쇠↑ = 정상)")
    if lam is not None:
        print(f"[공간상수] PSP 감쇠 λ = {lam:.0f} µm (실험 근사 {rng[0]}–{rng[1]}µm; "
              f"논문 BPAP λ 모델 {REFS.ATTENUATION['bpap_lambda_um']['model_paper']}µm)")


if __name__ == "__main__":
    main()
