# -*- coding: utf-8 -*-
"""12_lfp/e4a_fepsp.py  —  E4a: 상세형태 대표 PC의 SC 자극 세포외 fEPSP 계산

sc_epsp_test.py 보일러플레이트 계승:
  대표 추체세포 1개 로드 -> apical(SR)에 SC 시냅스(Ecker PC->PC E2 대용, 결정론) ->
  단일 볼리 / paired-pulse 자극 -> 전 세그먼트 총 막전류 i_membrane_(nA) 기록 ->
  자체 LSA 전달행렬로 세포외 전위 V(mV) 계산.

전극:
  (a) SR 단일전극  : 근위 SR 시냅스에서 깊이축 수직 ~50um
  (b) 깊이 프로파일: 소마->원위 apical 깊이축을 따라 24점(SO<-SP->SR->SLM), 측면 40um offset
프로토콜:
  (P1) 단일 볼리 t=50ms       -> SR 파형 + 깊이 프로파일(극성반전)
  (P2) paired-pulse IPI=50ms  -> PPR(slope2/slope1)

무한 균질 매질(sigma=0.3 S/m). MEA 슬라이스 3층 영상법은 E4b.
실행: <ca1sim>/python.exe 12_lfp/e4a_fepsp.py
"""
import os
import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BRAIN = os.path.dirname(ROOT)
SHARED = os.path.join(BRAIN, "shared")
PAPER = os.path.join(BRAIN, "papers", "01_Ecker2020_CA1_synaptic")
sys.path.insert(0, SHARED)
sys.path.insert(0, os.path.join(PAPER, "03_synapses"))
sys.path.insert(0, os.path.join(PAPER, "04_network"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "13_net_fepsp"))

from mea_postproc import measure_fepsp as _measure_fepsp, SLOPE_METHOD
from common.nrn_env import h
from common.cell_loader import load_cell
import network_lib as net
import params_table3 as P3
from synapse_pair import build_synapse
import lfp_calc as L

MODELS = os.path.join(SHARED, "models")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
SIGMA = 0.3          # S/m 무한 균질 매질
STIM_T = 50.0        # 첫 자극 시각 ms
IPI = 50.0           # paired-pulse 자극간격 ms
TSTOP = 150.0
NC_DELAY = 1.0
N_SYN = 40           # SR 분산 SC 시냅스 수(축소모델 ~60/PC 반영, 동기 볼리)
SR_BAND = (0.30, 0.68)   # SR 대역(중~원위): sink를 소마서 분리 -> 명확한 dipole


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def perp_dir(axis):
    """axis 에 수직인 단위벡터 하나."""
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(unit(axis), ref)) > 0.95:
        ref = np.array([1.0, 0.0, 0.0])
    return unit(np.cross(axis, ref))


def measure_fepsp(t, v, t0, dur=30.0):
    """공용 자(`13_net_fepsp/mea_postproc.measure_fepsp`)를 그대로 쓴다 — 얇은 감싸개.

    ★예전에는 이 파일이 같은 함수를 **따로 복사해** 두고 있었다. 기울기 정의를
      교차시각 방식으로 바꾼 뒤에도 이 사본만 옛 표본회귀로 남아, E4a의 slope·PPR이
      MEA 쪽과 다른 자로 재어지고 있었다. 사본을 지우고 한 곳으로 모은다.
    pre=0.0 — 기준선을 창 첫 표본으로 잡던 옛 동작을 그대로 유지한다(짝펄스 2발째의
      기준선 창이 1발째 감쇠구간에 걸치는 것을 피하려면 이쪽이 안전하다).
    단위는 mV(자체 LSA 출력) — 함수는 단위를 가리지 않는다.
    """
    return _measure_fepsp(t, v, t0, dur, 0.0)


def run_protocol(imem_vecs, tvec, number):
    """number 발(interval=IPI) 자극 후 imem 행렬(N_seg, N_t)·t 반환."""
    for ns in _NETSTIMS:
        ns.number = number
        ns.interval = IPI
        ns.start = STIM_T
        ns.noise = 0
    h.celsius = 34.0
    h.cvode_active(0)
    h.dt = 0.025
    h.finitialize(-70.0)
    h.continuerun(TSTOP)
    t = np.array(tvec)
    I = np.array([np.array(v) for v in imem_vecs])   # (N_seg, N_t) nA
    return t, I


_NETSTIMS = []


def main():
    # --- 대표 PC 로드 ---
    type_dir = net.load_representatives(MODELS)
    cell, tname = load_cell(type_dir["PC"], gid=0)
    h.define_shape()   # 3D 점 없는 섹션(축삭 stub/myelin)에 좌표 부여 -> (0,0,0) 이상점 제거
    print(f"[세포] 대표 추체 {tname}", flush=True)

    seclist = list(cell.all)
    geom = L.collect_segments(seclist)
    N = geom["mid"].shape[0]
    print(f"[기하] 세그먼트 {N}개 (섹션 {len(seclist)})", flush=True)

    # --- 깊이축: 소마 -> 원위 apical ---
    soma = cell.soma[0]
    soma_c = L.seg_point(soma, 0.5)
    h.distance(0, soma(0.5))
    apic = [s for s in seclist if ".apic" in s.name()]
    apic_d = sorted(((h.distance(s(0.5)), s) for s in apic), reverse=True)
    top = apic_d[:max(1, len(apic_d) // 10)]
    distal_c = np.mean([L.seg_point(s, 0.5) for _, s in top], axis=0)
    depth_axis = unit(distal_c - soma_c)
    lateral = perp_dir(depth_axis)
    dmax_path = apic_d[0][0]
    print(f"[깊이축] soma{np.round(soma_c,0)} -> apical(경로 dmax {dmax_path:.0f}um), 축 {np.round(depth_axis,2)}", flush=True)

    # --- SC 시냅스: SR 대역에 N_SYN개 분산(경로거리 균등), 동기 볼리 ---
    p = P3.CLASSES["PC->PC (E2)"]
    lo, hi = SR_BAND[0] * dmax_path, SR_BAND[1] * dmax_path
    sr_secs = [s for s in apic if lo <= h.distance(s(0.5)) <= hi]
    if len(sr_secs) < N_SYN:
        sr_secs = sorted(apic, key=lambda s: abs(h.distance(s(0.5)) - 0.35 * dmax_path))[:N_SYN]
    # 경로거리 균등 표집
    sr_secs = sorted(sr_secs, key=lambda s: h.distance(s(0.5)))
    idx = np.linspace(0, len(sr_secs) - 1, N_SYN).round().astype(int)
    chosen = [sr_secs[i] for i in idx]
    ns = h.NetStim()
    _NETSTIMS.append(ns)
    syns, syn_pos_list = [], []
    for s in chosen:
        syn = build_synapse(s(0.5), p, seeds=(1, 1, 1), deterministic=True)
        nc = h.NetCon(ns, syn)
        nc.weight[0] = p["g_nS"]
        nc.delay = NC_DELAY
        syns.append((syn, nc))
        syn_pos_list.append(L.seg_point(s, 0.5))
    syn_pos = np.array(syn_pos_list)
    syn_c = syn_pos.mean(axis=0)
    syn_depths = (syn_pos - soma_c) @ depth_axis
    print(f"[시냅스] SR 분산 {len(chosen)}개 (경로 {h.distance(chosen[0](0.5)):.0f}~{h.distance(chosen[-1](0.5)):.0f}um) "
          f"깊이 {syn_depths.min():.0f}~{syn_depths.max():.0f}um", flush=True)
    print(f"[구동] Ecker PC->PC(E2) g={p['g_nS']}nS x{N_SYN} NMDA비={p['NMDA_ratio']} 결정론 동기볼리", flush=True)

    # --- 전극 배치 ---
    elec_SR = syn_c + 50.0 * lateral                         # (a) SR 단일전극(시냅스 무게중심 옆)
    depths = np.linspace(-250.0, 700.0, 28)                  # (b) 깊이 프로파일(SO~SLM)
    elec_profile = np.array([soma_c + d * depth_axis + 40.0 * lateral for d in depths])
    syn_depth = float(syn_depths.mean())                     # 시냅스 평균 깊이좌표
    print(f"[전극] SR 단일 {np.round(elec_SR,0)} | 깊이프로파일 {len(depths)}점 [-250,700]um (시냅스 평균깊이 {syn_depth:.0f}um)", flush=True)

    M_SR = L.lsa_matrix(geom, [elec_SR], SIGMA)              # (1, N)
    M_prof = L.lsa_matrix(geom, elec_profile, SIGMA)         # (24, N)
    M_SR_psa = L.psa_matrix(geom, [elec_SR], SIGMA)          # 비교용

    # --- 막전류 기록 세팅 ---
    imem_vecs, cv = L.setup_imem(geom["segs"])
    tvec = h.Vector().record(h._ref_t)
    vsoma = h.Vector().record(soma(0.5)._ref_v)

    # --- P1 단일 볼리 ---
    t1, I1 = run_protocol(imem_vecs, tvec, number=1)
    cons_max, i_max = L.current_conservation(I1)
    print(f"[전류보존] max|sumI|={cons_max:.3e} nA, max|I|={i_max:.3e} nA, 비율={cons_max/max(i_max,1e-12):.2e}", flush=True)
    V_SR1 = L.compute_lfp(M_SR, I1)[0]                       # (N_t,) mV
    V_SR1_psa = L.compute_lfp(M_SR_psa, I1)[0]
    V_prof1 = L.compute_lfp(M_prof, I1)                      # (28, N_t) mV
    v_soma1 = np.array(vsoma)
    vpk = v_soma1[(t1 >= STIM_T) & (t1 < STIM_T + 30)].max()
    spiked = vpk > -20.0
    print(f"[소마] 자극후 최대 Vm {vpk:.1f}mV -> {'스파이크(역치초과)' if spiked else '역치하(순수 시냅스 fEPSP)'}", flush=True)

    f_SR = measure_fepsp(t1, V_SR1, STIM_T + NC_DELAY)
    print(f"[P1 SR fEPSP] 음성피크 {f_SR['amp']*1e3:.2f} uV, slope {f_SR['slope']*1e3:.3f} uV/ms, t_pk {f_SR['tpk']:.1f}ms", flush=True)

    # 깊이 프로파일 극성(각 전극 음성피크 진폭)
    prof_amp = np.array([measure_fepsp(t1, V_prof1[j], STIM_T + NC_DELAY)["amp"] for j in range(len(depths))])
    # 각 전극의 자극후 극값(양/음 모두) — 극성반전 확인용
    prof_ext = np.array([V_prof1[j][(t1 >= STIM_T) & (t1 < STIM_T + 30)][
        np.argmax(np.abs(V_prof1[j][(t1 >= STIM_T) & (t1 < STIM_T + 30)]))] for j in range(len(depths))])
    print(f"[P1 깊이극성] SR쪽(sink) 최음 {prof_ext.min()*1e3:.2f}uV @깊이{depths[np.argmin(prof_ext)]:.0f} | "
          f"SP/SO쪽(source) 최양 {prof_ext.max()*1e3:.2f}uV @깊이{depths[np.argmax(prof_ext)]:.0f}", flush=True)
    # 극성반전 깊이(모든 부호변화 = 다극성 구조 판정)
    sign = np.sign(prof_ext)
    cross = [0.5 * (depths[k-1] + depths[k]) for k in range(1, len(depths))
             if sign[k] != sign[k-1] and sign[k-1] != 0]
    npole = "삼중극(source-sink-source)" if len(cross) >= 2 else \
            ("이중극(dipole)" if len(cross) == 1 else "단일극")
    print(f"[극성] {npole} 반전 {['%.0f' % c for c in cross]}um (소마 0, 시냅스 평균 {syn_depth:.0f}um)", flush=True)
    revs = cross

    # --- P2 paired-pulse ---
    t2, I2 = run_protocol(imem_vecs, tvec, number=2)
    V_SR2 = L.compute_lfp(M_SR, I2)[0]
    f1 = measure_fepsp(t2, V_SR2, STIM_T + NC_DELAY)
    f2 = measure_fepsp(t2, V_SR2, STIM_T + IPI + NC_DELAY)
    ppr_slope = abs(f2["slope"]) / max(abs(f1["slope"]), 1e-12)
    ppr_amp = abs(f2["amp"]) / max(abs(f1["amp"]), 1e-12)
    print(f"[P2 paired IPI={IPI:.0f}ms] E1 slope {f1['slope']*1e3:.3f} amp {f1['amp']*1e3:.2f}uV | "
          f"E2 slope {f2['slope']*1e3:.3f} amp {f2['amp']*1e3:.2f}uV", flush=True)
    print(f"[PPR] slope비 {ppr_slope:.3f}, 진폭비 {ppr_amp:.3f} ({'facilitation' if ppr_slope>1 else 'depression'})", flush=True)

    # --- 저장 ---
    out = os.path.join(FIG, "_e4a_results.npz")
    np.savez(out,
             t1=t1, V_SR1=V_SR1, V_SR1_psa=V_SR1_psa, V_prof1=V_prof1, v_soma1=v_soma1,
             t2=t2, V_SR2=V_SR2, depths=depths, prof_amp=prof_amp, prof_ext=prof_ext,
             revs=np.array(revs, float),
             syn_pos=syn_pos, syn_depth=syn_depth, soma_c=soma_c, distal_c=distal_c,
             depth_axis=depth_axis, lateral=lateral, elec_SR=elec_SR, elec_profile=elec_profile,
             seg_mid=geom["mid"], seg_p0=geom["p0"], seg_p1=geom["p1"],
             stim_t=STIM_T, ipi=IPI, nc_delay=NC_DELAY, sigma=SIGMA,
             fSR_amp=f_SR["amp"], fSR_slope=f_SR["slope"], fSR_tpk=f_SR["tpk"],
             ppr_slope=ppr_slope, ppr_amp=ppr_amp,
             e1_slope=f1["slope"], e2_slope=f2["slope"], e1_amp=f1["amp"], e2_amp=f2["amp"],
             slope_method=SLOPE_METHOD,
             fSR_slope_legacy=f_SR["slope_legacy"], e1_slope_legacy=f1["slope_legacy"],
             e2_slope_legacy=f2["slope_legacy"], fSR_n_band=f_SR["n_band"],
             cons_ratio=cons_max / max(i_max, 1e-12), tname=str(tname), n_seg=N,
             g_nS=p["g_nS"], n_syn=N_SYN, vpk=vpk, spiked=spiked)
    print(f"[저장] {out}", flush=True)
    return out


if __name__ == "__main__":
    main()
