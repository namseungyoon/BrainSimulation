# Source: Ecker(2020) §2.6 — in silico paired recording 공용 함수 (라이브러리 모듈)
# 번호 파일(6_calibrate_ghat)과 시각화 파일(animate/filmstrip)이 공유. 번호 파일은 서로 import 불가라 분리.
import os
import numpy as np

from common.nrn_env import h, MODELS_DIR
from common.cell_loader import load_cell

# paired recording 기본 상수
V_HOLD = -70.0       # 후시냅스 고정 전위 (mV)
E_REV = 0.0          # EMS mod 의 AMPA/NMDA reversal (e=0)
PSP_EXP = 0.5        # 실험 목표 PSP 진폭 (mV) — 대표값
N_SYN = 5            # 연결당 시냅스 수
T_SPIKE = 50.0       # presynaptic 발화 시각 (ms)
TSTOP = 90.0
DT = 0.1             # EMS mod 는 cvode 비호환 → 고정 dt (역치하라 0.1ms 충분)


def load_pc():
    """대표 PC 세포 1개 로드. (cell, template_name) 반환."""
    pc_root = os.path.join(MODELS_DIR, "pyramidal")
    pc_dir = os.path.join(pc_root, sorted(os.listdir(pc_root))[0])
    return load_cell(pc_dir)


def place_synapses(cell, p, n_syn=N_SYN):
    """PC 첨단수상돌기(apical)에 AMPA 시냅스 n개 배치."""
    apics = list(cell.apic)
    idxs = np.linspace(len(apics) * 0.2, len(apics) * 0.8, n_syn).astype(int)
    syns, ncs, keep = [], [], []
    for i in idxs:
        syn = h.ProbAMPANMDA_EMS(apics[int(i)](0.5))
        syn.tau_r_AMPA = p["tau_r_AMPA"]; syn.tau_d_AMPA = p["tau_d_AMPA"]
        syn.NMDA_ratio = p["NMDA_ratio"]
        syn.Use, syn.Dep, syn.Fac, syn.Nrrp = p["Use"], p["Dep"], p["Fac"], int(p["Nrrp"])
        vs = h.VecStim(); tv = h.Vector([T_SPIKE]); vs.play(tv)
        nc = h.NetCon(vs, syn); nc.delay = 0.0
        syns.append(syn); ncs.append(nc); keep += [vs, tv]
    return syns, ncs, keep


def measure_psp(syns, ncs, vsoma, tvec, g_nS, n_trials=6):
    """주어진 g_hat(nS)에서 소마 EPSP 평균(mV) + 마지막 trace 반환."""
    for nc in ncs:
        nc.weight[0] = g_nS
    psps = []
    for k in range(n_trials):
        for j, syn in enumerate(syns):
            syn.setRNG(7, k + 1, j + 1)
        h.finitialize(V_HOLD)
        h.continuerun(TSTOP)
        t, v = np.array(tvec), np.array(vsoma)
        i0 = np.searchsorted(t, T_SPIKE - 1.0)
        psps.append(v[i0:].max() - v[i0])
    return float(np.mean(psps)), (np.array(tvec).copy(), np.array(vsoma).copy())
