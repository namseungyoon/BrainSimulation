"""
synapse_pair.py — in silico 단일 연결(pre->post) 빌더
============================================================================
Source: Ecker et al. (2020) §2.3-2.6, Fig.5.
보유 BBP stochastic TM 시냅스(ProbAMPANMDA_EMS / ProbGABAAB_EMS)를
대표 postsynaptic 구획에 배치하고 Table 3 파라미터를 주입한다.

핵심:
  - 흥분성 -> ProbAMPANMDA_EMS, 억제성 -> ProbGABAAB_EMS
  - Use/Dep/Fac/Nrrp + tau_r/d + NMDA_ratio 주입
  - setRNG(seed1, seed2, seed3) : Random123 확률 방출 스트림 초기화(시행별 시드)
  - presynaptic 스파이크열은 VecStim 으로 재생, NetCon weight = ĝ(nS)/1000 [µS]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.nrn_env import h, load_project_mechanisms  # noqa: E402

load_project_mechanisms()
h.load_file("stdrun.hoc")


def make_passive_post(e_pas: float = -65.0, name: str = "post") -> "h.Section":
    """수동(passive) 단일 구획 postsynaptic 세포 (정성적 STP 데모용).

    STP 프로파일(억압/촉진)은 시냅스 TM 파라미터로 결정되므로, 정성 재현에는
    단순 passive 구획이면 충분하다. 실제 형태/e-model 은 Phase 1/3 에서 사용.
    """
    sec = h.Section(name=name)
    sec.L = sec.diam = 20.0           # 단일 구획
    sec.cm = 1.0
    sec.insert("pas")
    sec.g_pas = 1.0 / 10000.0         # Rm ~ 10 kOhm*cm^2
    sec.e_pas = e_pas
    return sec


def build_synapse(seg, p: dict, seeds=(1, 1, 1), deterministic=False):
    """class 파라미터 dict(p)로 시냅스 1개 생성·설정.
    deterministic=False → 확률 EMS(ProbAMPANMDA_EMS/ProbGABAAB_EMS),
    deterministic=True  → 확률 미포함(DetAMPANMDA/DetGABAAB, Nrrp·setRNG 없음)."""
    if p["receptor"] == "AMPANMDA":
        syn = (h.DetAMPANMDA if deterministic else h.ProbAMPANMDA_EMS)(seg)
        syn.tau_r_AMPA = p.get("tau_r_AMPA", 0.2)
        syn.tau_d_AMPA = p["tau_d_AMPA"]
        syn.NMDA_ratio = p["NMDA_ratio"]
    else:
        syn = (h.DetGABAAB if deterministic else h.ProbGABAAB_EMS)(seg)
        syn.tau_r_GABAA = p.get("tau_r_GABAA", 0.2)
        syn.tau_d_GABAA = p["tau_d_GABAA"]
        # GABA_B 성분은 논문 가정(순수 GABA_A)에 따라 0 으로
        if hasattr(syn, "GABAB_ratio"):
            syn.GABAB_ratio = 0.0

    syn.Use = p["Use"]
    syn.Dep = p["Dep"]
    syn.Fac = p["Fac"]
    if not deterministic:
        syn.Nrrp = p["Nrrp"]
        syn.setRNG(int(seeds[0]), int(seeds[1]), int(seeds[2]))
    return syn


def make_netcon(syn, g_nS: float, spike_times):
    """VecStim presynaptic 소스 + NetCon.

    weight 는 nS 단위로 직접 넣는다 — EMS mod 가 PARAMETER `gmax=.001 (uS)`
    (nS→µS 변환 인자)로 곱하므로 g_AMPA_peak = gmax * weight = weight[nS].
    """
    vs = h.VecStim()
    tvec = h.Vector(list(spike_times))
    vs.play(tvec)
    nc = h.NetCon(vs, syn)
    nc.weight[0] = g_nS               # nS (mod 의 gmax 가 µS 로 변환)
    nc.delay = 0.0
    # VecStim/Vector 가 GC 되지 않도록 핸들 유지
    return nc, vs, tvec


def spike_train(n_pulses=8, freq_hz=20.0, t_start=100.0, recovery_delay=500.0):
    """Fig.5 프로토콜: n_pulses 자극열 + 마지막 펄스 뒤 recovery_delay 후 회복 스파이크 1발."""
    isi = 1000.0 / freq_hz
    train = [t_start + i * isi for i in range(n_pulses)]
    train.append(train[-1] + recovery_delay)   # recovery spike
    return train


def run_trial(class_params, seeds, n_pulses=8, freq_hz=20.0,
              t_start=100.0, recovery_delay=500.0, v_hold=-65.0,
              dt=0.025, pad=150.0, deterministic=False):
    """단일 시행 시뮬레이션. (t, v, spike_times) 반환.
    deterministic=True 면 확률 미포함(Det) 시냅스로 항상 같은 결과."""
    post = make_passive_post(e_pas=v_hold)
    spikes = spike_train(n_pulses, freq_hz, t_start, recovery_delay)
    syn = build_synapse(post(0.5), class_params, seeds=seeds, deterministic=deterministic)
    nc, vs, tvec = make_netcon(syn, class_params["g_nS"], spikes)

    t_vec = h.Vector().record(h._ref_t)
    v_vec = h.Vector().record(post(0.5)._ref_v)

    h.dt = dt
    h.celsius = 34.0                  # 논문 시뮬 온도
    h.finitialize(v_hold)
    h.continuerun(spikes[-1] + pad)

    import numpy as np
    return np.array(t_vec), np.array(v_vec), spikes
