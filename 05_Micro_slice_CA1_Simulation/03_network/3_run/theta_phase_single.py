# -*- coding: utf-8 -*-
"""
03_network/3_run/theta_phase_single.py
  나-2-② 프로토콜 ③ — θ 위상 기반 타이밍 장기가소성 (단일 시냅스 티어)

  문헌(그대로 재현):
   · Huerta & Lisman 1995 (Neuron 15:1053) — in vitro, CCh 50µM로 θ(7-9Hz) 유도,
     θ 중 단일 버스트(4 shocks @100Hz)를 peak→LTP / trough→LTD.
     "자극량이 아니라 위상이 방향을 결정."
   · Hyman et al. 2003 (J Neurosci 23:11725) — in vivo 행동 중, 국소 θ 위상 트리거,
     peak 자극 → +17.9±0.94% (LTP), trough 자극 → -12.9±1.03% (LTD).

  단일 시냅스 재현(결정론 .mod GBPlasticityStpProbSyn):
   - SP_PC 1개 세포 · SR(str.radiatum) 수상돌기에 SC→PC 시냅스 1개
   - θ 배경 = 소마에 8Hz 정현파 전류 주입 → 막전위 peak/trough 정의
   - conditioning = 4펄스 100Hz 버스트를 θ 위상 φ에 정렬해 pre 인가
       · --ncyc 1  : 단일 버스트 (Huerta & Lisman)
       · --ncyc N  : θ 주기마다 버스트 반복 N회 (Hyman tetanic 근사)
   - 측정 = 버스트 전/후 시냅스 효능 ρ 변화(Δρ). peak Δρ>0 / trough Δρ<0 재현 여부.
   - 위상 스윕(--sweep) → 위상-Δρ 곡선(양방향 스위치)
   - 3D UI용: t·Vm·칼슘 c·ρ·g·θ전류·자극시점 전 시계열 npz 저장(용량 무관)

  실행(ca1sim + 컴파일된 mechanism):
    CA1_MECH_LIB=$PWD/scratch/mechbuild/x86_64/libnrnmech.so \
      python 03_network/3_run/theta_phase_single.py --smoke        # peak vs trough 빠른 확인
    python 03_network/3_run/theta_phase_single.py --sweep --ncyc 1  # 위상 스윕(H&L)
    python 03_network/3_run/theta_phase_single.py --peak-trough --ncyc 20  # Hyman 근사
"""
import os, sys, json, time
import numpy as np
from neuron import h

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "lib"))
SCR = os.path.join(ROOT, "scratch")
OUTD = os.path.join(ROOT, "04_experiments", "Ex_theta_phase", "data")
os.makedirs(OUTD, exist_ok=True)


def arg(flag, default, cast=float):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


# ── 파라미터 ──
GID = int(arg("--gid", -1))                      # -1이면 첫 SP_PC 자동
THETA_HZ = arg("--thetaHz", 8.0)                 # H&L 7-9Hz
A_THETA = arg("--A", 0.35)                        # θ 정현파 진폭 (nA) — 소마
DC = arg("--DC", 0.0)                             # θ DC 오프셋 (nA)
BAMP = arg("--Bamp", 0.6)                          # 수렴 볼리 프록시 진폭 (nA) — 버스트 펄스마다 소마
BDUR = arg("--Bdur", 3.0)                          # 볼리 펄스 폭 (ms)
NPULSE = int(arg("--npulse", 4))                  # 버스트 펄스 수 (4 shocks)
BURST_HZ = arg("--burstHz", 100.0)                # 버스트 내 빈도 (100 Hz)
NCYC = int(arg("--ncyc", 1))                      # 버스트 반복 횟수 (1=H&L 단일)
SETTLE = arg("--settle", 300.0)
OBS = arg("--obs", 200.0)                          # 마지막 버스트 후 관측(칼슘 소멸+ρ 안정)
SMOKE = "--smoke" in sys.argv
SWEEP = "--sweep" in sys.argv
PEAK_TROUGH = "--peak-trough" in sys.argv
REC_DT = arg("--recdt", 0.5)                       # 시계열 기록 간격(ms) — 3D UI용

# SC→PC STP (mpi_plasticity.sc_stp SP_PC와 동일)
SC_U, SC_D, SC_F, SC_NRRP = 0.14, 186.0, 129.0, 12
SC_GMAX = 0.8 / 1000.0

TPER = 1000.0 / THETA_HZ                           # θ 주기(ms)


def pick_pc_gid(B):
    if GID >= 0:
        return GID
    idx = np.where(B.mt == "SP_PC")[0]
    return int(idx[0])


def pick_sr_dend(cell):
    """소마 경로거리 ~120-250µm apical 수상돌기 중점 → str.radiatum SC 표적."""
    soma = cell.soma[0]
    h.distance(0, soma(0.5))
    best = None
    for sec in cell.all:
        nm = sec.name()
        if "apic" not in nm and "dend" not in nm:
            continue
        if "axon" in nm or "soma" in nm:
            continue
        d = h.distance(sec(0.5))
        if 120.0 <= d <= 250.0:
            best = (sec, 0.5, d)
            if "apic" in nm:                       # apical 우선
                return best
    if best:
        return best
    # 폴백: 가장 먼 apical
    far = None
    for sec in cell.all:
        if "apic" in sec.name():
            d = h.distance(sec(0.5))
            if far is None or d > far[2]:
                far = (sec, 0.5, d)
    return far


def build(B, gid):
    cell = B.build_cell(gid)
    sec, x, dist = pick_sr_dend(cell)
    syn = h.GBPlasticityStpProbSyn(sec(x))
    syn.Use = SC_U; syn.Dep = SC_D; syn.Fac = SC_F; syn.Nrrp = SC_NRRP
    syn.gmax = SC_GMAX
    syn.ca_stp = 0; syn.rho0 = 0.5                 # 가소성 ON, gamma는 .mod 기본(1645·313)
    syn.setRNG(gid + 1, 1, 3)
    return cell, syn, sec, x, dist


def theta_vectors(tstop, t_peak0):
    """θ 정현파 amp 벡터: peak(최대 탈분극)이 t_peak0 + kT 에 오도록 cos 정렬."""
    tt = np.arange(0.0, tstop + REC_DT, REC_DT)
    amp = DC + A_THETA * np.cos(2 * np.pi * (tt - t_peak0) / TPER)
    return tt, amp


def burst_times(t_center):
    """버스트 중심 t_center에 4펄스 100Hz(폭 (NPULSE-1)*ISI) 정렬."""
    isi = 1000.0 / BURST_HZ
    span = (NPULSE - 1) * isi
    t0 = t_center - span / 2.0
    return [t0 + k * isi for k in range(NPULSE)]


def run_one(B, gid, phase_deg, label):
    """위상 φ(deg)에서 conditioning 실행. peak=0°, trough=180°."""
    cell, syn, sec, x, dist = build(B, gid)
    soma = cell.soma[0]

    # θ 첫 peak 기준시각: settle 이후 첫 주기 peak
    t_peak0 = SETTLE + TPER
    tstop = t_peak0 + NCYC * TPER + OBS

    # θ 전류 주입 (소마)
    ic = h.IClamp(soma(0.5)); ic.delay = 0; ic.dur = 1e9
    ttv, ampv = theta_vectors(tstop, t_peak0)
    tvec = h.Vector(ttv); avec = h.Vector(ampv)
    avec.play(ic._ref_amp, tvec, True)

    # 버스트(들): θ 주기마다 중심을 (t_peak0 + k*T + φ/360*T)에 정렬
    off = phase_deg / 360.0 * TPER
    all_spk = []
    for k in range(NCYC):
        c = t_peak0 + k * TPER + off
        all_spk += burst_times(c)
    all_spk = sorted(all_spk)

    # 수렴 볼리 프록시: 버스트 펄스마다 소마 전류 펄스(Bamp, Bdur). θ peak면 발화, trough면 무발화.
    ic_v = h.IClamp(soma(0.5)); ic_v.delay = 0; ic_v.dur = 1e9
    vt = np.arange(0.0, tstop + REC_DT, REC_DT); vv = np.zeros_like(vt)
    for ts in all_spk:
        vv[(vt >= ts) & (vt < ts + BDUR)] = BAMP
    vtv = h.Vector(vt); vvv = h.Vector(vv); vvv.play(ic_v._ref_amp, vtv, True)

    vs = h.VecStim(); sv = h.Vector(all_spk); vs.play(sv)
    ncp = h.NetCon(vs, syn); ncp.weight[0] = 1.0; ncp.delay = 1.0
    ncs = h.NetCon(soma(0.5)._ref_v, syn, sec=soma); ncs.weight[0] = -1.0; ncs.threshold = -10
    apc = h.APCount(soma(0.5)); apc.thresh = -10

    # 기록 (3D UI용 전 시계열)
    rt = h.Vector(); rt.record(h._ref_t)
    rv = h.Vector(); rv.record(soma(0.5)._ref_v)
    rc = h.Vector(); rc.record(syn._ref_c)
    rr = h.Vector(); rr.record(syn._ref_rho)
    rg = h.Vector(); rg.record(syn._ref_g)

    h.celsius = 34; h.v_init = -70
    h.finitialize(-70)
    h.dt = 0.025
    # settle
    h.continuerun(t_peak0 + off - 1.0)             # 첫 버스트 직전
    rho_init = float(syn.rho); c_pre_first = float(syn.c)
    h.continuerun(tstop)
    rho_final = float(syn.rho)
    npost = int(apc.n)

    T = np.array(rt); V = np.array(rv); C = np.array(rc); R = np.array(rr); G = np.array(rg)
    # θ 전류 재구성(기록시각 기준)
    Ith = DC + A_THETA * np.cos(2 * np.pi * (T - t_peak0) / TPER)
    Ivol = np.zeros_like(T)
    for ts in all_spk:
        Ivol[(T >= ts) & (T < ts + BDUR)] = BAMP
    dr = rho_final - rho_init
    direction = "LTP" if dr > 0.02 else ("LTD" if dr < -0.02 else "무변화")
    print(f"  [{label}] φ={phase_deg:3.0f}° · post AP {npost} · c_peak {C.max():.3f} "
          f"· ρ {rho_init:.3f}→{rho_final:.3f} (Δ{dr:+.3f}) · {direction}", flush=True)
    return dict(phase=phase_deg, label=label, rho_init=rho_init, rho_final=rho_final,
                dr=dr, direction=direction, npost=npost, c_peak=float(C.max()),
                dist=float(dist), tstop=tstop, t_peak0=t_peak0, ncyc=NCYC,
                spk=all_spk,
                trace=dict(t=T, v=V, c=C, rho=R, g=G, ith=Ith, ivol=Ivol))


def save(results, tag):
    # 요약 JSON
    summ = [{k: v for k, v in r.items() if k != "trace"} for r in results]
    for s in summ:
        s["spk"] = [round(x, 3) for x in s["spk"]]
    meta = dict(gid=int(results[0].get("gid", -1)) if results else -1,
                thetaHz=THETA_HZ, A_theta=A_THETA, DC=DC, Bamp=BAMP, Bdur=BDUR, npulse=NPULSE,
                burstHz=BURST_HZ, ncyc=NCYC, settle=SETTLE, obs=OBS,
                theta_d=1.0, theta_p=1.3, C_pre=1.0, C_post=0.275865, tau_ca=48.8373,
                sc=dict(U=SC_U, D=SC_D, F=SC_F, Nrrp=SC_NRRP, gmax=SC_GMAX),
                note="network-calibrated GB params; spike-triggered calcium (no voltage-dependent NMDA Ca)",
                results=summ)
    with open(os.path.join(OUTD, f"theta_phase_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    # 전 시계열 npz (3D UI)
    npz = {}
    for r in results:
        p = int(r["phase"])
        for k, v in r["trace"].items():
            npz[f"p{p}_{k}"] = v
        npz[f"p{p}_spk"] = np.array(r["spk"])
    np.savez_compressed(os.path.join(OUTD, f"theta_phase_{tag}_traces.npz"), **npz)
    print(f"[저장] {OUTD}/theta_phase_{tag}.json (+_traces.npz)", flush=True)


def main():
    t0 = time.time()
    import net_build as nb
    B = nb.NetBuilder()
    gid = pick_pc_gid(B)
    print(f"[대상] gid {gid} · mtype {B.mt[gid]} · θ {THETA_HZ}Hz A={A_THETA}nA DC={DC} "
          f"· 버스트 {NPULSE}p@{BURST_HZ}Hz ×{NCYC}주기 · settle {SETTLE} obs {OBS}", flush=True)

    if SMOKE or PEAK_TROUGH:
        phases = [(0.0, "peak"), (180.0, "trough")]
        tag = "smoke" if SMOKE else f"peaktrough_ncyc{NCYC}"
    elif SWEEP:
        phases = [(p, f"{int(p)}deg") for p in [0, 45, 90, 135, 180, 225, 270, 315]]
        tag = f"sweep_ncyc{NCYC}"
    else:
        phases = [(0.0, "peak"), (180.0, "trough")]
        tag = f"default_ncyc{NCYC}"

    results = []
    for ph, lab in phases:
        r = run_one(B, gid, ph, lab)
        r["gid"] = gid
        results.append(r)
    save(results, tag)
    print(f"[완료] {len(results)}조건 · 총 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
