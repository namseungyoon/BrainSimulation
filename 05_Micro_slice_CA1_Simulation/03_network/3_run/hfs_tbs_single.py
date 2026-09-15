# -*- coding: utf-8 -*-
"""
03_network/3_run/hfs_tbs_single.py
  나-2-② 프로토콜 ① — 주파수 기반 LTP 유도 (HFS·TBS, 단일 시냅스 티어)

  문헌(그대로 재현):
   · HFS-LTP: Bliss & Collingridge 1993; Hernandez et al. 2005 — 100 Hz × 1초(100펄스), 1~4열 → LTP
   · TBS-LTP: Larson & Munkácsy 2015 (원조 Larson & Lynch 1986)
              — 4펄스 100 Hz 버스트를 5 Hz theta로 10회 = 1에폭(2초) → LTP, 가장 생리적

  대조군 성격: 현 GB 칼슘 모델은 '빈도 의존'으로 설계 → 고빈도 C_pre 대량 누적으로 θp 초과 → LTP.
   (θ위상 프로토콜의 '위상 의존 재현 실패'와 대비 — 빈도는 되고 위상은 안 됨을 보임)

  구성: 추체 1개(SP_PC) + SC→PC 시냅스 1개. pre(VecStim)로 유도열 인가.
   유도 전/후 시냅스 효능 ρ 변화(및 가중치 w %)를 판독. 전 시계열 저장(3D UI).

  실행:
    python 03_network/3_run/hfs_tbs_single.py --proto hfs        # 100Hz×100
    python 03_network/3_run/hfs_tbs_single.py --proto tbs        # TBS 1에폭
    python 03_network/3_run/hfs_tbs_single.py --proto both
"""
import os, sys, json, time
import numpy as np
from neuron import h

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "lib"))
OUTD = os.path.join(ROOT, "04_experiments", "Ex_hfs_tbs", "data")
os.makedirs(OUTD, exist_ok=True)


def arg(flag, default, cast=float):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


GID = int(arg("--gid", -1))
PROTO = sys.argv[sys.argv.index("--proto") + 1] if "--proto" in sys.argv else "both"
SETTLE = arg("--settle", 300.0)
OBS = arg("--obs", 500.0)                          # 유도 후 ρ 안정 관측
BAMP = arg("--Bamp", 0.6)                           # 수렴 볼리 프록시(테타너스가 후세포 발화 동원)
BDUR = arg("--Bdur", 3.0)
NOVOLLEY = "--novolley" in sys.argv                 # 볼리 끄고 순수 C_pre 기여만 보기
NTEST = int(arg("--ntest", 60))                     # 유도 전/후 test-EPSP 시행 수(확률방출 분포)
TESTISI = arg("--testisi", 120.0)                   # test 펄스 간격(ms) — 칼슘 감쇠 충분

SC_U, SC_D, SC_F, SC_NRRP = 0.14, 186.0, 129.0, 12
SC_GMAX = 0.8 / 1000.0


def pick_pc_gid(B):
    if GID >= 0:
        return GID
    return int(np.where(B.mt == "SP_PC")[0][0])


def pick_sr_dend(cell):
    soma = cell.soma[0]; h.distance(0, soma(0.5)); best = None
    for sec in cell.all:
        nm = sec.name()
        if ("apic" not in nm and "dend" not in nm) or "axon" in nm or "soma" in nm:
            continue
        d = h.distance(sec(0.5))
        if 120.0 <= d <= 250.0:
            best = (sec, 0.5, d)
            if "apic" in nm:
                return best
    if best:
        return best
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
    syn.gmax = SC_GMAX; syn.ca_stp = 0; syn.rho0 = 0.5
    syn.setRNG(gid + 1, 1, 3)
    return cell, syn, dist


def induction_spikes(proto):
    """유도 자극열 시각(ms), settle 이후 시작."""
    t0 = SETTLE + 10.0
    if proto == "hfs":                              # 100 Hz × 100펄스 = 1초
        isi = 10.0; n = 100
        return [t0 + k * isi for k in range(n)], "HFS 100Hz×100 (1초)"
    if proto == "tbs":                              # 4펄스 100Hz 버스트 ×10 @ 5Hz = 2초
        spk = []; burst_isi = 10.0; theta_isi = 200.0
        for b in range(10):
            bt = t0 + b * theta_isi
            spk += [bt + k * burst_isi for k in range(4)]
        return spk, "TBS 4p@100Hz ×10 @5Hz (2초)"
    raise ValueError(proto)


def run_one(B, gid, proto):
    cell, syn, dist = build(B, gid)
    soma = cell.soma[0]
    spk, lab = induction_spikes(proto)
    tstop = spk[-1] + OBS

    vs = h.VecStim(); sv = h.Vector(spk); vs.play(sv)
    ncp = h.NetCon(vs, syn); ncp.weight[0] = 1.0; ncp.delay = 1.0
    ncs = h.NetCon(soma(0.5)._ref_v, syn, sec=soma); ncs.weight[0] = -1.0; ncs.threshold = -10
    apc = h.APCount(soma(0.5)); apc.thresh = -10

    # 수렴 볼리 프록시 (테타너스가 후세포 발화 동원)
    if not NOVOLLEY:
        ic_v = h.IClamp(soma(0.5)); ic_v.delay = 0; ic_v.dur = 1e9
        vt = np.arange(0.0, tstop + REC_DT, REC_DT); vv = np.zeros_like(vt)
        for ts in spk:
            vv[(vt >= ts) & (vt < ts + BDUR)] = BAMP
        vtv = h.Vector(vt); vvv = h.Vector(vv); vvv.play(ic_v._ref_amp, vtv, True)

    rt = h.Vector(); rt.record(h._ref_t)
    rv = h.Vector(); rv.record(soma(0.5)._ref_v)
    rc = h.Vector(); rc.record(syn._ref_c)
    rr = h.Vector(); rr.record(syn._ref_rho)
    rg = h.Vector(); rg.record(syn._ref_g)

    h.celsius = 34; h.v_init = -70
    h.finitialize(-70); h.dt = 0.025
    h.continuerun(spk[0] - 1.0)
    rho_init = float(syn.rho)
    h.continuerun(tstop)
    rho_final = float(syn.rho); npost = int(apc.n)

    T = np.array(rt); V = np.array(rv); C = np.array(rc); R = np.array(rr); G = np.array(rg)
    dr = rho_final - rho_init
    # 가중치 w = w0 + rho(b*w0 - w0) → EPSP% 근사: (w(final)/w(init)-1). b·w0는 .mod 기본, ρ만으로 부호 판정
    direction = "LTP" if dr > 0.02 else ("LTD" if dr < -0.02 else "무변화")
    print(f"  [{lab}] post AP {npost} · c_peak {C.max():.3f} · ρ {rho_init:.3f}→{rho_final:.3f} "
          f"(Δ{dr:+.3f}) · {direction}", flush=True)
    return dict(proto=proto, label=lab, rho_init=rho_init, rho_final=rho_final, dr=dr,
                direction=direction, npost=npost, c_peak=float(C.max()), dist=float(dist),
                nspk=len(spk), tstop=tstop, spk=spk,
                trace=dict(t=T, v=V, c=C, rho=R, g=G))


def save(results, gid):
    summ = [{k: v for k, v in r.items() if k != "trace"} for r in results]
    for s in summ:
        s["spk"] = [round(x, 2) for x in s["spk"]]
    meta = dict(gid=gid, settle=SETTLE, obs=OBS, Bamp=(0 if NOVOLLEY else BAMP), Bdur=BDUR,
                theta_d=1.0, theta_p=1.3, C_pre=1.0, C_post=0.275865, tau_ca=48.8373,
                sc=dict(U=SC_U, D=SC_D, F=SC_F, Nrrp=SC_NRRP, gmax=SC_GMAX), results=summ)
    tag = PROTO
    json.dump(meta, open(os.path.join(OUTD, f"hfs_tbs_{tag}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    npz = {}
    for r in results:
        for k, v in r["trace"].items():
            npz[f"{r['proto']}_{k}"] = v
        npz[f"{r['proto']}_spk"] = np.array(r["spk"])
    np.savez_compressed(os.path.join(OUTD, f"hfs_tbs_{tag}_traces.npz"), **npz)
    print(f"[저장] {OUTD}/hfs_tbs_{tag}.json (+_traces.npz)", flush=True)


def main():
    t0 = time.time()
    import net_build as nb
    B = nb.NetBuilder(); gid = pick_pc_gid(B)
    protos = ["hfs", "tbs"] if PROTO == "both" else [PROTO]
    print(f"[대상] gid {gid} · {B.mt[gid]} · 볼리 {'off' if NOVOLLEY else BAMP} · 프로토콜 {protos}", flush=True)
    results = [run_one(B, gid, p) for p in protos]
    save(results, gid)
    print(f"[완료] {len(results)}조건 · 총 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
