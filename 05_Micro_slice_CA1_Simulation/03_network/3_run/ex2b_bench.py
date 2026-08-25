# -*- coding: utf-8 -*-
"""Ex2b 2세포 벤치 (다중 ISI) — (pre→post) 연결의 uPSP·PPR·STP곡선·kinetics 격리 측정.
전시냅스 세포를 IClamp로 발화 → 후시냅스 Vm 기록. 배치·파라미터는 ex3_io 로직 재사용.
확률방출(Nrrp)이라 N시행 평균. ISI 여러 개로 STP(촉진/억압)의 시간축을 측정.

실행: python ex2b_bench.py --pre SP_PC --post SP_PC [--isis 20,50,100,200] [--ntrial 20] [--amp 1.2] [--save]
"""
import os, sys, json, io
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from neuron import h

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "lib"))
DER = os.path.join(ROOT, "data", "derived")
import ex3_io as X
import net_build as nb


def arg(f, d):
    return type(d)(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else d


PRE = arg("--pre", "SP_PC"); POST = arg("--post", "SP_PC")
ISIS = [float(x) for x in arg("--isis", "20,50,100,200").split(",")]
NTRIAL = int(arg("--ntrial", 20)); AMP = arg("--amp", 1.2)
HOLD_V = arg("--holdv", -50.0)                    # 억제(IPSC) 측정용 전압클램프 유지전위(mV)


def post_record(P, post_soma):
    """후세포 응답 기록 설정. 억제=전압클램프(-50mV) IPSC 전류, 흥분=전류클램프 Vm.
    반환: (기록ref, SEClamp객체 or None, 단위). E_GABA≈rest라 IPSP는 rest에서 안 보여 VC 필요."""
    if P["mech"] == "I":
        se = h.SEClamp(post_soma(0.5)); se.rs = 0.001; se.dur1 = 1e9; se.amp1 = HOLD_V
        return se._ref_i, se, "nA"                # IPSC (전류)
    return post_soma(0.5)._ref_v, None, "mV"      # EPSP (전압)
SAVE = "--save" in sys.argv
MORPH3D = "--morph3d" in sys.argv                # 세그먼트별 Vm + 시냅스 전류 3D 기록(대표 경로)
STIM = 100.0; TAIL = 80.0
REP_ISI = 50.0                                   # UI 대표 파형 ISI


def seg_positions(cell, xyzg, rot):
    """축삭 제외 전 세그먼트의 global 3D 위치 + Vm ref + soma여부."""
    pos, refs, soma = [], [], []
    for sec in cell.all:
        nm = sec.name(); n = int(sec.n3d())
        if n < 2 or ("axon" in nm) or ("node" in nm) or ("myelin" in nm):
            continue
        arc = np.array([sec.arc3d(i) for i in range(n)]); Lt = arc[-1] or 1.0
        xs = np.array([sec.x3d(i) for i in range(n)]); ys = np.array([sec.y3d(i) for i in range(n)]); zs = np.array([sec.z3d(i) for i in range(n)])
        so = "soma" in nm
        for seg in sec:
            a = seg.x * Lt
            loc = np.array([np.interp(a, arc, xs), np.interp(a, arc, ys), np.interp(a, arc, zs)])
            pos.append(xyzg + rot.apply(loc)); refs.append(seg); soma.append(so)
    return pos, refs, soma


def record_morph3d(P, XYZ, Q, isi=REP_ISI, rec_dt=0.25):
    """대표 ISI 1시행: 두 세포 세그먼트 Vm(t) + 시냅스 전류 i(t) + 3D 위치 기록."""
    pre_soma, post_soma = P["pre_soma"], P["post_soma"]
    rotq = Rot.from_quat(Q[P["qg"]][[1, 2, 3, 0]]); rotp = Rot.from_quat(Q[P["pg"]][[1, 2, 3, 0]])
    pos_q, ref_q, soma_q = seg_positions(P["post"], XYZ[P["qg"]], rotq)
    pos_p, ref_p, soma_p = seg_positions(P["pre"], XYZ[P["pg"]], rotp)
    tv = h.Vector(); tv.record(h._ref_t, rec_dt)
    vq = [h.Vector() for _ in ref_q]; [v.record(s._ref_v, rec_dt) for v, s in zip(vq, ref_q)]
    vp = [h.Vector() for _ in ref_p]; [v.record(s._ref_v, rec_dt) for v, s in zip(vp, ref_p)]
    si = [h.Vector() for _ in P["syns"]]; spos = []
    for (syn, nc, k), sv in zip(P["syns"], si):
        try: sv.record(syn._ref_i, rec_dt)
        except Exception: sv.record(syn._ref_g, rec_dt)
        seg = syn.get_segment(); sec = seg.sec; n = int(sec.n3d())
        arc = np.array([sec.arc3d(i) for i in range(n)]); Lt = arc[-1] or 1.0
        a = seg.x * Lt
        loc = np.array([np.interp(a, [sec.arc3d(i) for i in range(n)], [getattr(sec, ax)(i) for i in range(n)]) for ax in ("x3d", "y3d", "z3d")])
        spos.append(XYZ[P["qg"]] + rotq.apply(loc))
    ic1 = h.IClamp(pre_soma(0.5)); ic1.delay = STIM; ic1.dur = 3.0; ic1.amp = AMP
    ic2 = h.IClamp(pre_soma(0.5)); ic2.delay = STIM + isi; ic2.dur = 3.0; ic2.amp = AMP
    _ih = hold_post(P, post_soma)                 # 억제면 후세포 탈분극 홀딩
    for (syn, nc, k) in P["syns"]:
        syn.setRNG(P["qg"] + 1, 900000 + k, 7 if P["mech"] == "E" else 4)
    h.dt = 0.025; h.finitialize(-70); h.continuerun(STIM + isi + TAIL)
    return dict(t=np.array(tv),
                pos_q=np.array(pos_q), vq=np.array([np.array(v) for v in vq]), soma_q=np.array(soma_q),
                pos_p=np.array(pos_p), vp=np.array([np.array(v) for v in vp]), soma_p=np.array(soma_p),
                spos=np.array(spos), si=np.array([np.array(v) for v in si]))


def build_pair(B, XYZ, Q, seed, radial, rules, ipre, ipost, insyn, irule, igsyn, imech):
    mt = B.mt
    sel = np.where((mt[ipre] == PRE) & (mt[ipost] == POST))[0]
    if len(sel) == 0:
        return None
    ci = sel[int(np.argsort(np.abs(insyn[sel] - np.median(insyn[sel])))[0])]
    pg, qg, ns = int(ipre[ci]), int(ipost[ci]), int(insyn[ci])
    rl = rules.get(int(irule[ci])); gs = float(igsyn[ci]); mech = imech[ci]
    post = B.build_cell(qg); pre = B.build_cell(pg)
    rot = Rot.from_quat(Q[qg][[1, 2, 3, 0]])
    secs, xs, comp = X.compartments(post, XYZ[qg], rot, seed, radial)
    pool = comp[X.target_comp(PRE)]
    if len(pool) == 0:
        pool = comp["dend"] if len(comp["dend"]) else comp["soma"]
    rng = np.random.default_rng(qg * 1000 + pg)
    ksel = pool[rng.integers(0, len(pool), ns)]
    pre_soma, post_soma = pre.soma[0], post.soma[0]
    syns = []
    for k in ksel:
        sec, x = secs[int(k)], xs[int(k)]
        if mech == "E":
            syn = h.GBPlasticityStpProbSyn(sec(x))
            if rl:
                syn.Use = rl["U"]; syn.Dep = rl["D"]; syn.Fac = rl["F"]; syn.Nrrp = rl["NRRP"]
            syn.gmax = gs / 1000.0; syn.gamma_p = 0.0; syn.gamma_d = 0.0; w = 1.0
        else:
            syn = h.ProbGABAAB_EMS(sec(x))
            if rl:
                syn.Use = rl["U"]; syn.Dep = rl["D"]; syn.Fac = rl["F"]; syn.Nrrp = rl["NRRP"]
            w = gs / 1000.0
        nc = h.NetCon(pre_soma(0.5)._ref_v, syn, sec=pre_soma)
        nc.threshold = -10.0; nc.weight[0] = w; nc.delay = 1.0
        syns.append((syn, nc, int(k)))
    return dict(pre=pre, post=post, pre_soma=pre_soma, post_soma=post_soma, syns=syns,
                pg=pg, qg=qg, ns=ns, rl=rl, gs=gs, mech=mech)


def run_train(P, freq, npulse, ntrial):
    """freq(Hz)에서 npulse 트레인 N시행 평균 → 펄스별 상대진폭(1번째=1)."""
    pre_soma, post_soma = P["pre_soma"], P["post_soma"]
    isi = 1000.0 / freq
    ts = [STIM + k * isi for k in range(npulse)]
    tstop = ts[-1] + TAIL
    ics = []
    for t0 in ts:
        ic = h.IClamp(pre_soma(0.5)); ic.delay = t0; ic.dur = 3.0; ic.amp = AMP; ics.append(ic)
    qrec, _se, _unit = post_record(P, post_soma)  # 억제=VC IPSC, 흥분=Vm
    tv = h.Vector(); tv.record(h._ref_t)
    qv = h.Vector(); qv.record(qrec)
    acc = None
    for tr in range(ntrial):
        for (syn, nc, k) in P["syns"]:
            syn.setRNG(P["qg"] + 1 + tr * 7919, 900000 + k + tr * 131, 7 if P["mech"] == "E" else 4)
        h.dt = 0.025; h.finitialize(-70); h.continuerun(tstop)
        v = np.array(qv); acc = v if acc is None else acc + v
    v = acc / ntrial; t = np.array(tv); base = v[t < STIM].mean()
    amps = []; s = None
    for k, t0 in enumerate(ts):
        loc = v[(t >= t0 - 1.5) & (t < t0)].mean() if k else base
        w = (t >= t0 + 1.0) & (t < t0 + 1.0 + min(WFIX, isi - 2))
        if not w.any():
            amps.append(0.0); continue
        dw = v[w] - loc
        if s is None:                                  # 1번째로 응답 방향 결정
            a0 = float(dw[int(np.argmax(np.abs(dw)))]); s = 1.0 if a0 >= 0 else -1.0; amps.append(a0)
        else:
            amps.append(float(s * np.max(s * dw)))     # 같은 방향의 peak
    a1 = amps[0] if abs(amps[0]) > 1e-4 else 1.0
    return [round(a / a1, 3) for a in amps]


def run_isi(P, isi, ntrial):
    """ISI 페어펄스 N시행 평균 post Vm + 대표 pre Vm 반환."""
    pre_soma, post_soma = P["pre_soma"], P["post_soma"]
    tstop = STIM + isi + TAIL
    ic1 = h.IClamp(pre_soma(0.5)); ic1.delay = STIM; ic1.dur = 3.0; ic1.amp = AMP
    ic2 = h.IClamp(pre_soma(0.5)); ic2.delay = STIM + isi; ic2.dur = 3.0; ic2.amp = AMP
    qrec, _se, _unit = post_record(P, post_soma)  # 억제=VC IPSC, 흥분=Vm
    tv = h.Vector(); tv.record(h._ref_t)
    pv = h.Vector(); pv.record(pre_soma(0.5)._ref_v)
    qv = h.Vector(); qv.record(qrec)
    acc = None; nsp = 0; pre_last = None
    for tr in range(ntrial):
        for (syn, nc, k) in P["syns"]:
            syn.setRNG(P["qg"] + 1 + tr * 7919, 900000 + k + tr * 131, 7 if P["mech"] == "E" else 4)
        tsp = h.Vector(); ncsp = h.NetCon(pre_soma(0.5)._ref_v, None, sec=pre_soma); ncsp.threshold = -10; ncsp.record(tsp)
        h.dt = 0.025; h.finitialize(-70); h.continuerun(tstop)
        nsp += int(tsp.size()); v = np.array(qv)
        acc = v if acc is None else acc + v; pre_last = np.array(pv)
    return np.array(tv), acc / ntrial, pre_last, nsp / ntrial


WFIX = min(min(ISIS) - 2.0, 30.0)                 # 응답 측정창(모든 ISI 공통 → a1 ISI 독립·PPR 일관)


def amp_pp(t, v, base, isi):
    """페어펄스 두 응답 진폭. 두 응답 모두 '고정폭 창'에서 '1번째와 같은 방향'의 peak로 측정.
    고정창 → a1이 ISI에 무관(일관 PPR). 방향인식 peak → 음수아티팩트 방지 + 느린 IPSC 포착.
    (주의: 수상돌기 억제는 소마 VC 공간클램프로 과소·저속 측정될 수 있음 — 실제 실험과 동일 한계)"""
    w1 = (t >= STIM + 1.0) & (t < STIM + 1.0 + WFIX)
    d1 = v - base
    if not w1.any():
        return 0.0, 0.0
    k1 = int(np.argmax(np.abs(d1[w1]))); a1 = float(d1[w1][k1]); s = 1.0 if a1 >= 0 else -1.0
    pre2 = v[(t >= STIM + isi - 1.5) & (t < STIM + isi)].mean() if isi > 4 else base
    w2 = (t >= STIM + isi + 1.0) & (t < STIM + isi + 1.0 + WFIX)
    a2 = float(s * np.max(s * (v[w2] - pre2))) if w2.any() else 0.0
    return a1, a2


def kinetics(t, v, base):
    """단발(1번째) 응답 kinetics: 잠복·상승(10-90%)·감쇠τ(peak→1/e)."""
    w = (t >= STIM) & (t < STIM + 80)
    tw, dw = t[w], (v - base)[w]
    if len(tw) < 5:
        return 0.0, 0.0, 0.0
    pk = int(np.argmax(np.abs(dw))); peak = dw[pk]
    if abs(peak) < 1e-3:
        return 0.0, 0.0, 0.0
    s = np.sign(peak); sd = dw * s; ap = abs(peak)      # 응답 방향 양수화
    def back(frac):                                     # peak에서 역스캔 → 10%/90% 교차(온셋)
        for i in range(pk, 0, -1):
            if sd[i] < frac * ap:
                return tw[i]
        return tw[0]
    t10 = back(0.1); t90 = back(0.9)
    lat = t10 - STIM; rise = t90 - t10
    tau = 0.0                                            # peak→37% 감쇠
    for i in range(pk, len(sd)):
        if sd[i] <= 0.37 * ap:
            tau = tw[i] - tw[pk]; break
    return float(lat), float(rise), float(tau)


def main():
    B = nb.NetBuilder()
    wc = np.load(DER + "/window_cells.npz", allow_pickle=True); XYZ = wc["xyz"]; Q = wc["orientation_wxyz"]
    cfg = json.load(io.open(ROOT + "/config/window_layout.json", encoding="utf-8"))
    fr = cfg["frame_um"]; seed = np.array(fr["seed"]); radial = np.array(fr["radial_dir"])
    rules = {r["id"]: r for r in json.load(io.open(ROOT + "/config/synapse_rules.json", encoding="utf-8"))["internal_rules"]}
    di = np.load(DER + "/synapses_internal.npz", allow_pickle=True)
    p = np.load(DER + "/synapse_params.npz", allow_pickle=True)
    P = build_pair(B, XYZ, Q, seed, radial, rules,
                   di["pre_gid"], di["post_gid"], di["n_syn"],
                   p["internal_rule"], p["internal_gsyn"], p["internal_mech"].astype(str))
    if P is None:
        print(f"[Ex2b] 연결 없음 {PRE}->{POST}", flush=True); return
    print(f"[Ex2b] {PRE}->{POST}: pre gid{P['pg']} post gid{P['qg']} · {P['ns']}시냅스 · "
          f"type={P['rl']['type'] if P['rl'] else '?'} mech={P['mech']} gsyn={P['gs']}nS · ISI {ISIS} · {NTRIAL}시행", flush=True)
    if MORPH3D:
        wc2 = np.load(DER + "/window_cells.npz", allow_pickle=True)
        M = record_morph3d(P, wc2["xyz"], wc2["orientation_wxyz"])
        outd = os.path.join(ROOT, "04_experiments", "Ex2b_connection_matrix", "traces"); os.makedirs(outd, exist_ok=True)
        np.savez(os.path.join(outd, f"morph3d_{PRE}__{POST}.npz"),
                 pre=PRE, post=POST, cls=(P["rl"]["type"] if P["rl"] else "?"), mech=P["mech"],
                 ns=P["ns"], gsyn=P["gs"], stim=STIM, isi=REP_ISI, **M)
        print(f"[morph3d] {M['vq'].shape[0]}+{M['vp'].shape[0]}세그 · {M['t'].size}프레임 · 시냅스 {M['si'].shape[0]} -> traces/morph3d_{PRE}__{POST}.npz", flush=True)
        return
    isis, a1s, a2s, pprs = [], [], [], []
    rep_t = rep_v = rep_pre = None; lat = rise = tau = 0.0; presp = 0.0; base0 = -70.0
    for isi in ISIS:
        t, v, preV, sp = run_isi(P, isi, NTRIAL)
        base = v[t < STIM].mean()
        a1, a2 = amp_pp(t, v, base, isi)
        ppr = (a2 / a1) if abs(a1) > 1e-4 else float("nan")
        isis.append(isi); a1s.append(a1); a2s.append(a2); pprs.append(ppr)
        if isi == max(ISIS):                            # kinetics = 가장 잘 분리된 1번째 응답
            lat, rise, tau = kinetics(t, v, base); presp = sp; base0 = base
        if abs(isi - REP_ISI) < 1e-6 or (rep_t is None and isi == ISIS[0]):
            rep_t, rep_v, rep_pre = t, v, preV
        print(f"  ISI {isi:5.0f}ms: u{'EPSP' if P['mech']=='E' else 'IPSP'}1 {a1:.3f} · 2 {a2:.3f} · PPR {ppr:.2f}", flush=True)
    a1m = float(np.nanmean([abs(x) for x in a1s]))
    ntr_tr = max(8, NTRIAL // 2)
    train20 = run_train(P, 20.0, 8, ntr_tr)                # 20Hz 8펄스 트레인
    freqs = [5.0, 10.0, 20.0, 40.0]
    freqR = [round(run_train(P, f, 10, ntr_tr)[-1], 3) for f in freqs]   # 주파수별 정상상태
    print(f"[결과] pre 발화 {presp:.1f}회 · u1 {a1m:.3f}mV · PPR@50 {pprs[isis.index(50.0)] if 50.0 in isis else float('nan'):.2f} · "
          f"잠복 {lat:.2f}ms · τ {tau:.1f}ms · train20 정상상태 {train20[-1]:.2f} · freqR {freqR}", flush=True)
    if SAVE:
        outd = os.path.join(ROOT, "04_experiments", "Ex2b_connection_matrix", "traces"); os.makedirs(outd, exist_ok=True)
        lo = STIM - 8; hi = STIM + REP_ISI + TAIL
        m = (rep_t >= lo) & (rep_t <= hi)
        np.savez(os.path.join(outd, f"pair_{PRE}__{POST}.npz"),
                 pre=PRE, post=POST, cls=(P["rl"]["type"] if P["rl"] else "?"), mech=P["mech"],
                 ns=P["ns"], gsyn=P["gs"], base=base0, presp=presp,
                 U=(P["rl"]["U"] if P["rl"] else 0), D=(P["rl"]["D"] if P["rl"] else 0), F=(P["rl"]["F"] if P["rl"] else 0),
                 isis=np.array(isis), a1s=np.array(a1s), a2s=np.array(a2s), pprs=np.array(pprs),
                 train20=np.array(train20), freqs=np.array(freqs), freqR=np.array(freqR),
                 lat=lat, rise=rise, tau=tau, stim=STIM, rep_isi=REP_ISI,
                 t=rep_t[m].astype(np.float32), v=rep_v[m].astype(np.float32), preV=rep_pre[m].astype(np.float32))
        print(f"[저장] traces/pair_{PRE}__{POST}.npz", flush=True)


if __name__ == "__main__":
    main()
