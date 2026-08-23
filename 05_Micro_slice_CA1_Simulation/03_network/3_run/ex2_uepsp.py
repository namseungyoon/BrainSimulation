# -*- coding: utf-8 -*-
"""Ex2 — Schaffer collateral 단발 uEPSP 특성화 (미세슬라이스 맥락).

단일 SC 섬유 1개를 활성 → 그 섬유가 접촉하는 추체(PC) 타깃들에서 uEPSP(막전위)를 기록.
uEPSP는 역치하라 타 세포로 전파 안 함 → 타깃 PC + 해당 시냅스만 소형 빌드(전체망 불필요).
확률방출(Nrrp)이라 N시행 평균 → 평균 uEPSP·진폭분포·실패율. 페어펄스(ISI)로 PPR(촉진).

측정: 소마 uEPSP 진폭(mV)·지연(ms)·10-90% 상승시간·감쇠τ·PPR·실패율.
대조: Sayer 1990 단위 uEPSP · 04 벤치 · HippocampusHub gsyn.

실행: cd ~/mechbuild_gpu && python ex2_uepsp.py [--fiber F] [--ntrial 50] [--isi 50]
결과: scratch/ex2_uepsp.npz + .json
"""
import os, sys, time, json
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from neuron import h

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "lib"))
DERIVED = os.path.join(ROOT, "data", "derived")


def arg(flag, d):
    return type(d)(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else d

NTRIAL = arg("--ntrial", 15)        # 시행수 (확률방출 평균)
ISI = arg("--isi", 50.0)
FIBER = arg("--fiber", -1)          # -1 = 자동 선택(추체 타깃 많은 섬유)
SC_GMAX = 0.8 / 1000.0              # µS (mpi_baseline과 동일)
SETTLE = 40.0                       # 안정화(rest 도달, finit 빠름)
OBS = 50.0                          # 자극 후 관찰창(ms) — uEPSP τ~11ms → 충분
FAIL_MV = 0.05                      # 실패 판정 임계(mV)


def sc_stp_pc():
    return 0.14, 186.0, 129.0, 12   # SP_PC용 Use/Dep/Fac/Nrrp


def seg_kdtree(cell):
    P, ref = [], []
    for sec in cell.all:
        n = int(sec.n3d())
        if n < 2:
            continue
        Lt = sec.arc3d(n - 1) or 1.0
        for i in range(n):
            P.append((sec.x3d(i), sec.y3d(i), sec.z3d(i)))
            ref.append((sec, min(max(sec.arc3d(i) / Lt, 0.0), 1.0)))
    return cKDTree(np.array(P)), ref


def epsp_metrics(t, v, t0):
    """t0 이후 uEPSP: 진폭·지연·10-90% 상승·감쇠τ. baseline = t0 직전 평균."""
    t = np.asarray(t); v = np.asarray(v)
    pre = (t >= t0 - 5) & (t < t0)
    base = v[pre].mean() if pre.any() else v[0]
    win = (t >= t0) & (t <= t0 + OBS)
    if not win.any():
        return dict(amp=0.0, lat=np.nan, rise=np.nan, decay=np.nan, base=base)
    tw, vw = t[win], v[win]
    ipk = int(np.argmax(vw)); amp = float(vw[ipk] - base)
    lat = rise = decay = np.nan
    if amp > FAIL_MV:
        thr10, thr90 = base + 0.1 * amp, base + 0.9 * amp
        i10 = np.argmax(vw >= thr10); i90 = np.argmax(vw >= thr90)
        lat = float(tw[i10] - t0)
        rise = float(tw[i90] - tw[i10]) if i90 > i10 else np.nan
        # 감쇠: 피크 후 1/e 도달까지
        vt = base + amp / np.e
        after = vw[ipk:]; ta = tw[ipk:]
        j = np.argmax(after <= vt)
        decay = float(ta[j] - tw[ipk]) if j > 0 else np.nan
    return dict(amp=amp, lat=lat, rise=rise, decay=decay, base=base)


def main():
    import net_build as nb
    B = nb.NetBuilder()
    h.load_file("stdrun.hoc")
    pc = h.ParallelContext()          # psolve 경로(EMS 시냅스 호환) — continuerun 대신
    h.cvode_active(0)                 # 고정 dt 확정
    wc = np.load(os.path.join(DERIVED, "window_cells.npz"), allow_pickle=True)
    XYZ = wc["xyz"].astype(float); Q = wc["orientation_wxyz"]; mt = B.mt
    sc = np.load(os.path.join(DERIVED, "sc_synapses.npz"), allow_pickle=True)
    fib = np.load(os.path.join(DERIVED, "sc_fibers.npz"), allow_pickle=True)
    scpost = sc["post_gid"]; scxyz = sc["xyz"].astype(float); fid = fib["fiber_id"]
    is_pc = (mt == "SP_PC")

    # ── 섬유 선택: 추체 타깃 많은 섬유(한 번에 uEPSP 여러 개) ──
    pcsyn = is_pc[scpost]
    MAXT = arg("--maxt", 12)
    if FIBER < 0:
        uf, cnt = np.unique(fid[pcsyn], return_counts=True)
        fsel = int(uf[np.argmin(np.abs(cnt - MAXT))])   # PC 타깃 ~MAXT개인 대표 섬유
    else:
        fsel = FIBER
    sel = np.where((fid == fsel) & pcsyn)[0]     # 이 섬유의 추체 시냅스 인덱스
    targets = sorted(set(int(scpost[si]) for si in sel))[:MAXT]
    sel = sel[np.isin(scpost[sel], targets)]     # 상한 타깃으로 재필터
    print(f"[Ex2] 섬유 {fsel} · 추체 타깃 {len(targets)}개 · SC 시냅스 {len(sel)}개", flush=True)

    # ── 타깃 PC 빌드 + SC 시냅스 배치 + Vm 기록 ──
    t0 = time.time()
    fiber = h.VecStim()
    syns = []           # (syn, target_gid, si)
    recs = {}           # gid -> (t_vec, soma_v, dend_v)
    for g in targets:
        cell = B.build_cell(g)
        tree, ref = seg_kdtree(cell); rot = Rot.from_quat(Q[g][[1, 2, 3, 0]])
        soma = cell.soma[0]
        sv = h.Vector(); sv.record(soma(0.5)._ref_v)
        dv = None
        for si in sel[scpost[sel] == g]:
            mp = rot.inv().apply(scxyz[si] - XYZ[g]); _, k = tree.query(mp, k=1); sec, x = ref[k]
            syn = h.GBPlasticityStpProbSyn(sec(x))
            U, D, Fa, Nr = sc_stp_pc()
            syn.Use = U; syn.Dep = D; syn.Fac = Fa; syn.Nrrp = Nr; syn.gmax = SC_GMAX
            syn.gamma_p = 0.0; syn.gamma_d = 0.0
            syn.setRNG(g + 1, int(si) + 1, 3)      # 확률방출 RNG(필수)
            nc = h.NetCon(fiber, syn); nc.weight[0] = 1.0; nc.delay = 1.0
            ncs = h.NetCon(soma(0.5)._ref_v, syn, sec=soma); ncs.weight[0] = -1.0  # 소마 sentinel(post 신호)
            syns.append((syn, g, int(si), nc, ncs))
            if dv is None:                         # 첫 시냅스 수상돌기 세그먼트 Vm
                dv = h.Vector(); dv.record(sec(x)._ref_v)
        recs[g] = (sv, dv)
    print(f"[Ex2] 빌드 완료 {len(targets)}세포·{len(syns)}시냅스 · {time.time()-t0:.0f}s", flush=True)

    trec = h.Vector(); trec.record(h._ref_t)   # 섹션 존재 후 시간 기록
    h.celsius = 34; h.dt = 0.025

    ISO = 80.0                                     # 시행 간격(uEPSP τ~11ms → 충분히 감쇠)

    def play_run(times):
        tv = h.Vector(times); fiber.play(tv)
        pc.set_maxstep(10)
        tf = time.time(); h.finitialize(-70)
        tstop = times[-1] + OBS; tp = time.time(); pc.psolve(tstop)
        print(f"  [run] finit {tp-tf:.1f}s · psolve {tstop:.0f}ms {time.time()-tp:.1f}s", flush=True)
        return np.array(trec)

    # ── 단발: 1회 안정화 후 NTRIAL회 자극 반복 (RNG 스트림 자연 진행 → 시행별 확률방출 변동) ──
    stims = [SETTLE + k * ISO for k in range(NTRIAL)]
    t = play_run(stims)
    single = {g: [] for g in targets}
    for g in targets:
        sv = np.array(recs[g][0])
        for st in stims:
            single[g].append(epsp_metrics(t, sv, st)["amp"])
    g0 = targets[0]
    trace0 = (t.copy(), np.array(recs[g0][0]), np.array(recs[g0][1]) if recs[g0][1] is not None else None, g0)

    # ── 페어펄스: 쌍(간격 ISI)을 PAIR_ISO 간격으로 반복 → PPR (쌍 간 오염 방지) ──
    PAIR_ISO = ISI + 100.0
    pstims = []
    for k in range(NTRIAL):
        p = SETTLE + k * PAIR_ISO; pstims += [p, p + ISI]
    t2 = play_run(pstims)
    ppr = []                       # 타깃별 mean(EPSP2)/mean(EPSP1) (표준 PPR)
    pair_a1, pair_a2 = [], []      # 전체 a1/a2 풀
    for g in targets:
        sv = np.array(recs[g][0])
        a1s, a2s = [], []
        for k in range(NTRIAL):
            p = SETTLE + k * PAIR_ISO
            a1s.append(max(0.0, epsp_metrics(t2, sv, p)["amp"]))
            a2s.append(max(0.0, epsp_metrics(t2, sv, p + ISI)["amp"]))
        m1, m2 = float(np.mean(a1s)), float(np.mean(a2s))
        pair_a1 += a1s; pair_a2 += a2s
        if m1 > FAIL_MV:
            ppr.append(m2 / m1)

    # ── 집계 ──
    def agg(gid):
        amps = np.array(single[gid]); succ = amps[amps > FAIL_MV]
        return dict(gid=gid, mean_amp=float(succ.mean()) if succ.size else 0.0,
                    fail=float(np.mean(amps <= FAIL_MV)), n=len(amps))
    per = [agg(g) for g in targets]
    all_succ = np.concatenate([np.array(single[g])[np.array(single[g]) > FAIL_MV] for g in targets]) \
        if any(np.array(single[g]).max() > FAIL_MV for g in targets) else np.array([0.0])
    pa1 = np.array(pair_a1); pa2 = np.array(pair_a2)
    ppr_pooled = float(pa2.mean() / pa1.mean()) if pa1.size and pa1.mean() > FAIL_MV else np.nan   # 풀 mean(E2)/mean(E1)
    all_ppr = np.array(ppr) if ppr else np.array([np.nan])
    # 대표 세포 · 성공(최대진폭) 시행의 상세 metric
    t, sv0, dv0, g0 = trace0
    amps0 = np.array(single[g0]); kbest = int(np.argmax(amps0)) if amps0.size else 0
    m0 = epsp_metrics(t, sv0, stims[kbest])

    summary = dict(fiber=fsel, n_targets=len(targets), n_syn=len(syns), ntrial=NTRIAL, isi=ISI,
                   mean_uEPSP_mV=float(np.mean(all_succ)), median_uEPSP_mV=float(np.median(all_succ)),
                   fail_rate=float(np.mean([p["fail"] for p in per])),
                   PPR=ppr_pooled, PPR_perTarget=float(np.nanmean(all_ppr)),
                   rep_latency_ms=m0["lat"], rep_rise_ms=m0["rise"], rep_decay_ms=m0["decay"], rep_gid=g0)
    print("[Ex2] === uEPSP 특성 ===", flush=True)
    print(f"  평균 uEPSP {summary['mean_uEPSP_mV']:.3f} mV · 중앙 {summary['median_uEPSP_mV']:.3f} mV · 실패율 {summary['fail_rate']*100:.0f}%", flush=True)
    print(f"  PPR {ppr_pooled:.2f} (풀 mean E2/E1, >1=촉진) · 타깃평균 {np.nanmean(all_ppr):.2f} · 대표: 지연 {m0['lat']:.2f}ms · 상승 {m0['rise']:.2f}ms · 감쇠τ {m0['decay']:.2f}ms", flush=True)

    np.savez_compressed(os.path.join(ROOT, "scratch", "ex2_uepsp.npz"),
                        t=t, soma_v=sv0, dend_v=(dv0 if dv0 is not None else np.array([])),
                        single_amps=np.array([single[g] for g in targets], dtype=object),
                        pair_a1=np.array(pair_a1), pair_a2=np.array(pair_a2),
                        targets=np.array(targets), stims=np.array(stims), kbest=kbest,
                        settle=SETTLE, iso=ISO, isi=ISI, fiber=fsel)
    json.dump(summary, open(os.path.join(ROOT, "scratch", "ex2_uepsp.json"), "w"), indent=1)
    print(f"[Ex2] 저장 scratch/ex2_uepsp.npz · 총 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
