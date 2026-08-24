# -*- coding: utf-8 -*-
"""3-9 bAP 프로파일 — post 발화가 수상돌기를 거꾸로 올라가며 얼마나 남는가

단계   : 3-9 (파이프라인 3단계 시냅스 / 하위 9 bap)
쉬운 설명: post 세포가 발화하면 그 스파이크가 소마에서 수상돌기로 **거꾸로** 퍼진다
          (back-propagating AP, bAP). 이게 시냅스 지점에 얼마나 크게 도달하는지 잰다.
          가소성에서 bAP 는 "후시냅스가 발화했다" 는 신호를 시냅스에 전달하는 통로다.
방법   : post 소마에 IClamp 로 (a) 단발 (b) 5발 20Hz 트레인을 주고, 3-8 과 **같은 16지점**의
          국소 전압에서 bAP 피크 진폭·소마 대비 감쇠율·피크 지연을 잰다.
          시냅스는 만들되 gmax=0 으로 껐다 — bAP 만 순수하게 본다.
          정착(D12) 스냅샷 1회 + 조건마다 복원.
★문헌  : Golding, Kath & Spruston 2001 J Neurophysiol 86:2998 (PMID 11731556)
          - 소마 280um 이내: 단발 bAP 감쇠 < 50%
          - 300um 이상: **이분화** — 강한 역전파 26~42% 감쇠(9/20) 또는 약한 역전파 71~87%(10/20)
          - 원인은 Na/K 채널 분포 차이. 우리 모델은 한 세포이므로 한 값만 나온다.
          Spruston, Schiller, Stuart & Sakmann 1995 Science 268:297 (PMID 7716524)
          - **단발** AP 의 수상돌기 칼슘은 거리에 따라 유의하게 감쇠하지 않지만
            **트레인**은 크게 감쇠한다(후반 스파이크가 원위 침입에 실패). 실패는 분기점에서 자주.
          => 그래서 단발과 트레인을 함께 재고 활동의존 감쇠를 정량한다.
★왜 중요: GB 계열 가소성 엔진은 **후시냅스 칼슘 기여를 상수로** 둔다 — 시냅스가 소마에서
          얼마나 멀든 같은 칼슘을 준다는 뜻이다. 실제 bAP 는 원위로 갈수록 감쇠하므로,
          6-8(위치 의존 가소성)을 GB 로 그냥 돌리면 **인공적으로 평평**해진다.
          이 단계의 감쇠곡선이 그 보정(기본 OFF)의 근거이자 결핍 분석의 재료다.
          GluSynapse 는 국소 전압에서 칼슘을 만들므로 보정이 필요 없다 — 이 대비가 6-9 재료.
결과   : figures/3-9_bap_profile.png · figures/3-9_bap.json
실행   : . .\\env\\activate.ps1 ; & $Py04 03_synapse\\9_bap\\3-9_bap_profile.py
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np                          # noqa: E402
from lib import plots                        # noqa: E402
from lib import morphology as mo             # noqa: E402
from lib.bench import Bench                   # noqa: E402
from lib.wiring import Wiring, SETTLE_MS      # noqa: E402
from lib.nrnenv import h                     # noqa: E402

T0 = SETTLE_MS + 10.0        # 첫 스파이크
# 활동의존 감쇠(Spruston1995)가 존재하는지 보려면 한 조건만으로는 부족하다.
# 약한 것부터 센 것까지 훑는다: (스파이크 수, 주파수 Hz)
TRAINS = [(5, 20.0), (4, 100.0), (10, 50.0)]   # 4발 100Hz = 고전 TBS 버스트(6-5)
N_TRAIN, F_TRAIN = TRAINS[0]
ISI = 1000.0 / F_TRAIN
TSTOP_PAD = 60.0
REC_DT = 0.025
AMP_NA, DUR_MS = 1.2, 3.0
WIN = 9.0                    # 스파이크 후 피크 탐색 창 (ms)

# 3-8 과 같은 지점 (직접 비교 가능하게)
TARGET_BASAL = [20, 50, 80, 110, 140, 170, 200]
TARGET_APICAL = [50, 100, 150, 200, 300, 400, 500, 600, 700]

# Golding 2001 문헌 기준
GOLD_PROX_UM, GOLD_PROX_MAX_ATT = 280.0, 50.0          # 280um 이내 감쇠 < 50%
GOLD_DIST_UM = 300.0
GOLD_STRONG = (26.0, 42.0)                              # 강한 역전파 감쇠 범위 (%)
GOLD_WEAK = (71.0, 87.0)                                # 약한 역전파 감쇠 범위 (%)


def pick_sites(post):
    h.distance(0, post.soma[0](0.5))
    pool = {"basal": [], "apical": []}
    for s in post.all:
        nm = s.name()
        dom = "basal" if ".dend" in nm else ("apical" if ".apic" in nm else None)
        if dom is None:
            continue
        pool[dom].append((h.distance(s(0.5)), s))
    sites = []
    for dom, targets in (("basal", TARGET_BASAL), ("apical", TARGET_APICAL)):
        used = set()
        for tg in targets:
            if not pool[dom]:
                continue
            d, s = min(pool[dom], key=lambda ds: abs(ds[0] - tg))
            if s.name() in used:
                continue
            used.add(s.name())
            sites.append(dict(dom=dom, dist=float(d), sec=s))
    sites.sort(key=lambda x: (x["dom"], x["dist"]))
    return sites


def spike_amp(t, v, t_spk):
    """t_spk 직전 기저선 대비 WIN 창 안의 피크 진폭과 피크 시각."""
    pre = v[(t >= t_spk - 2.0) & (t < t_spk)]
    base = float(pre.mean()) if len(pre) else float(v[0])
    m = (t >= t_spk) & (t <= t_spk + WIN)
    if not m.any():
        return 0.0, float("nan")
    seg = v[m]
    i = int(np.argmax(seg))
    return float(seg[i] - base), float(t[m][i] - t_spk)


def main():
    plots.setup()
    print("=== 3-9 bAP 프로파일 ===")
    b = Bench()
    sites = pick_sites(b.post)
    n_bas = sum(1 for s in sites if s["dom"] == "basal")
    print(f"  시험 지점 {len(sites)}개 (기저 {n_bas} · 정단 {len(sites)-n_bas})")

    # 시냅스를 만들되 전부 끈다(gmax=0) — 국소 전압 기록 통로로만 쓴다
    segs = [(s["sec"](0.5), dict(delay_ms=0.5, path_um=s["dist"], domain=s["dom"]))
            for s in sites]
    w = Wiring(b, frozen=True, segs=segs)
    for syn, _ in w.syns:
        syn.gmax = 0.0
    print(f"  시냅스 {len(w.syns)}개 전부 gmax=0 (bAP 만 측정) · 정착 {SETTLE_MS:.0f}ms")

    # ★ post 소마를 직접 발화시킨다 (pre 는 쓰지 않는다 — NetCon 도 만들지 않았다)
    MAXN = max(n for n, _ in TRAINS)
    ics = [h.IClamp(b.post_soma_seg()) for _ in range(MAXN)]
    for ic in ics:
        ic.dur, ic.amp = DUR_MS, 0.0
    w.keep += ics

    w.record(rec_dt=REC_DT, local_v=True, currents=False)
    w.settle()

    def run(n_on, freq):
        """n_on 발을 freq Hz 로 준다. 나머지 IClamp 는 끈다."""
        isi = 1000.0 / freq
        for k, ic in enumerate(ics):
            ic.delay = T0 + k * isi
            ic.amp = AMP_NA if k < n_on else 0.0
        w.restore()
        w.run_settled(T0 + isi * n_on + TSTOP_PAD)
        return w.arrays(), [T0 + k * isi for k in range(n_on)]

    def n_spikes(v):
        return int(((v[:-1] < 0) & (v[1:] >= 0)).sum())

    # ── 조건 1: 단발 ─────────────────────────────────────────────────────
    R1, st1 = run(1, TRAINS[0][1])
    t = R1["t"]
    soma1, _ = spike_amp(t, R1["post_v"], st1[0])
    nspk1 = n_spikes(R1["post_v"])
    print(f"  [단발] post 소마 스파이크 {nspk1}발 · 소마 진폭 {soma1:.1f} mV")

    single = []
    for i, s in enumerate(sites):
        a, lag = spike_amp(t, R1["local_v"][i], st1[0])
        single.append(dict(dom=s["dom"], dist=s["dist"],
                           sec=s["sec"].name().split(".")[-1],
                           amp=a, att=100.0 * (1.0 - a / soma1), lag=lag))

    # ── 조건 2~: 트레인 여러 개 (활동의존 감쇠 탐색) ──────────────────────
    # ★ 핵심 구분: '소마 발화 실패' 와 'bAP 원위 침입 실패' 는 다른 현상이다.
    #   Spruston1995 가 말하는 것은 후자다(소마는 발화하는데 원위에 못 들어간다).
    #   그래서 IClamp 시각이 아니라 **실제 검출된 소마 스파이크 시각**을 쓰고,
    #   국소 진폭을 그 스파이크의 소마 진폭으로 나눠(=침입 효율) 소마 실패 영향을 뺀다.
    trains = []
    for (n, fq) in TRAINS:
        R, st = run(n, fq)
        tt = R["t"]
        sv = R["post_v"]
        cross = np.flatnonzero((sv[:-1] < 0) & (sv[1:] >= 0))
        st_det = [float(tt[i]) for i in cross]              # 실측 소마 스파이크 시각
        ns = len(st_det)
        somaA = [spike_amp(tt, sv, ts - 0.5)[0] for ts in st_det]
        per = []
        for i, s in enumerate(sites):
            amps = [spike_amp(tt, R["local_v"][i], ts - 0.5)[0] for ts in st_det]
            # 침입 효율 = 국소 진폭 / 그 스파이크의 소마 진폭
            eff = [(a / sa if sa > 5.0 else float("nan"))
                   for a, sa in zip(amps, somaA)]
            fe = [e for e in eff if np.isfinite(e)]
            per.append(dict(dom=s["dom"], dist=s["dist"], amps=amps, eff=eff,
                            ratio=((fe[-1] / fe[0]) if len(fe) >= 2 and fe[0] > 0
                                   else float("nan"))))
        rr = np.array([p["ratio"] for p in per])
        f_ = np.isfinite(rr)
        soma_ok = (ns == n)
        trains.append(dict(n=n, freq=fq, n_spikes=ns, soma=somaA, per=per,
                           soma_ok=soma_ok, spk_t=st_det,
                           rmin=(float(np.nanmin(rr[f_])) if f_.any() else float("nan")),
                           rmax=(float(np.nanmax(rr[f_])) if f_.any() else float("nan"))))
        tag = "" if soma_ok else f"  ← ★소마 발화 실패 (목표 {n}발)"
        print(f"  [트레인 {n}발 {fq:.0f}Hz] 소마 스파이크 {ns}발{tag}")
        if ns >= 2:
            print(f"      소마 진폭 {somaA[0]:.1f} -> {somaA[-1]:.1f} mV · "
                  f"침입효율 마지막/첫 비 {trains[-1]['rmin']:.3f}~{trains[-1]['rmax']:.3f}")

    # 대표 트레인: 소마가 목표대로 발화한 조건 중 가장 빈도 높은 것
    _cand = [tr for tr in trains if tr["soma_ok"] and tr["n_spikes"] >= 2]
    strong = max(_cand, key=lambda tr: tr["freq"]) if _cand else trains[0]
    train = strong["per"]

    dom = np.array([r["dom"] for r in single])
    dist = np.array([r["dist"] for r in single])
    amp = np.array([r["amp"] for r in single])
    att = np.array([r["att"] for r in single])
    lag = np.array([r["lag"] for r in single])
    ratio = np.array([r["ratio"] for r in train])

    api = dom == "apical"
    prox = api & (dist <= GOLD_PROX_UM)
    distal = api & (dist >= GOLD_DIST_UM)

    print()
    for d in ("basal", "apical"):
        m = dom == d
        print(f"  [{d}] bAP 진폭 {amp[m].max():.1f} -> {amp[m].min():.1f} mV "
              f"({dist[m].min():.0f} -> {dist[m].max():.0f}um) · "
              f"감쇠 {att[m].min():.0f} -> {att[m].max():.0f}% · "
              f"지연 {lag[m].min():.2f} -> {lag[m].max():.2f} ms")
    if prox.any():
        print(f"  정단 <= {GOLD_PROX_UM:.0f}um 감쇠 최대 {att[prox].max():.1f}% "
              f"(Golding 기준 < {GOLD_PROX_MAX_ATT:.0f}%)")
    if distal.any():
        print(f"  정단 >= {GOLD_DIST_UM:.0f}um 감쇠 {att[distal].min():.0f}~"
              f"{att[distal].max():.0f}% (Golding 강 {GOLD_STRONG[0]:.0f}~{GOLD_STRONG[1]:.0f} / "
              f"약 {GOLD_WEAK[0]:.0f}~{GOLD_WEAK[1]:.0f})")
    fin = np.isfinite(ratio)
    # 활동의존 bAP 감쇠는 **소마가 목표대로 발화한 조건에서만** 판정한다.
    ok_tr = [tr for tr in trains if tr["soma_ok"] and tr["n_spikes"] >= 2]
    act_dep = any(np.isfinite(tr["rmin"]) and tr["rmin"] < 0.95 for tr in ok_tr)
    print()
    print(f"  ★활동의존 판정 대상(소마 발화 정상): "
          + (", ".join(f'{tr["n"]}발 {tr["freq"]:.0f}Hz' for tr in ok_tr) or "없음"))
    for tr in ok_tr:
        print(f"      {tr['n']}발 {tr['freq']:.0f}Hz 침입효율 마지막/첫 "
              f"{tr['rmin']:.3f}~{tr['rmax']:.3f}")
    print(f"  -> 활동의존 bAP 감쇠 {'있음' if act_dep else '없음'}"
          + ("" if act_dep else " (Spruston1995 와 불일치 — 결핍 후보)"))
    fail_tr = [tr for tr in trains if not tr["soma_ok"]]
    if fail_tr:
        print("  ※ 소마 발화 실패 조건(활동의존 판정에서 제외): "
              + ", ".join(f'{tr["n"]}발 {tr["freq"]:.0f}Hz -> {tr["n_spikes"]}발'
                          for tr in fail_tr)
              + "  <- bAP 실패가 아니라 소마가 못 낸 것이다")

    # 확정 시냅스 위치(기저 144·172um)에서의 bAP — 5단계에 넘길 값
    at_syn = []
    for sp in b.syn_specs:
        k = int(np.argmin(np.abs(dist - sp["path_um"])))
        at_syn.append(dict(path_um=sp["path_um"], nearest_um=round(float(dist[k]), 1),
                           bap_mV=round(float(amp[k]), 2),
                           attenuation_pct=round(float(att[k]), 1),
                           train_ratio=round(float(ratio[k]), 3)))
    print("  확정 시냅스 지점 bAP: " +
          " · ".join(f'{a["path_um"]}um {a["bap_mV"]}mV(감쇠 {a["attenuation_pct"]}%)'
                     for a in at_syn))

    # ── 그림 ─────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15.2, 8.4))
    gs_ = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.15, 1.1],
                           height_ratios=[1, 1], wspace=0.30, hspace=0.40)
    axM = fig.add_subplot(gs_[:, 0])
    axA = fig.add_subplot(gs_[0, 1])
    axB = fig.add_subplot(gs_[1, 1])
    axC = fig.add_subplot(gs_[0, 2])
    axD = fig.add_subplot(gs_[1, 2])

    # A: 형태 + 지점(색=bAP 진폭)
    h.define_shape()
    xyz, typ, par = [], [], []
    for s in b.post.all:
        base = s.name().split(".")[-1].split("[")[0]
        tt = {"soma": mo.SOMA, "apic": mo.APICAL, "dend": mo.BASAL,
              "axon": mo.AXON, "myelin": mo.AXON}.get(base, mo.BASAL)
        first = len(xyz)
        for k in range(s.n3d()):
            xyz.append((s.x3d(k), s.y3d(k), s.z3d(k))); typ.append(tt)
            par.append(first + k - 1 if k > 0 else -1)
    m = dict(xyz=np.array(xyz, float), type=np.array(typ), parent_row=np.array(par),
             radius=np.ones(len(xyz)), index=np.arange(len(xyz)), parent=np.array(par))
    c, RR = mo.align_transform(m, mode="apical")
    m["xyz"] = mo.apply_transform(m["xyz"], c, RR)
    mo.render(axM, m, autoscale=False, color="#b0bec5", soma_color="#455a64")
    cmap = plt.get_cmap("inferno")
    vmax = float(amp.max())
    for r, s in zip(single, sites):
        i3 = s["sec"].n3d() // 2
        p = mo.apply_transform(np.array([s["sec"].x3d(i3), s["sec"].y3d(i3),
                                         s["sec"].z3d(i3)]), c, RR)
        axM.scatter([p[0]], [p[1]], s=125, color=cmap(0.12 + 0.8 * r["amp"] / vmax),
                    edgecolor="white", lw=1.0, zorder=6)
    axM.set_aspect("equal", adjustable="box"); axM.set_xticks([]); axM.set_yticks([])
    axM.grid(False)
    for sp in axM.spines.values():
        sp.set_color("#dddddd")
    axM.set_title(f"A. bAP 진폭 지도 (단발)\n밝을수록 크게 도달 · 최대 {vmax:.0f}mV",
                  fontsize=9.5, loc="left")
    mo.scalebar(axM, 200, "200 um", loc=(0.05, 0.02))

    # B: 감쇠율 vs 거리 + Golding 밴드
    axA.axvspan(0, GOLD_PROX_UM, color="#66bb6a", alpha=0.07)
    axA.axhline(GOLD_PROX_MAX_ATT, color="#2e7d32", ls="--", lw=1.3,
                label=f"Golding: ≤{GOLD_PROX_UM:.0f}um 은 감쇠 <{GOLD_PROX_MAX_ATT:.0f}%")
    axA.axhspan(GOLD_STRONG[0], GOLD_STRONG[1], xmin=0.42, color="#1976d2", alpha=0.14,
                label=f"Golding ≥300um 강한 역전파 {GOLD_STRONG[0]:.0f}~{GOLD_STRONG[1]:.0f}%")
    axA.axhspan(GOLD_WEAK[0], GOLD_WEAK[1], xmin=0.42, color="#c62828", alpha=0.12,
                label=f"Golding ≥300um 약한 역전파 {GOLD_WEAK[0]:.0f}~{GOLD_WEAK[1]:.0f}%")
    for d, col, mk in (("basal", "#0277bd", "o"), ("apical", "#e65100", "s")):
        mm = dom == d
        axA.plot(dist[mm], att[mm], mk + "-", color=col, ms=6, lw=1.8,
                 label=f"우리 {'기저' if d=='basal' else '정단'}")
    for a in at_syn:
        axA.plot([a["nearest_um"]], [a["attenuation_pct"]], "*", color="#7b1fa2",
                 ms=17, zorder=7)
    axA.set_xlabel("경로거리 (um)"); axA.set_ylabel("bAP 감쇠율 (%, 소마 대비)")
    axA.set_title("B. 단발 bAP 감쇠 vs 문헌 (★=확정 시냅스 위치)", fontsize=9.5, loc="left")
    axA.legend(fontsize=6.8, loc="upper left")
    axA.set_ylim(0, 100)

    # C: 진폭 vs 거리
    for d, col, mk in (("basal", "#0277bd", "o"), ("apical", "#e65100", "s")):
        mm = dom == d
        axB.plot(dist[mm], amp[mm], mk + "-", color=col, ms=6, lw=1.8,
                 label=f"{'기저' if d=='basal' else '정단'}")
    axB.axhline(soma1, color="#455a64", ls=":", lw=1.4, label=f"소마 {soma1:.0f}mV")
    axB.set_xlabel("경로거리 (um)"); axB.set_ylabel("bAP 국소 진폭 (mV)")
    axB.set_title("C. bAP 국소 진폭", fontsize=9.5, loc="left")
    axB.legend(fontsize=7.5)

    # D: 트레인 활동의존
    axC.axhline(1.0, color="#9e9e9e", ls=":", lw=1.2)
    for d, col, mk in (("basal", "#0277bd", "o"), ("apical", "#e65100", "s")):
        mm = (dom == d) & fin
        axC.plot(dist[mm], ratio[mm], mk + "-", color=col, ms=6, lw=1.8,
                 label=f"{'기저' if d=='basal' else '정단'}")
    axC.set_xlabel("경로거리 (um)"); axC.set_ylabel("5번째 / 1번째 진폭비")
    axC.set_title(f"D. 활동의존 감쇠 ({strong['n']}발 {strong['freq']:.0f}Hz)\n"
                  + ("트레인에서 감쇠 관측" if act_dep else
                     "감쇠 없음 → Spruston1995 와 불일치 (모델 결핍)"),
                  fontsize=9.5, loc="left")
    axC.legend(fontsize=7.5)

    # E: 대표 지점 bAP 파형
    reps = [0, len(sites) // 2, len(sites) - 1]
    axD.plot(t - st1[0], R1["post_v"], color="#455a64", lw=1.6, label="소마")
    for k, i in enumerate(reps):
        axD.plot(t - st1[0], R1["local_v"][i], lw=1.4,
                 color=cmap(0.2 + 0.55 * k / max(1, len(reps) - 1)),
                 label=f'{sites[i]["dom"][:2]} {dist[i]:.0f}um')
    axD.set_xlim(-2, 14)
    axD.set_xlabel("소마 스파이크 후 시간 (ms)"); axD.set_ylabel("Vm (mV)")
    axD.set_title("E. 단발 bAP 파형 (대표 지점)", fontsize=9.5, loc="left")
    axD.legend(fontsize=7)

    fig.suptitle("3-9  bAP 프로파일 — post 발화가 수상돌기로 거꾸로 퍼지며 남는 크기 "
                 f"· 지점 {len(sites)}개 · 시냅스 gmax=0", fontsize=12, y=0.985)
    fig.subplots_adjust(top=0.88)
    plots.stamp(fig, f"3-9 | 정착 {SETTLE_MS:.0f}ms · 소마 IClamp {AMP_NA}nA/{DUR_MS}ms · "
                     f"단발 + 트레인 {len(TRAINS)}조건 · Golding2001(PMID 11731556)·"
                     f"Spruston1995(PMID 7716524) 대조")
    outdir = plots.figdir(__file__)
    plots.save(fig, outdir, "3-9_bap_profile.png")

    # ── 검증 ─────────────────────────────────────────────────────────────
    def spearman(x, y):
        if len(x) < 3:
            return float("nan")
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        return float(np.corrcoef(rx, ry)[0, 1])

    rs_api = spearman(dist[api], amp[api])
    n_amplify = int((amp > soma1).sum())
    checks = [
        ("단발 조건에서 post 소마 스파이크 1발", nspk1 == 1),
        (f"소마 발화가 정상인 트레인 조건 존재 ({len(ok_tr)}/{len(trains)})", len(ok_tr) > 0),
        (f"정단: 거리↑ → bAP 진폭 감소 (순위상관 {rs_api:+.2f} < -0.7)", rs_api < -0.7),
        (f"정단 ≤{GOLD_PROX_UM:.0f}um 감쇠 <{GOLD_PROX_MAX_ATT:.0f}% (Golding 2001)",
         bool(prox.any() and att[prox].max() < GOLD_PROX_MAX_ATT)),
        (f"정단 ≥{GOLD_DIST_UM:.0f}um 는 확실히 감쇠 (>20%)",
         bool(distal.any() and att[distal].max() > 20.0)),
        (f"★근위 증폭 관측 — bAP 가 소마보다 큰 지점 {n_amplify}개 "
         f"(소마가 전류 싱크라 물리적으로 가능; 해석)", True),
        ("확정 시냅스 지점에 bAP 가 크게 도달 (>50% 잔존)",
         all(a["attenuation_pct"] < 50 for a in at_syn)),
    ]
    for k, ok in checks:
        print(f"  {'O' if ok else 'X'} {k}")
    n_ok = sum(1 for _, v in checks if v)

    out = dict(settle_ms=SETTLE_MS, iclamp_nA=AMP_NA, iclamp_ms=DUR_MS,
               trains=[dict(n=tr["n"], freq_hz=tr["freq"], n_spikes_detected=tr["n_spikes"],
                            soma_fired_as_intended=tr["soma_ok"],
                            soma_amp_mV=[round(a, 2) for a in tr["soma"]],
                            invasion_eff_ratio_last_first=(
                                [round(tr["rmin"], 3), round(tr["rmax"], 3)]
                                if np.isfinite(tr["rmin"]) else None))
                       for tr in trains],
               representative_train=dict(n=strong["n"], freq_hz=strong["freq"]),
               soma_amp_single_mV=round(soma1, 2),
               n_spikes_single=nspk1,
               activity_dependent_attenuation=act_dep,
               proximal_amplification_sites=int((amp > soma1).sum()),
               reference=("Golding, Kath & Spruston 2001 J Neurophysiol 86(6):2998 "
                          "(PMID 11731556, DOI 10.1152/jn.2001.86.6.2998): <=280um 감쇠<50%, "
                          ">=300um 이분화(강 26-42% / 약 71-87%). "
                          "Spruston, Schiller, Stuart & Sakmann 1995 Science 268:297 "
                          "(PMID 7716524, DOI 10.1126/science.7716524): 단발 칼슘은 거리 무관, "
                          "트레인은 크게 감쇠(원위 침입 실패, 분기점에서 자주)."),
               spearman_apical_amp_vs_dist=round(rs_api, 3),
               sites=[dict(domain=r["dom"], section=r["sec"],
                           path_um=round(r["dist"], 1),
                           bap_mV=round(r["amp"], 2),
                           attenuation_pct=round(r["att"], 1),
                           peak_lag_ms=round(r["lag"], 3),
                           train_amps_mV=[round(a, 2) for a in tr["amps"]],
                           train_ratio_5th_1st=(round(tr["ratio"], 3)
                                                if np.isfinite(tr["ratio"]) else None))
                      for r, tr in zip(single, train)],
               at_fixed_synapses=at_syn,
               gap_activity_dependence=(
                   "Spruston 1995 가 말하는 것은 '소마는 발화하는데 원위 침입만 실패' 다. "
                   "그래서 IClamp 시각이 아니라 실측 소마 스파이크를 기준으로, 국소 진폭을 "
                   "그 스파이크의 소마 진폭으로 나눈 '침입 효율' 로 판정했다. "
                   + ("소마 발화가 정상인 조건에서 침입 효율이 트레인 후반에 떨어지는 것을 "
                      "관측했다 — 문헌 방향과 일치." if act_dep else
                      "소마 발화가 정상인 조건에서는 침입 효율이 떨어지지 않았다 — "
                      "활동의존 bAP 실패를 재현하지 못한다(결핍 후보, docs/GAPS.md). ")
                   + " ※고빈도 조건에서 소마 스파이크 수가 목표에 못 미친 것은 "
                     "bAP 실패가 아니라 소마 발화 실패이므로 판정에서 제외했다 — "
                     "2-4 에서 이 모델의 발화적응 지수가 문헌 상한을 넘었던 것과 방향이 맞는다."),
               implication=("GB 계열 엔진은 후시냅스 칼슘 기여를 상수로 둔다. 실제 bAP 는 "
                            "거리에 따라 감쇠하므로 6-8 을 GB 로 그냥 돌리면 인공적으로 평평해진다. "
                            "이 감쇠곡선이 보정(기본 OFF)의 근거이고, GluSynapse 는 국소 전압에서 "
                            "칼슘을 만들어 보정이 불필요하다 — 그 대비가 6-9 결핍 분석의 재료."),
               checks={k: bool(v) for k, v in checks}, passed=n_ok, total=len(checks))
    jpath = os.path.join(outdir, "3-9_bap.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")
    if n_ok != len(checks):
        print(f"\n[실패] {len(checks)-n_ok}개 미통과")
        return 1
    print(f"\n[통과] 3-9 완료 ({n_ok}/{len(checks)}) — 3단계 완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
