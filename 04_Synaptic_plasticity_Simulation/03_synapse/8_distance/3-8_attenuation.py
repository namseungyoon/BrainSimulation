# -*- coding: utf-8 -*-
"""3-8 거리별 감쇠·전달 지연 — 시냅스 위치를 바꾸며 soma EPSP 가 얼마나 줄고 늦는지

단계   : 3-8 (파이프라인 3단계 시냅스 / 하위 8 distance)
쉬운 설명: 같은 세기의 시냅스를 소마에서 가까운 곳/먼 곳에 놓아 본다. 먼 곳에 놓으면 신호가
          수상돌기를 타고 오며 작아지고 늦어진다(수동 케이블 필터). 얼마나인지 잰다.
방법   : post 세포의 기저·정단 수상돌기에서 경로거리를 훑는 지점들을 고르고, 각 지점에
          시냅스를 하나만 켠다(나머지는 gmax=0). g 는 확정값(3-7)을 그대로.
          NetCon 지연은 모든 지점에 동일한 상수 -> 측정된 지연 차이는 순수 수상돌기 케이블 지연.
          정착(D12) 스냅샷 1회 + 지점마다 복원.
★비교  : Magee & Cook 2000 Nat Neurosci 3(9):895 (PMID 10966620) — CA1 정단 수상돌기에서
          시냅스 전도도가 거리에 따라 점진적으로 커져 **소마 EPSP 진폭이 위치에 거의 무관**해진다
          (수상돌기 국소 EPSP 는 멀수록 커지고, 그것이 케이블 필터를 상쇄).
          우리 모델은 g 가 위치와 무관한 단일값이므로 **그 보정이 없다** -> 감쇠가 그대로 보인다.
          그래서 이 단계는 "각 거리에서 소마 EPSP 를 균일하게 만들려면 g 가 얼마여야 하는가"
          (= Magee-Cook 식 보정 곡선)를 함께 산출한다. 6-8(위치 의존 가소성)이 이 값을 쓴다.
          ⚠️ Magee&Cook 은 정단(SC 입력) 결과다. 우리 연결(PC->PC)의 기본 위치는 기저수상돌기이고
             기저의 보정 여부는 그 논문 범위 밖이다 — 외삽임을 명기한다.
★주의  : 3-7/D13 에서 국소 전압이 높으면 기저수상돌기 국소 스파이크가 났다. 원위는 입력저항이
          커서 같은 g 로도 국소 전압이 높아지므로, 지점마다 스파이크 여부를 함께 판정한다.
결과   : figures/3-8_attenuation.png · figures/3-8_attenuation.json
실행   : . .\\env\\activate.ps1 ; & $Py04 03_synapse\\8_distance\\3-8_attenuation.py
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
from lib import measure                       # noqa: E402
from lib import morphology as mo             # noqa: E402
from lib.bench import Bench                   # noqa: E402
from lib.wiring import Wiring, SETTLE_MS      # noqa: E402
from lib.nrnenv import h                     # noqa: E402

T_SPIKE = SETTLE_MS + 10.0
TSTOP = T_SPIKE + 70.0
REC_DT = 0.025
SYN_DELAY = 0.5          # 모든 지점 동일 (지연 차이를 케이블 성분만으로 만들기 위해)
# 국소 스파이크 판정은 lib.measure.is_dendritic_spike (공용, 국소 최고 전압 문턱)
TARGET_BASAL = [20, 50, 80, 110, 140, 170, 200]
TARGET_APICAL = [50, 100, 150, 200, 300, 400, 500, 600, 700]


def pick_sites(post):
    """목표 경로거리에 가장 가까운 구획을 도메인별로 고른다(중복 제거)."""
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
            key = s.name()
            if key in used:
                continue
            used.add(key)
            sites.append(dict(dom=dom, target=tg, dist=float(d), sec=s))
    sites.sort(key=lambda x: (x["dom"], x["dist"]))
    return sites


def main():
    plots.setup()
    print("=== 3-8 거리별 감쇠·전달 지연 ===")
    b = Bench()
    sites = pick_sites(b.post)
    n_bas = sum(1 for s in sites if s["dom"] == "basal")
    print(f"  시험 지점 {len(sites)}개 (기저 {n_bas} · 정단 {len(sites)-n_bas})")

    # 모든 지점에 시냅스를 만들고 한 번에 하나만 켠다 (Wiring 을 여러 번 만들면
    # 이전 시냅스도 같이 발화하는 함정이 있다)
    segs = [(s["sec"](0.5), dict(delay_ms=SYN_DELAY, path_um=s["dist"], domain=s["dom"]))
            for s in sites]
    w = Wiring(b, frozen=True, segs=segs)
    g_nS = float(w.p["g_nS"])
    print(f"  클래스 {w.class_name} · g = {g_nS} nS (3-7 확정값) · 지연 모두 {SYN_DELAY} ms")

    w.drive_pre_iclamp([T_SPIKE], amp_nA=1.2, dur_ms=3.0)
    w.record(rec_dt=REC_DT, local_v=True, currents=False)
    t_event = T_SPIKE + SYN_DELAY

    w.settle()
    rows = []
    for i, s in enumerate(sites):
        w.restore()
        for j, (syn, _) in enumerate(w.syns):
            syn.gmax = (g_nS / 1000.0) if j == i else 0.0     # i 번만 켠다
        w.run_settled(TSTOP)
        R = w.arrays()
        fs = measure.epsp_features(R["t"], R["post_v"], t_event)
        fl = measure.epsp_features(R["t"], R["local_v"][i], t_event)
        vloc_pk = float(R["local_v"][i].max())
        dspike = measure.is_dendritic_spike(vloc_pk)
        rows.append(dict(dom=s["dom"], dist=s["dist"], sec=s["sec"].name().split(".")[-1],
                         soma=fs["amp_mV"], local=fl["amp_mV"], vloc=vloc_pk,
                         lat=fs["latency_ms"], rise=fs["rise_ms"], hw=fs["halfwidth_ms"],
                         dspike=dspike))
        print(f"  {s['dom']:6s} {s['dist']:6.1f}um ({rows[-1]['sec']:>10s}) -> "
              f"soma {fs['amp_mV']:7.4f} mV · 국소 {fl['amp_mV']:7.3f} mV "
              f"(최고 {vloc_pk:7.2f}) · 감쇠 {fs['amp_mV']/max(fl['amp_mV'],1e-9):5.3f} · "
              f"지연 {fs['latency_ms']:5.2f} ms" + ("  <-- 국소 스파이크" if dspike else ""))

    dom = np.array([r["dom"] for r in rows])
    dist = np.array([r["dist"] for r in rows])
    soma = np.array([r["soma"] for r in rows])
    local = np.array([r["local"] for r in rows])
    lat = np.array([r["lat"] for r in rows])
    rise = np.array([r["rise"] for r in rows])
    dsp = np.array([r["dspike"] for r in rows], dtype=bool)
    atten = soma / np.maximum(local, 1e-9)

    # Magee-Cook 식 보정: 각 지점의 soma EPSP 를 '가장 근위 지점' 값으로 맞추는 g
    #   soma EPSP 가 g 에 거의 비례(3-7 수동 구간)하므로 1차 근사로 비례 역산한다.
    ok = ~dsp
    i_ref = int(np.argmin(np.where(ok, dist, np.inf)))     # 스파이크 없는 최근위 지점
    soma_ref = float(soma[i_ref])
    g_needed = np.where(soma > 0, g_nS * soma_ref / np.maximum(soma, 1e-9), np.nan)

    for d in ("basal", "apical"):
        m = (dom == d) & ok
        if m.sum() >= 2:
            print(f"\n  [{d}] soma EPSP {soma[m].max():.4f} -> {soma[m].min():.4f} mV "
                  f"({dist[m].min():.0f} -> {dist[m].max():.0f}um) = "
                  f"{soma[m].min()/soma[m].max():.3f}배로 감쇠")
            print(f"       국소 EPSP {local[m].min():.2f} ~ {local[m].max():.2f} mV "
                  f"(멀수록 국소는 커진다 = 입력저항 증가)")
            print(f"       지연 {lat[m].min():.2f} -> {lat[m].max():.2f} ms")
            print(f"       균일화에 필요한 g: {np.nanmin(g_needed[m]):.2f} ~ "
                  f"{np.nanmax(g_needed[m]):.2f} nS (기준 {dist[i_ref]:.0f}um 의 {soma_ref:.3f}mV)")

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15.0, 5.6))
    gs_ = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.15, 1.05], height_ratios=[1, 1],
                           wspace=0.30, hspace=0.42)
    axM = fig.add_subplot(gs_[:, 0])
    axA = fig.add_subplot(gs_[:, 1])
    axL = fig.add_subplot(gs_[0, 2])
    axR = fig.add_subplot(gs_[1, 2])

    # ---- A: 형태 + 시험 지점 ----
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
    mo.render(axM, m, autoscale=False, color="#90a4ae", soma_color="#455a64")
    for r, s in zip(rows, sites):
        i3 = s["sec"].n3d() // 2
        p = mo.apply_transform(np.array([s["sec"].x3d(i3), s["sec"].y3d(i3), s["sec"].z3d(i3)]),
                               c, RR)
        col = "#c62828" if r["dspike"] else ("#0277bd" if r["dom"] == "basal" else "#e65100")
        axM.scatter([p[0]], [p[1]], s=110, marker="o", color=col,
                    edgecolor="white", lw=1.0, zorder=6)
    axM.set_aspect("equal", adjustable="box"); axM.set_xticks([]); axM.set_yticks([])
    axM.grid(False)
    for sp in axM.spines.values():
        sp.set_color("#dddddd")
    axM.set_title(f"A. 시험 지점 {len(sites)}개\n파랑=기저 · 주황=정단 · 빨강=국소 스파이크",
                  fontsize=9.5, loc="left")
    mo.scalebar(axM, 200, "200 um", loc=(0.05, 0.02))

    # ---- B: soma / 국소 EPSP vs 거리 ----
    for d, col, mk in (("basal", "#0277bd", "o"), ("apical", "#e65100", "s")):
        mm = (dom == d) & ok
        if mm.any():
            axA.plot(dist[mm], soma[mm], mk + "-", color=col, ms=6, lw=1.8,
                     label=f"soma EPSP ({'기저' if d=='basal' else '정단'})")
    if dsp.any():
        axA.plot(dist[dsp], soma[dsp], "x", color="#c62828", ms=10, mew=2,
                 label="국소 스파이크 (제외)")
    axA.set_xlabel("시냅스 경로거리 (um)"); axA.set_ylabel("post 소마 EPSP 진폭 (mV)")
    axA2 = axA.twinx()
    for d, col, mk in (("basal", "#4fc3f7", "o"), ("apical", "#ffb74d", "s")):
        mm = (dom == d) & ok
        if mm.any():
            axA2.plot(dist[mm], local[mm], mk + "--", color=col, ms=4, lw=1.2, alpha=0.9,
                      label=f"국소 EPSP ({'기저' if d=='basal' else '정단'})")
    axA2.set_ylabel("시냅스 국소 EPSP 진폭 (mV)", color="#777")
    axA2.tick_params(axis="y", labelcolor="#777")
    hA, lA = axA.get_legend_handles_labels(); hB, lB = axA2.get_legend_handles_labels()
    axA.legend(hA + hB, lA + lB, fontsize=7.5, loc="upper right")
    axA.set_title("B. 멀어지면 소마 EPSP 는 작아지고 국소 EPSP 는 커진다\n"
                  "(수동 케이블 감쇠 + 원위 입력저항 증가) — g 는 위치와 무관한 단일값",
                  fontsize=9.5, loc="left")

    # ---- C(위): 지연·상승 vs 거리 ----
    axL.plot(dist[ok & (dom == "basal")], lat[ok & (dom == "basal")], "o-",
             color="#0277bd", ms=5, lw=1.5, label="기저 개시지연")
    axL.plot(dist[ok & (dom == "apical")], lat[ok & (dom == "apical")], "s-",
             color="#e65100", ms=5, lw=1.5, label="정단 개시지연")
    axL.plot(dist[ok], rise[ok], ".", color="#7b1fa2", ms=7, alpha=0.7, label="상승시간(20-80%)")
    axL.set_ylabel("ms"); axL.set_xticklabels([])
    axL.set_title(f"C. 케이블 지연 (NetCon 지연은 전 지점 {SYN_DELAY}ms 동일)",
                  fontsize=9.5, loc="left")
    axL.legend(fontsize=7, loc="upper left")

    # ---- C(아래): 균일화에 필요한 g (Magee-Cook 식 보정) ----
    axR.plot(dist[ok & (dom == "basal")], g_needed[ok & (dom == "basal")], "o-",
             color="#0277bd", ms=5, lw=1.5, label="기저")
    axR.plot(dist[ok & (dom == "apical")], g_needed[ok & (dom == "apical")], "s-",
             color="#e65100", ms=5, lw=1.5, label="정단")
    axR.axhline(g_nS, color="#2e7d32", ls="--", lw=1.3, label=f"현재 단일값 {g_nS} nS")
    axR.set_xlabel("시냅스 경로거리 (um)"); axR.set_ylabel("필요 g (nS)")
    axR.set_title("D. 소마 EPSP 를 균일하게 만들려면 필요한 g\n"
                  "(Magee&Cook 2000 이 실제 정단에서 관찰한 보정 — 우리 모델엔 없음)",
                  fontsize=9.5, loc="left")
    axR.legend(fontsize=7, loc="upper left")

    fig.suptitle(f"3-8  거리별 감쇠·전달 지연 — {w.class_name} · g={g_nS}nS 고정 · "
                 f"시험 지점 {len(sites)}개", fontsize=12, y=0.985)
    fig.subplots_adjust(top=0.86)
    plots.stamp(fig, f"3-8 | 정착 {SETTLE_MS:.0f}ms · 지연 상수 {SYN_DELAY}ms(케이블 성분 분리) · "
                     f"기준 {dist[i_ref]:.0f}um {soma_ref:.3f}mV · 국소 스파이크 {int(dsp.sum())}개 제외")
    outdir = plots.figdir(__file__)
    plots.save(fig, outdir, "3-8_attenuation.png")

    # ---- 검증 ----
    def spearman(x, y):
        """순위상관 — 값의 크기 대신 순서만 본다(도메인별 범위 차이에 둔감)."""
        if len(x) < 3:
            return float("nan")
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        return float(np.corrcoef(rx, ry)[0, 1])

    rs = {}
    for d in ("basal", "apical"):
        mm = (dom == d) & ok
        rs[d] = dict(soma=spearman(dist[mm], soma[mm]), local=spearman(dist[mm], local[mm]))
        print(f"  [{d}] 순위상관: soma-거리 {rs[d]['soma']:+.2f} · 국소-거리 {rs[d]['local']:+.2f}")

    # ★ 산포: 같은 경로거리대라도 감쇠가 다르다 (국소 가지 굵기·분지 영향)
    #   경로거리만으로 위치를 파라미터화할 수 없다는 뜻 -> 6-8 에서 주의해야 한다.
    nonmono = {}
    for d in ("basal", "apical"):
        mm = (dom == d) & ok
        o = np.argsort(dist[mm])
        ss = soma[mm][o]
        nonmono[d] = int(np.sum(np.diff(ss) > 0))     # 거리가 늘었는데 EPSP 가 커진 횟수

    def mono_dec(d):
        mm = (dom == d) & ok
        if mm.sum() < 3:
            return True
        o = np.argsort(dist[mm])
        return bool(np.all(np.diff(soma[mm][o]) < 0))

    def mono_inc_lat(d):
        mm = (dom == d) & ok
        if mm.sum() < 3:
            return True
        o = np.argsort(dist[mm])
        return bool(lat[mm][o][-1] > lat[mm][o][0])

    checks = [
        (f"기저: 거리↑ → soma EPSP 감소 경향 (순위상관 {rs['basal']['soma']:+.2f} < -0.7)",
         bool(rs["basal"]["soma"] < -0.7)),
        (f"정단: 거리↑ → soma EPSP 감소 경향 (순위상관 {rs['apical']['soma']:+.2f} < -0.7)",
         bool(rs["apical"]["soma"] < -0.7)),
        ("거리↑ → 개시지연 증가 (케이블 지연)", mono_inc_lat("basal") and mono_inc_lat("apical")),
        (f"기저: 거리↑ → 국소 EPSP 증가 (순위상관 {rs['basal']['local']:+.2f} > 0.5)",
         bool(rs["basal"]["local"] > 0.5)),
        (f"정단: 거리↑ → 국소 EPSP 증가 (순위상관 {rs['apical']['local']:+.2f} > 0.5)",
         bool(rs["apical"]["local"] > 0.5)),
        ("소마/국소 감쇠비 < 1 (전 지점)", bool(np.all(atten[ok] < 1.0))),
        ("확정 g 의 기본 위치(기저)가 스파이크 없음",
         bool(not dsp[(dom == "basal")].any())),
        (f"★산포 확인 — 경로거리만으로 감쇠가 정해지지 않음 "
         f"(역전 기저 {nonmono['basal']}회·정단 {nonmono['apical']}회)",
         (nonmono["basal"] + nonmono["apical"]) > 0),
    ]
    for k, okk in checks:
        print(f"  {'O' if okk else 'X'} {k}")
    n_ok = sum(1 for _, v in checks if v)

    out = dict(cls=w.class_name, g_nS=g_nS, settle_ms=SETTLE_MS, syn_delay_ms=SYN_DELAY,
               reference="Magee & Cook 2000 Nat Neurosci 3(9):895 (PMID 10966620) — "
                         "정단 시냅스는 거리에 따라 g 가 커져 소마 EPSP 를 균일화. "
                         "우리 모델은 단일 g 여서 그 보정이 없다(외삽 주의: 기저는 논문 범위 밖).",
               ref_site_um=round(float(dist[i_ref]), 1), ref_soma_mV=round(soma_ref, 4),
               sites=[dict(domain=r["dom"], section=r["sec"], path_um=round(r["dist"], 1),
                           soma_mV=round(r["soma"], 4), local_mV=round(r["local"], 3),
                           local_peak_mV=round(r["vloc"], 2),
                           attenuation=round(r["soma"] / max(r["local"], 1e-9), 4),
                           latency_ms=round(r["lat"], 3), rise_ms=round(r["rise"], 3),
                           halfwidth_ms=round(r["hw"], 3),
                           g_needed_nS=(round(float(g), 3) if np.isfinite(g) else None),
                           dendritic_spike=r["dspike"])
                      for r, g in zip(rows, g_needed)],
               n_dendritic_spike=int(dsp.sum()),
               dspike_vloc_threshold_mV=measure.DSPIKE_VLOC_MV,
               spearman=dict((d, dict(soma=round(rs[d]["soma"], 3),
                                      local=round(rs[d]["local"], 3)))
                             for d in rs),
               nonmonotonic_steps=nonmono,
               finding_scatter=("같은 경로거리대라도 soma EPSP 가 다르다 — 감쇠는 경로거리보다 "
                                "국소 가지 굵기·분지 구조에 좌우된다. 6-8(위치 의존 가소성)에서 "
                                "위치를 경로거리 하나로 파라미터화하면 안 되고 지점별 실측 감쇠를 쓴다."),
               checks={k: bool(v) for k, v in checks}, passed=n_ok, total=len(checks))
    jpath = os.path.join(outdir, "3-8_attenuation.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")
    if n_ok != len(checks):
        print(f"\n[실패] {len(checks)-n_ok}개 미통과")
        return 1
    print(f"\n[통과] 3-8 완료 ({n_ok}/{len(checks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
