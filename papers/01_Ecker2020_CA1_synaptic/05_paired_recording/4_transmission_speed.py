"""
4_transmission_speed.py — 전달속도(수상돌기 전파 지연) 테스트
============================================================================
Source: Ecker(2020) 단일세포 수상돌기 특성(신호 전달). paired recording 의
        "전달속도(단일세포 수상돌기 특성) 실험" 부분.

흥분성 시냅스(PC->PC AMPA+NMDA)를 **PC 정점수상돌기의 소마경로거리별**로 하나씩
배치하고 동일한 단일 입력을 주어, **소마에서 측정한 PSP** 의
    (1) 개시지연 latency   (5% 도달 시각)
    (2) 정점도달시간 peak-time
    (3) 상승시간 rise      (20→80%)
를 거리의 함수로 잰다. 지연-거리 기울기의 역수가 **유효 전달속도(µm/ms)** 이며,
진폭은 거리에 따라 **감쇠**(케이블 필터링)한다.

확률 방출의 타이밍 잡음을 없애기 위해 **Use=1.0(확정 방출)** 로 두어 순수
"전파+필터링" 효과만 분리한다. EMS mod = cvode 비호환 → 고정 dt.

함수 흐름:
  load_post(PC) → dist_targeted_segs(거리타깃) → [거리별] 시냅스 1개 배치·구동·실행
  → _measure(소마 PSP 특징) → 선형적합(전파속도) → 그림

실행:
  <ca1sim python> .../05_paired_recording/4_transmission_speed.py          # 저장만
  <ca1sim python> .../05_paired_recording/4_transmission_speed.py --show   # 창 띄움
"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(THIS)
ROOT = os.path.dirname(os.path.dirname(PAPER))
SHARED = os.path.join(ROOT, "shared")
sys.path.insert(0, SHARED)
sys.path.insert(0, os.path.join(PAPER, "03_synapses"))
sys.path.insert(0, THIS)

SHOW = "--show" in sys.argv
import matplotlib
if not SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.collections import LineCollection     # noqa: E402

from common.plotstyle import set_korean_font          # noqa: E402
import params_table3 as P3                             # noqa: E402
import paired_experiment as pe                         # noqa: E402
from paired_experiment import h                        # noqa: E402

set_korean_font()

OUT = os.path.join(THIS, "figures")
CLASS = "PC->PC (E2)"                  # 전파 측정용 흥분성 시냅스(정점 AMPA+NMDA)
TARGETS = [25.0, 50.0, 100.0, 150.0, 200.0, 300.0]    # 목표 소마경로거리(µm)


# ─────────────────────────── 형태 그리기 보조 ───────────────────────────
def morph_lines(cell):
    """cell.all 의 3D 점들을 (xs, ys, kind) 선분 목록으로. kind∈{soma,apic,dend,axon}."""
    if sum(int(h.n3d(sec=s)) for s in cell.all) < 4:
        h.define_shape()
    lines = []
    for s in cell.all:
        n = int(h.n3d(sec=s))
        if n < 2:
            continue
        xs = [h.x3d(i, sec=s) for i in range(n)]
        ys = [h.y3d(i, sec=s) for i in range(n)]
        nm = s.name()
        if "soma" in nm:
            kind = "soma"
        elif ".apic" in nm:
            kind = "apic"
        elif "axon" in nm:
            kind = "axon"
        else:
            kind = "dend"
        lines.append((xs, ys, kind))
    return lines


def seg_xy(seg):
    """seg 가 속한 구획의 중간 3D 점 (x, y)."""
    s = seg.sec
    n = int(h.n3d(sec=s))
    if n < 1:
        return None
    i = n // 2
    return h.x3d(i, sec=s), h.y3d(i, sec=s)


# ─────────────────────────── PSP 특징 측정 ───────────────────────────
def _measure(t, v, t_spike):
    """소마 트레이스에서 PSP (amp, peak-time, latency, rise) 측정 (흥분성, 기준선 대비)."""
    i0 = np.searchsorted(t, t_spike - 1.0)
    base = float(v[np.searchsorted(t, t_spike - 3.0):i0].mean())
    tseg = t[i0:] - t_spike
    defl = v[i0:] - base
    pki = int(defl.argmax())
    amp = float(defl[pki])
    ptime = float(tseg[pki])
    up = defl[:pki + 1]
    c05 = np.where(up >= 0.05 * amp)[0]
    c20 = np.where(up >= 0.20 * amp)[0]
    c80 = np.where(up >= 0.80 * amp)[0]
    lat = float(tseg[c05[0]]) if len(c05) else np.nan
    rise = float(tseg[c80[0]] - tseg[c20[0]]) if (len(c20) and len(c80)) else np.nan
    return dict(amp=amp, ptime=ptime, lat=lat, rise=rise, t=tseg, v=defl)


# ─────────────────────────── 거리별 시뮬레이션 ───────────────────────────
def run_distance_scan():
    """PC 후세포 + 거리별 단일 흥분성 시냅스 → 소마 PSP 특징 측정."""
    p = dict(P3.CLASSES[CLASS])                       # 동역학은 PC->PC, 방출만 확정으로 덮어씀
    cell, tname = pe.load_post("PC")
    segs = pe.dist_targeted_segs(cell, TARGETS, region="apical")

    # 같은 구획에 중복 매칭된 타깃 제거(실제 거리 기준 정렬·유일화)
    uniq, seen = [], set()
    for d, seg in sorted(segs, key=lambda x: x[0]):
        key = round(d, 1)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((d, seg))

    tvec = h.Vector().record(h._ref_t)
    vsoma = h.Vector().record(cell.soma[0](0.5)._ref_v)
    h.dt = 0.025                                       # 역치하 EPSP — 미세 dt로 지연 분해(EMS 안정)
    h.celsius = 34.0

    rows, keep = [], []
    for d, seg in uniq:
        syn = h.ProbAMPANMDA_EMS(seg)
        syn.tau_r_AMPA = p.get("tau_r_AMPA", 0.2)
        syn.tau_d_AMPA = p["tau_d_AMPA"]
        syn.NMDA_ratio = p["NMDA_ratio"]
        syn.Use = 1.0                                  # 확정 방출(타이밍 잡음 제거)
        syn.Dep = p["Dep"]; syn.Fac = p["Fac"]; syn.Nrrp = 1
        vstim = h.VecStim(); tv = h.Vector([pe.T_SPIKE]); vstim.play(tv)
        nc = h.NetCon(vstim, syn); nc.weight[0] = p["g_nS"]; nc.delay = 0.0
        vloc = h.Vector().record(seg._ref_v)
        syn.setRNG(7, 1, 1)
        h.finitialize(pe.V_HOLD); h.continuerun(pe.T_SPIKE + 120.0)

        t = np.array(tvec); v = np.array(vsoma); vl = np.array(vloc)
        m = _measure(t, v, pe.T_SPIKE)
        i0 = np.searchsorted(t, pe.T_SPIKE - 1.0)
        lbase = float(vl[np.searchsorted(t, pe.T_SPIKE - 3.0):i0].mean())
        m["lamp"] = float((vl[i0:] - lbase).max())     # 국소(시냅스 부위) 진폭
        m["d"] = d
        m["xy"] = seg_xy(seg)
        rows.append(m)
        keep += [syn, vstim, tv, nc, vloc]

    return cell, tname, rows, keep


# ─────────────────────────── 선형적합(전파속도) ───────────────────────────
def fit_speed(dists, delays):
    """지연(ms) vs 거리(µm) 선형적합 → (기울기 ms/µm, 속도 µm/ms, 절편)."""
    d = np.asarray(dists, float)
    y = np.asarray(delays, float)
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(d[ok], y[ok], 1)
    speed = (1.0 / slope) if slope > 1e-9 else np.nan
    return slope, speed, intercept


# ─────────────────────────── 그림 + main ───────────────────────────
def plot(cell, tname, rows, show):
    dists = [r["d"] for r in rows]
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(min(dists), max(dists))
    col = {r["d"]: cmap(norm(r["d"])) for r in rows}

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(f"전달속도 — 수상돌기 전파 지연 (PC 정점, {CLASS} 시냅스)  ·  후세포 {tname}",
                 fontsize=14, fontweight="bold")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.25, 1.25], hspace=0.30, wspace=0.32)

    # (A) 형태 + 거리별 시냅스 위치 ---------------------------------------
    axm = fig.add_subplot(gs[:, 0])
    ck = {"soma": "k", "apic": "0.55", "dend": "0.78", "axon": "0.9"}
    segs_lc = {"soma": [], "apic": [], "dend": [], "axon": []}
    for xs, ys, kind in morph_lines(cell):
        segs_lc[kind].append(np.column_stack([xs, ys]))
    for kind in ("axon", "dend", "apic", "soma"):
        if segs_lc[kind]:
            axm.add_collection(LineCollection(segs_lc[kind], colors=ck[kind],
                                              linewidths=1.6 if kind == "soma" else 0.6, zorder=1))
    sx, sy = (h.x3d(0, sec=cell.soma[0]), h.y3d(0, sec=cell.soma[0]))
    axm.plot(sx, sy, "v", color="navy", ms=11, zorder=5)
    axm.annotate("기록(소마)", (sx, sy), textcoords="offset points", xytext=(8, -2),
                 fontsize=8, color="navy")
    for r in rows:
        if r["xy"]:
            axm.plot(*r["xy"], "o", color=col[r["d"]], ms=9, mec="k", mew=0.6, zorder=6)
            axm.annotate(f"{r['d']:.0f}", r["xy"], textcoords="offset points",
                         xytext=(6, 4), fontsize=7.5, color="k")
    axm.set_title("(A) 시냅스 위치 (색=소마경로거리 µm)", fontsize=10)
    axm.set_aspect("equal"); axm.axis("off"); axm.autoscale()

    # (B) 소마 PSP 트레이스 (거리별) --------------------------------------
    axb = fig.add_subplot(gs[0, 1])
    for r in rows:
        axb.plot(r["t"], r["v"], color=col[r["d"]], lw=1.7,
                 label=f"{r['d']:.0f} µm")
    axb.axvline(0, color="0.6", ls=":", lw=1)
    axb.set_xlim(-2, 60); axb.set_xlabel("시냅스전 스파이크 이후 시간 (ms)")
    axb.set_ylabel("소마 PSP (mV)")
    axb.set_title("(B) 소마 EPSP — 멀수록 늦고 작고 느림", fontsize=10)
    axb.legend(fontsize=7, title="거리", ncol=2, loc="upper right")

    # (C) 지연 vs 거리 + 전파속도 적합 ------------------------------------
    axc = fig.add_subplot(gs[1, 1])
    lat = [r["lat"] for r in rows]; pt = [r["ptime"] for r in rows]
    axc.scatter(dists, lat, c=[col[d] for d in dists], s=55, ec="k", lw=0.6, zorder=4, label="개시지연")
    axc.scatter(dists, pt, c=[col[d] for d in dists], s=55, marker="s", ec="k", lw=0.6,
                zorder=4, label="정점도달")
    s_lat, v_lat, b_lat = fit_speed(dists, lat)
    s_pt, v_pt, b_pt = fit_speed(dists, pt)
    xx = np.linspace(0, max(dists) * 1.05, 50)
    if np.isfinite(s_lat):
        axc.plot(xx, s_lat * xx + b_lat, color="navy", lw=1.4, ls="--")
    if np.isfinite(s_pt):
        axc.plot(xx, s_pt * xx + b_pt, color="firebrick", lw=1.4, ls="--")
    txt = ["[전달속도 = 지연-거리 기울기의 역수]"]
    if np.isfinite(v_pt):
        txt.append(f"정점도달  ~ {v_pt:.0f} µm/ms  ({s_pt*1000:.1f} µs/µm)")
    if np.isfinite(v_lat):
        txt.append(f"개시지연  ~ {v_lat:.0f} µm/ms")
    axc.text(0.03, 0.97, "\n".join(txt), transform=axc.transAxes, va="top", fontsize=8.5,
             bbox=dict(fc="#FFF6D5", ec="0.6", alpha=0.95))
    axc.set_xlabel("소마경로거리 (µm)"); axc.set_ylabel("지연 (ms)")
    axc.set_title("(C) 전달속도 — 정점도달이 거리에 비례해 늦어짐", fontsize=10)
    axc.legend(fontsize=8, loc="lower right")

    # (D) 국소 vs 소마 진폭 (감쇠 = 둘의 간극) + (E) 상승시간 --------------
    sub = gs[:, 2].subgridspec(2, 1, hspace=0.34)
    axd = fig.add_subplot(sub[0])
    lamp = [r["lamp"] for r in rows]; samp = [r["amp"] for r in rows]
    axd.plot(dists, lamp, "-o", color="#C44E52", lw=1.4, ms=6, label="국소(시냅스 부위)")
    axd.plot(dists, samp, "-s", color="navy", lw=1.4, ms=6, label="소마")
    axd.set_yscale("log")
    axd.set_xlabel("소마경로거리 (µm)"); axd.set_ylabel("EPSP 진폭 (mV, log)")
    axd.set_title("(D) 감쇠 — 원위로 갈수록 국소↑·소마와 간극↑", fontsize=10)
    axd.legend(fontsize=7.5, loc="upper left")
    axd.text(0.97, 0.05, "원위부 고임피던스+NMDA →\n국소 큰 탈분극(소마는 감쇠)",
             transform=axd.transAxes, ha="right", va="bottom", fontsize=7,
             color="0.3", bbox=dict(fc="white", ec="0.8", alpha=0.85))

    axr = fig.add_subplot(sub[1])
    axr.scatter(dists, [r["rise"] for r in rows], c=[col[d] for d in dists],
                s=55, ec="k", lw=0.6, zorder=4)
    axr.plot(dists, [r["rise"] for r in rows], color="0.5", lw=1, zorder=2)
    axr.set_xlabel("소마경로거리 (µm)"); axr.set_ylabel("상승시간 20→80% (ms)")
    axr.set_title("(E) 상승시간 — 멀수록 느려짐(케이블 필터링)", fontsize=10)

    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, "4_transmission_speed.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[그림] {out}", flush=True)
    if show:
        plt.show()


def main():
    cell, tname, rows, keep = run_distance_scan()
    print("\n  거리(um)  진폭(mV)  개시지연(ms)  정점도달(ms)  상승(ms)  국소진폭(mV)", flush=True)
    for r in rows:
        print(f"  {r['d']:7.0f}  {r['amp']:7.3f}  {r['lat']:10.3f}  "
              f"{r['ptime']:10.3f}  {r['rise']:7.3f}  {r['lamp']:9.2f}", flush=True)
    _, v_lat, _ = fit_speed([r["d"] for r in rows], [r["lat"] for r in rows])
    _, v_pt, _ = fit_speed([r["d"] for r in rows], [r["ptime"] for r in rows])
    print(f"\n  [전파속도] 개시지연 ~ {v_lat:.0f} um/ms,  정점도달 ~ {v_pt:.0f} um/ms", flush=True)
    plot(cell, tname, rows, SHOW)


if __name__ == "__main__":
    main()
