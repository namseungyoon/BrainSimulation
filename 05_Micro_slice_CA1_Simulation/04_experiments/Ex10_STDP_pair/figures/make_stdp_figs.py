# -*- coding: utf-8 -*-
"""나-1-② STDP 실측 결과 그림 — scratch/stdp_pair_*.json(run_stdp_all.sh 산출)을 읽어 실제 곡선 3종 생성.
가-15 타이밍곡선 · 가-16 burst수 의존 · 가-17 주파수 의존. 모식도(*_expected.png)를 이 실측본으로 교체.
matplotlib만 사용(NEURON 불필요). 결과 파일 없으면 해당 그림은 건너뜀."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for fam in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
    try:
        matplotlib.rcParams["font.family"] = fam; break
    except Exception:
        pass
matplotlib.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))  # 05_Micro_slice_CA1_Simulation
SCR = os.path.join(ROOT, "scratch")
C0, C1 = "#1f77b4", "#d62728"   # ca_stp=0 원본, ca_stp=1 확장
LBL = {0: "ca_stp=0  원본(Graupner-Brunel)", 1: "ca_stp=1  확장(확률방출 연동)"}


def load(tag):
    p = os.path.join(SCR, "stdp_pair" + tag + ".json")
    if not os.path.isfile(p):
        print("  (없음)", os.path.basename(p)); return None
    return json.load(open(p, encoding="utf-8"))


def by_ca(res, cs):
    return sorted([r for r in res if r.get("ca_stp") == cs], key=lambda r: r["dt"])


def fig_timing():
    d = load("_timing")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.axhline(0, color="#888", lw=0.8); ax.axvline(0, color="#888", lw=0.8, ls=":")
    for cs, col in [(0, C0), (1, C1)]:
        rr = by_ca(d["res"], cs)
        if not rr:
            continue
        x = [r["dt"] for r in rr]; y = [r["drho"] for r in rr]
        ax.plot(x, y, "-o", color=col, lw=2.2, ms=5, label=LBL[cs], ls=("-" if cs == 0 else "--"))
    ax.set_xlabel("스파이크 시간차 Δt (ms)   [pre 기준, +면 pre 먼저]")
    ax.set_ylabel("시냅스 효능 변화 Δrho")
    ax.set_title(f"가-15. STDP 타이밍 곡선 (실측 · gid {d.get('gid','?')} · {d.get('npair','?')}짝)", fontsize=11)
    ax.legend(loc="best", fontsize=8.5)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "stdp_curve.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    print("  저장 stdp_curve.png")


def fig_burstnum():
    ds = [(n, load("_burst%d" % n)) for n in (1, 2, 4)]
    ds = [(n, d) for n, d in ds if d]
    if not ds:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.axhline(0, color="#888", lw=0.8)
    w = 0.34
    for cs, col, off in [(0, C0, -w / 2), (1, C1, w / 2)]:
        ns, ys = [], []
        for n, d in ds:
            rr = [r for r in d["res"] if r.get("ca_stp") == cs]
            if rr:
                ns.append(n); ys.append(rr[0]["drho"])
        if ns:
            ax.bar(np.array(ns) + off, ys, w, color=col, label=LBL[cs])
    ax.set_xticks([n for n, _ in ds]); ax.set_xticklabels(["%d발" % n for n, _ in ds])
    ax.set_xlabel("후시냅스 버스트 발수 (Δt=+10 ms 고정)"); ax.set_ylabel("시냅스 효능 변화 Δrho")
    gid = ds[0][1].get("gid", "?")
    ax.set_title(f"가-16. post-burst 수 의존 (실측 · Wittenberg 재현 · gid {gid})", fontsize=10.5)
    ax.legend(loc="best", fontsize=8.5)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "stdp_burstnum.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    print("  저장 stdp_burstnum.png")


def fig_freq():
    ivs = [1000, 200, 50, 20]
    ds = [(1000.0 / iv, load("_freq%d" % iv)) for iv in ivs]
    ds = [(f, d) for f, d in ds if d]
    if not ds:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.axhline(0, color="#888", lw=0.8)
    for cs, col, mk in [(0, C0, "o"), (1, C1, "s")]:
        fs, ys = [], []
        for f, d in ds:
            rr = [r for r in d["res"] if r.get("ca_stp") == cs]
            if rr:
                fs.append(f); ys.append(rr[0]["drho"])
        if fs:
            ax.plot(fs, ys, "-" + mk if cs == 0 else "--" + mk, color=col, lw=2.2, label=LBL[cs])
    ax.set_xscale("log"); xs = sorted({f for f, _ in ds})
    ax.set_xticks(xs); ax.set_xticklabels([("%g" % x) for x in xs])
    ax.set_xlabel("유도 짝짓기 반복 빈도 (Hz)"); ax.set_ylabel("시냅스 효능 변화 Δrho")
    gid = ds[0][1].get("gid", "?")
    ax.set_title(f"가-17. pairing 주파수 의존 (실측 · Sjöström 재현 · gid {gid})", fontsize=10.5)
    ax.legend(loc="best", fontsize=8.5)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "stdp_freq.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    print("  저장 stdp_freq.png")


if __name__ == "__main__":
    print("[make_stdp_figs] scratch 읽기:", SCR)
    fig_timing(); fig_burstnum(); fig_freq()
    print("[make_stdp_figs] 완료")
