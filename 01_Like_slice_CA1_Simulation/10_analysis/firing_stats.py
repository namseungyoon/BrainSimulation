# -*- coding: utf-8 -*-
"""
10_analysis/firing_stats.py  —  E1: baseline 발화율·E/I 검증 (새 시뮬 불필요)

완주 데이터(FULL_spikes_all.csv + slice_cells.npz)로 유형·층별 발화율을 집계하고
**in vivo CA1 문헌값과 대조**해 우리 baseline이 생리적인지 판정. 구동 과다 여부 진단.

산출: figures/E1_firing_baseline.png + 콘솔 판정
실행: python 10_analysis/firing_stats.py
"""
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
CELLS = os.path.join(ROOT, "05_placement", "slice_cells.npz")
SPK = os.path.join(ROOT, "09_run", "spikes", "FULL_spikes_all.csv")
TSTOP = 1000.0

# in vivo awake rat CA1 문헌 발화율(Hz) 참고 band [저값, 고값] + 대표값
#  PC: Mizuseki&Buzsaki 2013(로그정규, 중앙~0.3, 평균~1Hz) → 성긴 발화
#  INT: 빠른발화 바스켓/축삭 ~10-40Hz, OLM 등 ~5-25Hz
LIT = {
    "PC(추체)": (0.3, 2.0, "Mizuseki&Buzsaki 2013: 평균~1Hz, 로그정규 성긴발화"),
    "INT(인터뉴런)": (8.0, 40.0, "빠른발화 바스켓/축삭 ~10-40Hz, 기타 ~5-25Hz"),
}
ETYPE_TO_T4 = {"cACpyr": "PC", "cNAC": "PV", "cAC": "cAC", "bAC": "bAC"}


def main():
    c = np.load(CELLS, allow_pickle=True)
    etype = c["etype"].astype(str); mtype = c["mtype"].astype(str)
    layer = c["layer"].astype(str); sclass = c["sclass"].astype(str); N = len(etype)

    nspk = np.zeros(N, int); ts_by = {}
    with open(SPK, encoding="utf-8") as f:
        rd = csv.reader(f); next(rd, None)
        for row in rd:
            g = int(row[0]); nspk[g] += 1; ts_by.setdefault(g, []).append(float(row[2]))
    rate = nspk / (TSTOP / 1000.0)

    def stat(mask):
        r = rate[mask]
        return dict(n=int(mask.sum()), mean=float(r.mean()), median=float(np.median(r)),
                    active=float((nspk[mask] > 0).mean()) * 100)

    print("=" * 70)
    print("E1  baseline 발화율 (전체 슬라이스 1초 완주)")
    print("=" * 70)
    exc = sclass == "EXC"; inh = sclass == "INH"
    sE, sI = stat(exc), stat(inh)
    print(f"[흥분(PC)]  n={sE['n']:,}  평균 {sE['mean']:.1f}Hz  중앙 {sE['median']:.1f}Hz  활성 {sE['active']:.0f}%")
    print(f"[억제(INT)] n={sI['n']:,}  평균 {sI['mean']:.1f}Hz  중앙 {sI['median']:.1f}Hz  활성 {sI['active']:.0f}%")
    print("-" * 70)
    print("문헌 대조:")
    pc_lo, pc_hi, pc_ref = LIT["PC(추체)"]
    over = sE["mean"] / ((pc_lo + pc_hi) / 2)
    print(f"  PC 평균 {sE['mean']:.1f}Hz vs in vivo {pc_lo}-{pc_hi}Hz ({pc_ref})")
    print(f"  → PC가 문헌 대표값의 약 {over:.0f}배. {'⚠️ 과활성(구동 과다 의심)' if sE['mean']>pc_hi else 'OK'}")
    int_lo, int_hi, int_ref = LIT["INT(인터뉴런)"]
    print(f"  INT 평균 {sI['mean']:.1f}Hz vs in vivo {int_lo}-{int_hi}Hz → "
          f"{'상한 근처/초과' if sI['mean']>int_hi else '범위 내'}")

    # e-type별 / 층별
    print("-" * 70); print("e-type별 평균 발화율:")
    for tn in ["PC", "PV", "cAC", "bAC"]:
        m = np.array([ETYPE_TO_T4.get(e, "cAC") for e in etype]) == tn
        if m.any():
            print(f"  {tn:4s} n={int(m.sum()):5d}  {rate[m].mean():.1f}Hz")
    print("층별 평균 발화율:")
    for L in ["SO", "SP", "SR", "SLM"]:
        m = layer == L
        if m.any():
            print(f"  {L:4s} n={int(m.sum()):5d}  {rate[m].mean():.1f}Hz")

    # 시간 정상성(50ms bin)
    seg_ms = 50.0; edges = np.arange(0, TSTOP + seg_ms, seg_ms)
    allt = np.concatenate([np.array(v) for v in ts_by.values()])
    h, _ = np.histogram(allt, bins=edges)
    pr = h / N / (seg_ms / 1000.0)
    print("-" * 70)
    print(f"시간 정상성: 집단발화율 세그평균 {pr.mean():.1f}Hz, 표준편차 {pr.std():.1f}Hz "
          f"(seg0={pr[0]:.1f} → 정상상태 {pr[1:].mean():.1f})")

    # ── 그림 ──────────────────────────────────────────────────────────
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17, 5))
    # (A) 유형 발화율 vs 문헌 band
    cats = ["PC(추체)", "INT(인터뉴런)"]; vals = [sE["mean"], sI["mean"]]
    a1.bar(cats, vals, color=["#DD8452", "#4C72B0"])
    for i, (cat, v) in enumerate(zip(cats, vals)):
        lo, hi, _ = LIT[cat]
        a1.axhspan(lo, hi, xmin=i/2+0.05, xmax=(i+1)/2-0.05, color="green", alpha=0.15)
        a1.text(i, v, f" {v:.1f}Hz", ha="center", va="bottom", fontsize=10)
    a1.set_ylabel("평균 발화율 (Hz)")
    a1.set_title("(A) 우리 발화율(막대) vs in vivo 문헌 band(초록)\nPC 과활성 여부 판정")
    # (B) 집단발화율 시간
    a2.plot((edges[:-1]+edges[1:])/2, pr, "o-", color="#333", ms=3)
    a2.set_xlabel("시간 (ms)"); a2.set_ylabel("집단 발화율 (Hz/세포)")
    a2.set_title(f"(B) 시간 정상성 (평균 {pr[1:].mean():.1f}Hz)")
    # (C) PC 발화율 분포(로그)
    pcrate = rate[np.array([ETYPE_TO_T4.get(e, "cAC") for e in etype]) == "PC"]
    a3.hist(pcrate, bins=40, color="#DD8452", alpha=0.85)
    a3.axvspan(pc_lo, pc_hi, color="green", alpha=0.2, label="in vivo 범위")
    a3.axvline(pcrate.mean(), color="red", ls="--", label=f"우리 평균 {pcrate.mean():.1f}Hz")
    a3.set_xlabel("PC 발화율 (Hz)"); a3.set_ylabel("세포 수"); a3.legend(fontsize=8)
    a3.set_title("(C) 추체세포 발화율 분포")
    fig.suptitle("E1-a  baseline 발화율 검증 — in vivo CA1 문헌 대조 (구동 과다 진단)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(FIG, "E1a_firing_baseline.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print("=" * 70)
    print(f"[그림] {out}")
    verdict = ("과활성 — 외부 Poisson 구동 하향 필요" if sE["mean"] > pc_hi else "생리적 범위")
    print(f"[판정] PC baseline: {verdict}")


if __name__ == "__main__":
    main()
