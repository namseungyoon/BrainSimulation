"""
6_efeature_distributions.py — 뉴런 검증: e-특징 분포 vs 실험 (세포 20개)
============================================================================
Source: Ecker(2020) §2.2; full-scale CA1 논문(references) 단일세포 검증; eFEL(BBP).

20개 세포(PC1+인터뉴런19)의 e-특징을 `efeature_worker.py`로 추출 → **e-type별 분포**로
그리고 **실험 기준 범위(experimental_refs)** 띠를 겹쳐 "모델이 실험 분포 안에 드는가 /
e-type 정의(적응 AC vs 비적응 NAC)에 부합하는가"를 검증한다.

비교 특징: 입력저항 Rin · AP진폭 · AP반치폭 · fAHP깊이 · 적응지수 · sag · rheobase · 최대발화율
        + f-I 곡선.

⚠️ 실험 띠는 문헌 근사 범위(experimental_refs 주석 참고). 정밀 채점은 `10_hippounit_validation.py`.

실행: <ca1sim python> papers/01_Ecker2020_CA1_synaptic/02_neurons/6_efeature_distributions.py
"""
import os
import sys
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(THIS)))
SHARED = os.path.join(ROOT, "shared")
MODELS = os.path.join(SHARED, "models")
OUT = os.path.join(THIS, "figures")
sys.path.insert(0, SHARED)
sys.path.insert(0, THIS)

import numpy as np                                  # noqa: E402
import matplotlib                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.patches import Rectangle            # noqa: E402
from matplotlib.lines import Line2D                 # noqa: E402
from common.plotstyle import set_korean_font        # noqa: E402
from common.model_naming import etype_of            # noqa: E402
import experimental_refs as REFS                    # noqa: E402

set_korean_font()
WORKER = os.path.join(THIS, "efeature_worker.py")


def list_cells():
    cells = []   # (etype, dir)
    pyr = os.path.join(MODELS, "pyramidal")
    for d in sorted(os.listdir(pyr)):
        cells.append(("cACpyr", os.path.join(pyr, d)))
    intd = os.path.join(MODELS, "interneurons")
    for d in sorted(os.listdir(intd)):
        et = etype_of(d) or "?"
        cells.append((et, os.path.join(intd, d)))
    return cells


def _run_one(et, d):
    r = subprocess.run([sys.executable, WORKER, "--cell", d], capture_output=True, text=True)
    line = next((l for l in r.stdout.splitlines() if l.startswith("EFEAT_JSON ")), None)
    if not line:
        tail = (r.stderr.strip().splitlines() or ["?"])[-1]
        return (et, d, None, tail)
    data = json.loads(line[len("EFEAT_JSON "):])
    data["etype"] = et
    return (et, d, data, None)


def collect():
    cells = list_cells()
    results = []
    print(f"[검증] 세포 {len(cells)}개 병렬(6) 격리 프로세스로 e-특징 추출 …", flush=True)
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(_run_one, et, d) for et, d in cells]
        for fut in as_completed(futs):
            et, d, data, err = fut.result()
            if data is None:
                print(f"  [실패] {os.path.basename(d)[:40]}: {err}", flush=True)
                continue
            results.append(data)
            print(f"  {et:7s} Rin={_f(data.get('Rin_MOhm'),0)} APw={_f(data.get('AP_width_ms'),2)} "
                  f"adapt={_f(data.get('adaptation_index'),3)} fmax={_f(data.get('f_at_max_Hz'),0)}", flush=True)
    return results


def _f(x, nd):
    return "  -  " if x is None else f"{x:.{nd}f}"


def feature_panel(ax, results, feat, label):
    """e-type 축 위에 모델 값 산점 + 실험 띠(음영)."""
    order = REFS.ETYPE_ORDER
    for xi, et in enumerate(order):
        # 실험 띠
        b = REFS.band(et, feat)
        if b is not None:
            lo, hi = b
            ax.add_patch(Rectangle((xi - 0.32, lo), 0.64, hi - lo,
                                   color=REFS.ETYPE_COLOR[et], alpha=0.16, lw=0))
        # 모델 값(지터 산점)
        vals = [r.get(feat) for r in results if r["etype"] == et and r.get(feat) is not None]
        if vals:
            xj = xi + (np.random.RandomState(xi + 7).rand(len(vals)) - 0.5) * 0.34
            ax.scatter(xj, vals, s=34, color=REFS.ETYPE_COLOR[et], edgecolor="k",
                       linewidth=0.5, zorder=3)
            ax.hlines(np.mean(vals), xi - 0.34, xi + 0.34, color="k", lw=2, zorder=4)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["PC\n(cACpyr)" if e == "cACpyr" else e for e in order], fontsize=8)
    ax.set_title(label, fontsize=10)
    ax.margins(x=0.08)


def main():
    os.makedirs(OUT, exist_ok=True)
    results = collect()
    if not results:
        print("[중단] 추출 결과 없음"); return

    fig, axes = plt.subplots(3, 3, figsize=(17, 13))
    fig.suptitle("뉴런 검증 — e-특징 분포 vs 실험 기준 범위 (20개 세포, eFEL)\n"
                 "음영=실험 근사 범위(experimental_refs) · 점=모델 세포 · 검은선=모델 평균",
                 fontsize=13, fontweight="bold")

    # (0,0) f-I 곡선
    axfi = axes[0, 0]
    for r in results:
        fI = r.get("fI") or []
        amps = [a for a, n in fI]; ns = [n for a, n in fI]
        axfi.plot(amps, ns, "-", color=REFS.ETYPE_COLOR.get(r["etype"], "0.5"),
                  lw=1, alpha=0.6)
    axfi.set_title("(f-I) 전류→발화수", fontsize=10)
    axfi.set_xlabel("주입 전류 (nA)"); axfi.set_ylabel("스파이크 수 (600ms)")
    axfi.legend(handles=[Line2D([0], [0], color=REFS.ETYPE_COLOR[e], lw=2) for e in REFS.ETYPE_ORDER],
                labels=["PC", "bAC", "cAC", "cNAC"], fontsize=8)

    # 나머지 8개 패널 = 특징
    panels = [axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1],
              axes[1, 2], axes[2, 0], axes[2, 1], axes[2, 2]]
    for ax, (feat, label) in zip(panels, REFS.FEATURES):
        feature_panel(ax, results, feat, label)
    # 적응지수 패널에 0선·주석
    for ax, (feat, _) in zip(panels, REFS.FEATURES):
        if feat == "adaptation_index":
            ax.axhline(0, color="0.4", ls="--", lw=1)
            ax.text(0.02, 0.97, "AC>0(적응) · NAC~0(비적응)", transform=ax.transAxes,
                    fontsize=8, va="top", color="0.3")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT, "6_efeature_distributions.png")
    fig.savefig(out, dpi=120)
    print(f"[그림] {out}")

    # ── 일관성 체크(콘솔) ──────────────────────────────────────────
    def mean_of(et, feat):
        vs = [r.get(feat) for r in results if r["etype"] == et and r.get(feat) is not None]
        return float(np.mean(vs)) if vs else None
    print("\n[e-type별 평균]")
    print(f"  {'특징':<18}{'PC':>9}{'bAC':>9}{'cAC':>9}{'cNAC':>9}")
    for feat, label in REFS.FEATURES:
        row = "".join(f"{(_f(mean_of(e, feat),2)):>9}" for e in REFS.ETYPE_ORDER)
        print(f"  {label.split('(')[0].strip():<18}{row}")
    # 정의 부합: 적응(cAC,bAC) > 비적응(cNAC) ; AP폭 cNAC < PC
    ad = {e: mean_of(e, "adaptation_index") for e in REFS.ETYPE_ORDER}
    apw = {e: mean_of(e, "AP_width_ms") for e in REFS.ETYPE_ORDER}
    print("\n[정의 부합 점검]")
    if None not in (ad["cAC"], ad["cNAC"]):
        ok = ad["cAC"] >= ad["cNAC"]
        print(f"  적응지수 cAC({ad['cAC']:.3f}) >= cNAC({ad['cNAC']:.3f}) ? {'OK' if ok else 'NO(분리 약함)'}")
    if None not in (apw["cNAC"], apw["cACpyr"]):
        ok = apw["cNAC"] <= apw["cACpyr"]
        print(f"  AP반치폭 cNAC({apw['cNAC']:.2f}ms) <= PC({apw['cACpyr']:.2f}ms) ? {'OK' if ok else 'NO'}")


if __name__ == "__main__":
    main()
