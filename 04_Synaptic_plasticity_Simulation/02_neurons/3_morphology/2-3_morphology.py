# -*- coding: utf-8 -*-
"""2-3 형태 지표 — 선별한 두 세포의 수상돌기 구조를 정량화한다

단계   : 2-3 (파이프라인 2단계 뉴런 / 하위 3 morphology)
방법   : NEURON 에 로드한 pre·post 의 각 구획을 소마 경로거리로 분류해, 경로거리 대비 수상돌기
         길이 분포(sholl 유사)·직경 분포·정단/기저 분기를 잰다. 3-2(시냅스 배치)가 쓸
         'SR 대역(정단 100~300um)에 실제로 얼마나 많은 막이 있는가'를 여기서 확정한다.
근거   : docs/PIPELINE.md 산출물 규약 · config/cells.yaml
재료   : lib/cells.py · lib/morphology.py · config/cells.yaml
결과   : figures/2-3_morphology.png · figures/2-3_morphology.json

실행:
  . .\\env\\activate.ps1
  & $Py04 02_neurons\\3_morphology\\2-3_morphology.py
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np                          # noqa: E402
import yaml                                 # noqa: E402
from lib import plots                        # noqa: E402
from lib import cells                        # noqa: E402
from lib.nrnenv import h                     # noqa: E402

# SR 근위 대역(3-2 에서 SC 대리 시냅스를 놓을 곳). config 로 뺄 값이지만 2-3 에서 처음 정의.
SR_MIN, SR_MAX = 100.0, 300.0
BIN = 25.0                                    # 경로거리 히스토그램 간격 um


def dendrite_profile(cell):
    """각 구획의 (도메인, 소마 경로거리 중점, 길이, 평균직경)을 모은다.

    h.distance() 로 소마(0.5) 기준 경로거리를 잰다. 구획 중점 거리 = 시작거리 + L/2.
    """
    soma = cell.soma[0]
    h.distance(0, soma(0.5))
    recs = []
    for sec in cell.all:
        nm = sec.name().split(".")[-1].split("[")[0]
        if nm not in ("apic", "dend"):
            continue
        dom = "apical" if nm == "apic" else "basal"
        d0 = h.distance(sec(0.0))
        mid = d0 + sec.L / 2.0
        diam = np.mean([seg.diam for seg in sec])
        recs.append((dom, mid, sec.L, diam))
    return recs


def summarize(recs):
    ap = [(m, L, d) for dom, m, L, d in recs if dom == "apical"]
    ba = [(m, L, d) for dom, m, L, d in recs if dom == "basal"]
    def total(rs):
        return sum(L for _, L, _ in rs)
    # SR 대역 정단 막 길이
    sr = sum(L for m, L, _ in ap if SR_MIN <= m <= SR_MAX)
    return dict(
        apical_len=round(total(ap), 1),
        basal_len=round(total(ba), 1),
        apical_sections=len(ap),
        basal_sections=len(ba),
        sr_apical_len=round(sr, 1),
        sr_fraction=round(sr / total(ap), 3) if total(ap) else 0.0,
    )


def hist_by_dist(recs, dom, dmax):
    edges = np.arange(0, dmax + BIN, BIN)
    vals = np.zeros(len(edges) - 1)
    for d, m, L, _ in [(r[0], r[1], r[2], r[3]) for r in recs]:
        if d != dom:
            continue
        b = int(m // BIN)
        if 0 <= b < len(vals):
            vals[b] += L
    return edges, vals


def main():
    plots.setup()
    with open(os.path.join(ROOT, "config", "cells.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["pair"]
    models_root = os.path.join(REPO, "Models")

    print("=== 2-3 형태 지표 ===")
    data = []
    for role in ("pre", "post"):
        bdir = os.path.join(models_root, cfg[f"{role}_bundle"])
        cell, _ = cells.load_cell(bdir, role)
        recs = dendrite_profile(cell)
        summ = summarize(recs)
        data.append(dict(role=role, tag=cfg[f"{role}_tag"], recs=recs, summ=summ))
        print(f"  [{role}] {cfg[f'{role}_tag']}")
        print(f"        정단 {summ['apical_len']:.0f} um ({summ['apical_sections']}구획)  "
              f"기저 {summ['basal_len']:.0f} um ({summ['basal_sections']}구획)")
        print(f"        SR대역(정단 {SR_MIN:.0f}~{SR_MAX:.0f}um) 막길이 {summ['sr_apical_len']:.0f} um "
              f"(정단의 {summ['sr_fraction']*100:.1f}%)")

    # 최대 경로거리(축 공유)
    dmax = 0.0
    for d in data:
        for _, m, _, _ in d["recs"]:
            dmax = max(dmax, m)
    dmax = np.ceil(dmax / BIN) * BIN

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))

    for col, d in enumerate(data):
        # 위: 경로거리별 수상돌기 길이 (정단 위 / 기저 아래로)
        ax = axes[0][col]
        eap, vap = hist_by_dist(d["recs"], "apical", dmax)
        eba, vba = hist_by_dist(d["recs"], "basal", dmax)
        centers = eap[:-1] + BIN / 2
        ax.bar(centers, vap, width=BIN * 0.92, color="#e53935", label="정단")
        ax.bar(centers, -vba, width=BIN * 0.92, color="#1e88e5", label="기저")
        ax.axvspan(SR_MIN, SR_MAX, color="#ffb300", alpha=0.16, zorder=0)
        ax.axhline(0, color="#616161", lw=0.8)
        ax.set_xlabel("소마 경로거리 (um)")
        ax.set_ylabel("구간 내 막 길이 (um)   기저 <- | -> 정단")
        role_ko = "pre (자극)" if d["role"] == "pre" else "post (기록)"
        ax.set_title(f"{role_ko}  {d['tag']}\n경로거리별 수상돌기 분포", fontsize=10.5, loc="left")
        ax.legend(fontsize=8.5, loc="upper right")
        ax.text(SR_MIN + 5, ax.get_ylim()[1] * 0.9,
                f"SR 대역\n정단막 {d['summ']['sr_apical_len']:.0f} um",
                fontsize=8, color="#b26a00", va="top")

        # 아래: 정단 직경 vs 경로거리 (원위로 갈수록 가늘어지는가)
        ax2 = axes[1][col]
        ap = [(m, dm) for dom, m, L, dm in d["recs"] if dom == "apical"]
        ba = [(m, dm) for dom, m, L, dm in d["recs"] if dom == "basal"]
        if ap:
            am, ad = zip(*ap)
            ax2.scatter(am, ad, s=14, color="#e53935", alpha=0.6, label="정단")
        if ba:
            bm, bd = zip(*ba)
            ax2.scatter(bm, bd, s=14, color="#1e88e5", alpha=0.6, label="기저")
        ax2.axvspan(SR_MIN, SR_MAX, color="#ffb300", alpha=0.16, zorder=0)
        ax2.set_xlabel("소마 경로거리 (um)")
        ax2.set_ylabel("구획 평균 직경 (um)")
        ax2.set_title("직경 대 경로거리 — 원위로 갈수록 가늘어짐", fontsize=10, loc="left")
        ax2.legend(fontsize=8.5, loc="upper right")

    for ax in axes.ravel():
        ax.set_xlim(0, dmax)

    fig.suptitle("2-3  두 세포 형태 지표 — 경로거리별 수상돌기 분포·직경 (SR 대역 강조)",
                 fontsize=12.5, y=0.985)
    fig.subplots_adjust(top=0.90, hspace=0.32, wspace=0.22)
    plots.stamp(fig, f"2-3 | SR 대역 정단 {SR_MIN:.0f}~{SR_MAX:.0f}um | 3-2 시냅스 배치의 근거")
    outdir = plots.figdir(__file__)
    plots.save(fig, outdir, "2-3_morphology.png")

    checks = [
        ("pre 정단 최대거리 > 500um", max(m for dom, m, _, _ in data[0]["recs"] if dom == "apical") > 500),
        ("post 정단 최대거리 > 500um", max(m for dom, m, _, _ in data[1]["recs"] if dom == "apical") > 500),
        ("pre SR대역 정단막 > 0", data[0]["summ"]["sr_apical_len"] > 0),
        ("post SR대역 정단막 > 0", data[1]["summ"]["sr_apical_len"] > 0),
    ]
    n_ok = sum(1 for _, ok in checks if ok)
    for k, ok in checks:
        print(f"  {'O' if ok else 'X'} {k}")

    out = dict(sr_band=[SR_MIN, SR_MAX], bin_um=BIN,
               pre=dict(tag=cfg["pre_tag"], **data[0]["summ"]),
               post=dict(tag=cfg["post_tag"], **data[1]["summ"]),
               checks={k: bool(v) for k, v in checks},
               checks_passed=n_ok, checks_total=len(checks))
    jpath = os.path.join(outdir, "2-3_morphology.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")

    if n_ok != len(checks):
        print(f"\n[실패] {len(checks)-n_ok}개 미통과")
        return 1
    print(f"\n[통과] 2-3 완료 ({n_ok}/{len(checks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
