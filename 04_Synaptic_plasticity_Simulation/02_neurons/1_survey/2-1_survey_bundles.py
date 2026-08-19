# -*- coding: utf-8 -*-
"""2-1 뉴런 선별 — CA1 추체 번들 13종의 실제 형태를 그려서 pre/post 를 고른다

단계   : 2-1 (파이프라인 2단계 뉴런 / 하위 1 survey)
방법   : 각 번들의 SWC 를 직접 읽어 소마 원점·정단수상돌기 상방으로 정렬한 뒤 나란히 그린다.
         형태 지표(정단 길이·최대 경로거리·기저 길이)를 함께 재서 선별 근거를 남긴다.
         NEURON 을 거치지 않는 이유: 13종이 **템플릿 이름이 전부 `CA1_PC_cAC_sig` 로 같아서**
         한 프로세스에 동시 로드가 불가능하다. NEURON 인스턴스화는 2-2 에서 한다.
근거   : docs/PIPELINE.md 산출물 규약 — 표가 아니라 모형을 그린다.
재료   : ../Models/CA1_pyr_cACpyr_*_model_files/morphology/*.swc (13종)
결과   : figures/2-1_morphology_grid.png · 2-1_metrics.png · 2-1_survey.json

실행:
  .venv\\Scripts\\python.exe 02_neurons\\1_survey\\2-1_survey_bundles.py
"""
import os
import re
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

import numpy as np                              # noqa: E402
from lib import plots                            # noqa: E402
from lib import morphology as mo                 # noqa: E402

MODELS = os.path.join(REPO, "Models")


def find_bundles():
    pat = re.compile(r"^CA1_pyr_cACpyr_(.+)_\d{14}_model_files$")
    out = []
    for d in sorted(os.listdir(MODELS)):
        m = pat.match(d)
        if m and os.path.isdir(os.path.join(MODELS, d)):
            out.append((m.group(1), os.path.join(MODELS, d)))
    return out


def main():
    plots.setup()
    import matplotlib.pyplot as plt

    bundles = find_bundles()
    print(f"=== 2-1 뉴런 선별 : 추체 번들 {len(bundles)}종 ===")

    data = []
    for tag, d in bundles:
        swc = mo.bundle_swc(d)
        m = mo.align(mo.load_swc(swc), mode="apical")
        met = mo.metrics(m)
        data.append(dict(tag=tag, dir=os.path.basename(d), m=m, met=met))
        print(f"  {tag:24s} 포인트 {met['n_points']:6d}  "
              f"정단 {met['len_4']:8.0f} um  기저 {met['len_3']:7.0f} um  "
              f"정단최대거리 {met['maxdist_apical']:6.0f} um")

    # ---------- 그림 1: 형태 격자 (실제 모형) --------------------------------
    ncol = 5
    nrow = int(np.ceil(len(data) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.05 * ncol, 3.5 * nrow))
    axes = np.atleast_1d(axes).ravel()

    # 모든 세포를 같은 축척으로 → 크기 비교가 성립한다.
    # 축척은 **수상돌기 기준**으로 잡는다. 축삭을 포함하면 옆으로 멀리 뻗은 세포 하나
    # (mpg141017_a1-2_idC) 때문에 전체가 작아져 정작 봐야 할 수상돌기가 안 보인다.
    dend = (mo.SOMA, mo.BASAL, mo.APICAL)
    xmax = ymin = ymax = 0.0
    for rec in data:
        m = rec["m"]
        sel = np.isin(m["type"], dend)
        xy = m["xyz"][sel][:, :2]
        xmax = max(xmax, np.percentile(np.abs(xy[:, 0]), 99.8))
        ymin = min(ymin, np.percentile(xy[:, 1], 0.2))
        ymax = max(ymax, np.percentile(xy[:, 1], 99.8))
    half = xmax * 1.08
    ylo, yhi = ymin * 1.15, ymax * 1.08

    for ax, rec in zip(axes, data):
        mo.render(ax, rec["m"], types=(mo.SOMA, mo.BASAL, mo.APICAL, mo.AXON),
                  autoscale=False)
        ax.set_xlim(-half, half)
        ax.set_ylim(ylo, yhi)
        # adjustable="box": 데이터 한계를 늘리지 않고 축 상자를 줄여 aspect 를 맞춘다.
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_color("#dddddd")
        ax.set_title(rec["tag"], fontsize=8.5, pad=4)
        met = rec["met"]
        ax.text(0.03, 0.975,
                f"정단 {met['len_4']/1000:.1f} mm\n최대 {met['maxdist_apical']:.0f} um",
                transform=ax.transAxes, va="top", fontsize=7.4, color="#455a64")
    for ax in axes[len(data):]:
        ax.axis("off")
    mo.scalebar(axes[0], 200, "200 um")

    handles = [plt.Line2D([], [], color=mo.TYPE_COLOR[t], lw=2.4,
                          label=mo.TYPE_KO[t])
               for t in (mo.SOMA, mo.APICAL, mo.BASAL, mo.AXON)]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.002))
    fig.suptitle("2-1  CA1 추체세포 13종 실제 형태 (소마 원점 · 정단수상돌기 상방 정렬 · 동일 축척)",
                 fontsize=12, y=0.985)
    fig.subplots_adjust(top=0.93, bottom=0.075, hspace=0.16, wspace=0.06)
    plots.stamp(fig, "2-1 | SWC 직접 파싱 | 축삭은 NEURON replace_axon 으로 교체되므로 참고용")
    outdir = plots.figdir(__file__)
    plots.save(fig, outdir, "2-1_morphology_grid.png")

    # ---------- 그림 2: 지표 비교 + 선별 ------------------------------------
    tags = [r["tag"] for r in data]
    ap = np.array([r["met"]["len_4"] / 1000 for r in data])      # mm
    ba = np.array([r["met"]["len_3"] / 1000 for r in data])
    md = np.array([r["met"]["maxdist_apical"] for r in data])

    # 선별 기준: 정단 최대 경로거리가 충분히 길고(원위 SLM 까지), 정단 길이가 중앙값 근처인 것.
    # 극단값(가장 크거나 작은 세포)은 대표성이 떨어져 피한다.
    score = np.abs(ap - np.median(ap))
    ok = md >= 500.0
    cand = [i for i in np.argsort(score) if ok[i]]
    pre_i, post_i = (cand + list(np.argsort(score)))[:2]

    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(13.2, 5.0),
                                  gridspec_kw={"width_ratios": [1.5, 1]})
    order = np.argsort(-ap)
    ypos = np.arange(len(tags))
    a1.barh(ypos, ap[order], color="#e53935", label="정단수상돌기")
    a1.barh(ypos, -ba[order], color="#1e88e5", label="기저수상돌기")
    a1.set_yticks(ypos)
    a1.set_yticklabels([tags[i] for i in order], fontsize=8)
    a1.invert_yaxis()
    a1.axvline(0, color="#616161", lw=0.9)
    a1.set_xlabel("수상돌기 총 길이 (mm)   <- 기저 | 정단 ->")
    a1.set_title("수상돌기 길이 비교", loc="left", pad=10)
    a1.legend(fontsize=8.5, loc="lower right")
    for k, i in enumerate(order):
        a1.text(ap[i] + 0.15, k, f"{ap[i]:.1f}", va="center", fontsize=7.5, color="#b71c1c")
        a1.text(-ba[i] - 0.15, k, f"{ba[i]:.1f}", va="center", ha="right",
                fontsize=7.5, color="#0d47a1")
        if i in (pre_i, post_i):
            a1.text(-ba[i] - 1.35, k, "선택", va="center", ha="right",
                    fontsize=8, color=plots.OK, fontweight="bold")
    a1.set_xlim(-max(ba) * 1.55, max(ap) * 1.22)

    a2.scatter(md, ap, s=46, color="#9e9e9e", zorder=3)
    for i, t in enumerate(tags):
        if i in (pre_i, post_i):
            a2.scatter([md[i]], [ap[i]], s=150, facecolor="none",
                       edgecolor=plots.OK, lw=2.2, zorder=4)
            # 두 선택 세포가 가까이 있으면 라벨이 겹친다 → pre 는 아래, post 는 위로 분리.
            role = "pre(자극)" if i == pre_i else "post(기록)"
            dy = -16 if i == pre_i else 12
            va = "top" if i == pre_i else "bottom"
            a2.annotate(f"{t}  {role}", (md[i], ap[i]), textcoords="offset points",
                        xytext=(10, dy), va=va, fontsize=8, color=plots.OK,
                        fontweight="bold")
    a2.axvline(500, ls="--", lw=1.1, color=plots.NG)
    a2.text(505, a2.get_ylim()[0], " 최소 500 um", fontsize=8, color=plots.NG,
            va="bottom")
    a2.axhline(np.median(ap), ls=":", lw=1.1, color="#616161")
    a2.text(a2.get_xlim()[0], np.median(ap), " 정단 길이 중앙값 ", fontsize=8,
            color="#616161", va="bottom")
    a2.set_xlabel("정단 최대 경로거리 (um)")
    a2.set_ylabel("정단수상돌기 총 길이 (mm)")
    a2.set_title("선별 기준 — 원위까지 닿고 극단값이 아닌 것", loc="left", pad=10)

    plots.stamp(fig2, f"2-1 | pre={tags[pre_i]} · post={tags[post_i]}")
    plots.save(fig2, outdir, "2-1_metrics.png")

    # ---------- 산출 ---------------------------------------------------------
    out = dict(
        n_bundles=len(data),
        template_name_shared="CA1_PC_cAC_sig",
        note=("13종이 템플릿 이름을 공유하지만 hoc 안의 생체물리 파라미터는 서로 다르다"
              "(BluePyOpt 개별 최적화). 2-2 에서 템플릿 이름을 번들별로 바꿔 로드해야"
              "각 세포가 자기 파라미터를 갖는다."),
        selected=dict(pre=data[pre_i]["dir"], post=data[post_i]["dir"],
                      pre_tag=tags[pre_i], post_tag=tags[post_i]),
        bundles=[dict(tag=r["tag"], dir=r["dir"],
                      **{k: v for k, v in r["met"].items()}) for r in data],
    )
    jpath = os.path.join(outdir, "2-1_survey.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")
    print(f"\n선별  pre(자극) = {tags[pre_i]}")
    print(f"      post(기록) = {tags[post_i]}")
    print("\n[통과] 2-1 완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
