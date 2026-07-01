# -*- coding: utf-8 -*-
"""
05b_memap/map_models.py  —  단계 5b: me-type 매핑 (V2c)

목적:
  slice400 의 17,647 세포 각각의 (m-type, e-type)를 우리 검증된 23 me-model 에 매핑.
    - 정확히 같은 (m,e) 모델이 있으면 그것을 사용.
    - 없으면 같은 m-type 내 다른 e-type 모델로 대체(substitution). (계획 "갭은 m-type 내 대체")
  23개 모델은 형태(morphology) 1개씩뿐이므로, 각 세포는 해당 모델을 '복제(clone)'해 인스턴스화.
  -> 모델별 복제(재사용) 수 · 층별 분포 · 대체 건수 집계.

검증 (V2c): 모든 세포가 모델 배정 = (m,e) 100% 해소.

노드 m-type 는 언더스코어(SP_PC), 레지스트리는 하이픈(SP-PC) → 정규화.

산출: model_assignment.npz(+summary.json), figures/V2c_*.png
"""
import os
import json
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # like_slice_CA1
REGISTRY = os.path.join(ROOT, "..", "shared", "models", "models_registry.json")
CELLS = os.path.join(ROOT, "05_placement", "slice_cells.npz")
NODES_H5 = os.path.join(ROOT, "data", "circuit", "networks", "nodes",
                        "hippocampus_neurons", "nodes.h5")
MORPH_LIB = os.path.join(ROOT, "data", "morphology_library")  # (단계6에서 압축해제)
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

LAYER_ORDER = ["SO", "SP", "SR", "SLM"]
LAYER_COLOR = {"SO": "#4C72B0", "SP": "#DD8452",
               "SR": "#55A868", "SLM": "#C44E52"}


def norm_m(m):
    """nodes 'SP_PC' -> registry 'SP-PC'."""
    return m.replace("_", "-")


def main():
    reg = json.load(open(REGISTRY, encoding="utf-8"))
    models = reg["models"]                          # 23개
    # (m,e) -> model 이름,  m -> [models]
    by_me = {}
    by_m = defaultdict(list)
    model_layer = {}
    for md in models:
        key = md["mtype"] + "|" + md["etype"]
        name = f'{md["mtype"]}_{md["etype"]}'
        by_me[key] = name
        by_m[md["mtype"]].append(md)
        model_layer[name] = md["layer"]
    model_names = [f'{m["mtype"]}_{m["etype"]}' for m in models]

    d = np.load(CELLS, allow_pickle=True)
    mtype = np.array([norm_m(s) for s in d["mtype"]])
    etype = d["etype"].astype(str)
    layer = d["layer"].astype(str)
    N = len(mtype)

    # 세포별 형태(morphology) = Romani 가 nodes.h5 에 지정한 변이 형태 (라이브러리 .swc)
    import h5py
    with h5py.File(NODES_H5, "r") as f:
        g = f["nodes/hippocampus_neurons/0"]
        mlib = [s.decode() for s in g["@library"]["morphology"][:]]
        midx = g["morphology"][:]
    morphology = np.array(mlib, dtype=object)[midx][d["node_id"]].astype(str)

    assigned = np.empty(N, dtype=object)
    substituted = np.zeros(N, bool)
    sub_detail = Counter()      # (m, e_cell -> e_model)
    for i in range(N):
        m, e = mtype[i], etype[i]
        key = m + "|" + e
        if key in by_me:
            assigned[i] = by_me[key]
        else:
            # 같은 m-type 내 대체: 첫 모델 사용
            cand = by_m.get(m)
            if not cand:
                assigned[i] = "UNRESOLVED"
                continue
            md = cand[0]
            assigned[i] = f'{md["mtype"]}_{md["etype"]}'
            substituted[i] = True
            sub_detail[f'{m} {e}→{md["etype"]}'] += 1

    n_unres = int((assigned == "UNRESOLVED").sum())
    n_sub = int(substituted.sum())
    print(f"[V2c] 세포 {N:,}  미해소 {n_unres}  대체 {n_sub:,}  "
          f"정확매칭 {N-n_sub-n_unres:,}")
    print(f"[V2c] (m,e) 해소율 = {100*(N-n_unres)/N:.2f}%")

    # 모델별 복제(재사용) 수
    rep = Counter(assigned)
    print("\n=== 모델별 복제(인스턴스) 수 ===")
    for name in sorted(model_names, key=lambda n: -rep.get(n, 0)):
        if rep.get(name, 0):
            print(f"  {name:22s} {rep[name]:>7,}  [{model_layer[name]}]")

    # 모델 × 층
    ml = defaultdict(lambda: Counter())
    for i in range(N):
        ml[assigned[i]][layer[i]] += 1

    print("\n=== 대체 상세 ===")
    for k, v in sub_detail.most_common():
        print(f"  {k}: {v:,}")

    # 형태 다양성 집계 (m-type별 고유 형태 수)
    morph_div = {}
    for m in sorted(set(d["mtype"].astype(str))):
        sel = d["mtype"].astype(str) == m
        morph_div[m] = {"n_cells": int(sel.sum()),
                        "n_unique_morph": int(len(set(morphology[sel])))}
    n_uniq_all = int(len(set(morphology)))
    print(f"\n[형태 다양성] slice400 고유 형태 {n_uniq_all:,}종 "
          f"(세포 {N:,}) — 동일복제 아님, 세포별 Romani 변이형태")

    # 저장
    np.savez_compressed(os.path.join(HERE, "model_assignment.npz"),
                        node_id=d["node_id"], model=assigned.astype("U40"),
                        substituted=substituted,
                        morphology=morphology.astype("U80"))
    summary = {
        "step": "5b me-map (V2c)", "n_cells": int(N),
        "n_models_used": int(sum(1 for n in model_names if rep.get(n, 0))),
        "n_models_total": len(model_names),
        "resolved_pct": round(100 * (N - n_unres) / N, 2),
        "n_substituted": n_sub, "n_exact": int(N - n_sub - n_unres),
        "replication_per_model": {n: int(rep.get(n, 0)) for n in model_names},
        "substitution_detail": dict(sub_detail),
        "by_model_layer": {n: dict(ml[n]) for n in model_names if rep.get(n, 0)},
        "morphology": {"unique_total": n_uniq_all, "by_mtype": morph_div,
                       "note": "e-model(biophysics)=23모델 재사용, morphology=Romani 세포별 변이형태(라이브러리)"},
    }
    json.dump(summary, open(os.path.join(HERE, "model_assignment_summary.json"),
                            "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    _fig_counts(model_names, rep, model_layer)
    _fig_model_layer(model_names, ml, rep)
    _fig_substitution(sub_detail, n_sub, N)
    _fig_diversity(morph_div, n_uniq_all, N)
    print(f"\n[OK] model_assignment + figures -> {FIG}")


def _fig_counts(model_names, rep, model_layer):
    used = [(n, rep[n]) for n in model_names if rep.get(n, 0)]
    used.sort(key=lambda kv: kv[1])
    names = [n for n, _ in used]; vals = [v for _, v in used]
    colors = [LAYER_COLOR[model_layer[n]] for n in names]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(names, vals, color=colors)
    ax.set_xscale("log"); ax.set_xlabel("복제(인스턴스) 수 — 로그")
    ax.set_title(f"V2c-1  23 me-model 별 복제 수 (사용 {len(names)}종, 색=층)\n"
                 "각 모델 1개 형태를 세포 수만큼 복제·재사용")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:,}", va="center", fontsize=7)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=LAYER_COLOR[L], label=L) for L in LAYER_ORDER],
              loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "V2c_1_model_counts.png"), dpi=130)
    plt.close(fig)


def _fig_model_layer(model_names, ml, rep):
    used = [n for n in model_names if rep.get(n, 0)]
    mat = np.array([[ml[n].get(L, 0) for L in LAYER_ORDER] for n in used])
    fig, ax = plt.subplots(figsize=(6, max(6, len(used) * 0.35)))
    disp = np.log10(mat + 1)
    im = ax.imshow(disp, cmap="magma", aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels(LAYER_ORDER)
    ax.set_yticks(range(len(used))); ax.set_yticklabels(used, fontsize=7)
    for i in range(len(used)):
        for j in range(4):
            if mat[i, j]:
                ax.text(j, i, f"{mat[i,j]:,}", ha="center", va="center",
                        fontsize=6, color="w" if disp[i, j] > disp.max()*0.5 else "k")
    fig.colorbar(im, ax=ax, label="log10(세포수+1)")
    ax.set_title("V2c-2  모델 × 소마 층 배치수")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "V2c_2_model_by_layer.png"), dpi=130)
    plt.close(fig)


def _fig_substitution(sub_detail, n_sub, N):
    fig, ax = plt.subplots(figsize=(9, 5))
    exact = N - n_sub
    ax.bar(["정확 매칭\n(exact (m,e))", "m-type 내 대체\n(substituted)"],
           [exact, n_sub], color=["#55A868", "#DD8452"])
    for i, v in enumerate([exact, n_sub]):
        ax.text(i, v, f"{v:,}\n({100*v/N:.1f}%)", ha="center", va="bottom")
    ax.set_ylabel("세포 수")
    title = f"V2c-3  (m,e) 매핑: 정확 {exact:,} vs 대체 {n_sub:,} / 총 {N:,}"
    if sub_detail:
        title += "\n대체: " + ", ".join(f"{k}({v:,})" for k, v in sub_detail.most_common())
    else:
        title += "\n대체 없음 — 모든 (m,e) 정확 매칭"
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "V2c_3_substitution.png"), dpi=130)
    plt.close(fig)


def _fig_diversity(morph_div, n_uniq_all, N):
    """m-type별 고유 형태 수 vs 세포 수 (형태 다양성)."""
    ms = sorted(morph_div, key=lambda m: -morph_div[m]["n_cells"])
    cells = [morph_div[m]["n_cells"] for m in ms]
    uniq = [morph_div[m]["n_unique_morph"] for m in ms]
    x = np.arange(len(ms))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - 0.2, cells, 0.4, label="세포 수", color="#4C72B0")
    ax.bar(x + 0.2, uniq, 0.4, label="고유 형태 수", color="#DD8452")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(ms, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("개수 (로그)")
    for i in range(len(ms)):
        ax.text(i + 0.2, uniq[i], str(uniq[i]), ha="center", va="bottom", fontsize=7)
    ax.legend()
    ax.set_title(f"V2c-4  형태 다양성: m-type별 세포수 vs 고유 형태수\n"
                 f"slice400 전체 고유 형태 {n_uniq_all:,}종 / {N:,}세포 "
                 f"(Romani 라이브러리 세포별 변이 — 동일복제 아님)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "V2c_4_morph_diversity.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
