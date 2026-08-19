# -*- coding: utf-8 -*-
"""
03_network/1_connectome/prune_structural.py  —  3-1(b) pruning 1단계(구조적)

Reimann 2015 touch+prune의 **구조적 pruning**: 각 축삭이 만들 수 있는 시냅스
총량을 bouton 밀도로 제한한다.
  - 축삭당 bouton 예산 = 전체 축삭 길이 × bouton 밀도(0.2260/µm, Hub Connection Anatomy)
  - apposition을 이 예산에 맞춰 분배(쌍별 apposition 수 비례) → 시냅스 수 확정
  - 다음 2단계(기능적, 연결확률·시냅스수/연결 Bezaire&Soltesz)로 추가 pruning 예정
결과: data/derived/synapses_internal_s1.npz + 그림 3-3_prune_s1.png

재료: data/derived/appositions.npz · window_cells.npz · morphology_library
실행: python 03_network/1_connectome/prune_structural.py
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DERIVED = os.path.join(ROOT, "data", "derived")
LIB = os.path.join(ROOT, "data", "morphology_library", "morphology_library")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
BOUTON_DENSITY = 0.2260   # µm^-1 (Hub Connection Anatomy, 미특성화 평균)
MTYPES = ["SP_PC", "SP_Ivy", "SP_PVBC", "SP_CCKBC", "SO_OLM", "SP_BS",
          "SO_Tri", "SR_SCA", "SO_BS", "SLM_PPA", "SP_AA", "SO_BP"]


def axon_length(path):
    rows = np.loadtxt(path, comments="#")
    idx = rows[:, 0].astype(int); typ = rows[:, 1].astype(int)
    xyz = rows[:, 2:5]; par = rows[:, 6].astype(int)
    id2 = {i: k for k, i in enumerate(idx)}
    L = 0.0
    for k in range(len(idx)):
        if typ[k] == 2 and par[k] in id2:
            L += np.linalg.norm(xyz[k] - xyz[id2[par[k]]])
    return L


def main():
    ap = np.load(os.path.join(DERIVED, "appositions.npz"))
    pre = ap["pre_gid"]; post = ap["post_gid"]; napp = ap["n_app"].astype(float)
    wc = np.load(os.path.join(DERIVED, "window_cells.npz"), allow_pickle=True)
    morph = wc["morphology"].astype(str); mt = wc["mtype"].astype(str)

    # 축삭 길이(형태별 캐시)
    cache = {}
    axlen = np.zeros(len(morph))
    for i in range(len(morph)):
        if morph[i] not in cache:
            cache[morph[i]] = axon_length(os.path.join(LIB, morph[i] + ".swc"))
        axlen[i] = cache[morph[i]]
    budget = axlen * BOUTON_DENSITY   # 축삭당 시냅스 예산

    # 축삭(pre)별 apposition 총합 → 예산 비례 분배
    rng = np.random.default_rng(0)
    tot_app = np.zeros(len(morph))
    np.add.at(tot_app, pre, napp)
    exp_syn = np.where(tot_app[pre] > 0, budget[pre] * napp / tot_app[pre], 0.0)
    # 확률적 반올림
    nsyn = np.floor(exp_syn).astype(int) + (rng.random(len(exp_syn)) < (exp_syn - np.floor(exp_syn)))
    keep = nsyn >= 1
    Pre, Post, Ns = pre[keep], post[keep], nsyn[keep]

    np.savez_compressed(os.path.join(DERIVED, "synapses_internal_s1.npz"),
                        pre_gid=Pre.astype(np.int32), post_gid=Post.astype(np.int32),
                        n_syn=Ns.astype(np.int32), bouton_density=BOUTON_DENSITY)
    print(f"=== 3-1(b) pruning 1단계(구조적·bouton 밀도 {BOUTON_DENSITY}/µm) ===")
    print(f"[축삭 길이] 평균 {axlen.mean():.0f}µm · 예산 평균 {budget.mean():.0f} 시냅스/축삭")
    print(f"[apposition→시냅스] 후보 쌍 {len(pre):,} → 연결쌍 {keep.sum():,} (남은 비율 {100*keep.mean():.1f}%)")
    print(f"[시냅스] 총 {Ns.sum():,}개 · 쌍당 중앙 {np.median(Ns):.0f}·평균 {Ns.mean():.1f}·최대 {Ns.max()}")
    # PC 입력/출력 수렴도 감(참고)
    is_pc = mt == "SP_PC"
    pc_in = np.sum(np.isin(Post, np.where(is_pc)[0]))
    print(f"[참고] post=PC 연결쌍 {pc_in:,} · PC 1개당 평균 입력쌍 {pc_in/max(is_pc.sum(),1):.0f}")

    fig_matrix(Pre, Post, Ns, mt)
    print(f"[3-3] 저장 -> data/derived/synapses_internal_s1.npz · 그림 -> {FIG}/3-3_prune_s1.png")


def fig_matrix(Pre, Post, Ns, mt):
    mi = {m: k for k, m in enumerate(MTYPES)}
    mat = np.zeros((12, 12))
    for a, b in zip(mt[Pre], mt[Post]):
        mat[mi[a], mi[b]] += 1
    fig, ax = plt.subplots(figsize=(10, 8.5))
    lm = np.log10(mat + 1)
    im = ax.imshow(lm, cmap="viridis", aspect="auto")
    ax.set_xticks(range(12)); ax.set_xticklabels(MTYPES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(12)); ax.set_yticklabels(MTYPES, fontsize=8)
    ax.set_xlabel("post"); ax.set_ylabel("pre")
    for i in range(12):
        for j in range(12):
            if mat[i, j] > 0:
                ax.text(j, i, f"{int(mat[i,j])}", ha="center", va="center", fontsize=6,
                        color="white" if lm[i, j] < lm.max()*0.6 else "black")
    fig.colorbar(im, label="log10(연결쌍 +1)")
    ax.set_title(f"3-1(b) pruning 1단계(구조적) — mtype×mtype 연결쌍\n"
                 f"bouton 밀도 {BOUTON_DENSITY}/µm · 2단계(기능적) 예정")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "3-3_prune_s1.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
