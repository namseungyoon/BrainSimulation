# -*- coding: utf-8 -*-
"""
03_network/1_connectome/prune_functional.py  —  3-1(b) pruning 2단계(기능적)

Reimann 2015 touch+prune의 **기능적 pruning**: pathway별 연결확률·시냅스수/연결을
Bezaire & Soltesz 2013 수렴도 목표치에 맞춰 apposition을 최종 시냅스로 솎는다.
  - 억제(INT→post): 수렴도 그대로(국소 성립), 가용 apposition으로 cap
  - 흥분(PC→post): 창 PC비율 f=N_win_PC/311,500 로 국소 스케일(재귀흥분 희박)
  - 각 pre유형: n_connection = target_synapses / (syn/connection), apposition 가중 선택
결과: data/derived/synapses_internal.npz(최종) + 3단계 비교그래프 3-1b_prune_stages.png

재료: appositions.npz · synapses_internal_s1.npz · window_cells.npz
실행: python 03_network/1_connectome/prune_functional.py
근거: docs/connection_numbers.md (Bezaire&Soltesz 2013 §3.3, Megias 2001)
"""
import os
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
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
MTYPES = ["SP_PC", "SP_Ivy", "SP_PVBC", "SP_CCKBC", "SO_OLM", "SP_BS",
          "SO_Tri", "SR_SCA", "SO_BS", "SLM_PPA", "SP_AA", "SO_BP"]
N_CA1_PC = 311500   # Bezaire&Soltesz 2013 (11% interneuron 가정)

# PC 1개당 억제 수렴도(synapses/PC) — Bezaire&Soltesz 2013 §3.3 (근사=*)
TARGET_PC_INH = {"SP_PVBC": 193, "SP_CCKBC": 96, "SP_BS": 104, "SP_Ivy": 422,
                 "SR_SCA": 14, "SLM_PPA": 12, "SO_OLM": 100, "SO_BS": 104,
                 "SO_Tri": 20, "SO_BP": 20, "SP_AA": 10}   # OLM/BS/Tri/BP/AA=*근사
SYNCONN = {"SP_BS": 10, "SO_BS": 10, "SP_Ivy": 10, "SR_SCA": 6, "SLM_PPA": 6,
           "SO_OLM": 10, "SP_PVBC": 6, "SP_CCKBC": 6, "SP_AA": 6, "SO_Tri": 10,
           "SO_BP": 10, "SP_PC": 1.5}
PC_PC_SYN = 197        # PC→PC 수렴도(전체 CA1)
PC_INT_SYN = 2211      # PC→interneuron boutons/INT (전체 CA1)
INT_INT_SYN = 692      # INT→INT boutons/INT (전체 CA1)


def targets(post_mt, f, is_pc):
    """post 세포에 대한 pre_mtype별 목표 시냅스 수."""
    t = {}
    if post_mt == "SP_PC":
        t.update(TARGET_PC_INH)                 # 억제: 그대로
        t["SP_PC"] = PC_PC_SYN * f               # 흥분 재귀: 국소 스케일
    else:  # post = interneuron
        t["SP_PC"] = PC_INT_SYN * f              # PC→INT: 국소 스케일
        for m in MTYPES[1:]:                     # INT→INT: 총 692 분배(유형수로)
            t[m] = INT_INT_SYN / (len(MTYPES) - 1)
    return t


def main():
    ap = np.load(os.path.join(DERIVED, "appositions.npz"))
    pre = ap["pre_gid"].astype(np.int64); post = ap["post_gid"].astype(np.int64); napp = ap["n_app"].astype(float)
    wc = np.load(os.path.join(DERIVED, "window_cells.npz"), allow_pickle=True)
    mt = wc["mtype"].astype(str)
    n_win_pc = int(np.sum(mt == "SP_PC"))
    f = n_win_pc / N_CA1_PC
    rng = np.random.default_rng(0)

    # post별 그룹 (정렬)
    order = np.argsort(post, kind="stable")
    post_s = post[order]; pre_s = pre[order]; napp_s = napp[order]
    bnd = np.searchsorted(post_s, np.arange(post_s.max() + 2))

    Pre, Post, Ns = [], [], []
    for pc in range(len(bnd) - 1):
        a, b = bnd[pc], bnd[pc + 1]
        if a == b:
            continue
        cand_pre = pre_s[a:b]; cand_app = napp_s[a:b]
        tgt = targets(mt[pc], f, mt[pc] == "SP_PC")
        pre_mt = mt[cand_pre]
        for m, tsyn in tgt.items():
            if tsyn <= 0:
                continue
            sel = np.where(pre_mt == m)[0]
            if len(sel) == 0:
                continue
            sc = SYNCONN.get(m, 5)
            n_conn = int(round(tsyn / sc))
            n_conn = min(max(n_conn, 0), len(sel))
            if n_conn == 0:
                continue
            p = cand_app[sel] / cand_app[sel].sum()
            chosen = rng.choice(sel, n_conn, replace=False, p=p)
            Pre.append(cand_pre[chosen]); Post.append(np.full(n_conn, pc, np.int64))
            Ns.append(np.full(n_conn, max(1, int(round(sc))), np.int32))
    Pre = np.concatenate(Pre); Post = np.concatenate(Post); Ns = np.concatenate(Ns)

    np.savez_compressed(os.path.join(DERIVED, "synapses_internal.npz"),
                        pre_gid=Pre.astype(np.int32), post_gid=Post.astype(np.int32),
                        n_syn=Ns.astype(np.int32), window_pc_fraction=f)
    print(f"=== 3-1(b) pruning 2단계(기능적·Bezaire&Soltesz 2013) ===")
    print(f"[창 PC비율 f] {n_win_pc}/{N_CA1_PC} = {f:.4f} (PC→ 흥분 국소 스케일)")
    print(f"[최종 연결쌍] {len(Pre):,} · 시냅스 {Ns.sum():,}")
    is_pc = mt == "SP_PC"
    pcpost = np.isin(Post, np.where(is_pc)[0])
    print(f"[PC로 가는 연결] {pcpost.sum():,} · PC당 평균 입력쌍 {pcpost.sum()/max(is_pc.sum(),1):.0f}")
    pcpc = pcpost & (mt[Pre] == "SP_PC")
    print(f"[PC→PC(국소)] {pcpc.sum():,} · PC당 {pcpc.sum()/max(is_pc.sum(),1):.1f} (희박)")

    stages_graph(pre, post, mt, Pre, Post, Ns)
    matrix_final(Pre, Post, mt)
    print(f"[3-1b] 저장 -> data/derived/synapses_internal.npz")
    print(f"[3-1b] 그림 -> {FIG}/3-1b_prune_stages.png · 3-1b_matrix_final.png")


def stages_graph(pre0, post0, mt, Pre, Post, Ns):
    s1 = np.load(os.path.join(DERIVED, "synapses_internal_s1.npz"))
    is_pc = np.where(mt == "SP_PC")[0]
    def pcpc_pairs(pr, po):
        m = np.isin(po, is_pc) & (mt[pr] == "SP_PC"); return int(m.sum())
    # 3단계: 가지치기전(touch) / 1단계(구조) / 2단계(기능)
    pairs = [len(pre0), len(s1["pre_gid"]), len(Pre)]
    syns = [np.load(os.path.join(DERIVED, "appositions.npz"))["n_app"].sum(),
            int(s1["n_syn"].sum()), int(Ns.sum())]
    pcpc = [pcpc_pairs(pre0, post0), pcpc_pairs(s1["pre_gid"], s1["post_gid"]), pcpc_pairs(Pre, Post)]
    labels = ["가지치기 전\n(touch)", "1단계 후\n(구조·bouton)", "2단계 후\n(기능·Bezaire)"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
    # (data, 제목, 색, apposition막대 인덱스(단위 다름 표시))
    panels = [(pairs, "연결쌍 수", "#4C72B0", None),
              (syns, "시냅스 수  (첫 막대=apposition 접촉후보·서브샘플, 시냅스 아님)", "#DD8452", 0),
              (pcpc, "PC→PC 연결쌍 (국소 희박화)", "#C44E52", None)]
    for ax, (data, ttl, col, appidx) in zip(axes, panels):
        bars = ax.bar(labels, data, color=col, alpha=0.85)
        if appidx is not None:   # apposition 막대: 단위 다름 → 회색 빗금
            bars[appidx].set_facecolor("#b0b0b0"); bars[appidx].set_hatch("//"); bars[appidx].set_edgecolor("#666")
        ax.set_title(ttl, fontsize=10.5); ax.set_yscale("log")
        for k, (bx, v) in enumerate(zip(bars, data)):
            note = "\n(접촉후보)" if appidx is not None and k == appidx else ""
            ax.text(bx.get_x() + bx.get_width()/2, v, f"{int(v):,}{note}", ha="center", va="bottom", fontsize=8.5)
        ax.set_ylim(0.5, max(data) * 3.5)
    fig.suptitle("3-1(b) 내부 커넥텀 pruning — 가지치기 전 → 1단계(구조) → 2단계(기능) (log축)\n"
                 "※ 시냅스 패널 첫 막대만 apposition(접촉후보·서브샘플)로 단위가 다름 — 시냅스는 1·2단계 값", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "3-1b_prune_stages.png"), dpi=130)
    plt.close(fig)


def matrix_final(Pre, Post, mt):
    mi = {m: k for k, m in enumerate(MTYPES)}
    mat = np.zeros((12, 12))
    for a, b in zip(mt[Pre], mt[Post]):
        mat[mi[a], mi[b]] += 1
    fig, ax = plt.subplots(figsize=(10, 8.5))
    lm = np.log10(mat + 1)
    im = ax.imshow(lm, cmap="cividis", aspect="auto")
    ax.set_xticks(range(12)); ax.set_xticklabels(MTYPES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(12)); ax.set_yticklabels(MTYPES, fontsize=8)
    ax.set_xlabel("post"); ax.set_ylabel("pre")
    for i in range(12):
        for j in range(12):
            if mat[i, j] > 0:
                ax.text(j, i, f"{int(mat[i,j])}", ha="center", va="center", fontsize=6,
                        color="white" if lm[i, j] < lm.max()*0.6 else "black")
    fig.colorbar(im, label="log10(연결쌍 +1)")
    ax.set_title("3-1(b) 최종 내부 커넥텀 (2단계 pruning 후) — mtype×mtype")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "3-1b_matrix_final.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
