# -*- coding: utf-8 -*-
"""
07_connectivity/prune_connectome.py  —  Romani 방식 pruning (문헌 실측 기반)

입력: touch_connectivity.npz (raw 형태접촉 커넥텀; pre,post,n_syn=apposition,dist)
방법 (Romani 2024, PDF p.33 + Table S9):
  1) 경로별 연결당 시냅스수를 **실측 평균(S9)·변동계수 CV=0.50**(Romani)으로 배정
     (raw apposition 수는 점기반 과대라, 시냅스수는 문헌값으로 대체 — 핵심 pruning)
  2) 연결 수(연결확률)는 전체 CA1 pruned 결과(821M 시냅스)를 **슬라이스 비율로 스케일**한
     목표 총시냅스에 맞춰 apposition 가중 다운샘플 (연결확률 보정 근사)

Table S9 (연결당 시냅스수, 실측) → 9클래스 매핑:
  PC->PC 1.17 | PC->SOM+ 2.83(PC→OLM) | SOM+->PC 10(OLM→PC)
  PV+->PC 7.63(PVBC11·BS6·AA5.89 평균) | CCK+->PC 6.82(CCKBC8.3·SCA5.33 평균)
  CCK+->CCK+ 3.5(SCA→SCA) | CCK-->CCK- 1.55(PVBC→PV)
  PC->SOM- 2.5(근사) | NOS+->PC 5.0(Ivy, 근사)

산출: pruned_summary.json, figures/V4p_*.png
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "..", "papers",
                                "01_Ecker2020_CA1_synaptic", "05_paired_recording"))
from pathway_map import pathway_class            # noqa: E402

NPZ = os.path.join(HERE, "touch_connectivity.npz")
CELLS = os.path.join(ROOT, "05_placement", "slice_cells.npz")
FIG = os.path.join(HERE, "figures")
CV = 0.50
N_SLICE, N_CA1 = 17647, 456378
CA1_SYN = 821e6                    # Romani 전체 CA1 pruned 내부 시냅스
SEED = 0

# S9 실측 연결당 시냅스수 (직접값 D / 근사 A)
S9 = {"PC->PC (E2)": (1.17, "D"), "PC->SOM+ (E1)": (2.83, "D"),
      "SOM+->PC (I2)": (10.0, "D"), "PV+->PC (I2)": (7.63, "D(avg)"),
      "CCK+->PC (I3)": (6.82, "D(avg)"), "CCK+->CCK+ (I1)": (3.5, "D"),
      "CCK-->CCK- (I2)": (1.55, "D"), "PC->SOM- (E2)": (2.5, "A"),
      "NOS+->PC (I3)": (5.0, "A")}


def main():
    d = np.load(NPZ)
    pre, post, appo = d["pre"], d["post"], d["n_syn"].astype(np.int64)
    c = np.load(CELLS, allow_pickle=True)
    mt = np.array([m.replace("_", "-") for m in c["mtype"].astype(str)])
    E = len(pre)
    rng = np.random.default_rng(SEED)
    print(f"[load] raw 연결 {E:,} · raw apposition {appo.sum():,}")

    # 연결별 클래스 (벡터화: m-type코드 → 클래스행렬)
    mts = sorted(set(mt)); code = {m: i for i, m in enumerate(mts)}
    cc = np.array([code[m] for m in mt])
    classes = sorted(S9.keys())
    cidx = {cl: i for i, cl in enumerate(classes)}
    CID = -np.ones((len(mts), len(mts)), int)
    for pi, p in enumerate(mts):
        for qi, q in enumerate(mts):
            cl = pathway_class(p, q)
            if cl:
                CID[pi, qi] = cidx[cl]
    conn_cls = CID[cc[pre], cc[post]]              # 각 연결의 클래스 idx

    # 1) 연결당 시냅스수 배정: Gamma(mean=target, cv=CV), 반올림, 최소1
    tgt = np.array([S9[classes[i]][0] for i in range(len(classes))])
    m_arr = tgt[conn_cls]
    k = 1.0 / CV**2; theta = m_arr * CV**2
    syn = np.maximum(1, np.round(rng.gamma(k, theta)).astype(np.int64))
    total_keepall = int(syn.sum())
    print(f"[pruned/keepall] 연결 {E:,} 유지 · 시냅스 {total_keepall:,} "
          f"(평균 {total_keepall/E:.2f}/연결)")

    # 2) 전체CA1 스케일 목표에 맞춰 연결 다운샘플 (apposition 가중)
    target_total = CA1_SYN * N_SLICE / N_CA1
    print(f"[target] 전체CA1 821M × ({N_SLICE}/{N_CA1}) = {target_total/1e6:.1f}M 시냅스")
    if total_keepall > target_total:
        # apposition 큰 연결 우선 유지: 확률 = f * (appo/mean_appo) 클립
        f = target_total / total_keepall
        w = appo / appo.mean()
        keep_p = np.clip(f * w, 0, 1)
        keep = rng.random(E) < keep_p
    else:
        keep = np.ones(E, bool)
    pre_k, post_k, syn_k, cls_k = pre[keep], post[keep], syn[keep], conn_cls[keep]
    total_cal = int(syn_k.sum())
    print(f"[pruned/calibrated] 연결 {keep.sum():,} · 시냅스 {total_cal:,} "
          f"(평균 {total_cal/max(keep.sum(),1):.2f}/연결)")

    # 클래스별 집계 (calibrated)
    per = {}
    for i, cl in enumerate(classes):
        m = cls_k == i
        per[cl] = {"target_syn_per_conn": S9[cl][0], "src": S9[cl][1],
                   "model_syn_per_conn": float(syn_k[m].mean()) if m.any() else 0.0,
                   "connections": int(m.sum()), "synapses": int(syn_k[m].sum())}

    indeg = np.bincount(post_k, minlength=N_SLICE)
    outdeg = np.bincount(pre_k, minlength=N_SLICE)
    summary = {
        "method": "Romani-style pruning (S9 syn/conn, CV0.50; conn downsample to CA1-scaled total)",
        "raw_connections": E, "raw_appositions": int(appo.sum()),
        "keepall_synapses": total_keepall,
        "target_total_synapses": int(target_total),
        "pruned_connections": int(keep.sum()), "pruned_synapses": total_cal,
        "mean_syn_per_conn": total_cal / max(int(keep.sum()), 1),
        "mean_in_degree": float(indeg[indeg > 0].mean()) if (indeg > 0).any() else 0,
        "mean_out_degree": float(outdeg[outdeg > 0].mean()) if (outdeg > 0).any() else 0,
        "per_class": per, "CV": CV,
    }
    json.dump(summary, open(os.path.join(HERE, "pruned_summary.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=2)
    np.savez_compressed(os.path.join(HERE, "pruned_connectivity.npz"),
                        pre=pre_k.astype(np.int32), post=post_k.astype(np.int32),
                        n_syn=syn_k.astype(np.int16), cls=cls_k.astype(np.int8),
                        classes=np.array(classes, dtype="U20"))

    _fig_synconn_validate(classes, per)
    _fig_before_after(E, appo.sum(), total_keepall, keep.sum(), total_cal, target_total)
    print(f"[OK] -> {FIG}")


def _fig_synconn_validate(classes, per):
    """모델 연결당 시냅스수 vs 문헌 S9 (일치 검증)."""
    order = sorted(range(len(classes)), key=lambda i: -per[classes[i]]["target_syn_per_conn"])
    cl = [classes[i] for i in order]
    tg = [per[c]["target_syn_per_conn"] for c in cl]
    md = [per[c]["model_syn_per_conn"] for c in cl]
    src = [per[c]["src"] for c in cl]
    x = np.arange(len(cl)); h = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.barh(x + h/2, tg, h, color="#4C72B0", label="문헌 S9 (실측)")
    ax.barh(x - h/2, md, h, color="#DD8452", label="우리 모델 (배정 결과)")
    ax.set_yticks(x); ax.set_yticklabels([f"{c}  [{s}]" for c, s in zip(cl, src)], fontsize=8)
    ax.set_xlabel("연결당 시냅스 수")
    for i in range(len(cl)):
        ax.text(tg[i], x[i]+h/2, f" {tg[i]:.2f}", va="center", fontsize=7)
    ax.legend()
    ax.set_title("V4p-1  연결당 시냅스수: 문헌(S9) vs 우리 모델 (D=직접·A=근사)\n"
                 "CV=0.50로 배정 → 평균이 문헌값에 일치 (pruning 핵심 제약)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "V4p_1_synconn_validate.png"), dpi=130)
    plt.close(fig)


def _fig_before_after(E0, appo0, keepall, conn1, syn1, target):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].bar(["raw 연결", "pruned 연결"], [E0, conn1], color=["#bbbbbb", "#55A868"])
    ax[0].set_yscale("log"); ax[0].set_ylabel("연결 수 (로그)")
    for i, v in enumerate([E0, conn1]):
        ax[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax[0].set_title("연결 수: raw → pruned")
    ax[1].bar(["raw apposition", "pruned 시냅스", "목표(CA1스케일)"],
              [appo0, syn1, target], color=["#bbbbbb", "#DD8452", "#4C72B0"])
    ax[1].set_yscale("log"); ax[1].set_ylabel("시냅스 수 (로그)")
    for i, v in enumerate([appo0, syn1, target]):
        ax[1].text(i, v, f"{v/1e6:.0f}M", ha="center", va="bottom", fontsize=9)
    ax[1].set_title("시냅스 수: raw(17억) → pruned")
    fig.suptitle("V4p-2  pruning 전후 규모 (Romani 문헌 기반)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "V4p_2_before_after.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
