# -*- coding: utf-8 -*-
"""
08_synapses/assign_synapses.py  —  단계 8: 시냅스 생리 주입 (V4)

목적:
  pruned 커넥텀(9클래스)에 Ecker(2020) params_table3 의 **9클래스 EMS 파라미터**
  (g_nS·Use·Dep·Fac·Nrrp·NMDA비·E_rev·STP프로파일)를 주입.

검증 (V4, 기존 Ecker 검증 재사용):
  - 9클래스 파라미터 표
  - STP 프로파일: Tsodyks-Markram 모델로 스파이크 열에 대한 PSP 진폭 재현
    → E1/I1 촉진, E2/I2 억압, I3 pseudo-linear 확인
  - 우리 커넥텀의 클래스별 시냅스 수(파라미터가 실제로 몇 개 시냅스에 적용되는지)

산출: synapse_assignment_summary.json, figures/V4_*.png
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
                                "01_Ecker2020_CA1_synaptic", "03_synapses"))
from params_table3 import CLASSES                # noqa: E402

PRUNED = os.path.join(ROOT, "07_connectivity", "pruned_connectivity.npz")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
STP_COLOR = {"E1": "#2f8f4e", "E2": "#DD8452", "I1": "#4C72B0",
             "I2": "#8172B3", "I3": "#937860"}


def tm_psp(U, D, F, spikes_ms):
    """Tsodyks-Markram: 스파이크열에 대한 정규화 PSP 진폭 열."""
    u = U; R = 1.0; out = []
    prev = None
    for t in spikes_ms:
        if prev is None:
            u = U; R = 1.0
        else:
            dt = t - prev
            u = U + u * (1 - U) * np.exp(-dt / F)
            R = 1 + (R - u_prev * R - 1) * np.exp(-dt / D)
        out.append(u * R)
        u_prev = u
        prev = t
    return np.array(out)


def main():
    d = np.load(PRUNED, allow_pickle=True)
    cls = d["cls"]; nsyn = d["n_syn"].astype(np.int64)
    classes = list(d["classes"].astype(str))
    E = len(cls)
    print(f"[load] pruned 연결 {E:,} · 시냅스 {nsyn.sum():,}")

    # 클래스별 우리 커넥텀 통계 + EMS 파라미터
    rows = []
    for i, cl in enumerate(classes):
        m = cls == i
        p = CLASSES.get(cl, {})
        rows.append({
            "class": cl, "ei": p.get("ei"), "stp": p.get("stp"),
            "g_nS": p.get("g_nS"), "tau_d": p.get("tau_d_AMPA", p.get("tau_d_GABAA")),
            "Use": p.get("Use"), "Dep": p.get("Dep"), "Fac": p.get("Fac"),
            "Nrrp": p.get("Nrrp"), "NMDA_ratio": p.get("NMDA_ratio"),
            "e_rev": p.get("e_rev"),
            "our_connections": int(m.sum()), "our_synapses": int(nsyn[m].sum()),
        })
    for r in rows:
        print(f"  {r['class']:16s} {r['stp']} g={r['g_nS']} U={r['Use']} "
              f"D={r['Dep']} F={r['Fac']} → 연결 {r['our_connections']:,} 시냅스 {r['our_synapses']:,}")
    json.dump({"n_connections": E, "n_synapses": int(nsyn.sum()), "by_class": rows},
              open(os.path.join(HERE, "synapse_assignment_summary.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=2)

    _fig_param_table(rows)
    _fig_stp(rows)
    _fig_our_counts(rows)
    print(f"[OK] -> {FIG}")


def _fig_param_table(rows):
    cols = ["ei", "stp", "g_nS", "tau_d", "Use", "Dep", "Fac", "Nrrp", "NMDA_ratio", "e_rev"]
    cell = [[r["class"]] + [r[c] for c in cols] for r in rows]
    fig, ax = plt.subplots(figsize=(14, 4.5)); ax.axis("off")
    t = ax.table(cellText=cell, colLabels=["클래스"] + cols, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.6)
    # 헤더/EI 색
    for (r, c), cellobj in t.get_celld().items():
        if r == 0:
            cellobj.set_facecolor("#333"); cellobj.get_text().set_color("w")
        elif c == 0:
            ei = rows[r-1]["ei"]
            cellobj.set_facecolor("#fdece0" if ei == "E" else "#eaf0f7")
    ax.set_title("V4-1  9클래스 EMS 시냅스 파라미터 (Ecker 2020 Table 3)\n"
                 "g_nS=peak전도도 · Use=방출확률 · Dep/Fac=억압/촉진시상수(ms) · Nrrp=방출자리 · NMDA_ratio",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "V4_1_param_table.png"), dpi=130)
    plt.close(fig)


def _fig_stp(rows):
    """20Hz 8-스파이크 + 회복 스파이크에 대한 정규화 PSP (STP 프로파일)."""
    spikes = list(np.arange(8) * 50.0) + [400.0 + 550.0]   # 20Hz 8발 + 550ms 후 회복
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for r in rows:
        psp = tm_psp(r["Use"], r["Dep"], r["Fac"], spikes)
        psp = psp / psp[0]
        ax.plot(range(1, len(psp) + 1), psp, "o-", color=STP_COLOR.get(r["stp"], "#999"),
                label=f"{r['class']} [{r['stp']}]", alpha=0.85)
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("스파이크 순번 (20Hz 8발 + 회복)"); ax.set_ylabel("정규화 PSP 진폭 (1번=1)")
    ax.set_title("V4-2  단기가소성(STP) 프로파일 — Tsodyks-Markram 재현\n"
                 "E1/I1 촉진(↑) · E2/I2 억압(↓) · I3 pseudo-linear")
    ax.legend(fontsize=7, ncol=1, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "V4_2_stp_profiles.png"), dpi=130)
    plt.close(fig)


def _fig_our_counts(rows):
    order = sorted(range(len(rows)), key=lambda i: -rows[i]["our_synapses"])
    cl = [rows[i]["class"] for i in order]
    syn = [rows[i]["our_synapses"] for i in order]
    conn = [rows[i]["our_connections"] for i in order]
    isE = [rows[i]["ei"] == "E" for i in order]
    y = np.arange(len(cl)); h = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.barh(y + h/2, syn, h, color=["#DD8452" if e else "#4C72B0" for e in isE], label="시냅스 수")
    ax.barh(y - h/2, conn, h, color=["#f0c9a8" if e else "#a9c0e0" for e in isE], label="연결 수")
    ax.set_yticks(y); ax.set_yticklabels(cl, fontsize=8); ax.set_xscale("log")
    for i in range(len(cl)):
        ax.text(syn[i], y[i]+h/2, f" {syn[i]:,}", va="center", fontsize=7)
    ax.set_xlabel("개수 (로그)"); ax.legend()
    ax.set_title("V4-3  우리 pruned 커넥텀 — 클래스별 연결·시냅스 수 (파라미터 적용 대상)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "V4_3_our_class_counts.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
