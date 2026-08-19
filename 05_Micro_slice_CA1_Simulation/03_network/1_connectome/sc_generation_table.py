# -*- coding: utf-8 -*-
"""
03_network/1_connectome/sc_generation_table.py  —  3-1 SC 시냅스 생성 규모 표

Hub 수렴도(convergence)로 우리 창 5,610세포에 생성될 Schaffer 시냅스 수를 계산.
전 생물학적 수렴도는 계산상 과대 → 축소(스케일) 필요를 함께 표시.
결과: figures/3-1_sc_generation.png  (+ 콘솔 수치)

실행: python 03_network/1_connectome/sc_generation_table.py
"""
import os
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

# 창 세포 수 (2-1 결과)
N_PC = 5040
N_INT = 570
# Hub 수렴도(세포당 SC 시냅스) + STP (SC 재구성 데이터)
CONV_PC = 20878        # SC → CA1 PC
CONV_INT = 12714       # SC → CB1R+/CB1R- 억제뉴런
STP = {"PC": "U0.14 D186 F129 NRRP12", "INT": "U0.11 D307 F195 NRRP4~8"}


def main():
    full_pc = CONV_PC * N_PC
    full_int = CONV_INT * N_INT   # 상한(전 억제뉴런 가정)
    full_tot = full_pc + full_int
    print(f"[전 생물학적 수렴도 기준]")
    print(f"  SC→PC   : {CONV_PC:,} × {N_PC:,} = {full_pc:,}")
    print(f"  SC→INT  : {CONV_INT:,} × {N_INT:,} = {full_int:,} (상한)")
    print(f"  합계    : ~{full_tot:,}  (≈ {full_tot/1e6:.0f}M) → 직접 시뮬 과대")

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("off")
    rows = [
        ["후시냅스 대상", "수렴도\n(SC시냅스/세포)", "세포수", "총 SC 시냅스", "배치 층", "단기가소성(STP)"],
        ["추체 SP_PC", f"{CONV_PC:,}", f"{N_PC:,}", f"{full_pc:,}", "SR 정단 + SO 기저", STP["PC"]],
        ["억제뉴런 CB1R±\n(CCKBC/PVBC 등)", f"{CONV_INT:,}", f"≤{N_INT:,}", f"≤{full_int:,}", "SR/SO 수상돌기", STP["INT"]],
        ["합계(전 생물학적)", "—", f"{N_PC+N_INT:,}", f"~{full_tot:,}", "—", "—"],
    ]
    tb = ax.table(cellText=rows, cellLoc="center", loc="center", bbox=[0.0, 0.42, 1.0, 0.5])
    tb.auto_set_font_size(False); tb.set_fontsize(11); tb.scale(1, 2.1)
    for j in range(6):
        tb[0, j].set_facecolor("#333"); tb[0, j].set_text_props(color="white", fontweight="bold")
    for j in range(6):
        tb[3, j].set_facecolor("#FBE9E7"); tb[3, j].set_text_props(fontweight="bold")
    tb[1, 0].set_text_props(fontweight="bold")

    ax.text(0.5, 0.95, "3-1(a)  Schaffer 시냅스 생성 규모 (Hub 수렴도 기준)", ha="center",
            fontsize=15, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.36,
            "생성 방식: 각 세포의 수상돌기 세그먼트 중 SR·SO 층 안에 있는 것을 골라,\n"
            "대상별 수렴도만큼 SC 시냅스를 확률적으로 배치 (SLM tuft·축삭 제외)",
            ha="center", va="top", fontsize=11, transform=ax.transAxes, color="#333")
    ax.text(0.5, 0.20,
            "주의: 전 생물학적 수렴도면 총 ~1.1억 시냅스 → NEURON 직접 시뮬 불가.\n"
            "→ 슬라이스 축소 적용: (a) 400µm 슬래브 내 잔존 수상돌기 비율로 감소,\n"
            "   (b) 활성화는 E3 근처 다발만(자극세기), (c) 필요시 유효시냅스로 그룹화",
            ha="center", va="top", fontsize=10.5, transform=ax.transAxes,
            color="#B71C1C", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(FIG, "3-1_sc_generation.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"[3-1] 저장 -> {out}")


if __name__ == "__main__":
    main()
