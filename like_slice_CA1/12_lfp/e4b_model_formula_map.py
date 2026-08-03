# -*- coding: utf-8 -*-
"""12_lfp/e4b_model_formula_map.py  —  '어떤 수식·모델로 만들어졌나' 주석 지도

E4b fEPSP 그래프(count/band)와 애니메이션 GIF가 각각 어떤 스파이크/막전류 모델과
어떤 세포외전위 공식으로 만들어지는지 소스 대조 결과를 한 장에 라벨링.
모든 수치·모델명은 소스 대조 확정값(lfp_calc.py / e4b_count.py / e4b_anim_data.py /
params_table3.py / cell_seed3_0.hoc).
실행: <ca1sim>/python.exe 12_lfp/e4b_model_formula_map.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")

C_A = "#B9722E"      # 애니메이션(교육용) 톤
C_B = "#1F6FB2"      # 실제 결과 톤
C_MID = "#5A5A57"
C_EQ = "#2C2C2A"
BG_A = "#FBF1E7"
BG_B = "#EAF2FA"
BG_MID = "#F1EFE8"
BG_EQ = "#F6F4EE"


def box(ax, x, y, w, h, fc, ec, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                fc=fc, ec=ec, lw=lw, zorder=2))


def arrow(ax, x0, y0, x1, y1, color=C_MID, lw=2.2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18,
                                 lw=lw, color=color, zorder=3, shrinkA=0, shrinkB=0))


def main():
    fig, ax = plt.subplots(figsize=(14.2, 9.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.5, 0.975, "이 그래프들은 어떤 수식·모델로 만들어지나",
            ha="center", fontsize=18, fontweight="bold", color=C_EQ)
    ax.text(0.5, 0.945, "발화/막전류 소스 모델  →  막전류 i_membrane_  →  세포외전위 공식  (모든 값 소스 대조 확정)",
            ha="center", fontsize=11, color=C_MID)

    # ================= ① 두 소스 모델 =================
    ax.text(0.5, 0.905, "①  발화 / 막전류 소스 모델", ha="center", fontsize=13,
            fontweight="bold", color=C_EQ)

    # --- 모델 A : 애니메이션 GIF (교육용 HH) ---
    box(ax, 0.035, 0.60, 0.44, 0.285, BG_A, C_A)
    ax.text(0.255, 0.862, "모델 A — 애니메이션 GIF (교육용)", ha="center", fontsize=12,
            fontweight="bold", color=C_A)
    ax.text(0.255, 0.836, "E4b_fepsp_play.gif · e4b_anim_data.py", ha="center", fontsize=8.5,
            color=C_MID, style="italic")
    ax.text(0.055, 0.808, "고전 Hodgkin–Huxley 볼-앤-스틱", fontsize=10.5, color=C_EQ, fontweight="bold")
    lines_a = [
        "• soma.insert(\"hh\")  —  Na · K · leak (HH 3전류)",
        "• dend : pas 15세그 (수동)  +  Exp2Syn 시냅스",
        "•   τ1=0.5 · τ2=3.0 ms · e=0 mV · weight 0.05",
        "• celsius = 6.3  (고전 오징어축삭 온도)",
        "• dt = 0.025 ms (고정)",
    ]
    for i, s in enumerate(lines_a):
        ax.text(0.055, 0.778 - i * 0.028, s, fontsize=9.3, color=C_EQ)
    ax.text(0.255, 0.618, "⇒  진짜 활동전위(스파이크) 발생 · Vm 약 +38mV",
            ha="center", fontsize=9.8, color=C_A, fontweight="bold")

    # --- 모델 B : 실제 결과 (BBP me-model) ---
    box(ax, 0.525, 0.60, 0.44, 0.285, BG_B, C_B)
    ax.text(0.745, 0.862, "모델 B — count · band fEPSP (실제 결과)", ha="center", fontsize=12,
            fontweight="bold", color=C_B)
    ax.text(0.745, 0.836, "E4b_count.png · E4b_band_mea · e4b_count.py", ha="center", fontsize=8.5,
            color=C_MID, style="italic")
    ax.text(0.545, 0.808, "BBP CA1  SP-PC / cACpyr  me-model", fontsize=10.5, color=C_EQ, fontweight="bold")
    lines_b = [
        "• 실측 형태학 morphology.swc · 템플릿 CA1_PC_cAC_sig",
        "• 활성 채널 12종: nax(Na) · kdr·kap·kmb·kad(K)",
        "•   kca·cagk(Ca-act K) · hd(Ih) · can·cal·cat(Ca) · cacum",
        "• 시냅스 40개: DetAMPANMDA (결정론) · PC→PC(E2)",
        "•   g=0.6nS · τ 0.2/3.0ms · NMDA×1.22 · Use 0.50",
        "• celsius = 34 · dt = 0.025 ms (고정)",
    ]
    for i, s in enumerate(lines_b):
        ax.text(0.545, 0.780 - i * 0.026, s, fontsize=9.0, color=C_EQ)
    ax.text(0.745, 0.617, "⇒  역치하 복합 EPSP (스파이크 아님 = 진짜 fEPSP)",
            ha="center", fontsize=9.6, color=C_B, fontweight="bold")

    # ================= ② 막전류 추출 =================
    box(ax, 0.13, 0.475, 0.74, 0.085, BG_MID, C_MID)
    ax.text(0.5, 0.535, "②  막전류 추출 (두 모델 공통)", ha="center", fontsize=12,
            fontweight="bold", color=C_EQ)
    ax.text(0.5, 0.508, "h.CVode().use_fast_imem(1)  →  세그먼트별  i_membrane_  기록 (단위 nA)",
            ha="center", fontsize=10.3, color=C_EQ)
    ax.text(0.5, 0.485, "부호 규약:  흥분성 시냅스 sink(막 내향) → i_membrane_ < 0 → 전극 전위 음성(fEPSP 음편향)",
            ha="center", fontsize=9.2, color=C_MID)

    arrow(ax, 0.255, 0.60, 0.35, 0.562)
    arrow(ax, 0.745, 0.60, 0.65, 0.562)

    # ================= ③ 세포외전위 공식 =================
    box(ax, 0.035, 0.045, 0.93, 0.37, BG_EQ, C_EQ, lw=1.6)
    ax.text(0.5, 0.383, "③  세포외전위 공식  (lfp_calc.py · Holt&Koch 1999 · Ness 2015)",
            ha="center", fontsize=13, fontweight="bold", color=C_EQ)
    arrow(ax, 0.5, 0.475, 0.5, 0.417)

    # 공통 골격
    ax.text(0.5, 0.345, "공통 골격 :  전극 전위 = 모든 세그먼트 막전류의 거리가중 선형합",
            ha="center", fontsize=10.2, color=C_MID)
    ax.text(0.5, 0.305, r"$V_j(t)\;=\;\sum_{i}\,M_{ji}\,I_i(t)$",
            ha="center", fontsize=17, color=C_EQ)
    ax.text(0.5, 0.272, "M = 전달행렬(전극 j × 세그먼트 i),  I = i_membrane_(nA).  좌표 µm·σ S/m·I nA → V mV (1e-3 환산 내장)",
            ha="center", fontsize=8.6, color=C_MID)

    # 좌: LSA (모델 A)
    ax.plot([0.5, 0.5], [0.06, 0.245], color="#CFC9BC", lw=1, zorder=1)
    ax.text(0.255, 0.225, "선전류원  LSA  —  모델 A(애니메이션)에 사용", ha="center",
            fontsize=10.3, fontweight="bold", color=C_A)
    ax.text(0.255, 0.176,
            r"$M=\frac{1}{4\pi\sigma L}\left[\sinh^{-1}\frac{L-s_0}{\rho}-\sinh^{-1}\frac{-s_0}{\rho}\right]$",
            ha="center", fontsize=14, color=C_EQ)
    ax.text(0.255, 0.108,
            "s0 = 전극의 세그먼트축 투영 · ρ = 축까지 수직거리(반경 하한)\nσ = 0.3 S/m (무한 균질 매질).  길이~0 세그먼트는 점전류원 1/(4πσd)",
            ha="center", fontsize=8.6, color=C_MID)

    # 우: MoI (모델 B)
    ax.text(0.745, 0.225, "MEA 3층 영상법  MoI  —  모델 B(실제 fEPSP)에 사용", ha="center",
            fontsize=10.3, fontweight="bold", color=C_B)
    ax.text(0.745, 0.178,
            r"$M=\frac{1}{4\pi\sigma_T}\left[2g(z')+2\sum_{n=1}^{N} W_{TS}^{\,n}\left(g(2nh-z')+g(2nh+z')\right)\right]$",
            ha="center", fontsize=11.5, color=C_EQ)
    ax.text(0.745, 0.128,
            r"$g(w)=\frac{1}{\sqrt{\rho^2+w^2}}\qquad W_{TS}=\frac{\sigma_T-\sigma_S}{\sigma_T+\sigma_S}=-\frac{2}{3}$",
            ha="center", fontsize=11.5, color=C_EQ)
    ax.text(0.745, 0.075,
            "유리 MEA면 z=0(σ_G=0 절연,반사 2배) · 조직 z∈[0,h] σ_T=0.3 · 식염수 σ_S=1.5 S/m\n"
            "ρ=전극-소스 수평거리 · z'=소스 높이(슬라이스 내 클램프) · n_img=20",
            ha="center", fontsize=8.6, color=C_MID)

    fig.tight_layout()
    out = os.path.join(FIG, "E4b_model_formula_map.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved:", out)


if __name__ == "__main__":
    main()
