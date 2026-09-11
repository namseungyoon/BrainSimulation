# -*- coding: utf-8 -*-
"""나-1-② STDP 예상/모식도 그림 4종 — 자리표시용(실제 데이터로 교체 예정).
가-14 프로토콜 도해(실제 프로토콜) · 가-15 예상 STDP 타이밍곡선 · 가-16 예상 버스트수 의존 · 가-17 예상 주파수 의존.
결과 모식도 3종은 'SCHEMATIC 예상·모식도(실제 데이터 아님)' 명시. matplotlib만 사용(NEURON 불필요)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

for fam in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
    try:
        matplotlib.rcParams["font.family"] = fam; break
    except Exception:
        pass
matplotlib.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); os.makedirs(HERE, exist_ok=True)
SCHEM = "SCHEMATIC  예상·모식도 (실제 데이터 아님)"
C0, C1 = "#1f77b4", "#d62728"   # ca_stp=0(원본), ca_stp=1(확장)


def tag(ax):
    ax.text(0.5, 1.16, SCHEM, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10, color="#b00", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="#fff3f3", ec="#b00", lw=1.0))


# 가-14 프로토콜 도해 (실제 프로토콜)
def fig_protocol():
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    dt = 12.0
    ax.axhline(1.0, color="#ccc", lw=0.8); ax.axhline(0.0, color="#ccc", lw=0.8)
    ax.vlines(0, 1.0, 1.8, color=C0, lw=2.2)
    ax.text(0, 1.9, "pre 단일 스파이크\n(VecStim, SC 섬유)", ha="center", va="bottom", color=C0, fontsize=9)
    post_t = [dt + j * 10.0 for j in range(4)]   # 4발 @ 100 Hz
    ax.vlines(post_t, 0.0, 0.8, color=C1, lw=2.2)
    ax.text(np.mean(post_t), -0.35, "post 버스트 4발 @ 100 Hz\n(소마 전류주입 강제발화)", ha="center", va="top", color=C1, fontsize=9)
    ax.add_patch(FancyArrowPatch((0, 0.5), (dt, 0.5), arrowstyle="<->", mutation_scale=12, color="k", lw=1.2))
    ax.text(dt / 2, 0.58, "Δt", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.text(70, 1.4, "5 Hz로 유도 반복 30회\n→ rho 판독", ha="left", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#999"))
    ax.set_xlim(-25, 120); ax.set_ylim(-0.9, 2.3); ax.set_yticks([])
    ax.set_xlabel("시간 (ms)"); ax.set_title("가-14. STDP 유도 프로토콜 도해 (SC→PC 쌍)", fontsize=11)
    for s in ["top", "left", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "stdp_protocol.png"), dpi=150); plt.close(fig)


# 가-15 예상 STDP 타이밍 곡선 (모식도)
def fig_curve():
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    dts = np.linspace(-100, 100, 400)
    # 목표(문헌) 비대칭 창: +Δt LTP, -Δt LTD (Bi&Poo류). 정성 모식도.
    ltp = 0.42 * np.exp(-dts / 18.0) * (dts > 0)
    ltd = -0.30 * np.exp(dts / 22.0) * (dts < 0)
    base = ltp + ltd                          # ca_stp=0 원본(기준)
    ext = 1.18 * ltp + 0.72 * ltd             # ca_stp=1 확장(강화쪽 증폭 가정)
    ax.axhline(0, color="#888", lw=0.8); ax.axvline(0, color="#888", lw=0.8, ls=":")
    ax.plot(dts, base, color=C0, lw=2.4, label="ca_stp=0  원본(Graupner-Brunel, 기준선)")
    ax.plot(dts, ext, color=C1, lw=2.4, ls="--", label="ca_stp=1  확장(확률방출 연동)")
    ax.text(55, 0.30, "LTP\n(pre→post)", ha="center", color="#2a7", fontsize=10, fontweight="bold")
    ax.text(-58, -0.24, "LTD\n(post→pre)", ha="center", color="#a33", fontsize=10, fontweight="bold")
    ax.set_xlabel("스파이크 시간차 Δt (ms)   [pre 기준, +면 pre 먼저]")
    ax.set_ylabel("시냅스 효능 변화 Δrho")
    ax.set_title("가-15. 예상 STDP 타이밍 곡선", fontsize=11)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax.set_ylim(-0.45, 0.6); tag(ax)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "stdp_curve_expected.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


# 가-16 예상 버스트 수 의존 (모식도)
def fig_burstnum():
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    n = np.array([1, 2, 4])
    base = np.array([-0.05, 0.07, 0.20])      # ca_stp=0: 단일=약억압/무 → 버스트=강화
    ext = np.array([0.02, 0.14, 0.27])        # ca_stp=1: 전반 강화쪽
    w = 0.34
    ax.bar(n - w / 2, base, w, color=C0, label="ca_stp=0 원본")
    ax.bar(n + w / 2, ext, w, color=C1, label="ca_stp=1 확장")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xticks(n); ax.set_xticklabels(["1발", "2발", "4발"])
    ax.set_xlabel("후시냅스 버스트 발수 (Δt 고정)"); ax.set_ylabel("시냅스 효능 변화 Δrho")
    ax.set_title("가-16. 예상 post-burst 수 의존 (Wittenberg 재현 목표)", fontsize=10.5)
    ax.text(1, -0.05, "단일=무/억압", ha="center", va="top", fontsize=8.5, color="#a33")
    ax.text(4, 0.27, "버스트=강화", ha="center", va="bottom", fontsize=8.5, color="#2a7")
    ax.legend(loc="upper left", fontsize=8.5); ax.set_ylim(-0.15, 0.35); tag(ax)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "stdp_burstnum_expected.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


# 가-17 예상 주파수 의존 (모식도)
def fig_freq():
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    f = np.array([1, 5, 20, 50])
    base = np.array([-0.18, -0.08, 0.10, 0.26])   # 저빈도 LTD → 고빈도 LTP 전환
    ext = np.array([-0.10, 0.00, 0.18, 0.31])
    ax.axhline(0, color="#888", lw=0.8)
    ax.plot(f, base, "-o", color=C0, lw=2.2, label="ca_stp=0 원본")
    ax.plot(f, ext, "--s", color=C1, lw=2.2, label="ca_stp=1 확장")
    ax.set_xscale("log"); ax.set_xticks(f); ax.set_xticklabels([str(x) for x in f])
    ax.set_xlabel("유도 짝짓기 반복 빈도 (Hz)"); ax.set_ylabel("시냅스 효능 변화 Δrho")
    ax.set_title("가-17. 예상 pairing 주파수 의존 (Sjöström 재현 목표)", fontsize=10.5)
    ax.text(1.3, -0.16, "저빈도\nLTD", ha="center", color="#a33", fontsize=9)
    ax.text(45, 0.27, "고빈도\nLTP", ha="center", color="#2a7", fontsize=9)
    ax.legend(loc="upper left", fontsize=8.5); ax.set_ylim(-0.28, 0.4); tag(ax)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "stdp_freq_expected.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig_protocol(); fig_curve(); fig_burstnum(); fig_freq()
    print("saved:", ", ".join(sorted(os.listdir(HERE))))
