"""
pipeline_flowchart.py — 시뮬레이션 파이프라인 플로우차트(함수명 포함, 보고서용)
============================================================================
두 "최종 시뮬레이션"의 함수 흐름을 한 장에:
  (좌) 네트워크 마이크로서킷  2_run_and_analyze.py → network_lib.*
  (우) paired recording        2_paired_schematic.py → paired_experiment.*
  (하) 공유 빌딩블록           cell_loader · params_table3 · synapse_pair
실행: <ca1sim python> papers/01_Ecker2020_CA1_synaptic/slides/pipeline_flowchart.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(THIS)))
sys.path.insert(0, os.path.join(ROOT, "shared"))
from common.plotstyle import set_korean_font   # noqa: E402
set_korean_font()

NAVY = "#1E2761"; BLUE = "#DCE6F7"; GREEN = "#DCF0DC"; GRAY = "#ECECEC"; PHASE = "#C9D6F0"


def box(ax, cx, cy, w, h, title, sub, fc, ec="0.3", tfs=10, sfs=7.6):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.004,rounding_size=0.010",
                                fc=fc, ec=ec, lw=1.3, zorder=2))
    ax.text(cx, cy + h * 0.18, title, ha="center", va="center", fontsize=tfs,
            fontweight="bold", color=NAVY, zorder=3)
    if sub:
        ax.text(cx, cy - h * 0.24, sub, ha="center", va="center", fontsize=sfs,
                color="0.25", zorder=3)


def arrow(ax, x1, y1, x2, y2, color="0.35", ls="-", lw=1.6):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls), zorder=1)


def vchain(ax, cx, w, h, ys, items, fc):
    for i, (cy, (t, s)) in enumerate(zip(ys, items)):
        box(ax, cx, cy, w, h, t, s, fc)
        if i:
            arrow(ax, cx, ys[i - 1] - h / 2, cx, cy + h / 2)


def main():
    fig, ax = plt.subplots(figsize=(16, 11.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.suptitle("축소 CA1 in silico — 시뮬레이션 파이프라인 (함수 흐름)", fontsize=16, fontweight="bold")

    # ── 상단: 프로젝트 단계 ──
    phases = ["01_setup\n환경·mod", "02_neurons\n뉴런 검증", "03_synapses\n시냅스 9클래스",
              "04_network\n마이크로서킷", "05_paired\nrecording"]
    n = len(phases); pw = 0.165; x0 = 0.06
    for i, ph in enumerate(phases):
        cx = x0 + i * (pw + 0.02) + pw / 2
        box(ax, cx, 0.945, pw, 0.045, ph.split("\n")[0], ph.split("\n")[1], PHASE, tfs=9, sfs=7.5)
        if i:
            arrow(ax, cx - pw - 0.02 + pw, 0.945, cx - pw / 2, 0.945, lw=1.3)

    W, H = 0.40, 0.062
    ax.text(0.25, 0.885, "■ 네트워크 시뮬레이션  (2_run_and_analyze.py → network_lib)",
            ha="center", fontsize=11.5, fontweight="bold", color=NAVY)
    ax.text(0.75, 0.885, "■ paired recording  (2_paired_schematic.py → paired_experiment)",
            ha="center", fontsize=11.5, fontweight="bold", color=NAVY)

    ys = [0.82, 0.725, 0.63, 0.535, 0.44, 0.345, 0.25]
    # 좌: 네트워크
    net_items = [
        ("1_build_connectivity.py : build()", "세포 배치·거리의존 연결·9클래스 배정 → connectivity.json"),
        ("load_connectivity()  [+ subsample()]", "연결도 읽기 (--demo 면 타입균형 축소)"),
        ("load_representatives()", "대표 4종(PC·PV·cAC·bAC) 모델 폴더"),
        ("build_cells()", "세포 인스턴스화  → cell_loader.load_cell"),
        ("wire_synapses()", "9클래스 EMS 시냅스 연결  → params_table3, synapse_pair"),
        ("add_external_drive() → record_spikes()", "Poisson 외부 구동 + 스파이크 기록"),
        ("run_network() → analyze_activity()", "고정 dt 실행 → raster · e-type별 발화율"),
    ]
    vchain(ax, 0.25, W, H, ys, net_items, BLUE)

    # 우: paired recording
    pr_items = [
        ("build_and_run()", "PVBC→PC 결합 paired recording 구성"),
        ("load_by_mtype() / load_post()", "전세포·후세포 로드  → cell_loader.load_cell"),
        ("perisomatic_segs() → place_synapses()", "주변표적 위치 + 9클래스 시냅스  → params_table3"),
        ("drive_train()", "전세포 발화 트레인(자극, IClamp 펄스)"),
        ("connect()", "전세포 소마 스파이크 → 시냅스 NetCon"),
        ("run_paired()", "확률 시행 실행 → 전세포 Vm · 후세포 PSP"),
        ("plot_all_trials() / 그림", "입력 · 시행별 출력(1/N…N/N) · 파라미터 작용"),
    ]
    vchain(ax, 0.75, W, H, ys, pr_items, GREEN)

    # ── 하단: 공유 빌딩블록 ──
    ax.text(0.5, 0.165, "공유 빌딩블록 (shared/common · 03_synapses)", ha="center",
            fontsize=11, fontweight="bold", color=NAVY)
    sh = [("cell_loader.load_cell()", "me-model 로드"),
          ("params_table3.CLASSES", "9클래스 Table 3 파라미터"),
          ("synapse_pair.build_synapse()", "EMS 확률 시냅스 생성")]
    sx = [0.22, 0.5, 0.78]
    for cx, (t, s) in zip(sx, sh):
        box(ax, cx, 0.10, 0.27, 0.058, t, s, GRAY, tfs=9, sfs=7.3)
    # 의존 화살표(점선)
    arrow(ax, 0.25, 0.535 - H / 2, 0.22, 0.10 + 0.058 / 2, color="0.6", ls=":", lw=1.2)   # build_cells→load_cell
    arrow(ax, 0.25, 0.44 - H / 2, 0.5, 0.10 + 0.058 / 2, color="0.6", ls=":", lw=1.2)      # wire→params
    arrow(ax, 0.75, 0.63 - H / 2, 0.78, 0.10 + 0.058 / 2, color="0.6", ls=":", lw=1.2)     # place→build_synapse
    arrow(ax, 0.75, 0.725 - H / 2, 0.22, 0.10 + 0.058 / 2, color="0.6", ls=":", lw=1.2)    # load_post→load_cell

    out = os.path.join(THIS, "pipeline_flowchart.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[플로우차트] {out}")


if __name__ == "__main__":
    main()
