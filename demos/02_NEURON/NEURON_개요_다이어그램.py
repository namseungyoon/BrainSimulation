# -*- coding: utf-8 -*-
"""
NEURON 시뮬레이터 설명용 다이어그램 생성 스크립트
matplotlib 로 직접 그려서 PNG 4장으로 저장한다.
  1) NEURON_01_개요와특징.png      - 개요 / 주요 기능 / 활용 분야
  2) NEURON_02_시뮬레이션구조.png  - Section / Segment / Topology 구조
  3) NEURON_03_워크플로우.png       - 모델링 -> 시뮬레이션 -> 분석 흐름
  4) NEURON_04_NMODL메커니즘.png    - .mod (NMODL) 파일 처리 과정
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 공통 색상 팔레트
C_BLUE = "#2E5E8C"
C_LBLUE = "#D6E6F2"
C_GREEN = "#3A7D44"
C_LGREEN = "#DDEFD9"
C_ORANGE = "#D9822B"
C_LORANGE = "#FBE6CF"
C_PURPLE = "#6A4C93"
C_LPURPLE = "#E6DBF2"
C_GRAY = "#555555"


def rounded_box(ax, xy, w, h, text, fc, ec, fontsize=11, fontcolor="black",
                weight="normal", ha="center"):
    """둥근 모서리 박스 + 텍스트"""
    box = FancyBboxPatch((xy[0], xy[1]), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.08",
                         linewidth=1.6, edgecolor=ec, facecolor=fc)
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=fontcolor, weight=weight, wrap=True)
    return box


def arrow(ax, p1, p2, color=C_GRAY, lw=2.0, style="-|>", rad=0.0):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=18,
                        linewidth=lw, color=color,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)


# ============================================================
# 1) 개요와 특징
# ============================================================
def fig1_overview():
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(6.5, 7.6, "NEURON 시뮬레이터 — 개요와 특징", ha="center",
            fontsize=20, weight="bold", color=C_BLUE)
    ax.text(6.5, 7.15,
            "생물물리학적으로 사실적인 단일 뉴런 ~ 대규모 신경망 시뮬레이션 도구",
            ha="center", fontsize=12, color=C_GRAY)

    # 중앙 정의 박스
    rounded_box(ax, (4.3, 5.7), 4.4, 1.0,
                "NEURON\n(Yale Univ., Hines & Carnevale)\n전기생리 모델 전용 시뮬레이터",
                C_LBLUE, C_BLUE, fontsize=12, weight="bold")

    # 4개 특징 카드
    cards = [
        ((0.4, 3.9), "핵심 개념", C_LGREEN, C_GREEN,
         "• 케이블 방정식 기반\n• 막전위/이온 채널 수치해석\n• 구획 모델(compartmental)"),
        ((3.6, 3.9), "주요 기능", C_LORANGE, C_ORANGE,
         "• HH 등 이온 채널 모델\n• 시냅스 / 가소성\n• 네트워크(스파이크 전달)"),
        ((6.8, 3.9), "사용 방식", C_LPURPLE, C_PURPLE,
         "• Python / HOC 스크립트\n• GUI(그래프·셀빌더)\n• NMODL(.mod) 확장"),
        ((10.0, 3.9), "병렬/성능", C_LBLUE, C_BLUE,
         "• 가변시간 적분(CVODE)\n• MPI 병렬 시뮬레이션\n• 대규모 네트워크"),
    ]
    for (xy, title, fc, ec, body) in cards:
        rounded_box(ax, xy, 2.6, 1.6, "", fc, ec)
        ax.text(xy[0] + 1.3, xy[1] + 1.35, title, ha="center",
                fontsize=12, weight="bold", color=ec)
        ax.text(xy[0] + 0.15, xy[1] + 0.75, body, ha="left", va="center",
                fontsize=9.5, color="black")
        arrow(ax, (6.5, 5.7), (xy[0] + 1.3, xy[1] + 1.6), color=ec, lw=1.4)

    # 활용 분야 띠
    rounded_box(ax, (0.4, 1.0), 12.2, 2.0, "", "#F4F4F4", C_GRAY)
    ax.text(6.5, 2.75, "활용 분야", ha="center", fontsize=13,
            weight="bold", color=C_GRAY)
    fields = [
        "단일 뉴런\n전기생리 재현", "시냅스 가소성\n(STP/STDP)",
        "신경회로망\n(피질·해마)", "약물·질환\n효과 모델링",
        "뇌 시뮬레이션\n(Blue Brain 등)",
    ]
    for i, f in enumerate(fields):
        x = 1.1 + i * 2.35
        rounded_box(ax, (x, 1.25), 2.0, 1.1, f, "white", C_BLUE,
                    fontsize=9.5)

    plt.tight_layout()
    fig.savefig("NEURON_01_개요와특징.png", dpi=130, bbox_inches="tight")
    print("[saved] NEURON_01_개요와특징.png")
    plt.close(fig)


# ============================================================
# 2) 시뮬레이션 구조 (Section / Segment / Topology)
# ============================================================
def fig2_structure():
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(6.5, 7.6, "NEURON 시뮬레이션 구조 — 뉴런의 표현 방법",
            ha="center", fontsize=20, weight="bold", color=C_BLUE)

    # --- (A) 형태학적 계층 ---
    ax.text(3.2, 6.9, "① 형태(Topology) — Section 연결", ha="center",
            fontsize=13, weight="bold", color=C_GREEN)

    # soma + dendrite + axon 모식도
    rounded_box(ax, (2.4, 5.6), 1.5, 0.9, "soma", C_LORANGE, C_ORANGE,
                fontsize=11, weight="bold")
    rounded_box(ax, (0.3, 5.75), 1.8, 0.55, "dendrite", C_LGREEN, C_GREEN,
                fontsize=10)
    rounded_box(ax, (4.2, 5.75), 1.9, 0.55, "axon", C_LBLUE, C_BLUE,
                fontsize=10)
    arrow(ax, (2.1, 6.05), (2.4, 6.05), color=C_GRAY)
    arrow(ax, (3.9, 6.05), (4.2, 6.05), color=C_GRAY)
    ax.text(3.2, 5.2, "soma.connect(dend), axon.connect(soma)",
            ha="center", fontsize=9, color=C_GRAY, style="italic")

    # --- (B) Section -> Segment (nseg) ---
    ax.text(9.7, 6.9, "② 구획화 — Section을 Segment로 분할",
            ha="center", fontsize=13, weight="bold", color=C_ORANGE)

    # 하나의 Section 막대
    sec_x, sec_y, sec_w = 7.0, 5.7, 5.4
    rounded_box(ax, (sec_x, sec_y), sec_w, 0.7, "", C_LORANGE, C_ORANGE)
    ax.text(sec_x + sec_w / 2, sec_y + 1.0, "1개 Section (nseg=5)",
            ha="center", fontsize=10, color=C_ORANGE, weight="bold")
    # 분할선
    n = 5
    for i in range(1, n):
        xx = sec_x + sec_w * i / n
        ax.plot([xx, xx], [sec_y, sec_y + 0.7], color=C_ORANGE, lw=1.2,
                ls="--")
    for i in range(n):
        cx = sec_x + sec_w * (i + 0.5) / n
        ax.text(cx, sec_y + 0.35, f"seg\n{i+1}", ha="center", va="center",
                fontsize=8, color="black")
    # 0 ~ 1 위치 좌표
    ax.text(sec_x, sec_y - 0.25, "x=0", ha="center", fontsize=9,
            color=C_GRAY)
    ax.text(sec_x + sec_w, sec_y - 0.25, "x=1", ha="center", fontsize=9,
            color=C_GRAY)
    ax.text(9.7, 5.0, "각 segment 마다 막전위 V, 이온 농도 등을 개별 계산",
            ha="center", fontsize=9, color=C_GRAY, style="italic")

    # --- (C) 한 segment 의 등가회로 ---
    ax.text(6.5, 4.3, "③ 각 Segment = 등가 RC 회로 + 이온 채널 (케이블 방정식)",
            ha="center", fontsize=13, weight="bold", color=C_PURPLE)
    rounded_box(ax, (1.0, 2.3), 11.0, 1.7, "", C_LPURPLE, C_PURPLE)
    ax.text(6.5, 3.6,
            r"$C_m \dfrac{dV}{dt} = -\,\sum_{ion} g_{ion}(V - E_{ion})"
            r" + \dfrac{1}{r_a}\dfrac{\partial^2 V}{\partial x^2} + I_{stim}$",
            ha="center", fontsize=16, color="black")
    ax.text(6.5, 2.75,
            "막 전기용량(Cm) · 이온 채널 전도도(g) · 축방향 저항(ra) · 시냅스/자극 전류",
            ha="center", fontsize=10, color=C_GRAY)

    # --- (D) 객체 계층 요약 ---
    ax.text(6.5, 1.75, "주요 객체 계층", ha="center", fontsize=12,
            weight="bold", color=C_BLUE)
    chain = ["Section\n(형태 단위)", "Segment\n(계산 단위)",
             "Mechanism\n(채널/시냅스)", "PointProcess\n(시냅스·자극)",
             "NetCon\n(스파이크 연결)"]
    for i, c in enumerate(chain):
        x = 0.7 + i * 2.45
        rounded_box(ax, (x, 0.5), 2.1, 0.95, c, "white", C_BLUE,
                    fontsize=9.5)
        if i < len(chain) - 1:
            arrow(ax, (x + 2.1, 0.97), (x + 2.45, 0.97), color=C_GRAY)

    plt.tight_layout()
    fig.savefig("NEURON_02_시뮬레이션구조.png", dpi=130, bbox_inches="tight")
    print("[saved] NEURON_02_시뮬레이션구조.png")
    plt.close(fig)


# ============================================================
# 3) 워크플로우
# ============================================================
def fig3_workflow():
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(6.5, 7.6, "NEURON 시뮬레이션 워크플로우",
            ha="center", fontsize=20, weight="bold", color=C_BLUE)

    steps = [
        ("1. 형태 정의", C_GREEN, C_LGREEN,
         "Section 생성\nsoma = h.Section()\n길이(L)·지름(diam)·nseg\nconnect 로 연결"),
        ("2. 생물물리 설정", C_ORANGE, C_LORANGE,
         "이온 채널 삽입\nsoma.insert('hh')\nCm, Ra, 전도도\n(.mod 메커니즘)"),
        ("3. 자극·시냅스", C_PURPLE, C_LPURPLE,
         "IClamp 전류주입\nExp2Syn 시냅스\nNetStim 스파이크원\nNetCon 연결"),
        ("4. 기록 설정", C_BLUE, C_LBLUE,
         "h.Vector()\nv.record(soma(0.5)._ref_v)\n시간·막전위·\n스파이크 기록"),
        ("5. 실행", C_GREEN, C_LGREEN,
         "h.finitialize(-65)\nh.continuerun(tstop)\n시간 적분\n(고정/가변 dt)"),
        ("6. 분석·시각화", C_ORANGE, C_LORANGE,
         "matplotlib plot\n막전위 파형\n래스터/주파수\n결과 저장"),
    ]
    # 2행 3열 배치 + 화살표
    positions = [(0.5, 4.3), (4.55, 4.3), (8.6, 4.3),
                 (8.6, 1.4), (4.55, 1.4), (0.5, 1.4)]
    bw, bh = 3.7, 2.2
    for (title, ec, fc, body), (x, y) in zip(steps, positions):
        rounded_box(ax, (x, y), bw, bh, "", fc, ec)
        ax.text(x + bw / 2, y + bh - 0.32, title, ha="center",
                fontsize=12.5, weight="bold", color=ec)
        ax.text(x + 0.25, y + (bh - 0.7) / 2, body, ha="left", va="center",
                fontsize=10, color="black")

    # 화살표: 1->2->3 (윗줄 오른쪽), 3->4 (아래로), 4->5->6 (아랫줄 왼쪽)
    arrow(ax, (0.5 + bw, 4.3 + bh / 2), (4.55, 4.3 + bh / 2))
    arrow(ax, (4.55 + bw, 4.3 + bh / 2), (8.6, 4.3 + bh / 2))
    arrow(ax, (8.6 + bw / 2, 4.3), (8.6 + bw / 2, 1.4 + bh))
    arrow(ax, (8.6, 1.4 + bh / 2), (4.55 + bw, 1.4 + bh / 2))
    arrow(ax, (4.55, 1.4 + bh / 2), (0.5 + bw, 1.4 + bh / 2))

    ax.text(6.5, 0.7,
            "※ 형태(구조) → 물성(채널) → 입력(자극/시냅스) → 기록 → 실행 → 분석 의 순서",
            ha="center", fontsize=10.5, color=C_GRAY, style="italic")

    plt.tight_layout()
    fig.savefig("NEURON_03_워크플로우.png", dpi=130, bbox_inches="tight")
    print("[saved] NEURON_03_워크플로우.png")
    plt.close(fig)


# ============================================================
# 4) NMODL (.mod) 메커니즘
# ============================================================
def fig4_nmodl():
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(6.5, 7.6, "NMODL (.mod) — 메커니즘(모델) 정의와 처리 과정",
            ha="center", fontsize=19, weight="bold", color=C_BLUE)

    # 왼쪽: .mod 파일 구조 카드
    rounded_box(ax, (0.5, 2.3), 5.3, 4.6, "", "#F7F7F7", C_GREEN)
    ax.text(3.15, 6.55, ".mod 파일의 블록 구조", ha="center",
            fontsize=13, weight="bold", color=C_GREEN)
    blocks = [
        ("NEURON", "공개 변수·전류 선언 (SUFFIX, RANGE)"),
        ("PARAMETER", "고정 파라미터 (전도도, 시정수 등)"),
        ("STATE", "상태 변수 (m, h, n / u, x ...)"),
        ("INITIAL", "초기값 설정"),
        ("BREAKPOINT", "매 스텝 전류 계산 (SOLVE)"),
        ("DERIVATIVE", "미분방정식 (상태변수 변화)"),
        ("NET_RECEIVE", "시냅스 이벤트 처리 (스파이크 수신)"),
    ]
    for i, (b, d) in enumerate(blocks):
        y = 6.0 - i * 0.52
        ax.text(0.85, y, b, ha="left", va="center", fontsize=10.5,
                weight="bold", color=C_BLUE, family="monospace")
        ax.text(2.55, y, d, ha="left", va="center", fontsize=8.8,
                color="black")

    # 오른쪽: 처리 파이프라인
    ax.text(9.5, 6.55, "처리 파이프라인", ha="center",
            fontsize=13, weight="bold", color=C_ORANGE)
    pipe = [
        ("synapse.mod\n(NMODL 코드 작성)", C_LGREEN, C_GREEN),
        ("nrnivmodl / mknrndll\n(컴파일 → C → 바이너리)", C_LORANGE, C_ORANGE),
        ("NEURON 실행 시 자동 로드\n(메커니즘 등록)", C_LPURPLE, C_PURPLE),
        ("Python/HOC에서 사용\nsoma.insert('hh')\nsyn = h.Exp2Syn(...)", C_LBLUE, C_BLUE),
    ]
    px, pw, ph = 7.0, 5.2, 0.95
    for i, (txt, fc, ec) in enumerate(pipe):
        y = 5.6 - i * 1.3
        rounded_box(ax, (px, y), pw, ph, txt, fc, ec, fontsize=10)
        if i < len(pipe) - 1:
            arrow(ax, (px + pw / 2, y), (px + pw / 2, y - 0.35), color=C_GRAY)

    # 하단 설명
    rounded_box(ax, (0.5, 0.6), 12.0, 1.3, "", C_LBLUE, C_BLUE)
    ax.text(6.5, 1.5, "핵심 요약", ha="center", fontsize=12,
            weight="bold", color=C_BLUE)
    ax.text(6.5, 1.0,
            ".mod 파일 = 이온 채널·시냅스 등 '메커니즘(모델)'의 수식 정의서 →"
            " 컴파일 후 NEURON에 붙여 사용  |  HH·STP·STDP 모두 .mod로 구현 가능",
            ha="center", fontsize=10, color="black")

    plt.tight_layout()
    fig.savefig("NEURON_04_NMODL메커니즘.png", dpi=130, bbox_inches="tight")
    print("[saved] NEURON_04_NMODL메커니즘.png")
    plt.close(fig)


if __name__ == "__main__":
    fig1_overview()
    fig2_structure()
    fig3_workflow()
    fig4_nmodl()
    print("완료: 4개 다이어그램 생성")
