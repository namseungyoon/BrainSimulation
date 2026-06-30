"""
5_morphology_3d.py — 뉴런 형태(morphology) 3D 시각화 (.swc 직접 파싱, matplotlib)
============================================================================
4_morphology.py 는 x·y만 그린 2D 투영이었다. 여기서는 .swc 의 z 까지 살려
matplotlib(mplot3d)로 진짜 3D 로 그린다. 추가 설치 불필요.

구역 색: 소마(검정) · 첨단수상돌기 apical(빨강) · 기저수상돌기 basal(파랑) · 축삭 axon(회색).

산출(기본 실행, 헤드리스 OK):
  - figures/5_morphology_3d_montage.png : 20개 세포 3D 한 각도 몽타주(한눈에 비교)
  - figures/5_morphology_3d_<cell>.gif  : 한 세포가 빙글 도는 회전 GIF (기본 PC)

인터랙티브(로컬, 마우스로 돌리기):
  python 5_morphology_3d.py --show PC        # 이름에 PC 포함 세포를 회전 가능한 창으로
  python 5_morphology_3d.py --show 980120A   # 형태 ID 일부로도 선택

옵션:
  --cell <substr>  회전 GIF/창 대상 세포 선택(기본: PC=피라미드)
  --no-gif         GIF 생략(몽타주만)  /  --no-montage  몽타주 생략

실행: conda activate ca1sim
      python papers/01_Ecker2020_CA1_synaptic/02_neurons/5_morphology_3d.py
"""
import os
import sys
import glob
import argparse

try:                                  # Windows 콘솔(cp949)에서 µ 등 깨짐 방지
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(THIS)))
SHARED = os.path.join(ROOT, "shared")
sys.path.insert(0, SHARED)
MODELS = os.path.join(SHARED, "models")
OUT = os.path.join(THIS, "figures")

TYPE_COLOR = {1: "black", 2: "0.6", 3: "tab:blue", 4: "tab:red"}   # soma/axon/basal/apical
TYPE_Z = {1: 5, 2: 0, 3: 1, 4: 1}   # 그릴 때 소마를 위로

# --show 인 경우에만 GUI 백엔드 사용(그 외엔 Agg 로 저장)
SHOW_TARGET = None
if "--show" in sys.argv:
    i = sys.argv.index("--show")
    if i + 1 < len(sys.argv):
        SHOW_TARGET = sys.argv[i + 1]
import matplotlib
if SHOW_TARGET is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt                         # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402
from matplotlib.lines import Line2D                       # noqa: E402
from common.plotstyle import set_korean_font             # noqa: E402

set_korean_font()


# ----------------------------- .swc 파싱(z 유지) -----------------------------
def parse_swc_3d(path):
    """반환: segs dict {type: ndarray(N,2,3)} — 각 노드와 부모를 잇는 3D 선분."""
    nodes = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            # id type x y z radius parent
            nodes[int(p[0])] = (int(p[1]), float(p[2]), float(p[3]), float(p[4]), int(p[6]))
    by_type = {1: [], 2: [], 3: [], 4: []}
    allpts = []
    for nid, (typ, x, y, z, par) in nodes.items():
        allpts.append((x, y, z))
        if par in nodes:
            px, py, pz = nodes[par][1], nodes[par][2], nodes[par][3]
            by_type.setdefault(typ, by_type.get(typ, []))
            by_type[typ].append([[px, py, pz], [x, y, z]])
    allpts = np.array(allpts, float)
    ctr = allpts.mean(0)                      # 무게중심 → 원점 정렬
    segs = {}
    for t, lst in by_type.items():
        if lst:
            segs[t] = np.array(lst, float) - ctr
    zr = float(allpts[:, 2].max() - allpts[:, 2].min())
    nseg = sum(len(v) for v in segs.values())
    return segs, zr, nseg, (allpts - ctr)


def downsample(segs, target=2000):
    """선분이 너무 많으면 줄임(소마=type1 은 항상 유지). 회전 GIF 속도용."""
    total = sum(len(v) for v in segs.values())
    if total <= target:
        return segs
    step = max(1, total // target)
    out = {}
    for t, arr in segs.items():
        out[t] = arr if t == 1 else arr[::step]
    return out


# ----------------------------- 세포 목록 -----------------------------
def swc_of(model_dir):
    f = glob.glob(os.path.join(model_dir, "morphology", "*.swc"))
    return f[0] if f else None


def list_cells():
    cells = []   # (label, role, etype, morph, dir)
    pyr = os.path.join(MODELS, "pyramidal")
    for d in sorted(os.listdir(pyr)):
        p = d.split("_")
        cells.append((f"PC · {p[3] if len(p) > 3 else d}", "PC", "cACpyr",
                      p[3] if len(p) > 3 else "", os.path.join(pyr, d)))
    intd = os.path.join(MODELS, "interneurons")
    rows = []
    for d in sorted(os.listdir(intd)):
        p = d.split("_")
        et = p[2] if len(p) > 2 else "?"
        mo = p[3] if len(p) > 3 else ""
        rows.append((et, mo, os.path.join(intd, d)))
    order = {"bAC": 0, "cAC": 1, "cNAC": 2}
    for et, mo, dd in sorted(rows, key=lambda r: (order.get(r[0], 9), r[1])):
        cells.append((f"{et} · {mo}", "INT", et, mo, dd))
    return cells


# ----------------------------- 그리기 헬퍼 -----------------------------
def draw_cell(ax, segs, lw_soma=2.2, lw=0.6):
    for t in (2, 3, 4, 1):     # 소마 마지막(위로)
        if t in segs:
            lc = Line3DCollection(segs[t], colors=TYPE_COLOR[t],
                                  linewidths=(lw_soma if t == 1 else lw))
            ax.add_collection3d(lc)


# 색 범례: type 4=첨단(빨강) · 3=기저(파랑) · 2=축삭(회색) · 1=소마(검정)
LEGEND_ITEMS = [
    (4, "첨단수상돌기 apical"),
    (3, "기저수상돌기 basal"),
    (2, "축삭 axon"),
    (1, "소마 soma"),
]


def add_color_legend(ax, loc="upper right"):
    """선 색 ↔ 구역 범례를 축에 추가(프록시 핸들 사용 — Line3DCollection은 자동 인식 안 됨)."""
    handles = [Line2D([0], [0], color=TYPE_COLOR[t], lw=3, label=lab)
               for t, lab in LEGEND_ITEMS]
    ax.legend(handles=handles, loc=loc, fontsize=8, framealpha=0.85,
              labelspacing=0.3, borderpad=0.4)


def set_equal(ax, pts):
    mn = pts.min(0); mx = pts.max(0); c = (mn + mx) / 2.0; r = (mx - mn).max() / 2.0 or 1.0
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def style_axis(ax, title, planar=False):
    ax.set_title(title + (" (평면)" if planar else ""), fontsize=8)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    try:
        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.pane.set_edgecolor((1, 1, 1, 0))
    except Exception:
        pass


# ----------------------------- 산출물 -----------------------------
def make_montage(cells, path):
    n = len(cells)
    ncol = 5; nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(4.0 * ncol, 4.0 * nrow))
    fig.suptitle("뉴런 형태 3D (morphology) — 빨강=첨단 · 파랑=기저 · 회색=축삭 · 검정=소마  "
                 "[20개 세포, elev=15° azim=-70°]", fontsize=14, fontweight="bold")
    zinfo = []
    for k, (label, role, et, mo, d) in enumerate(cells):
        segs, zr, nseg, pts = parse_swc_3d(swc_of(d))
        planar = zr < 5.0
        zinfo.append((label, role, et, nseg, zr, planar))
        ax = fig.add_subplot(nrow, ncol, k + 1, projection="3d")
        draw_cell(ax, segs)
        set_equal(ax, pts)
        ax.view_init(elev=15, azim=-70)
        style_axis(ax, label, planar)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return zinfo


def find_cell(cells, substr):
    s = (substr or "PC").lower()
    for c in cells:
        if s in c[0].lower() or s in os.path.basename(c[4]).lower():
            return c
    return cells[0]


def make_gif(cell, path):
    from matplotlib.animation import FuncAnimation, PillowWriter
    label, role, et, mo, d = cell
    segs, zr, nseg, pts = parse_swc_3d(swc_of(d))
    segs = downsample(segs, 2000)
    fig = plt.figure(figsize=(6, 7))
    ax = fig.add_subplot(111, projection="3d")
    draw_cell(ax, segs)
    set_equal(ax, pts)
    style_axis(ax, f"{label}   (z={zr:.0f}µm)", zr < 5.0)
    add_color_legend(ax)

    def update(az):
        ax.view_init(elev=12, azim=az)
        return []
    anim = FuncAnimation(fig, update, frames=range(-180, 180, 10), interval=80, blit=False)
    anim.save(path, writer=PillowWriter(fps=12))
    plt.close(fig)
    return zr, nseg


def show_interactive(cell):
    label, role, et, mo, d = cell
    segs, zr, nseg, pts = parse_swc_3d(swc_of(d))
    fig = plt.figure(figsize=(7, 8))
    ax = fig.add_subplot(111, projection="3d")
    draw_cell(ax, segs)
    set_equal(ax, pts)
    style_axis(ax, f"{label}   (z={zr:.0f}µm, 선분 {nseg}개)", zr < 5.0)
    add_color_legend(ax)
    ax.view_init(elev=12, azim=-70)
    print(f"[인터랙티브] {label}  z범위={zr:.0f}µm  — 마우스로 드래그하여 회전하세요. 창을 닫으면 종료.")
    plt.show()


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", default=None, help="이 문자열이 포함된 세포를 인터랙티브 창으로")
    ap.add_argument("--cell", default="PC", help="회전 GIF 대상 세포(부분 문자열)")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--no-montage", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    cells = list_cells()

    if args.show is not None:
        show_interactive(find_cell(cells, args.show))
        return

    if not args.no_montage:
        mpath = os.path.join(OUT, "5_morphology_3d_montage.png")
        zinfo = make_montage(cells, mpath)
        print(f"[그림] {mpath}")
        print("  세포별 z-깊이(3D 유효성):")
        print(f"    {'세포':<16}{'역할':<5}{'etype':<7}{'선분':>6}{'z범위(µm)':>11}  평면?")
        for label, role, et, nseg, zr, planar in zinfo:
            print(f"    {label:<16}{role:<5}{et:<7}{nseg:>6}{zr:>11.1f}  {'예(납작)' if planar else '아니오'}")
        npl = sum(1 for *_, planar in zinfo if planar)
        print(f"  → 평면 재구성(z<5µm) {npl}/{len(zinfo)}개. 나머지는 입체 형태 확인 가능.")

    if not args.no_gif:
        cell = find_cell(cells, args.cell)
        gname = cell[0].replace(" · ", "_").replace(" ", "")
        gpath = os.path.join(OUT, f"5_morphology_3d_{gname}.gif")
        try:
            zr, nseg = make_gif(cell, gpath)
            print(f"[GIF] {gpath}  ({cell[0]}, z={zr:.0f}µm, 선분 {nseg}개)")
        except Exception as e:
            print(f"[GIF 실패] {type(e).__name__}: {e} — Pillow 미설치 시 회전 GIF만 생략됩니다(몽타주는 정상).")


if __name__ == "__main__":
    main()
