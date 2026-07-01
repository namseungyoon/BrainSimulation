"""
2_paired_schematic.py — Fig.4(a): paired recording 모식 + 입력/출력 (3D 형태, 결합)
============================================================================
PVBC → PC 결합 paired recording을 **파이프라인 함수**(paired_experiment)로 구성:
  로드 → 시냅스 배치(주변표적) → 자극(전세포 트레인) → 연결(NetCon) → 실행.
전세포 PVBC 의 **실제 스파이크**가 PC 시냅스를 구동(결합) → 입력→출력 인과·동기화.

산출:
  (상) 입력: 전세포 PVBC 소마 Vm — 트레인 발화
  (중) 형태(3D): PVBC(초록계) + PC(파랑계), axon/dendrite 색 구분, 시냅스·자극·기록 마커
  (하) 출력: 후세포 PC 소마 PSP — 얇은 색=시행, 검정 굵은선=평균
  + 확대 그림(시냅스·자극·기록 위치)
실행:
  <ca1sim python> .../2_paired_schematic.py            # PNG 저장(헤드리스)
  <ca1sim python> .../2_paired_schematic.py --show     # 인터랙티브 3D 창 열기
"""
import os
import sys
import glob

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SHOW = "--show" in sys.argv
import matplotlib
if not SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.lines import Line2D
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(THIS)
ROOT = os.path.dirname(os.path.dirname(PAPER))
SHARED = os.path.join(ROOT, "shared")
sys.path.insert(0, SHARED)
sys.path.insert(0, os.path.join(PAPER, "03_synapses"))
sys.path.insert(0, THIS)
from common.nrn_env import h, MODELS_DIR            # noqa: E402
from common.plotstyle import set_korean_font         # noqa: E402
import params_table3 as P3                            # noqa: E402
from synapse_pair import spike_train                  # noqa: E402
from paired_experiment import (load_post, load_by_mtype, find_mtype_dir,   # noqa: E402
                               perisomatic_segs, place_synapses, drive_train,
                               connect, run_paired, N_SYN, T_SPIKE)

set_korean_font()
h.load_file("stdrun.hoc")
OUT = os.path.join(THIS, "figures")

PRE_MTYPE, POST_MTYPE, CLASS = "SP-PVBC", "SP-PC", "PV+->PC (I2)"
N_TRIALS = 15
# axon/dendrite 색 구분(세포별)
PRE_COL = {1: "darkgreen", 2: "yellowgreen", 3: "seagreen", 4: "seagreen"}
POST_COL = {1: "black", 2: "tab:orange", 3: "tab:blue", 4: "tab:blue"}
LEGEND = [
    Line2D([0], [0], color="seagreen", lw=2, label="PVBC 수상돌기"),
    Line2D([0], [0], color="yellowgreen", lw=2, label="PVBC 축삭"),
    Line2D([0], [0], color="tab:blue", lw=2, label="PC 수상돌기(첨단+기저)"),
    Line2D([0], [0], color="tab:orange", lw=2, label="PC 축삭"),
    Line2D([0], [0], color="k", lw=3, label="소마"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markeredgecolor="k", ms=9, label="시냅스 (PV+→PC)"),
    Line2D([0], [0], marker="*", color="w", markerfacecolor="gold", markeredgecolor="k", ms=15, label="자극 위치(전세포)"),
    Line2D([0], [0], marker="v", color="w", markerfacecolor="navy", markeredgecolor="k", ms=11, label="기록 위치(후세포 PSP)"),
]


def swc_segs3d(model_dir):
    """.swc → ({type: (N,2,3)}, soma_xyz). type 1소마/2축삭/3기저/4첨단."""
    sw = glob.glob(os.path.join(model_dir, "morphology", "*.swc"))[0]
    nodes = {}
    with open(sw) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            q = line.split()
            nodes[int(q[0])] = (int(q[1]), float(q[2]), float(q[3]), float(q[4]), int(q[6]))
    by = {1: [], 2: [], 3: [], 4: []}
    soma = (0.0, 0.0, 0.0)
    for nid, (typ, x, y, z, par) in nodes.items():
        if typ == 1:
            soma = (x, y, z)
        if par in nodes:
            px, py, pz = nodes[par][1], nodes[par][2], nodes[par][3]
            by.setdefault(typ, by.get(typ, [])).append([(px, py, pz), (x, y, z)])
    return {k: np.array(v) for k, v in by.items() if v}, np.array(soma)


def build_and_run():
    """파이프라인: 로드 → 배치 → 자극 → 연결 → 실행 (함수로 흐름 명시)."""
    pre, pv_t = load_by_mtype(PRE_MTYPE)              # 전세포(PVBC)
    post, pc_t = load_post("PC")                       # 후세포(PC)
    p = P3.CLASSES[CLASS]
    train = spike_train(n_pulses=8, freq_hz=20.0, t_start=T_SPIKE, recovery_delay=500.0)
    segs = perisomatic_segs(post, N_SYN)               # 주변표적 위치
    syns = place_synapses(post, p, segs)               # 시냅스 생성
    ics = drive_train(pre, train)                      # 전세포 발화 자극(트레인) — 참조 유지(GC 방지)
    ncs = connect(pre, syns, p["g_nS"], delay=1.0)     # 전세포 스파이크 → 시냅스 연결 — 참조 유지
    pre_vm, post_traces = run_paired(pre, post, syns, train[-1] + 120.0, N_TRIALS)
    return dict(pv_t=pv_t, pc_t=pc_t, train=train, pre_vm=pre_vm, post_traces=post_traces, p=p,
                _keep=(pre, post, syns, ics, ncs))     # 핸들 유지(GC 방지)


def draw_morph3d(ax, segs, colmap, delta=(0, 0, 0), alpha=0.8):
    d = np.array(delta, float)
    for t, arr in segs.items():
        ax.add_collection3d(Line3DCollection(arr + d, colors=colmap.get(t, "0.5"),
                                             linewidths=(2.0 if t == 1 else 0.5), alpha=alpha))


def set_equal3d(ax, pts):
    mn, mx = pts.min(0), pts.max(0); c = (mn + mx) / 2; r = (mx - mn).max() / 2 or 1.0
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def plot_all_trials(data):
    """새 그림: 입력 + 시행별(1/N…N/N) 출력 전체 + 파라미터가 어떻게 작용했는지 설명."""
    p = data["p"]; train = data["train"]
    tpre, vpre = data["pre_vm"]; post_traces = data["post_traces"]
    n = len(post_traces); cmap = plt.get_cmap("turbo"); tt = post_traces[0][0]
    g = p["g_nS"]; U = p["Use"]; D = p["Dep"]; F = p["Fac"]; N = int(p["Nrrp"])
    td = p.get("tau_d_GABAA", p.get("tau_d_AMPA"))

    fig = plt.figure(figsize=(15, 11))
    fig.suptitle(f"시행별 입력/출력 + 파라미터 작용 — {PRE_MTYPE}→{POST_MTYPE}  [{CLASS}]  (N={n})",
                 fontsize=14, fontweight="bold")
    gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1.0], wspace=0.16)
    gsl = gs[0].subgridspec(2, 1, height_ratios=[1, 8], hspace=0.07)

    # 입력(전세포 발화 트레인)
    axin = fig.add_subplot(gsl[0])
    axin.plot(tpre, vpre, color="darkgreen", lw=0.8)
    for ts in train:
        axin.axvline(ts, color="0.8", lw=0.5, ls="--")
    axin.set_xlim(T_SPIKE - 20, train[-1] + 120); axin.set_xticklabels([])
    axin.set_ylabel("입력 Vm", fontsize=8)
    axin.set_title("입력: 전세포 PVBC 발화 트레인", fontsize=10, loc="left")

    # 시행별 출력(오프셋 스택, 1/N … N/N)
    axtr = fig.add_subplot(gsl[1], sharex=axin)
    step = 1.8
    for i, (_, v) in enumerate(post_traces):
        axtr.plot(tt, v + i * step, color=cmap(i / max(n - 1, 1)), lw=0.8)
        axtr.text(T_SPIKE - 16, -70 + i * step, f"{i+1}/{n}", fontsize=7, va="center",
                  ha="right", color=cmap(i / max(n - 1, 1)), fontweight="bold")
    for ts in train:
        axtr.axvline(ts, color="0.9", lw=0.4, ls="--")
    axtr.set_xlim(T_SPIKE - 30, train[-1] + 120); axtr.set_yticks([])
    axtr.set_xlabel("시간 (ms)"); axtr.set_ylabel("시행 (아래→위 1→N, 오프셋)")
    axtr.set_title(f"출력: {n}개 시행 각각 (1/{n} … {n}/{n}) — 확률 방출로 시행마다 변동", fontsize=10, loc="left")

    # 파라미터 작용 설명
    axp = fig.add_subplot(gs[1]); axp.axis("off")
    blocks = [
        (f"● 시냅스 파라미터 (Table 3, {CLASS})", "b"),
        (f"ĝ = {g} nS  → PSP 진폭 크기", ""),
        (f"U_SE = {U}  → 첫 방출확률(낮을수록 변동 CV↑)", ""),
        (f"D = {D} ms  → 방출자원 회복(클수록 억압↑)", ""),
        (f"F = {F} ms  → 촉진 시간상수(작으면 촉진 없음)", ""),
        (f"N_RRP = {N}  → 방출 자리 수(작을수록 CV↑)", ""),
        (f"τ_decay = {td} ms  → PSP 감쇠", ""),
        ("", ""),
        ("● 이 그림에서 관측된 작용", "b"),
        (f"· 첫 PSP 최대→점차 억압: D={D}ms 큼·F={F}ms 작음 ⇒ I2 억압형", "g"),
        (f"· 시행마다 크기·실패 변동: U_SE={U}, N_RRP={N}", "g"),
        ("   ⇒ 확률 다소포 방출이라 시행마다 다름(CV↑)", "g"),
        ("· 마지막 회복 펄스 다시 커짐 ⇒ 자원 회복(D)", "g"),
        ("", ""),
        (f"색 = 시행 순서(1→{n}, turbo), 굵은 흐름 = 억압 추세", "0"),
    ]
    y = 0.99
    for txt, style in blocks:
        if txt == "":
            y -= 0.03; continue
        c = {"b": "navy", "g": "tab:green", "0": "0.45", "": "0.15"}[style]
        axp.text(0.0, y, txt, transform=axp.transAxes, color=c, va="top",
                 fontsize=11 if style == "b" else 9.5,
                 fontweight=("bold" if style == "b" else "normal"))
        y -= 0.065

    out = os.path.join(OUT, "2_paired_alltrials.png")
    fig.savefig(out, dpi=120)
    print(f"[시행별] {out}")


def main():
    os.makedirs(OUT, exist_ok=True)
    data = build_and_run()
    train = data["train"]; tpre, vpre = data["pre_vm"]; post_traces = data["post_traces"]

    pre_segs, pre_soma = swc_segs3d(find_mtype_dir(PRE_MTYPE))
    post_segs, post_soma = swc_segs3d(os.path.join(MODELS_DIR, "pyramidal",
                                      sorted(os.listdir(os.path.join(MODELS_DIR, "pyramidal")))[0]))
    delta = post_soma - pre_soma + np.array([-130.0, 0.0, 0.0])     # PVBC 를 PC 근처로
    pre_xyz = pre_soma + delta
    rs = np.random.RandomState(1)
    syn_xyz = np.array([post_soma + rs.uniform(-18, 18, 3) for _ in range(N_SYN)])

    # ── 메인 그림 (3행: Vm / 3D 형태 / PSP) ──
    fig = plt.figure(figsize=(13, 13))
    fig.suptitle(f"In silico paired recording — {PRE_MTYPE} → {POST_MTYPE}  [{CLASS}]  (결합·동기화)",
                 fontsize=14, fontweight="bold")
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 3.6, 1.5], hspace=0.25)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(tpre, vpre, color="darkgreen", lw=0.9)
    for ts in train:
        ax1.axvline(ts, color="0.8", lw=0.6, ls="--", zorder=0)
    ax1.set_xlim(T_SPIKE - 20, train[-1] + 120); ax1.set_ylabel("PVBC Vm (mV)")
    ax1.set_title("(상) 입력: 전세포 PVBC 소마 Vm — 트레인 발화(자극)", fontsize=10, loc="left")

    ax2 = fig.add_subplot(gs[1], projection="3d")
    draw_morph3d(ax2, pre_segs, PRE_COL, delta=delta, alpha=0.7)
    draw_morph3d(ax2, post_segs, POST_COL, alpha=0.9)
    ax2.scatter(syn_xyz[:, 0], syn_xyz[:, 1], syn_xyz[:, 2], c="red", s=55,
                edgecolor="k", linewidth=0.5, depthshade=False, zorder=6)
    ax2.scatter(*pre_xyz, c="gold", s=420, marker="*", edgecolor="k", linewidth=0.8, depthshade=False)
    ax2.scatter(*post_soma, c="navy", s=160, marker="v", edgecolor="w", linewidth=1.0, depthshade=False)
    ax2.text(*pre_xyz, f"  자극+Vm 기록\n  PVBC ({data['pv_t']})", color="darkgreen", fontsize=9, fontweight="bold")
    ax2.text(*post_soma, f"  PSP 기록\n  PC ({data['pc_t']})", color="navy", fontsize=9, fontweight="bold")
    post_pts = np.vstack([s.reshape(-1, 3) for s in post_segs.values()])
    pre_pts = np.vstack([s.reshape(-1, 3) for s in pre_segs.values()]) + delta
    set_equal3d(ax2, np.vstack([post_pts, pre_pts]))
    ax2.set_title("(중) 형태(3D): PVBC→PC · 시냅스·자극·기록", fontsize=10, loc="left")
    ax2.set_xlabel("x (µm)"); ax2.set_ylabel("y (µm)"); ax2.set_zlabel("z (µm)")
    ax2.view_init(elev=12, azim=-72)
    ax2.legend(handles=LEGEND, loc="upper left", fontsize=7.5, framealpha=0.9)

    ax3 = fig.add_subplot(gs[2])
    for ts in train:
        ax3.axvline(ts, color="0.85", lw=0.6, ls="--", zorder=0)
    stack = np.array([v for _, v in post_traces]); tt = post_traces[0][0]
    n_tr = len(post_traces)
    cmap = plt.get_cmap("turbo")
    for i, (_, v) in enumerate(post_traces):                       # 시행마다 다른 색
        ax3.plot(tt, v, color=cmap(i / max(n_tr - 1, 1)), lw=0.7, alpha=0.75)
    ax3.plot(tt, stack.mean(0), color="k", lw=2.6, label=f"평균 (N={n_tr})")   # 검정 굵은선=평균
    ax3.set_xlim(T_SPIKE - 20, train[-1] + 120)
    ax3.set_xlabel("시간 (ms)"); ax3.set_ylabel("PC 소마 전위 (mV)")
    ax3.set_title(f"(하) 출력: 후세포 PC 소마 PSP — 색=시행({n_tr}회, 확률방출 변동) · 검정 굵은선=평균 (IPSP, 억압)",
                  fontsize=10, loc="left")
    ax3.text(0.015, 0.07, f"N = {n_tr} 시행", transform=ax3.transAxes, fontsize=9.5,
             color="0.2", fontweight="bold")
    ax3.legend(fontsize=8, loc="lower right")

    out = os.path.join(OUT, "2_paired_schematic_PVBC-PC.png")
    fig.savefig(out, dpi=120)
    print(f"[그림] {out}")

    # ── 확대 그림(3D): 시냅스·자극·기록 위치 ──
    figz = plt.figure(figsize=(9, 8)); axz = figz.add_subplot(111, projection="3d")
    draw_morph3d(axz, pre_segs, PRE_COL, delta=delta, alpha=0.6)
    draw_morph3d(axz, post_segs, POST_COL, alpha=0.95)
    axz.scatter(syn_xyz[:, 0], syn_xyz[:, 1], syn_xyz[:, 2], c="red", s=90, edgecolor="k", depthshade=False, zorder=6)
    axz.scatter(*pre_xyz, c="gold", s=600, marker="*", edgecolor="k", linewidth=1.0, depthshade=False)
    axz.scatter(*post_soma, c="navy", s=300, marker="v", edgecolor="w", linewidth=1.3, depthshade=False)
    axz.text(*pre_xyz, "  자극(전세포)", color="darkgreen", fontsize=11, fontweight="bold")
    axz.text(*post_soma, "  기록(PSP)+시냅스", color="navy", fontsize=11, fontweight="bold")
    ctr = (pre_xyz + post_soma) / 2; half = 150.0
    axz.set_xlim(ctr[0] - half, ctr[0] + half); axz.set_ylim(ctr[1] - half, ctr[1] + half)
    axz.set_zlim(ctr[2] - half, ctr[2] + half)
    try:
        axz.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    axz.view_init(elev=12, azim=-72)
    axz.set_title("확대(3D): 시냅스 · 자극 · 기록 위치", fontsize=13, fontweight="bold")
    axz.legend(handles=LEGEND, loc="upper left", fontsize=8, framealpha=0.9)
    outz = os.path.join(OUT, "2_paired_schematic_zoom.png")
    figz.savefig(outz, dpi=130)
    print(f"[확대] {outz}")

    # 새 창: 입력 + 시행별(1/N…N/N) 출력 전체 + 파라미터 작용
    plot_all_trials(data)

    if SHOW:
        print("[show] 인터랙티브 3D 창 — 마우스로 회전. 창 닫으면 종료.")
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
