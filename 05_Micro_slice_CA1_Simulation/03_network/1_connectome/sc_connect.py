# -*- coding: utf-8 -*-
"""
03_network/1_connectome/sc_connect.py  —  3-1(a) SC 시냅스 실제 생성

각 세포의 수상돌기(.swc)를 배치+회전해 SR·SO 층 세그먼트에 Schaffer 시냅스를
확률 배치한다. **유효시냅스 그룹화(검증방식)** — Amsalem 2020(Neuron_Reduce,
Nat Commun) · Wybo 2020(eLife)의 두 조건에 맞춤:
  (1) 구역별 보존: 세그먼트 **길이 비례** 배치 → 각 그룹시냅스가 자기 국소
      구역의 몫만 대표(전역 균일 뭉침 아님). 완전형태 유지(fEPSP 공간정보 보존).
  (2) 개별 타이밍: 시냅스별 n_represented(=대표 생물학적 시냅스 수)를 저장 →
      구동 단계에서 동시발화가 아니라 그 수만큼 독립 이벤트로 발화하도록.
  - 대상: SP_PC 200 · 억제뉴런(SO_OLM 제외) 122(수렴도 비례) · SO_OLM 0
  - 판정: 국소 r로 SR(25~450)·SO(-400~-65), SP·SLM tuft·축삭 제외
  - 총 전도도 = 시냅스수 × n_represented ≈ 생물학적 수렴도(20,878/PC) 보존
결과: data/derived/sc_synapses.npz + 그림 3-1_sc_synapses.png

재료: data/derived/window_cells.npz · config/window_layout.json · data/morphology_library
실행: python 03_network/1_connectome/sc_connect.py
"""
import os
import sys
import json
import logging

import numpy as np
from scipy.spatial.transform import Rotation as Rot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
NPZ = os.path.join(ROOT, "data", "derived", "window_cells.npz")
CFG = os.path.join(ROOT, "config", "window_layout.json")
LIB = os.path.join(ROOT, "data", "morphology_library", "morphology_library")
DERIVED = os.path.join(ROOT, "data", "derived")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

# 1-5 층 경계(국소 r) → SC 대상 층
SR_R = (25.0, 450.0)      # 정단 SR
SO_R = (-400.0, -65.0)    # 기저 SO
CONV_PC, CONV_INT = 20878, 12714
# PC당 배치수(그룹화) — CLI로 조절: --npc N (구동 여유시 늘려 정밀도↑, 그룹화인자↓)
N_PC = int(sys.argv[sys.argv.index("--npc") + 1]) if "--npc" in sys.argv else 200
N_INT = int(round(N_PC * CONV_INT / CONV_PC))   # 수렴도 비례 자동 (200→122)


def dend_points(path):
    """수상돌기(basal+apical) 점 + 각 점의 세그먼트 길이(부모까지 거리)."""
    rows = np.loadtxt(path, comments="#")
    idx = rows[:, 0].astype(int); typ = rows[:, 1].astype(int)
    xyz = rows[:, 2:5].astype(np.float64); par = rows[:, 6].astype(int)
    id2row = {i: k for k, i in enumerate(idx)}
    seglen = np.zeros(len(idx))
    for k in range(len(idx)):
        p = par[k]
        if p in id2row:
            seglen[k] = np.linalg.norm(xyz[k] - xyz[id2row[p]])
    m = (typ == 3) | (typ == 4)   # basal + apical (soma·axon 제외)
    return xyz[m], seglen[m]


def main():
    d = np.load(NPZ, allow_pickle=True)
    xyz = d["xyz"]; Q = d["orientation_wxyz"]; mt = d["mtype"].astype(str); morph = d["morphology"].astype(str)
    cfg = json.load(open(CFG, encoding="utf-8"))
    fr = cfg["frame_um"]
    seed = np.array(fr["seed"]); L = np.array(fr["long_dir"])
    R = np.array(fr["radial_dir"]); Tk = np.array(fr["thick_dir"])
    Mrows = np.column_stack([L, R, Tk])
    e3 = np.array(next(e["xyz_um"] for e in cfg["electrodes"]["list"] if e["role"] == "stim"))

    rng = np.random.default_rng(0)
    cache = {}
    POST, WXYZ, UVW, LAYER, WF = [], [], [], [], []
    n_by = {"PC": 0, "INT": 0, "OLM_skip": 0}
    for i in range(len(xyz)):
        m = mt[i]
        if m == "SO_OLM":
            n_by["OLM_skip"] += 1; continue
        is_pc = (m == "SP_PC")
        N = N_PC if is_pc else N_INT
        conv = CONV_PC if is_pc else CONV_INT
        if morph[i] not in cache:
            cache[morph[i]] = dend_points(os.path.join(LIB, morph[i] + ".swc"))
        pts, seglen = cache[morph[i]]
        world = xyz[i] + Rot.from_quat(Q[i][[1, 2, 3, 0]]).apply(pts)
        loc = (world - seed) @ Mrows
        r = loc[:, 1]
        in_sr = (r >= SR_R[0]) & (r <= SR_R[1])
        in_so = (r >= SO_R[0]) & (r <= SO_R[1])
        elig = np.where(in_sr | in_so)[0]
        if len(elig) == 0:
            continue
        # (1) 구역별 보존: 세그먼트 길이 비례 배치
        wlen = seglen[elig].astype(float)
        prob = wlen / wlen.sum() if wlen.sum() > 0 else None
        pick = rng.choice(elig, N, replace=len(elig) < N, p=prob)
        POST.append(np.full(N, i, np.int64))
        WXYZ.append(world[pick]); UVW.append(loc[pick])
        LAYER.append(np.where(in_sr[pick], "SR", "SO").astype("U3"))
        WF.append(np.full(N, conv / N, np.float32))
        n_by["PC" if is_pc else "INT"] += 1

    POST = np.concatenate(POST); WXYZ = np.concatenate(WXYZ); UVW = np.concatenate(UVW)
    LAYER = np.concatenate(LAYER); WF = np.concatenate(WF)
    distE3 = np.linalg.norm(WXYZ - e3, axis=1).astype(np.float32)

    np.savez_compressed(os.path.join(DERIVED, "sc_synapses.npz"),
                        post_gid=POST, xyz=WXYZ.astype(np.float32), uvw=UVW.astype(np.float32),
                        layer=LAYER, n_represented=WF, dist_e3=distE3,
                        n_pc=N_PC, n_int=N_INT, conv_pc=CONV_PC, conv_int=CONV_INT, e3_xyz=e3)

    print(f"=== 3-1(a) SC 시냅스 생성 (검증방식: 구역별 보존 + 개별타이밍 대비) ===")
    print(f"[배치 세포] PC {n_by['PC']:,} · 억제(비OLM) {n_by['INT']:,} · OLM 제외 {n_by['OLM_skip']:,}")
    print(f"[총 SC 시냅스] {len(POST):,}개 (PC {N_PC}/셀·INT {N_INT}/셀) — 세그먼트 길이비례 배치")
    print(f"[층 분포] SR {np.sum(LAYER=='SR'):,} · SO {np.sum(LAYER=='SO'):,}")
    print(f"[그룹화] n_represented ≈ {WF.mean():.1f} (시냅스 1개가 대표하는 생물학적 시냅스 수)")
    print(f"         → 3-2 전도도 = n_represented×g_single · 3-3 구동 = n_represented개 독립이벤트")
    print(f"[E3 거리] 중앙 {np.median(distE3):.0f}µm · 200µm 이내 {np.sum(distE3<200):,}개")
    print(f"\n[3-1] 저장 -> data/derived/sc_synapses.npz")
    fig_syn(UVW, LAYER, distE3, cfg, e3, seed, Mrows, len(POST))
    print(f"[3-1] 그림 -> {FIG}/3-1_sc_synapses.png")


def fig_syn(UVW, LAYER, distE3, cfg, e3, seed, Mrows, n):
    w = cfg["window_um"]; c = w["center_local"]
    e3loc = (e3 - seed) @ Mrows
    rng = np.random.default_rng(1)
    s = rng.choice(len(UVW), min(45000, len(UVW)), replace=False)
    U, Rr, W = UVW[s, 0], UVW[s, 1], UVW[s, 2]; Ls = LAYER[s]
    col = {"SR": "#55A868", "SO": "#4C72B0"}
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    for panel, (X, Y, xl, yl, cx, cy, hx, hy, ex, ey, ttl) in enumerate([
        (U, Rr, "종축 u (µm)", "층관통 r (µm)", c["u"], c["r"], w["long"]/2, w["radial"]/2, e3loc[0], e3loc[1], "u-r (정면)"),
        (Rr, W, "층관통 r (µm)", "두께 w (µm)", c["r"], c["w"], w["radial"]/2, w["thick"]/2, e3loc[1], cfg["electrodes"]["mea_face_w_um"], "r-w (단면)")]):
        a = ax[panel]
        for Ln in ["SR", "SO"]:
            m = Ls == Ln
            a.scatter(X[m], Y[m], s=2, c=col[Ln], alpha=0.25, linewidths=0, label=f"SC on {Ln}")
        a.add_patch(Rectangle((cx-hx, cy-hy), 2*hx, 2*hy, fill=False, ec="black", lw=1.6, ls="--"))
        a.scatter([ex], [ey], s=240, marker="*", c="red", edgecolors="black", zorder=6)
        a.annotate("E3(자극)", (ex, ey), fontsize=9, ha="center", va="bottom", xytext=(0, 10), textcoords="offset points")
        a.set_aspect("equal"); a.set_xlabel(xl); a.set_ylabel(yl); a.set_title(ttl)
    ax[0].legend(loc="upper right", fontsize=9, markerscale=4)
    fig.suptitle(f"3-1(a)  Schaffer 시냅스 배치 — {n:,}개 (표본 {len(s):,} 표시) · SR+SO · E3", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "3-1_sc_synapses.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
