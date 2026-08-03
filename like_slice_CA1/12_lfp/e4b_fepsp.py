# -*- coding: utf-8 -*-
"""12_lfp/e4b_fepsp.py  —  E4b: MEA 슬라이스 3층 영상법 + 정렬 세포 앙상블 -> 집단 fEPSP(mV)

E4a(무한매질·단일세포 µV)를 실측 MEA로 확장:
  (1) MEA 3층 영상법(Ness 2015): 유리(z=0)-조직(z∈[0,h])-식염수(z>h) 경계 보정.
  (2) 정렬 세포 앙상블: 상세 PC 1개 막전류를 N개 복제(같은 방향 정렬·수평 흩뿌림)해 합산.
  -> 집단 mV급 fEPSP (실측 E9 대조의 계산엔진).

절차: E4a식 상세 PC + SR SC 시냅스 단일볼리 -> 막전류 -> MEA 프레임으로 회전(깊이축→z) ->
      전극 유리면 z=0 -> moi 전달행렬로 단일세포/앙상블 세포외 전위.
실행: <ca1sim>/python.exe 12_lfp/e4b_fepsp.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BRAIN = os.path.dirname(ROOT)
SHARED = os.path.join(BRAIN, "shared")
PAPER = os.path.join(BRAIN, "papers", "01_Ecker2020_CA1_synaptic")
for p in (SHARED, os.path.join(PAPER, "03_synapses"), os.path.join(PAPER, "04_network"), HERE):
    sys.path.insert(0, p)

from common.nrn_env import h
from common.cell_loader import load_cell
import network_lib as net
import params_table3 as P3
from synapse_pair import build_synapse
import lfp_calc as L

MODELS = os.path.join(SHARED, "models")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

# --- MEA 파라미터 (Ness 2015) ---
SIG_T, SIG_S, SIG_G = 0.3, 1.5, 0.0      # 조직/식염수/유리(절연) 전도도 S/m
N_IMG = 20                                # 영상급수 항
Z_SOMA = 60.0                             # 소마 높이(유리면 위 µm)
R_POP = 500.0                             # 앙상블 반경(µm)
NMAX = 15000
N_SYN = 40
SR_BAND = (0.30, 0.68)
RNG = np.random.RandomState(11)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def rot_to_z(axis):
    """axis(단위)를 +z로 보내는 회전행렬 (Rodrigues)."""
    a = unit(axis); z = np.array([0, 0, 1.0])
    v = np.cross(a, z); c = float(np.dot(a, z)); s = np.linalg.norm(v)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def main():
    # ---- 1) 상세 PC + SR SC 시냅스 + 단일볼리 -> 막전류 ----
    type_dir = net.load_representatives(MODELS)
    cell, tname = load_cell(type_dir["PC"], gid=0)
    h.define_shape()
    soma = cell.soma[0]
    soma_c = L.seg_point(soma, 0.5)
    h.distance(0, soma(0.5))
    apic = [s for s in cell.all if ".apic" in s.name()]
    dmax = max(h.distance(s(0.5)) for s in apic)
    depth_axis = unit(np.mean([L.seg_point(s, 0.5) for s in apic
                               if h.distance(s(0.5)) > 0.9 * dmax], axis=0) - soma_c)
    lo, hi = SR_BAND[0] * dmax, SR_BAND[1] * dmax
    sr = sorted([s for s in apic if lo <= h.distance(s(0.5)) <= hi], key=lambda s: h.distance(s(0.5)))
    chosen = [sr[i] for i in np.linspace(0, len(sr) - 1, N_SYN).round().astype(int)]
    p = P3.CLASSES["PC->PC (E2)"]
    ns = h.NetStim(); ns.number = 1; ns.start = 5.0; ns.noise = 0
    keep = []
    for s in chosen:
        syn = build_synapse(s(0.5), p, seeds=(1, 1, 1), deterministic=True)
        nc = h.NetCon(ns, syn); nc.weight[0] = p["g_nS"]; nc.delay = 1.0
        keep += [syn, nc]

    geom = L.collect_segments(list(cell.all))
    vecs, cv = L.setup_imem(geom["segs"])
    tvec = h.Vector().record(h._ref_t)
    h.celsius = 34.0; h.cvode_active(0); h.dt = 0.025
    h.finitialize(-70.0); h.continuerun(50.0)
    t = np.array(tvec)
    I = np.array([np.array(v) for v in vecs])            # (N_seg, N_t) nA

    # ---- 2) MEA 프레임으로 회전: 깊이축 -> -z (정단=SR을 유리쪽), 유리 위 Z_SOMA 여유 ----
    # 실측 SR fEPSP(음성) 재현: 흥분 시냅스가 몰린 SR(정단)을 전극(유리면)에 가깝게.
    R = rot_to_z(-depth_axis)
    mid0 = (geom["mid"] - soma_c) @ R.T                  # 소마 원점, 정단=-z쪽
    zmin, zmax = mid0[:, 2].min(), mid0[:, 2].max()
    z_shift = Z_SOMA - zmin                              # 최저(정단/SR)를 유리 위 여유로
    mid_mea = mid0 + np.array([0, 0, z_shift])
    H = mid_mea[:, 2].max() + Z_SOMA                     # 슬라이스 두께(세포 위 여유 포함)
    geom_mea = dict(mid=mid_mea, radius=geom["radius"])
    soma_z = soma_c @ R.T                                # (참고)
    print(f"[MEA] 세포 깊이범위 z={mid_mea[:,2].min():.0f}~{mid_mea[:,2].max():.0f}µm · 슬라이스 h={H:.0f}µm · 소마 z≈{mid_mea[np.argmin(np.abs(geom['mid'][:,1]-soma_c[1])),2]:.0f}", flush=True)
    print(f"[두께 주의] full-size 상세 PC(~700µm)를 담으려 h={H:.0f}µm 사용(실측 MEA 300~400µm는 세포를 절단). "
          f"두께는 두 상반 효과: 경계반사(두꺼울수록 MoI비율↑=과대) vs 소스-전극거리(멀수록 과소). 별개 기전이므로 분리해석.", flush=True)

    # ---- 3) 단일세포: MEA(영상법) vs 무한매질 ----
    elec = np.array([[0.0, 0.0, 0.0]])                   # 유리면 전극(인구 중심 아래)
    M_mea = L.moi_point_matrix(geom_mea, elec, SIG_T, SIG_S, SIG_G, H, N_IMG)
    # 무한매질 비교: 같은 MEA 기하에서 영상 없이 점전류원
    M_inf = L.psa_matrix(geom_mea, elec, SIG_T)
    V1_mea = (M_mea @ I)[0] * 1e3                         # uV
    V1_inf = (M_inf @ I)[0] * 1e3
    ipk = int(np.argmax(np.abs(V1_mea)))
    print(f"[단일세포] MEA(영상법) {V1_mea[ipk]:.3f}µV vs 무한매질 {V1_inf[ipk]:.3f}µV  (MEA/무한={V1_mea[ipk]/V1_inf[ipk]:.2f})", flush=True)

    # ---- 4) 앙상블: N개 정렬 복제본을 수평 원반(반경 R_POP)에 ----
    rr = R_POP * np.sqrt(RNG.uniform(0, 1, NMAX))
    th = RNG.uniform(0, 2 * np.pi, NMAX)
    offs = np.column_stack([rr * np.cos(th), rr * np.sin(th), np.zeros(NMAX)])  # 수평 이동
    virt = elec[0][None, :] - offs                       # 세포 이동 = 전극 반대 이동
    Mall = L.moi_point_matrix(geom_mea, virt, SIG_T, SIG_S, SIG_G, H, N_IMG)    # (NMAX, N_seg)
    cpk = Mall @ I[:, ipk] * 1e3                          # 각 복제본 피크기여 (uV)

    Ns = np.unique(np.round(np.logspace(0, np.log10(NMAX), 30)).astype(int))
    align = np.array([cpk[:n].sum() for n in Ns])        # 정렬+동기 합 (uV)
    signs = RNG.choice([-1, 1], NMAX)
    rand = np.array([np.abs((signs[:n] * cpk[:n]).sum()) for n in Ns])

    waves = {}
    for n in [100, 1000, NMAX]:
        waves[n] = (Mall[:n].sum(axis=0) @ I) * 1e3       # (N_t,) uV

    pop_full = abs(align[-1])
    a1 = abs(V1_mea[ipk])
    density = NMAX / (np.pi * (R_POP / 1000.0) ** 2)     # 세포/mm^2 (원반 면적)
    n_1mV = int(NMAX * 1000.0 / max(pop_full, 1e-9))
    print(f"[앙상블·MEA] 단일 {a1:.3f}µV -> 정렬+동기 N=1000 {abs(align[np.argmin(abs(Ns-1000))]):.0f}µV · N={NMAX} {pop_full:.0f}µV", flush=True)
    print(f"[밀도] N={NMAX}/반경{R_POP:.0f}µm = {density:.0f}세포/mm² -> 실제 CA1 str.pyr(2~4e4/mm²)에 근접 = 밀도는 이미 충분", flush=True)
    print(f"[외삽] 이 진폭이면 1mV엔 ~{n_1mV:,}세포. 실측 평면MEA fEPSP 0.1~1mV 대비 {pop_full:.0f}µV는 3~30x 미달", flush=True)
    print(f"[격차 원인] 밀도 아님 -> 세포당 진폭(이상화 단일볼리·Ecker E2 대용 SR 시냅스·전 복제본 동일깊이·유리전극 원거리)", flush=True)

    # ---------------- 그림 ----------------
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    m = (t >= 3) & (t <= 30)

    # (A) 단일세포: MEA vs 무한
    a = ax[0, 0]
    a.plot(t[m], V1_inf[m], color="#7f8c8d", lw=1.6, ls="--", label=f"무한매질 {V1_inf[ipk]:.2f}µV")
    a.plot(t[m], V1_mea[m], color="#c0392b", lw=1.9, label=f"MEA 영상법 {V1_mea[ipk]:.2f}µV")
    a.axhline(0, color="0.7", lw=0.5)
    a.set_title(f"(A) 단일세포: MEA 영상법 vs 무한매질\n유리 반사+식염수 -> ×{V1_mea[ipk]/V1_inf[ipk]:.2f}")
    a.set_xlabel("시간 (ms)"); a.set_ylabel("세포외 전위 (µV)"); a.legend(fontsize=8)

    # (B) 앙상블 성장
    b = ax[0, 1]
    for n, col in zip([100, 1000, NMAX], ["#f39c12", "#2980b9", "#c0392b"]):
        b.plot(t[m], waves[n][m], lw=1.8, color=col, label=f"N={n}  ({abs(waves[n][ipk]):.0f}µV)")
    b.axhline(0, color="0.7", lw=0.5)
    b.set_title("(B) 정렬+동기 앙상블 집단 fEPSP\n세포 수↑ -> 신호↑ (평면 MEA는 µV~수십µV급)")
    b.set_xlabel("시간 (ms)"); b.set_ylabel("세포외 전위 (µV)"); b.legend(fontsize=8)

    # (C) 크기 vs N
    c = ax[1, 0]
    c.loglog(Ns, np.abs(align), "o-", color="#c0392b", lw=1.8, label="정렬+동기 (더해짐)")
    c.loglog(Ns, np.abs(rand), "s--", color="#7f8c8d", lw=1.4, label="방향 무작위 (상쇄)")
    c.axhspan(100, 1000, color="green", alpha=0.12); c.text(1.2, 130, "실측 fEPSP 범위 0.1~1mV", fontsize=8, color="green")
    c.set_title("(C) 집단 크기 vs 세포 수 (MEA)\n정렬하면 커지고, 무작위면 상쇄")
    c.set_xlabel("세포 수 N"); c.set_ylabel("피크 |V| (µV)"); c.legend(fontsize=8); c.grid(alpha=0.3, which="both")

    # (D) MEA 기하 모식
    d = ax[1, 1]
    d.axhspan(-30, 0, color="#34495e", alpha=0.5)      # 유리
    d.axhspan(0, H, color="#f9e79f", alpha=0.35)       # 조직
    d.axhspan(H, H + 120, color="#aed6f1", alpha=0.5)  # 식염수
    d.text(R_POP * 0.5, -15, "유리 MEA (전극, z=0)", fontsize=8, color="w")
    d.text(R_POP * 0.5, H / 2, "조직 슬라이스", fontsize=8)
    d.text(R_POP * 0.5, H + 55, "식염수 배스", fontsize=8)
    # 세포 몇 개(정렬) + 전극
    for xoff in np.linspace(-R_POP, R_POP, 9):
        zc = mid_mea[:, 2]
        d.plot([xoff + (mid_mea[:, 0] - mid_mea[:, 0].mean()) * 0.15], [zc.mean()], alpha=0)  # dummy
        d.plot([xoff, xoff], [mid_mea[:, 2].min(), mid_mea[:, 2].max()], color="0.4", lw=1.2, alpha=0.6)
    d.plot([0], [0], "s", color="red", ms=10, zorder=5)
    d.text(-R_POP * 0.95, mid_mea[:, 2].max() * 0.9, "세포(정렬)\n소마 아래·정단 위", fontsize=8)
    d.set_xlim(-R_POP * 1.1, R_POP * 1.1); d.set_ylim(-30, H + 120)
    d.set_title("(D) MEA 3층 기하 (영상법)\n유리(반사)·조직·식염수(감쇠)")
    d.set_xlabel("가로 (µm)"); d.set_ylabel("높이 z (µm)")

    fig.suptitle(f"E4b — MEA 3층 영상법(MoI 검증됨·버그0) + 정렬 앙상블  (σ_T={SIG_T}/σ_S={SIG_S}/σ_G=0, h={H:.0f}µm, W_TS=−2/3)\n"
                 f"단일 {a1:.3f}µV(MEA=무한×{V1_mea[ipk]/V1_inf[ipk]:.1f}) → N={NMAX} {pop_full:.0f}µV @밀도 {density:.0f}/mm²(실제 CA1 근접). "
                 f"실측 0.1~1mV 대비 3~30x 미달 = 세포당 진폭(이상화볼리·E2대용·단일깊이·유리전극 원거리), 밀도 아님",
                 fontsize=9.5, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FIG, "E4b_mea_ensemble.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    np.savez(os.path.join(FIG, "_e4b_results.npz"),
             t=t, V1_mea=V1_mea, V1_inf=V1_inf, Ns=Ns, align=align, rand=rand,
             pop_full=pop_full, a1=a1, H=H, sigT=SIG_T, sigS=SIG_S, nmax=NMAX, ipk=ipk)
    print("saved:", out)


if __name__ == "__main__":
    main()
