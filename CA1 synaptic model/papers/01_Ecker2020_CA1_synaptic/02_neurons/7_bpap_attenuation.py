"""
7_bpap_attenuation.py — 뉴런 검증: 역전파 활동전위(BPAP) 감쇠 (PC)
============================================================================
Source: full-scale CA1 논문 S9; Golding/Spruston·Magee 계열 BPAP 실험.

체세포에서 활동전위(AP)를 1발 유발 → 첨단수상돌기(apical) **거리별로 역전파된 AP**를
동시에 기록한다. 멀수록 진폭이 줄어드는 감쇠를 지수 적합해 공간상수 λ를 구하고
논문 S9(모델 155.6µm, 실험 235.2µm)와 비교한다.

패널(사용자 스펙):
  (형태) PC 모델에 **근위(≈78µm)·원위(≈505µm)** 기록 지점을 정확히 표시(소마=AP 유발)
  (A) **역전파 AP 파형 — 소마 제외**(수상돌기 근/원위만; 멀수록 작고 느림)
  (B) **BPAP 진폭 vs 거리** + 지수적합 λ + **BPAP 뜻 설명**

캐싱: 시뮬(1회)을 npz 로 저장 → 라벨·점·색 수정은 재시뮬 없이 즉시.
  <ca1sim py> 7_bpap_attenuation.py              # 캐시 없으면 시뮬, 있으면 그림만
  <ca1sim py> 7_bpap_attenuation.py --recompute  # 강제 재시뮬
"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.optimize import curve_fit

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(THIS)))
SHARED = os.path.join(ROOT, "shared")
sys.path.insert(0, SHARED)
sys.path.insert(0, THIS)
from common.plotstyle import set_korean_font           # noqa: E402
import experimental_refs as REFS                        # noqa: E402

set_korean_font()
OUT = os.path.join(THIS, "figures")
CACHE = os.path.join(OUT, "7_bpap_cache.npz")
RECOMPUTE = "--recompute" in sys.argv

T_STIM, DUR, AMP, TSTOP, WIN = 50.0, 2.0, 2.5, 120.0, 12.0
NEAR_UM, FAR_UM = 78.0, 505.0          # 표시할 근위·원위 목표 거리(µm)


# ───────────────── 형태(SWC) — NEURON 불필요 ─────────────────
def parse_swc(path):
    pts, by_type = {}, {1: [], 2: [], 3: [], 4: []}
    soma_pts = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            c = ln.split()
            if len(c) < 7:
                continue
            i = int(c[0]); t = int(c[1]); x = float(c[2]); y = float(c[3]); par = int(c[6])
            pts[i] = (x, y, t)
            if t == 1:
                soma_pts.append((x, y))
            if par in pts:
                px, py, _ = pts[par]
                by_type.setdefault(t if t in by_type else 3, []).append([[px, py], [x, y]])
    sx = float(np.mean([p[0] for p in soma_pts])) if soma_pts else 0.0
    sy = float(np.mean([p[1] for p in soma_pts])) if soma_pts else 0.0
    return by_type, (sx, sy)


# ───────────────── NEURON 시뮬 (compute) ─────────────────
def seg_xy(h, sec):
    n = int(h.n3d(sec=sec))
    if n < 1:
        return (0.0, 0.0)
    i = n // 2
    return (float(h.x3d(i, sec=sec)), float(h.y3d(i, sec=sec)))


def compute():
    from common.nrn_env import h
    from common.cell_loader import load_cell
    h.load_file("stdrun.hoc")
    pyr = os.path.join(SHARED, "models", "pyramidal")
    pc_dir = os.path.join(pyr, sorted(os.listdir(pyr))[0])
    cell, tname = load_cell(pc_dir)
    soma = cell.soma[0]
    h.distance(0, soma(0.5))

    apics = sorted([(h.distance(s(0.5)), s) for s in cell.apic if h.distance(s(0.5)) > 5],
                   key=lambda x: x[0])
    # 곡선용 ~20점 균등 + 근/원위 목표 포함
    idxs = set(np.linspace(0, len(apics) - 1, 20).astype(int).tolist())
    for tgt in (NEAR_UM, FAR_UM):
        idxs.add(min(range(len(apics)), key=lambda k: abs(apics[k][0] - tgt)))
    chosen = [apics[i] for i in sorted(idxs)]

    ic = h.IClamp(soma(0.5)); ic.delay = T_STIM; ic.dur = DUR; ic.amp = AMP
    tvec = h.Vector().record(h._ref_t)
    vsoma = h.Vector().record(soma(0.5)._ref_v)
    recs = [(d, s, h.Vector().record(s(0.5)._ref_v)) for d, s in chosen]
    h.celsius = 34.0; h.cvode_active(1)
    h.finitialize(-70.0); h.continuerun(TSTOP)

    t = np.array(tvec)
    Vmat = np.array([np.array(v) for _, _, v in recs])
    XY = np.array([seg_xy(h, s) for _, s, _ in recs])
    dists = np.array([d for d, _, _ in recs])
    soma_xy = seg_xy(h, soma)

    i0, i1 = np.searchsorted(t, T_STIM - 1), np.searchsorted(t, T_STIM + WIN)
    b0, b1 = np.searchsorted(t, T_STIM - 2), np.searchsorted(t, T_STIM)

    def amp_of(v):
        return float(v[i0:i1].max() - v[b0:b1].mean())

    soma_amp = amp_of(np.array(vsoma))
    amps = np.array([amp_of(v) for v in Vmat])
    lam, A = np.nan, soma_amp
    try:
        popt, _ = curve_fit(lambda d, lam, A: A * np.exp(-d / lam), dists, amps,
                            p0=[150.0, soma_amp], bounds=([20, 1], [2000, 300]), maxfev=10000)
        lam, A = float(popt[0]), float(popt[1])
    except Exception as e:
        print(f"[경고] 지수 적합 실패: {e}", flush=True)

    print(f"[BPAP] 소마 AP {soma_amp:.0f}mV → 원위({dists.max():.0f}µm) {amps[-1]:.0f}mV "
          f"· λ={lam:.0f}µm", flush=True)
    return dict(t=t, Vmat=Vmat, XY=XY, dists=dists, vsoma=np.array(vsoma),
                soma_xy=np.array(soma_xy), soma_amp=soma_amp, amps=amps,
                lam=lam, A=A, tname=tname)


def save_cache(d):
    np.savez(CACHE, **d)


def load_cache():
    if not os.path.isfile(CACHE):
        return None
    z = np.load(CACHE, allow_pickle=True)
    return {k: (z[k] if z[k].ndim else z[k].item()) for k in z.files}


# ───────────────── 4 e-type 비교 아님 — 단일 PC 3패널 ─────────────────
def draw(d, swc, soma_xy_swc):
    t, Vmat, dists, XY = d["t"], d["Vmat"], d["dists"], d["XY"]
    near = int(np.argmin(np.abs(dists - NEAR_UM)))
    far = int(np.argmin(np.abs(dists - FAR_UM)))
    ref = REFS.ATTENUATION["bpap_lambda_um"]

    fig = plt.figure(figsize=(16.5, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.85, 1.2, 1.3], wspace=0.28)
    fig.suptitle(f"뉴런 검증 — 역전파 활동전위(BPAP) 감쇠 (PC: {d['tname']})",
                 fontsize=13, fontweight="bold")

    # ── (형태) 근위/원위 지점 표시 ──
    a0 = fig.add_subplot(gs[0, 0])
    style = {1: ("black", 1.8, 5, 1.0), 2: ("0.8", 0.3, 0, 0.35),
             3: ("tab:blue", 0.6, 2, 0.9), 4: ("0.45", 0.6, 2, 0.9)}
    for tt in (2, 3, 4, 1):
        segs = swc.get(tt)
        if segs:
            col, lw, z, al = style[tt]
            a0.add_collection(LineCollection(segs, colors=col, linewidths=lw, zorder=z, alpha=al))
    sx, sy = soma_xy_swc
    a0.plot(sx, sy, "*", color="gold", ms=20, mec="k", mew=0.8, zorder=10)
    a0.annotate("소마: AP 유발", (sx, sy), textcoords="offset points", xytext=(10, -16),
                fontsize=8.5, fontweight="bold")
    nx, ny = XY[near]; fx, fy = XY[far]
    a0.plot(nx, ny, "o", color="tab:green", ms=11, mec="k", mew=0.7, zorder=11)
    a0.plot(fx, fy, "o", color="tab:red", ms=11, mec="k", mew=0.7, zorder=11)
    a0.annotate(f"근위 {dists[near]:.0f}µm", (nx, ny), textcoords="offset points",
                xytext=(11, 5), fontsize=8.5, color="tab:green", fontweight="bold")
    a0.annotate(f"원위 {dists[far]:.0f}µm", (fx, fy), textcoords="offset points",
                xytext=(8, -2), fontsize=8.5, color="tab:red", fontweight="bold")
    a0.set_title("(형태) 기록 지점 — 근위·원위", fontsize=10)
    a0.set_aspect("equal"); a0.axis("off"); a0.autoscale()

    # ── (A) 역전파 AP 파형 — 소마 제외 ──
    aA = fig.add_subplot(gs[0, 1])
    aA.plot(t, Vmat[near], color="tab:green", lw=1.8, label=f"근위 {dists[near]:.0f}µm")
    aA.plot(t, Vmat[far], color="tab:red", lw=1.8, label=f"원위 {dists[far]:.0f}µm")
    aA.set_xlim(T_STIM - 3, T_STIM + 22)
    aA.set_xlabel("시간 t (ms)"); aA.set_ylabel("수상돌기 막전위 (mV)")
    aA.set_title("(A) 역전파 AP 파형 (소마 제외) — 멀수록 작고 느림", fontsize=10)
    aA.legend(fontsize=8.5); aA.grid(alpha=0.3)

    # ── (B) 진폭 vs 거리 + λ + BPAP 설명 ──
    aB = fig.add_subplot(gs[0, 2])
    aB.plot(dists, d["amps"], "o", color="tab:purple", ms=5, label="모델 BPAP 진폭")
    aB.plot(dists[near], d["amps"][near], "o", color="tab:green", ms=11, mec="k", zorder=5)
    aB.plot(dists[far], d["amps"][far], "o", color="tab:red", ms=11, mec="k", zorder=5)
    if np.isfinite(d["lam"]):
        dd = np.linspace(0, dists.max(), 100)
        aB.plot(dd, d["A"] * np.exp(-dd / d["lam"]), "-", color="k", lw=1.8,
                label=f"지수적합 λ={d['lam']:.0f}µm")
    aB.set_xlabel("소마로부터 거리 (µm)"); aB.set_ylabel("역전파 AP 진폭 (mV)")
    aB.set_title("(B) BPAP 진폭 vs 거리", fontsize=10)
    aB.legend(fontsize=8.5, loc="upper right"); aB.grid(alpha=0.3)
    aB.text(0.03, 0.97,
            "■ BPAP = Back-Propagating AP (역전파 활동전위)\n"
            "  소마/AIS에서 점화된 AP가 수상돌기로 역방향 전파.\n"
            "  수상돌기 Na채널+케이블 → 거리↑ 일수록 진폭↓(감쇠).\n"
            "  STDP 등에서 '후세포 발화'를 수상돌기에 알리는 신호.\n"
            f"  λ(우리)={d['lam']:.0f} · 논문 모델 {ref['model_paper']:.0f} · "
            f"실험 {ref['exp_paper']:.0f} µm",
            transform=aB.transAxes, va="top", fontsize=7.8,
            bbox=dict(fc="#FFF6D5", ec="0.6", alpha=0.95))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(OUT, "7_bpap_attenuation.png")
    fig.savefig(out, dpi=125)
    print(f"[그림] {out}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    d = None if RECOMPUTE else load_cache()
    if d is None:
        print("[시뮬] 캐시 없음/재계산 → NEURON 실행 …", flush=True)
        d = compute()
        save_cache(d)
        print(f"[캐시저장] {CACHE}", flush=True)
    else:
        print(f"[캐시] {CACHE} 사용 → 재시뮬 없이 그림만", flush=True)

    pyr = os.path.join(SHARED, "models", "pyramidal")
    pc_dir = os.path.join(pyr, sorted(os.listdir(pyr))[0])
    swc, soma_xy_swc = parse_swc(os.path.join(pc_dir, "morphology", "morphology.swc"))
    draw(d, swc, soma_xy_swc)


if __name__ == "__main__":
    main()
