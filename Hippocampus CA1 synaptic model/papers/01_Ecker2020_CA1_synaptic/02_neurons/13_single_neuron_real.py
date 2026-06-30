"""
13_single_neuron_real.py — 단일세포 검증 4패널 (실제 CA1 e-model)
============================================================================
실제 Hippocampus Hub me-model(NEURON)로 4패널 그림. HH 데모(single_neuron_validation.py)
는 교육용으로 보존.

4패널(사용자 스펙):
  (1) 형태 + 자극/측정/개시 지점   : 소마=전류주입(IClamp)+Vm측정, AIS=AP 개시 추정
  (2) 주입 전류 + 막전위 스파이크   : 300~800ms step 자극에 유발된 발화(자발 아님)
  (3) f-I 곡선                      : 자극 세기 vs 발화 빈도
  (4) 스파이크(활동전위) 1개 확대   : 역치·진폭·반치폭·AHP 를 점선으로 표시

전기생리:
  - 자극·측정 모두 **소마**(IClamp @ soma, Vm @ soma) = 실험 whole-cell patch 와 동일.
  - AP 는 **AIS(축삭 stub)** 에서 점화→소마로 역전파되어 측정. **자발 발화 아님**(전류 유발).
  - EMS 시냅스 없음 → 고정 dt(=0.025) 안전.

캐싱(빠른 반복):
  시뮬(f-I·demo 트레이스)을 npz 로 1회 저장 → 라벨·점선·색 등 **그림 수정은 재시뮬 없이 즉시**.
  실행:
    <ca1sim py> 13_single_neuron_real.py               # 캐시 없으면 시뮬, 있으면 캐시로 그림
    <ca1sim py> 13_single_neuron_real.py --recompute   # 강제 재시뮬(전류·세포·dt 바꿀 때)
    <ca1sim py> 13_single_neuron_real.py --mtype SP-PVBC
"""
import os
import sys
import json

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

THIS = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(THIS)
ROOT = os.path.dirname(os.path.dirname(PAPER))
SHARED = os.path.join(ROOT, "shared")
sys.path.insert(0, SHARED)
from common.plotstyle import set_korean_font           # noqa: E402

set_korean_font()
OUT = os.path.join(THIS, "figures")
REGISTRY = os.path.join(SHARED, "models", "models_registry.json")

DT = 0.025
CELSIUS = 34.0
V_INIT = -65.0
T_ON, DUR, TSTOP = 300.0, 500.0, 900.0     # 자극 300→800ms, 사전 300ms 안정화
CUR = "#E8820C"                            # 자극 전류 색(주황)
RECOMPUTE = "--recompute" in sys.argv


# ───────────────────────── 모델 선택 ─────────────────────────
def pick_model():
    with open(REGISTRY, encoding="utf-8") as f:
        models = json.load(f)["models"]
    argv = sys.argv
    mt = argv[argv.index("--mtype") + 1] if "--mtype" in argv else None
    et = argv[argv.index("--etype") + 1] if "--etype" in argv else None
    cand = [m for m in models
            if (mt is None or m["mtype"] == mt) and (et is None or m["etype"] == et)]
    if not cand:
        cand = [m for m in models if m["role"] == "PC"] or models
    return cand[0]


# ───────────────────────── 형태(SWC) — NEURON 불필요 ─────────────────────────
def parse_swc(path):
    """morphology.swc → (by_type 선분, soma_xy, axon_pts)."""
    pts = {}
    by_type = {1: [], 2: [], 3: [], 4: []}
    soma_pts, axon_pts = [], []
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
            elif t == 2:
                axon_pts.append((x, y))
            if par in pts:
                px, py, _ = pts[par]
                by_type.setdefault(t if t in by_type else 3, []).append([[px, py], [x, y]])
    soma_xy = (float(np.mean([p[0] for p in soma_pts])),
               float(np.mean([p[1] for p in soma_pts]))) if soma_pts else (0.0, 0.0)
    return by_type, soma_xy, axon_pts


def model_channels(model):
    """모델 mechanisms 폴더의 .mod = 사용 이온채널 목록(어떤 스파이크 모델인지 표기용)."""
    md = os.path.join(ROOT, model["dir"], "mechanisms")
    if not os.path.isdir(md):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(md) if f.endswith(".mod"))


def ais_xy(soma_xy, axon_pts):
    """소마에서 가장 가까운 축삭 점 = AIS(개시) 추정 위치."""
    if not axon_pts:
        return None
    sx, sy = soma_xy
    d = [((ax - sx) ** 2 + (ay - sy) ** 2, (ax, ay)) for ax, ay in axon_pts]
    d.sort(key=lambda z: z[0])
    # 소마에 너무 붙은 첫 점보다 약간 떨어진(개시분절) 점 선택
    idx = min(len(d) - 1, 8)
    return d[idx][1]


# ───────────────────────── NEURON 시뮬 (compute) ─────────────────────────
def compute(model):
    """f-I 스윕 + demo 트레이스 계산 → dict. (NEURON 필요)"""
    from common.nrn_env import h
    from common.cell_loader import load_cell
    h.load_file("stdrun.hoc")
    cell, tname = load_cell(os.path.join(ROOT, model["dir"]))
    ic = h.IClamp(cell.soma[0](0.5))
    tv = h.Vector().record(h._ref_t)
    vv = h.Vector().record(cell.soma[0](0.5)._ref_v)
    h.dt = DT; h.celsius = CELSIUS

    def run_step(amp):
        ic.delay = T_ON; ic.dur = DUR; ic.amp = amp
        h.finitialize(V_INIT); h.continuerun(TSTOP)
        return np.array(tv), np.array(vv)

    def nspk(t, v):
        w = (t[:-1] >= T_ON) & (t[:-1] <= T_ON + DUR)
        return int(np.sum((v[:-1] < -10) & (v[1:] >= -10) & w))

    amps = np.round(np.arange(0.0, 1.05, 0.1), 3)
    print("[f-I] 스윕 …", flush=True)
    rates = np.array([nspk(*run_step(a)) / (DUR / 1000.0) for a in amps])
    rheo = float(amps[np.argmax(rates > 0)]) if np.any(rates > 0) else float("nan")
    I_demo = float(amps[int(np.argmax(rates))]) if rates.max() > 0 else float(amps[-1])
    t, v = run_step(I_demo)
    n_sp = nspk(t, v)
    print(f"[demo] I={I_demo}nA, 스파이크 {n_sp}개", flush=True)
    return dict(amps=amps, rates=rates, rheo=rheo, I_demo=I_demo, n_sp=n_sp,
                t=t, v=v, tname=tname)


def cache_path(model):
    return os.path.join(OUT, f"13_cache_{model['mtype']}_{model['etype']}.npz")


def save_cache(model, d):
    np.savez(cache_path(model), amps=d["amps"], rates=d["rates"], t=d["t"], v=d["v"],
             rheo=d["rheo"], I_demo=d["I_demo"], n_sp=d["n_sp"], tname=d["tname"])


def load_cache(model):
    p = cache_path(model)
    if not os.path.isfile(p):
        return None
    z = np.load(p, allow_pickle=True)
    return dict(amps=z["amps"], rates=z["rates"], t=z["t"], v=z["v"],
                rheo=float(z["rheo"]), I_demo=float(z["I_demo"]),
                n_sp=int(z["n_sp"]), tname=str(z["tname"]))


# ───────────────────────── 특징 추출 (그림 단계 — 빠름) ─────────────────────────
def ap_features(t, v, dt):
    """첫 스파이크 특징 + 점선 표시용 좌표."""
    peaks = np.where((v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:]) & (v[1:-1] > -10.0))[0] + 1
    if len(peaks) == 0:
        return None
    pk = int(peaks[0])
    dvdt = np.gradient(v, dt)
    w0 = max(0, pk - int(5.0 / dt))                    # 정점 직전 5ms 창(자극 onset 오인 방지)
    pre = np.where(dvdt[w0:pk] >= 15.0)[0]
    thr_idx = int(w0 + pre[0]) if len(pre) else max(pk - 1, 0)
    v_thr, v_peak = float(v[thr_idx]), float(v[pk])
    amp = v_peak - v_thr
    half = v_thr + amp / 2.0
    left = pk
    while left > 0 and v[left] > half:
        left -= 1
    right = pk
    while right < len(v) - 1 and v[right] > half:
        right += 1
    hw = (right - left) * dt
    a1 = min(pk + int(8 / dt), len(v))                 # fast AHP (정점 후 ~8ms)
    ahp_idx = pk + int(np.argmin(v[pk:a1]))
    v_ahp = float(v[ahp_idx])
    return dict(pk=pk, thr_idx=thr_idx, v_thr=v_thr, v_peak=v_peak, amp=amp,
                half=half, left=left, right=right, hw=hw, ahp_idx=ahp_idx,
                v_ahp=v_ahp, ahp=v_ahp - v_thr)


# ───────────────────────── 4-패널 그림 ─────────────────────────
def draw(model, d, swc, soma_xy, axon_pts):
    t, v = d["t"], d["v"]
    feat = ap_features(t, v, DT)
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"단일세포 검증 4패널 (실제 CA1 e-model)   ·   {model['mtype']} · {model['etype']} · "
                 f"{model['layer']} · {d['tname']}", fontsize=14, fontweight="bold")

    # ── (1) 형태 + 자극/측정/개시 ──
    a1 = ax[0, 0]
    style = {1: ("black", 1.8, 5, 1.0), 2: ("0.78", 0.35, 0, 0.4),
             3: ("tab:blue", 0.7, 2, 0.95), 4: ("tab:red", 0.7, 2, 0.95)}
    for tt in (2, 3, 4, 1):
        segs = swc.get(tt)
        if segs:
            col, lw, z, al = style[tt]
            a1.add_collection(LineCollection(segs, colors=col, linewidths=lw, zorder=z, alpha=al))
    sx, sy = soma_xy
    a1.plot(sx, sy, "*", color="gold", ms=22, mec="k", mew=0.8, zorder=10)
    a1.plot(sx, sy, "v", color="navy", ms=10, mec="w", mew=0.6, zorder=11)
    a1.annotate("소마: 전류 주입(★) + Vm 측정(▼)", (sx, sy), textcoords="offset points",
                xytext=(12, 10), fontsize=9, fontweight="bold",
                bbox=dict(fc="white", ec="0.6", alpha=0.85))
    ais = ais_xy(soma_xy, axon_pts)
    if ais:
        a1.plot(*ais, "s", color="crimson", ms=9, mec="k", mew=0.6, zorder=10)
        a1.annotate("AP 개시 (AIS)", ais, textcoords="offset points", xytext=(8, -12),
                    fontsize=8.5, color="crimson")
    a1.set_title("(1) 형태 · 자극/측정=소마 · 개시=AIS", fontsize=11)
    a1.set_aspect("equal"); a1.axis("off"); a1.autoscale()

    # ── (2) 주입 전류 + Vm 스파이크 ──
    a2 = ax[0, 1]
    Itr = np.where((t >= T_ON) & (t <= T_ON + DUR), d["I_demo"], 0.0)
    a2.plot(t, v, color="navy", lw=0.8)
    a2.set_title(f"(2) 자극 유발 발화  ·  스파이크 {d['n_sp']}개  (I={d['I_demo']:.2f} nA, 300~800ms)",
                 fontsize=11)
    a2.set_xlabel("시간 (ms)"); a2.set_ylabel("막전위 Vm (mV)", color="navy")
    a2.tick_params(axis="y", labelcolor="navy"); a2.grid(alpha=0.3)
    axI = a2.twinx()
    axI.plot(t, Itr, color=CUR, lw=1.3); axI.fill_between(t, 0, Itr, color=CUR, alpha=0.12)
    axI.set_ylabel("주입 전류 I (nA)", color=CUR); axI.tick_params(axis="y", labelcolor=CUR)
    axI.set_ylim(-0.03, max(d["I_demo"] * 2.2, 0.1))

    # ── (3) f-I 곡선 ──
    a3 = ax[1, 0]
    a3.plot(d["amps"], d["rates"], "o-", color="darkgreen", ms=5)
    a3.set_title("(3) f-I 곡선 — 자극↑ → 발화↑", fontsize=11)
    a3.set_xlabel("주입 전류 (nA)"); a3.set_ylabel("발화 빈도 (Hz)"); a3.grid(alpha=0.3)
    if np.any(d["rates"] > 0):
        ri = int(np.argmax(d["rates"] > 0))
        a3.annotate(f"rheobase ~ {d['rheo']:.2f} nA", xy=(d["amps"][ri], d["rates"][ri]),
                    xytext=(d["amps"][ri] + 0.12, max(d["rates"]) * 0.5),
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.2), color="red", fontsize=9)

    # ── (4) 스파이크 1개 확대 + 점선 특징 ──
    a4 = ax[1, 1]
    if feat:
        pk = feat["pk"]
        peaks_all = np.where((v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:]) & (v[1:-1] > -10.0))[0] + 1
        nxt = next((p for p in peaks_all if p > pk), None)
        i0 = int(np.searchsorted(t, 295.0))               # 295ms 부터
        i1 = min(pk + int(12 / DT), len(t))
        if nxt is not None:
            i1 = min(i1, int(nxt - 1 / DT))               # 다음 스파이크 직전까지(1개만)
        i1 = max(i1, pk + int(9 / DT))
        sl = slice(i0, i1)
        tt, vv = t[sl], v[sl]
        # 주입 전류 함께(twin, 주황) — Vm·주석은 위 레이어로
        Itr4 = np.where((tt >= T_ON) & (tt <= T_ON + DUR), d["I_demo"], 0.0)
        ax4I = a4.twinx()
        ax4I.plot(tt, Itr4, color=CUR, lw=1.3); ax4I.fill_between(tt, 0, Itr4, color=CUR, alpha=0.12)
        ax4I.set_ylabel("주입 전류 I (nA)", color=CUR); ax4I.tick_params(axis="y", labelcolor=CUR)
        ax4I.set_ylim(-0.03, max(d["I_demo"] * 3.0, 0.1))
        a4.set_zorder(ax4I.get_zorder() + 1); a4.patch.set_visible(False)
        a4.plot(tt, vv, color="navy", lw=1.9, zorder=5)   # 패널(2) 스파이크와 동일 색
        x0, x1 = tt[0], tt[-1]
        tpk, tthr = t[pk], t[feat["thr_idx"]]
        tL, tR = t[feat["left"]], t[feat["right"]]
        tahp = t[feat["ahp_idx"]]
        vthr, vpk, vahp, half = feat["v_thr"], feat["v_peak"], feat["v_ahp"], feat["half"]

        # 역치/정점 수평 점선
        a4.hlines(vthr, x0, x1, color="0.5", ls=":", lw=1.2)
        a4.hlines(vpk, x0, x1, color="0.5", ls=":", lw=0.9)
        a4.plot(tthr, vthr, "k^", ms=8, zorder=6); a4.plot(tpk, vpk, "kv", ms=8, zorder=6)
        a4.text(x0, vthr, f" 역치 {vthr:.0f}mV", fontsize=8.5, va="bottom", color="0.3")

        # 진폭 (역치→정점 세로 화살표, 파랑)
        xa = tpk + (x1 - tpk) * 0.18
        a4.annotate("", xy=(xa, vpk), xytext=(xa, vthr),
                    arrowprops=dict(arrowstyle="<->", color="tab:blue", lw=1.6))
        a4.text(xa + (x1 - x0) * 0.01, (vthr + vpk) / 2, f"진폭\n{feat['amp']:.0f}mV",
                color="tab:blue", fontsize=9, va="center", fontweight="bold")

        # 반치폭 (절반 높이 가로 점선+화살표, 초록)
        a4.hlines(half, tL, tR, color="tab:green", ls=":", lw=1.4)
        a4.annotate("", xy=(tR, half), xytext=(tL, half),
                    arrowprops=dict(arrowstyle="<->", color="tab:green", lw=1.6))
        a4.text((tL + tR) / 2, half, f"반치폭 {feat['hw']:.2f}ms", color="tab:green",
                fontsize=9, ha="center", va="bottom", fontweight="bold")

        # AHP (정점 후 최저점, 역치 기준 세로 점선+화살표, 보라)
        a4.hlines(vahp, tpk, x1, color="purple", ls=":", lw=1.0)
        a4.annotate("", xy=(tahp, vahp), xytext=(tahp, vthr),
                    arrowprops=dict(arrowstyle="<->", color="purple", lw=1.5))
        a4.text(tahp, (vthr + vahp) / 2, f" AHP {feat['ahp']:.0f}mV", color="purple",
                fontsize=9, va="center", fontweight="bold")
    a4.set_title("(4) 스파이크(활동전위) 1개 확대 — 역치·진폭·반치폭·AHP  [CA1 e-model · HH 아님]",
                 fontsize=10.5)
    a4.set_xlabel("시간 (ms)"); a4.set_ylabel("막전위 Vm (mV)", color="navy")
    a4.tick_params(axis="y", labelcolor="navy"); a4.grid(alpha=0.3)

    chans = model_channels(model)
    fig.text(0.5, 0.030, "자극·측정 = 소마(whole-cell) · AP 개시 = AIS · 자발 아닌 전류 유발 발화",
             ha="center", fontsize=9, color="0.35")
    fig.text(0.5, 0.007, f"스파이크 생성 모델: {model['mtype']} {model['etype']} Hub CA1 e-model · "
             f"HH 형식 전압의존 다채널(고전 HH 아님): {', '.join(chans)}",
             ha="center", fontsize=7.8, color="0.4")
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    out = os.path.join(OUT, f"13_single_neuron_real_{model['mtype']}_{model['etype']}.png")
    fig.savefig(out, dpi=135); plt.close(fig)
    print(f"[그림] {out}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    m = pick_model()
    print(f"[모델] {m['mtype']} · {m['etype']} · {m['layer']} · {m['morph']}", flush=True)

    d = None if RECOMPUTE else load_cache(m)
    if d is None:
        print("[시뮬] 캐시 없음/재계산 → NEURON 실행 …", flush=True)
        d = compute(m)
        save_cache(m, d)
        print(f"[캐시저장] {cache_path(m)}", flush=True)
    else:
        print(f"[캐시] {cache_path(m)} 사용 → 재시뮬 없이 그림만", flush=True)

    swc, soma_xy, axon_pts = parse_swc(os.path.join(ROOT, m["dir"], "morphology", "morphology.swc"))
    draw(m, d, swc, soma_xy, axon_pts)


if __name__ == "__main__":
    main()
