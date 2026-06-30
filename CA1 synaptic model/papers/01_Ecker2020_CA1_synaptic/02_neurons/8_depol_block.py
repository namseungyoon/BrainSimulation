"""
8_depol_block.py — 뉴런 검증: 탈분극 블록(depolarization block) (PC)
============================================================================
Source: HippoUnit DepolarizationBlockTest; Bianchi et al. (2012) CA1 PC.

전류를 점점 키우면 발화수가 늘다가, **너무 강하면 막이 탈분극된 채로 멈춰**(depolarization
block) 발화가 사라진다. 전류 스윕으로 발화수 곡선의 상승→하강과 **블록 개시 전류**를 찾고,
실험 근사 범위와 비교한다. eFEL `depol_block_bool` 로 교차확인.

실행: <ca1sim python> papers/01_Ecker2020_CA1_synaptic/02_neurons/8_depol_block.py
"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(THIS)))
SHARED = os.path.join(ROOT, "shared")
sys.path.insert(0, SHARED)
sys.path.insert(0, THIS)
from common.nrn_env import h                   # noqa: E402
from common.cell_loader import load_cell       # noqa: E402
from common.plotstyle import set_korean_font   # noqa: E402
import experimental_refs as REFS               # noqa: E402

set_korean_font()
h.load_file("stdrun.hoc")
OUT = os.path.join(THIS, "figures")

AMPS = [round(a, 2) for a in np.arange(0.4, 3.01, 0.2)]
DELAY, DUR, TSTOP = 300.0, 600.0, 1000.0


def main():
    os.makedirs(OUT, exist_ok=True)
    import efel
    try:
        efel.set_setting("Threshold", -20.0)
    except Exception:
        pass
    pyr = os.path.join(SHARED, "models", "pyramidal")
    pc_dir = os.path.join(pyr, sorted(os.listdir(pyr))[0])
    cell, tname = load_cell(pc_dir)
    soma = cell.soma[0]
    ic = h.IClamp(soma(0.5)); ic.delay = DELAY; ic.dur = DUR
    tv = h.Vector().record(h._ref_t)
    vv = h.Vector().record(soma(0.5)._ref_v)
    h.celsius = 34.0
    # 고전류 탈분극 블록 스윕 → cvode 가 불안정 구간에서 멈추므로 고정 dt 사용(빠르고 안정)
    h.cvode_active(0)
    h.dt = 0.025

    counts, vends, blocks, traces = [], [], [], {}
    for amp in AMPS:
        ic.amp = amp
        h.finitialize(-70.0)
        h.continuerun(TSTOP)
        t = np.array(tv); v = np.array(vv)
        tr = {"T": t, "V": v, "stim_start": [DELAY], "stim_end": [DELAY + DUR]}
        sc = efel.get_feature_values([tr], ["Spikecount"], raise_warnings=False)[0].get("Spikecount")
        n = int(sc[0]) if sc is not None and len(sc) else 0
        db = efel.get_feature_values([tr], ["depol_block_bool"], raise_warnings=False)[0].get("depol_block_bool")
        blocked = bool(db[0]) if db is not None and len(db) else False
        vend = float(v[np.searchsorted(t, DELAY + DUR - 100):np.searchsorted(t, DELAY + DUR)].mean())
        counts.append(n); vends.append(vend); blocks.append(blocked)
        traces[amp] = (t, v)

    counts = np.array(counts); vends = np.array(vends)
    amps = np.array(AMPS)
    # 블록 개시: 발화수 정점 이후 처음으로 (발화수 < 0.4*정점) AND (Vend > -45mV)
    ipk = int(np.argmax(counts))
    I_block = None
    for k in range(ipk + 1, len(amps)):
        if counts[k] < 0.4 * counts[ipk] and vends[k] > -45:
            I_block = amps[k]; break
    if I_block is None:  # eFEL 플래그 백업
        for k, b in enumerate(blocks):
            if b:
                I_block = amps[k]; break

    rng = REFS.DEPOL_BLOCK["cACpyr"]["I_block_nA"]

    # ── 그림 ───────────────────────────────────────────────────────
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16.5, 4.9))
    fig.suptitle(f"뉴런 검증 — 탈분극 블록 (PC: {tname})", fontsize=13, fontweight="bold")

    # (A) 발화수 vs 전류 (상승→하강)
    axA.plot(amps, counts, "o-", color="tab:blue", lw=1.8, ms=5)
    axA.axvspan(rng[0], rng[1], color="tab:gray", alpha=0.15, label=f"실험 블록 근사 {rng[0]}–{rng[1]}nA")
    if I_block is not None:
        axA.axvline(I_block, color="tab:red", ls="--", lw=1.8, label=f"블록 개시 {I_block:.1f}nA")
    axA.set_title("(A) 발화수 vs 전류", fontsize=10)
    axA.set_xlabel("주입 전류 (nA)"); axA.set_ylabel("스파이크 수 (600ms)"); axA.legend(fontsize=8)

    # (B) 예시 트레이스: 발화 vs 블록
    amp_fire = amps[ipk]
    amp_blk = I_block if I_block is not None else amps[-1]
    tf, vf = traces[amp_fire]; tb, vb = traces[amp_blk]
    axB.plot(tf, vf, color="tab:green", lw=0.8, label=f"발화 {amp_fire:.1f}nA ({counts[ipk]}개)")
    axB.plot(tb, vb, color="tab:red", lw=0.8, label=f"블록 {amp_blk:.1f}nA")
    axB.set_xlim(DELAY - 20, DELAY + DUR + 20); axB.set_title("(B) 발화 vs 블록 트레이스", fontsize=10)
    axB.set_xlabel("시간 t (ms)"); axB.set_ylabel("막전위 (mV)"); axB.legend(fontsize=8)

    # (C) 정상상태 막전위 vs 전류(블록이면 탈분극 고정)
    axC.plot(amps, vends, "s-", color="tab:orange", lw=1.8, ms=5)
    axC.axhline(-45, color="0.5", ls=":", lw=1); axC.text(amps[0], -43, "탈분극 고정선(-45mV)", fontsize=7, color="0.4")
    axC.set_title("(C) 자극말 정상상태 전위", fontsize=10)
    axC.set_xlabel("주입 전류 (nA)"); axC.set_ylabel("자극말 V (mV)")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUT, "8_depol_block.png")
    fig.savefig(out, dpi=120)
    print(f"[그림] {out}")
    print(f"[검증] 발화수 정점 {counts[ipk]}개 @ {amps[ipk]:.1f}nA → "
          f"블록 개시 {'%.1f nA' % I_block if I_block is not None else '미검출(범위 내 무블록)'}")
    print(f"  (실험 근사 {rng[0]}–{rng[1]}nA; eFEL depol_block_bool 플래그 {sum(blocks)}개 계단)")


if __name__ == "__main__":
    main()
