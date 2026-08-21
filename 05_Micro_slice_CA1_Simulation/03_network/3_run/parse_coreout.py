# -*- coding: utf-8 -*-
"""파일모드 GPU 결과(special-core out.dat)를 baseline npz로 변환.
사용: python parse_coreout.py <coreout_dir/out.dat> <coredat_dir/meta.json>
결과: scratch/mpi_baseline{tag}.npz + .json (viz_baseline_full.py가 읽는 포맷)
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DERIVED = os.path.join(ROOT, "data", "derived")

outdat = sys.argv[1]
metaj = sys.argv[2]
meta = json.load(open(metaj, encoding="utf-8"))
N = int(meta["n"]); STIM_T = float(meta["stim_t"]); SETTLE = float(meta["settle"])
TSTOP = float(meta["tstop"]); RADIUS = float(meta["radius"]); NOSTIM = int(meta.get("nostim", 0))
TAG = meta.get("tag", "")

# out.dat: 각 줄 "time gid" (탭/공백). gid < N = 실제 뉴런, 그 이상 = 섬유 VecStim
raw = np.loadtxt(outdat) if os.path.getsize(outdat) > 0 else np.empty((0, 2))
if raw.ndim == 1:
    raw = raw.reshape(1, -1)
st_all = raw[:, 0].astype(float); id_all = raw[:, 1].astype(int)
cellmask = id_all < N
st = st_all[cellmask]; sid = id_all[cellmask]

wc = np.load(os.path.join(DERIVED, "window_cells.npz"), allow_pickle=True)
mt = wc["mtype"].astype(str)
is_pc = np.array([mt[g] == "SP_PC" for g in range(N)])
fired = set(sid.tolist())
firedmask = np.array([g in fired for g in range(N)])
nE = int(np.sum(firedmask & is_pc)); nI = int(np.sum(firedmask & ~is_pc))
totE = int(is_pc.sum()); totI = N - totE
obs_ms = TSTOP - SETTLE
rate = 1000.0 * len(st) / max(obs_ms, 1) / N

np.savez_compressed(os.path.join(ROOT, "scratch", f"mpi_baseline{TAG}.npz"),
                    spk_t=st, spk_id=sid, fired=firedmask, is_pc=is_pc,
                    radius=RADIUS, stim_t=STIM_T, settle=SETTLE, n=N, nostim=NOSTIM)
json.dump({"n": N, "sc": meta.get("sc"), "internal": meta.get("internal"),
           "spikes": int(len(st)), "active": int(firedmask.sum()), "activeE": nE, "activeI": nI,
           "radius": RADIUS, "nostim": NOSTIM, "obs_ms": obs_ms, "rate_hz": rate, "mode": "filemode_gpu"},
          open(os.path.join(ROOT, "scratch", f"mpi_baseline{TAG}.json"), "w"))

note = "무자극 자발" if NOSTIM else "자극 volley 1회"
print(f"[변환] out.dat -> scratch/mpi_baseline{TAG}.npz")
print(f"  총 뉴런 스파이크 {len(st):,} · 발화세포 {int(firedmask.sum())}/{N} ({100*firedmask.mean():.0f}%) ({note})")
print(f"  추체 {nE}/{totE} ({100*nE/max(totE,1):.0f}%) · 억제 {nI}/{totI} ({100*nI/max(totI,1):.0f}%) · 망평균 {rate:.3f} Hz/세포")
