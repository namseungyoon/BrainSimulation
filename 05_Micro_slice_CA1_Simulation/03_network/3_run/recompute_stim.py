# -*- coding: utf-8 -*-
"""SC 자극 기준점 이동 → dist_e3 재계산 (시냅스 위치·커넥텀 불변, 거리만).
문헌(SR·양극전극·경로자극)에 맞춰 자극 locus를 두께 중앙(w=0)·SR로 이동.
현재 w=-200(바닥면)은 옛 config 인공물. 옛 dist_e3·e3_xyz는 백업.
실행: python recompute_stim.py [--u -279 --r 253 --w 0]
"""
import os, sys, json, shutil
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DERIVED = os.path.join(ROOT, "data", "derived")


def arg(f, d):
    return type(d)(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else d

NU = arg("--u", -279.0); NR = arg("--r", 253.0); NW = arg("--w", 0.0)   # 국소 목표(장축·층·두께)


def main():
    cfg = json.load(open(os.path.join(ROOT, "config", "window_layout.json"), encoding="utf-8"))
    fr = cfg["frame_um"]; seed = np.array(fr["seed"])
    L = np.array(fr["long_dir"]); R = np.array(fr["radial_dir"]); T = np.array(fr["thick_dir"])
    new_pt = seed + NU * L + NR * R + NW * T                       # 새 자극점 global
    print(f"[stim] 새 기준점 local(u={NU},r={NR},w={NW}) → global {np.round(new_pt,1)}", flush=True)

    p = os.path.join(DERIVED, "sc_synapses.npz")
    bak = os.path.join(DERIVED, "sc_synapses_w-200_backup.npz")
    if not os.path.exists(bak):
        shutil.copy(p, bak); print(f"[stim] 백업 → {os.path.basename(bak)}", flush=True)
    z = np.load(p, allow_pickle=True); d = {k: z[k] for k in z.files}
    old_e3 = d["e3_xyz"].astype(float); old_dist = d["dist_e3"].astype(float)
    scxyz = d["xyz"].astype(float)
    new_dist = np.linalg.norm(scxyz - new_pt, axis=1).astype(np.float32)
    d["dist_e3"] = new_dist; d["e3_xyz"] = new_pt.astype(float)
    np.savez_compressed(p, **d)
    print(f"[stim] dist_e3 재계산 완료. 옛 기준 {np.round(old_e3,1)} → 새 {np.round(new_pt,1)}", flush=True)
    print(f"[stim] dist_e3<150 시냅스: 옛 {int(np.sum(old_dist<150)):,} → 새 {int(np.sum(new_dist<150)):,}", flush=True)
    print(f"[stim] 새 dist 중앙 {np.median(new_dist):.0f}µm · 최소 {new_dist.min():.0f}", flush=True)


if __name__ == "__main__":
    main()
