# -*- coding: utf-8 -*-
"""Ex12 리플레이 UI 빌더 — baseline npz → 데이터 임베드 → 자립형 ex12_replay.html.

사용: python build_ex12_replay.py [--npz scratch/mpi_baseline.npz]
- 템플릿 `ex12_replay_tpl.html`(같은 폴더)의 `__INJECT__` 자리에 시뮬 데이터 JSON을 삽입.
- 세포 3D 위치(슬라이스 국소 프레임 u=장축·r=방사·w=두께) + 스파이크(gid,자극기준 시각) + 메타(소요시간 등).
- 어떤 baseline npz(볼리/무자극/다른 자극)로도 재생성 가능.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DERIVED = os.path.join(ROOT, "data", "derived")

npz = sys.argv[sys.argv.index("--npz") + 1] if "--npz" in sys.argv else os.path.join(ROOT, "scratch", "mpi_baseline.npz")
d = np.load(npz, allow_pickle=True)
wc = np.load(os.path.join(DERIVED, "window_cells.npz"), allow_pickle=True)
cfg = json.load(open(os.path.join(ROOT, "config", "window_layout.json"), encoding="utf-8"))

XYZ = wc["xyz"].astype(float); mt = wc["mtype"].astype(str)
fr = cfg["frame_um"]; seed = np.array(fr["seed"], float)
L = np.array(fr["long_dir"], float); R = np.array(fr["radial_dir"], float); T = np.array(fr["thick_dir"], float)
P = XYZ - seed; u = P @ L; r = P @ R; w = P @ T
N = int(d["n"]); is_pc = np.array([mt[g] == "SP_PC" for g in range(N)])
cells = [[round(float(u[i]), 1), round(float(r[i]), 1), round(float(w[i]), 1), int(is_pc[i])] for i in range(N)]

def loc(xyz):
    p = np.array(xyz, float) - seed
    return [round(float(p @ L), 1), round(float(p @ R), 1), round(float(p @ T), 1)]

# 전극 3개(E1 SO·E2 SP·E3 SR) 국소좌표
els = [{"id": e["id"], "layer": e.get("layer", ""), "role": e.get("role", "rec"), "pos": loc(e["xyz_um"])}
       for e in cfg["electrodes"]["list"]]
e3 = next((e["pos"] for e in els if e["id"] == "E3"), els[-1]["pos"])

# 층(SO/SP/SR/SLM) 방사축 범위 — 세포 layer 필드로 산출
lay = wc["layer"].astype(str)
uni = sorted(set(lay.tolist()), key=lambda nm: float(r[lay == nm].mean()))
layers = [{"name": nm, "rmin": round(float(r[lay == nm].min()), 1),
           "rmax": round(float(r[lay == nm].max()), 1)} for nm in uni]
uMin, uMax = round(float(u.min()), 1), round(float(u.max()), 1)

stim_t = float(d["stim_t"])
st = d["spk_t"].astype(float); sid = d["spk_id"].astype(int)
spikes = [[int(g), round(float(t - stim_t), 2)] for t, g in zip(st, sid)]; spikes.sort(key=lambda x: x[1])

# 관측 창(자극기준): -5 ~ 최대 스파이크(+2) 또는 최소 30ms
tmax = max(30.0, (max((s[1] for s in spikes), default=30.0) + 2))
jpath = os.path.splitext(npz)[0] + ".json"
jd = json.load(open(jpath)) if os.path.exists(jpath) else {}
meta = {"n": N, "stim_t": stim_t, "radius": float(d["radius"]), "e3": e3,
        "electrodes": els, "layers": layers, "uMin": uMin, "uMax": uMax,
        "psolve_s": float(jd.get("psolve_s", 0.0)), "spikes_total": len(spikes),
        "active": int(jd.get("active", 0)), "activeE": int(jd.get("activeE", 0)),
        "activeI": int(jd.get("activeI", 0)), "nostim": int(d["nostim"]) if "nostim" in d else 0,
        "tmin": -5.0, "tmax": round(float(tmax), 1)}

data = json.dumps({"meta": meta, "cells": cells, "spikes": spikes}, separators=(",", ":"))
tpl = open(os.path.join(HERE, "ex12_replay_tpl.html"), encoding="utf-8").read()
out_html = tpl.replace("__INJECT__", data)
_uidir = os.path.join(ROOT, "04_experiments", "Ex1_baseline", "ui"); os.makedirs(_uidir, exist_ok=True)
out_path = os.path.join(_uidir, "ex12_replay.html")
open(out_path, "w", encoding="utf-8").write(out_html)
print(f"[Ex12] {os.path.basename(npz)} -> ex12_replay.html ({len(out_html)//1024}KB)")
print(f"  세포 {N} · 스파이크 {len(spikes):,} · 발화 {meta['active']}/{N} · 소요 {meta['psolve_s']/60:.0f}분 · 창 {meta['tmin']}~{meta['tmax']}ms")
