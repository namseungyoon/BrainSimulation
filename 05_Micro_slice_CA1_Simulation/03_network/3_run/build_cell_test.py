# -*- coding: utf-8 -*-
"""
03_network/3_run/build_cell_test.py  —  3-3 검증: 세포 1개 인스턴스화

emodel hoc(고유 rename) + morphology_library의 .swc 로 실제 생물물리 세포를
NEURON에 짓고, 정상 동작(구획 수·정지막전위·전류주입 발화)을 확인한다.
축소 없음 — 완전형태를 그대로 NEURON에 올림.

실행(반드시 ca1sim + 컴파일된 mechanism):
  python 03_network/3_run/build_cell_test.py
"""
import os
import re
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.abspath(os.path.join(ROOT, ".."))
DERIVED = os.path.join(ROOT, "data", "derived")
MORPHLIB = os.path.join(ROOT, "data", "morphology_library", "morphology_library")
MECH = os.path.join(ROOT, "scratch", "mechbuild", "x86_64", "libnrnmech.so")
SCR = os.path.join(ROOT, "scratch")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

from neuron import h


def load_emodel(hoc_path, uniq):
    """hoc 읽어 템플릿 이름을 고유하게 rename 후 로드. 반환: 새 템플릿 이름."""
    txt = open(hoc_path).read()
    m = re.search(r"begintemplate\s+(\w+)", txt)
    tname = m.group(1)
    new = f"{tname}_{uniq}"
    txt = re.sub(r"\b" + tname + r"\b", new, txt)
    tmp = os.path.join(SCR, f"emodel_{uniq}.hoc")
    open(tmp, "w").write(txt)
    h.load_file(tmp.replace("\\", "/"))
    return new


def main():
    h.load_file("stdrun.hoc")
    h.nrn_load_dll(MECH.replace("\\", "/"))
    h.celsius = 34
    h.v_init = -70

    wc = np.load(os.path.join(DERIVED, "window_cells.npz"), allow_pickle=True)
    mt = wc["mtype"].astype(str); morph = wc["morphology"].astype(str)
    tpl = np.array([t.replace("hoc:", "") for t in wc["model_template"]])
    mm = json.load(open(os.path.join(DERIVED, "memodel_map.json"), encoding="utf-8"))["template_map"]

    # E3 근처 추체 1개 (gid 501)
    gid = 501
    print(f"[대상] gid {gid} · mtype {mt[gid]} · morphology {morph[gid]}")
    print(f"       emodel {tpl[gid]}")
    hoc_rel = mm[tpl[gid]]["hoc"]
    hoc_path = os.path.join(REPO, hoc_rel)
    print(f"       hoc {hoc_rel}")

    tname = load_emodel(hoc_path, "c0")
    print(f"[템플릿] {tname} 로드")
    cell = getattr(h, tname)(MORPHLIB.replace("\\", "/"), morph[gid] + ".swc")
    print(f"[인스턴스화 성공]")

    nsec = sum(1 for _ in cell.all)
    nseg = sum(s.nseg for s in cell.all)
    print(f"[구획] section {nsec}개 · segment {nseg}개")

    # 정지막전위
    h.finitialize(-70); h.continuerun(200)
    vrest = cell.soma[0](0.5).v
    print(f"[정지 막전위] {vrest:.1f} mV (200ms 후)")

    # 전류 계단 → F-I 곡선 + 대표 전압파형
    amps = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5]
    dur, delay = 400.0, 100.0
    ic = h.IClamp(cell.soma[0](0.5)); ic.delay = delay; ic.dur = dur
    apc = h.APCount(cell.soma[0](0.5)); apc.thresh = -20
    tvec = h.Vector(); tvec.record(h._ref_t)
    vsoma = h.Vector(); vsoma.record(cell.soma[0](0.5)._ref_v)
    counts = []; traces = {}
    for a in amps:
        ic.amp = a; h.finitialize(-70); h.continuerun(delay + dur + 100)
        counts.append(int(apc.n))
        if a in (0.4, 0.8, 1.5):
            traces[a] = (np.array(tvec), np.array(vsoma))
    rate = [c / (dur / 1000.0) for c in counts]   # Hz
    print("[F-I] " + " · ".join(f"{a}nA:{c}sp" for a, c in zip(amps, counts)))

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1.3]})
    ax[0].plot(amps, rate, "o-", color="#C44E52", lw=2, ms=7)
    ax[0].set_xlabel("주입 전류 (nA)"); ax[0].set_ylabel("발화율 (Hz)")
    ax[0].set_title(f"(a) F-I 곡선 — 추체 gid {gid}\nrheobase ≈ 0.5nA")
    ax[0].grid(alpha=0.3)
    for a, (t, v) in traces.items():
        ax[1].plot(t, v, lw=0.8, label=f"{a} nA ({counts[amps.index(a)]}sp)")
    ax[1].set_xlabel("시간 (ms)"); ax[1].set_ylabel("소마 전압 (mV)")
    ax[1].set_title("(b) 소마 전압 파형 (전류 계단)"); ax[1].legend(fontsize=9)
    fig.suptitle(f"3-3 세포 검증 — {mt[gid]} 완전형태({nsec}sec·{nseg}seg)·정지 {vrest:.1f}mV·emodel {tpl[gid][:28]}", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "3-3_cell_test.png"), dpi=130); plt.close(fig)
    print(f"\n[검증 완료] 세포 완전형태 인스턴스화·동작 확인 · 그림 -> {FIG}/3-3_cell_test.png")


if __name__ == "__main__":
    main()
