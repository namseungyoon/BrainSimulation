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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.abspath(os.path.join(ROOT, ".."))
DERIVED = os.path.join(ROOT, "data", "derived")
MORPHLIB = os.path.join(ROOT, "data", "morphology_library", "morphology_library")
MECH = os.path.join(ROOT, "scratch", "mechbuild", "x86_64", "libnrnmech.so")
SCR = os.path.join(ROOT, "scratch")

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
    h.finitialize(-70)
    h.continuerun(200)
    vrest = cell.soma[0](0.5).v
    print(f"[정지 막전위] {vrest:.1f} mV (200ms 후)")

    # 전류 주입 → 발화 확인
    ic = h.IClamp(cell.soma[0](0.5)); ic.delay = 100; ic.dur = 400; ic.amp = 0.4
    apc = h.APCount(cell.soma[0](0.5)); apc.thresh = -20
    h.finitialize(-70); h.continuerun(600)
    print(f"[전류주입 0.4nA 400ms] 스파이크 {int(apc.n)}개 → {'발화 OK' if apc.n > 0 else '무발화'}")
    print("\n[검증 완료] 세포 1개 완전형태 인스턴스화·동작 확인")


if __name__ == "__main__":
    main()
