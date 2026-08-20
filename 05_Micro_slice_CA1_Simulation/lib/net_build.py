# -*- coding: utf-8 -*-
"""
lib/net_build.py  —  3-2/3-3 NEURON 네트워크 빌더 (import 모듈)

emodel hoc(고유 rename) + morphology_library .swc 로 세포를 인스턴스화하고,
시냅스(ProbAMPANMDA_EMS/ProbGABAAB_EMS, STP)를 삽입하는 유틸.
축소 없음 — 완전형태.
"""
import os
import re
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(ROOT, ".."))
DERIVED = os.path.join(ROOT, "data", "derived")
MORPHLIB = os.path.join(ROOT, "data", "morphology_library", "morphology_library")
MECH = os.path.join(ROOT, "scratch", "mechbuild", "x86_64", "libnrnmech.so")
SCR = os.path.join(ROOT, "scratch")


class NetBuilder:
    def __init__(self):
        from neuron import h
        self.h = h
        h.load_file("stdrun.hoc")
        h.nrn_load_dll(MECH.replace("\\", "/"))
        h.celsius = 34
        self.wc = np.load(os.path.join(DERIVED, "window_cells.npz"), allow_pickle=True)
        self.mt = self.wc["mtype"].astype(str)
        self.morph = self.wc["morphology"].astype(str)
        self.tpl = np.array([t.replace("hoc:", "") for t in self.wc["model_template"]])
        self.mm = json.load(open(os.path.join(DERIVED, "memodel_map.json"), encoding="utf-8"))["template_map"]
        self._tpl_cache = {}   # emodel_template -> new template name
        self.cells = {}        # gid -> cell object

    def _load_template(self, emodel):
        if emodel in self._tpl_cache:
            return self._tpl_cache[emodel]
        hoc_path = os.path.join(REPO, self.mm[emodel]["hoc"])
        txt = open(hoc_path).read()
        tname = re.search(r"begintemplate\s+(\w+)", txt).group(1)
        uniq = f"{tname}_{len(self._tpl_cache)}"
        txt = re.sub(r"\b" + tname + r"\b", uniq, txt)
        # MPI: 랭크(프로세스)별로 임시 hoc 파일 분리 — 동시 write 충돌 방지
        tmp = os.path.join(SCR, f"emodel_tpl_p{os.getpid()}_{len(self._tpl_cache)}.hoc")
        open(tmp, "w").write(txt)
        self.h.load_file(tmp.replace("\\", "/"))
        self._tpl_cache[emodel] = uniq
        return uniq

    def build_cell(self, gid):
        if gid in self.cells:
            return self.cells[gid]
        tname = self._load_template(self.tpl[gid])
        cell = getattr(self.h, tname)(MORPHLIB.replace("\\", "/"), self.morph[gid] + ".swc")
        self.cells[gid] = cell
        return cell

    def build_cells(self, gids):
        for g in gids:
            self.build_cell(int(g))
        return len(self.cells)

    def counts(self):
        nsec = nseg = 0
        for c in self.cells.values():
            for s in c.all:
                nsec += 1; nseg += s.nseg
        return nsec, nseg


def rss_mb():
    try:
        for ln in open("/proc/self/status"):
            if ln.startswith("VmRSS:"):
                return int(ln.split()[1]) / 1024.0
    except Exception:
        return -1.0
    return -1.0
