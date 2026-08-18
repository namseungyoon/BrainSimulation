# -*- coding: utf-8 -*-
"""lib/atlas_query.py — 크롭 atlas에서 임의 3D 점의 층·정규화깊이 질의.

3단계(01_tissue/3_atlas_prep/crop_atlas.py)가 만든 data/derived/atlas_crop.npz 를 읽어,
배치(02_neurons)·방향·fEPSP 등에서 세포/전극 좌표의 층(SO/SP/SR/SLM)과
정규화깊이 nd(0=SO 하단 → 1=SLM 상단)를 벡터화 질의한다.

사용:
    from lib.atlas_query import AtlasQuery
    aq = AtlasQuery("data/derived/atlas_crop.npz")
    layers = aq.layer(xyz)        # (N,) 문자열
    nd     = aq.depth_norm(xyz)   # (N,) float, 밖=nan
"""
import numpy as np

LAYERS = {1: "SO", 2: "SP", 3: "SR", 4: "SLM"}


class AtlasQuery:
    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=True)
        self.regions = d["regions"]
        self.phy = d["phy"]; self.base = d["base"]; self.top = d["top"]
        self.origin = np.asarray(d["origin"], float)
        self.vs = float(d["vsize"])
        self.dims = np.array(self.regions.shape)

    def _vox(self, xyz):
        xyz = np.atleast_2d(np.asarray(xyz, float))
        vi = np.floor((xyz - self.origin) / self.vs).astype(int)
        inb = ((vi >= 0) & (vi < self.dims)).all(1)
        vi = np.clip(vi, 0, self.dims - 1)
        return vi, inb

    def layer(self, xyz):
        vi, inb = self._vox(xyz)
        lab = self.regions[vi[:, 0], vi[:, 1], vi[:, 2]]
        out = np.array([LAYERS.get(int(l), "밖") for l in lab], dtype=object)
        out[~inb] = "밖"
        return out

    def depth_norm(self, xyz):
        vi, inb = self._vox(xyz)
        base = self.base[vi[:, 0], vi[:, 1], vi[:, 2]]
        top = self.top[vi[:, 0], vi[:, 1], vi[:, 2]]
        phy = self.phy[vi[:, 0], vi[:, 1], vi[:, 2]]
        t = top - base
        nd = (phy - base) / np.where(t != 0, t, np.nan)
        nd[~inb] = np.nan
        return nd
