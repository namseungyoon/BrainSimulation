# -*- coding: utf-8 -*-
"""lib/fepsp_record.py — NEURON 막전류 기반 fEPSP 기록기 (재사용 인프라).

fast_imem로 세그먼트별 막전류 I(t)[nA]를 기록하고, mea_forward로 전극(E1/E2/E3)의
장전위 V(t)[mV]를 계산. MPI에서 각 랭크가 자기 세포 기여분(부분합)을 만들고 allreduce로
전체 합산 → 중앙 저장 불필요. Ex3·Ex4·Ex5·Ex8·Ex9·Ex11 등 모든 필드 실험이 공용.

사용 패턴:
    rec = FEPSPRecorder(elec_xyz)                  # 전극 global xyz (n_elec,3)
    for g in mine:
        rec.add_cell(cells[g], XYZ[g], rot_g)      # 세그먼트 등록 (global 위치)
    rec.finalize(rec_dt=0.1)                        # fast_imem on + 가중치·기록벡터 (finitialize 前)
    ... h.finitialize(-70); pc.psolve(TSTOP) ...
    V = rec.potential_allreduce(comm)              # (n_elec, n_t) mV, 전 랭크 합산
    t = rec.times()                                # (n_t,) ms

주의: fast_imem의 i_membrane_ 단위 = nA(세그먼트 총 막전류). mea_forward도 nA·µm·S/m→mV.
"""
import numpy as np
from neuron import h
try:
    import mea_forward as mf
except ImportError:  # 패키지로 import될 때
    from . import mea_forward as mf


class FEPSPRecorder:
    def __init__(self, elec_xyz, sigma=mf.SIGMA, rmin=mf.RMIN, skip=("axon", "node", "myelin"), stride=1):
        self.elec = np.asarray(elec_xyz, float).reshape(-1, 3)
        self.sigma = sigma
        self.rmin = rmin
        self.skip = skip
        self.stride = int(stride)    # 세그먼트 서브샘플(대규모 setup O(n²) 회피). W를 stride배 보정
        self._ctr = 0
        self._pos = []       # 세그먼트 global 중심 xyz
        self._refs = []      # 세그먼트 객체(i_membrane_ 접근용)
        self._soma = []      # 세그먼트가 soma인가 (시각화용)
        self._diam = []      # 세그먼트 지름 (시각화용)
        self.W = None        # (n_elec, n_seg) 가중치 [mV per nA]
        self.vecs = []       # 세그먼트별 기록 Vector
        self.tvec = None

    def add_cell(self, cell, xyz_global, rot):
        """cell의 (축삭 제외) 세그먼트 중심을 global 좌표로 등록. rot=scipy Rotation."""
        xyz_global = np.asarray(xyz_global, float)
        for sec in cell.all:
            nm = sec.name(); n = int(sec.n3d())
            if n < 2 or any(s in nm for s in self.skip):
                continue
            arc = np.array([sec.arc3d(i) for i in range(n)]); Lt = arc[-1] or 1.0
            xs = np.array([sec.x3d(i) for i in range(n)])
            ys = np.array([sec.y3d(i) for i in range(n)])
            zs = np.array([sec.z3d(i) for i in range(n)])
            issoma = "soma" in nm
            for seg in sec:
                self._ctr += 1
                if self.stride > 1 and (self._ctr % self.stride) != 0:
                    continue                                  # 서브샘플: 이 세그먼트 건너뜀
                a = seg.x * Lt
                loc = np.array([np.interp(a, arc, xs), np.interp(a, arc, ys), np.interp(a, arc, zs)])
                self._pos.append(xyz_global + rot.apply(loc))
                self._refs.append(seg)
                self._soma.append(issoma)
                self._diam.append(float(seg.diam))

    def n_seg(self):
        return len(self._refs)

    def finalize(self, rec_dt=0.1, rec_tvec=None):
        """fast_imem 활성 + 전극 가중치 계산 + 세그먼트 막전류 기록 벡터 부착. finitialize 前 호출.
        rec_tvec(h.Vector, 명시 시각들) 주면 그 시각만 기록(관측창만 → 메모리 절약)."""
        h.cvode.use_fast_imem(1)
        segxyz = np.array(self._pos) if self._pos else np.zeros((0, 3))
        self.W = (mf.psa_weights(self.elec, segxyz, self.sigma, self.rmin)
                  if len(segxyz) else np.zeros((len(self.elec), 0)))
        self.W = self.W * self.stride    # 서브샘플 보정: 기록 세그먼트가 stride개를 대표
        if rec_tvec is not None:
            self.tvec = rec_tvec
            self.vecs = [h.Vector() for _ in self._refs]
            for v, seg in zip(self.vecs, self._refs):
                v.record(seg._ref_i_membrane_, rec_tvec)
        else:
            self.tvec = h.Vector(); self.tvec.record(h._ref_t, rec_dt)
            self.vecs = []
            for seg in self._refs:
                v = h.Vector(); v.record(seg._ref_i_membrane_, rec_dt)
                self.vecs.append(v)
        return self

    def potential_local(self):
        """이 랭크 세그먼트 기여분 V_local (n_elec, n_t) [mV]."""
        nt = int(self.tvec.size())
        if not self.vecs:
            return np.zeros((len(self.elec), nt))
        I = np.array([np.array(v)[:nt] for v in self.vecs])   # (n_seg, n_t) nA
        return self.W @ I

    def potential_allreduce(self, comm):
        """전 랭크 합산 V (n_elec, n_t) [mV]."""
        from mpi4py import MPI
        Vloc = self.potential_local()
        return comm.allreduce(Vloc, op=MPI.SUM)

    def times(self):
        return np.array(self.tvec)

    @staticmethod
    def fepsp_slope(V, t, win=(0.3, 1.5), t0=0.0):
        """fEPSP 초기 기울기(mV/ms): t0+win 구간 선형회귀. V:(...,T), t:(T,)."""
        t = np.asarray(t, float)
        m = (t >= t0 + win[0]) & (t <= t0 + win[1])
        if m.sum() < 2:
            return np.full(np.asarray(V).shape[:-1], np.nan)
        A = np.vstack([t[m], np.ones(m.sum())]).T
        coef, *_ = np.linalg.lstsq(A, np.asarray(V)[..., m].T, rcond=None)
        return coef[0]
