# -*- coding: utf-8 -*-
"""
lib/mea_forward.py  —  fEPSP 순방향(forward) 모델: 막전류 → 전극 장전위

세그먼트별 막전류 I_seg(t)[nA]와 세그먼트 3D 위치로부터, 전극 e에서의 장전위
V_e(t)[mV] 를 준정적(quasi-static) 근사로 계산.

    V_e(t) = 1/(4πσ) · Σ_seg  I_seg(t) / r(e, seg)

단위: I[nA], r[µm], σ[S/m] ⇒ V[mV]  (유도: 1e-9A / (4πσ · 1e-6 m) = 1e-3 V = 1 mV)

3기법:
  PSA — 점원(세그먼트 중심 1점)          : 가장 단순, 전극이 뉴로필에서 조금 떨어지면 충분
  LSA — 선원(세그먼트를 선분으로, Holt&Koch 1999): 전극이 돌기에 가까울 때 정확
  MoI — 3층 영상법(Ness 2015)            : 슬라이스(조직/유리/식염수) 경계 반사 보정 (차후 이식)

MPI 사용법: 각 랭크가 자기 세포의 W_local(전극×로컬세그) 를 만들고, I_local(t) 를
기록 → V_local = W_local @ I_local → allreduce 합산 = 전체 V_e(t). 중앙 저장 불필요.

참고 상수: 뇌 조직 세포외 전도도 σ ≈ 0.3 S/m (0.3~0.4 통용, Ness 2015).
"""
import numpy as np

SIGMA = 0.3          # S/m, 세포외 전도도
RMIN = 3.0           # µm, 특이점 방지 최소 거리(세그먼트 반경 규모)
K_MV = 1.0           # V[mV] = K_MV/(4πσ) · ΣI_nA/r_µm  (단위계수 흡수됨=1)


def psa_weights(elec_xyz, seg_xyz, sigma=SIGMA, rmin=RMIN):
    """점원(PSA) 가중치 W[n_elec, n_seg]. V_mV(t) = W @ I_nA(t).

    elec_xyz: (E,3) µm · seg_xyz: (S,3) µm
    """
    elec = np.asarray(elec_xyz, float).reshape(-1, 3)
    seg = np.asarray(seg_xyz, float).reshape(-1, 3)
    # r[e,s] = |elec_e - seg_s|
    d = elec[:, None, :] - seg[None, :, :]           # (E,S,3)
    r = np.sqrt((d * d).sum(-1))                      # (E,S)
    r = np.maximum(r, rmin)
    return (K_MV / (4.0 * np.pi * sigma)) / r         # (E,S)  [mV per nA]


def lsa_weights(elec_xyz, seg_start, seg_end, sigma=SIGMA, rmin=RMIN):
    """선원(LSA, Holt & Koch 1999) 가중치 W[n_elec, n_seg].

    각 세그먼트를 start→end 선분으로 보고 선적분:
      V = I/(4πσ L) · ln| (√(h²+ρ²) − h) / (√(l²+ρ²) − l) |
      L=선분길이, ρ=선(연장)에서 전극의 수직거리, l=start까지 세로거리, h=l−L(end까지).
    """
    elec = np.asarray(elec_xyz, float).reshape(-1, 3)
    a = np.asarray(seg_start, float).reshape(-1, 3)
    b = np.asarray(seg_end, float).reshape(-1, 3)
    ab = b - a                                        # (S,3)
    L = np.linalg.norm(ab, axis=1)                    # (S,)
    L = np.maximum(L, 1e-6)
    u = ab / L[:, None]                               # 단위벡터 (S,3)
    W = np.zeros((len(elec), len(a)))
    for e in range(len(elec)):
        pe = elec[e][None, :] - a                     # (S,3) 전극-시작점
        l = (pe * u).sum(1)                           # 세로거리 start기준 (S,)
        # 수직거리 ρ
        perp = pe - l[:, None] * u
        rho = np.linalg.norm(perp, axis=1)            # (S,)
        rho = np.maximum(rho, rmin)
        h = l - L                                     # end 기준 세로거리
        num = np.sqrt(h * h + rho * rho) - h
        den = np.sqrt(l * l + rho * rho) - l
        # den/num 부호 안전
        val = np.log(np.maximum(np.abs(num), 1e-12) / np.maximum(np.abs(den), 1e-12))
        W[e] = (K_MV / (4.0 * np.pi * sigma * L)) * val
    return W                                          # (E,S) [mV per nA]


class MEAForward:
    """전극 세트에 대한 순방향 계산기. weights를 한 번 만들고 여러 I(t)에 재사용."""

    def __init__(self, elec_xyz, sigma=SIGMA, rmin=RMIN):
        self.elec = np.asarray(elec_xyz, float).reshape(-1, 3)
        self.sigma = sigma
        self.rmin = rmin
        self.W = None
        self.method = None

    def build_psa(self, seg_xyz):
        self.W = psa_weights(self.elec, seg_xyz, self.sigma, self.rmin)
        self.method = "PSA"
        return self

    def build_lsa(self, seg_start, seg_end):
        self.W = lsa_weights(self.elec, seg_start, seg_end, self.sigma, self.rmin)
        self.method = "LSA"
        return self

    def potential(self, I_nA):
        """I_nA: (n_seg,) 또는 (n_seg, n_time). 반환 V_mV: (n_elec,) 또는 (n_elec, n_time)."""
        return self.W @ np.asarray(I_nA, float)

    @staticmethod
    def slope(V, t, win=(0.5, 2.0), stim_t=0.0):
        """fEPSP 초기 기울기(mV/ms): 자극 후 win(ms) 구간 선형회귀 기울기.
        V:(...,T), t:(T,) [ms, stim 기준이면 stim_t=0]."""
        t = np.asarray(t, float)
        m = (t >= stim_t + win[0]) & (t <= stim_t + win[1])
        if m.sum() < 2:
            return np.full(V.shape[:-1], np.nan)
        tt = t[m]
        A = np.vstack([tt, np.ones_like(tt)]).T
        coef, *_ = np.linalg.lstsq(A, np.asarray(V)[..., m].T, rcond=None)
        return coef[0]                                # 기울기 (…,)


def _selftest():
    """단위·부호 점검: SR(깊은) 전극에 흥분성 sink(-)를 두면 음전위가 나와야 함."""
    # 전극 3개 (SO/SP/SR 모사: y가 깊이)
    elec = np.array([[0, 0, 0], [0, 100, 0], [0, 200, 0]], float)  # E1,E2,E3
    # 시냅스 sink: SR 근처(y=200)에 유입전류(막전류 음수 = 세포내로 유입 → 세포외 sink)
    seg = np.array([[10, 200, 0]], float)   # SR 위치 근처 세그먼트
    I = np.array([-1.0])                     # -1 nA (sink)
    fw = MEAForward(elec).build_psa(seg)
    V = fw.potential(I)
    print("[selftest] PSA V(E1,E2,E3) mV =", np.round(V, 4))
    assert V[2] < V[0], "SR 전극이 sink에 더 가까워 더 음전위여야 함"
    # LSA도 유한값
    fw2 = MEAForward(elec).build_lsa(np.array([[10, 195, 0]]), np.array([[10, 205, 0]]))
    print("[selftest] LSA V(E1,E2,E3) mV =", np.round(fw2.potential(I), 4))
    print("[selftest] OK")


if __name__ == "__main__":
    _selftest()
