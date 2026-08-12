# -*- coding: utf-8 -*-
"""12_lfp/lfp_calc.py  —  E4 세포외 LFP/fEPSP 순방향 계산기 (무의존, numpy만)

준정적(quasistatic)·옴성(ohmic)·등방(isotropic)·균질(homogeneous) 매질 가정에서
세포외 전위 = 세그먼트 총 막전류의 선형합:  V_j(t) = sum_i M_ji * I_i(t)

  - I_i  : 세그먼트 i의 총 막전류 (NEURON i_membrane_, 단위 nA, use_fast_imem 필요)
  - M_ji : 전달행렬 (전극 j x 세그먼트 i). 선전류원(LSA) 또는 점전류원(PSA)
  - 좌표 um, sigma S/m, I nA -> V 는 곧바로 mV (환산계수 10^-3 내장:
           nA/um = 1e-3 A/m, /(S/m) = 1e-3 V)

부호 규약(중요):
  NEURON i_membrane_ 은 '외향(+)/내향(-)' 총 막전류.
  volume-conductor 식 V=(1/4pi.sigma).sum I/r 에 그대로 넣으면
  흥분성 시냅스 sink(막 내향, i_membrane_<0) 근처 전극에서 V<0 (fEPSP 음성 편향).
  즉 I_i = i_membrane_ 를 부호 그대로 사용한다.

근거: Holt & Koch 1999 (LSA log 공식); Linden 2014 (LFPy); Ness 2021 (준정적 정리).
LSA 는 asinh 형식으로 구현(Holt&Koch log 형과 수학적 등가, 수치 안정·부호 케이스 분기 불요):
  V = I/(4pi.sigma.L) * [ asinh((L - s0)/rho) - asinh((-s0)/rho) ]
  s0  = 전극의 세그먼트축 투영(시작점 기준),  rho = 축까지 수직거리(반경으로 하한 클램프).
"""
import numpy as np


# ----------------------------------------------------------------------------
# 세그먼트 기하 수집 (NEURON cell -> 배열)
# ----------------------------------------------------------------------------
def seg_point(sec, x):
    """섹션 sec 의 정규화 위치 x(0~1)에서 3D 좌표(um). arc3d 보간(sc_epsp_placement 방식)."""
    n = int(sec.n3d())
    if n == 0:
        return np.zeros(3)
    if n == 1:
        return np.array([sec.x3d(0), sec.y3d(0), sec.z3d(0)], float)
    arcs = np.array([sec.arc3d(i) for i in range(n)], float)
    tot = arcs[-1] if arcs[-1] > 0 else sec.L
    arcs = arcs / tot
    xs = np.array([sec.x3d(i) for i in range(n)], float)
    ys = np.array([sec.y3d(i) for i in range(n)], float)
    zs = np.array([sec.z3d(i) for i in range(n)], float)
    return np.array([np.interp(x, arcs, xs), np.interp(x, arcs, ys), np.interp(x, arcs, zs)])


def collect_segments(seclist):
    """모든 섹션의 모든 세그먼트 기하 수집.
    반환 dict: p0(N,3) 시작, p1(N,3) 끝, mid(N,3) 중점, length(N), radius(N), segs(list)."""
    p0, p1, mid, length, radius, segs = [], [], [], [], [], []
    for sec in seclist:
        nseg = sec.nseg
        for k, seg in enumerate(sec):
            a = k / nseg
            b = (k + 1) / nseg
            xa = seg_point(sec, a)
            xb = seg_point(sec, b)
            p0.append(xa)
            p1.append(xb)
            mid.append(0.5 * (xa + xb))
            length.append(float(np.linalg.norm(xb - xa)))
            radius.append(float(seg.diam) / 2.0)
            segs.append(seg)
    n_bad = sum(1 for sec in seclist if int(sec.n3d()) == 0)
    if n_bad:
        print(f"[lfp_calc][경고] 3D점 없는 섹션 {n_bad}개 -> 원점(0,0,0) 배치 위험. "
              f"collect_segments 전에 h.define_shape() 호출 권장.", flush=True)
    return dict(p0=np.array(p0), p1=np.array(p1), mid=np.array(mid),
                length=np.array(length), radius=np.array(radius), segs=segs)


def setup_imem(segs):
    """use_fast_imem 활성화 + 각 세그먼트 i_membrane_(nA) 기록 Vector 등록.
    반환 (vecs, cv). 고정 dt 유지: cv.active 는 호출하지 않는다."""
    from neuron import h
    cv = h.CVode()
    cv.use_fast_imem(1)
    vecs = []
    for seg in segs:
        v = h.Vector()
        v.record(seg._ref_i_membrane_)
        vecs.append(v)
    return vecs, cv


# ----------------------------------------------------------------------------
# 전달행렬
# ----------------------------------------------------------------------------
def lsa_matrix(geom, electrodes, sigma=0.3):
    """선전류원(LSA) 전달행렬 M[j,i]  (전극 J x 세그먼트 N). 단위 nA*um*S/m -> mV.
    V = M @ I."""
    p0 = geom["p0"]
    p1 = geom["p1"]
    L = geom["length"]
    rad = geom["radius"]
    mid = geom["mid"]
    E = np.atleast_2d(np.asarray(electrodes, float))
    J = E.shape[0]
    N = p0.shape[0]
    Lsafe = np.where(L > 1e-9, L, 1e-9)
    axis = (p1 - p0) / Lsafe[:, None]
    short = L <= 1e-6              # 3D 점이 겹쳐 길이~0 인 세그먼트는 점전류원으로
    M = np.zeros((J, N))
    for j in range(J):
        rel = E[j] - p0                                   # 전극 - 시작점 (N,3)
        s0 = np.einsum("ij,ij->i", rel, axis)             # 축방향 투영(시작점 기준)
        perp2 = np.einsum("ij,ij->i", rel, rel) - s0 ** 2
        rho = np.sqrt(np.clip(perp2, 0.0, None))
        rho = np.maximum(rho, rad)                        # 반경 하한(로그 발산 방지)
        line = (np.arcsinh((L - s0) / rho) - np.arcsinh((-s0) / rho)) / (4.0 * np.pi * sigma * Lsafe)
        d = np.maximum(np.linalg.norm(E[j] - mid, axis=1), rad)
        point = 1.0 / (4.0 * np.pi * sigma * d)
        M[j] = np.where(short, point, line)
    return M


def psa_matrix(geom, electrodes, sigma=0.3):
    """점전류원(PSA) 전달행렬. 원거리 수렴 검증·비교용."""
    mid = geom["mid"]
    rad = geom["radius"]
    E = np.atleast_2d(np.asarray(electrodes, float))
    J = E.shape[0]
    N = mid.shape[0]
    M = np.zeros((J, N))
    for j in range(J):
        d = np.maximum(np.linalg.norm(E[j] - mid, axis=1), rad)
        M[j] = 1.0 / (4.0 * np.pi * sigma * d)
    return M


def moi_point_matrix(geom, electrodes, sigma_T=0.3, sigma_S=1.5, sigma_G=0.0,
                     h=800.0, n_img=20):
    """MEA 슬라이스 3층 영상법(method of images, Ness 2015) 전달행렬 M[j,i].

    기하: 전극은 유리(MEA)면 z=0 · 조직 슬라이스 z∈[0,h] · 식염수 z>h.
    유리 절연(sigma_G=0) 표준 -> 소스가 유리면에서 완전 반사(전위 2배).
    식염수 경계 반사계수 W_TS=(sigma_T-sigma_S)/(sigma_T+sigma_S) (기본 -2/3)로 영상급수.

      M = (1/4πσ_T) [ 2·g(z') + 2·Σ_{n=1}^{n_img} W_TS^n ( g(2nh−z') + g(2nh+z') ) ],
      g(w) = 1/√(ρ² + w²),  ρ=전극과 소스의 수평거리, z'=소스 높이(슬라이스 내로 클램프).

    전제: 전극은 z=0 평면. 소스(세그먼트)는 슬라이스 내부(0<z'<h). 단위 nA·µm·S/m -> mV.
    극한 검증: h→∞ 또는 W_TS=0 이면 M → 2×(무한매질 점전류원)  (유리 반사 2배).
    """
    mid = geom["mid"]
    rad = geom["radius"]
    E = np.atleast_2d(np.asarray(electrodes, float))
    # 가정 명시(오용 시 조용히 틀리지 않고 실패): 전극은 유리면 z=0 · 유리 절연 sigma_G=0
    if E.shape[1] > 2:
        assert np.all(np.abs(E[:, 2]) < 1e-6), "moi_point_matrix: 전극은 유리면 z=0 가정(E[:,2]≈0)"
    assert abs(sigma_G) < 1e-12, "moi_point_matrix: 유리 절연(sigma_G=0)만 구현 — sigma_G!=0 미지원"
    W = (sigma_T - sigma_S) / (sigma_T + sigma_S)
    dx = E[:, 0][:, None] - mid[:, 0][None, :]
    dy = E[:, 1][:, None] - mid[:, 1][None, :]
    rho2 = dx * dx + dy * dy                       # (J, N)
    zc = np.clip(mid[:, 2], 0.0, h)[None, :]       # (1, N) 소스 높이
    radb = rad[None, :]

    def g(w):
        return 1.0 / np.maximum(np.sqrt(rho2 + w * w), radb)

    S = 2.0 * g(zc)
    for n in range(1, n_img + 1):
        S = S + 2.0 * (W ** n) * (g(2 * n * h - zc) + g(2 * n * h + zc))
    return S / (4.0 * np.pi * sigma_T)


def compute_lfp(M, imem):
    """V = M @ I.  imem (N_seg, N_t) nA -> V (N_elec, N_t) mV."""
    return M @ np.asarray(imem)


def current_conservation(imem):
    """매 시각 sum_i I_i(t). 전극 주입 없으면 ~0 이어야 함(전류보존 게이트).
    반환: (max|sumI|, max|I|)  ->  비율이 작으면 정상."""
    I = np.asarray(imem)
    s = np.abs(I.sum(axis=0))
    return float(s.max()), float(np.abs(I).max())
