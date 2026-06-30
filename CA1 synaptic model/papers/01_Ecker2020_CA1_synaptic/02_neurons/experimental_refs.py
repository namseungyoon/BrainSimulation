"""
experimental_refs.py — 단일세포 e-특징·감쇠의 실험 기준값 (라이브러리, 번호 X)
============================================================================
자체완결 검증(6_·3_·7_·8_)이 모델 분포에 오버레이할 **실험 기준 범위**의 단일 진실원천.

⚠️ 주의: 아래 값은 **문헌 기반 근사 범위**(전형적 관찰 범위 ~ 평균±SD 수준)이다.
BBP 단일세포 최적화에 쓰인 *정확한* 실험 타깃(평균±SD)은 다운로드 모델 패키지에
포함되어 있지 않다 → 그 정밀 검증은 HippoUnit 단계(`10_hippounit_validation.py`)가 담당.
여기서는 "모델이 실험 분포의 대략적 범위 안에 들어오는가 / e-type 정의에 부합하는가"를 본다.

e-type: cACpyr(피라미드 PC), bAC/cAC/cNAC(인터뉴런).
  - AC = accommodating(적응: 발화 간격이 점점 벌어짐 → adaptation_index 양수)
  - NAC = non-accommodating(비적응: 일정 → adaptation_index ~0)

출처(범주): Hippocampome.org; Bezaire & Soltesz (2013); Pawelzik et al. (2002);
            Tricoire et al. (2011); Magee & Cook (2000); Migliore et al. (2018);
            full-scale CA1 모델 논문(references PDF, S9 등). 정밀 인용은 후속.
"""

# 각 값 = (lo, hi) 대략적 실험 범위. None 이면 해당 e-type 기준 미설정.
REF = {
    # ── CA1 피라미드 세포(PC) ──────────────────────────────────────────
    "cACpyr": {
        "Rin_MOhm":        (50, 180),    # 입력저항(성체 CA1 PC ~50–110, 모델 다소 높음)
        "Vrest_mV":        (-72, -62),   # 정지막전위
        "AP_amplitude_mV": (75, 110),    # 활동전위 진폭
        "AP_width_ms":     (0.7, 1.6),   # AP 반치폭(PC 는 넓은 편)
        "fAHP_depth_mV":   (2, 15),      # 빠른 AHP 깊이(역치 대비)
        "adaptation_index": (0.0, 0.30), # 적응함 → 양수
        "sag_ratio":       (0.02, 0.25), # Ih 에 의한 sag
        "rheobase_nA":     (0.05, 0.35),
        "f_at_max_Hz":     (3, 35),      # +0.4 nA 부근 발화율(완만)
    },
    # ── bAC: bursting Accommodating 인터뉴런 ───────────────────────────
    "bAC": {
        "Rin_MOhm":        (80, 320),
        "Vrest_mV":        (-70, -55),
        "AP_amplitude_mV": (50, 95),
        "AP_width_ms":     (0.4, 1.1),
        "fAHP_depth_mV":   (5, 25),
        "adaptation_index": (0.0, 0.35),  # 적응(초기 폭발 후)
        "sag_ratio":       (0.0, 0.20),
        "rheobase_nA":     (0.02, 0.40),
        "f_at_max_Hz":     (15, 130),
    },
    # ── cAC: continuous Accommodating 인터뉴런 ─────────────────────────
    "cAC": {
        "Rin_MOhm":        (70, 300),
        "Vrest_mV":        (-70, -55),
        "AP_amplitude_mV": (50, 95),
        "AP_width_ms":     (0.35, 1.0),
        "fAHP_depth_mV":   (5, 28),
        "adaptation_index": (0.0, 0.30),  # 적응함 → 양수
        "sag_ratio":       (0.0, 0.18),
        "rheobase_nA":     (0.03, 0.45),
        "f_at_max_Hz":     (20, 160),
    },
    # ── cNAC: continuous Non-Accommodating(속발화형, PV 유사) ──────────
    "cNAC": {
        "Rin_MOhm":        (60, 260),
        "Vrest_mV":        (-70, -55),
        "AP_amplitude_mV": (45, 90),
        "AP_width_ms":     (0.2, 0.6),    # 좁은 AP(속발화)
        "fAHP_depth_mV":   (8, 30),       # 크고 빠른 AHP
        "adaptation_index": (-0.05, 0.08),# 비적응 → ~0
        "sag_ratio":       (0.0, 0.15),
        "rheobase_nA":     (0.05, 0.55),
        "f_at_max_Hz":     (40, 220),     # 높은 발화율
    },
}

# 어떤 특징을 분포 그림에 넣을지(라벨·단위) — 6_ 가 사용
FEATURES = [
    ("Rin_MOhm",        "입력저항 Rin (MΩ)"),
    ("AP_amplitude_mV", "AP 진폭 (mV)"),
    ("AP_width_ms",     "AP 반치폭 (ms)"),
    ("fAHP_depth_mV",   "fAHP 깊이 (mV)"),
    ("adaptation_index", "적응지수 (AC>0, NAC~0)"),
    ("sag_ratio",       "Sag 비율 (Ih)"),
    ("rheobase_nA",     "Rheobase (nA)"),
    ("f_at_max_Hz",     "발화율 @최대전류 (Hz)"),
]

# 수상돌기 감쇠 — 공간상수(λ) 실험/모델 기준 (full-scale CA1 논문 S9)
ATTENUATION = {
    "bpap_lambda_um": {"model_paper": 155.6, "exp_paper": 235.2},  # 역전파 AP 공간상수(S9)
    # PSP 국소→소마 전기긴장 감쇠 길이(근사). 원위 시냅스일수록 소마 EPSP 가 더 감쇠.
    "psp_lambda_um": {"approx_range": (40, 150),
                      "note": "국소→소마 전기긴장 감쇠 길이(근사; Magee & Cook 2000 계열)"},
}

# 탈분극 블록(depolarization block) 개시 전류 — CA1 PC 근사
DEPOL_BLOCK = {
    "cACpyr": {"I_block_nA": (0.5, 1.8)},   # 강한 전류에서 발화 정지(근사 범위)
}

ETYPE_ORDER = ["cACpyr", "bAC", "cAC", "cNAC"]
ETYPE_COLOR = {"cACpyr": "tab:red", "bAC": "tab:green", "cAC": "tab:blue", "cNAC": "tab:orange"}


def band(etype, feature):
    """(lo, hi) 또는 None."""
    return REF.get(etype, {}).get(feature)
