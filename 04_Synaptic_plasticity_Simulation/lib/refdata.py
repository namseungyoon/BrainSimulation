# -*- coding: utf-8 -*-
"""lib/refdata.py — 문헌 기준값 표 (번호 없음 = import 전용 모듈)

⚠️ 아래 값은 문헌 기반 대략 범위(전형적 관찰 범위 수준)다. BBP 단일세포 최적화의 정확한
타깃(평균±SD)은 다운로드 번들에 없다. 여기서는 "모델이 대략 범위 안인가 / e-type 정의에
부합하는가"를 본다. 정밀 검증은 HippoUnit 몫(04 범위 밖).
출처(범주): Hippocampome.org · Migliore et al. 2018 · Magee & Cook 2000 · full-scale CA1 논문.
"""

# CA1 추체세포(cACpyr) e-특징 대략 범위 (lo, hi)
CACPYR = {
    "Rin_MOhm":         (50, 180),     # 입력저항
    "Vrest_mV":         (-72, -62),    # 정지막전위
    "AP_amplitude_mV":  (75, 110),     # 활동전위 진폭
    "AP_halfwidth_ms":  (0.7, 1.6),    # AP 반치폭
    "AP_threshold_mV":  (-55, -40),    # 발화 역치
    "sag_ratio":        (0.02, 0.25),  # Ih 에 의한 sag
    "rheobase_nA":      (0.05, 0.35),  # 발화 시작 최소전류
    "adaptation_index": (0.0, 0.60),   # 발화 적응(양수)
}
