# -*- coding: utf-8 -*-
"""lib/refdata.py — 문헌 기준값 표 (번호 없음 = import 전용 모듈)

⚠️ 아래 값은 문헌 기반 대략 범위(전형적 관찰 범위 수준)다. BBP 단일세포 최적화의 정확한
타깃(평균±SD)은 다운로드 번들에 없다. 여기서는 "모델이 대략 범위 안인가 / e-type 정의에
부합하는가"를 본다. 정밀 검증은 HippoUnit 몫(04 범위 밖).
출처(범주): Hippocampome.org · Migliore et al. 2018 · Magee & Cook 2000 · full-scale CA1 논문.
"""

# ★ 이 벤치(CA1 PC -> CA1 PC)의 uEPSP 기준 문헌 — Deuchars & Thomson 1996
#   Neuroscience 74(4):1009-1018 (PMID 8895869, DOI 10.1016/0306-4522(96)00251-5)
#   성체 흰쥐 CA1 추체세포 989쌍 동시 세포내기록 -> 단일시냅스 흥분성 연결 9개(6쌍 분석가능).
#   post Vm -67~-70mV 에서: 진폭 0.7±0.5 mV (0.17~1.5) · 10-90% 상승 2.7±0.9 ms (1.5~3.8)
#   · 반치폭 16.8±4.1 ms (11.6~25). AMPA+NMDA 매개. 4쌍 전부 짝펄스 억압.
#   완전 재구성된 1쌍: 전시냅스 축삭이 post 3차 기저수상돌기에 접촉 2개(스파인1·shaft1),
#   그 쌍의 EPSP 진폭 1.5mV · 상승 2.8ms · 반치폭 11.6ms.
#   ⚠️ 상승시간은 10-90% 기준이다(우리 measure 는 20-80%) -> 우리 값이 체계적으로 작게 나온다.
DEUCHARS1996 = {
    "amp_mV":       {"min": 0.17, "max": 1.5, "mean": 0.7, "sd": 0.5},
    "rise_ms":      {"mean": 2.7, "sd": 0.9, "min": 1.5, "max": 3.8, "note": "10-90%"},
    "halfwidth_ms": {"mean": 16.8, "sd": 4.1, "min": 11.6, "max": 25.0},
    "n_contacts":   2,          # 완전 재구성 쌍의 접촉 수 (기저수상돌기)
    "n_pairs_tested": 989, "n_connected": 9,
    "src": "Deuchars & Thomson 1996 Neuroscience 74(4):1009 (PMID 8895869)",
}

# 참고(다른 연결): 단일 연결 uEPSP CA3->CA1 = Schaffer collateral
#   ⚠️ 이 벤치의 연결이 아니다(D9). SC 자극 실험과 대조할 때만 참고로 쓴다.
# Sayer, Friedlander & Redman 1990 J Neurosci 10(3):826
#   PubMed 확인(PMID 2319304, DOI 10.1523/JNEUROSCI.10-03-00826.1990). 기니피그 슬라이스.
#   71개 EPSP: 진폭 30~665 uV(평균 131), 상승시간 3.9±1.8 ms, 반치폭 19.5±8.0 ms.
#   양자증분 278 uV. ★ '단일 CA3 세포 활성화'로 유발 = 연결 단위(다접촉 포함).
SAYER1990 = {
    "amp_mV":       {"min": 0.030, "max": 0.665, "mean": 0.131},
    "rise_ms":      {"mean": 3.9, "sd": 1.8},
    "halfwidth_ms": {"mean": 19.5, "sd": 8.0},
    "src": "Sayer/Friedlander/Redman 1990 J Neurosci 10(3):826 (PMID 2319304)",
}

# Ecker et al. 2020 서지 — 원문 확보·대조 완료 (D15, 2026-08-24)
#   Hippocampus 30(11):1129-1145 · DOI 10.1002/hipo.23220
#   99_references/Ecker2020_CA1_synaptic_physiology_in_silico.pdf (gitignore)
ECKER2020 = {
    "cite": "Ecker A, Romani A, Saray S, Kali S, Migliore M, Falck J, Lange S, Mercer A, "
            "Thomson AM, Muller E, Reimann MW, Ramaswamy S (2020) Hippocampus 30(11):1129-1145",
    "doi": "10.1002/hipo.23220",
    # Table 3 "PC to PC (E2)" 원문 값 (평균 ± SD). 상첨자 a = 체감각 피질(Markram 2015)에서 일반화
    "PC_PC": {"g_nS": (0.6, 0.1), "tau_d_AMPA": (3.0, 0.2),
              "Use": (0.5, 0.02, "a"), "Dep_ms": (671, 17, "a"), "Fac_ms": (17, 5, "a"),
              "Nrrp": 2, "NMDA_ratio": 1.22,
              "NMDA_tau_r": 3.9, "NMDA_tau_d": 148.5},
    # 연결당 시냅스 수 (본문 §3.3, Supplementary Table S3) — Fig.3b 는 실험 대조 그림
    "nsyn_per_conn": {"E_E": (1.26, 0.6), "I_E": (8.2, 2.1),
                      "E_I_PC_OLM": (2.8, 1.2), "I_I": (2.8, 0.2)},
    "erev_exc_mV": -8.5, "erev_inh_mV": -73.0,   # p4, Moradi & Ascoli 2020
    "ca_o_mM_calibration": 2.5,                   # p13: 전도도를 2.5mM PSP 진폭에 맞춰 보정
    "note": "PC->PC 는 in vitro [Ca2+]o 2~2.5mM 에서 E2(억압), in vivo 수준 1.1~1.3mM 에서는 "
            "E3(준선형·낮은 진폭·큰 변동·실패)이 된다(p8). 우리 Use=0.50 은 슬라이스 조건 값이다.",
    "table3_has_no_schaffer": True,               # 16경로 전부 CA1 내부 — D9 근거
}

# 참고 대조용 — Ecker 2020 Table 3 의 촉진형 흥분성 클래스 PC->SOM+ (E1).
#   ⚠️ 이 벤치의 연결이 아니다(표적이 개재뉴런). 억압 vs 촉진 대비를 보일 때
#   없는 클래스를 만들지 않고 이 클래스를 쓴다 (D9 원칙).
#   ⚠️ USE/D/F 는 상첨자 a = 체감각 피질 일반화값이고 SD 가 매우 크다(F 670±830).
ECKER_E1_CONTRAST = {
    "name": "PC->SOM+ (E1)",
    "g_nS": 0.8, "tau_d_AMPA": 1.7, "NMDA_ratio": 0.28,
    "Use": 0.09, "Dep_ms": 138.0, "Fac_ms": 670.0, "Nrrp": 1,
    "sd": {"g_nS": 0.05, "tau_d_AMPA": 0.14, "Use": 0.12, "Dep_ms": 211.0, "Fac_ms": 830.0},
    "src": "Ecker 2020 Hippocampus 30(11):1129 Table 3 (PC->O-LM / PC->SOM+), DOI 10.1002/hipo.23220",
}

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
