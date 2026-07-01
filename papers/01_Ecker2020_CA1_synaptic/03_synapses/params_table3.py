"""
params_table3.py — Ecker et al. (2020) Table 3 시냅스 파라미터 (단일 진실원천)
============================================================================
Source: Ecker, Romani, ... Ramaswamy (2020) "Data-driven integration of
        hippocampal CA1 synaptic physiology in silico", Hippocampus 30:1129-1145.
        Table 3 ("Parameters and generalization to nine classes") + 본문 §2.3, §3.5.

논문은 모든 연결을 9개 일반화 클래스로 묶었다(아래 CLASSES = 굵게 표기된 클래스 대표값).
INDIVIDUAL_PATHWAYS 에는 클래스를 구성하는 개별 경로값도 함께 보관한다.

EMS mod RANGE 변수 매핑:
    U_SE -> Use,  D -> Dep(ms),  F -> Fac(ms),  N_RRP -> Nrrp,
    tau_decay -> tau_d_AMPA / tau_d_GABAA,  NMDA/AMPA -> NMDA_ratio,
    g_hat(nS) -> NetCon weight(µS, /1000),  E_rev -> 구동력 계산용(Eq.11).

주의:
  - g_hat 단위 nS. NEURON NetCon weight 는 µS 이므로 g_nS/1000 사용.
  - tau_rise: 본문에 PC->PC NMDA(τr=3.9, τd=148.5)만 명시. AMPA τr 등 일부는
    Supplementary Table S1(미보유) 값 → mod 기본값/근사 사용하고 NOTE 로 표시.
  - E_rev_eff: 흥분성 -8.5 mV, 억제성 -73 mV (본문 §2.6, Moradi & Ascoli 2020 보정).
"""

# 흥분성 -8.5 mV, 억제성(GABA_A) -73 mV : ĝ 보정(Eq.11) 구동력 df=|E_rev - V_SS| 에 사용
E_REV_EXC = -8.5
E_REV_INH = -73.0

# NMDA/AMPA peak conductance 비 (본문 §2.3)
NMDA_RATIO_PC_PC = 1.22     # PC->PC
NMDA_RATIO_PC_CCK = 0.86    # PC->CCK+ interneuron
NMDA_RATIO_PC_INT = 0.28    # PC->그 외 interneuron

# PC->PC NMDA 동역학 (본문 §2.3, Q10 보정 후)
PC_PC_NMDA_TAU_R = 3.9
PC_PC_NMDA_TAU_D = 148.5


def _exc(g, tau_d, use, dep, fac, nrrp, nmda_ratio, tau_r=0.2):
    """흥분성(AMPA+NMDA) 연결 파라미터 dict."""
    return dict(receptor="AMPANMDA", e_rev=E_REV_EXC,
                g_nS=g, tau_r_AMPA=tau_r, tau_d_AMPA=tau_d,
                Use=use, Dep=dep, Fac=fac, Nrrp=nrrp, NMDA_ratio=nmda_ratio)


def _inh(g, tau_d, use, dep, fac, nrrp, tau_r=0.2):
    """억제성(GABA_A) 연결 파라미터 dict. (논문 가정: 순수 GABA_A)"""
    return dict(receptor="GABAAB", e_rev=E_REV_INH,
                g_nS=g, tau_r_GABAA=tau_r, tau_d_GABAA=tau_d,
                Use=use, Dep=dep, Fac=fac, Nrrp=nrrp, NMDA_ratio=0.0)


# === 9개 일반화 클래스 (Table 3 굵은 대표값) ====================================
# STP profile 라벨: E1=흥분성 촉진, E2=흥분성 억압, I1=억제성 촉진,
#                   I2=억제성 억압, I3=억제성 pseudo-linear
CLASSES = {
    "PC->PC (E2)":      dict(pre="PC",   post="PC",   ei="E", stp="E2",
                             **_exc(0.6, 3.0, 0.50, 671, 17, 2, NMDA_RATIO_PC_PC,
                                    tau_r=0.2)),
    "PC->SOM+ (E1)":    dict(pre="PC",   post="SOM+", ei="E", stp="E1",
                             **_exc(0.8, 1.7, 0.09, 138, 670, 1, NMDA_RATIO_PC_INT)),
    "PC->SOM- (E2)":    dict(pre="PC",   post="SOM-", ei="E", stp="E2",
                             **_exc(2.35, 4.12, 0.23, 410, 10, 1, NMDA_RATIO_PC_INT)),
    "PV+->PC (I2)":     dict(pre="PV+",  post="PC",   ei="I", stp="I2",
                             **_inh(2.0, 11.1, 0.13, 1122, 9.3, 1)),
    "CCK+->PC (I3)":    dict(pre="CCK+", post="PC",   ei="I", stp="I3",
                             **_inh(2.0, 8.8, 0.16, 168, 13, 1)),
    "SOM+->PC (I2)":    dict(pre="SOM+", post="PC",   ei="I", stp="I2",
                             **_inh(1.4, 8.3, 0.30, 1250, 2, 1)),
    "NOS+->PC (I3)":    dict(pre="NOS+", post="PC",   ei="I", stp="I3",
                             **_inh(0.48, 16.0, 0.32, 144, 62, 1)),  # Ivy->PC
    "CCK-->CCK- (I2)":  dict(pre="CCK-", post="CCK-", ei="I", stp="I2",
                             **_inh(4.5, 2.67, 0.26, 930, 1.6, 6)),
    "CCK+->CCK+ (I1)":  dict(pre="CCK+", post="CCK+", ei="I", stp="I1",
                             **_inh(4.5, 4.5, 0.11, 115, 1542, 1)),
}

# === 클래스 구성 개별 경로 (Table 3 비대표 행) =================================
INDIVIDUAL_PATHWAYS = {
    # PC -> SOM+ (E1)
    "PC->OLM":     dict(cls="PC->SOM+ (E1)", **_exc(0.8, 1.7, 0.09, 138, 670, 1, NMDA_RATIO_PC_INT)),
    # PC -> SOM- (E2)
    "PC->PVBC":    dict(cls="PC->SOM- (E2)", **_exc(2.0, 4.12, 0.23, 410, 10, 1, NMDA_RATIO_PC_INT)),
    "PC->CCKBC":   dict(cls="PC->SOM- (E2)", **_exc(3.5, 4.12, 0.23, 410, 10, 1, NMDA_RATIO_PC_CCK)),
    "PC->BS":      dict(cls="PC->SOM- (E2)", **_exc(1.65, 4.12, 0.23, 410, 10, 1, NMDA_RATIO_PC_INT)),
    "PC->Ivy":     dict(cls="PC->SOM- (E2)", **_exc(2.3, 4.12, 0.50, 671, 17, 1, NMDA_RATIO_PC_INT)),
    # PV+ -> PC (I2)
    "PVBC->PC":    dict(cls="PV+->PC (I2)", **_inh(2.15, 5.94, 0.16, 965, 8.6, 6)),
    "AA->PC":      dict(cls="PV+->PC (I2)", **_inh(2.4, 11.2, 0.10, 1278, 10, 1)),
    "BS->PC":      dict(cls="PV+->PC (I2)", **_inh(1.6, 16.1, 0.13, 1122, 9.3, 1)),
    # CCK+ -> PC (I3)
    "CCKBC->PC":   dict(cls="CCK+->PC (I3)", **_inh(1.8, 9.35, 0.16, 153, 12, 1)),
    "SCA->PC":     dict(cls="CCK+->PC (I3)", **_inh(2.15, 8.3, 0.15, 185, 14, 1)),
    # SOM+ -> PC (I2)
    "Tri->PC":     dict(cls="SOM+->PC (I2)", **_inh(1.4, 7.75, 0.30, 1250, 2, 1)),
    # CCK- -> CCK- (I2)
    "PVBC->PVBC":  dict(cls="CCK-->CCK- (I2)", **_inh(4.5, 2.67, 0.26, 930, 1.6, 6)),
    "PVBC->AA":    dict(cls="CCK-->CCK- (I2)", **_inh(4.5, 2.67, 0.24, 1730, 3.5, 1)),
    # CCK+ -> CCK+ (I1)
    "CCKBC->CCKBC": dict(cls="CCK+->CCK+ (I1)", **_inh(4.5, 4.5, 0.11, 115, 1542, 1)),
}


# === 연결당 시냅스 수 (Nsyn/conn) — 해부 통계, EMS 보정 파라미터 아님 ===========
# 논문 단계2(Fig.2): "한 연결 = 여러 시냅스 접촉". 클래스별 평균 접촉 수.
#   값 = (Nsyn, confidence, source). confidence: "documented" | "estimated".
# 출처:
#   - E→E 1.3 / PC→O-LM 2.8 / PC→INT(주변표적) 8.2 : Ecker(2020) Fig.3b·§3.3 [documented]
#   - PV+→PC(주변표적 바스켓) 10–12 : Megías(2001) EM, Bezaire&Soltesz(2013) [documented]
#   - CCK+→PC 3–5 / OLM(SOM+)→PC 2–3 / Ivy(NOS+)→PC 1–2 / I-I 2.8–4 :
#       Bezaire&Soltesz(2013) 수렴-발산 추정 범위, Romani(2024) S14/S16/S17 [estimated]
# ⚠️ 대표 단일값. 실제는 m-type쌍·수상돌기 구획별 분포(CV~0.5, Romani 2024).
NSYN_PER_CONNECTION = {
    "PC->PC (E2)":      (1.3,  "documented", "Ecker2020 Fig.3b (E→E)"),
    "PC->SOM+ (E1)":    (2.8,  "documented", "Ecker2020 Fig.3b (PC→O-LM)"),
    "PC->SOM- (E2)":    (8.2,  "documented", "Ecker2020 Fig.3b (PC→INT 주변표적)"),
    "PV+->PC (I2)":     (11.0, "documented", "Megías2001 EM / Bezaire&Soltesz2013 (10–12)"),
    "CCK+->PC (I3)":    (4.0,  "estimated",  "Bezaire&Soltesz2013 (3–5, 수상돌기표적)"),
    "SOM+->PC (I2)":    (2.5,  "estimated",  "Bezaire&Soltesz2013 (OLM→PC 2–3, 원위)"),
    "NOS+->PC (I3)":    (1.5,  "estimated",  "Bezaire&Soltesz2013 (Ivy→PC 1–2, 희소)"),
    "CCK-->CCK- (I2)":  (3.0,  "estimated",  "Bezaire&Soltesz2013 (I-I 2.8–4)"),
    "CCK+->CCK+ (I1)":  (4.0,  "estimated",  "interneuron 상호연결 (I-I 3–5)"),
}


def nsyn_of(name: str) -> float:
    """클래스의 대표 연결당 시냅스 수(Nsyn/conn)."""
    return NSYN_PER_CONNECTION[name][0]


def get_class(name: str) -> dict:
    return CLASSES[name]


if __name__ == "__main__":
    # 표 출력 + facilitation/depression 정성 분류 (F>D 이면 촉진 우세)
    hdr = f"{'class':18s} {'ei':2s} {'stp':3s} {'g(nS)':6s} {'tau_d':6s} {'Use':5s} {'Dep':6s} {'Fac':6s} {'Nrrp':4s}"
    print(hdr)
    print("-" * len(hdr))
    for name, p in CLASSES.items():
        print(f"{name:18s} {p['ei']:2s} {p['stp']:3s} "
              f"{p['g_nS']:<6.2f} {p.get('tau_d_AMPA', p.get('tau_d_GABAA')):<6.2f} "
              f"{p['Use']:<5.2f} {p['Dep']:<6.0f} {p['Fac']:<6.0f} {p['Nrrp']:<4d}")
    print(f"\n총 {len(CLASSES)}개 일반화 클래스, "
          f"{len(INDIVIDUAL_PATHWAYS)}개 개별 경로 등록.")
