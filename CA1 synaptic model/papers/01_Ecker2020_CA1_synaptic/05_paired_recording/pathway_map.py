# Source: Ecker(2020) Table 3 일반화 + 개별경로. m-type 쌍 → 9 클래스 매핑(근사).
"""m-type 쌍(전→후) 을 9개 일반화 클래스 중 하나로 매핑한다.

근거: params_table3 의 개별경로(PC->PVBC, PVBC->PC 등) + 카테고리 추정.
카테고리: PV+(주변표적 FS) · CCK+(CCK 발현) · SOM+(SOM 발현 수상돌기표적) · NOS+(Ivy/NGF).
⚠️ m-type→카테고리·클래스 배정은 **근사**(논문 정확 표 미보유). bistratified(BS)는
   PV+(출력)·SOM+(PC 입력 대상) 이중성 → 출력은 PV+, PC→BS 는 SOM+ 로 둠.
"""
MORDER = ["SLM-PPA", "SR-SCA", "SP-AA", "SP-BS", "SP-CCKBC", "SP-Ivy",
          "SP-PC", "SP-PVBC", "SO-BS", "SO-BP", "SO-OLM", "SO-Tri"]

PVPLUS = {"SP-PVBC", "SP-AA", "SP-BS", "SO-BS"}      # 주변표적/FS (PV+)
CCKPLUS = {"SP-CCKBC", "SR-SCA", "SLM-PPA"}          # CCK 발현
SOMPLUS = {"SO-OLM", "SO-Tri", "SO-BP"}              # SOM 발현(수상돌기표적)
NOSPLUS = {"SP-Ivy"}                                  # nNOS/Ivy
PC_TO_SOMPLUS = {"SO-OLM", "SP-BS", "SO-BS"}          # PC→SOM+ (촉진 E1) 대상


def pathway_class(pre, post):
    """(pre_mtype, post_mtype) → 클래스명 또는 None(연결 없음)."""
    if pre == "SP-PC":                                # 흥분성(PC 출력)
        if post == "SP-PC":
            return "PC->PC (E2)"
        if post in PC_TO_SOMPLUS:
            return "PC->SOM+ (E1)"
        return "PC->SOM- (E2)"                        # PC→그 외 인터뉴런
    # pre = 인터뉴런(억제성)
    if post == "SP-PC":
        if pre in PVPLUS:
            return "PV+->PC (I2)"
        if pre in CCKPLUS:
            return "CCK+->PC (I3)"
        if pre in SOMPLUS:
            return "SOM+->PC (I2)"
        if pre in NOSPLUS:
            return "NOS+->PC (I3)"
        return None
    # 인터뉴런 → 인터뉴런 (근사)
    if pre in PVPLUS:
        return "CCK-->CCK- (I2)"
    if pre in CCKPLUS:
        return "CCK+->CCK+ (I1)"
    if pre in SOMPLUS:
        return "SOM+->PC (I2)"
    if pre in NOSPLUS:
        return "NOS+->PC (I3)"
    return None
