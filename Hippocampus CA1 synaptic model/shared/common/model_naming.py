# Source: 프로젝트 공용 — 모델 폴더명 파서
"""모델 폴더 이름에서 e-type·m-type·형태ID 를 추출한다.

명명 규칙(권장): CA1_{pyr|int}_[{mtype}_]{etype}_{morph}_{timestamp}_model_files
  예) CA1_int_cAC_990611HP2_..._model_files              (m-type 없음 — 구형)
      CA1_int_SR-SCA_cAC_990611HP2_..._model_files       (m-type 포함 — 신형)

e-type 를 알려진 목록으로 찾으므로 **m-type 유무·위치와 무관**하게 동작한다
(폴더를 하나씩 바꾸는 도중에도 안전). 폴더 basename 을 넘길 것.
주의: m-type 토큰엔 밑줄(_) 금지 — 하이픈 사용(예: SR-SCA, SP-PVBC, SO-OLM).
"""
ETYPES = ("cACpyr", "cNAC", "cAC", "bAC")     # exact 토큰 매치(부분문자 아님)
_CATEGORIES = ("int", "pyr")                   # m-type 자리에 오면 m-type 아님


def parse(name):
    """폴더 basename → {'etype','mtype','morph'} (못 찾으면 None)."""
    toks = name.split("_")
    for i, t in enumerate(toks):
        if t in ETYPES:
            prev = toks[i - 1] if i >= 1 else None
            mtype = prev if (prev is not None and prev not in _CATEGORIES) else None
            morph = toks[i + 1] if i + 1 < len(toks) else None
            return {"etype": t, "mtype": mtype, "morph": morph}
    return {"etype": None, "mtype": None, "morph": None}


def etype_of(name):
    return parse(name)["etype"]


def morph_of(name):
    return parse(name)["morph"]


def mtype_of(name):
    return parse(name)["mtype"]
