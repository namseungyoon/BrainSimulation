# -*- coding: utf-8 -*-
"""lib/engines.py — 가소성 엔진 레지스트리 + 어댑터 계약 (번호 없음 = import 전용)

★ 왜 필요한가.
   엔진 6종은 **상태 구조가 다르다** — GB 계열은 효능 하나(rho), 고전 STDP 는 가중치와
   두 흔적, GluSynapse 는 전·후시냅스 두 축이다. 문자열 필드 하나로는 표현되지 않는다.
   그래서 각 엔진이 **자기 능력을 선언**하고, 배선·검증 코드는 **이름이 아니라 선언을 읽고**
   분기한다. 엔진을 추가할 때 검증 코드를 고치지 않아도 되는 것이 이 모듈의 설계 목표다.

필수 계약 3가지 (docs/ENGINE_SPEC.md · PLAN)
  1. 모든 엔진의 efficacy() 는 **같은 정규화 척도**([0,1])를 반환한다.
  2. 후시냅스 스파이크 전달 방식은 **엔진이 선언**한다(post_nc). 배선 코드가 이름으로
     분기하면 안 된다 — GluSynapse 처럼 국소 전압에서 칼슘을 만드는 엔진에 sentinel
     NetCon 을 붙이면 이중계산 또는 무증상 오작동이 된다.
  3. **동결은 행동으로 검증한다**(5-11). 선언만으로는 부족하다 — D21 에서 gamma=0 만으로는
     자율항이 살아 rho0 가 안정 고정점이 아니면 표류한다는 것을 실측했다.

능력 필드
  stp      단기가소성 보유 (5-9 대상을 자동 결정)
  ltp      장기가소성(효능 상태) 보유
  prob     확률 방출 (RNG 시딩 필수 — 안 하면 무증상 오작동, 5-5 실측)
  post_nc  후시냅스 스파이크를 weight<0 sentinel NetCon 으로 받아야 하는가
  gmax_via 전도도 단위 규약 "param"(syn.gmax, uS) / "weight"(NetCon weight, nS) — D22(1)
  states   기록 가능한 상태변수
  freeze   가소성을 끄는 파라미터
  freeze_rho0        동결이 **실제로 불변**인 rho0 (자율항의 고정점 전부)
  freeze_rho0_robust 그 중 **안정**한 것만 — 대조군으로 써도 되는 값.
                     rho*=0.5 는 고정점이지만 **불안정**하다(칼날 균형). 결정론 실행에서는
                     움직이지 않아 계약 검사는 통과하지만 조금만 흔들려도 어느 우물로든
                     떨어지므로 **대조군으로 쓰면 안 된다**. 둘을 구분해 선언한다(D21·5-11).
  ref      대조할 numpy 참조 모듈 이름 (lib.refs.*)
  own      04 가 직접 작성한 mod 인가
"""

# ---------------------------------------------------------------------------
# 레지스트리. 키는 짧은 별칭, mech 는 NEURON POINT_PROCESS 이름.
# ---------------------------------------------------------------------------
ENGINES = {
    "det": dict(
        mech="DetAMPANMDA", label="det (전달만)", own=False,
        stp=True, ltp=False, prob=False, post_nc=False, gmax_via="weight",
        states=("g", "i"),
        stp_keys=("Use", "Dep", "Fac"),
        freeze={}, freeze_rho0=None, freeze_rho0_robust=None,
        ref="tm",
        note="장기가소성 상태가 아예 없다 — 구조적 기준선(5-2).",
    ),
    "A": dict(
        mech="GBPlasticitySyn", label="A (순수 GB)", own=False,
        stp=False, ltp=True, prob=False, post_nc=True, gmax_via="param",
        states=("g", "i", "c", "rho", "w"),
        stp_keys=(),
        freeze={"gamma_p": 0.0, "gamma_d": 0.0},
        freeze_rho0=(0.0, 0.5, 1.0), freeze_rho0_robust=(0.0, 1.0),
        ref="gb",
        note="Graupner & Brunel 2012. 버스트의 네 펄스를 똑같이 취급한다.",
    ),
    "B": dict(
        mech="GBPlasticityStpSyn", label="B (GB+TM)", own=False,
        stp=True, ltp=True, prob=False, post_nc=True, gmax_via="param",
        states=("g", "i", "c", "rho", "w", "pr_last", "ca_last"),
        stp_keys=("Use", "Dep", "Fac"),
        freeze={"gamma_p": 0.0, "gamma_d": 0.0},
        freeze_rho0=(0.0, 0.5, 1.0), freeze_rho0_robust=(0.0, 1.0),
        ref="gb",
        conventions={"norm_Pr": 1, "ca_stp": 0},     # D24: 기준은 논문 원본 칼슘
        note="관례 ca_stp 가 결론을 뒤집는다(D24). 기준은 ca_stp=0.",
    ),
    "C": dict(
        mech="GBPlasticityStpProbSyn", label="C (B+확률방출)", own=False,
        stp=True, ltp=True, prob=True, post_nc=True, gmax_via="param",
        states=("g", "i", "c", "rho", "w", "pr_last", "ca_last",
                "ves_last", "n_pre", "n_rel"),
        stp_keys=("Use", "Dep", "Fac", "Nrrp"),
        freeze={"gamma_p": 0.0, "gamma_d": 0.0},
        freeze_rho0=(0.0, 0.5, 1.0), freeze_rho0_robust=(0.0, 1.0),
        ref="gb",
        conventions={"norm_Pr": 1, "ca_stp": 0},
        note="ca_stp=0 이면 확률성이 가소성에 전혀 영향을 주지 않는다(GAPS G5).",
    ),
    "stdp": dict(
        mech="PairSTDPSyn", label="고전 STDP", own=True,
        stp=False, ltp=True, prob=False, post_nc=True, gmax_via="param",
        states=("g", "i", "w", "rho", "x_pre", "x_post", "dw_last",
                "n_pre", "n_post"),
        stp_keys=(),
        freeze={"A_p": 0.0, "A_d": 0.0},
        freeze_rho0=None, freeze_rho0_robust=None,   # 자율항이 없다
        ref="stdp",
        conventions={"all_to_all": 1},
        note="칼슘 상태가 없다 — 스파이크 짝만 본다. GB 계열의 대조군(5-6).",
    ),
    # 5-7 GluSynapseCa 는 미결#4(외부 소스 라이선스) 결정 후 등록한다.
    #   등록 시 post_nc=False (국소 전압에서 칼슘을 만들므로 sentinel 을 붙이면 이중계산),
    #   ref="glusyn" 이 될 예정이다. 여기 주석으로 남겨 계약을 미리 못박는다.
}

ORDER = ["det", "A", "B", "C", "stdp"]          # 그림·표의 고정 순서

# lib/synprobe.py 가 쓰는 형태(mech 이름 -> 능력)로 투영한다. 단일 출처 유지.
CAPS = {e["mech"]: dict(stp=e["stp"], ltp=e["ltp"], prob=e["prob"],
                        post_nc=e["post_nc"], gmax_via=e["gmax_via"],
                        states=e["states"])
        for e in ENGINES.values()}


def get(key):
    if key not in ENGINES:
        raise KeyError(f"알 수 없는 엔진 별칭 {key!r} — 등록된 것: {list(ENGINES)}")
    return ENGINES[key]


def mech(key):
    return get(key)["mech"]


def caps(key):
    e = get(key)
    return dict(stp=e["stp"], ltp=e["ltp"], prob=e["prob"], post_nc=e["post_nc"],
                gmax_via=e["gmax_via"])


def with_cap(name, value=True):
    """능력으로 엔진을 고른다. 5-9 가 stp=True 엔진을 자동으로 찾는 데 쓴다."""
    return [k for k in ORDER if ENGINES[k].get(name) == value]


def efficacy(key, syn):
    """비교용 스칼라 효능 — **모든 엔진이 같은 [0,1] 척도**를 준다 (계약 1).

    장기가소성 엔진은 rho 를 그대로 쓴다. GB 는 정의상 [0,1] 이고, PairSTDPSyn 은
    가중치 범위를 GB 축에 맞춰 rho = (w-w_min)/(w_max-w_min) 로 노출한다(5-6).
    장기가소성이 없는 엔진은 0.5(중립)를 준다 — '변하지 않는다' 를 수치로 말하려고.
    """
    e = get(key)
    if not e["ltp"]:
        return 0.5
    r = float(syn.rho)
    if not (-1e-9 <= r <= 1 + 1e-9):
        raise ValueError(f"{e['mech']}.rho={r} 가 [0,1] 밖이다 — 계약 1 위반")
    return r


def apply_params(syn, key, P, rho0=None, frozen=False, conventions=True):
    """config/synapse.yaml 값(P)을 엔진 선언에 따라 적용한다.

    이름으로 분기하지 않는다 — 선언(stp_keys · ltp · conventions · freeze)만 읽는다.
    """
    e = get(key)
    # 전달(수용체) 파라미터는 모든 엔진 공통
    for k_cfg, k_syn in (("e_rev_mV", "e"), ("tau_r_AMPA", "tau_r_AMPA"),
                         ("tau_d_AMPA", "tau_d_AMPA"), ("tau_r_NMDA", "tau_r_NMDA"),
                         ("tau_d_NMDA", "tau_d_NMDA"), ("NMDA_ratio", "NMDA_ratio"),
                         ("mg_mM", "mg")):
        if hasattr(syn, k_syn):
            setattr(syn, k_syn, P[k_cfg])
    # 단기가소성
    for k in e["stp_keys"]:
        src = {"Use": "Use", "Dep": "Dep_ms", "Fac": "Fac_ms", "Nrrp": "Nrrp"}[k]
        setattr(syn, k, P[src])
    # 관례 (D24 기준값)
    if conventions:
        for k, v in (e.get("conventions") or {}).items():
            if hasattr(syn, k):
                setattr(syn, k, v)
    # 효능 초기값
    if e["ltp"] and rho0 is not None:
        syn.rho0 = rho0
    # 동결
    if frozen:
        for k, v in e["freeze"].items():
            setattr(syn, k, v)
    return syn


def freeze_ok(key, rho0):
    """이 rho0 에서 동결이 **실제로 불변**인가 (자율항의 고정점인가). D21·5-11."""
    e = get(key)
    lim = e["freeze_rho0"]
    if lim is None:
        return True
    return any(abs(rho0 - v) < 1e-12 for v in lim)


def freeze_robust(key, rho0):
    """대조군으로 **써도 되는** rho0 인가 — 고정점이면서 **안정**해야 한다.

    rho*=0.5 는 고정점이라 결정론 실행에서는 움직이지 않지만 불안정하다(칼날 균형).
    대조군은 견고해야 하므로 여기서는 False 를 준다.
    """
    e = get(key)
    lim = e["freeze_rho0_robust"]
    if lim is None:
        return True
    return any(abs(rho0 - v) < 1e-12 for v in lim)


def table():
    """능력표 (그림·문서용). 행 순서는 ORDER."""
    rows = []
    for k in ORDER:
        e = ENGINES[k]
        rows.append(dict(key=k, mech=e["mech"], label=e["label"], own=e["own"],
                         stp=e["stp"], ltp=e["ltp"], prob=e["prob"],
                         post_nc=e["post_nc"], gmax_via=e["gmax_via"],
                         ref=e["ref"],
                         conventions=e.get("conventions") or {},
                         freeze=e["freeze"], freeze_rho0=e["freeze_rho0"],
                         freeze_rho0_robust=e["freeze_rho0_robust"],
                         note=e["note"]))
    return rows
