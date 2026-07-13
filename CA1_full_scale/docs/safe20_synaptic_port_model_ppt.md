---
marp: true
title: CA1 safe20 Synaptic Port Model
description: Why the ModelDB synaptic kinetics are lowered into a 20-port NEST-GPU receptor table, and how the table was constructed
paginate: true
---

# CA1 safe20 Synaptic Port Model

ModelDB synapse kinetics를 NEST-GPU 풀스케일 CA1에서 실행하기 위한 39→20 receptor-port lowering 설계

HITL decision deck

---

# 1. Executive Summary

`safe20`은 “20개 시냅스 타입만 쓰겠다”는 생물학적 주장도, 임의의 단순화도 아니다.

정확한 정의:

```text
ModelDB syndata120의 pathway-specific synaptic kinetics
  -> compartment-aware original mechanisms 39개
  -> NEST-GPU user model이 감당 가능한 20개 receptor/conductance port
  -> full-scale 338,740-cell CA1 network에서 실행 가능한 table
```

이 기법을 쓰는 이유는 세 가지다.

1. ModelDB의 pathway별 kinetics를 5-port 교과서 모델보다 훨씬 충실히 보존한다.
2. 5.145B synapse 풀스케일에서 per-synapse kinetics 저장/적분을 피한다.
3. NEST-GPU user model의 20-port 실행 제약 안에서 fail-loud provenance를 유지한다.

---

# 2. 질문의 핵심

질문:

> 왜 인자화된 통합 synapse model로 두지 않고, 20개 receptor port를 정의하나?

답:

상위 모델은 이미 인자화된 통합 모델로 보는 게 맞다.

```text
SynapseMechanism(
  receptor_class,
  E_rev,
  tau_rise,
  tau_decay,
  compartment
)
```

하지만 NEST-GPU 실행면은 synapse마다 `(E_rev, tau_rise, tau_decay)`를 들고 ODE를 푸는 구조가 아니다. 뉴런별 receptor/conductance state vector를 만들고, 각 synapse는 그중 하나의 port index를 가리킨다.

---

# 3. Port의 의미

port는 “projection 종류”가 아니라 **뉴런이 유지하는 receptor kinetics bucket**이다.

개념적으로 한 뉴런은 다음 상태를 가진다.

```text
V, w, ...
g_rise[0],  g_decay[0]   # AMPA_fast e0 tr0.07 td0.2 dend
g_rise[1],  g_decay[1]   # AMPA_fast e0 tr0.3  td0.6 dend
...
g_rise[19], g_decay[19]  # GABA_B em90 tr180 td200 dend
```

synapse event가 오면:

```text
event(weight, receptor=k)
  -> target neuron의 conductance state k에 weight를 누적
  -> 매 timestep port별 conductance ODE를 적분
```

---

# 4. Port 수 제한이 의미하는 것

`20 ports`는 네트워크 전체 연결 수 제한이 아니다.

잘못된 해석:

```text
네트워크에 시냅스 타입이 20개 이하만 있어야 한다
```

정확한 해석:

```text
각 뉴런 모델 인스턴스가 동시에 구분해 적분할 receptor kinetics channel이 20개
```

같은 port를 여러 projection이 공유할 수 있다.

예:

```text
CA3 -> Bistratified   port AMPA_fast tr2 td6.3
CA3 -> CCK_Basket     port AMPA_fast tr2 td6.3
CA3 -> Ivy            port AMPA_fast tr2 td6.3
```

---

# 5. 왜 20개인가

현재 프로젝트의 GPU backend는 `aglif_dend_cond_beta`를 NEST-GPU `user_m2`로 낮춘다.

코드상 user model port limit:

```text
_MAX_USER_MODEL_PORTS = 20
```

build 시:

```text
ngpu.Create(model_name, count, n_ports)
ngpu.SetStatus(nodes, {
  E_rev: [...],
  tau_rise: [...],
  tau_decay: [...],
  compartment: [...]
})
```

즉 20은 임의 상수가 아니라 현재 NEST-GPU user model 실행 표면의 hard budget이다.

근거: `src/ca1/sim/gpu_backend.py`

---

# 6. 왜 per-synapse 인자화를 하지 않나

풀스케일 synapse 수:

```text
total synapses ~= 5.145B
```

만약 synapse마다 kinetics parameter를 저장하면 최소:

```text
5.145B * 3 floats(E_rev, tau_rise, tau_decay) * 4 bytes
~= 61.7 GB
```

여기에 alignment, compartment, event state, indexing, kernel access cost까지 붙는다.

반면 port 방식은:

```text
338,740 cells * 20 ports * conductance states
```

5B 단위가 아니라 수백만 단위 상태로 떨어진다.

---

# 7. 계산량 차이

per-synapse kinetics 방식:

```text
각 synapse가 자기 kinetics 인자를 들고 있음
event마다 global memory에서 개별 parameter load
많은 synapse state 또는 event-level parameter access 필요
```

port 방식:

```text
동일/유사 kinetics event를 neuron-local port conductance에 누적
decay ODE는 port 수만큼만 적분
synapse는 receptor index와 weight/delay만 보유
```

핵심:

```text
ODE를 푸는 것은 맞지만,
5B synapse마다 푸는 것이 아니라
각 뉴런의 20개 conductance channel에 대해 푼다.
```

---

# 8. 원본 데이터: ModelDB syndata

원본은 `src/ca1/params/syndata_120.json`이다.

ModelDB synapse table 성격:

- pathway별 synaptic mechanism
- postsynaptic/presynaptic cell type
- target section filter
- reversal potential
- rise/decay time constants
- 일부 row는 A/B component를 가짐

프로젝트 문서 기준:

- `syndata_120.json`, `syndata_137.json` 각각 97 entries
- `syndata137`은 NGF→Pyramidal GABA_A reversal이 -75mV로 다른 variant
- spontaneous/theta run에서는 `syndata120`이 canonical path로 쓰임

---

# 9. 원본 조합 수

`syndata120`을 compartment-aware로 해석하면:

```text
entries: 97
primary/A component rows: 97
secondary/B component rows: 8
total component rows: 105
```

중복 제거 기준별 unique count:

| 기준 | 개수 |
|---|---:|
| raw kinetics `(E_rev, tau_rise, tau_decay)` | 32 |
| raw kinetics + compartment | 36 |
| receptor class + kinetics | 36 |
| receptor class + kinetics + compartment | 39 |
| safe20 compressed ports | 20 |

따라서 핵심 compression은 `39 original mechanisms -> 20 ports`다.

---

# 10. 39개 원본 조합의 분포

`receptor_class + E_rev + tau_rise + tau_decay + compartment` 기준:

```text
AMPA_fast      7
AMPA_slow      2
GABA_A_fast    6
GABA_A_slow   23
GABA_B         1
total         39
```

정보량의 대부분은 `GABA_A_slow`에 있다.

의미:

- excitatory drive는 상대적으로 적은 kinetics family로 표현 가능
- recurrent inhibition은 pathway/compartment별 다양성이 큼
- 현재 CA1 balance 실패가 inhibitory/source-location 쪽으로 몰리는 것은 이 구조와 일관됨

---

# 11. Pipeline 개요

```text
syndata_120.json
  |
  v
parse entries: pre, post, parameters, section_list
  |
  v
component extraction: A component + optional B component
  |
  v
derive receptor class from presynaptic source
  |
  v
derive compartment from section_list: soma/axon -> soma, else dend
  |
  v
construct original mechanism key
  |
  v
apply replacements + representative port selection
  |
  v
ReceptorConfig(names, E_rev, tau_rise, tau_decay)
  |
  v
NEST-GPU n_ports=20 + synapse receptor index
```

---

# 12. Step 1: receptor class 결정

presynaptic source로 receptor class를 결정한다.

```text
ECIII       -> AMPA_slow
CA3         -> AMPA_fast
Pyramidal   -> AMPA_fast
PV_Basket   -> GABA_A_fast
Axo         -> GABA_A_fast
others      -> GABA_A_slow
NGF B comp  -> GABA_B
```

이 규칙은 synapse row의 receptor field를 맹신하는 것이 아니라, source-derived class와 declared receptor consistency를 검증한다.

이 때문에 CA3/ECIII afferent가 각각 fast/slow AMPA로 분리된다.

---

# 13. Step 2: A/B component 처리

primary component:

```text
e_rev, tau_rise, tau_decay
```

또는:

```text
e_rev_A, tau_rise_A, tau_decay_A
```

secondary component:

```text
e_rev_B, tau_rise_B, tau_decay_B
```

`B` component는 Neurogliaform co-release의 `GABA_B`로 사용된다.

현재 counts:

```text
A/primary = 97
B/GABA_B = 8
total = 105
```

---

# 14. Step 3: compartment-aware split

`section_list`에서 soma/axon 여부를 읽는다.

```text
if "soma" or "axon" in section_list:
  compartment = soma
else:
  compartment = dend
```

이 split이 중요한 이유:

- 같은 kinetics라도 soma/dend input은 AGLIF-dend에서 다른 compartment current로 들어간다.
- CCK 같은 pathway는 같은 post target에도 soma/dend split이 생긴다.
- source-location transfer와 결합될 때 dendritic domain provenance가 필요하다.

---

# 15. Step 4: 원본 mechanism key

각 component는 다음 key로 표현된다.

```text
(
  receptor_class,
  E_rev,
  tau_rise,
  tau_decay,
  compartment
)
```

예:

```text
(AMPA_fast, 0, 0.07, 0.2, dend)
(GABA_A_fast, -60, 0.287, 2.67, soma)
(GABA_A_slow, -60, 0.432, 4.49, dend)
(GABA_B, -90, 180, 200, dend)
```

이 단계에서 unique count가 39개다.

---

# 16. Step 5: representative port set

20-port budget에 맞추기 위해 대표 kinetics set을 정의한다.

대표군 예:

```text
AMPA_fast:    0.07/0.2, 0.3/0.6, 0.5/3, 2/6.3
AMPA_slow:    0.5/3, 2/6.3
GABA_A_fast:  0.287/2.67, 0.28/8.4, 0.3/6.2
GABA_A_slow:  multiple soma/dend buckets
GABA_B:       180/200
```

대표 port는 receptor class와 E_rev를 보존하고, tau space에서 가장 가까운 representative를 선택한다.

거리 기준은 log tau distance:

```text
|log(tau_rise_a)-log(tau_rise_b)| +
|log(tau_decay_a)-log(tau_decay_b)|
```

---

# 17. Step 6: replacements

일부 source kinetics는 명시 replacement를 거친다.

예:

```text
AMPA_fast 2.0/8.0 -> AMPA_fast 2.0/6.3
GABA_A_fast 1.0/8.0 -> GABA_A_fast 0.28/8.4
GABA_A_slow 3.1/42 -> GABA_A_slow 8.0/39.0
GABA_A_slow 9.0/39 -> GABA_A_slow 8.0/39.0
```

compartment-aware replacement도 있다.

```text
AMPA_fast 0.1/1.5 -> AMPA_fast 0.3/0.6
AMPA_fast 0.11/0.25 -> AMPA_fast 0.07/0.2
GABA_A_fast 0.08/4.8 -> GABA_A_fast 0.287/2.67
```

---

# 18. Final 20-port table

Canonical `syndata120-compartment-aware-20port-budget_weighted`:

```text
00 AMPA_fast    e0   tr0.07  td0.2   dend
01 AMPA_fast    e0   tr0.3   td0.6   dend
02 AMPA_fast    e0   tr0.5   td3     dend
03 AMPA_fast    e0   tr2     td6.3   dend
04 AMPA_slow    e0   tr0.5   td3     dend
05 AMPA_slow    e0   tr2     td6.3   dend
06 GABA_A_fast  em60 tr0.287 td2.67  soma
07 GABA_A_fast  em60 tr0.28  td8.4   soma
08 GABA_A_fast  em60 tr0.3   td6.2   soma
09 GABA_A_slow  em60 tr0.11  td9.7   dend
10 GABA_A_slow  em60 tr0.25  td7.5   dend
11 GABA_A_slow  em60 tr0.25  td7.5   soma
12 GABA_A_slow  em60 tr0.287 td2.67  dend
13 GABA_A_slow  em60 tr0.432 td4.49  dend
14 GABA_A_slow  em60 tr0.432 td4.49  soma
15 GABA_A_slow  em60 tr1     td8     dend
16 GABA_A_slow  em60 tr1     td8     soma
17 GABA_A_slow  em60 tr2.9   td3.1   dend
18 GABA_A_slow  em60 tr8     td39    dend
19 GABA_B       em90 tr180   td200   dend
```

---

# 19. 39→20 compression: high-level map

```text
AMPA_fast    7 original mechanisms -> 4 ports
AMPA_slow    2 original mechanisms -> 2 ports
GABA_A_fast  6 original mechanisms -> 3 ports
GABA_A_slow 23 original mechanisms -> 10 ports
GABA_B       1 original mechanism  -> 1 port
```

가장 큰 압축은 `GABA_A_slow`에서 발생한다.

이것은 장점과 리스크를 동시에 만든다.

장점:

- full-scale GPU에서 실행 가능
- source-specific inhibition 다양성을 5-port보다 훨씬 보존

리스크:

- slow inhibition subfamily의 미세한 tau 차이가 병합됨
- phase/rhythm balance에 민감할 수 있음

---

# 20. Port별 병합 예시 1: Excitation

AMPA는 비교적 손실이 작다.

```text
AMPA_fast tr0.07 td0.2 dend
  <- 0.07/0.2
  <- 0.11/0.25

AMPA_fast tr0.3 td0.6 dend
  <- 0.1/1.5
  <- 0.3/0.6

AMPA_fast tr2 td6.3 dend
  <- 2.0/6.3
  <- 2.0/8.0

AMPA_slow tr2 td6.3 dend
  <- ECIII non-Pyramidal targets
```

해석:

- CA3/ECIII afferent distinction은 보존됨
- ultra-fast Pyramidal→PV AMPA는 별도 port로 보존됨

---

# 21. Port별 병합 예시 2: Fast inhibition

`GABA_A_fast`는 PV/Axo 계열이다.

```text
GABA_A_fast tr0.287 td2.67 soma
  <- 0.08/4.8
  <- 0.18/0.45
  <- 0.287/2.67

GABA_A_fast tr0.28 td8.4 soma
  <- 0.28/8.4
  <- 1.0/8.0

GABA_A_fast tr0.3 td6.2 soma
  <- 0.3/6.2
```

주의:

- PV_Basket→Bistratified의 매우 빠른 0.18/0.45 kinetics가 canonical budget_weighted에서는 0.287/2.67로 병합된다.
- 이를 보존하는 alternative strategy가 `preserve_fast_basket_bistratified`다.

---

# 22. Port별 병합 예시 3: Slow inhibition

`GABA_A_slow tr1 td8 dend`는 가장 많이 합쳐지는 bucket 중 하나다.

```text
0.6/15.0
0.728/10.0
0.728/20.2
1.0/8.0
1.1/11.0
1.3/10.2
```

`GABA_A_slow tr8 td39 dend`:

```text
3.1/42.0
8.0/39.0
9.0/39.0
```

해석:

- theta phase, recurrent inhibitory load, NGF/Ivy/SCA balance에 민감할 수 있는 곳
- 현재 네트워크 실패 분석에서 가장 HITL 검토 가치가 큰 영역

---

# 23. Backend lowering

`ReceptorConfig`가 생성되면 GPU backend는 이를 population 상태로 내려보낸다.

```text
receptor_status = {
  E_rev:      list(receptors.E_rev),
  tau_rise:  list(receptors.tau_rise),
  tau_decay: list(receptors.tau_decay),
  compartment: ...
}
```

connection에는 port index만 들어간다.

```text
port_idx = receptors.port_index(proj.receptor)
syn_spec = {
  weight: positive_weight,
  delay: proj.delay_ms,
  receptor: port_idx
}
```

억제는 negative weight가 아니라 negative `E_rev`를 가진 GABA port로 구현된다.

---

# 24. Why positive inhibitory weights

conductance model에서 current는 대략:

```text
I_syn = g_syn * (E_rev - V)
```

따라서 inhibition은:

```text
weight > 0
E_rev < resting / threshold
```

로 표현한다.

negative weight를 쓰면:

- conductance의 물리적 의미가 깨짐
- driving force와 sign을 이중으로 처리할 위험
- CPU/GPU backend consistency가 깨질 수 있음

이 프로젝트는 `GABA_A_fast`, `GABA_A_slow`, `GABA_B` port의 `E_rev`로 억제를 표현한다.

---

# 25. Provenance와 fail-loud gate

canonical receptor provenance:

```text
syndata120-compartment-aware-20port-budget_weighted;
sha256=26774704b306d1bd0461fd7df69491cfacd0e1a2e6385877ece2150c9e05e46c
```

final-tier gate는 다음을 거부한다.

- receptor port provenance missing
- strategy label mismatch
- sha256 mismatch
- noncanonical strategy를 final evidence로 승격

의미:

```text
"budget_weighted라고 주장하지만 table 내용이 바뀐 경우"를 탐지
```

이것은 hidden fallback 방지에 중요하다.

---

# 26. Source-location transfer와의 관계

`safe20`은 kinetics table이고, source-location transfer는 dendritic/somatic effect를 AGLIF-dend 축약 모델에 맞추는 별도 계층이다.

결합 관계:

```text
port name endswith __dend / __soma
  -> AGLIF-dend compartment status
  -> source_location_transfer table
  -> validated dendritic current scaling/domain
```

따라서 final-tier는 다음 둘을 함께 본다.

1. receptor port table이 canonical인가
2. source-location transfer table이 validated/final인가

둘 중 하나라도 diagnostic이면 최종 결론용 evidence가 아니다.

---

# 27. Alternative strategies

현재 strategy:

```text
budget_weighted
```

대안:

```text
preserve_fast_basket_bistratified
```

- PV_Basket→Bistratified fast kinetics를 보존
- 대신 다른 slow port 하나를 희생

```text
demix_pyramidal_olm_gabaa_slow_distal
```

- O_LM→Pyramidal slow GABA_A distal port를 분리
- CCK→O_LM mixed soma/dend issue를 재배치

대안들은 diagnostic 가치가 있지만, 현재 final-tier canonical은 `budget_weighted`다.

---

# 28. 왜 5-port 모델보다 나은가

기본 5-port 모델:

```text
AMPA_fast
AMPA_slow
GABA_A_fast
GABA_A_slow
GABA_B
```

문제:

- CA3/Pyramidal AMPA kinetics 차이 소실
- PV/Axo fast inhibition 다양성 소실
- CCK/O_LM/Ivy/NGF/SCA slow inhibition 차이 소실
- soma/dend routing 소실

safe20:

- class는 유지
- 주요 tau family 보존
- soma/dend split 보존
- GABA_B co-release 보존
- GPU 실행 가능

즉, 5-port보다 훨씬 source-faithful한 최소 실행 표현이다.

---

# 29. 왜 39-port를 그대로 쓰지 않나

이론적으로는 39-port가 더 충실하다.

하지만 현재 제약:

```text
NEST-GPU user_m1/user_m2 max ports = 20
```

39-port를 쓰려면:

1. user model kernel 확장
2. per-neuron state memory 증가
3. SetStatus/record/compartment arrays 확장
4. validation/provenance gate 재설계
5. full-scale memory/performance 재검증

따라서 지금은 20-port가 “표현력 vs 실행 가능성”의 practical frontier다.

---

# 30. 손실이 큰 지점

HITL이 주목해야 하는 병합:

1. `GABA_A_slow tr1 td8 dend`
   - 6개 original mechanisms 병합
   - O_LM/Ivy/Bistratified/CCK/SCA 관련 가능

2. `GABA_A_fast tr0.287 td2.67 soma`
   - very fast PV→Bistratified 0.18/0.45가 병합
   - PV/Bistratified timing에 민감할 수 있음

3. `GABA_A_slow tr8 td39 dend`
   - NGF slow inhibition family 병합
   - theta envelope / slow GABA balance에 영향 가능

4. CCK soma/dend split
   - source-location transfer와 강하게 결합

---

# 31. 현재 CA1 실패와의 연결

최근 full-scale exploration에서 반복된 현상:

- PV silence 또는 과도한 PV recruitment
- Pyramidal/CCK/SCA/Axo balance mismatch
- Ivy/NGF recruitment 문제
- theta phase는 일부 좋아지나 rate/CFC가 막힘
- scalar gain만으로는 통과 실패

이 현상은 단순 cell fitting보다 다음 계층과 더 관련 있다.

```text
receptor kinetics compression
source-location transfer
compartment routing
recurrent inhibitory load
afferent/recurrent source graph
```

따라서 safe20의 병합 지점을 검토하는 것은 합리적인 HITL 축이다.

---

# 32. 검증된 사실

현재 생성된 evidence:

```text
docs/generated/safe20_syndata120_summary.json
docs/generated/safe20_syndata120_mapping.csv
```

검증된 수치:

```text
entries = 97
component_rows = 105
typed_plus_compartment_unique = 39
compressed_ports = 20
canonical provenance sha256 = 267747...
```

설치 확인:

```text
gti installed globally
gti --version = 0.3.1
```

주의:

`gti` live image generation은 호출하지 않았다. private backend/network side effect 없이 자료 생성만 수행했다.

---

# 33. 사용해야 하는 이유

`safe20`을 유지해야 하는 이유:

1. full-scale GPU 실행 가능성
   - 20-port hard budget에 맞음

2. biological fidelity
   - 5-port보다 pathway kinetics를 많이 보존

3. memory/bandwidth efficiency
   - per-synapse kinetics 저장을 피함

4. backend consistency
   - CPU/GPU 모두 ReceptorConfig 기반

5. provenance
   - exact table hash로 hidden fallback 차단

6. investigation leverage
   - 병합 지점을 명시하므로 실패 원인을 구조적으로 추적 가능

---

# 34. 쓰지 말아야 하는 경우

다음 조건이면 `safe20`만으로 결론 내리면 안 된다.

- 특정 slow inhibitory kinetics가 theta phase/rate에 결정적이라는 증거가 나옴
- PV/Bistratified ultra-fast inhibition timing이 핵심 blocker로 좁혀짐
- source-location transfer가 mixed-domain port를 제대로 분리하지 못함
- syndata137의 -75mV NGF→Pyramidal reversal이 필수로 확인됨
- 20-port budget 때문에 주요 source family가 과도하게 병합됨

이 경우 선택지는:

```text
alternative 20-port strategy
-> user model 39-port 확장
-> E-GLIF/user_m custom extension
-> 2-compartment fallback
```

---

# 35. 어떤 절차로 만들었는가

현 구현 절차:

1. ModelDB `syndata_120.json` parser 확보
2. 97 entries에서 primary/secondary components 추출
3. presynaptic source로 receptor class 결정
4. `section_list`로 soma/dend compartment 결정
5. original mechanism key 39개 산출
6. representative 20-port candidate set 정의
7. replacement table로 near-duplicate/known variants 정렬
8. log tau distance로 nearest representative 선택
9. pair-specific receptor mapping 생성
10. `ReceptorConfig` 생성
11. provenance label + SHA256 부여
12. final-tier gate에서 label/hash 검증
13. GPU backend에서 `n_ports=20`으로 lowering

---

# 36. 구현 파일 지도

핵심 구현:

- `src/ca1/params/receptors.py`
  - syndata parsing
  - class/compartment derivation
  - pair receptor mapping

- `src/ca1/params/receptor_ports.py`
  - port strategy
  - representative set
  - replacements
  - provenance hash
  - final receptor-port failures

- `src/ca1/params/synapses.py`
  - Projection/Afferent 생성
  - NGF GABA_B co-release

- `src/ca1/sim/gpu_backend.py`
  - NEST-GPU `n_ports`
  - receptor arrays
  - synapse `receptor` index

---

# 37. 테스트/게이트 지도

주요 테스트:

- `tests/test_afferent_receptor_mapping.py`
  - CA3/ECIII AMPA split
  - NGF GABA_B scaling
  - pair kinetics preservation
  - syndata137 -75mV preservation
  - n_ports <= 20
  - compartment-aware split

- `tests/test_receptor_port_provenance.py`
  - canonical strategy final gate pass
  - table hash mutation detection
  - missing hash rejection
  - hash mismatch rejection

- `tests/test_gpu_backend_user_model_ports.py`
  - >20 ports reject
  - compartment status mapping

---

# 38. HITL decision points

결정 1: 현재 final canonical을 유지할 것인가?

```text
syndata120 + compartment-aware + budget_weighted + safe20
```

권고: 유지. 다만 실패 원인 분석에서 병합 민감도를 따로 검증.

결정 2: alternative strategy를 final 후보로 승격할 것인가?

권고: 지금은 diagnostic. 승격하려면 label/hash/gate를 새 canonical로 바꾸고 full validation 필요.

결정 3: 39-port 확장을 할 것인가?

권고: 바로 가지 말 것. 먼저 20-port 병합 민감도와 source-location transfer blocker를 확인.

---

# 39. 다음 검증 제안

GPU를 쓰기 전 가능한 정적/CPU 검증:

1. `39 original -> 20 port` 병합 손실 ranking
   - tau distance
   - affected pathway count
   - affected total synapse budget

2. source-location-sensitive port audit
   - mixed-domain ports
   - soma/dend shared kinetics
   - CCK/O_LM/Pyramidal paths

3. alternative 20-port strategy 비교표
   - budget_weighted
   - preserve_fast_basket_bistratified
   - demix_pyramidal_olm_gabaa_slow_distal

4. final-tier gate impact
   - which alternatives are diagnostic-only
   - what provenance change is required

---

# 40. 결론

`safe20`은 “임의로 20개를 고른 시냅스 타입 목록”이 아니다.

정확한 역할:

```text
ModelDB의 39개 compartment-aware synaptic mechanisms를
NEST-GPU full-scale 실행 가능한 20개 neuron-local conductance ODE bucket으로
컴파일한 backend lowering table
```

이 기법을 써야 하는 이유:

- 5-port보다 논문/ModelDB kinetics fidelity가 높음
- 39/per-synapse 모델보다 GPU 풀스케일 실행 가능성이 높음
- provenance hash로 hidden fallback을 막음
- 현재 실패 분석의 구조적 축을 명확히 드러냄

HITL 관점의 핵심 판단:

```text
safe20은 현재 baseline으로 유지하되,
slow inhibition / PV-Bistratified / source-location 병합 민감도를 다음 검증축으로 삼는다.
```

---

# Appendix A. Generated artifacts

이번 자료 생성을 위해 생성한 로컬 산출물:

```text
docs/generated/safe20_syndata120_summary.json
docs/generated/safe20_syndata120_mapping.csv
```

`god-tibo-imagen` 설치 확인:

```text
gti path: /home/seonghwankim/.nvm/versions/node/v26.3.0/bin/gti
gti version: 0.3.1
```

live image generation은 호출하지 않음.

---

# Appendix B. Source references

Project files:

- `src/ca1/params/receptors.py`
- `src/ca1/params/receptor_ports.py`
- `src/ca1/params/synapses.py`
- `src/ca1/sim/gpu_backend.py`
- `src/ca1/config.py`
- `src/ca1/validation/network_provenance.py`
- `docs/modeldb_dataset_notes.md`
- `docs/architecture.md`
- `tests/test_afferent_receptor_mapping.py`
- `tests/test_receptor_port_provenance.py`
- `tests/test_gpu_backend_user_model_ports.py`

