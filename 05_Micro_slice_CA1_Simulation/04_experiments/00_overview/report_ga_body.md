# 가. 시뮬레이션 반응과 생리학적 데이터 비교 검증

- MEA(Multi-Electrode Array, 다전극 어레이) 전극이 놓이는 국소 조직 규모에서 시냅스 가소성(LTP 장기강화·LTD 장기억압)을 검증하고, 나아가 새로운 가소성 모델을 개발·검증할 기반을 마련함
- 해마 CA1(Cornu Ammonis 1) 마이크로 슬라이스(MEA 전극 3개가 들어가는 최소 조직, 종축 500 × 측관통 800 × 두께 400 µm)를 Romani et al. 2024 아틀라스·커넥텀 파이프라인으로 in silico(컴퓨터 시뮬레이션) 재구성하고, 창 내부 모든 뉴런을 대표세포 축소 없이 전세포 완전형태 NEURON 모델로 인스턴트화함
- 마이크로 슬라이스 채택 근거: 전체 CA1(456,378세포)은 계산자원상 불가능하나, 실측 MEA와 직접 대응할 최소 단위가 필요함
- 검증 설계: 규모별(뉴런 쌍 → 마이크로 슬라이스 네트워크 → 실측 MEA)로 반응을 문헌·MEA 실측값과 대조하고, 각 규모에서 단기·장기 가소성을 차례로 적용하여 선행 연구를 재현하는지 확인함. 이를 통해 새 가소성 모델을 얹어 검증할 프로토콜 파이프라인을 확립함
  - 단기가소성(STP, Short-Term Plasticity; Ecker 2020) = 슬라이스·시냅스·회로가 옳음을 확인하는 검증된 기반(모델 무관) — 쌍: paired-pulse·burst / 네트워크: I-O·PPF·burst·fEPSP
  - 장기가소성(LTP/LTD; Graupner-Brunel → 신규 모델) = 기반 위에 얹는 교체 가능한 모델 — 쌍: STDP / 네트워크: 주파수의존 LTD/LTP (HFS·TBS·LFS) / 실측: MEA 대조
- 단기가소성으로 기반을 검증해 두면 어떤 장기가소성 모델이든 동일 프로토콜로 검증 가능함. 본 과제의 신규 모델은 장기가소성 자리에 교체·재검증되며, 실측 MEA 대조가 최종 목표임

| 규모 | 단기가소성 (Ecker) | 장기가소성 (Graupner-Brunel → 신규) |
|---|---|---|
| 뉴런 쌍 | paired-pulse · burst | STDP (타이밍 Δt · 버스트 수 · 반복 빈도, 각각 ca_stp 0·1 병렬) |
| 마이크로 슬라이스 네트워크 | I-O · PPF · burst · fEPSP | 주파수의존 LTD/LTP (HFS·TBS·LFS·주파수-응답) |
| 실측 MEA | — (해당 실험 없음) | MEA 실측 대조 (최종 목표) |

> 약어: I-O(Input-Output, 입출력), PPF(Paired-Pulse Facilitation), fEPSP(field Excitatory Postsynaptic Potential, 필드 흥분성 시냅스후전위), STDP(Spike-Timing-Dependent Plasticity, 스파이크 타이밍 의존 가소성), LFS(Low-Frequency Stimulation, 저빈도 자극), HFS(High-Frequency Stimulation, 고빈도 자극), TBS(Theta-Burst Stimulation, theta-burst 자극), BCM(Bienenstock-Cooper-Munro, 주파수-응답 가소성 이론). Δt = 전·후 스파이크 시간차(sweep = 여러 Δt를 훑어 STDP 곡선 산출).

### 완료한 실험과 문헌 검증 (요약)

현재까지 시뮬레이션으로 완료한 것은 뉴런 쌍 규모의 두 가지 단기가소성 실험, 곧 paired-pulse와 burst이며, 두 실험을 서로 다른 방식으로 문헌과 대조하여 검증하였다.

- paired-pulse 실험은 독립 문헌 검증에 해당한다. 시냅스 클래스별로 측정한 PPR, 즉 두 번째 응답 컨덕턴스를 첫 번째 응답 컨덕턴스로 나눈 비를 개별 쌍기록 문헌이 보고한 값과 대조하였다. 억압성 연결인 추체세포에서 추체세포로 가는 시냅스는 Deuchars와 Thomson 1996, PVBC에서 추체세포는 Kraushaar와 Jonas, CCKBC에서 추체세포는 Hefft와 Jonas의 값과 방향이 일치하였고, 촉진성인 추체세포에서 OLM으로 가는 시냅스도 촉진으로 일치하였다. 대표 여섯 쌍 가운데 네 쌍이 정량적으로 일치하였으며 클래스 사이의 차이도 통계적으로 유의하였다. 나머지 두 쌍인 OLM에서 추체세포, Ivy에서 추체세포는 단일 쌍의 확률적 방출과 파라미터 미세조정 부족에서 비롯된 것으로, OLM에서 추체세포의 경우 방출확률을 재조정하면 Maccaferri 등 2000이 보고한 무변조 거동을 재현하였다.
- burst 실험은 일관성 검증에 해당한다. 여덟 개의 펄스로 이루어진 자극열을 여러 주파수로 인가하고 펄스별 응답의 촉진과 억압 궤적을 Ecker 등 2020이 특성화한 클래스별 단기가소성 동역학과 대조하였다. 단기가소성 파라미터 자체가 Ecker의 피팅값이므로 이 실험은 독립 검증이 아니라, 본 시뮬레이터의 구현이 Ecker의 클래스 동역학을 제대로 재현하는지 확인하는 자기일관성 검증이다.

## 가) 구축

- 시뮬레이션은 세 층으로 구성함. 세그먼트와 막 수준의 **뉴런 모델**, 그 위에 얹는 **시냅스 가소성 모델**(단기와 장기), 그리고 이들을 조립하는 **규모**(뉴런 쌍과 네트워크)임. 뉴런 모델과 시냅스 모델은 규모와 무관하게 동일하므로 1)과 2)에서 한 번만 서술하고, 3)에서 두 규모의 조립과 자극과 기록 방식을 나누어 서술함

### 1) 뉴런 모델 (세포·세그먼트 수준)

- 각 뉴런은 실제 형태를 그대로 가진 전세포 완전형태 모델임. 형태는 morphology_library의 재구성 SWC(세포체·수상돌기·축삭의 3차원 좌표)를 그대로 사용하고, 축삭은 상세 가지 대신 60 µm stub(AIS, Axon Initial Segment, 축삭시작분절)으로 대체함. 각 형태는 여러 구획(section)으로 나뉘고 구획마다 다시 여러 세그먼트로 분할되어, 세그먼트 사이의 전위 전파를 케이블 방정식(cable equation)으로 계산함(막온도 34도)
- 세포의 전기생리 거동은 emodel(전기형 단일세포 모델)로 규정됨. 형태형(m-type)과 전기형(e-type) 조합마다 지정된 hoc 템플릿(memodel_map)이 각 구획에 이온채널을 고유 밀도로 분포시켜 소마·수상돌기·축삭에 걸친 능동·수동 막전류를 재현함. 채택한 채널 집합은 Romani et al. 2024 CA1 파이프라인의 Hodgkin-Huxley형 메커니즘으로, 나트륨(na3·nax), 칼륨(지연정류 kdr·kdrb, A형 kad·kap, M형 kmb, 칼슘의존 kca·cagk), 칼슘(L형 cal·N형 can·T형 cat 및 칼슘축적 cacum), 과분극활성 Ih(hd, HCN 계열), 수동 누설(passive)로 구성됨. 이 세그먼트 수준의 막전류가 이후 모든 관측량 곧 시냅스후전위·활동전위·필드전위의 물리적 근원임
- 지배식: 각 구획의 막전위는 구획형 케이블 방정식으로 동시에 풀리며, 구획 j는 그 위의 채널 전류 합과 이웃 구획과의 축방향 전류로 갱신됨

$$C_m\,\frac{dV_j}{dt} = -\sum_k \bar g_k\, m^p h^q (V_j - E_k) + \sum_{i \sim j} \frac{V_i - V_j}{r_{ij}}$$

[그림 가-1. 뉴런 모델 — 실제 SP_PC 추체세포 재구성 형태 위에 구획별 이온채널 표시 · 층(SLM/SR/SP/SO) · 축삭 60 µm stub(AIS)]  (`04_experiments/00_overview/figures/neuron_model_schematic.png`)

구획별 이온채널 분포(emodel):

| 구획 | 이온채널 메커니즘 |
|---|---|
| 소마·축삭 | Na (na3, nax) · Kdr (kdr, kdrb) · Km (kmb) |
| 수상돌기 | Ca (cal, can, cat) · Ih (hd, HCN) · A형 K (kad, kap) |
| 공통(전 구획) | 칼슘의존 K (kca, cagk) · 칼슘축적 (cacum) · 누설 (passive) |

### 2) 시냅스 모델 (규모 무관 공통)

- 시냅스는 흥분성과 억제성 두 메커니즘으로 구현하며, 아래 단기·장기 두 가소성 모델은 규모와 무관하게 동일한 시냅스 규칙임(쌍에서 검증한 규칙을 네트워크에 그대로 배치). 흥분성 시냅스(SC 구심과 내부 재귀 연결)는 **GBPlasticityStpProbSyn** 메커니즘으로 확률적 방출(다소포성 MVR, Multi-Vesicular Release)·단기가소성(TM)·장기가소성(Graupner-Brunel 칼슘)을 한 시냅스에 통합하고, 억제성 시냅스는 **ProbGABAAB_EMS** 메커니즘으로 구현함

#### ① 단기가소성 모델 (Ecker et al. 2020, Hippocampus)

- 단기가소성은 **Tsodyks-Markram(TM) 현상학 모델**로 기술됨. 시냅스를 한정된 방출자원으로 보고, 스파이크가 올 때 방출 세기를 이용률 u와 가용자원 R(Resources)의 곱(u·R)으로 정함. 한 번의 스파이크는 상반된 두 변화를 동시에 남김 — 이용률 u가 올라 다음 방출을 키우는 **촉진**(facilitation), 자원 R이 줄어 다음 방출을 깎는 **억압**(depression). 자극이 뜸해지면 u는 시상수 F로 기저 방출확률 U(Utilization)까지, R은 시상수 D로 완전회복까지 각각 이완함. NRRP(방출가능 소포부위 수, Number of Readily Releasable Pool)는 한 접합부에서 여러 소포가 확률적으로 방출되는 다소포성(multi-vesicular)을 반영함. 시냅스마다 다른 U·D·F·NRRP 네 값이 촉진형·억압형 거동을 가르며, 이 값들은 Ecker et al. 2020(Hippocampus)이 CA1 쌍기록으로 pathway별 보정한 실측치를 그대로 채택함
- 거동은 두 힘(감도↑ 촉진 vs 자원↓ 억압)의 우열로 갈림: PC→OLM(U 0.09·F 670)은 첫 반응이 약해도 감도가 오래 유지돼 연속자극에 **커지는 촉진형**, PC→PC(U 0.65)는 첫 스파이크에 자원을 크게 소진해 **작아지는 억압형**임
- 지배식: 스파이크마다 방출은 이용률 u와 자원 R의 곱 u·R이며, 스파이크 순간 u는 촉진으로(U만큼 증가) R은 억압으로(u·R만큼 감소) 갱신되고, 스파이크 사이에는 각각 시상수 F·D로 이완함

$$\text{방출} \propto u\,R,\qquad u \leftarrow u + U(1-u),\quad R \leftarrow R - u\,R \quad(\text{스파이크 순간})$$

$$\frac{du}{dt} = -\frac{u}{F},\qquad \frac{dR}{dt} = \frac{1-R}{D}\quad(\text{스파이크 사이})$$

[그림 가-2. Tsodyks-Markram 단기가소성 모델 — 방출확률 U·촉진 F·억압 D에 따른 촉진형·억압형 시냅스의 연속자극 반응 (Ecker et al. 2020 도해 인용)]  (논문 그림 삽입)

#### ② 장기가소성 모델 (Graupner-Brunel 2012)

- 장기가소성은 **칼슘 기반 양안정(calcium-based bistable) 모델**로 기술됨. 시냅스 효능 ρ(rho, synaptic efficacy)를 두 안정상태(DOWN ρ=0·UP ρ=1)를 오가는 변수로 보고, 시냅스후 칼슘 농도 궤적 c(t)가 강화·억압의 부호를 결정함 — 전스파이크는 NMDA 성분으로, 후스파이크는 역전파 활동전위(bAP, back-propagating Action Potential) 성분으로 각각 칼슘을 유입시킴. c(t)가 강화문턱 θp(potentiation threshold)를 넘는 구간에서 ρ가 UP으로(LTP), 억압문턱 θd(depression threshold)만 넘고 θp에 못 미치는 구간에서 ρ가 DOWN으로(LTD) 밀리며, 최종 가중치는 w = w₀ + ρ(w₁ − w₀)로 사상됨(ρ의 잡음 항이 자극 없는 상태의 장기 안정성·자발전이를 표현). 하나의 칼슘 기전으로 STDP 곡선(스파이크 타이밍 Δt 의존)과 자극빈도 의존성(LFS→LTD·HFS→LTP)을 동시에 재현하며(Graupner & Brunel 2012, PNAS), 본 과제가 개발하는 **새로운 가소성 모델**이 바로 이 자리에 교체·재검증되는 표준 슬롯임. 구현은 위 GBPlasticityStpProbSyn 메커니즘의 장기가소성 층(전·후 스파이크 검출기 + 칼슘 적분기 결합)
- 지배식: 시냅스후 칼슘 c(t)는 전스파이크(지연 D 뒤)와 후스파이크가 각각 C_pre·C_post만큼 올리고 시상수 τ_Ca로 감쇠함. 효능 ρ는 두 안정점(0·1) 사이에서, c가 강화문턱 θp를 넘으면 UP(속도 γp), 억압문턱 θd만 넘으면 DOWN(속도 γd)으로 움직이며(Θ는 계단함수, ρ*=0.5는 불안정점), 최종 가중치 w로 사상됨

$$c(t) = C_{pre}\!\sum_{t_{pre}} e^{-(t-t_{pre}-D)/\tau_{Ca}} + C_{post}\!\sum_{t_{post}} e^{-(t-t_{post})/\tau_{Ca}}$$

$$\tau\,\frac{d\rho}{dt} = -\rho(1-\rho)(\rho^{*}-\rho) + \gamma_p(1-\rho)\,\Theta[c-\theta_p] - \gamma_d\,\rho\,\Theta[c-\theta_d] + \text{Noise}$$

$$w = w_0 + \rho\,(b\,w_0 - w_0),\qquad b = w_1/w_0$$

[그림 가-3. Graupner-Brunel 칼슘 기반 양안정 모델 개념도 — 칼슘 궤적 c(t)와 문턱 θd·θp, 시냅스 효능 ρ의 UP/DOWN 전이 (Graupner & Brunel 2012, PNAS 도해 인용)]  (논문 그림 삽입)

### 3) 시뮬레이션 규모별 구성

- 위 뉴런 모델과 시냅스 모델을 두 규모로 조립함. 뉴런 쌍은 시냅스 규칙을 통제된 조건에서 검증하는 최소 단위이고, 마이크로 슬라이스 네트워크는 그 규칙이 조직 규모 반응(fEPSP)으로 합산되는지 보는 단위임. 두 규모의 차이는 세포 수와 자극·기록 방식일 뿐, 뉴런 모델과 시냅스 모델은 동일함

#### ① 뉴런 쌍

- 검증 대상 연결의 전시냅스(pre)·후시냅스(post) 두 뉴런을 실제 커넥텀 그대로 인스턴트화하고, 둘 사이에 커넥텀이 규정하는 시냅스를 형성하여 시뮬레이션함. 형태·시냅스 위치·수는 마이크로 슬라이스 커넥텀에서 그대로 가져오며, pre 소마에 전류주입(IClamp, Current Clamp)으로 발화시키고 post 소마 반응을 기록함. 두 세포만으로도 전체망 속 동일 연결과 등가이며, 전시냅스 발화(활동전위)까지 함께 검증 가능함
- 신호 흐름은 다음과 같음: pre 소마 활동전위를 소마에서 문턱(−10 mV) 통과로 감지하여 1 ms 지연 뒤 시냅스로 전달하고, 커넥텀 위치(흥분성=수상돌기, 세포체주변 억제=소마 근처)에서 수동 전파·감쇠하여 post 소마에서 기록됨. 측정은 전·후세포 막전위(Vm, membrane potential)를 병행 기록하고(대표 경로는 세그먼트 Vm·시냅스 전류 i를 3D로 저장), 단기가소성 지표 PPR(Paired-Pulse Ratio)은 시냅스 컨덕턴스 g(t)를 주 측정량으로 사용함(PPR = g₂/g₁) — 후세포 클램프·아티팩트를 우회해 흥분·억제를 동일하게 강건히 측정함

[그림 가-4. 두 뉴런 쌍 구성 — 전시냅스(파랑)·후시냅스(보라) 세포, 축삭 stub, 자극/기록 지점, 시냅스, NetCon 전달 점선]  (`04_experiments/Ex2c_stp_dynamics_pair/figures/pair_method_schematic.png`)

#### ② 마이크로 슬라이스 네트워크

- 창(종축 500 × 측관통 800 × 두께 400 µm) 내부에 존재하는 세포 5,610개를 대표세포로 축소하지 않고 각각 전세포 완전형태 모델로 인스턴트화하고, 세포 간 연결은 Romani 2024 파이프라인의 두 단계 — 형태끼리 물리적으로 맞닿는 접촉점을 찾는 **touch**와, 그 접촉점을 pathway별 생리적 연결확률로 솎아내는 **prune** — 로 배선함. 이렇게 형성된 시냅스는 총 5,907,100개로, 창 내부 세포끼리의 회로 연결 4,835,416개와 외부에서 들어오는 Schaffer collateral 입력 1,071,684개로 구성됨. SC(Schaffer Collateral, 샤퍼 곁가지 = CA3→CA1 흥분성 입력)는 CA3 세포를 실제로 두지 않고 가상 섬유 10,000개로 대체하여, 각 섬유가 자신이 담당하는 시냅스들에 스파이크를 전달하는 방식으로 입력함
- 쌍에서 pathway별로 확정한 시냅스 규칙(단기 TM 파라미터, 장기 칼슘 기전)을 이 5,907,100개 시냅스에 그대로 배치하므로, 쌍이 맞으면 네트워크 집단 반응을 신뢰 가능함. 관측량은 두 가지 — 실측 MEA와 직접 대응하는 조직 규모의 **fEPSP**(field EPSP, 세포외 필드전위)와, 회로 활동(E-I 동역학·과발화)을 보는 **집단 스파이크·발화율**임. fEPSP는 모든 세그먼트에서 계산되는 빠른 막전류(fast_imem)를 전극 위치까지 순방향 계산(forward modeling; **PSA** 점원근사·**LSA** 선원근사, Point/Line Source Approximation)하여 SO 층의 **E1**·SP 층의 **E2**·SR 층의 **E3** 세 전극에서 취득하며, 이 인프라는 독립 라이브러리 **LFPy**와 상관계수 1.0으로 일치·검증됨(Ex3d). 5,610세포 개별 막전위(Vm) 파형은 데이터 규모상 저장하지 않으나(뉴런 쌍 검증에서는 Vm을 직접 기록함), fEPSP·스파이크가 모두 완전형태 세포의 막전위 동역학에서 산출되므로 회로 반응을 온전히 반영함

[그림 가-5. 마이크로 슬라이스 네트워크 형태 — 5,610 전세포(추체 5,040·인터뉴런 570) 소마 배치 · 층 구조(SO/SP/SR/SLM) · fEPSP 전극 E1·E2·E3 층관통 정렬]  (`04_experiments/Ex2b_connection_matrix/figures/network_morphology.png`)

[그림 가-5B. 마이크로 슬라이스 CA1 커넥텀 구조 — 원형 connectogram·연결 매트릭스]  (`04_experiments/Ex2b_connection_matrix/ui/connectome_circular.html`)

## 나) 검증

### 1) 뉴런 쌍

#### ① 단기가소성 (Ecker)

- 구축한 뉴런 쌍(아래 표의 대표 연결들)을 그대로 사용해 두 가지 단기가소성 실험을 진행함. 두 실험 모두 전시냅스를 자극하여 시냅스 컨덕턴스 g(t)를 기록하고 쌍마다 100 시행을 평균하며 지표로 PPR을 사용함. paired-pulse는 2펄스를 여러 ISI(Inter-Stimulus Interval, 두 펄스 사이의 자극 간 간격)로 인가하여 짧은 시간규모의 단기가소성을 보고, burst는 여러 펄스를 이어지는 열(train)로 인가하여 긴 시간규모의 단기가소성을 봄

- 검증 대상 경로: HippocampusHub Connection Physiology(Kohus et al. 2016, Romani 2024)의 22개 pathway 규칙을 STP 프로파일에 따라 5개 클래스로 분류
  - E1 촉진성 흥분: 추체세포(PC, Pyramidal Cell) → 오리엔스 인터뉴런(OLM·Tri·BS·BP), 낮은 U·강한 촉진
  - E2 억압성 흥분: 추체세포 → 대부분 표적, 높은 U·억압
  - I1 촉진성 억제: CCK·SCA·PPA 인터뉴런 상호, 촉진
  - I2 강억압 억제: PV basket(PVBC)·축삭축삭세포(AA, axo-axonic)·OLM·bistratified(BS) → 추체세포, 강억압·수적 우세
  - I3 약억압 수상돌기 억제: CCK basket(CCKBC)·SCA·PPA·Ivy → 추체세포, 약억압
- 5개 클래스를 포괄하는 **15개 대표 경로**로 구성하며 각 경로를 100 시행 반복함. 연결마다 STP 파라미터가 고유하게 지정됨 — U(방출확률)·D(감쇠 시상수, ms)·F(촉진 시상수, ms)·NRRP(방출가능 소포부위 수, Number of Readily Releasable Pool). 아래 표의 이 15개 경로가 단기가소성 검증(paired-pulse·burst)의 대상임

| 클래스 | 대표 쌍 (pre → post) | U | D (ms) | F (ms) | NRRP |
|---|---|---|---|---|---|
| E1 촉진성 흥분 | PC → OLM | 0.09 | 138 | 670 | 1 |
| E2 억압성 흥분 | PC → PC | 0.65 | 671 | 17 | 2 |
| | PC → Ivy | 0.50 | 671 | 17 | 1 |
| | PC → PVBC | 0.23 | 410 | 10 | 1 |
| I1 촉진성 억제 | CCKBC → CCKBC | 0.11 | 115 | 1542 | 1 |
| I2 강억압 억제 | PVBC → PC | 0.16 | 965 | 8.6 | 9 |
| | AA → PC | 0.10 | 1278 | 10 | 1 |
| | BS → PC | 0.13 | 1122 | 9.3 | 1 |
| | OLM → PC | 0.30 | 1250 | 2 | 1 |
| | PVBC → AA | 0.24 | 1730 | 3.5 | 1 |
| | PVBC → Ivy | 0.26 | 930 | 1.6 | 1 |
| I3 약억압 수상돌기 억제 | CCKBC → PC | 0.16 | 153 | 12 | 1 |
| | SCA → PC | 0.15 | 185 | 14 | 1 |
| | PPA → PC | 0.16 | 168 | 13 | 1 |
| | Ivy → PC | 0.32 | 144 | 62 | 1 |

- U가 비슷해도 D·F가 달라 거동이 갈림(PC→OLM은 F 670으로 강촉진, AA→PC는 D 1278로 강억압)
- 인터뉴런 약칭: OLM(oriens-lacunosum moleculare), PVBC(parvalbumin basket cell, 파브알부민 basket), CCKBC(CCK basket cell), BS(bistratified), AA(axo-axonic), SCA(Schaffer collateral-associated), PPA(perforant path-associated), Ivy(ivy cell), Tri(trilaminar), BP(back-projection)

##### 1. paired-pulse — 독립 문헌 검증

- 각 대표 경로마다 전시냅스에 펄스를 두 번 주는데, 두 펄스 사이의 간격 ISI(Inter-Stimulus Interval, 자극 간 간격)를 20, 50, 100, 200 ms 네 값으로 바꿔 가며 인가함. 각 ISI에서 두 번째 응답의 시냅스 컨덕턴스 최고값을 첫 번째 응답의 최고값으로 나눈 PPR(Paired-Pulse Ratio)을 구하고, 이를 ISI에 대해 이어 PPR-vs-ISI 곡선을 산출함. 이 곡선은 짧은 간격에서 시냅스가 촉진되는지 억압되는지, 그리고 그 효과가 간격이 벌어지면서 얼마나 빨리 원래대로 회복되는지를 보여 줌. 산출한 곡선은 같은 값을 개별 쌍기록으로 측정한 문헌과 직접 대조함

[그림 가-6. paired-pulse 자극 신호 — 2펄스, 모든 ISI(20·50·100·200 ms) 타이밍]  (`04_experiments/Ex2c_stp_dynamics_pair/figures/ex2c_stim_pp.png`)

- 결과 — 15개 경로 전체 분석

15개 경로를 5개 클래스로 묶어 PPR의 ISI 곡선을 보면 방향이 뚜렷하게 갈림. 촉진성인 E1(PC→OLM, 평균 PPR 약 1.20)과 I1(CCKBC→CCKBC, 약 1.67)은 두 번째 응답이 커지고, 억압성인 E2(약 0.64)와 I2(약 0.80)는 작아지며, I3 약억압 수상돌기 억제(약 1.00)는 거의 변하지 않음. 짧은 ISI(20 ms)에서 촉진·억압 효과가 가장 크고 ISI가 길어질수록 무변조(PPR 1)로 회복하는 고전적 단기가소성 시간규모도 함께 나타남

[그림 가-6B. 대표 경로 15개 PPR-vs-ISI 곡선(클래스별 그룹) — E1·I1 촉진, E2·I2·I3 억압/약억압]  (`04_experiments/Ex2c_stp_dynamics_pair/figures/ex2c_ppr15_bypath.png`)

같은 15개 경로를 쌍별 개별 패널로 제시하며, 음영 밴드는 각 경로의 100 시행 부트스트랩 95% CI(Confidence Interval, 신뢰구간 — 참값이 그 안에 있을 것으로 95% 신뢰하는 범위)임. 예를 들어 PC→PC는 0.36(0.28–0.45)으로 억압, PC→OLM은 1.89(1.04–3.69)로 촉진으로 경로마다 방향이 명확함

[그림 가-6C. 대표 경로 15개 개별 PPR-vs-ISI(쌍별 패널, 밴드 = 100시행 부트스트랩 95% CI)]  (`04_experiments/Ex2c_stp_dynamics_pair/figures/ex2c_ppr15_grid.png`)

15개 경로의 실제 뉴런 쌍 형태와, 각 쌍에서 시냅스가 놓인 위치와 개수, 전류주입으로 자극하는 pre 소마와 반응을 기록하는 post 소마 지점을 제시함

[그림 가-6D. 대표 경로 15개 뉴런 쌍 — 전(파랑)·후(보라) 실제 형태 · 시냅스 개수/위치(커넥텀 실측) · 자극(pre 소마)·기록(post 소마)]  (`04_experiments/Ex2c_stp_dynamics_pair/figures/ex2c_pairs15.png`)

15개 경로에 등장하는 세포 유형 9종(추체세포와 인터뉴런 8종)의 실제 완전형태를 제시함

[그림 가-6E. 대표 경로에 등장하는 세포 유형별 실제 형태(추체세포 + 인터뉴런 8종, 각 유형 대표 1개 전세포 완전형태)]  (`04_experiments/Ex2c_stp_dynamics_pair/figures/ex2c_celltypes.png`)

방출확률 U가 클수록 PPR이 낮아지는 관계를 STP 곡선과 PPR-vs-U 산점도로 제시함

[그림 가-7. STP 곡선·PPR-vs-U 산점도]  (`ex2c_stp_curves_15groups.png`)

클래스 사이의 PPR 분포 차이가 통계적으로 유의함을 제시함(Kruskal-Wallis H = 22.80, p = 1.4×10⁻⁴). 방출확률 U와 PPR의 음의 상관(Spearman r = −0.317)은 Tsodyks-Markram 모델의 예측과 정합함(이론 대 실측 Pearson r = 0.551)

[그림 가-8. 클래스별 PPR 분포·Kruskal-Wallis·bootstrap forest plot]  (`statistical_test/figures/ex2d_stats.png`, `ex2d_bootstrap.png`)

- 문헌 비교 검증 — 위 15경로 중 개별 쌍기록 문헌이 존재하는 대표 6쌍. 문헌 PPR 정량값은 각 논문 PubMed 초록에서 확인
  - 6쌍 중 4쌍이 방향(촉진·억압)에서 일치함(개별 쌍기록이므로 독립 문헌 검증에 해당). 정량 크기는 문헌마다 자극 조건·측정 ISI가 달라 편차가 있음. 나머지 2쌍 중 OLM→PC는 불일치, Ivy→PC는 정량 PPR 문헌이 희소하여 판정을 보류하며, 각 사유를 아래에서 확인함

| 경로 | 시뮬 PPR | 문헌 PPR (정량) | 판정 | 문헌 |
|---|---|---|---|---|
| PC → PC | 0.34 | 억압 (ISI↑ 회복; 정량 PPR 미보고) | 방향 일치 | Deuchars & Thomson 1996 (CA1) |
| PVBC → PC | 0.81 | 약 0.63 (37 % PPD @10 ms) | 방향 일치 | Kraushaar & Jonas 2000 (⚠️치상회) |
| CCKBC → PC | 0.83 | 억압 (초록은 비동기 방출 중심, PPR 미보고) | 방향 참고 | Hefft & Jonas 2005 (⚠️치상회) |
| PC → OLM | 1.07 | 2.53 (@20 ms) | 방향 일치 (크기 약함) | Ali & Thomson 1998 (CA1) |
| OLM → PC | 0.62 | 0.93 (@100 ms, 무변조) | 불일치 | Maccaferri et al. 2000 (CA1) |
| Ivy → PC | 1.16 | PPR 미보고 (느린 억제만 기술) | 보류 | Fuentealba et al. 2008 (CA1) |

> 문헌 PPR은 각 논문 PubMed 초록 확인. ⚠️ Kraushaar 2000·Hefft 2005는 CA1이 아니라 치상회(dentate gyrus)의 basket·CCK 문헌으로 CA1 PVBC·CCKBC의 가장 가까운 대체 인용임. 특히 Hefft 2005는 PPR이 아니라 비동기 방출을 다룬 논문이라 억압 방향의 참고 근거로만 사용. 흥분성 2쌍(PC→PC·PC→OLM)만 CA1 직접 문헌이며, PC→OLM은 문헌 촉진(2.53 @20 ms)이 우리(1.07)보다 강함
- 기전: 억제성(PVBC·CCKBC·OLM→PC)의 억압과 흥분성(PC→OLM)의 촉진은 모두 전시냅스 방출확률 동역학(TM: 높은 U는 첫 방출로 자원 고갈→억압, 낮은 U에 촉진 F가 더해지면→촉진)에서 나오며, 부호가 문헌과 일치함
- 불일치한 OLM→PC는 시뮬이 억압(0.62)인 반면 문헌은 무변조(Maccaferri et al. 2000, PPR 0.93 @100 ms)임. 방출확률 U를 0.10으로 재조정하면 시뮬 PPR이 1.12로 이동해 무변조(≈1)를 재현하므로, 편차가 파라미터 미세조정 부족에서 비롯됨을 확인함
- 보류한 Ivy→PC는 시뮬에서 촉진(PPR 1.16)으로 나타나지만, Ivy 세포는 비교적 최근에 특성화되어(Fuentealba et al. 2008) 정량 PPR 문헌이 희소함(초록도 PC로부터의 느린 억제만 기술). 따라서 방향만 제시하고 판정은 보류하며, 향후 정량 문헌이 확보되면 재검토함

##### 2. burst — Ecker STP 프레임워크 재현

- 8개의 펄스로 이루어진 자극열을 5, 10, 20, 40 Hz의 여러 주파수로 각각 인가함. 주파수가 곧 펄스 사이 간격을 정하므로 펄스 간격은 각각 200, 100, 50, 25 ms로 서로 다름(간격은 1000을 주파수로 나눈 값). 이렇게 자극 빈도를 바꿔 가며 펄스별 응답이 어떻게 촉진되거나 억압되는지를 봄
- 단기가소성 파라미터가 Ecker 피팅값이므로 이 실험은 독립 문헌 검증이 아니라, 본 시뮬레이터의 구현이 Ecker가 특성화한 클래스별 단기가소성 동역학을 제대로 재현하는지 확인하는 자기일관성 검증에 해당함

[그림 가-9. burst 자극 신호 — 주파수 스윕(5·10·20·40 Hz), 각 8펄스 자극열 (간격 200·100·50·25 ms)]  (`04_experiments/Ex2c_stp_dynamics_pair/figures/ex2c_stim_burst.png`)

- 결과 — 15개 경로 전체 분석

15개 경로 모두 클래스별 특성이 재현됨. 촉진성인 E1(PC→OLM)과 I1(CCKBC→CCKBC)은 자극열이 진행되며 응답이 누적하여 커지고(I1은 여덟째 펄스에서 첫 펄스의 3배 이상), 억압성인 E2와 I2는 두 번째 펄스부터 급감하며, 자극 빈도가 높을수록(40 Hz) 억압이 더 강하게 나타남. I3 약억압은 초반에 완만히 감소한 뒤 회복하는 패턴을 보임

[그림 가-9B. 대표 경로 15개 burst(8펄스 자극열) — 주파수(5·10·20·40 Hz)별 펄스 응답, 촉진형 누적 증대·억압형 급감]  (`04_experiments/Ex2c_stp_dynamics_pair/figures/ex2c_burst15.png`)

대표 쌍에 대해서는 시냅스 전류가 만드는 세포외 전위장을 LFP로 보조 계산함

[그림 가-10. 대표 쌍별 LFP(Local Field Potential, 국소 장전위) 세포외 전위장(보조)]  (`Ex2e_pair_LFPy/figures/pairfield_*.png`)

- 소결: paired-pulse에서 독립 문헌 4/6 일치·클래스 차이 유의·TM 정합, burst에서 Ecker 클래스 동역학 재현으로 구현 일관성 확인. 불일치 1쌍(OLM→PC)은 방출확률 재조정으로 해소되고, 보류 1쌍(Ivy→PC)은 정량 PPR 문헌이 희소하여 방향만 확인하고 판정을 보류함

#### ② 장기가소성 (Graupner-Brunel)

- 목적은 Graupner-Brunel 칼슘 모델이 교체 가능한 표준 슬롯으로서 문헌 STDP(Spike-Timing-Dependent Plasticity)를 **부호와 창모양** 수준에서 재현하는지 보는 것임(정량 완전일치가 아니라 방향·형태 재현이 1차 목표). 신규 가소성 모델도 같은 프로토콜로 이 자리에서 재검증됨
- STDP는 전·후 스파이크 시간차 Δt가 입력인 **쌍 수준 현상**이라 타이밍을 통제할 수 있는 쌍에서만 검증 가능함(문헌도 모두 쌍·단일 시냅스). 셋업은 E3(SR 층) 근처 추체세포 한 개에 SC(Schaffer Collateral) 시냅스 한 개를 커넥텀 그대로 두고, 단기 검증에서 0으로 얼렸던 gamma_p·gamma_d를 되살려 효능 변수 rho를 활성화한 뒤, 전시냅스(VecStim)와 후시냅스(소마 전류주입)로 Δt를 부여해 유도 후 rho를 판독함
- 유도에는 후시냅스 버스트가 필요함 — 단일 스파이크쌍은 시냅스후 칼슘이 강화문턱 theta_p에 못 미쳐 변화가 없고(Inglebert et al. 2020), 버스트를 더해야 rho가 움직임(셋업 검증에서 확인)
- 세 실험으로 구성함 — **Δt 타이밍 곡선 · 후시냅스 버스트 발수 · 유도 반복 빈도**로, Graupner-Brunel의 검증 세 축(타이밍·버스트 수·빈도)을 쌍에서 재현함. 각 실험은 **ca_stp 0(칼슘 고정, Graupner-Brunel 원본·문헌 기준)과 1(BBP 확률방출 연동, 본 확장)을 병렬**로 산출해 확률방출이 곡선의 부호·창을 어떻게 바꾸는지 봄
- 문헌 대조 성격은 실험마다 다름 — 타이밍은 파라미터를 피팅하지 않은 **Bi & Poo 1998 독립 대조**, 버스트 수는 기본값 출처인 **Wittenberg & Wang 2006 구현 검증**임

##### 1. STDP 타이밍 곡선 — 부호·창 독립 대조

- 후시냅스 버스트를 고정한 채 전·후 스파이크 시간차 Δt를 음수에서 양수까지(−100에서 +100 ms) 여러 값으로 훑으며(sweep) 각 Δt에서 유도 후 rho 변화를 측정하여 STDP 곡선을 산출함. 유도 프로토콜은 전시냅스 단일 스파이크와 후시냅스 버스트를 Δt 간격으로 짝지어 5 Hz로 유도 반복 30회 인가하는 구성임

[그림 가-14. STDP 유도 프로토콜 도해 — 전시냅스 단일 스파이크와 후시냅스 버스트(4발), 시간차 Δt 정의, 5 Hz 유도 반복 30회]  (`04_experiments/Ex10_STDP_pair/figures/stdp_protocol.png`)

- 결과: E3 근처 대표 추체세포(gid 2720)에서 산출한 Δt 곡선은 두 ca_stp 모두 Δt 0 부근에서 강화가 최대인 정점형임(ca_stp 0 정점 Δρ +0.69 @Δt +5 ms, ca_stp 1 +0.68 @Δt +10 ms). ca_stp 0(원본)은 큰 양(+)Δt(+30 ms 이상)에서 Δρ가 0으로 급락하고 음(−)Δt로는 완만히 0까지 감소하는 좁은 창인 반면, ca_stp 1(확장)은 전 구간 바닥이 높아(음Δt에서도 Δρ 0.29 이상) 넓은 창임 — 확률방출이 성공하면 칼슘이 크게 튀어 타이밍과 무관하게 강화되기 때문임. 강한 post-burst(4발)가 타이밍 비대칭을 상당히 덮어 Bi & Poo 1998의 좁은 비대칭 창보다 넓게 나타나며, 이는 burst 강도·발수 조정으로 좁힐 수 있는 튜닝 지점임(아래에서 확인)

[그림 가-15. STDP 타이밍 곡선(실측 · 30짝 평균) — Δt 대 Δrho, ca_stp 0(원본)·1(확장) 병렬, Δt 0 부근 정점]  (`04_experiments/Ex10_STDP_pair/figures/stdp_curve.png`)

- 튜닝 확인: 후시냅스 버스트를 4발에서 2발로 줄이면 창이 크게 좁아짐(ca_stp 0). 강화가 ±20 ms 안으로 한정되고 정점이 Δt +5 ms로 이동하며, +5 ms(Δρ 0.35)가 −5 ms(0.22)보다 커 pre-before-post가 더 강한 STDP 비대칭이 드러남 — 강한 4발 버스트가 타이밍 비대칭을 덮던 것을 2발로 완화해 Bi & Poo류 좁은 창에 근접시킴. 버스트 강도가 STDP 창 폭을 조절하는 노브임을 실측으로 확인함

[그림 가-15B. STDP 타이밍 창 튜닝(실측 · ca_stp 0) — post-burst 4발(넓은 창)과 2발(좁은 창·+5 ms 비대칭) 비교]  (`04_experiments/Ex10_STDP_pair/figures/stdp_curve_tuning.png`)

##### 2. post-burst 수 의존 — Wittenberg 재현

- 시간차 Δt를 +10 ms로 고정한 채 후시냅스 버스트의 발수를 1발·2발·4발로 바꿔 가며 rho 변화를 측정함. 본 모델 경로(CA3에서 CA1 추체세포)와 기본값(Wittenberg 2006 피팅치)이 이 실험의 대조 문헌과 동일하므로, 버스트 수가 강화를 가르는 Wittenberg의 핵심 관찰을 직접 재현하는지 확인함

- 결과: ca_stp 0(원본)의 Δρ가 1발 +0.00, 2발 +0.31, 4발 +0.61로 단조 증가함 — 단일 스파이크만으로는 칼슘이 강화문턱 theta_p에 못 미쳐 변화가 없고, 버스트가 쌓여야 문턱을 넘어 강화되는 **Wittenberg & Wang 2006의 "버스트 필요"를 정량 재현**함. 반면 ca_stp 1(확장)은 1발에서 이미 +0.52로 강화되는데, 이는 단일소포(Nrrp=1) 방출 성공 시 칼슘이 과도 주입되는 확률방출 아티팩트로(mod 저자 경고와 일치), ca_stp 0이 생리적으로 타당함을 뒷받침함

[그림 가-16. post-burst 수 의존(실측 · Wittenberg 재현) — 후시냅스 1·2·4발 Δrho, ca_stp 0(단일=0 → 버스트=강화)·1(단일부터 강화)]  (`04_experiments/Ex10_STDP_pair/figures/stdp_burstnum.png`)

##### 3. pairing 주파수 의존 — Sjöström 재현

- 유도 반복의 빈도를 저빈도에서 고빈도까지(1·5·20·50 Hz) 바꿔 가며(후스파이크 단일) rho 변화를 측정함. 저빈도에서는 짝 사이 칼슘이 감쇠해 무변화, 고빈도에서는 칼슘이 누적돼 강화로 전환되는 빈도 의존이 재현 대상임(대조 문헌 Sjöström et al. 2001은 신피질 자료라 정성 대조)

- 결과: ca_stp 0(원본)의 Δρ가 1 Hz +0.00, 5 Hz +0.00, 20 Hz +0.61, 50 Hz +0.69로 **저빈도 무변화에서 고빈도 강화로 전환**됨 — 단일 후스파이크라도 짝짓기가 잦으면 칼슘이 감쇠보다 빨리 누적돼 강화문턱을 넘기 때문으로, 저빈도 무변화에서 고빈도 LTP로 갈리는 Sjöström의 빈도 의존을 정성 재현함. ca_stp 1(확장)은 1 Hz(+0.45)부터 강화되어 빈도와 무관하게 높음

[그림 가-17. pairing 주파수 의존(실측 · Sjöström 재현) — 반복 빈도 1~50 Hz Δrho, ca_stp 0 저빈도 무변화 → 고빈도 강화]  (`04_experiments/Ex10_STDP_pair/figures/stdp_freq.png`)

- 소결: 세 실험 모두 Graupner-Brunel 슬롯이 타이밍·버스트 수·빈도 세 축에서 문헌 방향을 재현함. 특히 **ca_stp 대조가 결정적** — ca_stp 0(원본)은 단일 스파이크·저빈도에서 무변화이고 버스트·고빈도라야 강화되는 생리적 문턱 거동(Wittenberg·Sjöström 일치)인 반면, ca_stp 1(확장)은 단일 방출에도 강화되는 확률방출 아티팩트를 보임. 따라서 문헌 대조와 신규 모델 검증의 기준선은 ca_stp 0이며, ca_stp 1은 확률방출 확장의 한계를 드러내는 대조군임. 타이밍 창이 넓게 나온 것(강한 burst가 비대칭을 덮음)은 burst 강도·발수 조정으로 좁히는 것이 후속 튜닝 과제임

- 유도 30회를 평균한 세포 위 신호 전파(SC 시냅스에서 EPSP가 소마로 전파된 뒤, 소마 스파이크의 역전파 활동전위 bAP가 수상돌기로 되돌아 올라가 시냅스 칼슘을 형성)를 세그먼트 막전위 색으로 3차원 재생하는 인터랙티브 뷰를 제공함

[그림 가-18. STDP 신호 전파(3D · 30짝 평균) — 세그먼트 Vm 색, EPSP 전파(시냅스→소마)·bAP 역전파(소마→수상돌기)]  (`04_experiments/Ex10_STDP_pair/ui/stdp_viz_dt10.html`)

### 2) 마이크로 슬라이스 네트워크

#### ① 단기가소성 (Ecker)

- 쌍에서 pathway별로 확정한 시냅스를 국소 조직 전체에 그대로 배치한 뒤, 실제 MEA가 측정하는 조직 규모의 fEPSP가 문헌과 일치하는지 확인함. 자극은 CA3에서 들어오는 Schaffer collateral(SC) 섬유 다발을 활성화하는 방식으로, 가상 섬유 10,000개 가운데 일부를 동시에 발화시키며 그 비율을 fiber volley(발화 섬유 %)로 조절함. 아래 세 실험(I-O·paired-pulse·burst)은 이 SC volley를 각각 세기·타이밍·빈도만 달리 주어 회로 반응을 봄

##### 1. I-O 곡선

- I-O(Input-Output, 입출력)는 자극 세기를 바꿔 가며 그에 따른 반응 크기를 보는 실험임. SC 자극 세기, 곧 동시에 발화하는 섬유 비율(fiber volley %)을 조절하여 각 세기에서 단발 SC volley 한 번을 주고 E3(SR 층) 전극의 fEPSP 피크를 측정함. 실측 실험실에서 자극 강도를 단계적으로 올려 반응 곡선을 그리는 관례와 같으며, 이 곡선으로 이후 실험에서 쓸 표준 자극 세기를 정함
- 실험은 두 단계로 진행함. 먼저 넓은 세기 범위(100 %까지)를 훑어 본 결과 fEPSP가 50 % 부근에서 최대에 이른 뒤 더 올려도 커지지 않고 오히려 감소하며 발화 세포도 75 % 이상에서 100 %로 포화되어, 고세기에서는 자극 세기를 구분해 볼 수 없음을 확인함. 이렇게 고세기가 포화되므로 세기에 따라 반응이 완만히 갈리는 실제 작업 구간인 저세기(0.5·1·2·4·8 %)를 다시 상세히 재실험하여 발화·fEPSP·파형을 정밀하게 봄
- 결과

저세기(0.5→8 %)에서는 자극 세기가 커질수록 발화 세포가 1.2→34.6 %, E3(SR) fEPSP가 −89→−1303 µV로 단조 증가하는 완만한 gradual I-O를 보임. 그림 가-11은 이 저세기 재실험 결과로, 왼쪽은 발화 세포 비율(firing I-O), 오른쪽은 E3(SR) fEPSP 피크가 세기에 따라 단조 증가함을 정상과 억제 차단을 겹쳐 보임

[그림 가-11. SC I-O — 저세기(0.5~8 %) 발화·fEPSP 곡선(정상 대 억제 차단)]  (`04_experiments/Ex3_io_inhibition/figures/ex3_io_curve.png`)

같은 저세기 구간의 실제 E3(SR) fEPSP 파형을 겹쳐 보면, 자극 세기가 커질수록 파형의 음성 피크가 단조로 깊어지는 과정이 직접 나타남

[그림 가-11B. I-O 세기별 E3(SR) fEPSP 파형 — 세기↑ → 진폭↑]  (`04_experiments/Ex3_io_inhibition/figures/ex3_io_waveforms.png`)

넓은 범위(0.5~100 %)로 보면 fEPSP는 50 %에서 −3130 µV로 최대에 이르고 75·100 %에서는 오히려 감소하는 비단조 거동을 보임(발화 세포가 50 %에 이미 ~97 %, 75 % 이상 100 %로 포화되기 때문). 그림 가-11C는 이 전범위 곡선을 로그축으로 보이며, 세기 단계별 탈억제(억제 차단과 정상의 fEPSP 차이 %)를 막대로 보이는데 단발 자극에서는 대부분 작아 억제의 필드 기여가 미미함

[그림 가-11C. SC I-O 전범위(0.5~100 %) + 세기별 탈억제(정상 대 억제 차단)]  (`04_experiments/Ex3_io_inhibition/figures/ex3_io_combined_0-100.png`)

- 이 I-O를 근거로 표준 자극 세기를 volley 8 %로 확정함(최대의 ~42 %, 증가·감소 모두 관측 여지 + 억제뉴런 일부 동원으로 E-I 동역학 관찰 가능). 단발 자극에서 억제 차단이 무효인 것(normal ≈ block)은 단시냅스 흥분(~3 ms)이 이중시냅스 억제(~5–8 ms)보다 먼저 도착하는 피드포워드 타이밍 때문임(정상)

##### 2. paired-pulse (PPF)

- paired-pulse는 SC volley를 두 번 주되 두 자극 사이 간격(ISI)을 20·50·100·200 ms로 바꿔 가며, 전체 망의 fEPSP로 PPR(두 번째 응답을 첫 번째 응답으로 나눈 비)을 측정함. 뉴런 쌍에서 한 paired-pulse 실험을 조직 규모에서 그대로 재현하는 것으로, 자극 세기는 위 I-O에서 정한 volley 8 %로 고정하고 자극 개수는 2발임
- 결과

전극별 PPR을 ISI에 대해 그려 보면, 주 측정 전극 E3(SR)의 PPR이 20 ms에서 0.96으로 짧은 간격에서 약간 눌리다가 50 ms에서 1.10으로 정점을 찍고 100·200 ms에서 1.07로 완만히 내려옴(E2·E1은 전체적으로 더 높음). 30–60 ms에서 촉진이 정점을 찍는 고전적 PPF 곡선과 일치하며(Leung & Fu 1994), 억제를 차단해도 곡선이 완전히 겹쳐(normal = block) "EPSP slope의 PPF는 GABA에 의존하지 않는 전시냅스성 현상"이라는 보고와 일치함

Leung & Fu 1994 전문(Fig 2·본문)에서 ISI별 거동을 확인해 대조하면 아래와 같음. 자극 경로·측정법·정점 ISI·ISI별 방향(단ISI 억압 → 중ISI 촉진 → 장ISI 수렴)·GABA 비의존성이 모두 일치하며, 정점 크기만 우리 쪽이 약함

| 비교 항목 | 본 시뮬 (E3/SR fEPSP) | Leung & Fu 1994 (EPSP slope) | 판정 |
|---|---|---|---|
| 자극 경로 | Schaffer collateral → CA1 | Schaffer collateral → CA1 첨두 수상돌기 | ✅ 동일 |
| 측정 지표 | fEPSP slope 비 E2/E1 | EPSP slope 비 E2/E1 (EPI), field·세포내 유사 크기 | ✅ 동일 |
| 20 ms | 0.96 (약간 억압) | E2 < E1 (10–20 ms 억압) | ✅ |
| 정점 ISI | 50 ms | 40–50 ms | ✅ 일치 |
| 정점 크기 | 1.10 | 약 1.4–1.5 (1.4×역치; 저강도서 더 큼) | 방향 일치·크기 약함 |
| 100 ms | 1.07 (>1) | E2 > E1 (30–100 ms 촉진) | ✅ |
| 200 ms | 1.07 | ~150 ms에서 unity로 수렴(저강도선 200 ms도 >1) | ✅ 대체로 |
| GABA 차단 영향 | 없음 (normal = block) | 없음 (bicuculline·GABAB·NMDA 무영향) | ✅ |

> 값은 Leung & Fu 1994 전문(Fig 2·본문)에서 확인(PubMed; DOI 10.1016/0006-8993(94)90209-7). EPI는 EPSP slope의 E2/E1이며 field·세포내·소마·수상돌기 측정이 유사 크기(원문). EPI가 자극세기에 의존(저강도서 최대·고강도서 감소)하는데, 우리 정점 크기(1.10)가 그들의 1.4×역치값(약 1.5)보다 약한 것은 집단 fEPSP가 순수 EPSP에 집단스파이크·피드포워드 억제·공간평균이 섞여 희석되고 volley 8%가 저역치 조건이 아니기 때문임(쌍 수준 SC→PC PPR은 2.11, Ex2)

- 기전과 판정 근거: 곡선 전체가 하나의 전시냅스 기전 곧 SC 말단의 잔여 칼슘(residual Ca)으로 설명됨. 첫 자극 뒤 칼슘이 수십 ms에 걸쳐 감쇠하므로 둘째 자극이 그 창에 도달하면 방출확률이 올라 촉진되며(정점 30–60 ms = 칼슘·촉진 시상수 F의 시간척도), 100 ms 이상이면 칼슘이 사라져 unity로 수렴함. 이는 본 모델의 TM 촉진(이용률 u 상승)과 Leung & Fu가 제시한 잔여칼슘 전시냅스 촉진이 동일한 기전임. '일치' 판정은 자극 경로·측정법이 같고, ISI별 곡선 형태(단ISI 억압 → 40–50 ms 정점 → 장ISI 수렴)가 같으며, GABA 비의존성이 같고, 그 바탕 기전이 같다는 네 축이 모두 맞는 데 근거함(크기만 집단 fEPSP 희석으로 약함)
- 20 ms 억압의 이유: 가장 짧은 간격에서는 촉진이 아직 정점에 못 올랐고, 집단 fEPSP의 둘째 응답 기울기가 첫 응답이 미처 사라지기 전에 측정되어 겉보기 slope가 눌림(합산·타이밍 효과). 억제를 차단해도 그대로(normal = block, bicuculline 무영향)이므로 피드포워드 억제가 아니며, Leung & Fu도 10–20 ms에서 E2 < E1을 같은 GABA 비의존 방식으로 관찰함

[그림 가-12. 네트워크 paired-pulse PPR-vs-ISI (전극별 · 정상 = 억제 차단, GABA 비의존)]  (`04_experiments/Ex3b_microSlice_PP/figures/ex3b_ppr_curve.png`)

같은 실험의 fEPSP를 각 ISI별로 3차원으로 재생·비교하는 인터랙티브 뷰로, 두 자극에 대한 필드 반응의 시공간 분포를 층(SO/SP/SR)별로 확인할 수 있음

[그림 가-12B. 네트워크 paired-pulse fEPSP(3D) — 모든 ISI(20·50·100·200 ms)]  (`04_experiments/Ex3b_microSlice_PP/ui/fepsp3d_isi{20,50,100,200}_norm.html`)

##### 3. burst

- burst는 SC volley 8발로 이루어진 자극열(train)을 8·40·100 Hz의 세 빈도로 주어 조직 규모 fEPSP의 주파수 의존 반응을 봄. 자극 세기는 volley 8 %로 고정하고 펄스 수는 8발이며, 빈도만 바꿔 가며 회로가 자극 빈도를 어떻게 걸러 내는지를 관찰함
- 결과

주 측정 전극 E3(SR)의 펄스별 정규화(P/P1) slope를 보면 감쇠가 자극 빈도에 따라 단조롭게 갈림 — 8 Hz는 자극열 끝까지 유지(P8/P1 slope 1.00, 초반 P2에서 1.20으로 오히려 촉진), 40 Hz는 중간 감쇠(P8/P1 0.82), 100 Hz는 강한 감쇠(P8/P1 0.24)로, 저빈도는 통과·유지하고 고빈도일수록 눌리는 저역통과 필터링을 보임. 이는 단기가소성의 방출자원 동역학(TM)에서 나오는 특성으로, 8 Hz(펄스 간격 125 ms)는 펄스 사이 자원이 회복돼 촉진성 SC 입력이 유지되지만 100 Hz(간격 10 ms)는 회복보다 고갈이 빨라 감쇠가 지배함. 네트워크 burst는 단일 시냅스 STP를 직접 검증하기보다 이 회로 수준 주파수 필터링을 관찰하는 성격으로, Ecker 2020 STP 파라미터를 얹은 망이 기대되는 빈도 의존 필터링을 창발적으로 재현함

[그림 가-13. 네트워크 burst — 전극별·주파수별(8·40·100 Hz) 펄스 응답 요약]  (`04_experiments/Ex3c_microSlice_burst/figures/ex3c_burst_summary.png`)

각 빈도의 네트워크 fEPSP를 3차원으로 재생하는 인터랙티브 뷰로, 자극열이 진행되며 필드 반응이 층(SO/SP/SR)별로 어떻게 누적·감쇠하는지를 빈도마다 확인할 수 있음

[그림 가-13B. 네트워크 burst fEPSP(3D) — 빈도별]  (`04_experiments/Ex3c_microSlice_burst/ui/fepsp3d_tr8_norm.html`, `04_experiments/Ex3c_microSlice_burst/ui/fepsp3d_tr40_norm.html`, `04_experiments/Ex3c_microSlice_burst/ui/fepsp3d_tr100_norm.html`)

#### ② 장기가소성 (Graupner-Brunel)

##### 1. 주파수의존 LTD/LTP — 진행 예정

- 동일 Graupner-Brunel 모델을 네트워크에 얹고, 표준 LTP/LTD 유도 프로토콜로 fEPSP slope % 변화를 측정하여 문헌과 대조
- 유도 프로토콜(각 문헌)
  - HFS-LTP: 고빈도 자극(100 Hz 테타너스 1초) → LTP (Bliss & Collingridge 1993; Hernandez et al. 2005)
  - TBS-LTP: theta-burst(4펄스 100 Hz를 5 Hz theta 리듬으로 반복) → LTP, 가장 생리적 (Larson & Munkácsy 2015)
  - LFS-LTD: 저빈도 자극(1 Hz × 900펄스) → LTD (Dudek & Bear 1992)
  - 주파수-응답(BCM): 1~100 Hz를 훑어 저빈도 LTD ↔ 고빈도 LTP로 부호가 갈리는 곡선 재현, 위 셋을 종합 (Dudek & Bear 1992)
- Graupner 모델이 칼슘(=주파수) 의존으로 가소성 부호를 결정하도록 설계되어, 이 주파수-응답이 핵심 검증 대상임
- 네트워크 유도 시뮬레이션은 조건당 수 시간에서 수십 시간이 드는 무거운 계산이므로, ca_stp는 쌍 검증에서 문헌과 더 잘 맞는 것으로 판정된 대표 한 값만 적용함(쌍처럼 0·1을 모두 돌리지 않음). 즉 쌍에서 방법론을 확정하고 그 결론을 네트워크에 적용하는 순서임
- 실행 가능성 단계: HFS(100 Hz 1초)와 TBS는 본 데스크톱에서 실행 가능하여 먼저 진행함. 반면 LFS-LTD(1 Hz를 7분에서 15분간 인가)는 자극 시간이 길어 본 데스크톱 단독으로는 비현실적이므로, 자극 시간 단축안 또는 클라우드 실행을 별도 대책으로 두고 이후 진행함

#### ③ 실측 · 장기가소성 (Graupner-Brunel) vs MEA

##### 1. MEA 실측 대조 — 진행 예정

- 위 장기가소성(현재 Graupner-Brunel, 향후 신규 모델)의 유도 결과를 자체 MEA 실측과 동일 프로토콜로 대조
- 시뮬 예측값이 실측과 동일 경향(부호·크기·주파수 의존)을 보이는지가 최종 검증임

---

- 본 과제의 신규 가소성 모델은 위 장기가소성 자리(나-1-② 쌍 STDP · 나-2-② 네트워크 주파수의존 LTD/LTP · 나-2-③ 실측 MEA)에 교체·재검증되며, 단기가소성으로 검증된 기반은 그대로 재사용됨

---

## 참고문헌 (References)

> 서지는 PubMed로 대조·검증함(제목·저널·권/쪽·DOI). "확인 필요" 표시 항목은 서지 최종확정 예정.

**단기가소성 — 쌍 paired-pulse·burst 대조 문헌**

- Deuchars J, Thomson AM (1996) CA1 pyramid-pyramid connections in rat hippocampus in vitro: dual intracellular recordings with biocytin filling. *Neuroscience* 74(4):1009–1018. https://doi.org/10.1016/0306-4522(96)00251-5
- Kraushaar U, Jonas P (2000) Efficacy and stability of quantal GABA release at a hippocampal interneuron-principal neuron synapse. *J Neurosci* 20(15):5594–5607. https://doi.org/10.1523/JNEUROSCI.20-15-05594.2000
- Hefft S, Jonas P (2005) Asynchronous GABA release generates long-lasting inhibition at a hippocampal interneuron-principal neuron synapse. *Nat Neurosci* 8(10):1319–1328. https://doi.org/10.1038/nn1542
- Maccaferri G, Roberts JD, Szucs P, Cottingham CA, Somogyi P (2000) Cell surface domain specific postsynaptic currents evoked by identified GABAergic neurones in rat hippocampus in vitro. *J Physiol* 524(1):91–116. https://doi.org/10.1111/j.1469-7793.2000.t01-3-00091.x
- Ali AB, Thomson AM (1998) Facilitating pyramid to horizontal oriens-alveus interneurone inputs: dual intracellular recordings in slices of rat hippocampus. *J Physiol* 507(1):185–199. https://doi.org/10.1111/j.1469-7793.1998.185bu.x
- Fuentealba P, Begum R, Capogna M, et al. (2008) Ivy cells: a population of nitric-oxide-producing, slow-spiking GABAergic neurons and their involvement in hippocampal network activity. *Neuron* 57(6):917–929. https://doi.org/10.1016/j.neuron.2008.01.034

**가소성 모델**

- Ecker A, Romani A, Sáray S, et al. (2020) Data-driven integration of hippocampal CA1 synaptic physiology in silico. *Hippocampus* 30(11):1129–1145. https://doi.org/10.1002/hipo.23220
- Graupner M, Brunel N (2012) Calcium-based plasticity model explains sensitivity of synaptic changes to spike pattern, rate, and dendritic location. *PNAS* 109(10):3991–3996. https://doi.org/10.1073/pnas.1109359109
- Chindemi G, Abdellah M, Amsalem O, et al. (2022) A calcium-based plasticity model for predicting long-term potentiation and depression in the neocortex. *Nat Commun* 13:3038. https://doi.org/10.1038/s41467-022-30214-w

**네트워크·장기가소성 — 유도/검증 문헌**

- Leung LS, Fu XW (1994) Factors affecting paired-pulse facilitation in hippocampal CA1 neurons in vitro. *Brain Res* 650(1):75–84. https://doi.org/10.1016/0006-8993(94)90209-7
- Dudek SM, Bear MF (1992) Homosynaptic long-term depression in area CA1 of hippocampus and effects of N-methyl-D-aspartate receptor blockade. *PNAS* 89(10):4363–4367. https://doi.org/10.1073/pnas.89.10.4363
- Bliss TVP, Collingridge GL (1993) A synaptic model of memory: long-term potentiation in the hippocampus. *Nature* 361(6407):31–39. https://doi.org/10.1038/361031a0
- Larson J, Munkácsy E (2015) Theta-burst LTP. *Brain Res* 1621:38–50. https://doi.org/10.1016/j.brainres.2014.10.034

**STDP — 쌍 장기가소성 검증 문헌 (진행 예정)**

- Bi GQ, Poo MM (1998) Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type. *J Neurosci* 18(24):10464–10472. https://doi.org/10.1523/JNEUROSCI.18-24-10464.1998
- Wittenberg GM, Wang SS (2006) Malleability of spike-timing-dependent plasticity at the CA3-CA1 synapse. *J Neurosci* 26(24):6610–6617. https://doi.org/10.1523/JNEUROSCI.5388-05.2006

**모델·데이터 기반 파이프라인**

- Romani A, et al. (2024) *PLoS Biology* — 전체 CA1 모델·아틀라스·커넥텀 파이프라인 원본. https://doi.org/10.1371/journal.pbio.3002861
- Reimann MW, et al. (2015) touch+prune 커넥텀 알고리즘. *Front Comput Neurosci* 9:120. https://doi.org/10.3389/fncom.2015.00120

**서지 확인 필요 (본문 인용, 최종확정 예정)**

- Inglebert Y, et al. (2020) — 생리적 칼슘 농도에서의 STDP (PubMed 후보 PMID 33328274, 최종 서지 확인 권장)
- Hernandez et al. (2005) — HFS-LTP (서지 미확정)
- Kohus et al. (2016) — pair recording, HippocampusHub Connection Physiology 출처 (서지 미확정)
