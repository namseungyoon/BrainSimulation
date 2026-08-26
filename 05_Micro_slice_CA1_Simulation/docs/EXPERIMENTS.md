# 실험 레지스트리 (Ex1~Ex12) — micro-slice CA1

> ⚠️ **명명 규칙 (2026-08-22 확정)**: **전극 = E1(SO)·E2(SP)·E3(SR)** · **실험 = Ex1~Ex11**. (과거 실험을 E1~E11로 쓰다 전극 E1~E3와 충돌 → 실험에 `Ex` 접두.)
> 단일 색인: **Notion번호 ↔ 코드경로 ↔ 그림**. 실험이 카테고리에 분산돼도 여기서 한눈에 추적.
> 공통 양식(각 실험): 목표 / 방법·입력 / 검증지표 / 결과·상태 / 근거(논문X→우리Y) / 한계·주의.
> 완료는 결과까지, 미실행은 "(예상·미실행)".

| ID | 실험 | 폴더 | Notion | 상태 |
|---|---|---|---|---|
| Ex1-A | 무자극 자발 발화율 | `04_experiments/Ex1_baseline` | 05페이지 | ✅ **0 Hz (완전 무음)** |
| Ex1-B | 단일 volley 구동 검증 | `04_experiments/Ex1_baseline` | 05페이지 | ✅ **39% (2,182/5,610)** |
| Ex2 | Schaffer collateral(CA3→CA1) 단발 uEPSP | `04_experiments/Ex2_schaffer` | 05페이지 | ✅ **uEPSP 0.43mV·PPR 2.11·τ9.95ms (Sayer1990 추세일치)** |
| Ex2b | 연결 검증 매트릭스 (전 경로 2세포 uPSP/PPR) | `04_experiments/Ex2b_connection_matrix` | 커넥톰 UI(원형+매트릭스) 완성, 132경로·5클래스 | 🔄 UI ✅ · 2세포 벤치 대기 |
| Ex3 | SC I-O + 억제 차단 (단발) | `04_experiments/Ex3_io_inhibition` | ✅ **gradual I-O(발화 1.2→34.6%·fEPSP 10배)** · 단발 억제차단 무효=피드포워드 타이밍(정답) · 표+I-O곡선+3D UI(저세기/포화) | ✅ 완료 |
| Ex3b | SC 페어펄스/train — 억제 동역학 | `04_experiments/Ex3b_paired_train` | TBD | ⬜ |
| Ex4 | fEPSP 계산기(LSA) | `04_experiments/Ex4_fepsp` | TBD | ⬜ |
| Ex4b | MEA 3층 영상법(MoI) 밴드 | `04_experiments/Ex4b_mea_band` | TBD | ⬜ |
| Ex4c | CSD/kCSD 분석 + 정답 검증 | `04_experiments/Ex4c_csd` | TBD | ⬜ |
| Ex5 | theta 변조 입력 + PAC | `04_experiments/Ex5_theta_pac` | TBD | ⬜ |
| Ex6 | 내측중격(MS) theta | `04_experiments/Ex6_ms_theta` | TBD | ⬜ |
| Ex7 | ACh 신경조절 | `04_experiments/Ex7_ach` | TBD | ⬜ |
| Ex8 | LTP/LTD(칼슘 가소성) | `04_experiments/Ex8_ltp` | TBD | ⬜ |
| Ex9 | 실측 MEA 대조(최종) | `04_experiments/Ex9_realdata_mea` | TBD | ⬜ |
| Ex10 | STDP 곡선(Wittenberg 2006) | `04_experiments/Ex10_stdp` | TBD | ⬜ |
| Ex11 | cholinergic theta 위상의존 양방향 가소성(Huerta & Lisman 1995) | `04_experiments/Ex11_chol_theta_plasticity` | TBD | ⬜ |
| Ex12 | **인터랙티브 SC 자극 워크벤치 (UI)** — 자극 설계 → 스파이크·fEPSP 리플레이 | `04_experiments/Ex12_ui_workbench` | TBD | ⬜ |

## 공용 도구
- fEPSP 순방향 3기법: `lib/mea_forward.py` — **PSA**(점원)·**LSA**(선원, Holt&Koch 1999)·**MoI**(3층 영상법, Ness 2015). 전극 **E1·E2·E3(SO·SP·SR) 모두 기록**, SC는 직접 시냅스 자극(locus=E3 SR위치 중심).
- 장기가소성 mod: `../shared/mechanisms`(GBPlasticity류) 참조 — Ex8·Ex10 공유.

## 최종목표 사슬
Ex4·Ex4b(fEPSP) → Ex8·Ex10(가소성) → **Ex9(실측 대조)**.

## Ex2 상세 — Schaffer collateral 단발 uEPSP 특성화 (다음 실험)
**목표**: 단일 SC 섬유 활성 → CA1 추체의 **단발 uEPSP**(진폭·지연·10-90% 상승시간·감쇠τ) + **페어펄스(PPR)** 를 미세슬라이스 커넥텀 맥락에서 재현. 04 벤치·Sayer 1990 실측과 대조.
- **단위 검증**(Ex2) → 집단 I-O(Ex3) → 관측량 fEPSP(Ex4) 순서의 첫 단계. Ex2가 맞아야 Ex3 신뢰.
- **방법**: 커넥텀(`sc_synapses`·`sc_fibers`)에서 **SC 섬유 1개 선택** → 그 섬유가 접촉하는 추체 타깃·시냅스 추출. uEPSP는 역치하(subthreshold)라 타 세포로 전파 안 함 → **타깃 세포 + 해당 시냅스만 빌드**(소형·빠름, 전체망 불필요). 형태·시냅스 위치는 실제 커넥텀 그대로.
- **기록(신규)**: 대표 세포 **Vm** — 소마 + 시냅스가 놓인 수상돌기 세그먼트 → EPSP가 수상돌기→소마로 감쇠·전파되는 것 관찰. (지금까지 스파이크만 저장 → Ex2부터 Vm 기록 추가)
- **프로토콜**: ① 단발(1 스파이크) → uEPSP. ② 페어펄스 ISI 50ms → PPR=EPSP2/EPSP1 (SC→PC 촉진성 → >1 기대). 확률시냅스(Nrrp>1)라 **다수 시행 평균**.
- **검증지표**: 소마 uEPSP 진폭(mV)·지연(ms)·상승시간·감쇠τ · PPR. **대조**: 04 벤치(Sayer 1990) · gsyn 규칙(HippocampusHub).

## Ex2b 상세 — 연결 검증 매트릭스 (전 경로 2세포 uPSP/PPR) (설계 2026-08-25)
**목표**: Ex2(SC→PC 한 쌍)를 **커넥텀 전 경로로 일반화** — 각 (pre-mtype→post-mtype) 연결의 uPSP/uIPSP·PPR을 격리 측정해 **시냅스 파라미터(U/D/F/gsyn)가 클래스별로 올바른 거동**을 내는지 검증. 시뮬 장점: 문헌은 핵심 경로만이지만 우리는 **전수 검증** 가능.
- **범위(실측 커넥텀)**: 내부 **132 경로**(pre-mtype→post-mtype, 실제 배선 598,204 연결) + SC 입력 11경로. **5 시냅스 클래스 전부**(E1·E2·I1·I2·I3) 덮음. 분포 E1:4·E2:7·I1:7·I2:110·I3:4.
- **방법 = 진짜 2세포**: 내부 연결은 pre·post 둘 다 빌드 → **pre를 IClamp로 발화**(단발/페어펄스) → post Vm 기록. SC는 CA3=외부라 **VecStim**(=Ex2). 커넥텀의 실제 접촉 시냅스·수상돌기 위치 그대로 사용.
  - *개념*: 시냅스는 스파이크 "이벤트"만 보므로 응답(uPSP/PPR)은 VecStim=실세포 동일. 진짜 2세포는 **전시냅스 발화(흥분성·AP)까지 통합 검증**하려는 선택.
- **측정**: post uPSP/uIPSP 진폭·부호·지연·상승/감쇠τ·**PPR**·실패율 + pre AP. 확률방출(Nrrp)→다시행 평균.
- **판정**: 규칙 U/D/F 예측(촉진/억압) vs 실측 거동 일치. 예: PC→OLM 강촉진(E1), PVBC→PC 억압(I2), SC→PC 촉진(=Ex2 PPR 2.11).
- **UI(완성 2026-08-25)**: `04_experiments/Ex2b_connection_matrix/ui/` — **원형 커넥토그램**(`connectome_circular.html`; 방향 화살표·클래스 색·연결수 굵기·노드 호버) + **연결 매트릭스**(`connectome_matrix.html`; pre세로×post가로 히트맵). 빌더 `03_network/3_run/ex2b_connectome_tpl.html`·`ex2b_matrix_tpl.html`, 데이터 `scratch/connectome_graph.json`.
- **mechanism 일관성**: 흥분 `GBPlasticityStpProbSyn`(Ex2·망 동일), 억제 `ProbGABAAB_EMS`, gsyn 내부=출처·SC=Ex2보정(0.8nS). 출처 HippocampusHub/Kohus 2016.

### Ex2b 확장 — 벤치 프로토콜 + 시각화 3종 (2026-08-25)
**벤치 프로토콜**(`ex2b_bench.py`, 2세포 격리, ex3_io 배치로직 재사용):
- **다중 ISI STP**: 페어펄스 ISI [20/50/100/200]ms → PPR(ISI) 곡선. 고정폭 창(WFIX)에서 방향인식 peak로 측정(a1 ISI독립·음수아티팩트 방지).
- **⭐ 측정 = 시냅스 컨덕턴스 g(t) 직접기록**(08-26 확정): 후세포 Vm/전류가 아니라 **시냅스 g(E: AMPA+NMDA, I: GABA_A)를 직접 기록·합산**. g는 시냅스 모델의 직접 출력(방출확률×STP×수용체)이라 **후세포 공간클램프·구동력·클램프 artifact 전부 우회**, E·I 동일하게 강건. PPR=g2/g1(후세포 인자 상쇄로 PSP-PPR과 사실상 동일), kinetics=g파형.
  - *경위*: rest에선 IPSP 무구동력→전압클램프(VC) IPSC 시도했으나 **SEClamp가 홀딩전류 artifact 측정(다른 경로 u1 동일값)** → 실패. **g 직접기록으로 전환해 해결** — 검증: PC→PC PPR 0.20(억압·회복)·잠복2.5ms·τ2ms(AMPA), PVBC→PC 0.58(억압), Ivy→PC 1.40(촉진)·τ8.2ms(GABA) 전부 물리적 정확.
  - 단위: g는 **nS**(절대 mV 아님) → 문헌 절대진폭 비교는 별도(흥분은 Ex2 0.43mV 활용). 매트릭스는 PPR로 색칠하니 무관.
- **⭐ Train/주파수 필터링**(신규): 20Hz 8펄스 train + 주파수별(5/10/20/40Hz) 정상상태 → 억압=저역통과·촉진=대역통과. TM의 상징 실험(Markram/Tsodyks).
- **kinetics**: 잠복·상승(10-90%)·감쇠τ. **morph3d 모드**(`--morph3d`): 세그먼트 Vm(t)+시냅스 전류 3D 기록(대표경로).
**시각화 3종**(예시=Tsodyks-Markram 이론 예측, 실측 교체):
- **A 매트릭스** `ex2b_matrix[_예시].html`: pre×post PPR 히트맵, **대표조합 6개 강조**(SC→PC·PC→PC·PVBC→PC·PC→OLM·CCKBC→PC·Ivy→PC, 문헌참조). 클릭→**STP곡선(실측 vs TM점선)·Train응답·대표파형·kinetics·지표표**.
- **B 3D 쌍** `ex2b_pair3d_예시.html`: pre+post 형태 **전압전파 시간재생 + 시냅스 전류 glow**, 재생커서와 **동기화 지표**(pre소마·전류·post소마), 속도조절(0.1~2×).
- **C 분석 6패널** `figures/ex2b_analysis_예시.png`: PPR-vs-U·클래스별·표적특이STP·STP곡선·**TM-vs-실측 산점도**·**Train 주파수필터링**.
- 빌더: `build_ex2b_results_ui.py`(실측 A) · `gen_ex2b_{example,analysis,morph3d_example}.py`(예시). 템플릿 `ex2b_{results,pair3d}_tpl.html`.
- **누락 조사 결과**: TM 표준실험 중 **train/주파수필터링이 빠져 추가**. 남은 것: CV·실패율(시행별), 장ISI 회복(500/1000ms).
- **의존/일정**: 전체망 조립 불요(2세포 소형). **Ex3 메모리 풀리면 132쌍 순차 실행**(한 쌍씩, OOM 방지) → 예시 자리에 교체. morph3d는 대표경로만.

## Ex3 상세 — SC I-O + 억제 차단 (설계 확정 2026-08-24)
**목표**: basal 시냅스 전달의 **I-O 곡선**(실측 MEA 포맷)을 재현. **y=fEPSP slope, x=fiber volley(발화 섬유 진폭)**의 전달함수.
- **세기축의 정체**: 실측에서 자극 전류↑ = **더 많은 SC 축삭 모집**(recruitment). fiber volley 진폭 ∝ 발화 섬유 수. 전도도(gsyn)는 생물물리 상수 → **세기축 아님**. ⇒ 우리 세기축 = **발화 섬유 비율**.
- **핵심 결정**:
  - **locus(자극 영역) 고정** — 전극 안 옮김. 직경 5종 × 세기 5종(=25) 방식 폐기.
  - 세기 = locus 안에서 **발화시키는 섬유 비율** 5단계 `[10,25,50,75,100]%` (locus 근처 축삭부터 문턱 넘음 → 범위 고정, 밀도만 변화).
  - **gsyn 0.8 nS 고정** (Ex2 검증값). 전도도 스케일 안 씀.
  - ×2: **정상 / 억제 차단**(GABA_A off, bicuculline 대응) ⇒ **총 10회**. 전체망 **1회 조립 재사용**(재빌드 0).
- **측정**: 전극 E1·E2·E3 **fEPSP slope**(LSA/MoI) · 발화 세포% · **반응 구름 반경**(세기↑ 시 발화 뉴런이 locus에서 퍼지는 범위 = 출력).
- **예상**(문헌): 정상 = 포화형 상승, 억제차단 = 더 가파르고 높음(탈억제). r=100%에서 발화 39%(Ex1 volley 앵커).
- **예보 UI**: `03_network/3_run/ex_forecast.html`(Ex3 카드 I-O) · `ex3_recruit3d.html`(실제 커넥텀 3D 모집 구름, `build_ex3_recruit3d.py`). 실행 후 실측으로 대체.

### Ex3 실측 결과 — 포화판(10~100%) · 2026-08-25 (E3 전극 버그 정정 08-26)
전체망(5610세포·5.9M시냅스) w=0 중앙자극. E3(SR) fEPSP 지표 (`ex3_metrics_saturated.md`, 생성: `ex3_metrics_table.py`).
> ⚠️ **정정(08-26)**: 이전 표는 `ex3_metrics_table.py`가 `"E3(SR)"` 매칭 실패로 **E2(SP, 집단스파이크)** 를 읽었었음. 전극 접두어매칭+baseline차감 수정 후 아래가 올바른 **E3(SR)** 값. (라이브 로그 fEPSP E3는 원래 정확했음.)

| 세기(volley%) | slope N (µV/ms) | slope B | tpeak N (ms) | tpeak B | peak N (µV) | peak B | 발화 N | 발화 B |
|---|---|---|---|---|---|---|---|---|
| 10 | -1170 | -1232 | 4.20 | 4.40 | -1465 | -1481 | 2173 (38%) | 2151 (38%) |
| 25 | -2916 | -2949 | 4.10 | 3.90 | -2187 | -2287 | 3877 (69%) | 3906 (69%) |
| 50 | -5254 | -5993 | 2.10 | 2.10 | -3131 | -3187 | 5268 (93%) | 5259 (93%) |
| 75 | -4746 | -4367 | 3.70 | 2.10 | -2490 | -2519 | 5475 (97%) | 5480 (97%) |
| 100 | -6727 | -7247 | 3.60 | 3.60 | -2788 | -2862 | 5521 (98%) | 5524 (98%) |

**소견**:
- **tpeak 세기의존 단축 4.2→3.1ms**: 약자극=순수 시냅스 sink(늦은 peak) → 강자극=**집단스파이크**(SP 소마 Na sink) 중첩으로 빨라짐. 교과서적 I-O 소견.
- **S자 포화**: 10%에서 이미 38% 발화 → 50%에서 93%. w=0 중앙자극이 강해 역치 위 포화.
- **억제차단(block) 고세기서 무효**(normal≈block): (a) 단시냅스 흥분 ~3ms가 이중시냅스 억제 ~5–8ms를 앞섬, (b) 포화 천장. ⇒ **저세기 재실행 필요**(아래).
- **발화 '띠'(band)**: 10% 발화세포 분포 u(장축)std146·r(SR)std38·w(두께)std67 = **장축으로 길고 SR에 얇은 라미나**. SC 섬유가 u축 다발(`sc_fibers.npz` by-construction) → 초점자극이 라미나 활성. 방향의 해부학적 타당성은 06(실제 궤적)에서 검증.

**창발 반응 vs 문헌 스코어카드** (우리가 코딩하지 않았는데 나온 것):

| 창발 반응 | 관찰 | 문헌 | 판정 |
|---|---|---|---|
| SR(E3) sink 최음성 | E3 −3~−10mV | SC→SR 수상돌기 sink | ✅ |
| 세기↑ 집단스파이크·tpeak 단축 | 4.2→3.1ms | 교과서 I-O | ✅ |
| S자 포화 I-O | 38→93% | Andersen 등 | ✅ |
| 단시냅스 잠복 ~3ms | tpeak 3~4ms | SC 단시냅스 | ✅ |
| SR 라미나 얇은 활성 | r std 38 | SC=SR 표적 | ✅ |
| 억제차단 고세기 무효 | N≈B | 포화+타이밍 | ⚠️ 저세기 검증중 |
| 활성 띠 방향(u) | u로 김 | SC 종/횡 성분 | ❓ 06 검증 |

- **3D UI**: `04_experiments/Ex3_io_inhibition/ui/ex3_io_3d.html`(포화판) · `ex3_io_3d_low.html`(저세기).

### Ex3 저세기 결과 — 최종 (0.5~8%, 50~800섬유) · 2026-08-26 완료
전체망 10조건(normal/block×5세기). E3(SR) fEPSP peak가 라이브 로그와 일치(검증). `ex3_metrics_lowvolley.md`.

| 세기 | slope N | slope B | tpeak N | tpeak B | peak N (µV) | peak B | 발화 N/B | 억제뉴런 N/B |
|---|---|---|---|---|---|---|---|---|
| 0.5% | -114 | -182 | 5.4 | 5.4 | -89 | -95 | 58/58 (1.2%) | 0/0 |
| 1% | -232 | -258 | 4.3 | 4.2 | -162 | -160 | 161/162 (3.2%) | 1/1 |
| 2% | -329 | -512 | 5.0 | 4.5 | -366 | -439 | 450/438 (8.9%) | 2/1 |
| 4% | -631 | -1216 | 4.7 | 4.3 | -697 | -731 | 1006/994 (19.6%) | 20/17 |
| 8% | -1532 | -1227 | 4.3 | 4.4 | -1303 | -1240 | 1815/1788 (34.6%) | 70/57 |

**⭐ 표준 자극 세기 확정(2026-08-26) = volley 8%** (잠정): 실측 실험실 관례처럼 I-O로 표준 세기를 정함. 8% 근거 — 최대 fEPSP(~3.1mV)의 **~42%**(LTP 표준 ~50% 근접, 증가·감소 둘 다 관측 여지) · **억제뉴런 70개 engage**(E-I 동역학 테스트 가능) · gradual(비포화). **→ 후속 실험(Ex3b·Ex4·Ex5·Ex8 LTP)의 기본 자극 세기로 사용.** 4·8% 반복평균 검증은 Ex2b 배치 후 큐잉(fEPSP 런 ~60분·5코어라 Ex2b와 충돌).

**최종 결론**:
- **① gradual I-O 재현 ✅**: 발화 1.2→34.6%, fEPSP peak −89→−1303µV(10배 단조), 포화 탈출. 흥분 전달 정확.
- **② 단발 억제차단 = 전 세기 무효 ✅(정답)**: block이 발화를 어디서도 못 늘림(block≤normal). fEPSP·slope 차이는 양방향=단발 잡음. 억제뉴런은 4%(20개)·8%(70개)로 켜지는데도 효과 없음.
- **③ 원인 = 타이밍(피드포워드)**: 이중시냅스 억제(~5-8ms)가 단시냅스 스파이크(~3ms) 뒤 도착 → 단발론 원리적으로 못 봄 = **억제회로가 피드포워드로 올바르게 배선됨의 증거**.
- **→ 억제 기능검증 = Ex3b(페어펄스/train)** 가 정답. (원하면 4·8%만 5회 반복평균으로 미세차이 확인 — 발화만이면 ~10시간.)
- **환경**: WSL conda ca1sim(`mpirun -np 5`, cwd=scratch/mechbuild), 집합통신 mpi4py(우선)+pc(폴백). 조건당 psolve ~60분(fast_imem 오버헤드), 단발.

## Ex3b 상세 — SC 페어펄스/train, 억제 동역학 (설계 2026-08-25)
**목표**: 단발 I-O(Ex3)에서 억제가 안 보인 이유(흥분 단일시냅스 ~3ms > 억제 이중시냅스 ~5–8ms, 강자극 포화)를 넘어, **연속 자극으로 억제의 시간 동역학**을 측정 = **E-I 회로 기능 검증**.
- **근거**: 억제 회로는 존재(GABA 4.79M·억제뉴런 발화 확인). 단발은 억제가 첫 스파이크에 늦어 무효 → **연속이면 앞 자극의 GABA(~10–40ms 지속)가 다음을 억제** → 억제 드러남.
- **① 페어펄스**: 2 볼리, ISI [20, 50, 100, 200]ms → **PPR = fEPSP2/fEPSP1** (또는 발화2/발화1). 단ISI에서 억제 우세(감소) vs 촉진. GABA off → 억제성 감소분 사라짐.
- **② train**: theta(8Hz)·gamma(40Hz)·HFS(100Hz) 열 → **억제 누적·주파수의존 gating**. GABA off → **탈억제 epileptiform 다중 population spike**(Schwartzkroin & Prince 1980).
- **측정**: PPR(ISI별) · train 응답 감쇠/증강 · 탈억제 bursting · fEPSP + 발화. 정상 vs GABA off.
- **검증 기준**: **여기선 억제 차단 효과가 뚜렷해야 정상** (단발과 달리 시간중첩으로 억제가 작동). 안 나오면 억제 배선/강도 점검.
- **의존/연결**: Ex3(단발) 후. Ex5/Ex6(theta)·Ex2(페어펄스 인프라)와 공유. 세기는 Ex3 역치값 사용.

## Ex4·Ex4b·Ex4c 상세 — fEPSP forward → CSD → 검증 (설계 2026-08-24)
**사슬:** 세그먼트 막전류 → [forward] → 전극 fEPSP → [inverse=CSD] → sink/source 밴드.

- **Ex4 (fEPSP forward)**: `lib/mea_forward.py` PSA/LSA(무한 균질 매질). 세그먼트 막전류(fast_imem, `lib/fepsp_record.py`) → 전극 V(t). ✅ 인프라 구축·검증 완료(2026-08-24).
  - **소규모 검증(50세포, `ex_fepsp_test.py`)**: locus 근처 PC 50개 SC volley → 막전류 fEPSP.
    - 크기 **~40µV/50세포** = 실측 스케일 정확(전체망 ~2천세포면 ~mV). biphasic **population spike**(~3ms) + 느린 시냅스 EPSP 꼬리.
    - 발화 시: E1(SO) −40.5 · E2(SP) **−42.9** · E3(SR) −28.6 µV — 셋 다 음성, **SP 최대 음성 = population spike(소마 Na sink) 지배**.
    - 역치하(gscale 0.02): 0발화, 필드 0.2µV·세 전극 거의 동일 → **far-field**.
  - **⚠️ 발견**: 자극 locus와 기록전극 E3 사이 **~400µm 오프셋**(E3 stim→rec 변경 잔재). 50세포 far-field 뭉갬은 **선별 인공물**(locus 근처만 빌드)일 가능성 → **전체망으로 판정 중**(Ex1 volley + fEPSP, `mpi_baseline.py --fepsp`).
  - **⚙️ 성능**: fEPSP 기록의 `Vector.record`가 O(n²)(전체망 705k세그/랭크) → **세그먼트 stride 서브샘플**(FEPSPRecorder stride, W×보정)로 해결.
  - **UI**: `04_experiments/00_overview/ui/fepsp_3d.html`(아티팩트 5378691e) — 세그먼트 막전류(sink/source 색)+3전극 fEPSP+막전류그래프+속도/전극/시냅스/자동회전/catchment 토글.
- **Ex4b (MoI forward, 경계보정)**: 슬라이스 3층(식염수/조직/유리 MEA)의 경계 반사를 **method of images**(Ness 2015)로 보정한 forward. 무한매질 가정의 오차를 잡아 **실측 슬라이스와 맞는** fEPSP·깊이 프로파일("밴드") 산출. `mea_forward`에 MoI 옵션 추가 예정.
- **Ex4c (CSD/kCSD 역분석 + 정답 검증, 신규)**: 전위 → sink/source **역추정**.
  - **naive CSD**: `-σ·∂²V/∂z²` (등간격 전극 2차미분). 3전극(SO/SP/SR, 간격 200µm)이면 가운데(SP) 한 점.
  - **kCSD**(Potworowski 2012, `kCSD-python`): 희소·불규칙 전극에서 커널 정규화로 **연속 CSD 재구성**(보강). 경계인식 변형은 MoI와 결합.
  - ⭐ **시뮬 정답 검증(실측 불가, 우리만 가능)**: ① **정답 CSD** = 실제 막전류를 깊이별 binning(전극 불필요) · ② 조밀 가상전극 → forward → naive CSD · ③ 3전극 → kCSD 보강. **②③를 ①과 대조** → forward+CSD 정당성 + **kCSD 희소전극 보강 정확도 정량**. `mea_forward`가 임의 위치 fEPSP를 주므로 가상전극 무한 증설 가능.
  - **근거**: Nicholson & Freeman 1975, Pettersen & Einevoll(CSD), Potworowski et al. 2012(kCSD), Ness et al. 2015(MoI). **방법론 기여 가능**(상세모델 정답 기반 CSD 방법 벤치마크).
  - **부호**: sink(전류 유입)→음성 CSD, source→양성.

## Ex8 상세 — LTP/LTD 유도 프로토콜 (가능성 확인 ✅)
- **인프라**: `../shared/mechanisms/GBPlasticity{Syn,StpSyn,StpProbSyn}.mod`(Graupner-Brunel 칼슘 가소성). c(t)→ρ 이중안정→w=w0+ρ(w1−w0).
- **Ex8-HFS (LTP)**: 100 Hz × 1초 테타너스 → LTP. 근거 Bliss & Lømo 1973; Bliss & Collingridge 1993. (짧아 CPU-MPI 3-4일 예산 가능)
- **Ex8-LFS (LTD)**: 1~3 Hz × 7~15분 → LTD. 근거 Dudek & Bear 1992; Mulkey & Malenka 1992. (긴 런 420~900초 → 데스크톱 불가: 클라우드 클러스터/프로토콜 단축/Jaxley 재구성 필요. **A6000 GPU는 이 모델 가속 못 함 — 2026-08-22 확정**)
- **Ex8-TBS**: theta-burst(기존). → 유도 프로토콜 3종(HFS·LFS·TBS) + Ex10 STDP.
- **측정**: 정규화 fEPSP slope(% baseline). **주의**: Graupner 파라미터 정량 검증 필요(방향은 재현 기대).

## Ex11 상세 — cholinergic theta 위상의존 양방향 가소성 (Huerta & Lisman 1995 재현)
- **프로토콜**: 카바콜 유발 theta 중 **단일 버스트(4펄스·100 Hz)**를 theta **위상**에 정렬 → **peak=LTP · trough=LTD**(이전 강화 시냅스) · heterosynaptic LTD.
- **의존**: Ex5/Ex6(theta) + Ex7(ACh·mAChR) + Ex8(Graupner mod) 결합 = **캡스톤**.
- **검증**: 위상별 부호(peak +, trough −) · NMDA + muscarinic 의존.
- **근거**: Huerta & Lisman 1995, *Neuron* 15(5):1053–1063 (PMID 7576649).

## Ex12 상세 — 인터랙티브 SC 자극 워크벤치 (UI)
**목표**: 사용자가 SC 자극을 **설계**하고, 그 응답(스파이크 + fEPSP)을 **시각적으로** 확인하는 인터랙티브 도구. 실측 프로토콜 탐색·데모·교육용. (실시간 계산 아님 — 상세 전체망은 ~10⁴배 느려 실시간 불가 → **설계 → 배치 계산 → 슬로모션 리플레이** 구조.)

**① 사용자 입력 (자극 설계)**:
- **위치(locus)**: SC 자극 중심 xyz(슬라이스 뷰에서 클릭) + 반경 → 활성 SC 섬유 선택 (`dist < R`)
- **파형 형태**: 단일 펄스 / 구형파(square) / 이상성(biphasic) / 버스트(N펄스) / 사인 / 사용자정의
- **길이**: 펄스폭 + 총 지속시간(bio ms)
- **주기/주파수**: 반복 Hz — 단발·theta(4~12Hz)·HFS(100Hz)·LFS(1~3Hz)
- **진폭/세기**: 활성 섬유 수 or weight

**② 처리 (백엔드 배치)**: 자극 config → 활성 섬유별 VecStim 시각벡터 생성 → 전체망 CPU-MPI sim(`mpi_baseline.py` 확장) → 스파이크(gid·시각) + fEPSP(전극 E1·E2·E3, `lib/mea_forward.py` LSA·MoI). **소요시간(compute wall-clock) 기록·표시.**

**③ 출력 (UI 리플레이, HTML/WebGL 아티팩트)**:
- **3D 발화 전파** 애니메이션 (자극 locus에서 퍼짐, E/I 색분리)
- **래스터** + **PSTH**
- **fEPSP 층별 파형** (E1 SO·E2 SP·E3 SR) — 실측 대조 신호
- **슬로모션 시간 슬라이더** (몇만배 느리게, 재생/일시정지/속도조절)
- **소요시간 표시** (bio X ms 계산에 wall-clock Y분)

**성격·경로**: 실시간 불가(설계→계산→리플레이). 여러 자극조건 미리 계산해두면 UI에서 골라 즉시 재생 = 준실시간 체감. **미래**: 축약/surrogate(Jaxley·GPU)면 진짜 실시간 클릭→발화.
**의존**: Ex1(구동)·Ex4(fEPSP). **명명**: 전극 E1/E2/E3(기록)에서 fEPSP, SC는 직접 시냅스 자극(locus).

---

_각 실험 상세는 진행 시 이 문서에 절을 추가한다._
