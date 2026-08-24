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
| Ex3 | SC I-O + 억제 차단 | `04_experiments/Ex3_io_inhibition` | TBD | ⬜ |
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
