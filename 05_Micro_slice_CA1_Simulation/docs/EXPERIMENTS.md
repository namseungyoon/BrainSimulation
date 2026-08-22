# 실험 레지스트리 (Ex1~Ex12) — micro-slice CA1

> ⚠️ **명명 규칙 (2026-08-22 확정)**: **전극 = E1(SO)·E2(SP)·E3(SR)** · **실험 = Ex1~Ex11**. (과거 실험을 E1~E11로 쓰다 전극 E1~E3와 충돌 → 실험에 `Ex` 접두.)
> 단일 색인: **Notion번호 ↔ 코드경로 ↔ 그림**. 실험이 카테고리에 분산돼도 여기서 한눈에 추적.
> 공통 양식(각 실험): 목표 / 방법·입력 / 검증지표 / 결과·상태 / 근거(논문X→우리Y) / 한계·주의.
> 완료는 결과까지, 미실행은 "(예상·미실행)".

| ID | 실험 | 폴더 | Notion | 상태 |
|---|---|---|---|---|
| Ex1 | baseline 발화율·구동 검증 | `04_experiments/Ex1_baseline` | 05페이지 | ✅ volley(39%) · 무자극 진행중 |
| Ex2 | Schaffer collateral(CA3→CA1) | `04_experiments/Ex2_schaffer` | TBD | ⬜ |
| Ex3 | SC I-O + 억제 차단 | `04_experiments/Ex3_io_inhibition` | TBD | ⬜ |
| Ex4 | fEPSP 계산기(LSA) | `04_experiments/Ex4_fepsp` | TBD | ⬜ |
| Ex4b | MEA 3층 영상법(MoI) 밴드 | `04_experiments/Ex4b_mea_band` | TBD | ⬜ |
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
