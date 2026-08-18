# 05 Micro-slice CA1 Simulation — 프로젝트 계획서

> **목표**: 해마 CA1의 **마이크로 슬라이스**(MEA 전극 2~3개가 들어가는 최소 조직)를 in silico로 구성하고,
> **MEA fEPSP → LTP/LTD → 실측 데이터 대조**까지 재현한다.
> **방법**: Romani(2024) CA1 아틀라스·커넥텀 **파이프라인**을 기반으로, 창 안의 **모든 뉴런을 전세포 완전형태 biophysical 모델**로 인스턴스화.
> **원칙**: 01_Like_slice 산출물 재사용 금지 — **05에서 백지 시작**. `../shared`·`../Models`는 경로 참조(복사 금지).

시작: 2026-08-13.

---

## 1. 개요 · 목적

- **마이크로 슬라이스 정의**: like-slice(17,647세포·SP밴드 2305×468µm)보다 훨씬 작은, MEA 전극 2~3개(200µm 간격)+여유가 들어가는 최소 CA1 조직.
- **최종 목표 사슬**: 조직 구축 → 구동 → **fEPSP 계산(E4/E4b)** → **LTP/LTD(E8)·STDP(E10)** → **실측 MEA 대조(E9)**.
- **트랙 관계**: 01_Like_slice(대규모, Romani 재사용)·02_full_scale(Bezaire)와 별개. 05는 **소규모·전세포 완전모델**로 백지 재구성.

## 2. 작업 규칙 (반드시 준수)

- **코드보다 계획 우선.** 단계(0→9)·실험(E1→E10)을 **하나씩** 진행하고, 각 단계 끝에 **✅검증(그림/수치)**을 사용자와 확인한 뒤 다음으로.
- 전체 파이프라인을 미리 자동 생성하지 않는다.
- **Notion + GitHub 동시 갱신** (결과·그림·보고서가 바뀌면 같은 흐름에서 둘 다). 05용 Notion 페이지는 신설 예정.
- **정직성**: 튜닝값 ≠ 측정값. 완료는 결과까지, 미실행은 계획·근거만(결과 날조 금지). 소스/실행 로그와 대조해 수치 확정.

## 3. 기하 (확정)

| 항목 | 값 | 비고 |
|---|---|---|
| footprint | **800 × 500 µm** | 800µm 축 = 층관통(SO/SP/SR), 500µm = 수직 |
| 두께 | **400 µm** | 일반 in vitro 급성 슬라이스 |
| MEA | 전극 간격 200 µm · 직경 10 µm · 개수 2~3 | 800µm 축에 사방 200µm 여유 |
| 예상 세포 수 | 약 3,000개(대부분 PC) | 정확값은 Stage5 배치에서 확정 |

- 크기 근거: fEPSP 기능적 기여 반경 ≈ SC 자극 동원 반경 ~200µm → 전극 스팬 400µm + 사방 200µm 여유 = 800µm. 자세한 근거·의사결정은 [docs/DECISIONS.md](docs/DECISIONS.md).
- ⚠️ fEPSP는 **광역 적분 신호**(전극당 유효기여 PC ≫ 국소). 마이크로 조직은 **절대 진폭이 축소** → **정규화 fEPSP·상대 LTP slope**로 비교(스케일 보정). 상세: [docs/DECISIONS.md](docs/DECISIONS.md).

## 4. 폴더 구조 규약

- 최상위 = **의미 카테고리 1~4**(인덱스): `01_tissue/` · `02_neurons/` · `03_network/` · `04_experiments/`.
- 각 카테고리 내부 하위폴더 = **단계(1-based 순번)**. **표기 통일: 「카테고리 N (이름) · M단계」** — 예: 카테고리1(tissue)·1단계 inspect · 카테고리1·2단계 bbox · 카테고리2(neurons)·2단계 placement. (카테고리를 "단계"라 부르지 않는다.) 실험은 E1~E11.
- 지원(인프라, 번호 없음): `config/`(기하·파라미터 YAML 단일 출처) · `docs/` · `lib/`(번호 없는 import 모듈) · `data/`(원자료, gitignore) · `env/`(빌드·구동 런처, 추적) · `scratch/`(일회성, gitignore).
- **번호 import 제약**: import되는 재사용 코드는 전부 `lib/`. 단계 폴더(`01_tissue/1_bbox` 등)는 실행 스크립트만 두고 import하지 않는다.
- **스크립트는 최상위에 두지 않는다** — 런처=`env/`(추적), 일회성=`scratch/`(제외).
- **추적 규칙**: 코드 + 결과 PNG만(~80MB). 대용량 원자료·산출물·중간 npz·로그는 `.gitignore`.

## 5. 필요 데이터 (Harvard Dataverse DOI 10.7910/DVN/TN3DUI)

| 파일 | 용량 | 내용 | 상태 |
|---|---|---|---|
| circuit.zip | 2.53 GB | SONATA nodes = 456,380세포 배치 | ✅ 확보(통합 zip) |
| atlas.zip | 297 MB | 부피·층·방향장 NRRD | ✅ 확보 |
| morphology_library.zip | 2.49 GB | 세포별 실제 형태학 | ✅ 확보(전세포 완전모델용) |
| single_cell_model_library.zip | 0.7 MB | Romani e-model | ✅ 확보 |

- 다운로드된 통합 `dataverse_files.zip`(4.83GB) → `data/incoming/`. 압축 해제·배치 절차: [data/README.md](data/README.md), [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

## 6. 재사용 자산 지도

- **재사용 O**(복사 아님, 경로 참조): `../shared/common`(nrn_env·cell_loader·model_naming·plotstyle·corrections)·`../shared/mechanisms`(컴파일 채널·EMS 시냅스 mod)·`../Models`(BBP 20 단일세포 번들).
- **재사용 X**: 01_Like_slice의 파이프라인 코드·데이터 산출물(slice_cells.npz 등) → `lib/`에서 마이크로 창 전용으로 자체 작성.
- 상세 경로 지도: [docs/REUSE_MAP.md](docs/REUSE_MAP.md).

## 7. 구성 파이프라인 (Romani 0~9) — 단계별 계획

> 상태 표기: ⬜ 미착수 · 🔄 진행 · ✅ 검증완료. **각 단계는 시작 전 보고 → 실행 → ✅검증 → 보고.**

> 폴더 = 카테고리(01~04) + 1-based 순번. 아래 "Romani단계"는 논문의 과학적 단계(0~9) 참조용.

| Romani단계 | 폴더 | 목표 | 검증(V) | 상태 |
|---|---|---|---|---|
| 0 데이터준비 | `data/` + `01_tissue/1_inspect` | 압축해제·SONATA/atlas 구조·세포수 확인 | V0(N=456,378·E:I 89:11·층4·mtype12) | ✅ |
| 1 슬라이스 bbox | `01_tissue/2_bbox` | 마이크로 창 정의(층관통_v1 500×800×400µm)·전극배치·config 내보내기 | V1a(창·전극 확정) | ✅ |
| 2 아틀라스 전처리 | `01_tissue/3_atlas_prep` | Romani atlas 창 크롭(78×97×80) + 국소질의 lib(층·nd) | V2p(전극 층검증·대응그림) | ✅ |
| 3 좌표·방향 벡터화 | `01_tissue/4_vectorize` | 좌표(l/t/r)+방사 방향장 | V1b 방사벡터 수직 | ⬜ |
| 4 층 구분 | `01_tissue/5_layers` | SO/SP/SR/SLM 경계·두께 | V1c | ⬜ |
| 4b 세포 조성 | `02_neurons/1_composition` | 층별 m/e-type·밀도·E:I | V2a 밀도·E:I | ⬜ |
| 5 배치 | `02_neurons/2_placement` | 창 내 전 뉴런 추출·좌표 배치 | V2b | ⬜ |
| 5b me-model 매핑 | `02_neurons/3_memodel_map` | (m,e)→완전형태 모델 매핑(대표축소 금지) | V2c (m,e) 100% | ⬜ |
| 6 방향성 주입 | `02_neurons/4_orientation` | 평행이동+quaternion 회전 | V2d 길이불변 | ⬜ |
| 7 커넥텀 | `03_network/1_connectome` | 9클래스 거리의존 연결 | V3 수렴발산 | ⬜ |
| 8 시냅스 | `03_network/2_synapses` | Ecker Table3 EMS 시냅스 주입 | V4 PSP/CV/STP | ⬜ |
| 9 구동 | `03_network/3_run` | build→wire→drive→run(고정 dt 0.025) | V5 raster·E/I | ⬜ |

- 마이크로 창 특성: 작아서 **전세포 완전모델 인스턴스화 가능**. 단 **경계효과**(가장자리 세포 연결 절단) 주의.

## 8. 실험 트랙 (E1~E10)

> 상세 레지스트리(Notion번호↔경로↔그림)는 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md). 완료는 결과까지, 미실행은 계획만.

| ID | 실험 | 폴더 | 상태 |
|---|---|---|---|
| E1 | baseline 발화율·구동 검증 | `04_experiments/E1_baseline` | ⬜ |
| E2 | Schaffer collateral(CA3→CA1) | `04_experiments/E2_schaffer` | ⬜ |
| E3 | SC I-O + 억제 차단 | `04_experiments/E3_io_inhibition` | ⬜ |
| E4 | fEPSP 계산기(LSA) | `04_experiments/E4_fepsp` | ⬜ |
| E4b | MEA 3층 영상법(MoI) 밴드 | `04_experiments/E4b_mea_band` | ⬜ |
| E5 | theta 변조 입력 + PAC | `04_experiments/E5_theta_pac` | ⬜ |
| E6 | 내측중격(MS) theta | `04_experiments/E6_ms_theta` | ⬜ |
| E7 | ACh 신경조절 | `04_experiments/E7_ach` | ⬜ |
| E8 | LTP/LTD(칼슘 가소성) | `04_experiments/E8_ltp` | ⬜ |
| E9 | 실측 MEA 대조(최종) | `04_experiments/E9_realdata_mea` | ⬜ |
| E10 | STDP 곡선(Wittenberg 2006) | `04_experiments/E10_stdp` | ⬜ |
| E11 | cholinergic theta 위상의존 양방향 가소성(Huerta & Lisman 1995) | `04_experiments/E11_chol_theta_plasticity` | ⬜ |

- fEPSP 3기법(공용, `lib/mea_forward.py`): **PSA**(점원)·**LSA**(선원, Holt&Koch 1999)·**MoI**(영상법 3층, Ness 2015).
- E8/E10 장기가소성 mod(GBPlasticity류)는 `../shared/mechanisms` 참조(05에서 신규 작성 안 함).

### E8 유도 프로토콜 (LTP/LTD) — 가능성 확인됨 ✅
- **인프라 확인**: `../shared/mechanisms/`에 Graupner-Brunel 칼슘 가소성 mod 존재 — `GBPlasticitySyn.mod`·`GBPlasticityStpSyn.mod`·`GBPlasticityStpProbSyn.mod`. 칼슘 c(t) → ρ 이중안정 → w=w0+ρ(w1−w0). 고빈도=고칼슘→LTP, 저빈도=중칼슘→LTD. → **고전 유도 프로토콜 재현 가능.**
- **E8-HFS (LTP)**: **100 Hz 고빈도 자극 1초**(테타너스) → ρ UP. 근거 Bliss & Lømo 1973; Bliss & Collingridge 1993. 짧은 런(수 초).
- **E8-LFS (LTD)**: **1~3 Hz 저빈도 자극 7~15분** → ρ DOWN. 근거 Dudek & Bear 1992; Mulkey & Malenka 1992. **긴 런(420~900초 시뮬)**.
- **마이크로 슬라이스 이점**: 조직이 작아 **7~15분 LFS 장시간 시뮬레이션이 현실적**(full-slice에선 비현실적). 단 LFS는 여전히 무거워 GPU/CoreNEURON(`env/gpu`) 권장.
- **측정**: **정규화 fEPSP slope(% baseline)** — 절대 진폭 아님(마이크로 조직 스케일 한계, §3 참조). 실험 측정 관행과도 일치.
- **⚠️ 주의**: Graupner 파라미터는 특정 데이터 피팅값 → 방향(HFS→LTP·LFS→LTD)은 재현 기대, **정량 크기는 파라미터 검증 필요**. 기존 E8 TBS·E10 STDP와 함께 **유도 프로토콜 3종 체계**(TBS·HFS·LFS+STDP).

### E11 프로토콜 (cholinergic theta 위상의존 양방향 가소성) — Huerta & Lisman 1995 재현
- **목표**: 콜린성(카바콜 유발) theta 진동 중 **단일 버스트(4펄스·100 Hz)**를 theta **위상**에 정렬해 인가 → **peak=LTP, trough=LTD**(이전 강화 시냅스) 양방향 재현.
- **방법**: E5/E6(theta 생성) + E7(ACh·mAChR 조절) + E8(Graupner 가소성 mod) **결합**. 단일 버스트를 theta peak/trough에 위상 정렬해 SC에 인가.
- **검증**: peak 버스트→LTP · trough 버스트→LTD · heterosynaptic LTD. **NMDA + muscarinic 수용체 의존**.
- **근거**: Huerta & Lisman 1995, *Neuron* 15(5):1053–1063 (PMID 7576649).
- **⚠️ 의존**: E5/E6(theta)·E7(ACh 기계론 Im/KM+mAChR)·E8(가소성 mod) 선행 필요 → **캡스톤 통합 실험**. muscarinic 조절 구현 여부가 관건.

## 9. 실행 환경

- NEURON(conda `ca1sim` 기준). **⚠️ 이 머신은 like-slice 실행 머신과 다름** — env 경로·컴파일 산출물 재확인 필요(현재 사용자 `USER`).
- BBP EMS 시냅스는 cvode 비호환 → **고정 dt 0.025**. NetCon weight = nS(→ µS는 /1000).
- MPI/CoreNEURON GPU 런처는 `env/`. matplotlib 한글 폰트(맑은 고딕) 결자 주의(`g_hat`·`-`·`~` 사용).

## 10. 로드맵 · 마일스톤

1. **파이프라인 구축**(0~9) → 조용한 baseline 슬라이스.
2. **SC 경로 + fEPSP**(E2·E4·E4b) → MEA식 유발 fEPSP.
3. **LTP/LTD·STDP**(E8·E10) → 가소성 재현.
4. **실측 대조**(E9) → 정규화 비교·스케일 보정.

각 단계·실험 끝에 ✅검증 그림/수치로 하나씩 확인 후 진행.
