# like-slice CA1 시뮬레이터 — 프로젝트 계획서

> ⚠️ **이 프로젝트는 Ecker(2020) 논문 검증이 아니라**, 해마 슬라이스와 유사한 **like-slice CA1 시뮬레이터** 구축(별도 트랙)입니다.
> **최종 목적**: like-slice 구축 → **MEA 측정 재현 → LTP/LTD 재현 → 실측 데이터 비교**.
> **방식**: Romani 2024의 **아틀라스·배치 재사용**(불러오기) + **우리 검증된 23 me-model(뉴런)** + **우리 검증된 9클래스 시냅스(Ecker)**. 연결은 우리 규칙으로 구성.

---

## ⚠️ 작업 규칙 (반드시 준수)
- **아직 파이썬 파일을 생성하지 말 것.** 아래 "프로젝트 구조"의 파일 목록은 *최종 목표 구조*일 뿐, 미리 만들지 않는다.
- **사용자의 지시에 따라 단계(0 → 1 → … → 9)를 하나씩** 진행한다. 해당 단계에 도달했을 때만 그 단계의 코드를 작성한다.
- **각 단계 끝에 ✅검증(그림/수치)을 사용자와 함께 확인**한 뒤 다음 단계로 넘어간다.
- 한 번에 여러 단계를 앞질러 구현하거나 전체 파이프라인을 자동 생성하지 않는다.

---

## 프로젝트 구조 (최종 목표 — 미리 만들지 말 것)
```
like_slice_CA1/
├── PLAN.md                 # 이 문서
├── data/                   # circuit.zip·atlas.zip 압축해제 (대용량)
├── lib/                    # slice_io.py · mtype_map.py · morph_transform.py · atlas_network_lib.py
├── 0_inspect_data.py       # nodes.h5 구조·세포수 확인 (V0)
├── 1_extract_slice.py      # 슬라이스 bbox 추출 → slice_nodes.json (1·4b·5)
├── 2_build_slice.py        # me-model 매핑·로드·회전배치 → slice_cells.json (5b·6)
├── 3_slice_connectivity.py # pathway_class 연결 → slice_connectivity.json (7)
├── 4_run_and_analyze.py    # wire→drive→run→analyze (8·9)
├── 5_visualize_slice.py    # 3D 층·배치·방향 + 검증 그림
└── figures/
```

## 재사용 (기존 자산 — import)
- `shared/` : `common/{nrn_env, cell_loader, model_naming, plotstyle}`, `models/`(23 me-model + registry), `mechanisms/`(EMS·VecStim mod).
- `papers/01_Ecker2020_CA1_synaptic/04_network/network_lib.py` : `wire_synapses·add_external_drive·record_spikes·run_network·analyze_activity·spikes_to_arrays·_placement·_dendrites`.
- `.../05_paired_recording/pathway_map.py` : `pathway_class(pre_mtype, post_mtype) → 9클래스`.
- `.../03_synapses/params_table3.py` : `CLASSES`(9클래스 시냅스 파라미터).

## 환경 / 도구
ca1sim (h5py 3.16 · scipy 1.15.3 · numpy 2.2.6 설치됨). 추가: `pip install pynrrd`(atlas NRRD). 전체경로 `C:\Users\SYNAM-OFFICE\.conda\envs\ca1sim\python.exe`.

---

## 필요 데이터 (Harvard Dataverse DOI 10.7910/DVN/TN3DUI)
| 받기 | 파일 | 용량 | 내용 |
|---|---|---|---|
| ✅필수 | **circuit.zip** | 2.53 GB | SONATA nodes = 456,380 세포 배치(좌표·방향·m/e-type·층) |
| ✅필수 | **atlas.zip** | 297 MB | 부피·층·방향장(복셀 NRRD) |
| 🔶선택 | morphology_library.zip | 2.49 GB | Romani 형태(변이/대조) |
| 🔶선택 | single_cell_model_library.zip | 0.7 MB | Romani e-model |
| ❌불필요 | synapses/touches/projections.xz* | ~595 GB | 연결(우리가 Ecker로 생성) |
다운로드: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/TN3DUI → circuit.zip + atlas.zip 만 선택 → `data/` 에 압축해제.

---

## 단계별 계획 (Romani 순서, ✅=검증)
- **0. 데이터 준비** — 다운로드·압축해제·`nodes.h5` 구조 확인. ✅V0: 세포수≈456,380·필드(x/y/z·orientation·mtype·etype·layer).
- **1. 아틀라스 슬라이스 3D 준비** — atlas 불러와 bbox(예: 횡300×종300µm×전층) 정의. ✅V1a: 층두께(atlas 실측 확정) **SO230/SP80/SR380/SLM199µm (합 887)**, 누적경계 0→230→310→692→887. [참고] 계획 초안의 SLM146/SR279/SP59/SO168(합652)은 동일비율·0.735배 다른 레퍼런스값 → atlas 실측으로 대체.
- **2. 아틀라스 전처리(Romani)** — Romani 후처리 atlas 그대로 사용(재전처리 불필요).
- **3. 좌표/방향 벡터화** — l/t/r + 방사 벡터장(orientation field 또는 세포별 quaternion). ✅V1b: 방사벡터 SO→SLM 수직.
- **4. 층 구분** — SO/SP/SR/SLM 경계·두께. ✅V1c.
- **4b. 세포 조성**(보완) — 층별 m/e-type 개수·밀도·E:I. ✅V2a: 밀도·E:I(≈89:11) 일치.
- **[슬라이스 채택 확정]** like-slice = **slice400** (atlas `nrrd_volumes/slices/slice400.nrrd`, 단일 400µm 절편, 4층 관통, 뉴런 17,647개). 실험 해마 절편에 가장 근접 → MEA/LTP 재현 목표 적합.
- **5. 뉴런 배치** — circuit nodes에서 슬라이스(slice400) 추출 → (좌표·방향·타입). ✅V2b.
- **5b. me-type 매핑+형태 준비**(보완) — (m,e)→우리 23모델; 갭은 m-type 내 대체; (선택)복제. ✅V2c: (m,e) 100% 해소.
- **6. 방향성 주입** — 형태 평행이동+quaternion 회전(`morph_transform`, `h.pt3dchange`). ⚠️quaternion 순서(w,x,y,z↔scipy x,y,z,w). ✅V2d: **길이 불변**·소마위치==목표·정점축≈orientation.
- **7. 커넥텀** — `pathway_class`+거리의존 → edges{pre,post,cls}. ✅V3: cls∈9클래스·수렴발산 타당.
- **8. 시냅스(Ecker)** — `params_table3` 9클래스 EMS 주입. ✅V4: PSP/CV/STP(기존 검증 재사용).
- **9. 구동** — network_lib build_and_place→wire→drive→run→analyze. ✅V5: --demo(약 250세포) 완료·raster·E/I.
- **10. 전체 슬라이스 실규모 구동(MPI)** ✅완료 — `09_run/run_mpi.py`(ParallelContext gid 배선). 전체 17,647세포×1초, 물리10코어 MPI 66.6h, 전세포 100% 발화(363,092 스파이크), 20kHz 전세포 소마 Vm(1.41GB) 저장. 다관점 분석 13종(V6/V7) + 발화 GIF 4종. → GitHub·Notion 갱신 완료.

---

## 11. 실험 및 검증 (MEA fEPSP → LTP/LTD)

> 규칙: 한 번에 하나씩 구현→보고→✅검증. **완료/진행 실험만 상세 기록, 미구현은 표만**(과장 금지). ID는 단일 체계 **E#**(구 EXP-*는 흡수·폐지). 정직성 감사(2026-07) 반영.

### 11.1 완료·진행 실험 (프로토콜·결과 — 실측)

**E1. Baseline 발화율·구동 검증** ✅완료 (`10_analysis/firing_stats.py`, `drive_sweep.py`)
- **E1-a**: 완주 데이터(전체 17,647세포×1초, 외부 Poisson 30Hz *가정*) 발화율 집계. PC 평균 **18.3Hz**·전세포 발화 → **과활성**(방향성 견고). ⚠️정직성: "in vivo의 몇 배"는 문헌밴드 선택에 따라 **9~37배**로 흔들림(대표 1Hz면 ~18배). "정상상태 20.1Hz"는 **E+I 혼합** 집단값(PC 아님). "층별 발화율"은 층 특성 아니라 **세포조성**(SO/SR/SLM 전량 인터뉴런, n=24~29).
- **E1-b**: 구동 weight 배율 스윕(760세포·60ms창). 배율 0.30→PC 0.17Hz, ≤0.15→침묵, 1.0→18Hz. ⚠️정직성: **0.30~1.0 구간 미표집**이라 "안정 저발화 구간 없음(bimodal)"은 **미확정**(표집 공백). 비자명 데이터=단 6(PC)+16(INT) 스파이크. → 결론은 "**MEA용 조용한 슬라이스 baseline 채택**"만 유효.

**E2. Schaffer collateral(CA3→CA1) 경로** (부분 완료)
- **E2-1** ✅ (`sc_epsp_test.py`): 단일 SC→PC EPSP. ⚠️**SC 전용 시냅스가 아니라 Ecker "PC→PC(E2)" 대용**(τ·NRRP·g 어느 것도 Romani SC-PC 아님). 근위SR ≈0.15mV. baseline 오염으로 진값(0.17mV) 대비 과소, Use=0.5로 유효 g≈0.30nS. → **"크기 근사"이지 "SC 구현/Romani 일치"가 아님**(실제 SC 파라미터면 ~0.40mV).
- **E2-2** (subset 배선 검증, `sc_network.py`): 조용한 baseline(PC 0) + SC 볼리 → PC 반응, gabazine 토글 존재. ⚠️실행값 **80syn·3nS는 코드 기본(12syn·0.6nS)의 명령행 오버라이드 튜닝값**(측정 아님). "PC 100% 발화"는 **과자극 포화**. 버그: 미사용 `frng`·SC배선 rng 오염(재현성 결함) → 수정 대상.

**E3. SC 자극 I-O + gabazine** 🔄 **진행중·미완** (`sc_io_curve.py`)
- 예비(108세포): I-O가 선형(R=0.962). **그러나 control ≈ gabazine (두 곡선 겹침) → 피드포워드 억제가 사실상 작동 안 함.** ⚠️정직성: **Fig4(FFI) 재현 아님**(R값은 우연히 선형일 뿐 메커니즘 아님). 저장 그림은 **108세포 예비본**, 1,200세포 전체는 **미완/중단**.
- 판정: **FFI 미재현** — SC→PC가 억제를 압도. 재작업 필요(SC↓ / 억제↑ / disynaptic 타이밍). "Fig4 재현" 주장 **보류**.

### 11.2 향후 실험 (미구현 — 목차/표만)

| ID | 실험 | 목표(요지) | 상태 |
|---|---|---|---|
| E2-3 | 전 슬라이스 SC 배치 | 15,723 PC에 Romani 층분포(SR67.9/SO24.7/SP7.1/SLM0.3%) SC, MPI | ⬜ 예정 |
| E3′ | SC I-O + gabazine 재작업 | FFI 작동시켜 control≠gabazine(진짜 Fig4) | ⬜ 재작업 |
| E4 | 세포외 LFP/fEPSP 계산기 | LFPykit 하이브리드 → SR층 fEPSP(sink) | ⬜ 예정 |
| E5 | theta 변조 SC 입력 + PAC | CA1 theta 추종·theta-nested gamma | ⬜ 예정 |
| E6 | 내측중격(MS) theta | 가상 페이스메이커 리듬 억제 주입 | ⬜ 예정 |
| E7 | ACh 신경조절 | 흥분성↑·theta 촉진(현상론→기계론) | ⬜ 예정 |
| E8 | LTP/LTD | Graupner 칼슘 가소성(papers/02 보유) | ⬜ 예정 |
| E9 | 실측 MEA 대조 | SpikeInterface 로드·정규화 fEPSP 비교 | ⬜ 예정 |

**slice400 기하(정직):** CA1 전용 횡단-유사 가상슬랩(소마 z~755µm=실험 300–400µm의 약 2배), 4층 정상(방사 ~905µm)·SR 확보 → MEA식 fEPSP엔 기하 부합. 단 **CA3/DG 부재→SC는 외부자극 대체, 물리절단 아님** → fEPSP 정량비교 시 스케일보정·한계 명시. Romani Fig4 규모=300µm·SP 101세포·SC 350축삭.

## 진행 약속
각 단계·실험 끝에 ✅검증 그림/수치로 **하나씩 확인** 후 다음으로.
