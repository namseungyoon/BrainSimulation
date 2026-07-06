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

## 실험 로드맵 — MEA fEPSP → LTP/LTD (Romani 2024 실험 기반)

> **규칙: 한 번에 하나씩** 구현 → 보고 → ✅검증 → 다음. Romani(2024) 풀스케일 CA1 논문의 in silico 실험을 우리 like-slice에 이식. 의존순서: **E1 → E2 → E3 → E4 → (11 fEPSP완성) → (12 LTP) → (13 실측)**.

### E1. Baseline 발화율·E/I 검증 + 구동 튜닝 (새 시뮬 불필요)
- **목표**: 완주 데이터가 생리적인지 판정. 현재 **PC 18.3Hz·전세포 100% 발화**는 in vivo CA1 추체세포(~0.5–2Hz, 성긴 발화) 대비 **과활성 의심** → Poisson 30Hz 외부구동이 과함일 가능성.
- **구현**: `10_analysis/firing_stats.py` — 유형·층별 발화율, seg별 정상성, raster + 문헌 발화율 대조표. 필요시 `network_lib.DRIVE_RATE` 스윕(짧은 재구동).
- **✅검증 E1**: 유형별 발화율이 문헌 범위로 수렴(PC 성기게·INT 높게), 구동강도 확정.

### E2. Schaffer collateral(CA3→CA1) 경로 명시화
- **목표**: 소마 Poisson 대용 → **실제 SC 시냅스**로 교체. PC apical(SR) 수상돌기에 AMPA/NMDA, CA3 스파이크원(`h.VecStim`).
- **구현**: `network_lib`에 SC 배치규칙(Romani 층분포 SR 67.9%/SO 24.7%/SP 7.1%/SLM 0.3%) + Ecker E2 AMPA/NMDA 재사용 + 자극 인터페이스(동기 볼리 vs Poisson).
- **✅검증 E2**: 단일 SC→PC EPSP가 Romani 0.15±0.12mV와 일치(`paired_recording.py` 확장).
- 의존: E1.

### E3. SC 자극 I-O + 피드포워드 억제(gabazine) — Romani Fig.4 재현
- **목표**: 300µm 슬라이스 조건, SC 축삭 활성비율 5–100% 스윕 → 발화세포수 I-O 곡선. 인터뉴런 연결 차단 = gabazine 모사.
- **구현**: SC 동기자극 프로토콜 + 인터뉴런(post=I) 시냅스 skip 토글 + I-O 곡선 그림.
- **✅검증 E3**: control I-O 선형(Romani R=0.992), GABA차단 시 포화 재현. → **MEA 자극-반응 곡선의 in silico 대응물**.
- 의존: E2.

### E4. 세포외 LFP/fEPSP 계산기 — **LFPykit 하이브리드** (조사 확정, 2026-07)
- **목표**: 막전류 → 가상 MEA 전극전위. SC 자극에 **SR층 전극에서 음성 fEPSP(sink)** 재현.
- **스택(확정)**: 계산코어 **LFPykit**(순수 파이썬, py3.10 정합, 의존성 numpy·scipy·MEAutility뿐 → 설치리스크 0. `RecMEAElectrode`=in vitro 슬라이스 3층[유리/조직/식염수] 전도도) + 전극기하 **MEAutility**. 자극은 NEURON 내장 **extracellular + xtra.mod**(초기엔 명시적 시냅스 볼리로 단순화). **LFPy 전면이식은 회피**(Cython/MSVC 빌드·mpi4py 전환·17k세포 재구성 부담). 원칙: *검증된 물리(forward model)는 라이브러리, 기록·좌표·MPI 합산은 직접 글루*.
- **구현**: `run_mpi.py`에 `h.cvode.use_fast_imem(1)`(고정 dt서 동작) + per-seg `i_membrane_` 기록(기존 Vm 훅 확장) → 세그 3D좌표(`h.x3d`+세포 offset)로 `CellGeometry` → 기하고정 M 1회산출 → `V=M·I_mem` → `pc.py_allreduce` 합산. 신규 `10_fepsp/{electrode.py, stim_extracellular.py, analyze.py}`. ⚠️ coarse nseg=1 근거리 왜곡·활성함수 불가 → **fEPSP/자극 대상 세포만 nseg 세분**(d_lambda).
- **✅검증 E4**: (0) 수백 세포 축소서 `use_fast_imem`×EMS 공존 확인 → (1) SC 자극 fEPSP 파형(음성 SR sink) + paired-pulse 비율.
- 의존: E2, E3. → **MEA fEPSP 재현 1차 마일스톤(=아래 11)**. 실측 대조(13)엔 SpikeInterface로 MEA 데이터 로드.

## 확장 (E1~E4 이후)
11) **MEA fEPSP 완성**(세포외 장 LFPy/LSA) · 12) **LTP/LTD**(Graupner 2012 칼슘모델 — `papers/02_Graupner2012...`에 이미 보유 + `demos/02_NEURON/02_STDP.py`를 회로 weight 훅으로 연결) · 13) **실측 MEA 비교**.

## 진행 약속
각 단계·실험 끝에 ✅검증 그림/수치로 **하나씩 확인** 후 다음으로.
