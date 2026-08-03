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
- `shared/` : `common/{nrn_env, cell_loader, model_naming, plotstyle}`, `models/`(23 me-model + registry), `mechanisms/`(BBP 확률 시냅스·VecStim mod).
- `papers/01_Ecker2020_CA1_synaptic/04_network/network_lib.py` : `wire_synapses·add_external_drive·record_spikes·run_network·analyze_activity·spikes_to_arrays·_placement·_dendrites`.
- `.../05_paired_recording/pathway_map.py` : `pathway_class(pre_mtype, post_mtype) → 9클래스`.
- `.../03_synapses/params_table3.py` : `CLASSES`(9클래스 시냅스 파라미터).

## 환경 / 도구
ca1sim (h5py 3.16 · scipy 1.15.3 · numpy 2.2.6 설치됨). 추가: `pip install pynrrd`(atlas NRRD). 전체경로 `C:\Users\SYNAM-O피드포워드 억제CE\.conda\envs\ca1sim\python.exe`.

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
- **8. 시냅스(Ecker)** — `params_table3` 9클래스 BBP 확률 시냅스 주입. ✅V4: PSP/CV/STP(기존 검증 재사용).
- **9. 구동** — network_lib build_and_place→wire→drive→run→analyze. ✅V5: --demo(약 250세포) 완료·raster·E/I.
- **10. 전체 슬라이스 실규모 구동(MPI)** ✅완료 — `09_run/run_mpi.py`(ParallelContext gid 배선). 전체 17,647세포×1초, 물리10코어 MPI 66.6h, 전세포 100% 발화(363,092 스파이크), 20kHz 전세포 소마 Vm(1.41GB) 저장. 다관점 분석 13종(V6/V7) + 발화 GIF 4종. → GitHub·Notion 갱신 완료.

---

## 11. 실험 및 검증 (MEA fEPSP → LTP/LTD)

> 규칙: 하나씩 구현→보고→✅검증. ID 단일체계 **E1~E10**. **완료는 결과까지, 미실행은 계획·근거만**(결과 날조 금지). 정직성 감사(2026-07) 반영 — 튜닝값≠측정값, "Fig4 재현"은 피드포워드 억제 실작동 시에만.

| ID | 실험 | 상태 |
|---|---|---|
| E1 | Baseline 발화율·구동 검증 | ✅ 완료 |
| E2 | Schaffer collateral 경로 | 🔄 E2-a·E2-b ✅ / E2-c subset 검증✅·전슬라이스⬜ |
| E3 | SC 자극 I-O + 억제 차단 | ✅ subset 완료(피드포워드 억제 작동, gap 71%p) |
| E4 | 세포외 LFP/fEPSP 계산기 | 🔄 E4a ✅(무의존 LSA·SR fEPSP·삼중극) · E4b MEA 예정 |
| E5 | theta 변조 SC 입력 + PAC | ⬜ 예정 |
| E6 | 내측중격(MS) theta | ⬜ 예정 |
| E7 | ACh 신경조절 | ⬜ 예정 |
| E8 | LTP/LTD (칼슘 가소성) | ⬜ 예정 |
| E9 | 실측 MEA 대조 | ⬜ 예정 |
| E10 | STDP 곡선 malleability (Wittenberg&Wang 2006) | ⬜ 예정 |

> **공통 양식** — 주제 E{N}: 주제설명 + 하위실험(E{N}-a…) + **결론**. 각 실험: 목표 / 방법·입력 / 검증지표 / 결과·상태 / 근거(논문X→우리Y) / 한계·주의. 결론=완료는 실제, 미실행은 "(예상·미실행) 실행 후 작성".

### E1. Baseline 발화율·구동 검증 ✅
- **목표**: 완주 baseline이 생리적인지 판정 + 외부구동 강도 결정.
- **방법·입력**: (E1-a) 완주 데이터(17,647세포×1초·외부 Poisson 30Hz) 유형/층별 발화율 집계. (E1-b) 구동 weight 배율 스윕(760세포·60ms창). `10_analysis/firing_stats.py`·`drive_sweep.py`.
- **검증지표**: 유형별 발화율 vs in vivo 문헌 밴드; 구동↔발화율 관계.
- **결과·상태**: PC 평균 **18.3Hz**·전세포 발화 = 과활성(E1-a). 배율 0.30→0.17Hz, ≤0.15→침묵, 1.0→18Hz(E1-b). → **조용한 슬라이스 baseline 채택**. 그림 E1-a·E1-b.
- **근거**: in vivo CA1 PC ~0.3–2Hz(Mizuseki & Buzsáki 2013).
- **⚠️ 한계·주의**: "몇 배"는 밴드 따라 9~37배(견고X). "정상상태 20.1Hz"=E+I 혼합(PC 아님). "층별 발화율"=세포조성(SO/SR/SLM 전량 INT, n=24~29). 0.30~1.0 미표집→"bimodal" 미확정.
- **결론**: baseline 과활성(PC 18.3Hz)·생리적 저발화 구간 부재 확인 → **MEA용 조용한 슬라이스 baseline 채택** 결정(방향성 견고, 정량 배수는 밴드 의존).
- **E1-G (GPU 재실행 ⬜)**: 전 슬라이스 baseline 1초를 GPU·확률적 TM 시냅스로 재실행(CPU 66.6h 대비 배속 측정). CoreNEURON GPU 빌드·검증 후 → `GPU_SETUP_PLAN.md`.

### E2. Schaffer collateral(CA3→CA1) 경로 🔄 (E2-a·E2-b ✅ / E2-c ⬜)
- **목표**: CA3 SC 입력을 명시적 시냅스로 배선 — 조용한 슬라이스에 자극원 제공.
- **방법·입력**: 가상 SC fiber **~800**(CA3 축삭 대용) → 세포당 SC 시냅스 PC **~60**(SR68/SO25/SP7/SLM0.3%)·INT **~40**, 무작위 fiber 연결. 시냅스=Ecker "PC→PC(E2)" AMPA/NMDA 대용. 전도도(튜닝) SC→PC~1.0nS/INT~4.0nS. `sc_epsp_test.py`·`sc_network.py`.
- **검증지표**: 단일 SC→PC EPSP=Romani 0.15±0.12mV; baseline 조용; 시냅스 층분포=Romani.
- **결과·상태**: **E2-a** ✅ 단일 EPSP 근위SR 0.15mV(그림 E2-a 파형·배치+SC모식). **E2-b** ✅ 조용한 baseline+볼리→PC 반응(subset). **E2-c** 🔄 **subset 2,000세포 SC 포아송 지속구동 9초 검증 완료**(dt 0.025·`sc_full_slice.py`): PC **10.6Hz 9초 내내 지속**·98% 발화·INT 0.2Hz, 그림 E2-c(발화율·raster·20kHz 막전위). **전 17,647 배치는 dt 0.025서 ~1초/실행 제약(약 64h/1초)→GPU 필요**.
- **근거**: Romani PC당 SC ~20,878(±5,867)·INT ~12,714·CA3 267,238세포·층분포(SR67.9/SO24.7/SP7.1/SLM0.3%). fiber ~800=수렴비율 7.8% 보존. 시냅스 ~1/350 축소+전도도 보정.
- **⚠️ 한계·주의**: SC=Ecker E2 대용(Romani 전용 SC-PC 0.85nS·τ0.4/12·NRRP12 아님, kinetics 다름). E2-a baseline 오염(진값 0.17mV)·Use=0.5로 유효 0.30nS. 전도도·fiber수=튜닝/설계값(측정 아님). **E2-c 지속구동은 감소 보정값**: SC 60시냅스(실제 ~20,000의 1/350)로 지속 발화시키려 fiber 150Hz·SC→PC 10nS로 세게 몲 → PC 10.6Hz는 **in-vivo(~1–2Hz)보다 높음**(검증용). SC→INT 3nS는 약해 정상상태 INT 0.2Hz(피드포워드 억제 미미). subset 2,000세포(전 슬라이스 아님).
- **결론**: CA3 부재를 가상 SC fiber로 대체하는 배선법 확립·3단 검증(EPSP 크기 근사 E2-a·볼리 반응 E2-b·포아송 지속발화 E2-c). **SC 경로가 조용한 슬라이스를 구동함 확인.** 전도도·fiber수·구동빈도=보정 튜닝값, 전 슬라이스 배선은 GPU 과제.
- **E2-c-G · E2-c 전슬라이스-G (GPU ⬜)**: E2-c(2,000세포+SC 9초)를 GPU·확률적 TM 시냅스로 재실행(결정론→**확률로 변경**), 그리고 전 슬라이스+SC 1초(E2-c 전슬라이스-G)도 GPU로 → `GPU_SETUP_PLAN.md`.

### E3. SC 자극 I-O + 억제 차단 ✅ subset 완료(피드포워드 억제 작동)
- **목표**: SC 세기별 자극→발화 PC I-O 곡선. 정상 vs 억제차단 비교로 피드포워드 억제가 반응을 조절함을 확인(Romani Fig.4).
- **방법·입력**: 1,200세포 subset(PC900+INT300), SC 볼리 활성비율 50/75/100% 스윕, 정상(억제 ON)/억제 차단(OFF). 실행 조합: SC→PC 1.0nS·SC→INT 6.0nS·억제 ×3. `sc_io_curve.py`.
- **검증지표**: 정상 I-O 선형성 R; **피드포워드 억제 gap(억제차단−정상) ≥10%p**.
- **결과·상태**: ✅ 정상(억제 ON ×3): 50%→3.2%·75%→24.0%·100%→60.7%(단계적). 억제 차단(OFF): 50%→41.6%·75%→95.0%·100%→100%(급포화). **gap 최대 71.0%p → 피드포워드 억제 작동**(Romani Fig.4 기전 재현). 정상 I-O R=0.893(시그모이드형). 그림 E3-a. 개념도 그림 E3.
- **근거**: Romani Fig.4(정상 선형 R=0.992·억제차단 포화); Pouille & Scanziani "window of opportunity".
- **⚠️ 한계·주의**: 1,200세포 subset(전 슬라이스 아님, I-O 스윕 ~56h 비현실적). SC→PC 1.0·SC→INT 6.0·억제 ×3 = **튜닝값(측정 아님)**. 정상 R=0.893은 시그모이드형으로 Romani 0.992보다 덜 선형(자극점 3개·조정 여지).
- **E3c (전슬라이스 GPU 결정론 I-O · ✅ 완주, 2026-07-30)** — 결과: 억제 gap 93.7%p(subset 71%p보다 강함)·control 100%서 26.6% vs block 60%서 94% 급포화·R=0.797·psolve 12회 ~2h44m. `sc_gpu_io.py`·그림 `E3c_io_curve.png`. 방법: subset(1,200·확률·CPU) I-O+억제차단을 **전슬라이스(17,647·결정론·GPU)**로 확장. E2-c 전슬라이스 GPU 완주로 실현 가능(과거 "전슬라이스 스윕 ~56h 비현실" 한계 해소). 방법: 결정론 전슬라이스 모델 **1회 빌드 → SC 볼리 활성비율(5~100%)×억제 정상/차단 루프 psolve(볼리 ~100ms 저렴) → [10,60)ms PC 발화 측정**(한 프로세스서 build-once-run-many). Start-Process 분리실행. 볼리 I-O 프로토콜(`sc_io_curve`)을 GPU 전슬라이스 파이프라인(`sc_full_slice`)에 이식 필요. ⚠️ 결정론=시행변동 없음(깨끗한 I-O·확률판과 성격 다름)·전도도/활성비율 튜닝값 유지.
- **결론**: **피드포워드 억제가 SC 반응 이득을 조절함을 subset 재현**(정상 vs 억제차단 gap 71%p, Romani Fig.4 기전). SC→PC 세기 "적정 창"이 관건. 튜닝값·시그모이드·1,200 subset 한계 → 기전 정성 재현, 정량·전 슬라이스는 향후.

### E4. 세포외 LFP/fEPSP 계산기 🔄 (E4a ✅ · E4b 예정)
- **목표**: 막전류→가상 전극 세포외전위, SC 자극에 SR층 음성 fEPSP(sink) 재현.
- **E4a ✅ (2026-07-31, 무의존 LSA 계산기)**: `h.CVode().use_fast_imem(1)`+per-seg `i_membrane_`(nA) 기록 → **자체 LSA(line-source) 전달행렬**(numpy만, LFPykit 미설치) → V=M·I(mV). 상세형태 대표 PC(723세그먼트)에 SC 40시냅스 SR 동기볼리. 결과: SR **음성 -0.55µV**·slope -0.641µV/ms, 깊이 **source-sink-source 삼중극**(반전 84·331µm), 전류보존 ΣI/max|I|=1.2e-14, PPR 0.56(depression, E2 대용). **7-에이전트 적대검증 통과**(LSA=Holt&Koch 오차1e-16·독립재구현 재현). `12_lfp/`.
- **E4b 🔄 (MoI ✅ 검증, 2026-08)**: `lfp_calc.moi_point_matrix` — 유리(z=0,절연)-조직-식염수 3층 영상법(Ness 2015). 적대검증 3중 독립재구현 기계정밀 일치·버그0(극한 h→∞·W_TS=0→정확히 2.0배·표준 1.770·W_TS=−2/3, sanity 고정). 상세 PC를 MEA 프레임 회전·정렬 N개 복제 합산 → 집단 fEPSP. `e4b_fepsp.py`·`e4b_mea_layout.py`. **결과**: 단일 0.028µV(=무한×1.9), 정렬+동기 N=15000→35µV @19,099/mm²(실제 CA1 밀도 근접). **★정직(적대검증 정정)**: 실측 0.1~1mV 대비 3~30x 미달 = **밀도 아님(충분) → 세포당 진폭**(이상화 볼리·E2 대용 시냅스·단일깊이·유리전극 원거리). 두께: 계획 h300µm이나 full-size 세포(~700µm) 수용 위해 실행 h811µm(실측 300~400µm는 세포 절단); 두께는 경계반사(과대)·소스거리(과소) 상반효과라 분리해석. MEA 8×8@200µm(1400µm)는 슬라이스 면(2577×1082µm)에 부분적합(31/64 조직 위, CA1 곡선). **★ 실제 밴드 재계산(2026-08, `e4b_band.py`·`e4b_mea_configs.py`)**: disk 대신 실제 PC 15,723개 실위치(SP 밴드 2305×468µm)로 각 전극 fEPSP → 밴드 전극 음성 fEPSP 8×8 중앙 |115|·**3×8(24/24 조직 위=확정 config) 중앙 |120|·최대 |171|µV** = **원반(35µV)의 3~5배 = 실측 0.1~1mV 저역대 도달**(disk footprint가 비관적이었음, 실 밴드 기하는 realistic-order). 단 완전정렬·동기·동일깊이라 상한값(실 지터가 낮춤). 얇은 밴드(468µm)엔 **3×8 최적**(8×8은 47%만 조직 위). 절대 mV·정량 대조는 실측 MEA 사양(슬라이스크기·측정부위)+생물학적 지터 반영 후 E9.
- **검증지표**: SR 전극 음성 fEPSP(sink) + 극성반전(삼중극) + paired-pulse 비율. (E4a 충족)
- **근거**: Holt&Koch 1999(LSA)·LFPy/LFPykit(Lindén 2014)·Ness 2015(슬라이스-MEA forward)·Colbert&Levy 1992·Teleńczuk 2020(J Physiol, 단일세포 sub-µV). 원 계획 `h.cvode…`는 8.2.7서 실패→`h.CVode()` 정정, LFPy 전면이식 회피→무의존 자체 LSA.
- **⚠️ 한계·주의**: 단일세포 sub-µV(집단 mV=향후 앙상블)·무한매질(MEA 영상법=E4b)·coarse nseg=1 부적합→상세형태 세포만 사용(723세그먼트)·PPR은 E2 대용 시냅스라 depression(실 SC PPF 아님).

### E5. theta 변조 SC 입력 + theta-nested gamma(PAC) ⬜ 예정
- **목표**: CA1이 입력 theta(8Hz) 추종하는지 + gamma의 theta 위상결합(PAC).
- **방법·입력**: SC fiber 발화율 정현파 변조 r(t)=r0(1+m·sin(2π·8t)), r0≈0.1–0.4Hz → inhomogeneous Poisson(Elephant/numpy) → `h.VecStim`.
- **검증지표**: population rate 스펙트럼 8Hz 피크; gamma(30–100Hz) 파워의 theta 위상결합(Tort MI).
- **결과·상태**: 미실행.
- **근거**: Romani가 CA3 theta를 정현파 SC로 주입→CA1 추종(생성기 아닌 추종기). gamma는 PING 자발생성.
- **⚠️ 한계·주의**: gamma 자발생성은 억제회로 파라미터 의존(자동 보장 아님).

### E6. 내측중격(MS) theta 페이스메이커 ⬜ 예정
- **목표**: MS-DBB의 GABA성 리드믹 탈억제로 theta 구동.
- **방법·입력**: MS 미보유→가상 입력원. PV-BC/OLM에 8Hz 정현파 억제입력(`h.VecStim`)→리듬 탈억제로 PC theta 위상 발화.
- **검증지표**: PC 발화의 theta 위상잠김(phase-locking); E5와 PAC 패턴 비교.
- **결과·상태**: 미실행.
- **근거**: 생체 theta 페이스메이커=MS-DBB(CA3 가상대체와 동일 철학). E5(입력변조)와 상보.
- **⚠️ 한계·주의**: 가상 입력원(실제 MS 세포 아님).

### E7. ACh(무스카린성) 신경조절 ⬜ 예정
- **목표**: PC 흥분성↑(M-전류 억제)·SC 전달↓·theta 촉진.
- **방법·입력**: 1차 현상론(SC weight↓+PC 흥분성/배경 조정), 2차 기계론(me-model에 Im/KM+mAChR mod).
- **검증지표**: ACh 조건 PC 발화율↑·theta 파워↑·SC EPSP 감쇠.
- **결과·상태**: 미실행.
- **근거**: Romani ACh Hill 용량반응.
- **⚠️ 한계·주의**: 기계론(Im/KM mod)은 현 23 me-model 보유 여부 선확인 필요.

### E8. LTP/LTD (칼슘 기반 가소성) 🔄 (E8.1 설계 · E8.2 결정론 GPU 포팅 ✅)
- **목표**: SC 자극(TBS/LFS)에 시냅스 weight 변화 재현.
- **핵심 구조(2026-07-24 확정)**: Graupner 모델은 **전달 모델이 아니라 가소성-전용 학습규칙**(효능 ρ만 변조, AMPA/NMDA 동역학 없음 — `papers/02/plasticity_model.py` 코드 대조 확인). 따라서 시냅스를 "대체"하는 게 아니라 **[전달 시냅스(AMPA/NMDA)] × [Graupner ρ 가중치 변조]** 결합 구조. 이는 Chindemi 2022가 Blue Brain 미세회로(163,271세포)에 적용한 방식과 동일: **결정론 이중안정 ρ + TM 전달**. Ecker/Reimann eLife(2024/25)는 211,712세포 전 흥분성 쌍에 적용 → 우리 계열에서 네트워크 규모 배치 입증.
- **방법·입력**: (E8.1) Graupner-Brunel 칼슘모델(`papers/02`, Wittenberg2006 fit)을 **결정론(σ=0) NMODL POINT_PROCESS**로 이식 — 칼슘 c ODE + ρ 이중안정(STATE, `DERIVATIVE cnexp`) + AMPA/NMDA 이중지수 전달 컨덕턴스 + `w=w0+ρ(w1−w0)` 스케일. pre 지연 D는 `net_send(D,2)` 단일 self-event(벡터 큐 금지=#638 회피), post는 소마 역치 NetCon. 단일연결 검증→회로. (E8.2) 위 mod가 GPU에서 돌려면 **결정론 전달이 GPU 친화**(Random123 불필요)임을 먼저 실증 — 기존 `DetAMPANMDA/DetGABAAB`의 `R/Pr/u/tsyn`를 NET_RECEIVE 인자→RANGE로 옮기고(#1067 회피, Prob* 방식) 지연연결 가드(#638) 후 CoreNEURON GPU(nvc++ 26.5) 컴파일+실행.
- **검증지표**: (E8.1) 신규 mod 단일 시냅스 출력이 오프라인 `integrate_rho`(σ=0)와 일치; TBS 후 ρ 분포 UP 이동(LTP)·저빈도 후 DOWN(LTD). (E8.2) Det\* mod GPU 컴파일 통과(NVC++-S 0) + repro 실행 RC=0(vs 확률 시냅스 Random123 세그폴트 RC=139) + GPU빌드 CPU모드=CPU 비트-동일.
- **결과·상태**: **E8.1 ✅ mod 구축·검증(2026-07-24)** — `shared/mechanisms/GBPlasticitySyn.mod` 작성(칼슘 c + ρ 이중안정 + AMPA/NMDA 전달, derivimplicit·net_send 단일 self-event·RNG 없음, Wittenberg2006 기본값). `papers/02/validate_gbmod.py`로 50Hz pre-post 프로토콜에서 **rho(t)가 Python `integrate_rho`(σ=0)와 일치**(mod 0.81598 vs Python 0.81598, 최대차 3e-5; 칼슘 최댓값도 일치, c의 1-샘플 점프-엣지 아티팩트 제외). **E8.2 ✅ 결정론 시냅스 GPU 포팅 실증(2026-07-24)** — `_wsl_det_gpu.py`로 원본 무변경·WSL `~/mods_det_gpu` 복사본에 가드+RANGE 리팩터 7개 앵커 적용 → nvc++ 26.5 GPU 빌드 **컴파일 통과(NVC++-S 0)** + repro **GPU 실행 RC=0 완주**(A6000·cell-permute type 1) · CPU 대조 RC=0. 확률 시냅스 Random123 GPU 세그폴트(RC=139) 대비 → **결정론 전달은 GPU 실행 가능 실증**. **★ 전슬라이스 완주(2026-07-24)**: 17,647세포·1초·결정론·SC를 A6000 GPU로 완주 — 스파이크 201,200·발화 99%·PC 12.37Hz·INT 3.45Hz·psolve 1.35h(층별 SP 11.55/SO 3.61/SLM 3.03/SR 1.96Hz; CPU 확률판 PC 13.50과 동일 대역·층별 순서 일치). **3관문 해결**: ①SC 포아송 GPU 세그폴트 진범=**NetStim Random123**(시냅스 아님; noise=0·위상무작위 빌드타임 구동으로 우회, 엄밀 포아송은 향후 PatternStim) ②OOM(호스트 pinned>84GB) → 랭크 **-n 4**로 컨텍스트 절감 ③분리실행 → **`Start-Process wsl.exe -e bash <script>`**(스크립트 `exec>log` 자체 리다이렉트)가 harness·세션 무관 3h 완주(설정: nrn-gpu MPI 재빌드 `~/nrn-gpu-mpi`·`~/mods_full_gpu_mpi`·`~/models_native`). `_wsl_mpi_gpu_fullrun_n4.sh`·출력 `sc_det_gpu/fullscale_n4`.
- **근거**: Chindemi 2022(상세망 결정론 ρ + 확률 TM, GPU/CoreNEURON은 future work로 명시)·Ecker/Reimann eLife 2024/25(네트워크 규모 유발 가소성)·Graupner 2012(칼슘 이중안정 원본, 단일 시냅스 STDP 검증)·Higgins 2014(이중안정=기억안정 기전).
- **⚠️ 한계·주의**: (1) **Romani에서 벗어남** — Romani는 확률 방출. 결정론 전달+Graupner ρ로 가는 이유: LTP가 요구하는 메커니즘 + GPU 실현성 + MEA는 집단 평균 신호(방출 변동 제거는 방어 가능한 단순화). "논문 X(확률 방출)→우리 Y(결정론 전달+ρ)→이유" 명시. Blue Brain 자신도 네트워크판 ρ는 결정론. (2) Graupner는 **초기 유도(분 단위)만** — 태깅-포획/공고화(시간~일) 없음. τ≈688s라 짧은 런에선 ρ 거의 불변(유도 신호 측정용). (3) Det\* RANGE 리팩터는 per-netcon→per-instance 이동 = 우리 모델(시냅스당 netcon 1개)에선 동치, 원본 BBP 대비 변경점. (4) 저[Ca]o(1mM) in vitro 조건 필요. (5) "가소성 규칙 GPU 실행"은 문헌상 아직 프런티어(Chindemi도 CoreNEURON GPU 미실증).

### E9. 실측 MEA fEPSP 대조 ⬜ 예정 (최종)
- **목표**: in silico fEPSP·LTP를 실측 HD-MEA와 정량 비교.
- **방법·입력**: SpikeInterface로 실측 MEA 로드, 정규화 fEPSP·kCSD로 in silico와 대조.
- **검증지표**: 정규화 fEPSP 파형·I-O·LTP 곡선이 실측 범위 내.
- **결과·상태**: 미실행.
- **근거**: EvoNES 2025(HD-CMOS-MEA SC 유발 fEPSP+네트워크 LTP 실측).
- **⚠️ 한계·주의**: slice400 축소·스케일 보정 반영 필요.

### E10. STDP 곡선 malleability 재현 (Wittenberg & Wang 2006) ⬜ 예정
- **목표**: CA3→CA1(SC) 시냅스의 스파이크-타이밍 의존 가소성(STDP, spike-timing-dependent plasticity) 곡선 재현. 특히 **후시냅스 활동 형태**(단일 스파이크 vs 2-스파이크 버스트=doublet)와 **페어링 빈도(theta 5Hz)·횟수**에 따라 동일 시냅스가 **① LTD-only · ② 양방향(sombrero) · ③ LTP-only** 세 규칙을 모두 보이는 "malleability(가변성)"를 그대로 재현.
- **모델 확정(2026-07-10 결정)**: 현재 SC 시냅스의 **단기가소성(STP: 방출확률 Use/회복 Dep/촉진 Fac)은 수백 ms 내 회복 → 지속적 LTP/LTD(장기가소성)를 구조적으로 못 만듦**(튜닝으로 해결 불가). 따라서 E10은 **장기가소성 NMODL 시냅스를 신규 구현**해 진행. Graupner-Brunel 칼슘기반 가소성(칼슘 c(t) → ρ 이중안정 상태변수 → 전도도 w=w0+ρ·(w1−w0))을 NMODL(POINT_PROCESS+NET_RECEIVE)로 이식하고, `papers/02`의 **Wittenberg2006 피팅 파라미터**(`PARAM_SETS["hippo_slice_Wittenberg2006"]`, Fig3/S10 fit) 사용. **E8(회로 LTP/LTD)과 mod 공유.**
- **방법·입력**: (1) 레퍼런스 — 오프라인 `papers/02/plasticity_model.py`(Wittenberg 파라미터 내장)로 세 조건 곡선 먼저 산출(`calcium_trace`→`integrate_rho`→`strength_change_ratio`). (2) 신규 NMODL 장기가소성 시냅스 작성·컴파일 → 단일 SC→PC 연결(`sc_epsp_test.py` 배선)에서 오프라인 모델과 일치 검증. (3) Δt(−100~+100ms) 스윕으로 세 조건(논문 그대로) 재현 — **E10-a** 단일 스파이크 70~100회@0.1~0.5Hz; **E10-b** doublet(스파이크 2개·간격~10ms) 5Hz×70~100회; **E10-c** doublet 5Hz×20~30회. (4) →E8 회로 확장.
- **검증지표**: (a) LTD-only 넓은 창(~0.84×baseline, 반치폭~113ms); (b) sombrero — 인과 LTP 창(Δt≈+10~+25ms, 정점~1.58×)을 anti-causal(Δt≈−20~−3ms, ~0.74×)·원거리 인과(Δt≈+25~+40ms, ~0.71×) LTD 창이 협공; (c) LTP-only(중심 Δt≈+4ms, 정점~1.29×). 곡선 **모양·부호가 논문 Fig1D/Fig3E와 정성 일치** + 신규 mod가 단일 시냅스에서 오프라인 모델과 수치 일치.
- **결과·상태**: ⬜ 미실행(설계·근거·모델 확정. 웹 1차출처 교차검증 완료. **모델=신규 장기가소성 NMODL mod로 확정**).
- **근거**: Wittenberg & Wang 2006 *J Neurosci* 26(24):6610-6617 (DOI 10.1523/JNEUROSCI.5388-05.2006). P14–21 SD rat 급성 슬라이스·whole-cell·SR의 SC 자극; doublet 간격 **10.3±0.9ms**; **5Hz(theta) 반복이 LTP 필수**(0.5Hz는 LTP 소실 ~0.80×); LTP는 **20~30회**·LTD는 **수백 회** 페어링 요구. 결론: "단일 STDP 규칙 하나로 활동→시냅스강도 매핑을 온전히 기술 불가". **우리(like-slice)**: STP는 장기가소성 구조적 불가 → **Graupner 칼슘모델을 NMODL로 신규 이식** — 파라미터가 **바로 이 논문 데이터에 피팅된 값**이라 재현 근거가 코드에 내장됨.
- **⚠️ 한계·주의**: **신규 NMODL 개발·검증 필요(개발량 있음)** — 저장소에 장기가소성 시냅스 mod(GluSynapse류) 부재가 선행과제. Graupner는 형태학·역전파 활동전위(bAP)·NMDA 칼슘유입을 명시적으로 풀지 않는 **현상론적 칼슘 트레이스**(pre=지연 D 후 C_pre 점프, post=C_post 점프, τ_ca 감쇠). 파라미터=Graupner **피팅값(측정 아님)**. 논문 doublet은 전류주입 유발(실측), 우리는 스파이크 시각 리스트로 모사. **정성 재현이 목표**(정량 오차·소수자리는 원문 PDF 대조 권장). E8(회로 TBS/LFS→fEPSP slope)과 구분: **E10 = 단일 시냅스 STDP Δt 곡선**, E8 = 회로 fEPSP.

**slice400 기하(정직):** CA1 전용 횡단-유사 가상슬랩(소마 z~755µm=실험 300–400µm의 약 2배), 4층 정상(방사 ~905µm)·SR 확보 → MEA식 fEPSP엔 기하 부합. 단 **CA3/DG 부재→SC 외부자극 대체, 물리절단 아님** → fEPSP 정량비교 시 스케일보정·한계 명시. Romani Fig4 규모=300µm·SP 101세포·SC 350축삭.

## 진행 약속
각 단계·실험 끝에 ✅검증 그림/수치로 **하나씩 확인** 후 다음으로.
