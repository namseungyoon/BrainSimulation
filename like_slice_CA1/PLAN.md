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
- **9. 구동** — network_lib build_and_place→wire→drive→run→analyze. ✅V5: --demo 완료·raster·E/I.

## 확장 (향후)
11) MEA fEPSP(세포외 장 LSA/LFPy) · 12) LTP/LTD(⚠️장기 가소성 모델 추가 — Graupner-Brunel/Chindemi 2022) · 13) 실측 MEA 비교.

## 진행 약속
각 단계 끝에 ✅검증 그림/수치로 **하나씩 확인** 후 다음 단계.
