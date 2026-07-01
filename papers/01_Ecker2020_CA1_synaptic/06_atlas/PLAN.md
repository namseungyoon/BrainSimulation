# like-slice 시뮬레이터 — 단계별 계획서 (Romani 순서 기반)

> **목적**: 해마 슬라이스와 비슷한 CA1 like-slice 시뮬레이터 구축 → (최종) **MEA 측정 재현 → LTP/LTD 재현 → 실측 데이터 비교**.
> **방식(요약)**: Romani 2024의 **아틀라스·배치를 재사용**(불러오기) + **우리 검증된 23 me-model(뉴런)** + **우리 검증된 9클래스 시냅스(Ecker)**. 연결은 우리 규칙으로 구성.

---

## 필요 데이터 (Harvard Dataverse DOI 10.7910/DVN/TN3DUI) — 용량 제한 없음
| 파일 | 용량 | 내용 | 받기 |
|---|---|---|---|
| **circuit.zip** | 2.53 GB | SONATA nodes = **456,380 세포 배치표**(좌표·방향·m/e-type·층) | ✅ 필수 |
| **atlas.zip** | 297 MB | 부피·**층·방향장**(복셀 NRRD) | ✅ 필수 |
| morphology_library.zip | 2.49 GB | Romani 형태 라이브러리 | 🔶 선택(변이/대조용; 우리 23 우선) |
| single_cell_model_library.zip | 0.7 MB | Romani e-model | 🔶 선택(우리 보유) |
| synapses/touches/projections.xz* | ~595 GB | Romani 연결(커넥톰) | ❌ (우리가 Ecker로 생성) |

**도구**: `h5py`·`scipy`·`numpy` (설치됨) + `pip install pynrrd`(atlas NRRD 읽기). *(libsonata/bluepysnap는 선택; h5py로 충분)*
**우리 자산(보유)**: 23 me-model(12 m-type), `params_table3`(9클래스), `network_lib`, `pathway_map`.

---

## 단계별 계획 (✅ = 우리가 확인할 검증 포인트)

### 0. 데이터 준비
- circuit.zip + atlas.zip 다운로드 → 압축해제 → `nodes.h5` 구조(필드명·orientation 표현) 확인.
- ✅ **검증 V0**: 세포 수 ≈ 456,380, 필드(x/y/z·orientation·mtype·etype·layer) 존재 확인.

### 1. 아틀라스 기반 슬라이스 3D 준비
- atlas.zip(복셀) 불러와 **슬라이스 영역(bbox)** 정의 (예: 횡 ~300µm × 종 ~300µm × radial 전층).
- ✅ **검증 V1a**: 슬라이스 부피·층두께가 Romani 정량값(SLM146/SR279/SP59/SO168µm)과 일치.

### 2. 아틀라스 전처리 (Romani 방법)
- Romani가 **이미 후처리한 atlas**(좌표계 보정·층 파라미터화)를 그대로 사용.
- (우리는 재전처리 불필요 — 재사용)

### 3. 아틀라스 좌표 정의 / 방향 벡터화
- l/t/r 3축 + **방사 벡터장**(세포 방향용). atlas의 orientation field 또는 세포별 quaternion 사용.
- ✅ **검증 V1b**: 방사벡터가 층에 수직(SO→SLM) 방향인지 quiver 그림 확인.

### 4. 층 구분
- SO/SP/SR/SLM 경계·두께 정의 (atlas 층 + Romani 두께).
- ✅ **검증 V1c**: 층 경계 그림 + 두께 수치.

### 4b. (보완) 세포 조성 — *빠졌던 단계*
- 슬라이스 내 **층별 m/e-type 개수·밀도·E:I 비율** 산출 (circuit nodes 통계에서 자동).
- ✅ **검증 V2a**: 밀도·E:I(≈89:11)·m-type 비율이 Romani와 일치.

### 5. 뉴런 배치
- **circuit.zip nodes에서 슬라이스 bbox 내 세포 추출** → 각 세포의 (좌표, 방향, m/e-type) → `slice_nodes.json`.
- ✅ **검증 V2b**: 배치된 소마가 층/밀도 분포에 부합.

### 5b. (보완) me-type 매핑 + 형태 준비 — *빠졌던 단계*
- (Romani m-type, e-type) → **우리 23 me-model 폴더** 매핑(registry `by_mtype`).
- **커버리지 점검**: 슬라이스 고유 (m,e) vs 우리 23 → 빠진 조합은 **같은 m-type 내 가용 e-type로 대체**. (선택: 변이 위해 형태 복제/스케일)
- ✅ **검증 V2c**: 슬라이스의 모든 (m,e)가 우리 모델(또는 대체)로 100% 해소(로그).

### 6. 뉴런 방향성 주입
- 로드한 me-model 형태를 **소마=목표좌표로 평행이동 + quaternion→회전행렬(scipy) 적용**(`morph_transform`, `h.pt3dchange`).
- ⚠️ quaternion 순서(`w,x,y,z` ↔ scipy `x,y,z,w`) 재정렬 주의.
- ✅ **검증 V2d**: 회전 후 **총 수상돌기 길이 불변**(강한 불변식) · 소마위치==목표 · 정점축≈orientation · 3D 그림에서 정점 SO→SLM 정렬.

### 7. 연결 커넥텀 구성
- 슬라이스 세포쌍에 **`pathway_class(pre_mtype, post_mtype)` + 거리의존(σ)** → `edges={pre,post,cls}`. (Romani edges 대신 우리 규칙; 축삭 표적층 반영)
- ✅ **검증 V3**: 모든 cls ∈ 9클래스 · 연결행렬 · 수렴/발산·연결수가 생물학적으로 타당.

### 8. 시냅스 모델 적용 (Ecker 방식)
- 각 edge에 `params_table3` 9클래스 **EMS 확률 시냅스**(g·U·D·F·Nrrp·τ·E_rev) 주입.
- ✅ **검증 V4**: 9클래스 PSP/CV/STP가 기존 paired recording 검증과 일치(재사용).

### 9. 시뮬레이터 구동
- `network_lib`: build_and_place → wire_synapses → 외부구동 → 실행(고정 dt) → 분석.
- ✅ **검증 V5**: `--demo` 10분 내 완료 · raster·발화율 · **E/I 작동**.

---

## 최종 확장 (별도 트랙, 향후)
11) **MEA fEPSP**: 막횡단 전류 → 세포외 장(LSA/LFPy, 전극 격자). 12) **LTP/LTD**: ⚠️ **장기 가소성 모델 추가 필요**(현재 단기 STP만; Graupner-Brunel/Chindemi 2022). 13) **실측 MEA 데이터와 비교·검증**.

---

## 산출 코드(06_atlas/)
`1_extract_slice.py`(0·1·4b·5) · `2_build_slice.py`(5b·6) · `3_slice_connectivity.py`(7) · `4_run_and_analyze.py`(8·9) · `5_visualize_slice.py`(검증 그림). 라이브러리: `slice_io.py`·`mtype_map.py`·`morph_transform.py`·`atlas_network_lib.py`.

## 단계별 진행 약속
각 단계 끝에 **✅검증 그림/수치**를 만들어 **하나씩 확인**하고 다음으로 넘어간다.
