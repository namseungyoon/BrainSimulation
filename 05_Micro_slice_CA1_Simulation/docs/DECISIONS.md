# 설계 결정 로그 — micro-slice CA1

> 결정·근거를 시간순으로 기록. 수치는 소스/실행 로그와 대조해 확정.

## 2026-08-13

### D1. 신규 트랙 · 백지 시작
- **결정**: 05_Micro_slice_CA1_Simulation을 신설. 01_Like_slice 산출물(코드·데이터)을 재사용하지 않고 백지에서 재구성.
- **근거**: like-slice보다 작은 규모의 CA1로 별도 테스트. `../shared`·`../Models`는 공용 인프라라 경로 참조.

### D2. 기하 — footprint 800 × 500 µm × 두께 400 µm
- **결정**: layer축 800 µm × 수직 500 µm × 두께 400 µm.
- **근거**:
  - 두께 400 µm = 일반 in vitro 급성 해마 슬라이스.
  - MEA 전극 3개(200 µm 간격) 스팬 = 400 µm. 바깥 전극이 절단 가장자리에 가깝지 않도록 **사방 200 µm 여유** → 800 µm.
  - 200 µm 여유의 물리적 근거 = **fEPSP 기능적 기여 반경 ≈ SC 자극 동원 반경 ~200 µm**(활성화 세포+근접장 보존).
  - 800×800이 아닌 800×500: "전극 2~3개"에 충실한 최소 구성(수직축은 근접장 폭만).

### D3. fEPSP는 광역 적분 신호 → 마이크로 조직은 정규화 비교
- **사실**(소스 `01/12_lfp`, 그림 `E4b_disk_contrib.png` 확정): 전극 1개당 국소 PC(150 µm) 중앙 **626**개 vs 유효 기여 Neff 중앙 **11,662**개(전체 PC의 ~74%). fEPSP는 국소가 아니라 넓은 면적을 적분.
- **함의**: 작은 조직은 먼 기여자가 잘려 **절대 fEPSP 진폭이 축소**됨.
- **결정**: 절대 mV가 아니라 **정규화 fEPSP·상대 LTP slope**로 비교, 필요 시 스케일 보정(like-slice E9와 동일 철학).
- **주의**: Neff 11,662는 "완전 정렬·동기·동일깊이" 이상화 상한(지터 −36%). 유발 fEPSP의 실질 기여는 SC로 활성화된 세포로 제한.

### D4. 전세포 완전형태 모델 (대표 축소 아님)
- **결정**: 창 안 모든 뉴런을 각자 실제 형태학 biophysical 모델로 인스턴스화.
- **근거**: 조직이 작아 계산 가능. like-slice의 4대표 축소보다 충실.
- **자산**: morphology_library.zip(2.49GB)·single_cell_model_library.zip 확보 → 실제 형태학 우선. `../Models` 20 번들은 보조/즉시검증.

### D5. fEPSP 3기법 확정
- **결정**: `lib/mea_forward.py`에 PSA·LSA(Holt&Koch 1999)·MoI(Ness 2015 3층) 구현(01 소스 이식·재구현).

### D6. 폴더 구조 — 하이브리드 카테고리(인덱스) + lib 단일 import
- **결정**: 최상위 카테고리 인덱스(`01_tissue`~`04_experiments`) + 지원 폴더(config/docs/lib/data/env/scratch). import 코드는 `lib/`만. 스크립트는 최상위에 두지 않음(env/ 추적, scratch/ 제외).
- **안전조치**: 루트 `.gitignore`에 `Models/` 추가(BBP 라이선스).

## 2026-08-14

### D7. 폴더 번호 통일 — 카테고리 1~4 + 하위 1-based 순번
- **결정**: 최상위 카테고리 = 단계 1~4(`01_tissue`~`04_experiments`). 각 카테고리 내부 하위폴더는 **1-based 순번**. `01_tissue`를 0-based→1-based로 리네임(`1_inspect`·`2_bbox`·`3_atlas_prep`·`4_vectorize`·`5_layers`). 02_neurons·03_network는 이미 1-based.
- **근거**: 01_tissue만 0으로 시작해 불일치·혼선. Romani 원단계(0~9)는 PLAN 표의 참조 컬럼으로만 유지.

### D8. LTP/LTD 유도 프로토콜 — 가능성 확인·E8에 추가
- **확인**: `../shared/mechanisms`에 Graupner-Brunel 칼슘 가소성 mod 3종(GBPlasticity{Syn,StpSyn,StpProbSyn}) 존재 → **유도 가능**.
- **추가**: E8-HFS(100 Hz × 1s → LTP; Bliss&Lømo 1973, Bliss&Collingridge 1993) · E8-LFS(1~3 Hz × 7~15min → LTD; Dudek&Bear 1992, Mulkey&Malenka 1992).
- **이점**: 마이크로 조직이라 장시간 LFS(7~15분) 시뮬 현실적. **측정 = 정규화 fEPSP slope**(절대 아님). **주의**: Graupner 파라미터 정량 검증 필요.

## 2026-08-18

### D9. 창·전극 배치 확정 (Stage 2 bbox) — `층관통_v1`
- **도구**: `01_tissue/2_bbox/window_picker.html`(인터랙티브 배치기, Artifact)로 확정. 원본 config = `config/window_layout.json`.
- **창**: 종축(proximodistal) 500 × 층관통 800 × 두께 400 µm, 각도 0°. 중심 물리 [2972.2, 612.2, 4656.4]µm. slice400 프레임(seed+단위벡터)으로 3D 복원.
- **전극**: **3×1 층관통 깊이 프로파일**(간격 200µm, 직경 10µm). **E1(SO)·E2(SP) 기록 · E3(SR) 자극** → SR 자극 → 층별 fEPSP/CSD.
- **근거**: 전극축을 슬라이스 긴 방향에 두어 기여반경(~200µm) 확보(D2 연장). 층관통 800으로 SO→SP→SR 전 층 관통.
- **⚠️**: UI 세포수는 표본(7,000/17,647) 추정 — 정확값은 Stage5 전세포 배치에서. 반영: `config/microslice.yaml`(window)·`config/mea.yaml`(layout).

---

## 2026-08-20

### D10. MEA 전극 면 뒤집기 (아래→위)
- **변경**: MEA 면을 두께축 아래(w=−200µm)에서 **위(w=+200µm)**로 뒤집음. 실제 MEA 장착 방향 반영.
- **반영**: `config/window_layout.json` `mea_face_w_um: -200 → +200`, 전극 E1/E2/E3 xyz 재계산. `make_picker.py` 생성식도 `+box.Lw/2`로. 3D/배치 그림은 config 구동 → 재실행 시 위 면 반영.

### D11. 자극 방식 — 현재(05)=후시냅스 point 직접자극 · 미래=06 섬유모델
- **현재(05)**: 국소 SC 시냅스 **직접 자극**(기존 논문 후시냅스 point 방식). 전극 E1/E2/E3 = 기록(위 표면). E1~E11 즉시 진행.
- **미래**: 전시냅스 섬유 기반 SC 자극(CA3 상류경로 자극·섬유 모집·전도전달)은 별도 트랙 **`06_Presynaptic_SC_Stimulation/`** 에서 재구현 → 완성 시 05에 소급 적용. 논문·특허 후보.

---

## 미결 (진행 시 결정)
- 층 경계 실측값(Stage4). 창 원점·회전(Stage1). SC 파라미터(E2). MEA 배치 상세(E4b).
