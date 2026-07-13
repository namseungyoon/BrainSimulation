# CA1 BSB+NEST 모델 리포지토리 심층 분석 보고서

> 분석 일자: 2026-06-08 · 분석 방식: 멀티 에이전트 워크플로우(서브시스템 7 + 교차검증 3 + 직접 검증)
> 대상 경로: `/data1/seonghwankim/workspace_studio/bsb-test`
> 분석 제외: `nest-simulator/`(벤더링된 NEST C++ 소스), `nest-build/`(빌드 산출물)

---

## 1. 총평 (Executive Summary)

이 저장소는 BSB v6(Brain Scaffold Builder)로 해마 **CA1 영역의 스파이킹 신경망을 구축**하고 **NEST로 시뮬레이션**한 뒤, **Bezaire et al. (2016, eLife) 풀스케일 CA1 모델**(원본 NEURON/HOC 소스는 `bezaire_modeldb/`)에 맞춰 검증하려는 계산신경과학 프로젝트다. 목표 자체는 명확하고 과학적으로 의미가 있으나, 현재 상태는 **연구급(research-grade)이 아니라 프로토타입/미완성 단계이며, 핵심 산출물(검증된 스파이크 출력)은 한 번도 디스크에 저장된 적이 없다.**

현재 단계 판정: **검증되지 않았고, 현 환경에서 실행 불가능하며, 사실상 2025년 9월 말 이후 동결된 프로젝트.**
- 직접 검증 결과 `.venv`에서 `import nest`가 `ModuleNotFoundError`로 실패한다. NEST 시뮬레이션 경로 전체가 **현 호스트에서 실행 불가**하며, 문서가 주장하는 모든 "검증 통과"는 **재현 불가능**하다(`.venv/bin/python -c "import nest"`).
- 소스 파일 mtime 기준 최신 실제 코드는 `run_bezaire_compliant_simulation.py`(2025-11-04) 단 하나이고, 나머지는 모두 2025-09-17~25에 집중된다. viz의 "11월 활동"은 `.DS_Store`/락파일 노이즈일 뿐이다.

가장 중요한 강점 3가지:
1. **세포 분류·개수의 충실도**: 9개 CA1 집단(Pyramidal + 8개 인터뉴런 클래스)과 풀스케일 개수(합계 338,740)가 `cellnumbers_101.dat` 및 Bezaire 논문과 정확히 일치한다(`ca1_model/parameters/bezaire_modeldb_true_connectivity.json:6-18`).
2. **연결성 추출의 데이터 기반성**: `indegree = total_connections / N_post` 공식이 ModelDB의 `fastconn.mod:187`("nconv = nconn*1.0/ncell")과 일치하며, 추출 결과 피라미드당 시냅스 ~16,850개가 논문의 ~17,000 목표와 1% 이내로 맞는다(`bezaire_modeldb_true_connectivity.json`).
3. **다운스케일 수학과 디버깅 방법론**: `build_scaled_network.py`(타입 힌트·dataclass 기반)와 `debug_*.py`(단일 뉴런 → Poisson → 소규모 망 → 풀스케일 단계적 격리)는 본질적으로 올바른 접근이다.

가장 심각한 리스크 3가지:
1. **실행 불가 + 검증 부재**: NEST 비임포트, `NestManager` 심볼 부재, 저장된 스파이크 출력 0건. 유일하게 "통과"로 문서화된 결과는 100개 피라미드 세포에 비생리적 드라이브(PV Poisson 1800 Hz, 가중치 500~1500배 fudge factor)를 가한 수작업 튜닝 산출물로, **창발적 검증이 아니라 튜닝 인공물**이다.
2. **데이터는 충실하나 사용은 부실**: 실제로 materialize되어 실행된 다운스케일은 p-preserve 모드라 **in-degree를 1/scale로 깎으면서 가중치 보정을 하지 않아** 네트워크를 침묵 쪽으로 편향시킨다. "가장 성숙한" 풀스케일 러너는 루프 버그로 발화율을 ~5배 부풀린다. "프로덕션" 스케일 러너는 이름 매핑 버그로 9개 중 8개 세포 타입에 피라미드 파라미터를 잘못 부여한다.
3. **아키텍처 부재 + 기술 부채**: 공유 라이브러리·패키지·버전관리가 전혀 없고, ~30개 독립 스크립트의 60~85%가 복붙이며, 8.96 GB의 재생성 가능한 바이너리가 트리 내에 방치되어 있다. 복붙은 이미 실제 결함(`build_full_ca1.py:76`)을 낳았다.

요약하면, **데이터 추출 코어는 올바르지만 그 위에 올라간 시뮬레이션·통합·검증·시각화·엔지니어링 레이어가 미완성이고 상호 모순적이다.** 이 저장소는 "버려진 작업대"에 가깝고, 가치 있는 보석(올바른 Bezaire 데이터 추출과 다운스케일 수식)이 그 안에 묻혀 있다.

---

## 2. 프로젝트 목적과 과학적 배경

- **BSB (Brain Scaffold Builder) v6**: 뉴런 배치(placement)와 연결성(connectivity)을 선언적 JSON 설정으로 구성하고 HDF5 스캐폴드로 컴파일하는 프레임워크. `bsb[arbor,nest,neuron]>=6.0.6`로 핀(`pyproject.toml`).
- **NEST**: 스파이킹 신경망 시뮬레이터. 본 저장소에서는 pip 패키지가 아니라 `nest-simulator/` 소스를 컴파일해 venv에 주입하는 방식(`install_nest.sh`).
- **Bezaire et al. (2016)**: 311,500개 피라미드 세포를 포함한 풀스케일 CA1 모델. 원본은 NEURON/HOC(`bezaire_modeldb/`의 `.mod`/`.hoc`, ModelDB 187604). 본 프로젝트의 **검증 기준점(ground truth)**.

목표 파이프라인: Bezaire ModelDB 원시 데이터(`conndata_*.dat`, `cellnumbers_*.dat`, `syndata_*.dat`) → JSON으로 추출 → BSB 설정으로 CA1 네트워크 구축 → NEST 시뮬레이션 → 세포 타입별 발화율을 Bezaire 목표와 비교. 목표 발화율: Pyr 0.1~5, PV 10~30, CCK/Ivy 2~15, O-LM 5~20 Hz.

**주의할 과학적 맥락**: 파라미터 도출·검증의 출처가 논문/ModelDB가 아니라 "ChatGPT"로 귀속되는 부분이 광범위하다. ChatGPT는 14개 파일에 걸쳐 67회 인용되며, `.claude/settings.local.json`에 `mcp__chatgpt-mcp__chatgpt` 권한이 실제로 부여되어 있어 LLM이 파이프라인을 구동한 운영 도구였음이 확인된다. 변환 공식 자체(`g_L=1000/Rin` 등)는 표준적이고 옳지만, "논문 정확값(논문 일치 확인)"이라는 라벨은 1차 출처로 독립 검증된 것이 아니다.

---

## 3. 전체 아키텍처와 파이프라인

핵심 문제는 **단일 파이프라인이 아니라 서로 약하게 연결된 두 개의 병렬 파이프라인**이 존재한다는 점이다.

```
                 ┌─────────────────────────── 참조 데이터 ───────────────────────────┐
                 │ bezaire_modeldb/datasets/{conndata,cellnumbers,syndata}_*.dat      │
                 └───────────────┬───────────────────────────────────────────────────┘
                                 │ extract_*/parse_* (macOS 경로 하드코딩, 일부 실행불가)
                                 ▼
                 parameters/bezaire_modeldb_true_connectivity.json  ← 유일하게 소비되는 권위 파일
                 parameters/{correct_aeif_parameters, synapses, complete_interneurons, cell_types,
                             bezaire_true(망가짐), bezaire_accurate(고아)}.json  ← 5+개 충돌
                                 │
          ┌──────────────────────┴───────────────────────────┐
          ▼ (경로 A: BSB)                                      ▼ (경로 B: 직접 NEST)
  configs/{basic,complete}/*.json                      build_scaled_network.py
          │ build_*.py → scaffold.compile()                    │ (p-preserve / indegree-preserve 수식)
          ▼                                                     ▼
  HDF5 스캐폴드 (placement + connectivity)            configs/scaled/scale0pXX_p-preserve.json
  ca1_full_scale.hdf5(6.1G, 8191셀), ...                       │ (BSB 설정 아님; 빌드 산출물)
          │                                                     ▼
          │ run_basic_simulation/run_nest_sim 만 로드   run_scaled_bezaire.py → nest.Create/Connect 직접
          │ (그나마 try/except로 폴백)                          │
          └───────────────► 나머지 13개 run_*.py는 BSB를 무시하고 JSON/인라인으로 NEST 망을 재구성
                                 │
                                 ▼
                 NEST 시뮬레이션 → 스파이크는 메모리 내 recorder에만 → stdout 출력 (디스크 저장 거의 없음)
                                 │
                                 ▼ (별개)
                 시각화: brain-viz / web(React) / electron / vue — HDF5의 placement만 렌더, connectivity 미렌더
```

이 다이어그램에서 드러나는 구조적 결함:
- **BSB가 계산한 연결성이 시뮬레이션에 거의 전달되지 않는다.** BSB의 네이티브 NEST 어댑터(`network.simulate("nest_sim")`)는 `test_bsb_nest.py:56`에서 주석 처리, `run_nest_sim.py:161-163`에서 hasattr 가드 뒤 폴백, `run_basic_simulation.py:20`에서 미사용 import로만 존재하는 **사실상 죽은 코드**다. 16개 NEST 스크립트는 모두 pynest를 직접 호출한다(스크립트당 `nest.Create` 4~17회). 결과적으로 BSB가 빌드한 그래프와 NEST가 실제 돌리는 그래프가 **조용히 분기(diverge)**한다.
- **검증 결과의 출처(provenance)가 불명확하다.** 문서가 보고하는 "통과" 발화율이 BSB-HDF5 경로에서 나왔는지, 직접-NEST scaled 경로에서 나왔는지 어느 문서도 명시하지 않는다.

---

## 4. 서브시스템별 심층 분석

### 4.1 네트워크 구축 (BSB build) — `ca1_model/scripts/build_*.py`, `configs/`, `parameters/`
- **목적**: CA1 4개 층(SO/SP/SR/SLM)을 partition으로 정의하고, RandomPlacement로 포인트 뉴런을 배치, FixedInDegree/FixedProbability/AllToAll로 연결해 HDF5로 영속화.
- **핵심 파일**: `ca1_complete_config.json`(632L, 9타입 338,740셀, 권위 설정), `build_complete_ca1.py`/`build_full_ca1.py`(같은 설정을 다른 출력 경로로), `build_scaled_network.py`(BSB 빌더가 아니라 해석적 스케일러).
- **주요 발견**:
  - **[high] `build_full_ca1.py:76`** `config['cell_types'].items()`로 순회(튜플 산출) → 형제 파일들은 `.keys()` 사용(`build_complete_ca1.py:123`, `build_scaled_ca1.py:83`). 모든 placement 조회가 broad `except`로 빠져 6.1 GB 빌드의 "Total placed"가 항상 0으로 보고됨. **복붙이 낳은 실제 결함.**
  - **[high] 세포 개수가 3중으로 충돌**: `ca1_complete_config.json`(합계 338,740, 권위)과 `build_scaled_ca1.py:41-50`의 하드코딩 `original_counts`(NGF 없는 별개 값), `build_full_ca1.py:3`의 헤더("8,191 cells, all 8 types", ~41배 차이)가 서로 다름.
  - **[high] 연결성 전략이 스케일 사다리에서 비일관**: scaled_1_50/small_test는 AllToAll(연결확률 1.0)인 반면 실제 Bezaire Pyr→PV 확률은 0.0026. complete는 FixedInDegree지만 in_degree 1~8로 임의값이며 실제 indegree(예: Pyr→O_LM 8231)와 무관. **어떤 BSB 연결 설정도 실제 Bezaire 통계를 담지 않는다**(오직 `build_scaled_network.py` 경로만 담음).
  - **[medium] AEIF 파라미터 중복·발산**: `correct_aeif_parameters.json` vs 설정 내장값 vs `cell_types.json`이 PV C_m(134.4 vs 100), V_th 등에서 불일치.
- **강점**: `build_scaled_network.py`는 깔끔하고 수학적으로 명시적(p-preserve는 edge ~ scale², indegree-preserve는 ~ scale¹). `build_basic_ca1.py`만 제대로 된 argparse/--validate를 갖춤(다른 빌더가 따르지 않은 좋은 템플릿). 층 명명(SO/SP/SR/SLM)과 배치-층 매핑은 해부학적으로 타당.

### 4.2 시뮬레이션 실행과 진화 — `ca1_model/scripts/run_*.py`, `debug_*.py`
(상세는 §5에서 별도 분석)
- **목적**: 스케일된 CA1 미세회로를 NEST로 시뮬레이션하고 타입별 발화율을 출력. ~15개 거의 동일한 러너가 각자 setup→create→connect→drive→analyze 5단계를 재구현.
- **주요 발견**:
  - **[high] 복붙 5,465 LOC, 공유 모듈 0개**. 타입별 AEIF dict가 스크립트마다 그대로 복제(`run_scaled_simulation.py`/`run_full_simulation.py`에 8개 동일 dict).
  - **[high] macOS 절대경로 하드코딩으로 다수 스크립트가 현 호스트에서 즉시 실패** (`run_fullscale_bezaire_compliant.py:82` 등).
  - **[high] "true" 가지의 가중치 문제**: §6의 과학적 평가로 정정됨 — afferent의 `weight_nS=1.0`은 플레이스홀더가 **아니라** 실제 변환값(`original_weight_uS=0.001`→1 nS). 진짜 문제는 `correct_aeif_parameters.json`의 별도 가중치 테이블이 ~1000배 차이 난다는 점.
  - **[low] `run_fullscale_true_bezaire.py:374-406` 청크 루프 버그**: `range(0,duration,10)`인데 매 반복 `nest.Simulate(50.0)` → 요청의 ~5배 시뮬레이션, 그러나 발화율은 명목 duration으로 나눠 **~5배 부풀림**.

### 4.3 Bezaire 데이터 추출 — `extract_*`, `parse_*`, `parameters/bezaire_*.json`
- **목적**: ModelDB 표 데이터를 JSON으로 변환해 연결성/시냅스 동역학 목표를 제공.
- **주요 발견**:
  - **[critical] `extract_bezaire_connectivity.py:55`의 indegree 공식이 물리적으로 불가능**: col4(네트워크 전체 총 연결수, 예 ca3→pyr=1,549,250,000)를 `synapses_per_connection`으로 나눠 세포당 수십억 개 indegree 산출. 결과 `bezaire_true_connectivity.json`은 전부 사용 불가(ECIII→Pyr=3,300,000,000).
  - **[high] 연결성 JSON 3종이 3~1000배 불일치, 단 하나만 옳고 소비됨**: `bezaire_modeldb_true_connectivity.json`(col4/N_post, CA3→Pyr=4973)만 권위이며 6개 스크립트가 소비. `bezaire_accurate_connectivity.json`은 ChatGPT 수기 추정(고아). `bezaire_true_connectivity.json`은 수십억 스케일(망가짐).
  - **[medium] 파싱한 syndata 동역학이 전혀 소비되지 않음**: `parse_syndata.py`는 실제 cell-pair별 kinetics를 충실히 추출하지만, 시뮬레이션은 하드코딩된 일반 receptor 상수 블록을 사용. 가장 충실한 데이터가 버려진다.
- **강점**: `parse_modeldb_tables.py`(pandas, 행수 검증, 상대경로)가 가장 잘 만들어진 파서지만 빌드에 연결돼 있지 않음. `bezaire_modeldb_true_connectivity.json`은 피라미드당 16,850 시냅스로 논문 ~17,000과 1% 이내 일치 — 정량 검증됨.

### 4.4 BSB↔NEST 통합 및 환경 — top-level 테스트, `install_nest.sh`, `pyproject.toml`
- **주요 발견**:
  - **[critical/검증완료] 시뮬레이션 서브시스템이 현 환경에서 실행 불가**: `.venv/bin/python -c "import nest"` → `ModuleNotFoundError`. `from bsb.simulation import NestManager` → `ImportError`(bsb 6.0.6에 없는 stale 심볼). `run_basic_simulation.py:20`은 import 시점에 크래시.
  - **[high] BSB NEST 어댑터 미실행**: §3 참조. 빌드 그래프와 시뮬레이션 그래프 분기 위험.
  - **[high] Python 버전 4중 모순**: `.python-version`=3.12, `pyproject`=`>=3.10,<3.13`, `install_nest.sh:61`=3.12 핀, 그러나 `README.md`는 "3.12는 NEST 호환성 이슈, 3.11 사용" 경고. 기본 해석 환경(3.12)이 문서가 "깨진다"고 한 바로 그 버전.
  - **[high] `install_nest.sh`는 macOS 전용**(`OSTYPE != darwin`이면 `exit 1`)인데 저장소는 Linux에서 동작 → NEST는 문서화되지 않은 경로로 빌드되었고 `uv.lock`에 캡처되지 않은 out-of-band 의존성.
  - **[high] 스모크 테스트 6종 모두 assert 0개**, try/except로 실패를 삼켜 무조건 exit 0. 검증 가치 없음.
- **강점**: `my_first_model/network_configuration.json`의 `simulations.nest_sim` 블록은 BSB 어댑터 스키마로 올바르게 작성되어 있어 네이티브 경로 복원이 어렵지 않음. `h5_inspector.py`는 bsb import 없이 BSB-HDF5 스키마를 이해하는 유용한 진단 도구. bsb 스택은 `uv.lock`에 6.0.6으로 일관 핀.

### 4.5 시각화 프론트엔드 (4종) — `brain-viz`, `web/bsb-visualizer`, `bsb-electron-viz`, `bsb-vue-viz`
- **주요 발견**:
  - **[high] 동일 기능을 4개 스택으로 4번 재구현(~4,647 LOC)**: 9타입 색상 블록이 6개 파일에 복붙, h5 로더 헬퍼가 3개 실 로더에 중복.
  - **[high] Electron은 잘못된 렌더러를 탑재**: `index.html`이 로드하는 `renderer-inline.js`는 타입당 1000셀로 캡하고 셀마다 개별 Mesh 생성(인스턴싱·포인트클라우드 없음). 풀기능 `renderer.js`(724L)는 bare ES import라 번들 없는 Electron에서 해석 불가 → **도달 불가 죽은 코드**.
  - **[high] Vue 앱은 HDF5 파싱을 전혀 하지 않음**: `Math.random()` 가짜 데이터 생성. 두 HTML 모두 데모.
  - **[medium] 어떤 뷰어도 connectivity를 렌더하지 않음** — CA1 연결성이 프로젝트 핵심인데 soma 포인트클라우드만 표시.
  - **[medium] Electron 보안 취약**: `webSecurity:false`, `--no-sandbox`, 3rd-party CDN 런타임 스크립트 로드.
- **강점**: React 앱(`web/bsb-visualizer`)이 가장 완성도 높고 관용적(three.js 라이프사이클·dispose, ResizeObserver, config JSON 구동 색상, 최신 의존성). `brain-viz`는 가장 강력한 로더(10GB 청크 스트리밍, 성능 디테일 티어). `brain-viz/vite.config.js`는 h5wasm에 필요한 COOP/COEP 헤더를 올바르게 설정.

### 4.6 문서·내러티브 — `README.md`, `CA1_*.md`, `docs/*.md`, 검증 스펙
- **주요 발견**:
  - **[high] README와 `docs/network_analysis.md`는 CA1이 아니라 일반 BSB 예제를 설명** — 1,560 base_type 셀, CA1/Bezaire 언급 0회. 프로젝트의 정문(front door)이 가장 부정확.
  - **[high] 검증 스펙이 권위 DATA_REPORT 및 실제 설정과 세포 개수에서 모순**: 검증 문서는 Ivy 2,203(vs 8,810, 4배), O-LM 251(vs 1,640, 6.5배), NGF 누락.
  - **[high] 뉴런 모델이 문서마다 3가지로 기술**: DATA_REPORT는 `aeif_cond_beta_multisynapse`, 검증 문서는 `aeif_cond_exp`(단일 시냅스인데 4-port receptor 표를 붙임 — 불가능).
  - **[high] CA3 입력 발화율이 4곳에서 200/250/0.65/10 Hz로 제각각.**
  - **[high] Phase-1 "100% 완료" 보고서가, 같은 프로젝트의 DATA_REPORT가 과학적으로 틀렸다고 한 모델(3타입 AllToAll, 음수 가중치)을 성과로 자축.**
- **강점**: `docs/modeldb_dataset_notes.md`는 1차 파일을 인용하며 README 오기를 바로잡는 가장 엄밀한 문서 — **신뢰의 닻**. DATA_REPORT는 과잉주장에도 불구하고 자기비판적으로 발견된 오류를 열거. 권위 순서가 품질과 역전(가장 잘 보이는 문서가 가장 부정확).

### 4.7 데이터 자산 — HDF5 네트워크, 출력
- **주요 발견(직접 h5py 검증)**:
  - **[high] 시뮬레이션 출력이 0건**: npz/csv/gdf/png 전무, `/tmp/nest_data` 없음. 7개 HDF5는 전부 네트워크 스캐폴드(placement+connectivity)뿐.
  - **[high] "full_scale"는 잘못된 라벨**: `ca1_full_scale.hdf5`는 6.1 GB지만 실제 **8,191셀 / 8타입 / NGF 누락**(설정은 9타입 338,740). 6 GB는 스케일이 아니라 조밀한 connectome(평균 indegree ~6,650) 때문이며 5.23 GB가 connectivity 그룹.
  - **[medium] 두 "complete" 빌드가 불일치**: `ca1_complete_network.hdf5`(2.5G)는 19 conn 타입(흥분성 pyr_to_* 누락), `ca1_full_scale.hdf5`(6.1G)는 28 타입. 2.5 GB는 열등한 폐기 빌드.
  - **[medium] 8.96 GB 바이너리가 `.gitignore` 미포함**(`*.hdf5` 무시 안 함), DVC/LFS 없음. 네트워크는 결정론적이라 완전 재생성 가능.
- **강점**: 모든 HDF5가 깔끔한 자기완결적 BSB v6 레이아웃. 66→180→332→1,303→8,191셀의 스케일 사다리는 점증 검증에 유용. `ca1_scaled_1_50.hdf5`(180셀, 29 conn 타입)가 가장 완전한 소형 connectome — 좋은 경량 fixture.

---

## 5. 시뮬레이션 진화 과정 분석 (핵심)

`run_*.py`의 명명 계보(basic→corrected→balanced→compliant→functional→perfected→true→fullscale)는 **버전관리 부재를 파일명으로 대체한 디버깅 일기**다. "침묵하는 네트워크/폭주 발화/잘못된 발화율"을 가중치·드라이브·연결성 수작업 튜닝으로 잡으려는 반복 시도이며, 상당수가 ChatGPT 오라클의 조언을 따른다.

| 단계 (파일) | 추정 문제 | 핵심 변경 | 결과/상태 |
|---|---|---|---|
| `run_basic_simulation.py` | 시작점 | BSB 스캐폴드 로드, 기본 뉴런, 무차원 대형 가중치(CA3 100, PV→Pyr -100) | NEST 3.x에서 제거된 kwargs(withgid) 사용, `NestManager` import 크래시 → 실행불가 |
| `run_nest_sim.py` | BSB 로드 실패 | from_storage 시도 후 실패 시 하드코딩 NEST로 폴백 | **BSB 시뮬레이션 포기 지점** |
| `run_scaled_simulation.py` / `run_full_simulation.py` | 독립 실행 필요 | 인라인 AEIF dict(8개) 패턴 확립 | 복붙 원형 |
| `run_corrected_simulation.py` | 파라미터 신뢰성 | JSON AEIF 로드, Bezaire g_max ×1000, 음수 억제 가중치, 배경 Poisson 800~1500 Hz | 비생리적 드라이브 |
| `run_balanced_simulation.py` | E/I 불균형 | 셀별 Poisson 수작업(PV 1800, O-LM 150 Hz), 억제 50~60% 스케일 | **문서가 "전 목표 통과"로 보고하는 유일한 런(그러나 100셀 토이)** |
| `debug_bezaire_network.py` | 왜 무발화? | DC/multisynapse/Poisson/rate×weight×indegree 스윕 | "참값 0.65 Hz는 너무 약해 발화 불가" 결론 |
| `debug_and_fix_spiking.py` | 발화 레시피 탐색 | 4단계 단계적 bring-up | `fixed_total_number+allow_multapses` 레시피 발견 |
| `run_bezaire_compliant_simulation.py` (2025-11-04, 최신) | 억제 모델링 오류 | `aeif_cond_beta_multisynapse` 4-port, **양수 가중치+음수 E_rev**(생리적으로 옳음) | 질적 도약, 최신 실제 편집 |
| `run_correct_bezaire_connectivity.py` / `run_perfected_bezaire.py` / `run_functional_bezaire.py` | 연결성 정합 | ChatGPT p-preservation `fixed_total_number` | macOS 경로 하드코딩으로 실행불가 |
| `run_scaled_bezaire.py` | 성숙한 CLI | argparse, scale knob | **이름 매핑 버그**(§6): 9개 중 8개 타입에 피라미드 파라미터 부여 |
| `run_scaled_true_bezaire.py` / `run_fullscale_true_bezaire.py` | ModelDB 충실도 | `indegree_true` 테이블 사용, 타입힌트·pathlib | **엔지니어링상 최선**이나 5배 루프 버그, 저장 결과 없음 |

**결론 — "최선"의 두 정의가 충돌**:
- **엔지니어링 품질 기준 최선**: `run_fullscale_true_bezaire.py`(외부 JSON, alias map, pathlib). 그러나 5배 발화율 버그가 있고 저장된 성공 결과가 없다.
- **보고된 결과 기준 최선**: `run_balanced_simulation.py`. 그러나 100셀 수작업 튜닝 토이로 창발적 검증이 아니다.
이 둘은 **양립 불가능한 다른 모델**(multisynapse+ModelDB-true vs aeif_cond_exp+수작업)이며, 어느 문서도 정본을 지정하지 않는다. 디렉터리만으로는 새 사용자가 어떤 스크립트를 돌려야 하는지 알 수 없다.

---

## 6. 과학적 타당성 평가

독립 재검증(ground-truth 데이터·파서·러너 직접 재독) 결과:

**충실한 부분(단단한 토대)**:
- **분류·개수**: 9개 CA1 집단과 풀스케일 개수가 `cellnumbers_101.dat`·논문과 정확히 일치(합계 338,740 + CA3 204,700 + EC 250,000).
- **연결성 추출**: `indegree=total/N_post`가 `fastconn.mod:187`과 일치, 피라미드당 ~16,850 시냅스가 논문 ~17,000과 1% 이내. `bezaire_modeldb_true_connectivity.json`은 신뢰 가능.

**Phase-1 발견에 대한 정정(과학 교차검증)**:
- afferent `weight_nS=1.0`은 플레이스홀더가 **아니다** — `original_weight_uS=0.001`→1 nS의 올바른 단위변환이다. 진짜 문제는 `correct_aeif_parameters.json:213-224`의 두 번째 가중치 테이블(g_max 0.0002~0.0019 nS)이 연결성 JSON과 ~1000배 어긋난다는 점이다.

**치명적 오용**:
- **[critical] p-preserve 다운스케일이 in-degree를 1/scale로 깎고 가중치 보정 없음**: scale 0.02에서 CA3→Pyr 유효 indegree 4973→99.5(50배 손실), 0.1에서 10배, 0.5에서 2배. `run_scaled_bezaire.py`는 가중치 스케일 기본 1.0이라 스케일 망이 ~1/scale만큼 적은 시냅스 드라이브를 받음 → **debug 스크립트가 보고한 "침묵"의 기계적 원인**. 통계적으로 옳은 preserve-indegree 모드는 존재하나 실행되지 않음.
- **[high] `run_scaled_bezaire.py:55-57` 이름 매핑 버그**: `pop.replace('cell','').capitalize()`가 'Pvbasket','Olm' 등을 만들어 JSON 키(PV_Basket,O_LM)와 불일치 → fallback이 모든 인터뉴런에 CA1_Pyramidal 파라미터를 부여. **"프로덕션" 스크립트가 9개 중 8개 타입을 잘못된 내재 동역학으로 시뮬레이션** → 타입별 발화율 비교 무효.
- **[high] `run_fullscale_true_bezaire.py:374-406` 5배 발화율 과대보고**(루프 버그).
- **[medium] afferent indegree가 소형 후집단에서 비현실적**: ECIII→Neurogliaform=58,240, ECIII→SCA=12,500(EC 전체 250,000 중 23%가 NGF 한 세포로 수렴). col4/N_post 정규화가 afferent에는 부적절(pre-population 풀을 써야 함)함을 시사 — **권위 파일조차 외부 드라이브에 대해 의심스럽다.**

**검증(미입증)**:
- 목표 발화율은 정의돼 있으나 충족 증거가 디스크에 0건. 유일한 "통과"는 `run_balanced_simulation.py`의 100셀 수작업 토이(드라이브 1800 Hz, indegree 1~6, 가중치 500~1500배 fudge, Pyr 1:3110 vs Ivy/O-LM 1:25로 E/I 비율 파괴). **이는 튜닝 인공물이지 창발적 CA1 동역학이 아니다.** 충실한 데이터 기반 망이 침묵/폭주가 아닌 theta/network 동역학을 재현한다는 증거는 없다.

**판정**: **프로토타입급, 연구급 아님, 아직 작동 미입증.** 데이터 추출 코어는 옳지만, 그 위의 다운스케일·시뮬레이션·검증이 모두 결함이 있거나 미입증이다. Bezaire(2016)의 과학적 재현으로서는 **현재 검증되지 않았고 end-to-end로 작동하지 않는다.**

---

## 7. 코드 품질 및 기술 부채

스태프 엔지니어 관점에서 **모든 구조 품질 축이 실패**한다.

- **[critical] 공유 코드 0**: `__init__.py`·패키지·교차 import 없음. 14개 러너(4,672 LOC)와 5개 빌더(873 LOC)가 동일 파이프라인을 각자 재구현. 러너의 ~60~75%(약 2,800~3,500 LOC), 빌더의 ~600 LOC가 기계적 복붙. **모든 BSB/NEST API 변경을 14곳·5~6곳에서 수정해야 함.** 복붙은 이미 `build_full_ca1.py:76` 결함을 유발.
- **[high] 파일명 버전관리**: 저장소는 git이 아니며(루트에 `.git` 없음; 유일한 VCS는 벤더링된 `bezaire_modeldb/`의 git+hg), 10개 스크립트가 'corrected/true/perfected/compliant' 형용사로 이력을 인코딩. 삭제된 것이 없어 정본 식별 불가.
- **[high] 8.96 GB 재생성 가능 바이너리가 트리 내 방치**, `.gitignore` 미포함, DVC/LFS 없음. 출력 위치도 비일관(일부 루트, `output/scaled`·`output/complete`는 빈 stub).
- **[high] 환경 재현 불가**: macOS 전용 인스톨러가 Linux에서, 잠금되지 않은 NEST, 모순된 Python 핀, **10개 파일 13곳의 `/Users/seonghwankim` macOS 절대경로**.
- **[high] 스모크 테스트 검증 가치 0**(assert 0개, 실패 삼킴). 과학 스크립트용 회귀 테스트 전무. 14개 중 1개만 출력 저장.
- **[medium] 파라미터 단일 출처 부재 — 5개+ 충돌**: 연결성 3종(true 망가짐 / accurate 고아 / modeldb_true 권위) + AEIF 3종. 추가로 `complete_interneurons.json`(실제 nS 스케일 g_max를 담은 **가장 충실한 시냅스 파일이지만 고아**)과 `synapses.json`(최신 스크립트가 소비)까지 합치면 충돌은 5개 이상.

**리팩터링 청사진(최소 변경)** — 단일 `ca1/` 패키지:
```
ca1/
  config.py      # 단일 설정/파라미터 로더 (권위 JSON만 참조, 충돌 파일 삭제)
  builder.py     # NetworkBuilder: BSB compile + 예외를 삼키지 않는 stats
  runner.py      # NestRunner: kernel/create/connect/devices (단일 정본 러너)
  analysis.py    # 발화율 계산 + 스파이크/메트릭 JSON 영속화
  cli.py         # config 파일 구동, 14개 러너·5개 빌더를 각 1개로 축약
```
이것만으로 3,000+ 중복 LOC 제거 가능. **우선순위**: (1) git init + `.gitignore`에 `*.hdf5` + 바이너리 삭제, (2) `ca1` 패키지·정본 러너 추출, (3) 파라미터/설정을 단일 출처로 통합, (4) 프론트엔드 1개(React) 선택.

---

## 8. 데이터 자산 현황 (직접 검증)

| 파일 | 크기 | 셀 수 | conn 타입 | 산출 스크립트 | 비고 |
|---|---|---|---|---|---|
| `ca1_model/output/full_scale/ca1_full_scale.hdf5` | 6.1 GB | **8,191 / 8타입(NGF 누락)** | 28 | `build_full_ca1.py` | "full"이지만 목표의 2.4%, 구조적 불완전 |
| `ca1_complete_network.hdf5` (루트) | 2.5 GB | 8,178 | 19(흥분성 누락) | `build_complete_ca1.py` | 열등·폐기 빌드 |
| `ca1_scaled_1_50.hdf5` (루트) | 272 MB | 180 | 29 | `build_scaled_ca1.py` | 가장 완전한 소형 connectome |
| `ca1_model/ca1_simple_network.hdf5` | 71 MB | 1,303 / 3타입 | 4 | (스크립트 없음) | Phase-1 "71MB 성공" |
| `network.hdf5` (루트) | 6.5 MB | base/top | A_to_B | BSB 튜토리얼 | **CA1 아님, 고아** |
| `ca1_small_test.hdf5` (루트) | 2.6 MB | 332 | 2 | `test_small_ca1.py` | 미완성 stub |
| `ca1_model/output/test/quick_test.hdf5` | 1.8 MB | 66 | 2 | `quick_sim_test.py` | 스모크 stub |

- **합계 8.96 GB**, 모두 2025-09-17 생성. **시뮬레이션 출력은 단 한 건도 없다**(직접 검증). 문서가 광고하는 338,740셀 풀스케일 모델은 **어떤 산출물에도 실현되지 않았다** — 최대가 8,191셀.
- **데이터 위생 불량**: 빌드 상태(성공/폐기/stub)를 표시하는 manifest·명명규칙·README 없음. 수동 h5py 검사 없이는 성공 빌드와 폐기 빌드를 구분 불가.

---

## 9. 리스크 및 이슈 (심각도 순)

| 심각도 | 이슈 | 근거 | 영향 |
|---|---|---|---|
| **critical** | NEST 비임포트, `NestManager` 부재 → 시뮬레이션 실행 불가 | `.venv` import 검증; `run_basic_simulation.py:20` | 모든 "검증 통과" 재현 불가 |
| **critical** | 저장된 시뮬레이션 출력 0건; 유일한 "통과"는 100셀 수작업 토이 | find 결과 0; `CA1_Model_Documentation...md:127-145` | 과학적 검증 미입증 |
| **critical** | p-preserve가 in-degree 1/scale 손실 + 가중치 보정 없음 | `build_scaled_network.py:95-100`; `run_scaled_bezaire.py:80-104` | 스케일 망 침묵 편향 |
| **critical** | `extract_bezaire_connectivity.py:55` 수십억 indegree | `bezaire_true_connectivity.json` | 잘못 채택 시 파국 |
| **high** | `run_scaled_bezaire.py:55` 이름 매핑 → 8/9 타입에 피라미드 파라미터 | `:55-57` vs `correct_aeif_parameters.json` | 타입별 비교 무효 |
| **high** | `run_fullscale_true_bezaire.py` 발화율 5배 과대 | `:374-406` | 보고 수치 신뢰 불가 |
| **high** | `build_full_ca1.py:76` `.items()` 버그로 placement 통계 0 | vs `:123 .keys()` | 6.1 GB 빌드 stats 손상 |
| **high** | full_scale에서 NGF 무단 누락(8/9 타입) | h5py 직접 검증 | 구조적 불완전 |
| **high** | 공유 코드 0, ~60~85% 복붙, git 부재 | `wc -l`, `find __init__.py` | 유지보수성 붕괴 |
| **high** | macOS 전용 인스톨러·10개 파일 절대경로·Python 핀 모순 | `install_nest.sh:24`, grep | 재현 불가 |
| **high** | 문서가 세포수·뉴런모델·CA3 rate를 2~4가지로 모순 기술 | §4.6 | 파라미터 출처 불신 |
| **medium** | afferent indegree 비현실(ECIII→NGF 58,240) | `bezaire_modeldb_true_connectivity.json` | 권위 파일도 외부드라이브 의심 |
| **medium** | 5개+ 파라미터 파일 충돌, 가장 충실한 것이 고아 | `complete_interneurons.json` 등 | 동역학이 파일 선택에 의존 |
| **medium** | 8.96 GB 비-gitignore 바이너리, DVC/LFS 없음 | `.gitignore`, `du` | 저장소 비대 |
| **medium** | viz 4종 중복·2종 비작동·connectivity 미렌더 | §4.5 | 노력 낭비 |
| **low** | `.DS_Store`×5, 이중 락파일, 빈 `CLAUDE.md`, 고아 `network.hdf5` | find | 클러터 |

---

## 10. 권장사항 (우선순위 순)

**즉시 (실행 가능성 복구)**
1. **환경 재현 경로 확립**: Linux용 NEST 빌드 절차 문서화 또는 컨테이너화. `import nest`가 성공하도록 만들고, `.python-version`/`pyproject`/installer/README의 Python 버전을 하나로 통일.
2. **`run_basic_simulation.py:20`의 `NestManager` import 제거**(bsb 6.0.6에 없는 심볼). 10개 파일의 `/Users` 절대경로를 `Path(__file__).resolve().parents[N]`로 교체(이미 `run_fullscale_true_bezaire.py`에 정답 존재 — 백포트).
3. **정본 러너 1개 지정** 후 나머지 명시적으로 폐기 표기. 엔지니어링상 `*_true_bezaire` 계열 권장하되 §6의 버그(5배 루프, 이름 매핑) 선수정.

**단기 (과학적 검증 강화)**
4. **다운스케일 보정**: scaled 설정을 `--preserve-indegree`로 재생성하거나, p-preserve 유지 시 가중치를 1/scale로 보정해 셀당 총 컨덕턴스 보존. scaled와 full-scale 발화율 일치를 검증 후에야 "다운스케일 충실"이라 주장.
5. **스파이크·메트릭 영속화**: 모든 러너가 스파이크 + 타입별 발화율 JSON을 저장. 데이터 기반 연결성 + 생리적 드라이브에서 목표 발화율이 창발하는지 회귀 검증 구축. **수작업 튜닝 단일 스크립트 결과를 Bezaire 검증으로 제시하지 말 것.**
6. **afferent 정규화 재검토**: 소형 후집단(NGF/SCA)의 비현실적 indegree(58,240/12,500)는 col4/N_post가 afferent에 부적절함을 시사 — pre-population 풀 기반 정규화로 재도출.

**중기 (구조·데이터·문서 정리)**
7. **`ca1/` 패키지 추출**(§7 청사진), 14개 러너·5개 빌더를 config 구동 단일 entrypoint로 축약.
8. **파라미터 단일 출처**: `bezaire_true_connectivity.json`(수십억)·`bezaire_accurate_connectivity.json`(고아) 삭제. `complete_interneurons.json`/`syndata`의 충실한 per-pair kinetics를 실제 receptor 설정에 연결.
9. **git init + `.gitignore`에 `*.hdf5`/`.DS_Store`/`.venv`** 추가, 8.96 GB 바이너리 삭제(결정론적 재생성) 또는 DVC/LFS 이관. 고아 `network.hdf5` 제거.
10. **문서 정리**: README를 CA1용으로 재작성, `docs/modeldb_dataset_notes.md` + `bezaire_modeldb_true_connectivity.json`를 권위로 명시하는 onboarding 노트(빈 `CLAUDE.md`)를 채움. 모순된 세포수·뉴런모델·CA3 rate를 단일 값으로 수렴.
11. **시각화 1개(React `web/bsb-visualizer`)로 통합**, 공유 h5-loader 모듈 추출, 나머지 3개 삭제. 필요 시 connectivity 렌더링 추가.

---

## 11. 미해결 질문 / 추가 조사 필요

1. **풀스케일(338,740셀) 네트워크가 실제로 빌드 시도되었는가?** 모든 산출물이 ~8,191셀에서 멈추고 NGF가 누락된 것은 메모리/패킹 실패를 시사하는가, 아니면 의도적 축소인가?
2. **NEST는 원래 어떤 Linux 경로로 빌드되었는가?** `install_nest.sh`는 Linux에서 실행 불가인데 `nest-build/` 산출물이 존재한다. 현재 `.venv`에서 NEST가 사라진 이유는?
3. **검증 문서의 "통과" 발화율은 어느 스크립트·어느 설정에서 나왔는가?** BSB-HDF5 경로인가 직접-NEST scaled 경로인가? 어느 문서도 산출 스크립트를 명시하지 않는다.
4. **정본 CA3/ECIII 드라이브는 무엇인가?** ModelDB DegreeStim 10 Hz vs 검증문서 200 Hz vs DATA_REPORT가 옳다고 한 ~0.65 Hz — 이 선택이 E/I 체제 전체를 좌우한다.
5. **afferent indegree에 N_post 나눗셈이 맞는가?** ECIII→NGF=58,240 같은 값은 외부 드라이브 정규화가 틀렸을 가능성을 강하게 시사한다.
6. **multisynapse(`*_compliant`/`*_true`) 가지가 실제로 Bezaire 일치 발화율을 낸 적이 있는가?** 저장된 결과가 없어 docstring의 성공 주장을 확인할 수 없다.
7. **(사용자 확인 필요)** 이 프로젝트를 재개할 계획인가, 아니면 동결 상태로 두는가? 재개라면 위 §10 즉시 항목부터, 보존이라면 §10-9(바이너리 정리)와 onboarding 노트만으로 충분하다.

---

### 부록: 분석 방법론 및 신뢰도

- 7개 서브시스템 분석 + 3개 교차검증(과학적 타당성/아키텍처/완전성 비판자) 에이전트가 파일:라인 근거를 첨부해 병렬 분석.
- 완전성 비판자가 Phase-1의 4개 고-레버리지 버그(`build_full_ca1.py:76`, 5배 루프, 수십억 indegree, GABA E_rev -65/-60)를 독립 재확인 — 모두 확정.
- 본 보고서 작성 전, 가장 결정적인 run-state 주장(NEST 비임포트, `NestManager` 부재, full_scale 8,191셀/NGF 누락, 시뮬레이션 출력 0건)을 분석자가 `.venv`/h5py로 **직접 재검증**해 확정.
- 주의: `.venv`는 본 분석 세션 중(2026-06-08 13:59) 에이전트의 `uv` 호출로 생성되었다. 분석 초기(13:50)에는 부재했으므로, "venv 부재" 초기 관측과 "venv 존재·NEST 비임포트" 후속 관측은 시점이 다를 뿐 둘 다 정확하다.
