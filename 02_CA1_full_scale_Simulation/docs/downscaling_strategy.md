# CA1 Downscaling Strategy for NEST

## 목표
단일 워크스테이션에서 베제어(2016) CA1 모델의 통계(평균 수렴도·합선 수·전도도 균형)를 유지한 채 실행 가능한 축소 규칙을 정의한다.

## 기본 표준 (Full scale)
- Cell counts: `NumData=101` → Pyr 311 500, interneuron 총 ~27 k
- 외부 입력: CA3 204 700, ECIII 250 000 포아송. 논문 theta-control
  조건은 각 afferent cell의 독립 Poisson `DegreeStim = 0.65 Hz`이다.
  `parameters.hoc`의 `DegreeStim = 10 Hz`는 ModelDB launcher 기본값일 뿐,
  Bezaire 2016 Figure 6의 7.8 Hz theta 검증 입력으로 쓰면 안 된다.
- Indegree/contact convergence: `ConnData=430` / `per_cell` 기준
  CA3→Pyr 5 985 contacts × 2 synapses = 11 970 synapses/cell,
  ECIII→Pyr 1 299 contacts × 2 synapses = 2 598 synapses/cell.
  이 값들이 논문 Table 1의 CA3→Pyr 3.73e9, ECIII→Pyr 8.09e8
  synapse totals와 일치한다.
- `synapses_per_connection` S: `conndata_430.dat` 참고. `ConnData=211`은
  ModelDB launcher default일 수 있으나 논문 Table 1 full-scale gate에는 쓰지 않는다.

## 축소 원칙
### A. 확률(p) 보존 구조 축소 (권장 기본)
- scale factor `s ∈ (0,1]`
- `N_pre = floor(s · N_pre_full)`, `N_post = floor(s · N_post_full)`
- `p = C_full / (N_pre_full · N_post_full)`
- `C = p · N_pre · N_post`, `Edges = C · S`
- NEST: `fixed_total_number`, `allow_multapses=True`, weight·delay·reversal 고정

### B. indegree 보존 (필요 시)
- `indegree_full = C_full / N_post_full`
- `Edges = indegree_full · N_post · S`
- NEST: `fixed_total_number`, `allow_multapses=True`
- 평균/분산 유지가 필요하면 모멘트 매칭 적용

### C. 외부 입력 압축
- CA3/ECIII generator 은행 M (예: 2 000) 구성
- `rate_per_gen = 0.65 Hz · (N_pre_full / M)` for theta-control. 다른
  `DegreeStim` 값은 조건별 provenance를 남긴 진단 실행에서만 사용한다.
- 각 Pyr이 K개 스트림 선택 (예: 40) → 평균 입력 유지, 겹침률 `K²/M` 낮게 설정(≪1)

### 모멘트 매칭(선택)
- 압축비 κ ⇒ 평균 유지: `w' = w / κ`
- 분산 유지: `w' = w / sqrt(κ)`
- 타협: `w' = w / κ^α (α∈[0,1])`

### 고정해야 할 것
- 가중치(nS), 지연(ms), rise/decay, reversal, S 값, E/I 비율, layer 배치

## 검증 체크
1. 평균 전도도: `μ ∝ Σ r · indegree · w · A`
2. 전도도 분산: `σ² ∝ Σ r · indegree · w² · B`
3. 목표 발화율: Pyr 0.1–5 Hz, PV 10–30 Hz

## 구현 경로
- 축소 계산 + NEST 연결 자동화를 위한 `ca1_model/scripts/build_scaled_network.py` 작성
- 설정 플래그: `scale_factor`, `preserve_indegree`, `compress_afferents`, `moment_matching`
- 예시: `uv run ca1_model/scripts/build_scaled_network.py --scale 0.02 --compress-ca3 4000 --compress-ec 5000 --compress-overlap 0.0005 --output ca1_model/configs/scaled/scale0p02_p-preserve.json`
