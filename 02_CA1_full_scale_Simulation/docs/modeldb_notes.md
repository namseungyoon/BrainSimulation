# ModelDB 데이터 해석 및 현재 정리

## 1. SimTracker 원본 파일 구조

### 1.1 `cellnumbers_101.dat`
- 첫 줄의 정수는 포함된 **cell group 개수**.
- 이후 각 행은 `cell_group mechanism count gid_flag external_flag` 형식.
- `count` 값이 베제어(eLife 2016) 풀스케일 CA1 모델의 세포 수와 정확히 일치함.
- `external_flag = 1` 행은 외부 입력(예: CA3, ECIII), `0`은 CA1 내부 뉴런.

### 1.2 `conndata_101.dat`
- 첫 줄은 행 개수 선언.
- 각 행 형식: `pre post weight number_of_connections synapses_per_connection`.
- `weight`: µS 단위의 g_max (접촉 1개당 전도도) → NEST/BSB에서는 **nS로 변환**(×1000).
- `number_of_connections`: 네트워크 전체에서 생성된 connection 객체 총합. **세포당 값이 아니므로** 반드시 `N_post`로 나눠 indegree 계산 필요.
- `synapses_per_connection`: 한 pre→post pair 당 형성되는 시냅스 접촉 수 (= BSB `synapses_per_pair`).
- delay 정보는 이 파일에 없고, SimTracker의 SynData 및 연결 생성 hoc 파일에서 정의됨.

## 2. 파서 스크립트
- 위치: `ca1_model/scripts/parse_modeldb_tables.py`
- 기능: pandas를 이용해 `cellnumbers`/`conndata`를 읽고, nS 변환·indegree·synapses per cell 등을 계산.
- CLI 옵션으로 입력 경로, CSV 출력, 누락 팝에 대한 처리 여부 설정 가능.
- 현재 샌드박스에서는 pandas import 시 신호 종료 → 로컬 환경에서 `.venv/bin/python` 등으로 실행 필요.

## 3. 주요 계산식
- `indegree_connections = number_of_connections / N_post`
- `total_synapses = number_of_connections × synapses_per_connection`
- `indegree_synapses = total_synapses / N_post`
- `weight_nS = weight_µS × 1000`

## 4. STP(Tsodyks–Markram) 관련 정리
- 베제어 원본 모델은 STP 파라미터(U, τ_rec, τ_fac)를 사용하지 않음 → ModelDB/SimTracker에도 해당 값 없음.
- CA3→Pyr, Pyr→PV, PV→Pyr, Pyr→OLM, OLM→Pyr 등 경로의 STP 파라미터는 **Ecker et al. (Hippocampus 2020) Live Papers** supplementary data에서 추출 필요.
- NEST에서는 `tsodyks2_synapse` 등에 U/τ_rec/τ_fac를 직접 매핑 가능. weight는 conductance 기반으로 변환 후 적용.

## 5. 남은 과제
1. 파서 결과를 사용해 BSB/NEST 설정의 indegree·synapses_per_pair·weight를 논문 수준으로 갱신.
2. delay/SynData 정보는 ModelDB의 `syn*.dat` 및 hoc 스크립트를 확인하여 반영.
3. Ecker 2020 자료에서 STP 값을 수집해 주요 경로에 순차 적용 후 시뮬레이션 검증.
