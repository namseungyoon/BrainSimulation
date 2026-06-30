# 03_synapses — 시냅스 모델: 8개 파라미터를 찾는 6단계 절차

Ecker et al. (2020) Figure 2. **시냅스 모델(biexponential + TM + 확률방출)은 파라미터 8개**:
`E_rev, τ_rise, τ_decay, U_SE, D, F, N_RRP, ĝ`.
이 8개를 정하는 **6단계 절차**를 단계별 파일(번호)로 구현한다.

## ★ 8개 파라미터 × 6단계 (마스터 표)
| 단계 | 파일 | 논문 | 이 단계가 찾는 파라미터 | 개수 | 상태 |
|:---:|------|------|------------------------|:---:|:---:|
| 1 | `1_innervation.py` | Fig.2-1, §2.1 | (해부: 축삭-수상돌기 분포) | 0/8 | ✅ |
| 2 | `2_num_synapses.py` | Fig.2-2, §3.3 | (해부: 연결당 시냅스 수) | 0/8 | ✅ |
| 3 | `3_biexp_conductance.py` | Fig.2-3, §2.3 식1–4 | **E_rev, τ_rise, τ_decay** | 3/8 | ✅ |
| 4 | `4_tm_stp.py` | Fig.2-4, §2.4 식5–6 | **U_SE, D, F** | 3/8 | ✅ |
| 5 | `5_stochastic_mvr.py` | Fig.2-5, §2.5 식7–10 | **N_RRP** | 1/8 | ✅ |
| 6 | `6_calibrate_ghat.py` | Fig.2-6, §2.6 식11 | **ĝ** (peak conductance) | 1/8 | ✅ |
|  |  |  | **합계** | **8/8** | |

> 8개 = 3(단계3)+3(단계4)+1(단계5)+1(단계6). **단계 1·2는 해부 검증**이라 8개에 안 들어감.

## 보조(검증/시각화) 파일 — 단계 번호 N-M 형식
실행 파일의 출력 그림명은 파일명과 동일(예: `3_biexp_conductance.py` → `figures/3_biexp_conductance.png`).
| 파일 | 소속 단계 | 역할 |
|------|:--:|------|
| `4-1_reproduce_fig5.py` | 4 | 9클래스 STP 종합 검증 (Fig.5) |
| `4-2_corrections_demo.py` | 4 | §2.7 칼슘·온도 보정 데모 |
| `5-1_stochastic_vs_deterministic.py` | 5 | 확률 포함 vs 미포함 비교 |
| `6-1_animate_calibration.py` | 6 | ĝ 보정 과정 GIF |
| `6-2_calibration_filmstrip.py` | 6 | ĝ 보정 과정 정적 PNG |
> 번호 파일은 서로 import 하지 않음(파이썬은 숫자-시작 모듈 import 불가). 공유 로직은 라이브러리에 둠.

## 라이브러리 모듈 (번호 X — 다른 파일이 import)
`params_table3.py`(Table 3 파라미터, 단일 진실원천) · `synapse_pair.py`(연결 빌더) ·
`tm_model.py`(simulate_tm) · `paired_recording.py`(load_pc·place_synapses·measure_psp + 상수)
> 파이썬은 숫자로 시작하는 모듈을 import 못 함 → 실행 단계만 번호, 라이브러리는 번호 없이.

## 실행
```powershell
conda activate ca1sim
python papers/01_Ecker2020_CA1_synaptic/03_synapses/3_biexp_conductance.py
```
각 파일은 같은 폴더 `figures/` 에 결과 PNG 저장.
