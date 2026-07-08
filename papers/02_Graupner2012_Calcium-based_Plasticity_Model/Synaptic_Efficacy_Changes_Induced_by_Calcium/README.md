# Synaptic Efficacy Changes Induced by Calcium

> 논문 첫 Results 섹션. **Fig 1(A–D)** 전체에 해당. 논문 읽는 순서대로, 이 섹션에
> 등장하는 개념을 하나씩 **설명 + 그래프**로 정리한다. (한글 라벨 matplotlib PNG)

## 이 섹션이 말하는 것 (요약)

단일 시냅스가 pre·post 활동전위 열을 받을 때, 그 상태를 **시냅스 효능 변수 ρ(t)** 로
기술한다. ρ 의 시간변화는 **Eq.1** 로 주어진다:

```
τ dρ/dt = -ρ(1-ρ)(ρ*-ρ)          # ① 활동 없을 때: cubic → 두 안정상태
          + γ_p (1-ρ) Θ[c-θ_p]    # ② 강화: 칼슘 c 가 θ_p 초과 시
          - γ_d ρ  Θ[c-θ_d]       # ③ 약화: 칼슘 c 가 θ_d 초과 시
          + Noise(t)              # ④ 활동 의존 잡음
```

- **①** cubic 항이 쉴 때(활동 없음) ρ 에 **두 안정상태**를 준다: `ρ=0`(DOWN, 낮은 효능),
  `ρ=1`(UP, 높은 효능). 경계(불안정 고정점)는 `ρ*=0.5`. → **쌍안정**.
- **②③** 칼슘 의존 신호전달(kinase→강화, phosphatase→약화)을 단순화한 항. 칼슘이
  강화문턱 θ_p / 약화문턱 θ_d 를 넘으면 ρ 를 위/아래로 민다.
- **④** 칼슘이 문턱 위일 때만 켜지는 백색잡음 → 상태 사이 확률적 전이를 만든다.
- **칼슘 c(t)**: pre 스파이크는 지연 D 후 C_pre 만큼, post 는 즉시 C_post 만큼 점프하고
  τ_Ca 로 지수 감쇠·선형 합산 (Fig 1A).
- 자극 중 ρ 의 **평균 정착값 ρ̄** 가 경계 ρ*=0.5 보다 크면 **LTP**, 작으면 **LTD** (Fig 1B,D).

## 항목별 그래프 로드맵

| # | 파일 | 개념 | 논문 | 상태 |
| --- | --- | --- | --- | --- |
| 1 | `1_bistable_synapse.py` | ① cubic 항 → 두 안정상태(DOWN/UP)와 경계 ρ*=0.5 | Fig 1D(rest) | ✅ |
| 2 | `2_calcium_transient.py` | 칼슘 c(t): 점프·감쇠·합산, 문턱 초과 시간(그늘) | Fig 1A | ⬜ |
| 3 | `3_time_above_threshold.py` | 칼슘 진폭 → 문턱 초과 시간 → 평균 정착값 ρ̄ | Fig 1B | ⬜ |
| 4 | `4_state_transitions.py` | 60 스파이크@1Hz + 잡음 → DOWN↔UP 확률적 전이 | Fig 1C | ⬜ |
| 5 | `5_potential_landscape.py` | 쉴 때 이중우물 vs 자극 중 단일우물(ρ̄), LTP/LTD 기준 | Fig 1D | ⬜ |

## 실행

```powershell
& "C:\Users\SYNAM-OFFICE\.conda\envs\ca1sim\python.exe" `
  "papers\02_Graupner2012_Calcium-based_Plasticity_Model\Synaptic_Efficacy_Changes_Induced_by_Calcium\<script>.py"
```

핵심 라이브러리 `plasticity_model.py`(상위 폴더)의 `drift·potential·integrate_rho·
calcium_trace·time_above_thresholds` 를 재사용. 파라미터: SI Table S1 **DP 세트**.
