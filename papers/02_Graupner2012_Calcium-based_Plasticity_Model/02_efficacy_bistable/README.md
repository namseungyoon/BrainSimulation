# 02_efficacy_bistable — 시냅스 효능 ODE (Eq.1) 이해

칼슘 트레이스 `c(t)`(단계 1)를 입력으로 받아, 시냅스 효능 변수 **ρ** 가 어떻게
움직이는지를 지배하는 **Eq.1** 을 항별로 분해한다. 논문 Fig 1D 에 해당.

```
τ dρ/dt = -ρ(1-ρ)(ρ*-ρ)        # (1) cubic: 활동 없을 때 두 안정상태(DOWN=0, UP=1)
          + γ_p (1-ρ) Θ[c-θ_p]  # (2) 강화: c>θ_p 면 ρ 를 위로
          - γ_d ρ  Θ[c-θ_d]     # (3) 약화: c>θ_d 면 ρ 를 아래로
          + Noise(t)            # (4) 잡음 (단계 2에서는 끔, 단계 4에서 다룸)
```

## ★ 단계 (마스터 표)

| 패널 | 논문 | 무엇을 보나 | 결과 |
| --- | --- | --- | --- |
| A | Eq.1 (1)항 | c=0 일 때 우변=cubic → 고정점 ρ=0,0.5,1 | 0·1 안정, 0.5 불안정 (쌍안정) |
| B | Fig 1D | 그 항의 포텐셜 U(ρ) | 두 우물(DOWN·UP)+장벽 ρ*=0.5 |
| C | Eq.1 (2)(3)항 | 고정 칼슘 3수준에서 ρ(t) | c>θ_p→0.617, θ_d<c<θ_p→0, c<θ_d→정지 |
| D | Fig 1C(inset) | 활동 없을 때 여러 ρ0 완화 | ρ*=0.5 경계로 0 또는 1 로 latching |

## 실행

```powershell
& "C:\Users\SYNAM-OFFICE\.conda\envs\ca1sim\python.exe" `
  "papers\02_Graupner2012_Calcium-based_Plasticity_Model\02_efficacy_bistable\1_efficacy_ode.py"
```
출력: `figures/1_efficacy_ode.png` + 콘솔 요약.

## 핵심 결과 (실행 로그 대조)

- cubic 항 고정점: ρ=0(안정), ρ*=0.5(불안정), ρ=1(안정) → **쌍안정**이 기억의 근거.
- 강화 구동(c=1.5 > θ_p) 정착 ρ ~ **0.617** = 이론 `γ_p/(γ_p+γ_d)` 와 정확히 일치 → 0.5 위 → **UP 운명**.
- 약화 구동(θ_d < c=1.15 < θ_p) 정착 ρ ~ **0.015** → 0.5 아래 → **DOWN 운명**.

## 모델

- `-dU/dρ = τ dρ/dt` 관계로 포텐셜 U(ρ) 정의 (라이브러리 `potential()`).
- 파라미터: SI Table S1 **DP 세트** (`plasticity_model.PARAM_SETS["DP"]`).
- 적분: `integrate_rho()` (Euler–Maruyama, 여기선 `noise=False` 결정론적).

## 다음 단계

단계 3(`03_thresholds_time`): 실제 스파이크 프로토콜에서 c(t) 가 θ_p/θ_d 위에
머무는 **시간 비율**(α_p, α_d)이 (2)(3)항의 구동량을 정한다 → STDP 곡선의 뿌리.
