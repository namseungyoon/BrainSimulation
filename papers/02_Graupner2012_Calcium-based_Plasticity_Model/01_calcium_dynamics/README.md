# 01_calcium_dynamics — 칼슘 트레이스 c(t)

논문: Graupner & Brunel (2012), Results 첫 문단 + **Fig 1A / Fig 2A**.

가소성 모델의 **입력 신호**인 칼슘 농도 `c(t)` 를 pre/post 스파이크 시각으로부터
계산한다. 이후 모든 단계(문턱 초과 시간 → 효능 변화 → STDP)가 이 트레이스 위에 쌓인다.

## 모델

```
c(t) = Σ_pre  C_pre  · exp(-(t - t_pre - D)/τ_Ca) · Θ(t - t_pre - D)   # pre: 지연 D 후 점프
     + Σ_post C_post · exp(-(t - t_post)  /τ_Ca) · Θ(t - t_post)       # post: 즉시 점프
```

- **C_pre / C_post**: 각 스파이크가 만드는 칼슘 점프 (θ 기준 정규화, 무차원)
- **D**: pre 유발 칼슘 전이의 지연 (NMDAR 활성 지연 반영). 본문 Fig1A: D=13.7 ms
- **τ_Ca**: 칼슘 감쇠 시간상수 (수 ms ~ 수십 ms)
- 전이들은 **선형 합산**된다 → pre·post 순서/간격 Δt 에 따라 봉우리가 달라진다.

## 스크립트

| 파일 | 내용 | 그림 |
|---|---|---|
| `1_calcium_trace.py` | pre-post 쌍(Δt=-20, +20 ms)의 c(t), θ_p/θ_d 선 표시 | `figures/1_calcium_trace.png` |
| `2_amplitude_delay.py` | C_pre·C_post·τ_Ca·D 변화가 파형/봉우리에 주는 영향 | `figures/2_amplitude_delay.png` |

## 실행

```powershell
& "C:\Users\SYNAM-OFFICE\.conda\envs\ca1sim\python.exe" `
  "papers\02_Graupner2012_Calcium-based_Plasticity_Model\01_calcium_dynamics\1_calcium_trace.py"
```

## 파라미터

**SI Table S1 의 DP-curve 세트** 사용 (출처 확정):
`C_pre=1, C_post=2, τ_Ca=20 ms, D=13.7 ms, θ_d=1.0, θ_p=1.3`.
`plasticity_model.PARAM_SETS["DP"]` (= `"demo_fig2"` 별칭).
