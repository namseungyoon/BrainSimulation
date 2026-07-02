# 02_Graupner2012 — 칼슘 기반 시냅스 가소성 모델

Graupner & Brunel (2012) *PNAS* 109(10):3991–3996,
**"Calcium-based plasticity model explains sensitivity of synaptic changes to
spike pattern, rate, and dendritic location"** 을 **bottom-up으로 하나씩** 재현한다.

이 논문은 **순수 수리모델**이다 (NEURON 불필요). 칼슘 농도 `c(t)` 와 시냅스 효능
`ρ(t)` 두 변수의 ODE 만 numpy/scipy로 적분한다. 프로젝트 최종 목표(LTP/LTD 재현)에
맞춰 가소성 규칙을 **나중에 NEURON CA1 네트워크에 결합**할 수 있도록 핵심
라이브러리(`plasticity_model.py`)를 스텝 단위 인터페이스로 설계했다.

## 핵심 방정식

```
τ dρ/dt = -ρ(1-ρ)(ρ*-ρ)              # 이중우물(bistable): DOWN(ρ=0)/UP(ρ=1)
          + γ_p(1-ρ)Θ[c-θ_p]         # 강화
          - γ_d ρ Θ[c-θ_d]           # 약화
          + Noise(t)                 # 활동 의존 잡음
c(t): pre 스파이크→지연 D 후 C_pre 점프, post→즉시 C_post 점프, τ_Ca 지수감쇠·선형합산
w = w0 + ρ(w1-w0),  ρ*=0.5 (두 안정상태 경계)
```

## ★ 단계 로드맵 (마스터 표)

| 단계 | 폴더 | 논문 | 이 단계가 다루는 것 | 상태 |
|---|---|---|---|---|
| 1 | `01_calcium_dynamics` | Fig 1A / 2A | pre·post 스파이크 → 칼슘 트레이스 c(t) | ✅ |
| 2 | `02_efficacy_bistable` | Fig 1D | 결정론적 ρ ODE, 이중우물 포텐셜 U(ρ), ρ*=0.5 | ⬜ |
| 3 | `03_thresholds_time` | Fig 1B / 2B | θ_p·θ_d 초과 시간 비율 α_p, α_d | ⬜ |
| 4 | `04_noise_transitions` | Fig 1C | 잡음 포함 확률적 ρ 궤적, DOWN↔UP 전이, w readout | ⬜ |
| 5 | `05_stdp_pairs` | Fig 2 | Δt 스윕 STDP 곡선의 다양성 (DP, DPD, P …) | ⬜ |
| 6 | `06_bursts_triplets` | Fig 3 | post 버스트 / 트리플렛·쿼드러플렛 | ⬜ |
| 7 | `07_firing_rate` | Fig 4 | 발화율 의존성 (주기 + Poisson) | ⬜ |
| 8 | `08_dendritic_location` | Fig 5 | 칼슘 진폭 감소 → LTP→LTD 전환, rescue | ⬜ |
| 9 | `09_analytical` | SI Appendix | 분석적 전이확률(Fokker-Planck/OU) vs 시뮬 (검증) | ⬜ |

각 단계 폴더 = `README.md` + 번호 스크립트(`1_*.py`, `2_*.py`) + `figures/`.

## 라이브러리 — `plasticity_model.py` (단일 진실 소스)

- `Params` / `PARAM_SETS` — 모델 파라미터 (데이터셋별 명명 세트)
- `calcium_trace()` · `drift()` · `potential()` · `time_above_thresholds()`
- `integrate_rho()` (Euler–Maruyama) · `synaptic_strength()`
- `class Synapse` — 이벤트 기반 스텝 인터페이스 (향후 NEURON 결합용)

## ⚠️ 파라미터 출처

모든 파라미터는 **SI Appendix (Corrected Nov 28, 2012) Table S1/S2/S3** 에서 확정
입력 완료. `plasticity_model.PARAM_SETS` 에 명명 세트로 정리:
- **Table S1** → STDP 곡선 7세트: `DP, DPD, DPD_prime, P, D, D_prime, BCM`
- **Table S2** → 실험 피팅 3세트: `hippo_slice_Wittenberg2006, hippo_culture_Wang2005, cortex_Sjostrom2001`
- **Table S3** → Fig S1 예시 2세트: `figS1_DPprime, figS1_DP`

표의 τ 단위는 **초(sec)** → 코드에선 ms(×1000)로 저장.

## 실행

```powershell
& "C:\Users\SYNAM-OFFICE\.conda\envs\ca1sim\python.exe" `
  "papers\02_Graupner2012_Calcium-based_Plasticity_Model\01_calcium_dynamics\1_calcium_trace.py"
```

그림은 각 단계의 `figures/` 에 스크립트명과 동일한 이름의 PNG로 저장된다.

## 출처 주석 규약

각 스크립트 상단 docstring 에 `Source: Graupner & Brunel (2012) §… / Fig …` 형식으로
논문 근거를 밝힌다. 수치는 소스코드/실행 로그와 대조해 확정한다(기억 의존 금지).
