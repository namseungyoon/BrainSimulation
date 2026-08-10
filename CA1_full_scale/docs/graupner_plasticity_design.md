# Graupner-Brunel (2012) 칼슘 기반 가소성 → 풀스케일 CA1 NEST-GPU 구현 설계서

대상 시냅스: CA3→Pyramidal (Schaffer collateral, AMPA_fast) | 근거: paper-02 구현 + NEST-GPU 포크 소스 + ca1 배관 + 검증 문헌을 소스/로그 대조로 확정 (design-graupner-plasticity 워크플로, 5 agents).

---

## 1. 결론 요약 (Feasibility verdict)

**충실한(faithful) Graupner-Brunel = HARD. 실용 권장안 = 2-티어 전략.**

- **핵심 충돌:** NEST-GPU fork의 시냅스 Update ABI는 **스파이크 이벤트 구동·nearest-neighbour 페어 전용**이며, per-synapse 영속 상태가 `float weight` **단 하나**뿐이다(`conn12b/16b`는 packed 2-word). Graupner는 칼슘 `c`와 효능 `ρ` 두 상태를 **비활동 구간에서도** 이중우물로 drift시켜야 한다. ABI가 이걸 원천 미제공.
- **권장 접근:**
  1. **Tier 1 (지금 당장, EASY-MODERATE):** 내장 pair-STDP를 CA3→Pyr에 배선, 배관(SynGroup 생성·부착·weight 스냅샷·HDF5·프로토콜 훅) 완성. 배관 80% 존재.
  2. **Tier 2 (충실 목표, HARD):** conn_struct 확장(`c`,`ρ`,`t_last`) + Graupner를 이벤트 구동 해석적 재구성 맵으로 재정식화해 device kernel 이식. paper-02 `integrate_rho`(σ=0)를 오라클로 회귀.
  3. 검증 게이트는 두 티어 동일 — like_slice NEURON 트랙 통과값(**+39.5% vs 엄격대조 −0.07%**)을 회귀 앵커로 재사용. ★2026-08-07 정정: 옛 앵커 +70.4% / −0.1% 는 **폐기된 기울기 정의**(20~80% 구간 표본 회귀) 값이다. 현행 정의는 **20%·80% 첫 교차점 기반** `slope = 0.6·amp/(t80−t20)` 이며, **같은 원자료**를 새 정의로 재계산하면 +39.5% / −0.07% 다(`13_net_fepsp/figures/_mea_ltp_plastic_csv/ltp_index.csv` 전극 #3).

> "graupner.cu 하나 추가"로 끝나지 않는다. 진짜 칼슘 모델은 **core 데이터구조 수술**이다.

---

## 2. 모델 사양 (paper-02 소스 대조 확정)

**칼슘 동역학:** `dc/dt = -c/τ_Ca`; pre spike는 지연 D 후 `c += C_pre`, post spike는 즉시 `c += C_post`.

**효능 ρ (이중우물, Eq.1):**
```
τ dρ/dt = -ρ(1-ρ)(ρ*-ρ) + γ_p(1-ρ)Θ[c-θ_p] - γ_d·ρ·Θ[c-θ_d] + Noise
```
Θ strict `>`; 고정점 ρ=0(DOWN)/0.5(불안정)/1(UP), ρ∈[0,1]. 구동 평형(σ=0): c>θ_p→ρ≈0.617; θ_d<c<θ_p→ρ≈0.015.
**readout:** `w = w0 + ρ(w1-w0)`, `w1 = b·w0`.

**CA1 파라미터 세트 `hippo_slice_Wittenberg2006`** (`plasticity_model.py:104-107`, `GBPlasticitySyn.mod:47-59`):

| 파라미터 | 값 | 비고 |
|---|---|---|
| C_pre | 1.0 | |
| C_post | 0.275865 | 단일 post는 θ_d 못 넘김 → LTD-only 원인 |
| τ_Ca | 48.8373 ms | |
| D | 18.8008 ms | pre 지연 |
| θ_d / θ_p | 1.0 / 1.3 | 고정 |
| γ_d / γ_p | 313.0965 / 1645.59 | |
| σ | 9.1844 | device 1차 이식은 σ=0 권장 |
| τ | 688.355 s | 매우 김 → §7 리스크 |
| b | 5.28145 | w1/w0 |
| β | 0.7 | 초기 DOWN 비율 |

주의: transmission 블록(AMPA/NMDA, NMDA_ratio 0.71)은 Chindemi2022 style이지 Graupner 아님 → NEST-GPU에선 **뉴런 모델(user_m2..m7) AMPA 포트가 컨덕턴스 담당**, syn_model은 `w`만 갱신.

---

## 3. NEST-GPU 구현

**Device Update ABI:** `__device__ void XxxUpdate(float* w, float Dt, float* param)`. 호출은 스파이크 시에만 — pre-spike(Dt<0, 최근 post 1개), post-spike(Dt>=0, 최근 pre 1개). **quiescent dt엔 호출 안 됨. param은 syn-group 공유.**

**Tier 1 (pair-STDP 등록, STDP 미러, 9 파일 터치):** `syn_model.h`(enum/switch/name 배열/class), `syn_model.cu`(CreateSynGroup), 신규 `graupner.h/.cu`, `CMakeLists.txt`. Python 레이어는 generic이라 편집 불필요.

**Tier 2 (진짜 칼슘, HARD):** per-synapse 상태 `c/ρ/t_last` 3개 필요 → conn_struct 확장(신규 변형 + 템플릿 스택 전체 스레딩 + reset kernel). 칼슘 c는 이벤트에서 `c·exp(-Dt/τ_Ca)+jump`로 정확 복원; ρ는 닫힌형 없음 → threshold-crossing 구간별 piecewise 적분. σ=0 시작. **최대 리스크: 연속 per-synapse 상태 진화 부재(nearest-neighbour hook).**

**Fallback:** Tier 2 실패 시 Tier 1 채택하되 칼슘 malleability(단일=LTD-only, doublet=causal LTP) 미재현 한계 명기.

---

## 4. ca1 배관 변경 (파일:라인 확정)

`synapse_group` 속성은 marshalling/ABI에 이미 배선됨. 빠진 것: group 생성 / syn_spec 삽입 / readback / 프로토콜.

| 단계 | 파일:라인 | 변경 |
|---|---|---|
| Group 생성 | `sim/gpu_backend.py:1014`(__init__) + `:1413`(build) | `CreateSynGroup("graupner")` + `SetSynGroupParam`로 §2 값 주입 |
| CA3→Pyr AMPA에만 부착 | `gpu_backend.py:1484-1546`(afferent syn_spec 4곳) | `synapse_group = gid if (CA3 and post==Pyramidal and receptor==AMPA_fast) else 0` |
| 프로토콜 spike train | `gpu_backend.py:1717-1784`(`_set_literal_source_spike_trains`) | CA3 source spike_times를 none/tetanus/TBS/Δt-pairing 분기 (`literal_source_graph` 필요) |
| weight readback | `gpu_backend.py:~1865` 신규 + `:2048` | `GetConnections`→`GetConnectionStatus` → `{source_idx,target_idx,weight_nS}` |
| SimResult | `types.py:445-457` | `synaptic_weights` 필드 |
| HDF5 기록 | `cli.py:443` + validate reader `:606-619` | `synaptic_weights` group (source_idx/target_idx/weight_nS) |
| config/CLI 훅 | `config.py` + `cli.py:733` | `--stim-protocol` (또는 env `CA1_STIM_PROTOCOL`) |

주의: per-synapse 가소성엔 `literal_source_graph` 필수(compound는 pre-cell 정체성 소실). afferent weight는 weight_compensation 미적용 → 진짜 per-synapse nS 스냅샷.

---

## 5. 검증 프로토콜 & 게이트

관측량 = 평균 CA3→PC weight 변화% = `100·(mean_after/mean_before − 1)`.

**프로토콜:** (a) 단일경로 Δt 페어링 sweep(single vs doublet, 60 pair@1Hz) — 메커니즘 체크. (b) TBS: 4 pulse@100Hz × 5 burst × 200ms(5Hz theta) — 네트워크 LTP 유도.

| Gate | 기준 | like_slice 선례 |
|---|---|---|
| 1 엔진 정확성 | GPU ρ ↔ Python `integrate_rho`(σ=0) `|Δρ|<1e-3` | NEURON 트랙 max~3e-4 통과 |
| 2 STDP malleability | single=LTD-only, doublet=causal LTP crossover | 통과 (Tier 1 STDP로는 불가) |
| 3 TBS LTP | plastic ≥ +20%, 엄격대조(γ=0) `|Δ|`≤5% | plastic **+39.5%** vs 대조 **−0.07%** (전극 #3, 현행 교차 기울기 정의). 옛 표기 +70.4% / −0.1% 는 폐기된 회귀 기울기 정의 값 |
| 4 주파수 의존(선택) | 저율 LTD, 고율 LTP, crossover ~5–15Hz | |

실행: baseline(`--stim-protocol none`) vs post(`tbs`) 시드 공유 2회 런 → (source_idx,target_idx) 정렬 후 per-synapse Δw. **엄격 γ=0 동일모델 대조만 Gate 3 카운트.**

---

## 6. 단계별 계획

| Phase | 내용 | 게이트 | 공수 |
|---|---|---|---|
| P0 | 배관 완성 (Tier 무관): SynGroup·부착·weight 스냅샷·HDF5·프로토콜 훅 | weight HDF5 라운드트립, static 불변 | 0.5–1일 |
| P1 | 단일-시냅스 Graupner 유닛테스트(Python), Wittenberg2006 STDP 곡선 재현 = Gate1·2 오라클 | 기존 자산 재사용 | 0.5일 |
| P2a | **Tier 1** NEST-GPU 등록 + build + smoke | 컴파일·weight 변화 관측 | 1–2일 |
| P2b | **Tier 2** conn_struct 확장 + event-driven 해석 map(σ=0) device kernel | GPU↔Python `|Δρ|<1e-3` | 1–2주 (HARD) |
| P3 | 풀스케일 CA3→PC LTP 런(TBS, baseline vs post) | Gate 2/3/4 | 2–3일 |

권장: **P0→P1→P2a(Tier1로 end-to-end 관통)→P2b(Tier2 충실)→P3.**

---

## 7. 리스크 & 미해결 질문 (사용자 결정)

1. **Tier 1로 충분한가 vs Tier 2 필수인가** — 칼슘 malleability(Gate 2)가 endgame(MEA fEPSP↔칼슘 LTP/LTD)에 필수면 Tier 2 core 수술(1–2주) 감수. **최대 갈림길.**
2. **τ=688s vs 런 길이 불일치** — rho0 초기화(0.0/0.5/β=0.7)와 프로토콜 스케일 결정 필요. 미결정 시 Gate 3 미달 위험.
3. **conn_struct 확장의 fork 유지보수 비용** — upstream rebase 충돌. plastic만 확장 struct로 분기할지.
4. **σ 노이즈 이식 여부** — 1차 σ=0 결정론 확정, 필요 시 device Random123 후속.
5. **Transmission 소유권** — user_mX AMPA 포트가 CA3→PC에 생리적으로 타당한지 확인.
6. **compound vs literal_source_graph** — per-synapse 가소성엔 후자 필수, 풀스케일 메모리 스케일 확인.
7. **pre 지연 D의 event-map 흡수** — Tier 2에서 c 재구성 시 D offset 수동 처리(미처리 시 Gate1 위상 오차).

---

## 참조 파일
- `papers/02_Graupner2012_Calcium-based_Plasticity_Model/plasticity_model.py`(:104-107 CA1 세트), `validate_gbmod.py`
- `shared/mechanisms/GBPlasticitySyn.mod`(:47-59 Wittenberg2006, σ=0)
- `like_slice_CA1/13_net_fepsp/{stdp_verify.py, mea_experiment.py:126-130, ltp_compare.py}` (cross-track 오라클/프로토콜/대조)
- NEST-GPU fork(WSL): `nest-gpu/src/{syn_model.h,syn_model.cu,stdp.h,stdp.cu,get_spike.h,rev_spike.h,rev_spike.cu,conn12b.h,conn16b.h,connect.h,CMakeLists.txt}`, `pythonlib/nestgpu.py`
- ca1 배관: `src/ca1/sim/{gpu_backend.py,nestgpu_api.py}`, `src/ca1/{types.py,cli.py,config.py}`, `src/ca1/params/synapses.py:328-343`
