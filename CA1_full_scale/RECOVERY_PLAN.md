# CA1 프로젝트 복구 설계 보고서

본 보고서는 다섯 개의 독립 리뷰 트랙(sim, scale, morph, fidelity, restructure)을 단일 의사결정 문서로 통합한다. 목표는 Bezaire et al. (2016, eLife e18566) full-scale CA1 모델을 BSB v6 + 포인트 스파이킹 네트워크로 재현·검증하는 프로젝트를 재구조화하고 복구하는 것이다. 결론을 먼저 제시하면: **현재 레포는 정적 해부(cell counts, layer geometry, synapse-count connectivity, intrinsic ephys, 실측 g_max)를 충실히 추출했으나, 논문의 실제 과학적 현상(intrinsic 7.8 Hz theta + 세포타입별 phase preference)을 추구하는 동역학·런타임 계층은 미완성이며 상호 모순적이다.** 복구는 "재구조화 후 회복(restructure-then-recover)"이며, 시뮬레이터를 NEST GPU로 교체하여 풀스케일을 GPU에서 직접 돌리는 것이 핵심 전략이다.

---

## 1. 하드웨어·현황 요약

### 1.1 검증된 하드웨어 (재확인 완료)
- **GPU**: 3x NVIDIA A40, 각 46068 MiB (약 46 GB), 합계 약 138 GB GPU 메모리. driver 580.159.04, CUDA 12.4, compute capability 8.6 (Ampere).
- **CPU/RAM**: 32 코어, 251 GB 시스템 RAM.
- **MPI**: mpirun/mpicc 존재 (멀티 GPU 파티셔닝 전제 충족).
- **툴체인**: uv 0.10.10, Python 3.12.11.

### 1.2 검증된 런타임 현황
- **NEST import 실패**: `uv run python -c "import nest"` -> `ModuleNotFoundError: No module named 'nest'` (재확인). `bsb.simulation.NestManager`는 bsb 6.0.6에 부재하는 stale symbol이며 `run_basic_simulation.py:20` import에서 크래시한다.
- **Arbor 설치됨**: `arbor 0.11.0` import 성공 (재확인). 폴백 후보로 즉시 사용 가능.
- **영구 출력 0건**: 프로젝트 역사상 디스크에 persist된 시뮬레이션 출력(spikes/rates)이 단 한 건도 없다. 유일한 "passing" 결과는 100-cell hand-tuned toy이며 emergent 결과가 아니다.
- **In-tree 바이너리**: `find . -name '*.hdf5'` 합계 **8.4 GB** (재확인). 현재 최대 단일 파일은 `ca1_complete_network.hdf5` 2.4G; 과거 보고된 6.1GB `ca1_full_scale.hdf5`(8,191 cells, NGF 누락)는 현 시점 트리에 없거나 정리되었을 수 있으나, 빌드 스크립트의 NGF-drop 버그 자체는 여전히 유효한 결함이다.

### 1.3 핵심 진단 한 줄 요약
정적 해부 계층은 신뢰 가능(논문 일치), 동역학/런타임/검증/엔지니어링 계층은 깨졌거나 미완성. 시뮬레이터 lock-in은 사실상 0(NEST가 import조차 안 되고 BSB sim 계층은 이미 우회됨)이므로, 시뮬레이터 교체 비용이 매우 낮다.

---

## 2. 시뮬레이터 선택 (다중 GPU)

대상 워크로드: 포인트 뉴런 conductance-based AdEx 네트워크. 모델은 정확히 하나 — `aeif_cond_beta_multisynapse` (4 receptor ports: AMPA_fast/slow E_rev=0, GABA_fast/slow E_rev<0), 연결은 `nest.Connect(..., syn_spec={"receptor_type": N})`, afferent는 `poisson_generator`, 기록은 `spike_recorder`. 풀스케일: 338,740 뉴런 + 약 5.19e9 synapses.

### 2.1 후보 비교표

| 후보 | 성숙도 | 멀티 GPU(1 네트워크 분할) | AdEx·다중수용체 | 풀스케일 적재(5.19B syn) | 마이그레이션 비용 | BSB 연동 |
|---|---|---|---|---|---|---|
| **NEST GPU v2.0** (Feb 2026) | 높음 | **가능** (MPI 1 rank/GPU + RemoteConnect) | **동일 모델명 verbatim** (tau_rise/tau_decay/E_rev 배열, receptor_type 포트) | **가능** (약 1.73B syn/GPU, 46GB 내) | **매우 낮음** (import 교체, 0-based 포트, n_receptors at Create) | 없음(직접 pynestgpu) |
| Arbor 0.11.0(설치됨)/0.12.x | 높음, 활발 | 가능 (MPI, 1 GPU/rank) | AdEx built-in(v0.12, 2025-04), 다중수용체 point synapse는 덜 검증 | 가능 | 중간~높음(파라미터/4-포트 재매핑) | **네이티브** (bsb[arbor]) |
| GeNN/PyGeNN v5.4 | 높음 | **불가**(설계상 단일 GPU) | 완전 커스텀 가능 | **불가**(46GB 단일카드 초과) | 중간(커스텀 모델 정의) | 없음 |
| Brian2CUDA v1.0a7 / Brian2GeNN | alpha / 베타 | **불가**(단일 GPU 선택만) | 가능 | **불가** | 중간 | 없음 |
| CoreNEURON+GPU | 높음 | 가능 | cable model 지향(point 과잉) | 가능하나 부적합 | 높음(OpenACC/NVHPC) | 부분 |
| ML SNN(snnTorch/Norse/BindsNET/SpikingJelly) | 높음(but 다른 목적) | 부분 | **부적합**(LIF-class, surrogate-gradient) | N/A | N/A | 없음 |

### 2.2 1순위 권장: NEST GPU (nest/nest-gpu v2.0)
근거(두 축 동시 우승):
1. **마이그레이션 비용 최저**: 동일 모델명 `aeif_cond_beta_multisynapse`을 verbatim 구현. `Create`/`Connect`/`poisson_generator` 동사 동일. **핵심**: NEST GPU는 프로젝트가 이미 채택한 "inhibitory port 가중치는 양수여야 한다"(positive weight + negative E_rev) 규약을 명시적으로 강제 — 기존 corrected 스크립트의 가중치 부호가 그대로 전이된다. correct_aeif_parameters.json의 g_L/C_m/V_th 값을 재유도할 필요 없음.
2. **풀스케일 멀티 GPU 적재가 유일하게 가능**: 5.19B synapses는 단일 46GB 카드에는 안 들어가나(8-20 B/syn 기준 42-104 GB 총량), 3x A40(138 GB)에 분할하면 약 1.73B syn/GPU. NEST GPU는 1M neurons/1B synapses를 11GB 2080 Ti 단일 카드에서 이미 실증했고(약 1.7x per card), 클러스터에서 4M/24B, 1024 GPU weak-scaling 입증. **풀스케일이 runnable해지며 downscaling이 불필요해진다** — 깨진 p-preserve 로직을 은퇴시킬 수 있음.

### 2.3 폴백: Arbor (bsb[arbor], 이미 설치됨)
BSB 네이티브, 진짜 멀티 GPU(MPI), AdEx built-in(2025-04~), 멀티노드 미래가 깨끗함. 단점: AdEx point cell에 대한 다중수용체 conductance synapse 경로가 NEST GPU의 목적 설계 multisynapse 모델만큼 검증되지 않아, 4-port AMPA/GABA 매핑과 파라미터 포트를 신중히 재구현해야 함(마이그레이션 비용 높음). GeNN/PyGeNN은 **다운스케일 단일 GPU 빠른 실험에만** 사용(풀스케일 불가).

### 2.4 마이그레이션 스케치 (NEST GPU primary, run_fullscale_bezaire_compliant.py를 템플릿으로)
1. **빌드**: nest-gpu v2.0를 CUDA 12.4 / compute 8.6 대상으로 소스 빌드, MPI 활성화. 깨진 NEST 설치는 재사용 금지. `import nestgpu as ngpu` 검증.
2. **커널 init**: `ResetKernel()`/`SetKernelStatus` -> ngpu 커널 init, time resolution 0.1 ms, rng seed. OpenMP thread 설정 제거(GPU가 병렬 담당).
3. **뉴런 생성**: `ngpu.Create('aeif_cond_beta_multisynapse', count, n_receptors=4)` 후 SetStatus로 AEIF 파라미터 + E_rev/tau_rise/tau_decay 배열. correct_aeif_parameters.json 값 유지.
4. **수용체 포트**: receptor_type을 1-based -> 0-based로 시프트({1,2,3,4}->{0,1,2,3}). GABA 포트에 **양수 가중치 유지**.
5. **연결**: `ngpu.Connect(...)` 직접 매핑. 풀스케일에서는 fixed-indegree 규칙 선호(각 rank가 local 연결만 빌드).
6. **Afferent**: poisson_generator를 per-source rate로(per-target loop 금지 — 과거 5x rate-inflation 버그 재현 방지).
7. **기록**: `CreateRecord`/`GetRecordData`로 spikes를 디스크에 persist(현재 출력 0건 결함 해소).
8. **런치**: `mpirun -np 3 python run_ca1_nestgpu.py`, 1 rank/A40, cross-rank projection은 RemoteConnect.
9. **첫 마일스톤(de-risk)**: 1/50 다운스케일을 1 GPU에서 먼저 검증(non-silent, 합리적 rate) -> 그 다음 3 GPU 풀스케일.

### 2.5 미해결 질문 (시뮬레이터)
- NEST GPU v2.0의 정확한 bytes-per-synapse(16 vs 20 B)와 46GB 카드 최종 헤드룸.
- driver 580.159.04 + CUDA 12.4 + compute 8.6에서 nest-gpu v2.0 클린 빌드 여부.
- pyramidal이 뉴런의 92%를 차지하므로 3-way population 파티셔닝 시 cross-GPU(RemoteConnect) spike 트래픽이 충분히 낮게 유지되는지.

---

## 3. 스케일 방법론 + 검증 하네스

### 3.1 현재 p-preserve의 정량 확인된 실패 모드
`build_scaled_network.py:97-98`은 `scaled_total = probability_full * pre.scaled_size * post.scaled_size`를 계산하므로 edges ∝ scale^2, in-degree K ∝ scale. `configs/scaled/scale0p1_p-preserve.json`에서 확인: **모든 쌍에서 K_scaled/K_full = 0.100** (scale=0.1). 예: PV_Basket->Pyramidal K 14.4->1.4, Axo->Pyramidal 25.7->2.6, O_LM->Pyramidal 21.8->2.2, CA3->Pyr 4973->497. **가중치/구동 보정이 전혀 없음**(compute_connection lines 95-100은 count만 재조정) -> mu와 sigma^2 동시 붕괴 -> silent network. `--preserve-indegree` 분기는 K를 보존하나 역시 가중치 보정이 없어, 이제 더 sparse하고 더 correlated된 recurrent pool에서 working point를 잘못 설정 -> over-drive. **두 모드 모두 mean-field-correct하지 않음.**

### 3.2 원칙적 다운스케일 (van Albada, Helias & Diesmann 2015, PLoS Comput Biol 11(9):e1004490)
**보존 가능한 것 (1차 통계)**: per-population mean firing rate, 그리고 total synaptic input current의 mean mu와 variance sigma^2(= mean-field "working point"). 방법:
- in-degree K 보존(가능하면), 또는 K가 k=K_scaled/K_full로 축소되면 recurrent 가중치를 **J -> J/k** (mean-preserving, J ∝ 1/K)로 스케일.
- mu 복원을 위해 per-type DC current I_dc 추가.
- 외부 balanced-Poisson rate(nu_ext_E, nu_ext_I)를 조정해 sigma^2의 부족분을 채움.
- afferent: per-target 유입 spike rate를 일정 유지 — required rate = K_aff_full * 0.65 Hz.

**보존 불가능한 것 (제목 정리)**: pairwise-averaged correlation의 시간 구조가 effective connectivity W=K*J와 one-to-one 대응. N을 줄이면 J ∝ 1/K가 올라가 internal variance가 팽창하고, 이를 상쇄하려면 external variance가 줄어 결국 0 floor(Eq.16)에 도달 — 그 아래로는 correlation 보존 불가. 따라서 **pairwise correlation, population synchrony, finite-size oscillation amplitude, 안정/불안정(진동) 경계가 다운스케일에서 일반적으로 보존되지 않는다.** 이 모델의 관심 현상(emergent theta/gamma synchrony)은 정확히 2차/correlation 현상이므로, **다운스케일은 headline 결과를 충실히 재현할 수 없다.**

### 3.3 GPU 풀스케일이 다운스케일을 불필요하게 만드는가 — 그렇다(주력 경로)
NEST GPU 풀스케일이 138 GB에 적재 가능하므로, theta/gamma/phase에 관한 모든 주장은 풀스케일에서 권위를 가진다. 따라서 **2-tier 전략**:
- **TIER 1 (권위적)**: 풀스케일(338,740 CA1 + 204,700 CA3 + 250,000 EC)을 3x A40에서 실행. theta power, peak frequency(목표 7.8 Hz), phase preference, cross-frequency coupling에 관한 **모든** 주장의 유일한 근거.
- **TIER 2 (빠른 반복 전용)**: 다운스케일은 wiring/parameter/per-type mean rate/정성적 phase 디버깅에만 사용. oscillation 결과 주장에 절대 사용 금지. 빌드는 in-degree-preserving + mean-field-compensated 방식.

`--preserve-indegree` 토글은 단일 원칙 경로로 교체: in-degree K 항상 보존을 기본으로, p-preserve는 `--debug-broken` 플래그 뒤로 격리(loud warning). 정량 비교는 scale 0.2-0.3 권장(0.1은 wiring/debug만), van-Albada external-variance floor 위반 또는 per-type N<50 / K<5이면 hard refusal.

### 3.4 검증 하네스 설계 (`ca1.validation` 순수함수 패키지)
입력: `spikes: Mapping[cell_type -> list[np.ndarray spike times s]]`, `meta: SimMeta(duration_s, dt_s, n_cells_per_type, scale, crop_first_ms=50, lfp_proxy)`.

**지표 (metrics.py)**: mean_rates(all + active-only, 첫 50ms crop), cv_isi(AI 체제 ~1), fano_factor, population_synchrony(chi), spike_density_function(Gaussian kernel), welch_psd, band_power_peak(theta=5-10 Hz, gamma=25-80 Hz), phase_preference(Hilbert phase, 0 deg = trough), ei_ratio.

**논문 기준값 (targets.py, Bezaire 2016에서 하드코드)**:
- theta_peak_hz = **7.8**, theta_band=(5,10), gamma_band=(25,80), gamma_peak 25-80 Hz 내 관대 수용(71 Hz도 보고됨).
- afferent_drive_hz = **0.65** (theta window 0.65-0.80 Hz).
- pyramidal active-only rate 앵커 ~1.8 Hz (band 0.3-3.0); model 자체 per-type rate는 Table 5(Pyr 0.74, PV+B 0.46 등).
- model phase (Table 5, deg): Pyramidal 339.7, PV_Basket 356.8, Bistratified 340.0, O_LM 334.7, Axo 163.4, CCK_Basket 202.8, Ivy 142.1, NGF 176.3, SCA 197.9. (trough-group {Pyr, PV+B, Bis, O-LM} vs rising-group {CCK, Ivy, NGF, Axo, SCA}.)
- experimental rate band (Table 6, 넓은 plausibility): Axo ~17.1, Bistratified ~5.9-30, CCK ~9.4, Ivy ~0.7-2.8, NGF ~6.0, O-LM ~4.9.

**합격 기준 (acceptance.py)**:
- `check_first_order`: per-type mean rate within band(rel tol 30%), CV(ISI) in [0.7,1.4], FF finite, pyramidal sparse-active. **SCALED 런 게이트.**
- `check_oscillation`: theta peak in 5-10 Hz AND |peak-7.8|<=1.5 Hz; theta power > gamma power(control); gamma peak in 25-80 Hz; theta-gamma CFC 존재. **FULL-SCALE 런 게이트 전용.**
- `check_phase`: per-type circular distance |mean_phase - target| <= 45 deg AND Rayleigh p<0.05; trough/rising group 순서 보존.

**비교 프로토콜 (report.py)**: `validate(spikes, meta, tier='scaled'|'full')`. tier='scaled'는 first_order+phase(정성)만 REQUIRED, oscillation은 WARN-only. tier='full'은 전 게이트 REQUIRED. `compare_scaled_vs_full_vs_paper`로 3-컬럼(scaled/full/paper) 마크다운/JSON 아티팩트 산출. 안정 PSD에 N>=200 cells/type, post-crop >=10 s; n_boot=1000 bootstrap 95% CI; seed 기록(결정론성).

### 3.5 미해결 질문 (스케일)
- 모델 자체의 control per-type emergent rate(Figure 5—source data 1-11 / ModelDB 187604 CSV)로 Table 6 proxy를 tight gate로 교체.
- 이 K~17,000 모델에서 van-Albada external-variance floor(Eq.16)가 정확히 어느 scale에서 1차 보존조차 불가능하게 만드는지(수치적 working-point 계산 필요).
- LFP proxy 선택(pyramidal-layer SDF / summed synaptic current / 100 um 내 LFP analog) — 0-deg-at-trough 규약 일치.

---

## 4. Morph -> Point 변환 + 뉴런 모델

### 4.1 원본 multi-compartment 복잡도
Bezaire CA1 소스 셀은 conductance-based multi-compartment NEURON 모델(`bezaire_modeldb/cells/class_*.hoc`). 모든 타입이 HH 채널 complement(Na, fast/slow Kdr, A-type Kv, 셀별 HCN/Ih, Ca, Ca-activated K)를 가지며, **세 가지 행동적 표현형**을 인코딩:
1. **Ih/sag**: O-LM 최대(Sag 26.5 mV / Sag Tau 42.5 ms, ch_HCNolm), SCA(12.9/41.7), CCK(9.2/45.6), 그리고 PV/Axo/Bistratified/Ivy/NGF는 sag ~0.
2. **fast-spiking**: PV basket / axo-axonic / bistratified (Membrane Tau 7-8 ms, Rheobase 300-350 pA, ch_Kdrfast).
3. **adaptation / late-spiking**: pyramidal·CCK(regular-spiking, slow AHP), Ivy·NGF(late-spiking, ch_Kdrslow).

### 4.2 현재 Rin/tau/rheobase -> AEIF가 잡는 것 / 놓치는 것
correct_aeif_parameters.json은 g_L=1000/Rin, C_m=tau_m*g_L, V_th=E_L+Delta_T+rheobase/g_L로 매핑.
- **잡는 것 (data-grounded)**: input resistance, membrane time constant, rheobase/threshold, RMP(=E_L). C_m/g_L/E_L/V_th는 타당한 first cut.
- **놓치는 것 (현재 placeholder)**: a/b/tau_w가 Bezaire 데이터에 fit되지 않음. pyramidal triple (a=4 nS, b=80.5 pA, tau_w=144 ms)은 **Brette-Gerstner 2005 교과서 기본값 verbatim**, 나머지는 round hand-set. 따라서 spike-frequency adaptation 크기/시정수, Ih sag·post-inhibitory rebound·resonance(단일 w-variable로 sag+rebound 동시 불가), AHP kinetics, fast-spiking K dynamics, dendritic synaptic filtering(약 16,850 synapse를 한 점으로 collapse)을 모두 놓침.

**theta 관점에서의 치명성**: 논문 perturbation은 theta가 PV+ basket, NGF, pyr-pyr recurrent, 그리고 interneuronal **diversity 자체**에 의존함을 보임(모든 interneuron을 PV-type으로 변환 시 theta 소멸). first-cut AdEx가 틀리는 특징들(O-LM Ih/resonance, PV fast-spiking gain, pyr/CCK adaptation)이 바로 diversity-defining 요소다.

### 4.3 권장 변환 파이프라인 (타입별)
방향 유지: `aeif_cond_beta_multisynapse` (포인트, NEST-native, conductance-based, 지수 spike nonlinearity, native subthreshold+spike adaptation, per-port E_rev/tau로 GABA_A reversal -60/-75 mV를 양수 가중치로 honor). 단, placeholder a/b/tau_w를 **per-type fit 값으로 교체**.

**Rigorous route**:
1. 소스 NEURON 모델에서 ground-truth 단일셀 데이터 생성(`nrnivmodl`로 *.mod 컴파일, f-I sweep + 과분극 step). NEURON 빌드 불가 시 논문 Appendix의 per-type current-sweep source data로 fit.
2. eFEL로 전기생리 feature 추출(mean_frequency, ISI_CV, adaptation_index, AHP_depth_slow, sag_amplitude, sag_time_constant 등).
3. BluePyOpt(CMA-ES/IBEA)로 AdEx 파라미터 fit: C_m/g_L/E_L 고정(이미 정확), a/b/tau_w/Delta_T/V_th/V_reset/t_ref 최적화.
4. **Pragmatic analytic fallback**: f-I steady-vs-onset gap에서 b, slow-AHP decay/ISI-ratio에서 tau_w, sag-derived conductance에서 a. PV/Axo/Bistratified는 a~0/small b/short tau_w(fast-spiking) 유지; pyr/CCK/Ivy/NGF/SCA는 nonzero b + longer tau_w.
5. **Ih/sag/rebound(O-LM 최강, SCA·CCK 약): 단일 w로 불가** — (a) a>0 + tau_w~sag_tau로 부분 sag, 또는 (b) NESTML-생성 aeif-with-Ih variant로 충실. O-LM phase/theta가 실패하면 (b)로 격상.
6. **Dendritic collapse**: per-pathway peak conductance를 soma로 합산, **타입별 단일 scaling factor**로 somatic PSP 진폭 보존(Rall equivalent-cylinder). complete_interneurons.json의 nS g_max x indegree 후 paired-recording source data로 calibrate. factor는 JSON에 감사 가능하게 기록.
7. **parsed-but-unused 데이터 wiring**: syndata_120/137.json(receptor kinetics, GABA E_rev)를 4 receptor 포트로, complete_interneurons.json(nS g_max)을 양수 가중치로 — negative-weight hack과 name-map 버그(run_scaled_bezaire.py:55) 제거.

### 4.4 뉴런 모델 선택
**AdEx via aeif_cond_beta_multisynapse 확정.** 유일하게 (1) 포인트·NEST-native·GPU/MPI 스케일러블, (2) conductance synapse + arbitrary receptor ports로 GABA reversal를 양수 가중치로 honor(8개 구식 스크립트의 negative-weight hack 직접 수리), (3) 두 상태변수(V,w)로 native subthreshold(a)+spike-triggered(b,tau_w) adaptation — fast-spiking vs regular/adapting vs late-spiking diversity를 표현하는 최소 장치, (4) 지수 spike nonlinearity로 2-variable 모델 중 최광 firing-pattern spectrum. GLIF(Allen)는 fit 가능성은 더 좋으나 NEST 1급 모델 아님(NESTML codegen 필요); Izhikevich는 fit·해석 난해; plain LIF은 adaptation/resonance 불가로 diversity 소거.

---

## 5. 논문 충실도 정밀 감사

### 5.1 스코어카드 (7차원)

| 차원 | 논문값 | 레포값 | 판정 |
|---|---|---|---|
| 1. Cell populations & full-scale counts | 338,740 = 311,500 pyr + 27,240 interneurons(9 types); CA3 204,700 / EC 250,000 | ca1_complete_config.json이 9개 카운트 정확 재현; afferent 일치 | **MATCH** (단, 6.1GB HDF5 빌드 스크립트가 NGF silently drop — spec은 맞고 빌드 버그) |
| 2. Layer geometry | 4000x1000 um; LayerHeights 4;100;50;200;100(=454) | x=4000,y=1000 일치; z=450, SO=120/SP=100/SR=170/SLM=60 — ModelDB와 divergence(SP 2x 두꺼움 등) | **PARTIAL** (포인트 뉴런엔 영향 미미) |
| 3. Connectivity | 5.19B synapses; Table 1=synapse counts(1-10/connection); 각 pyr이 197 pyr 접촉 | indegree=total/N_post 정규화 **정확**; EC->Pyr 10593 + CA3->Pyr 4973 + PC->PC 197 = 15,763 ~ 논문 ~17k; 단 runtime은 invented FixedInDegree 1-8 사용(실제 데이터 미소비) | **MATCH-with-caveat** (추출 일치, 런타임 미소비) |
| 4. Synaptic model | STP 없음(확인); NGF/dendritic은 **mixed GABA_A+GABA_B**(ExpGABAab; GABA_B 제거 시 theta 소멸) | STP 올바르게 부재; **GABA_B 전무**; E_rev 불일치(-60/-65/-80/-90); 실측 g_max 미사용 | **DIVERGE** |
| 5. Afferent drive | 454,700 independent Poisson @ **0.65 Hz** arrhythmic; theta window 0.65-0.80 Hz; 37,900 spikes/cycle | 100-2000 Hz shared generator(2-3 orders 과다); independent per-afferent 구조 부재 | **DIVERGE-SEVERE** |
| 6. Intrinsic properties | Appendix Rin/tau_m/RMP/rheobase | correct_aeif_parameters.json 유도법 타당; 그러나 runtime은 inferior set(cell_types.json) 소비; O-LM g_L=0.56(런타임) vs 3.735(정확) ~6x 오류 | **PARTIAL** |
| 7. **THE PHENOMENON (theta)** | intrinsic 7.8 Hz theta + gamma phase-locked + 세포타입별 phase preference, **rhythmic input 없이** emergent | LFP analog/power spectrum/theta-phase/CFC/Hilbert/Welch **전무**; success criteria가 mean-rate band(심지어 Table 5 모델값과 모순) | **MISSING-CRITICAL** |

### 5.2 Divergence 중요도 순위
1. **현상(theta)이 전혀 시도되지 않음** — 논문 제목이 "Interneuronal mechanisms of hippocampal theta". 레포에 oscillation 분석 0. mean-rate를 아무리 튜닝해도 논문 주장을 검증 불가.
2. **Afferent drive가 구조적으로 틀림** — 0.65 Hz arrhythmic independent ensemble 대신 100-2000 Hz shared. 논문 결과는 0.65-0.80 Hz 좁은 window에 민감. 단일 최중요 파라미터.
3. **런타임이 충실 추출 데이터를 폐기** — invented integer in-degree(1-8)·invented weight(-120~+50)·negative-weight hack 사용, 실측 nS g_max와 실제 synapse count 무시.
4. **GABA_B dual-component 부재** — 논문이 theta에 essential로 증명한 mechanism이 multisynapse 4-port에 없음.
5. **E_rev 불일치 / O-LM g_L 6x 오류** — 단일 SoT 부재.

### 5.3 명시적 정정·확인 사항
- **Afferent indegree 정규화는 정확하지만 canonical table 선택이 중요하다**:
  legacy `connectivity.json`/`conndata_101` 계열의 ECIII->NGF=58240,
  ECIII->SCA=12500은 해당 table의 synapse-per-cell 카운트로는 uncapped
  보존해야 한다. 그러나 논문 Table 1 full-scale gate는 `ConnData=430`
  / `per_cell`과 일치한다(예: ECIII->NGF=523 contacts × 2 = 1046
  synapses/cell). 따라서 final-tier는 raw ModelDB 430을 요구하고,
  legacy JSON 값은 diagnostic compatibility로만 취급한다.
  - 주의(restructure 트랙 관점차): restructure 트랙은 여전히 afferent N_post-divisor 재유도 + sanity assertion(afferent indegree <= pre-population size)을 권고하나, fidelity 트랙은 논문 근거로 정규화가 옳다고 본다. **권장 화해책**: 정규화 자체는 유지하되, 이 값을 distinct-cell 연결로 오해하는 런타임 코드 경로를 수정하고, multi-synapse aggregation을 명시. afferent를 distinct presynaptic cell로 강제 cap하는 assertion은 **추가하지 않음**(정확한 synapse budget을 깎을 위험).
- **theta/phase가 핵심 현상이며 레포는 이를 추구하지 않는다**: 레포는 mean firing rate matching을 추구하며 그 rate 타깃조차 논문 Table 5 모델값과 모순(레포의 "interneuron 10-30 Hz"는 Table 6 in-vivo이지 모델 emergent가 아님). **프로젝트 방향을 theta/phase 재현으로 pivot해야 한다.**

### 5.4 존재론적 위험 (Open Question)
**포인트 뉴런이 intrinsic theta를 재현할 수 있는가?** 논문 mechanism은 dendritic compartmentalization + dendritic GABA_B charge transfer + perisomatic/dendritic inhibition의 공간 분리에 의존하며, 포인트 AdEx는 이를 collapse한다. 완벽한 connectivity/drive에도 spontaneously oscillate하지 않을 수 있음 — 포인트 뉴런 선택의 근본적, 잠재적으로 fatal한 한계. **권장 접근**: 포인트 AdEx + dual GABA_A/GABA_B port를 first attempt(가장 싸고 빠른 falsify)로 시도하되, oscillate 실패를 가능성 높은 결과로 보고, 그 경우 minimal 2-compartment(soma+dendrite) 폴백을 정당화. theta-emergence 질문이 경험적으로 답해지기 전 과투자 금지.

---

## 6. 프로젝트 재구조화 계획

### 6.1 타깃 디렉터리 트리 (단일 설치형 패키지 `ca1`)
```
/data1/seonghwankim/workspace_studio/bsb-test/
  pyproject.toml          # [project] name=ca1; src layout; bsb[nest,neuron]; 단일 python pin
  README.md               # CA1/Bezaire용 재작성(generic BSB tutorial 아님)
  CLAUDE.md               # 권위 파일 안내: modeldb_dataset_notes.md + bezaire_modeldb_true_connectivity.json
  .gitignore              # *.hdf5 .DS_Store .venv/ output/ __pycache__/ *.pyc node_modules/ nest-build/
  .python-version         # 단일 값(installer/pyproject/README와 일치)
  install_nest_linux.sh   # macOS install_nest.sh 대체: system gcc/gsl/openmp, -Dwith-mpi=ON, pinned NEST
  Makefile or tasks.py    # build, sim, validate, regen-artifacts
  manifest.json           # 생성 아티팩트 레지스트리: {path, config, git_sha, n_cells, n_conn_types, checksum}
  src/ca1/
    __init__.py
    config.py             # 단일 config/param 로더(권위 JSON만 참조)
    extract/  modeldb_tables.py / connectivity.py / syndata.py
    build/    builder.py(예외 미은폐) / downscale.py(weight-compensated, FIXED)
    params/   neurons.py / synapses.py
              bezaire_modeldb_true_connectivity.json   # 권위 connectivity(verbatim)
              neuron_parameters.json                   # = correct_aeif_parameters.json
    sim/      backend.py(SimulatorBackend ABC) / nest_backend.py / gpu_backend.py(NEST GPU)
    validation/ harness.py / targets.py / acceptance.py / report.py
    analysis/  rates.py / spectral.py(theta/gamma/phase/CFC, NEW) / plots.py / h5_inspect.py
    cli.py    # ca1 build|sim|validate|regen CONFIG
  configs/  full_scale.yaml / scaled_1_50.yaml(preserve-indegree) / smoke_180.yaml
  tests/    test_downscale_conductance.py / test_alias_map.py / test_loop_rate.py /
            test_afferent_indegree.py / test_smoke_sim.py
  web/bsb-visualizer/     # KEEP(유일 React 프론트엔드)
  bezaire_modeldb/        # KEEP verbatim(vendored ModelDB 187604 ground truth)
  docs/  modeldb_dataset_notes.md / downscaling_strategy.md / architecture.md(NEW)
  DEEP_ANALYSIS_REPORT.md # KEEP(부채 provenance)
```

핵심 설계: **SimulatorBackend 추상 인터페이스**(build_populations/connect/attach_recorders/run/collect_spikes). BSB가 canonical graph를 한 번 빌드하고 모든 backend(NEST 오늘, NEST GPU 내일)가 **같은 graph를 소비** — "두 파이프라인이 silently diverge"하는 구조적 결함을 직접 해소.

### 6.2 파일별 마이그레이션 맵

**BUILDERS (5 -> 1)**: build_basic_ca1.py -> build/builder.py(REFACTOR, 유일하게 argparse/--validate). build_complete_ca1.py / build_full_ca1.py(:76 .items() 버그) / build_scaled_ca1.py -> DELETE. build_scaled_network.py -> build/downscale.py(REFACTOR + weight-compensation fix).

**RUNNERS (~14 -> 1, 두 생존자 병합)**:
- run_fullscale_true_bezaire.py -> sim/nest_backend.py **SKELETON**(최고 엔지니어링: pathlib :24, CELL_ALIAS_MAP :29-48, external-JSON 로더; :374-406 5x loop 버그 수정).
- run_bezaire_compliant_simulation.py -> nest_backend.py로 **병합**(유일 정확 모델: aeif_cond_beta_multisynapse 4-port positive weight :73-90).
- 나머지 모두 DELETE: run_basic_simulation.py(:20 crash), run_scaled_bezaire.py(:55 name-map 버그), run_corrected_simulation.py(800-1500 Hz drive), run_balanced_simulation.py(100-cell toy), /Users 하드코드 다수(run_correct_bezaire_connectivity.py, run_perfected_bezaire.py, run_scaled_true_bezaire.py, run_fullscale_bezaire_compliant.py 등), debug 다이어리들.

**EXTRACTORS**: parse_modeldb_tables.py -> extract/modeldb_tables.py(REFACTOR). extract_true_bezaire_connectivity.py -> extract/connectivity.py(/Users 제거). parse_syndata.py -> extract/syndata.py. extract_bezaire_connectivity.py(billions-scale broken) -> DELETE.

**PARAMETERS (9 -> 권위 set)**: KEEP bezaire_modeldb_true_connectivity.json(권위 connectivity) + correct_aeif_parameters.json(-> neuron_parameters.json). complete_interneurons.json + syndata_120/137.json -> params/synapses.py로 wire(드디어 소비). DELETE bezaire_true_connectivity.json(billions, broken), bezaire_accurate_connectivity.json(hand-estimate orphan), cell_types.json(3 types stale).

**VIZ (4 -> 1)**: web/bsb-visualizer KEEP. brain-viz는 COOP/COEP vite header + chunk-streaming 로더를 React 앱에 salvage 후 DELETE. bsb-electron-viz(webSecurity:false), bsb-vue-viz(fake data) DELETE.

**TOP-LEVEL**: h5_inspector.py -> analysis/h5_inspect.py. main.py(89-byte stub) / 6개 assert-0 smoke test / install_nest.sh / nest-build/ / nest-simulator/ / network.hdf5 / 모든 *.hdf5(8.4G) DELETE(regenerable, manifest 기록 후). DEEP_ANALYSIS_REPORT.md / docs / bezaire_modeldb KEEP.

### 6.3 환경·재현성 수정
- **git init** + 실제 .gitignore(현재 __pycache__/build/dist/.egg-info/.venv만 있음 -> *.hdf5, .DS_Store, output/, *.pyc, node_modules/, nest-build/ 추가).
- **8.4 GB HDF5 삭제**: 단, seed-determinism 먼저 검증하고 manifest.json에 checksum 기록 후 삭제(비-seeded RNG 사용 빌드는 regen 시 달라짐).
- **Linux+CUDA installer**: macOS install_nest.sh 대체(system gcc/gsl/OpenMP, -Dwith-mpi=ON, NEST 버전 pin).
- **Python pin 단일화**: README는 3.12가 NEST 깨진다 경고하나 .python-version은 3.12 pin — **import nest가 실제 성공하는 버전을 경험적으로 확정** 후 .python-version/pyproject/installer/README 일괄 갱신.
- **/Users 절대경로 10개 제거**: Path(__file__).resolve().parents[N](이미 run_fullscale_true_bezaire.py:24에 존재) 백포트.
- **nest-simulator/ (271M)**: Linux installer로 externalize + gitignore; 정확 재현이 필요하면 submodule pin.

---

## 7. 단계별 복구 시퀀스

의존성 순서. 각 단계는 done-criterion으로 게이트.

**Phase 0 — 재현성 바닥** (done: clean repo, sub-500MB, /Users 0건, 단일 install 경로 문서화)
- git init + .gitignore; seed-determinism 검증 후 8.4 GB HDF5 삭제(manifest 기록); Linux/CUDA installer 작성; Python pin 단일화; /Users 10개 제거.

**Phase 1 — ONE thing runnable + 첫 영구 출력** (done: `import nest`(또는 nestgpu) 성공 AND 180-cell sim이 spikes + per-type rate를 디스크에 persist)
- NEST 또는 NEST GPU import 성립(원래 빌드 방법 미상 -> Linux installer로 재빌드). run_basic_simulation.py:20 dead-symbol import 제거. 최소 fixture(scaled_1_50, Phase 0에서 재생성)로 병합 canonical runner 1회 실행 + spike_recorder/CreateRecord로 persist. **프로젝트 역사상 첫 영구 출력.** 물리 수정 전 plumbing 먼저 증명.

**Phase 2 — 검증된 정확성 수정 + 회귀 테스트** (done: scaled vs full per-type rate 일치, 각 버그에 가드 테스트)
- (a) downscaling weight compensation(per-cell total conductance 불변 테스트). (b) name-map 버그(run_scaled_bezaire.py:55 'Pvbasket' != 'PV_Basket') -> CELL_ALIAS_MAP. (c) 5x loop 버그(run_fullscale_true_bezaire.py:374-406) -> actual sim time으로 나눔. (d) afferent 정규화: **정규화는 유지**, 단 multi-synapse aggregation 명시 + 런타임의 distinct-cell 오해 경로 수정(fidelity/restructure 화해책 5.3 참조). (e) build_full_ca1.py:76 .items() 버그.
- 추가: GABA E_rev 단일화(~-60~-75 mV), O-LM g_L=3.735로 수정, correct_aeif_parameters.json + complete_interneurons.json을 단일 SoT로 채택.

**Phase 3 — backend 추상화 + NEST GPU 마이그레이션** (done: 동일 canonical graph가 NEST와 NEST GPU에서 tolerance 내 일치)
- SimulatorBackend ABC 정의. NestBackend(검증 oracle) 후 GpuBackend(NEST GPU) 추가. 불변식: 두 backend가 **같은 BSB HDF5 graph** 소비. config에 simulator:{backend,version} lock. 작은 fixture에서 양 backend rate 일치로 GPU 포트 정확성을 NEST oracle 대비 증명.

**Phase 4 — 풀스케일 빌드 + theta 검증 하네스** (done: 진짜 9-type/338,740-cell 빌드, 하네스가 theta/phase를 논문 타깃 대비 pass/fail로 보고)
- 진짜 풀스케일 빌드(현재 'full_scale'는 8,191 mislabeled, NGF drop 수정). afferent를 454,700 independent Poisson @ 0.65 Hz arrhythmic로 구동(0.5-0.9 Hz sweep으로 theta window 탐색). connectivity를 synapse-count + multi-synapse weight aggregation으로 wire. **NEW analysis**: LFP analog(pyramidal synaptic current, 100 um 내) + SDF + bandpass(theta 5-10, gamma 25-80) + Hilbert phase + Welch(목표 7.8 Hz) + theta-gamma CFC(scipy.signal로 충분, 신규 heavy dep 없음). success criteria를 mean-rate band에서 **theta peak 5-10 Hz(목표 7.8) + per-type phase(Table 5) + CFC**로 교체. provenance(backend/config/git SHA) 스탬프.

**Phase 5 (조건부) — 2-compartment 폴백**
- Phase 4에서 포인트 AdEx가 theta를 under-generate하거나 O-LM phase가 실패하면: O-LM(및 SCA/CCK)에 NESTML Ih variant 추가, 또는 minimal 2-compartment(soma+dendrite)로 dendritic GABA_B 분리 복원. 이 질문이 경험적으로 답해지기 전 과투자 금지.

---

## 8. 기타 이슈

- **트랙 간 명시적 충돌 1건 — afferent 정규화**: restructure 트랙은 N_post-divisor 재유도 + cap assertion을 권고, fidelity 트랙은 논문 근거로 정규화가 옳다고 봄. **본 보고서의 화해책**: 정규화 유지, multi-synapse aggregation 명시, distinct-cell 오해 런타임 경로만 수정, afferent를 pre-pop size로 cap하는 assertion은 추가하지 않음(정확한 synapse budget 손상 위험). 이는 "외부 구동 수치가 바뀌면 E/I regime 전체가 이동"하는 과학적 변경이므로 mechanical refactor가 아니라 재검증 대상.
- **GABA_B charge transfer**: 논문은 ExpGABAab의 비선형(K-channel-coupled) kinetics로 GABA_B를 모델링하며 이것이 theta에 essential. multisynapse의 slow beta-function port로 근사하면 charge-transfer 비선형성을 잃어 GABA_B 의존성이 재현 안 될 수 있음. 4 receptor port를 5개(AMPA, GABA_A, slow GABA_B E_rev~-90)로 확장 검토.
- **GABA_A E_rev 권위값 미확정**: SynData120=-60, SynData137=-75, synapses.json=-80. 추출 페이지에 단일 명시값 없음 — SimRun.m:36이 사용하는 SynData110에서 확정 필요.
- **풀스케일 메모리 헤드룸**: 16-20 B/syn에서 약 83-104 GB 총량, 138 GB에 modest headroom. connection metadata가 추정보다 무거우면 2 GPU에 population + 1 GPU에 afferent로 분할하거나 recording을 trim해야 함. spike 출력량이 크므로 incremental write로 host-memory blowup 방지.
- **레이어 두께 divergence (차원 2)**: 포인트 뉴런에는 영향 미미하나, distance-dependent 규칙을 쓰면 placement-based connectivity에 영향. ModelDB(SO~104/SP~50/SR~200/SLM~100)로 정정 권장.
- **Figure 5 source data 부재**: 모델 자체 per-type emergent rate(tight gate)는 main text에 없고 Figure 5—source data 1-11(ModelDB 187604 CSV)에 있음 — Table 6 experimental band를 looser proxy로 사용 중. 풀어야 tight model-vs-model gate 가능.
- **데이터 버그는 시뮬레이터 독립**: name-map mis-map, 5x rate-inflation loop, downscaling 미보정은 backend 선택과 무관하며, 수정하지 않으면 GPU 런이 silent/over-driven 네트워크를 충실히 재현. **GPU 포트 중 수정해야 함.**

---

## 9. 확정이 필요한 의사결정

복구 착수 전 사용자가 확정해야 할 핵심 fork(각 권장안 포함). 상세는 keyDecisions 참조.

1. **주력 시뮬레이터**: NEST GPU (권장) vs Arbon vs GeNN-단일GPU-다운스케일전용. -> **NEST GPU**: 동일 모델·가중치 규약 재사용, 풀스케일 멀티 GPU 유일 저비용 경로.
2. **최종 스케일 전략**: GPU 풀스케일 권위 + 다운스케일 디버그 전용 (권장) vs 다운스케일 oscillation 주장 허용 vs 풀스케일 전용. -> **GPU 풀스케일이 theta/gamma/phase의 유일 권위 config**(van Albada 2015가 correlation/synchrony 비보존을 증명, 이것이 본 모델의 deliverable).
3. **뉴런 모델 충실도 수준**: AdEx + hybrid(analytic seed -> BluePyOpt refine for theta-critical types) + dual GABA_B port + 조건부 2-compartment 폴백 (권장) vs 현 placeholder 유지 vs 즉시 2-compartment. -> **AdEx hybrid, 교과서 기본 a/b/tau_w 폐기, O-LM Ih와 2-compartment는 theta 실패 시에만 격상**(과투자 회피).
4. **재구조화 공격성 / 바이너리 삭제**: 8.4 GB HDF5 삭제+regen + ~14 runner를 1개로 병합 + 5 param을 2개로 + 4 viz를 1개로 (권장, aggressive) vs git-LFS 보관 vs in-tree+gitignore. -> **삭제+regen(seed-determinism 검증 + manifest checksum 선행)**, 단일 canonical runner는 두 생존자 병합.

추가로 방향성 확인이 필요한 과학적 fork: **theta/phase 재현으로 pivot할 것인가**(권장: 예 — mean-rate matching은 논문 주장을 검증하지 못하고 rate 타깃조차 Table 5와 모순), 그리고 **포인트 뉴런으로 theta가 안 나면 2-compartment를 받아들일 것인가**(권장: 포인트 + dual GABA_B port로 먼저 falsify, 실패를 likely outcome으로 두고 폴백 정당화).
