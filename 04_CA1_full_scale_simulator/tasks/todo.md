# CA1 복구 작업 체크리스트

목표: Bezaire et al. (2016) intrinsic 7.8 Hz theta를 BSB + 포인트 AdEx + NEST-GPU(3xA40)로 재현·검증.
확정된 결정: 시뮬레이터=NEST GPU / 권위=GPU 풀스케일(다운스케일=디버그) / 뉴런=AdEx hybrid fit + dual GABA_B + 조건부 2-comp / 재구조화=aggressive(삭제+regen).
근거 문서: `RECOVERY_PLAN.md`, `DEEP_ANALYSIS_REPORT.md`.

## Phase 0 — 재현성 바닥  [DONE]
- [x] git init (baseline a8b13a2, restructure 476f3c2)
- [x] .gitignore (*.hdf5, output/, .venv, .DS_Store, nest build trees, *.pdf, .omc)
- [x] 현재 인벤토리 캡처 (8.96 GB HDF5 / 7 files; 10 files with /Users)
- [x] manifest.json 생성 (legacy 아티팩트 레지스트리 + 향후 schema)
- [x] 8.96 GB stale HDF5 삭제 (buggy-builder 산출물, manifest 기록 후; 신 builder가 정확 재생성)
- [x] Python pin 단일화 (.python-version=3.12 / pyproject >=3.11,<3.13)
- [x] /Users 절대경로 10개 제거 (legacy 스크립트 삭제로 해소)
- [~] Linux+CUDA installer: docs/INSTALL_NESTGPU.md 작성 / 실행 스크립트 TODO

## Phase 1 — 패키지 + 첫 영구 출력
- [x] src/ca1 패키지 스켈레톤 + 핵심 계약 (types.py, sim/backend.py)
- [x] params/config 단일 SoT (neurons, synapses, config -> NetworkSpec; 9 types, O-LM g_L 수정)
- [x] extract 모듈 (ModelDB -> canonical JSON, /Users 제거, indegree=total/N_post)
- [x] build + downscale (weight-compensation FIX, NGF drop FIX, .items() FIX, 가드레일)
- [x] nest_backend (두 생존자 병합, 5x loop FIX, name-map FIX, spike persist) — 코드 완료, 실행은 NEST 빌드 후
- [x] gpu_backend (NEST GPU, n_receptors/0-based ports/positive inhib weights) — 코드 완료
- [x] 검증 하네스 (rates, spectral theta/gamma/phase/CFC, targets, acceptance) — 합성 7.81 Hz 검출 확인
- [x] analysis (h5_inspect, plots), cli, configs(full/scaled/smoke)
- [x] tests (downscale conductance invariant, spectral 합성 theta, afferent, alias) — 69 passing
- [x] dead code 삭제: 28 runner/builder, 5 param JSON, 3 viz->1(React), assert-0 smoke, macOS installer
- [x] NEST(CPU) 빌드 완료 (3.9, vendored source, CUDA/MPI, py3.12) -> import nest OK, aeif_cond_beta_multisynapse OK
- [x] **첫 영구 출력**: NestBackend scale-0.01(~3400셀) -> 2572 spikes(non-silent), pyramidal 18.7% sparse-active, results/smoke_0p01_nest.npz
- [x] 검증 하네스 실 데이터 스코어카드 동작 (Table 5 rate/phase 대조; Axo 이름버그 수정; 미보정 net이라 FAIL=예상)

## First-light 관찰 (scale 0.01, 미보정)
- Pyramidal 0.36 Hz(sparse, 방향 맞음) | O_LM 27 Hz(목표 17.4, 과발화) | Ivy 0.06 Hz(목표 43.3, 침묵) | PV 5.0 | Axo 12.4 | Bis 8.3 | CCK 5.8 | NGF 7.1 | SCA 8.9
- 해석: placeholder a/b/tau_w + 가중치 미보정(preserve-indegree at 0.01) + 0.65Hz drive(논문은 풀스케일 전제) -> 보정/피팅 필요. 파이프라인 자체는 정상.

## Phase 2 — 검증된 정확성 수정 + 회귀 테스트 (코드는 패키지에 반영)
- [ ] downscaling weight compensation (per-cell total conductance 불변 테스트)
- [ ] name-map 버그 -> CELL_ALIAS_MAP
- [ ] 5x loop 버그 -> 실제 sim time으로 나눔
- [ ] afferent 정규화: 유지 + multi-synapse aggregation 명시 (distinct-cell 오해 경로 수정)
- [ ] build_full NGF drop + .items() 버그
- [ ] GABA E_rev 단일화, O-LM g_L=3.735 수정

## Phase 3 — backend 추상화 + NEST GPU 마이그레이션  [DONE]
- [x] SimulatorBackend ABC — NestBackend(oracle) vs GpuBackend 동일 NetworkSpec 소비
- [x] NEST(CPU) 빌드 + NEST GPU v2.0 빌드 (CUDA 12.4/compute86/MPI, .venv) — import nest/nestgpu 성공, A40에서 모델 작동
- [x] GPU backend 4개 버그 수정(n_receptors, indegree, spike API, afferent aggregation)
- [x] GPU vs CPU @ scale0.01: 9타입 모두 동일 자릿수, 여러 타입 근접 일치(CCK 5.9/5.9, PV 5.3/5.0). 잔차는 finite-size+RNG
- [x] env.sh (nest_vars + NESTGPU_LIB + PYTHONPATH)

## Phase 4/5 — 보정·피팅·풀스케일 theta (다음)
- [ ] 뉴런 a/b/tau_w per-type 피팅: NEURON mod 컴파일 완료(feasible). class_*.hoc 셀을 f-I/sag sweep -> AdEx fit
- [ ] 보정: O_LM 과발화(66/27 vs 17.4) 등 per-type rate를 mean-field 가중치 보정 + 피팅으로 조정
- [ ] 풀스케일 GPU 빌드(진짜 9type/338,740) + 454,700 afferent @ 0.65 Hz arrhythmic
- [ ] theta 검증: 하네스로 7.8 Hz peak + per-type phase(Table 5) + CFC pass/fail
- [ ] 조건부 2-compartment (포인트로 theta 실패 시)

## Phase 4 — 풀스케일 + theta 검증
- [ ] 진짜 9-type/338,740-cell 빌드 (NGF 포함)
- [ ] afferent 454,700 independent Poisson @ 0.65 Hz arrhythmic (0.5-0.9 sweep)
- [ ] 하네스가 theta peak 5-10 Hz(목표 7.8) + per-type phase(Table 5) + CFC pass/fail 보고
- [ ] provenance(backend/config/git SHA) 스탬프

## Phase 5 (조건부) — 2-compartment 폴백
- [ ] 포인트 AdEx가 theta under-generate 또는 O-LM phase 실패 시에만: O-LM Ih variant 또는 minimal 2-comp

## Working Notes
- afferent indegree(ECIII->NGF 58,240)는 synapse-per-cell 카운트로 정규화 정확 (pre-pop cap 추가 금지).
- theta 메커니즘은 dendritic GABA_B 의존 -> 포인트 뉴런으로 안 나올 존재론적 위험. 포인트+dual GABA_B로 먼저 falsify.
- 권위 connectivity: src/ca1/params/connectivity.json. 권위 뉴런: neuron_parameters.json.
