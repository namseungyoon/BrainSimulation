# 파이프라인 색인 — N-M ↔ 폴더 ↔ 스크립트 ↔ 그림 ↔ Notion

> **이 트랙의 유일한 색인이다.** 번호가 어긋나면 여기서 잡힌다.
> Notion 페이지: `3c117cb5cdbd8091af80cba6a08ec5ee` — 절 번호가 아래 N-M 과 **완전히 일치**해야 한다.
>
> 상태: ⬜ 미착수 · 🔄 진행 · ✅ 검증완료

## 단계 구성 원칙

**한 단계 = 하나의 구성요소(또는 활동)이고, 그 구성요소의 검증은 그 단계 안에 둔다.**

- 단기가소성은 단계가 아니라 **엔진의 능력**이다 → 검증은 5-9(엔진 단계 안의 횡단 검증).
- 전달 검증도 단계가 아니라 **시냅스 구성요소의 검증**이다 → 3단계 안(3-5~3-9).

| 단계 | 성격 | 폴더 |
|---|---|---|
| 1 환경 | 인프라 구축 + 검증 | `01_env/` |
| 2 뉴런 | 구성요소 구축 + 검증 | `02_neurons/` |
| 3 시냅스·전달 | 구성요소 구축 + 검증 | `03_synapse/` |
| 4 구동·리듬 | 구성요소 구축 + 검증 | `04_drive/` |
| 5 가소성 엔진 | 구성요소 구축 + 검증 (단기·장기 모두) | `05_engines/` |
| 6 실험 | 사용 | `06_experiments/` |
| 7 보완 모델 | 산출 | `07_newmodel/` |

---

## 색인

| N-M | 하위 폴더 | 스크립트 | 주요 그림 | 상태 |
|---|---|---|---|---|
| **1-1** | `01_env/1_probe` | `env/probe_env.ps1` ※ | `1-1_env_probe.json` ※※ | ✅ |
| **1-2** | `01_env/2_python` | (설치 절차 → `docs/ENVIRONMENT.md`) | — | ⬜ |
| **1-3** | `01_env/3_neuron` | `env/activate.ps1` | `1-3_neuron_version.png` | ⬜ |
| **1-4** | `01_env/4_build` | `env/build_mechanisms.py` | `1-4_mech_inventory.png` | ⬜ |
| **1-5** | `01_env/5_verify` | `1-5_verify_mechanisms.py` | `1-5_mech_verify.png` | ⬜ |
| **2-1** | `02_neurons/1_survey` | `2-1_survey_bundles.py` | `2-1_bundle_survey.png` | ⬜ |
| **2-2** | `02_neurons/2_load` | `2-2_load_cell.py` | `2-2_cell_loaded.png` | ⬜ |
| **2-3** | `02_neurons/3_morphology` | `2-3_morphology.py` | `2-3_morphology.png` | ⬜ |
| **2-4** | `02_neurons/4_ephys` | `2-4_ephys_battery.py` | `2-4_ephys_battery.png` | ⬜ |
| **2-5** | `02_neurons/5_resonance` | `2-5_zap_resonance.py` | `2-5_zap_impedance.png` | ⬜ |
| **2-6** | `02_neurons/6_pair` | `2-6_two_cells.py` | `2-6_two_cells.png` | ⬜ |
| **2-7** | `02_neurons/7_distance` | `2-7_distance_map.py` | `2-7_distance_map.png` | ⬜ |
| **3-1** | `03_synapse/1_params` | `3-1_param_table.py` | `3-1_param_table.png` | ⬜ |
| **3-2** | `03_synapse/2_placement` | `3-2_placement.py` | `3-2_syn_sites.png` | ⬜ |
| **3-3** | `03_synapse/3_wiring` | `3-3_wiring.py` | `3-3_wiring_diagram.png` | ⬜ |
| **3-4** | `03_synapse/4_record` | `3-4_record.py` | `3-4_record_check.png` | ⬜ |
| **3-5** | `03_synapse/5_uepsp` | `3-5_uepsp.py` | `3-5_uepsp_trace.png` · `3-5_uepsp_stats.png` | ⬜ |
| **3-6** | `03_synapse/6_stochastic` | `3-6_stochastic.py` | `3-6_amp_hist.png` | ⬜ |
| **3-7** | `03_synapse/7_calibrate` | `3-7_calibrate_g.py` | `3-7_g_sweep.png` | ⬜ |
| **3-8** | `03_synapse/8_distance` | `3-8_attenuation.py` | `3-8_attenuation.png` | ⬜ |
| **3-9** | `03_synapse/9_bap` | `3-9_bap_profile.py` | `3-9_bap_profile.png` | ⬜ |
| **4-1** | `04_drive/1_modes` | `4-1_drive_modes.py` | `4-1_drive_modes.png` | ⬜ |
| **4-2** | `04_drive/2_natural_theta` | `4-2_natural_theta.py` | `4-2_zap_summary.png` · `4-2_spike_spectrum.png` | ⬜ |
| **4-3** | `04_drive/3_imposed_theta` | `4-3_imposed_theta.py` | `4-3_imposed_theta.png` | ⬜ |
| **4-4** | `04_drive/4_gamma` | `4-4_gamma.py` | `4-4_theta_gamma.png` | ⬜ |
| **4-5** | `04_drive/5_phase_align` | `4-5_phase_align.py` | `4-5_phase_align.png` | ⬜ |
| **4-6** | `04_drive/6_budget` | `4-6_runtime_budget.py` | `4-6_runtime_budget.png` | ⬜ |
| **5-1** | `05_engines/1_ref` | `5-1_refs.py` | `5-1_refs.png` | ⬜ |
| **5-2** | `05_engines/2_det` | `5-2_engine_det.py` | `5-2_engine_det.png` | ⬜ |
| **5-3** | `05_engines/3_gb_a` | `5-3_engine_a.py` | `5-3_engine_a.png` | ⬜ |
| **5-4** | `05_engines/4_gb_b` | `5-4_engine_b.py` | `5-4_engine_b.png` | ⬜ |
| **5-5** | `05_engines/5_gb_c` | `5-5_engine_c.py` | `5-5_engine_c.png` | ⬜ |
| **5-6** | `05_engines/6_stdp` | `5-6_stdp.py` | `5-6_stdp_window.png` | ⬜ |
| **5-7** | `05_engines/7_glusyn` | `5-7_glusyn.py` | `5-7_spine_calcium.png` | ⬜ |
| **5-8** | `05_engines/8_registry` | `5-8_registry.py` | `5-8_engine_matrix.png` | ⬜ |
| **5-9** | `05_engines/9_stp_verify` | `5-9_stp_verify.py` | `5-9_ppr.png` · `5-9_train.png` · `5-9_recovery.png` | ⬜ |
| **5-10** | `05_engines/10_calibrate` | `5-10_calibrate.py` | `5-10_first_pulse_match.png` | ⬜ |
| **5-11** | `05_engines/11_freeze` | `5-11_freeze_contract.py` | `5-11_freeze_identity.png` | ⬜ |
| **6-1** | `06_experiments/1_theta_phase` | `6-1_theta_phase.py` | `6-1_theta_phase.png` | ⬜ |
| **6-2** | `06_experiments/2_theta_gamma` | `6-2_theta_gamma.py` | `6-2_theta_gamma.png` | ⬜ |
| **6-3** | `06_experiments/3_stdp_single` | `6-3_stdp_single.py` | `6-3_stdp_single.png` | ⬜ |
| **6-4** | `06_experiments/4_stdp_burst` | `6-4_stdp_burst.py` | `6-4_stdp_burst.png` | ⬜ |
| **6-5** | `06_experiments/5_tbs` | `6-5_tbs.py` | `6-5_tbs_ltp.png` | ⬜ |
| **6-6** | `06_experiments/6_hfs` | `6-6_hfs.py` | `6-6_hfs_ltp.png` | ⬜ |
| **6-7** | `06_experiments/7_lfs` | `6-7_lfs.py` | `6-7_lfs_ltd.png` | ⬜ |
| **6-8** | `06_experiments/8_location` | `6-8_location.py` | `6-8_location.png` | ⬜ |
| **6-9** | `06_experiments/9_gap_analysis` | `6-9_gap_analysis.py` | `6-9_compare.png` → `docs/GAPS.md` | ⬜ |
| **7-1** | `07_newmodel/1_gaps` | — (`docs/GAPS.md` 확정) | — | ⬜ |
| **7-2** | `07_newmodel/2_design` | `7-2_design.py` | `7-2_design.png` | ⬜ |
| **7-3** | `07_newmodel/3_ref` | `7-3_newmodel_ref.py` | `7-3_newmodel_ref.png` | ⬜ |
| **7-4** | `07_newmodel/4_mod` | `mechanisms/*.mod` | — | ⬜ |
| **7-5** | `07_newmodel/5_verify` | `7-5_verify.py` | `7-5_verify.png` | ⬜ |
| **7-6** | `07_newmodel/6_compare` | `7-6_compare.py` | `7-6_compare.png` | ⬜ |

※ **1-1 만 PowerShell 이다.** 파이썬이 없는 상태를 진단하는 도구가 파이썬이면 순환이므로,
주 도구를 `env/probe_env.ps1` 로 두었다. 1-2 이후 파이썬 판을 추가해 재실행 가능하게 한다.

※※ **1-1 만 PNG 가 없다.** matplotlib 이 파이썬을 요구하는데, 그 파이썬이 없다는 것을 확인하는 단계라
그림을 만들 수 없다. 산출물은 **밑줄 없는 `1-1_env_probe.json`(추적)** 과
[ENVIRONMENT.md](ENVIRONMENT.md) 의 판정표다. 1-2 완료 후 같은 JSON 으로 PNG 를 생성해 채운다.

## 규약 요약

- 그림이 여러 장이면 **번호는 같고 slug만 다르다.** 알파벳 갈래(`4-2a`)는 쓰지 않는다.
- 중간 데이터는 `figures/_N-M_<slug>.npz` (밑줄 = gitignore). **추적 대상은 코드 + 결과 PNG만.**
- **번호 스크립트는 서로 import 하지 않는다.** 재사용 코드는 전부 `lib/`.
- 커밋 메시지: `04 N-M: 설명`
