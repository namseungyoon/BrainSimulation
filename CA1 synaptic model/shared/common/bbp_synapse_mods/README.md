# bbp_synapse_mods — BBP 공식 CA1 시냅스 메커니즘 (프로젝트 소유)

Ecker et al. (2020)이 쓴 stochastic TM 시냅스 NMODL. **단일세포 다운로드 모델엔 시냅스가 없어서**
(시냅스는 세포 간 연결) BBP 공식 공개 리포에서 받아 프로젝트 소유로 둔다. → `legacy/455999` 의존 제거.

## 출처
- 리포: **BlueBrain/neurodamus-models** (main)
  - `hippocampus/mod/ProbAMPANMDA_EMS.mod` (CA1 전용: NMDA τ_r=9, τ_d=61, Andrasfalvy&Magee 2001)
  - `hippocampus/mod/DetAMPANMDA.mod`
  - `common/mod/ProbGABAAB_EMS.mod` (hippocampus/mod 에서 이 파일로 symlink)
  - `common/mod/DetGABAAB.mod` (동일 symlink)
  - `common/mod/VecStim.mod`
- 검증: 보유 455999 사본과 `ProbAMPANMDA_EMS`/`ProbGABAAB_EMS` **바이트 동일(diff 0줄)** → 동일한 공식 코드.

## 역할
| 파일 | 용도 |
|------|------|
| ProbAMPANMDA_EMS | 흥분성 확률 TM 시냅스(AMPA+NMDA, MVR) — 논문 §2.5 핵심 |
| ProbGABAAB_EMS | 억제성 확률 TM 시냅스(GABA_A/B) |
| DetAMPANMDA / DetGABAAB | 결정론 버전(교차검증용) |
| VecStim | presynaptic 스파이크열 재생 |

## 핵심 인터페이스(파라미터 매핑은 02_synapse_model/params_table3.py)
`Use(U_SE), Dep(D), Fac(F), Nrrp(N_RRP), tau_r/d_AMPA·NMDA·GABAA, NMDA_ratio, e(E_rev)`,
`setRNG(s1,s2,s3)`(Random123), NetCon weight=ĝ[nS] (mod `gmax=.001`이 µS 변환).

이 폴더가 `s1_unify_mechanisms.py`의 시냅스 mod 소스다.
