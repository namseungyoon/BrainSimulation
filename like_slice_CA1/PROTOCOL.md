# 현재 시뮬레이션 프로토콜 — like-slice CA1 → MEA fEPSP

> 소스 대조 스냅샷(2026-08). 목적: "어떤 슬라이스에·어떤 세포를·어떤 시냅스로·어떻게 구동하고·어디서 측정하나"를 한 장에 정리.
> 수치는 모두 소스/실행 로그 집계(추측 없음). 관련 그림: `12_lfp/figures/E4b_neuron_metadata.png`, `E4b_model_formula_map.png`, `E4b_band_3x8_grid.png`.

## 1. 슬라이스 & 세포 배치
- **출처**: Romani(2024) CA1 아틀라스 커넥텀에서 절편(slice) 추출 후 실좌표 배치.
- **세포 17,647개** — `05_placement/slice_cells.npz`: `xyz`(µm), `quat_wxyz`(방향), `mtype`, `etype`, `layer`, `sclass`.
- **기하**: 3D 경계상자 span ≈ 2,428 × 1,282 × 755 µm. PC(SP) 밴드 면투영(PCA) **2,305 × 468 µm**, 면밀도 **14,560개/mm²**.

## 2. 뉴런 구성 (메타데이터)
- **E/I**: 흥분 15,723(89%) = `SP_PC` · 억제 1,924(11%) = 인터뉴런 11종.
- **층**: SP 17,330 · SO 264 · SLM 29 · SR 24.
- **12 m-type → 4개 대표 me-model(e-type별)로 축소**(형태학 복제):
  | e-type | 대표 형태학 | 세포 수 | E/I |
  | --- | --- | --- | --- |
  | cACpyr | CA1_pyr_SP-PC_cACpyr_mpg141017_a1-2_idC | 15,723 | EXC |
  | cNAC | CA1_int_SO-BP_cNAC_980120A | 707 | INH |
  | cAC | CA1_int_SO-BP_cAC_980120A | 495 | INH |
  | bAC | CA1_int_SLM-PPA_bAC_011127HP1 | 722 | INH |
- **PC 활성채널 12종**: `nax`(Na)·`kdr`·`kap`·`kmb`·`kad`(K)·`kca`·`cagk`(Ca-activated K)·`hd`(Ih)·`can`·`cal`·`cat`(Ca)·`cacum`(Ca buffer). celsius 34, v_init −70.

## 3. 시냅스 모델
- **Ecker(2020) Table 3** 9클래스 EMS 시냅스. 확률판 `ProbAMPANMDA/ProbGABAAB_EMS`, 결정론판 `DetAMPANMDA/DetGABAAB`.
- **단기가소성(STP)**: Tsodyks-Markram `Use`/`Dep`/`Fac`/`Nrrp`.
- BBP EMS는 cvode 비호환 → **fixed dt 0.025**. NetCon weight = `g_nS`.

## 4. SC(샤퍼 곁가지, CA3→CA1) 구동
- 별도 SC 시냅스 클래스가 없어 **`PC->PC (E2)`로 대용**(코드 `SC_CLASS`, 논문 근거 명시).
- **전-네트워크 구동**(`11_schaffer/sc_full_slice.py`): 섬유별 **포아송**(기본 3 Hz, CA3 발화 대용), 세포당 `sc_pc`/`sc_int` 시냅스.
  - 생리 보정(E3d): **`sc_g_pc` ≈ 7.5 nS → PC ~1 Hz**(Mizuseki/Romani 대조).
- **fEPSP 순방향(E4·E4b·10초 유발)**: 유발전위 목적이라 위 자발구동 대신 **동기 SC 볼리**를 사용 — 대표 PC의 SR 정단수상돌기에 **40개 `DetAMPANMDA`(PC→PC E2, g=0.6 nS)**를 동시 활성. (E4b-9와 동일 구성 → 결과 연속.)

## 5. 전극 배치 (MEA)
- **확정 3×8** · 간격 200 µm · 직경 10 µm · 회전 0° · **24/24 전극 조직 위(100%)**.
- 얇은 SP 밴드(468 µm)엔 3×8이 최적(8×8은 47%만 조직 위). *실제 MEA 실험 스펙(전극 깊이·슬라이스 두께)은 추후 반영 예정.*

## 6. fEPSP 측정 (세포외 순방향 계산)
- 준정적·옴성·등방·균질 볼륨전도체: **V_j(t) = Σ_i M_ji I_i(t)**, I = `i_membrane_`(`use_fast_imem`, nA). 부호: 흥분성 sink → I<0 → 음성 fEPSP.
- 전달행렬 M: 단일세포=**LSA**(Holt&Koch 1999), MEA=**MoI 3층 영상법**(Ness 2015; 유리 z=0 절연 σ_G0 · 조직 σ_T0.3 · 식염수 σ_S1.5 · n_img20). 적대검증 통과(rel 1.7e-16).
- **집단 fEPSP** = 정렬·동기 PC 복제본 기여 합. 완전 정렬·동기·동일깊이라 **상한값**(생물학적 지터 약 −36%).

## 7. 실행 환경
- conda `ca1sim`(NEURON). 전슬라이스 CoreNEURON: CPU 9.57 h/초, GPU(A6000) 포팅 진행(계획서 Stage A~D).
- fEPSP 순방향은 **단일 대표 PC 막전류 + 전달행렬**이라 경량(수 분).

## 로드맵 (실험 트랙)
E1 baseline → E2 SC 경로 → E3 I-O·억제 차단 → **E4 fEPSP 계산기 ✅** → **E4b MEA 밴드·24전극 ✅** → **[현재] 10초 자극-반응 유발 fEPSP** → E9 실측 대조 → E8/E10 LTP(칼슘 가소성·STDP).
