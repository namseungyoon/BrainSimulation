# 실험 인덱스 (Experiments Index)

> **Notion 번호 ↔ GitHub 경로 ↔ 그림** 정렬 마스터. 새 실험을 추가하면 이 표에 한 줄씩 더한다.
> 상세 목표·방법·근거·검증·정직성은 [PLAN.md](PLAN.md) §11 참조. 이 파일은 "무엇이 어디에 있는가" 색인.

## 번호 규칙
- `E{주제}` = 주제 (예: E2 = Schaffer collateral 경로)
- `E{주제}-{a,b,c…}` = 그 주제 안의 하위 실험. 새 궁금증이 생기면 d, e… 로 확장.
- 그림 파일 = `E{주제}{letter}_설명.png|gif` (예: `E2a_sc_epsp.png`, `E3b_tuning_comparison.png`)
- 폴더 = GitHub 구조(주제별 번호 폴더), 번호 = Notion 정렬용. 둘을 이 표가 연결.

## 문서 구조 규칙 (Notion §11 · 보고서 공통)
모든 주제·하위 실험에 **동일 구조**를 적용한다:
- **`E{주제}`** = ① 주제 설명(무엇을·왜) → ② 공통 구성/방법론(있으면) → ③ 하위 실험들 → ④ **결론**
- **`E{주제}-{letter}`** = **목표 / 방법 / 내용**(완료면 결과·그림·⚠️정직성 포함)
- **결론**(주제마다 필수): 완료 = 실제 결론(핵심 발견 + 한계), **미실행 = "(예상·미실행) … 실행 후 작성"**(결과 날조 금지)
- 그림 자리 = `> 🖼️ 그림 …(캡션) — 경로` + `> !파일명`(Notion 이미지 드래그용)

## 두 계열
- **V 계열** (`00_`~`09_`, 그림 `V0`~`V7`): 슬라이스 **구축·검증** 파이프라인(층·조성·배치·연결·시냅스·baseline). = 재료.
- **E 계열** (`10_`~, 그림 `E1`~): 실제 **실험**(아래 색인). = 연구.

## 진행 현황

| ID | 주제 | 상태 |
| --- | --- | --- |
| E1 | Baseline 발화율·구동 검증 | ✅ 완료 |
| E2 | Schaffer collateral 경로 | 🔄 E2-a·E2-b ✅ / E2-c subset✅·전슬라이스⬜ |
| E3 | SC 자극 I-O + 억제 차단 | ✅ subset 완료(피드포워드 억제 작동, gap 71%p) |
| E4 | 세포외 LFP/fEPSP 계산기 | ⬜ 예정 |
| E5 | theta 변조 SC 입력 + PAC | ⬜ 예정 |
| E6 | 내측중격(MS) theta | ⬜ 예정 |
| E7 | ACh 신경조절 | ⬜ 예정 |
| E8 | LTP/LTD (칼슘 가소성) | ⬜ 예정 |
| E9 | 실측 MEA 대조 | ⬜ 예정 |
| E10 | STDP 곡선 (Wittenberg & Wang 2006) | ⬜ 예정 (모델=신규 장기가소성 mod) |

## 상세 색인 (하위 실험 ↔ 코드 ↔ 그림)

| ID | 하위 실험 | 상태 | GitHub 코드 | 그림 파일 |
| --- | --- | --- | --- | --- |
| **E1** | *Baseline 발화율·구동* | | `10_analysis/` | |
| E1-a | baseline 발화율 검증 | ✅ | `firing_stats.py` | `10_analysis/figures/E1a_firing_baseline.png` |
| E1-b | 외부구동 강도 스윕 | ✅ | `drive_sweep.py` · `plot_drive_sweep.py` | `10_analysis/figures/E1b_drive_sweep.png` |
| **E2** | *Schaffer collateral 경로* | | `11_schaffer/` | |
| E2-a | 단일 SC→PC EPSP + 배치 | ✅ | `sc_epsp_test.py` · `sc_epsp_placement.py` | `11_schaffer/figures/E2a_sc_epsp.png` · `E2a_sc_epsp_placement.gif` |
| E2-b | 조용한 슬라이스 + 볼리 → PC 반응 (subset) | ✅ | `sc_network.py` | (콘솔 검증, 그림 없음) |
| E2-c | 포아송 지속구동 9초 (subset 2,000세포) | 🔄 subset✅ | `sc_full_slice.py` · `sc_full_analysis.py` · `sc_input_viz.py` | `11_schaffer/figures/E2c_sc_input.png` · `E2c_full_firing.png` |
| **E3** | *SC 자극 I-O + 억제 차단* | | `11_schaffer/` | |
| E3(개념) | 피드포워드 억제 회로 개념도 | ✅ | `e3_concept.py` | `11_schaffer/figures/E3_concept.png` |
| E3-a | I-O 곡선 + 억제 차단 결과 | ✅ | `sc_io_curve.py` | `11_schaffer/figures/E3a_sc_io_curve.png` |
| E3-b | 튜닝 전/후 비교 | ✅ | `e3_tuning_compare.py` | `11_schaffer/figures/E3b_tuning_comparison.png` |
| **E4** | *세포외 LFP/fEPSP* | ⬜ | (예정, LFPykit 기반) | |
| **E5** | *theta 변조 SC + PAC* | ⬜ | (예정) | |
| **E6** | *내측중격 theta* | ⬜ | (예정) | |
| **E7** | *ACh 신경조절* | ⬜ | (예정) | |
| **E8** | *LTP/LTD 칼슘 가소성* | ⬜ | (예정, `papers/02` Graupner + 신규 mod) | |
| **E9** | *실측 MEA 대조* | ⬜ | (예정, SpikeInterface) | |
| **E10** | *STDP (Wittenberg & Wang 2006)* | ⬜ | (예정, 신규 장기가소성 NMODL + `papers/02` Wittenberg fit) | |

## 정직성 원칙 (전 실험 공통)
- 완료는 결과까지, 미실행은 계획·근거만. **결과 날조 금지.**
- 튜닝값 ≠ 측정값. 감소 모델이라도 논문 근거로 "논문 X → 우리 Y → 이유" 명시.
- subset·보정값·한계를 캡션·⚠️에 반드시 표기.
