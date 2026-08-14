# 실험 레지스트리 (E1~E10) — micro-slice CA1

> 단일 색인: **Notion번호 ↔ 코드경로 ↔ 그림**. 실험이 카테고리에 분산돼도 여기서 한눈에 추적.
> 공통 양식(각 실험): 목표 / 방법·입력 / 검증지표 / 결과·상태 / 근거(논문X→우리Y) / 한계·주의.
> 완료는 결과까지, 미실행은 "(예상·미실행)".

| ID | 실험 | 폴더 | Notion | 상태 |
|---|---|---|---|---|
| E1 | baseline 발화율·구동 검증 | `04_experiments/E1_baseline` | TBD | ⬜ |
| E2 | Schaffer collateral(CA3→CA1) | `04_experiments/E2_schaffer` | TBD | ⬜ |
| E3 | SC I-O + 억제 차단 | `04_experiments/E3_io_inhibition` | TBD | ⬜ |
| E4 | fEPSP 계산기(LSA) | `04_experiments/E4_fepsp` | TBD | ⬜ |
| E4b | MEA 3층 영상법(MoI) 밴드 | `04_experiments/E4b_mea_band` | TBD | ⬜ |
| E5 | theta 변조 입력 + PAC | `04_experiments/E5_theta_pac` | TBD | ⬜ |
| E6 | 내측중격(MS) theta | `04_experiments/E6_ms_theta` | TBD | ⬜ |
| E7 | ACh 신경조절 | `04_experiments/E7_ach` | TBD | ⬜ |
| E8 | LTP/LTD(칼슘 가소성) | `04_experiments/E8_ltp` | TBD | ⬜ |
| E9 | 실측 MEA 대조(최종) | `04_experiments/E9_realdata_mea` | TBD | ⬜ |
| E10 | STDP 곡선(Wittenberg 2006) | `04_experiments/E10_stdp` | TBD | ⬜ |
| E11 | cholinergic theta 위상의존 양방향 가소성(Huerta & Lisman 1995) | `04_experiments/E11_chol_theta_plasticity` | TBD | ⬜ |

## 공용 도구
- fEPSP 순방향 3기법: `lib/mea_forward.py` — **PSA**(점원)·**LSA**(선원, Holt&Koch 1999)·**MoI**(3층 영상법, Ness 2015).
- 장기가소성 mod: `../shared/mechanisms`(GBPlasticity류) 참조 — E8·E10 공유.

## 최종목표 사슬
E4·E4b(fEPSP) → E8·E10(가소성) → **E9(실측 대조)**.

## E8 상세 — LTP/LTD 유도 프로토콜 (가능성 확인 ✅)
- **인프라**: `../shared/mechanisms/GBPlasticity{Syn,StpSyn,StpProbSyn}.mod`(Graupner-Brunel 칼슘 가소성). c(t)→ρ 이중안정→w=w0+ρ(w1−w0).
- **E8-HFS (LTP)**: 100 Hz × 1초 테타너스 → LTP. 근거 Bliss & Lømo 1973; Bliss & Collingridge 1993.
- **E8-LFS (LTD)**: 1~3 Hz × 7~15분 → LTD. 근거 Dudek & Bear 1992; Mulkey & Malenka 1992. (긴 런 420~900초 → 마이크로 조직이라 현실적, GPU 권장)
- **E8-TBS**: theta-burst(기존). → 유도 프로토콜 3종(HFS·LFS·TBS) + E10 STDP.
- **측정**: 정규화 fEPSP slope(% baseline). **주의**: Graupner 파라미터 정량 검증 필요(방향은 재현 기대).

## E11 상세 — cholinergic theta 위상의존 양방향 가소성 (Huerta & Lisman 1995 재현)
- **프로토콜**: 카바콜 유발 theta 중 **단일 버스트(4펄스·100 Hz)**를 theta **위상**에 정렬 → **peak=LTP · trough=LTD**(이전 강화 시냅스) · heterosynaptic LTD.
- **의존**: E5/E6(theta) + E7(ACh·mAChR) + E8(Graupner mod) 결합 = **캡스톤**.
- **검증**: 위상별 부호(peak +, trough −) · NMDA + muscarinic 의존.
- **근거**: Huerta & Lisman 1995, *Neuron* 15(5):1053–1063 (PMID 7576649).

---

_각 실험 상세는 진행 시 이 문서에 절을 추가한다._
