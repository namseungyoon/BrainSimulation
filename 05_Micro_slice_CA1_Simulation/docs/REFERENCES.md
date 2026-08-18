# 근거 문헌 (References) — 05 Micro-slice CA1

각 논문을 **어디에 왜 쓰는지**까지 명시. 인용은 검증 완료(웹/PubMed 대조).

## 1. 모델·데이터 기반
- **Romani et al. 2024**, *PLoS Biology* — 전체 CA1 모델·아틀라스·커넥텀 **파이프라인의 원본**. 05는 이 파이프라인을 마이크로슬라이스에 적용. DOI [10.1371/journal.pbio.3002861](https://doi.org/10.1371/journal.pbio.3002861)
- **Romani, A. 2024**, *Harvard Dataverse* "Rat CA1 model" — 회로(nodes)·atlas·morphology_library·단일세포모델 **데이터 출처**. DOI [10.7910/DVN/TN3DUI](https://doi.org/10.7910/DVN/TN3DUI)
- **Bezaire & Soltesz 2013**, *Hippocampus* 23:751–785 — CA1 **정량 연결체**(세포수·interneuron→PC 수렴도). 3-1(b) pruning 목표치(연결확률·시냅스수/연결)의 근거. DOI [10.1002/hipo.22141](https://doi.org/10.1002/hipo.22141) · PMID 23674373
- **Kohus et al. 2016** — pair recording. Hub Connection Physiology(STP 22 rules)가 인용한 **연결 생리 출처**. *(정확 서지 확인요 — Hub 인용 기준)*
- **HippocampusHub** (EBRAINS/Blue Brain Project) — Schaffer collateral 수렴도·STP, 연결 anatomy/physiology, 단일세포 모델 다운로드. https://www.hippocampushub.eu

## 2. 시냅스 축소 근거 (3-1(a) SC 그룹화)
- **Amsalem et al. 2020**, *Nature Communications* — **Neuron_Reduce**. 전달임피던스 동일 시냅스 병합 + 개별 타이밍 유지 → 전압/수상돌기 연산 보존. 시냅스 그룹화의 **검증 근거**.
- **Wybo et al. 2020**, *eLife* — 수상돌기 축소·**시냅스 그룹화 타당 조건**(전도도 변동 한계) 정량화.

## 3. 장기가소성 모델 (LTP/LTD 테스트, 예정)
- **Graupner & Brunel 2012**, *PNAS* 109:3991–3996 — **칼슘 기반 가소성 모델**(칼슘 문턱 θ_p/θ_d, bistable 효능). LTP/LTD 모델의 **기반**. DOI [10.1073/pnas.1109359109](https://doi.org/10.1073/pnas.1109359109)
- **Chindemi et al. 2022**, *Nature Communications* — 칼슘 가소성을 **커넥텀 스케일**로 구현(NMDA+VDCC 칼슘, 시냅스별). 연결쌍 적용 **구현 근거**. DOI [10.1038/s41467-022-30214-w](https://doi.org/10.1038/s41467-022-30214-w)

## 4. fEPSP 순방향 모델 (E4/fEPSP, 예정 — 01 트랙에서 검토됨)
- PSA(point-source) · LSA **Holt & Koch 1999** · MoI(method-of-images) **Ness et al. 2015** — 세포외 전위(LFP/fEPSP) 계산 기법. *(05 도입 시 정확 서지 재확인)*

---
**정직성:** DOI·연도 확정 항목은 웹/PubMed 대조 완료. "*확인요*" 표시 항목(Kohus 2016, fEPSP 기법 서지)은 도입 시점에 정확 서지 재확인 예정.
