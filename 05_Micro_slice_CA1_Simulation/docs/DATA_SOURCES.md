# 데이터 출처 — micro-slice CA1

## Romani(2024) CA1 — Harvard Dataverse
- DOI: **10.7910/DVN/TN3DUI**
- URL: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/TN3DUI
- 받는 방식: 여러 파일을 하나로 묶은 **통합 zip**(`dataverse_files.zip`, 4.83 GB) 다운로드.

### 포함 파일 (통합 zip 내부)
| 파일 | 용량(byte) | 용도 |
|---|---|---|
| circuit.zip | 2,526,564,872 | SONATA nodes = 456,380세포 배치(좌표·방향·m/e-type·층) |
| atlas.zip | 296,964,746 | 부피·층·방향장 NRRD |
| morphology_library.zip | 2,488,603,694 | 세포별 실제 형태학(.asc) — 전세포 완전모델용 |
| single_cell_model_library.zip | 725,696 | Romani e-model |
| README.md | 6,341 | 데이터셋 설명 |

합계 5,312,865,349 byte = 목록 총계 일치(손상 없음 확인 2026-08-13).

### 배치
- 통합 zip: `data/incoming/dataverse_files.zip` (다운로드 감시기가 자동 이동).
- 압축 해제 예정: `data/raw/{circuit,atlas,morphology_library,single_cell_model_library}/`.
- 절차: [../data/README.md](../data/README.md).

## BBP 단일세포 번들 (별도 보유)
- 위치: `../Models` (05 기준). 20종(PC cACpyr 1 + INT bAC/cAC/cNAC 19).
- 각 = hoc CCell 템플릿 + morphology.swc + mechanisms/*.mod(12채널) + neuron_simulation.py.
- BBP 라이선스 → **git 제외**(루트 `.gitignore`에 `Models/`).

## 실측 MEA 데이터 (E9, 추후)
- TBD — 실측 HD-MEA 샤퍼 곁가지 fEPSP 데이터셋.
