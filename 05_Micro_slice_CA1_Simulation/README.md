# 05_Micro_slice_CA1_Simulation

해마 **CA1 마이크로 슬라이스**(MEA 전극 2~3개 크기) in silico 시뮬레이터. Romani(2024) 파이프라인 기반, **전세포 완전형태 모델**, 백지 시작.
최종 목표: **MEA fEPSP → LTP/LTD → 실측 대조**.

- 전체 계획: [PLAN.md](PLAN.md)
- 확정 프로토콜 스냅샷: [PROTOCOL.md](PROTOCOL.md)
- 실험 레지스트리: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)

## 구조 (한눈에)

```
config/        기하·파라미터 단일 출처(YAML)          docs/     설계 노트·결정·데이터 출처·재사용 지도
lib/           번호 없는 import 모듈(유일 재사용 통로)   data/     Romani 원자료(gitignore)
01_tissue/     조직 기하        (Romani 0~4)
02_neurons/    뉴런 조성·배치·형태 (Romani 4b~6)
03_network/    연결·시냅스·구동 (Romani 7~9)
04_experiments/ 실험 E1~E10
env/           빌드·구동 런처(추적)                   scratch/  일회성(gitignore)
```

참조(복사 금지): `../shared/common` · `../shared/mechanisms` · `../Models`(BBP 20 단일세포 번들)

## 실행 원리

- import되는 재사용 코드 = `lib/`(번호 없음). 각 단계 폴더의 실행 스크립트는 번호 파일(예: `01_tissue/1_bbox/1_define_window.py`)로 두되 서로 import하지 않는다.
- 기하·dt 등 파라미터는 `config/*.yaml` 한 곳에서 읽는다(스크립트 하드코딩 금지).
- 추적: **코드 + 결과 PNG만**. 원자료·중간 산출물·로그는 `.gitignore`.

## 상태

⬜ 파이프라인 미착수 — 폴더·문서 골격만 존재(2026-08-13). 단계 0부터 사용자 확인 하에 진행.
