# CLAUDE.md — 프로젝트 작업 지침 (Hippocampal CA1 in silico)

> 새 작업을 시작할 때 이 지침을 먼저 확인한다. 핵심: **모든 결과물은 Notion + GitHub에 함께 반영**한다.

## 프로젝트 개요
로컬 하드웨어에서 해마 CA1 회로를 in silico로 재현·검증하는 연구. 한 저장소에 두 트랙:
- **Track A — Ecker (2020) 시냅스 생리 재현**: `papers/01_Ecker2020_CA1_synaptic/`
- **Track B — like-slice CA1 시뮬레이터**(Romani 아틀라스 재사용): `like_slice_CA1/`
- 공용 자산: `shared/common`, `shared/mechanisms`
- 최종 목표: like-slice → **MEA fEPSP 재현 → LTP/LTD 재현 → 실측 데이터 비교**

## ⭐ 핵심 규칙: Notion + GitHub 동시 갱신
코드·그림·보고서가 바뀌면 **같은 작업 흐름에서 둘 다** 갱신한다.
- **Notion 보고서** — 결과보고서 페이지 갱신
  - 01 Ecker: 페이지 id `38f17cb5cdbd808db50bef1aebaa0fd2`
  - 02 Like-Slice: 페이지 id `38f17cb5cdbd80458cc8d899c34a01e9`
  - 상위: "07_Research" ▸ "[과제]신개념선행연구사업"
- **GitHub** — 아래 절차로 push

### Notion 작성 기준 (정확성 최우선)
- 캡션·표의 **모든 수치는 소스코드/실행 로그와 대조**해 확정 (기억에 의존 금지). 실행으로만 나오는 값은 스크립트 재실행으로 확인.
- 데이터는 **노션 네이티브 표**(파이프 표 → `<table>` 렌더).
- Notion API는 로컬 PNG 자동삽입 불가 → `> 🖼️ **그림 N.** 자세하고 정확한 캡션 — 경로` 형식 placeholder를 두고, 사용자가 GitHub에서 받아 드래그 삽입.

## GitHub (단일 저장소)
- **저장소 = 이 폴더(`03_BrainSimulator`) 자체.** 원격 `https://github.com/namseungyoon/BrainSimulation`. 별도 복사본 폴더를 만들지 않는다(제자리 추적).
- GitHub 구조 = 실제 구조: `papers/01_Ecker2020_CA1_synaptic/`, `like_slice_CA1/`, `shared/`, `demos/`.
- **올라가는 것**: 코드 + 결과 그림(PNG)만 (약 80MB).
- **제외(.gitignore)**: `like_slice_CA1/data`(대용량) · `shared/models`(BBP 라이선스) · `references`(PDF) · `_archive` · `legacy` · `*.dll` · `__pycache__` · `*.h5/*.nrrd/*.zip` 등.
- ⚠️ `.gitignore`는 **인라인 주석 미지원** — 주석은 반드시 줄 맨 앞에. 이미 스테이징된 파일은 .gitignore로 안 빠지니 `.git/index` 삭제 후 재`add`.

### push 절차 (두 세션 공용 저장소)
01(Ecker)·02(like-slice) 두 Claude 세션이 **같은 저장소를 공유**한다. 각자 자기 폴더만 건드리면 충돌 없음.
```
cd "d:\Project_2025_2026_HIPPO\Workspace\03_BrainSimulator"
git pull                 # ⚠️ push 전 필수 (상대 세션 커밋 먼저 받기)
git add <바뀐 폴더>       # 예: papers/01_Ecker2020_CA1_synaptic  또는  like_slice_CA1
git commit -m "설명"
git push                 # 일반 push. ⚠️ force-push 금지
```
- **force-push 금지.** 원격 재구성이 필요하면 `merge -s ours`로 비파괴 처리(기존 이력 보존).
- `like_slice_CA1`의 옛 독립 `.git`은 `.git_local_backup`로 백업됨(gitignore).

## 실행 환경 (NEURON)
- conda env `ca1sim`. 셸에서 `conda activate`가 안 되므로 **전체 경로** 사용:
  `C:\Users\SYNAM-OFFICE\.conda\envs\ca1sim\python.exe`
- BBP EMS 시냅스(ProbAMPANMDA/ProbGABAAB)는 cvode 비호환 → **고정 dt**(0.025). NetCon weight = nS를 µS로(/1000).
- matplotlib 한글 폰트(Malgun Gothic): `ĝ`·`−`(유니코드 마이너스) 등 결자 주의 → `g_hat`·`-`·`~` 사용.
