# 전 슬라이스 fEPSP 3D 시각화 — 계획 & 체크리스트

> 2% 테스트 자극에 대한 **전 슬라이스(17,647세포)** 반응을, 예시 `fepsp_3d_full.html`과
> 같은 3D 뷰로 재현한다. 예시는 locus 근처 5,610세포의 국소 뷰였고, 우리는 이를
> **전 슬라이스로 확장**한다("세포 수가 더 많은 like-slice").

## 규칙
- 여기 적는 것: 무엇을·어떤 순서로·무엇을 통과로 볼 것인가.
- 측정값·서사는 여기 적지 않는다. 결정값은 코드/런처에 산다(fact-single-home).
- 단계마다: 설명 → 실행 → 통과기준 확인 → 사용자 이해 후 다음.

---

## A. 무엇을 만드나 (결정값)

| 항목 | 값 | 근거 |
| --- | --- | --- |
| 자극 세기 | **2.0% = 섬유 4/200** (`io_test`/`io_levels` 0.02) | 1단계 IO 확정. 발화 0·침습률 0 |
| 규모 | **전 슬라이스 17,647세포** (`--counts full`) | "전체" |
| 세포 표현 | 세포당 **2점**(소마점 + 수상돌기 대표점) ≈ 35k점 | 예시 방식(2.25점/세포)을 전 슬라이스로 |
| 저장 fidelity | 세포별 **소마/수상돌기 막전류 합** (진짜 전 세그먼트 아님) | 전 세그 = 383만 → 수 GB, 브라우저 불가 |
| 시뮬 길이 | tstop 140 ms · stim_t 100 ms | 검증된 2% 조건과 동일 |
| 런 소요 | 셋업 1.2 h + 140 ms×56.2 s/ms ≈ **3.4 h** | 계획서 비용모델 |
| 프레임 | rec_dt 0.4 ms → ~350프레임 저장, HTML은 ~100으로 다운샘플 | 데이터 ~10–15 MB |

## B. 저장 데이터 (예시 `D` 구조 대응)

io npz(`_mea_<TAG>.npz`)에 아래 추가 — `--save_cellcur`로 켜짐:
`cell_pos`(2N,3) · `cell_soma`(2N) · `cell_cur`(2N,nt) · `Isoma`/`Idend`(nt) ·
`syn_xyz`(M,3) · `viz_V`(24,nt) · `viz_elec`(24,3) · `viz_lay_name/r0/r1` · `viz_box`(3) ·
`viz_stim_locus`(3) · `viz_t`(nt). 좌표계 = (장축 u · 층관통 r · 두께 w).

## C. 파이프라인 & 체크리스트

| 단계 | 무엇을 | 상태 |
| --- | --- | --- |
| **1** | 코드 변경 — `--save_cellcur` (축약·취합·저장) + 런처 배선 | ✅ 완료 (문법 통과) |
| **2** | **스모크** — 소규모(380세포)·2랭크로 저장 포맷·MPI 취합 검증 | ✅ 완료 (rc=0, 552s) |
| **3** | npz 필드·모양 검증 | ✅ 통과 (14필드·소마380/수상380·NaN 없음·2랭크 합산) |
| **4** | **전규모 2% 재구동** (~3.4 h, WSL) | ⏸ 대기 (사용자 UI 확인 후) |
| **5** | **HTML 빌더** `make_fepsp3d.py` — npz → `D` JSON → HTML | ✅ 완료 (좌표버그 수정·적응형 색스케일·재중심화) |
| **6** | UI 검수 — 스모크로 시각 언어 확정 | ✅ 사용자 승인 (범례=ΔI색·소마/수상돌기 크기+연결선·전극 흰사각+라벨·시냅스 색만변화·자극파형·휠줌) |
| **보조** | `preview_slice.py` — 시뮬 없이 전 17,647세포 위치·층·전극 3D 미리보기 | ✅ 완료 (기하 검증용) |

### 통과기준
- **2**: `_mea_Sviz_smoke.npz`에 B의 필드가 전부 있고, `cell_cur` 모양 = (2·세포수, 프레임), `cell_soma` 절반이 True, `syn_xyz` ≥ 0행, 크래시·교착 없음.
- **4**: 위와 동일 필드 + `n=17,647`·발화 0(2%) 확인. 첫 청크 로그로 완료시각 보고.
- **5/6**: 예시의 모든 UI 동작(3D 드래그·자동회전·재생/시크/속도·시냅스토글·catchment·fEPSP차트·Isoma/Idend차트·층밴드·박스·자극locus·타일). 24전극은 대표 #18 기본 + 나머지 토글로 조정.

## D. 실행 명령 (WSL)

```bash
cd "/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/01_Like_slice_CA1_Simulation"
# 스모크(소규모·2랭크·수분)
NRANK=2 COUNTS=300,40,20,20 SAVE_CELLCUR=1 IO_LEVELS=0.02 TAG=Sviz_smoke bash _wsl_stage.sh 1 gb
# 전규모 2% (~3.4h) — 스모크 통과 후
SAVE_CELLCUR=1 IO_LEVELS=0.02 TAG=Sviz_full bash _wsl_stage.sh 1 gb
```
⚠️ `NRANK`(전규모 기본 16) ↔ `.wslconfig` `processors` 일치(과다구독 방지).

## E. 손댈 파일
- `13_net_fepsp/mea_experiment.py` — `--save_cellcur` 저장 로직 (완료)
- `_wsl_stage.sh` — `COUNTS`/`SAVE_CELLCUR` 배선 + `LS` 경로 수정 (완료)
- `_wsl_net_fepsp.sh` — `LS` 경로 수정 (완료)
- `13_net_fepsp/make_fepsp3d.py` — HTML 빌더 (5단계, 예정)
- `13_net_fepsp/figures/fepsp3d_full.html` — 산출물 (예정)

## F. 함정
- 진짜 전 세그먼트(383만) 저장 금지 — 수 GB, 브라우저 불가. 반드시 2점/세포 축약.
- `--save_cellcur`는 io 전용(ltp 6,260 ms면 데이터 폭발) — 코드가 가드.
- 24전극 fEPSP 차트에 24선을 다 그리면 뭉친다 — 선택 전극(기본 #18)만.
- ★**하니스 경로 버그(2026-08-27 발견)**: 폴더 재편으로 옛 `like_slice_CA1`가 삭제됐는데
  `_wsl_stage.sh`·`_wsl_net_fepsp.sh`가 `LS`를 옛 경로로 하드코딩하고 있었다 →
  cd 실패. 둘은 `01_Like_slice_CA1_Simulation`로 고침. **나머지 ~38개 `_wsl_*.sh`
  (벤치·GPU·calib 등 일회성)도 같은 옛 경로**를 갖고 있다 — 필요 시 일괄 수정.
- ★**좌표 export 버그(2026-08-27 발견·수정)**: 세포 위치를 `cellgeom`의 세그먼트 좌표
  (`real = geom["mid"]@Rc + xyz`)로 잡았더니 **형태 원점 offset**이 실려 w가 −1700µm로
  튀었다. LFP는 자기완결이라 무관하지만 절대 위치 저장은 깨진다. → 소마점은 **검증된 배치
  좌표(xyz 노드) `_soma_frame(g)`**, 수상돌기점은 **소마 + (수상−소마) 상대벡터**(offset 상쇄),
  시냅스도 동일 보정. 빌더는 세포 구름 중심으로 **재중심화**. 실제 슬라이스: 1,596×2,328×
  **두께 490µm**(soma p1~p99 ±200), 층 SO(−139)·SP(0)·SR(+392)·SLM(+791)µm.
- 스모크(`--counts` 소수)는 **중심 근처 공**만 뽑는다([mea_experiment.py] 나이스트-센터 선택) →
  슬라이스 단면이 아니다. 기하 검증은 `preview_slice.py`(전 세포)로, 전류는 전규모 런으로.
