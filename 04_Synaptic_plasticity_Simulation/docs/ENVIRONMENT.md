# 실행 환경 — 이 머신 전용 기록

> 이 문서는 **이 머신에서 실제로 한 일**만 적는다. 다른 머신의 절차를 옮겨 적지 않는다.
> 다른 트랙 문서가 전제하는 환경(`C:\Users\SYNAM-OFFICE\`)은 **이 머신이 아니다.**

## 1-1 진단 결과 (2026-08-19)

`powershell -ExecutionPolicy Bypass -File env\probe_env.ps1` 실행 결과.
원자료: `01_env/1_probe/figures/_1-1_probe.json`

| 항목 | 판정 | 실측값 |
|---|:---:|---|
| `python_on_path` | ⚠️ | `C:\Users\USER\AppData\Local\Microsoft\WindowsApps\python.exe` — **WindowsApps 스텁**(1KB 미만 reparse point). 실행하면 Microsoft Store 안내만 뜬다. **파이썬이 아니다** |
| `python_version` | ❌ | 없음 |
| `conda` | ❌ | 없음 |
| `conda_dirs` | ❌ | miniconda3 · anaconda3 · `.conda\envs` **어느 경로도 없음** |
| `NEURONHOME_env` | ❌ | 미설정 |
| `neuron_install` | ❌ | `%USERPROFILE%\nrn` · `C:\nrn` · `C:\Program Files\NEURON` 전부 없음 |
| `nrnivmodl` | ❌ | 없음 |
| `dll_local_04` | ❌ | `mechanisms\nrnmech.dll` 없음 (아직 빌드 안 함 — 정상) |
| `dll_shared` | ❌ | `shared\mechanisms\nrnmech.dll` 없음 |
| `mod_sources` | ✅ | `shared/mechanisms` 에 **`.mod` 23개** |
| `pyr_bundles` | ✅ | `Models/` 에 **CA1 추체 번들 13종** |
| `git` | ✅ | 2.55.0.windows.4 |
| `write_access` | ✅ | 쓰기 가능 |
| `free_space_GB` | ✅ | 1830.6 GB |

**요약: 14항목 중 5항목 존재.** 재료(mod 소스·세포 번들)는 다 있고 **실행 도구가 통째로 없다.**

### 판정

- **1-2(Python)·1-3(NEURON)이 트랙 전체의 블로커다.** 둘 다 설치 없이는 2단계 이후가 불가능하다.
- `python_on_path` 는 파일이 존재하지만 **스텁이라 존재로 세지 않는다**(판정 `!`).
  진단 스크립트가 크기 1KB 미만이면 `STUB` 접두어를 붙인다. 이걸 파이썬으로 착각하면 1-2에서 헤맨다.
  ⚠️ 최초 구현은 스텁을 `O` 로 세어 **6/14** 로 보고했다. 그림 스크립트는 스텁을 제외해 **5/14** 를 냈고,
  숫자가 둘로 갈렸다. 스텁은 쓸 수 없으므로 **5/14 가 맞고**, 진단 스크립트를 그에 맞춰 고쳤다.
- 저장소 자산(mod 23개·추체 13종)은 **확보되어 있으므로** 설치만 끝나면 바로 1-4 빌드로 갈 수 있다.

### ★ 동결 기준선과 현재 상태는 다른 파일이다

`env/probe_env.ps1` 은 산출을 둘로 나눈다.

| 파일 | 무엇 | 추적 |
|---|---|---|
| `figures/1-1_env_probe.json` | **동결 기준선** — 설치 전 상태. 한 번만 쓰고 **절대 덮어쓰지 않는다** | O |
| `figures/_env_probe_latest.json` | **현재 상태** — 언제든 재실행 | X (gitignore) |

동결하지 않으면 1-1 그림이 조용히 "나중 머신"을 설명하게 된다. 실제로 1-2 직후 재실행했을 때
그림이 설치된 파이썬을 못 보고 "없음"으로 그리는 일이 벌어졌다(진단기가 PATH만 봤고,
우리는 `PrependPath=0` 으로 설치해 PATH를 안 건드렸기 때문). 진단기는 이제 설치 경로와
04 venv 를 직접 확인한다.

## 1-2 Python — ✅ 완료 (2026-08-19)

**conda 를 쓰지 않는다.** 근거는 [DECISIONS.md](DECISIONS.md) D7 (사내 정책상 Anaconda 무료 사용 불가).

### 실제로 한 것

```powershell
# 1) 내려받기 (HTTPS, python.org 직접)
#    python-3.11.9-amd64.exe  25 MB
#    sha256 5EE42C4EEE1E6B4464BB23722F90B45303F79442DF63083F05322F1785F5FDDE
# 2) 사용자 단위 무인 설치 — 관리자 권한 불필요, 시스템 PATH 미변경
python-3.11.9-amd64.exe /quiet InstallAllUsers=0 TargetDir=C:\Users\USER\Python311 `
    PrependPath=0 Include_test=0 Include_launcher=0 AssociateFiles=0 Shortcuts=0 `
    Include_doc=0 Include_tcltk=0
# 3) 04 전용 venv
C:\Users\USER\Python311\python.exe -m venv <트랙루트>\.venv
# 4) 패키지
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install numpy scipy matplotlib pyyaml
```

### ★ 04 인터프리터 단일 출처

```
<트랙루트>\.venv\Scripts\python.exe
```

**이 경로만 쓴다.** 다른 트랙의 `ca1sim` 은 04와 무관하다. `.venv/` 는 gitignore 대상이다.

### 검증 결과 — `01_env/2_python/1-2_verify_python.py` (10/10 통과)

| 항목 | 값 |
|---|---|
| Python | 3.11.9 · 64bit |
| venv 안에서 실행 | 예 |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| matplotlib | 3.11.1 (백엔드 **Agg**) |
| pyyaml | 6.0.3 |
| 한글 폰트 | Malgun Gothic 존재 |
| tkinter | **없음(의도)** |

산출: `01_env/2_python/figures/1-2_python_env.png` · `1-2_python_env.json`

### 설치 옵션을 그렇게 준 이유

| 옵션 | 이유 |
|---|---|
| `PrependPath=0` | 시스템 PATH 를 건드리지 않는다. 다른 도구와 충돌하지 않는다. **대신 진단기가 PATH만 보면 새 파이썬을 못 찾으므로** 설치 경로를 직접 확인하도록 고쳤다 |
| `InstallAllUsers=0` | 관리자 권한 불필요 |
| `Include_tcltk=0` | tkinter 미설치 → 창을 띄울 수 없다. 어차피 규약이 `Agg` 고정이라 무해하고, 설치가 가벼워진다 |
| `Include_launcher=0` | 시스템 전역 `py` 런처를 만들지 않는다 |

## 1-3 NEURON — 예정 절차

⬜ **미실행.**

1. `nrn-8.2.7.w64-mingw-py-39-310-311-312-313-setup.exe` 를 `C:\Users\USER\nrn` 에 설치
2. `env/activate.ps1` 작성 — `NEURONHOME` · `PATH`(`bin`, `mingw\usr\bin`) · `PYTHONPATH`(`lib\python`) 설정
3. 검증: `python -c "from neuron import h; print(h.nrnversion())"`

### ★ 공유 파일을 고치지 않아도 되는 이유

`shared/common/nrn_env.py:23` 이

```python
NRN_HOME = os.environ.get("NEURONHOME", r"C:\Users\SYNAM-OFFICE\nrn")
```

이므로 **환경변수가 있으면 하드코딩 기본값은 무시된다.** `env/activate.ps1` 이 `NEURONHOME` 을 세우면
공유 파일을 건드릴 필요가 없고, 따라서 01·05 트랙에 영향이 없다.
(→ [DECISIONS.md](DECISIONS.md) D4)

⚠️ conda 의 `activate.d` 훅은 `conda env create` 로 복원되지 않는다. 손으로 만들거나
`env/activate.ps1` 을 매번 호출하는 방식 중 하나를 골라야 한다 — 1-3에서 결정하고 기록한다.

## 알려진 함정

- **NEURON 은 conda 패키지가 아니다.** `environment.yml` 에 없다. 별도 Windows 설치본이다.
- **`nrnivmodl` 은 mingw 툴체인을 쓴다.** `PATH` 에 `%NEURONHOME%\mingw\usr\bin` 이 없으면 빌드가 실패한다.
- **POINT_PROCESS 는 section 이 access 된 상태에서만 생성된다.** 1-5 검증 스크립트는
  더미 section 을 먼저 만들어야 한다.
