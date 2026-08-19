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

**요약: 14항목 중 6항목 존재.** 재료(mod 소스·세포 번들)는 다 있고 **실행 도구가 통째로 없다.**

### 판정

- **1-2(Python)·1-3(NEURON)이 트랙 전체의 블로커다.** 둘 다 설치 없이는 2단계 이후가 불가능하다.
- `python_on_path` 가 `[O]` 로 찍히지만 **스텁이다.** 진단 스크립트는 파일 크기 1KB 미만이면
  `STUB` 접두어를 붙이도록 되어 있다. 이걸 파이썬으로 착각하면 1-2에서 헤맨다.
- 저장소 자산(mod 23개·추체 13종)은 **확보되어 있으므로** 설치만 끝나면 바로 1-4 빌드로 갈 수 있다.

## 1-2 Python — 예정 절차

⬜ **미실행.** 아래는 계획이며, 실행 후 이 절에 **실제로 한 것**으로 교체한다.

1. Miniconda(Windows x86_64) 설치 → `C:\Users\USER\miniconda3`
2. `conda env create -f ..\environment.yml` → env 이름 `ca1sim` (python 3.10.20)
3. 검증: `python -V` · `import numpy, scipy, matplotlib`

⚠️ `environment.yml` 은 **win-64 빌드 문자열이 고정**되어 있어 그대로는 실패할 수 있다.
실패 시 빌드 문자열을 떼고 재시도하고, **그 사실을 여기 기록**한다.

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
