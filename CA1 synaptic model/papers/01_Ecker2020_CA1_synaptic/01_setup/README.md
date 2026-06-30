# 00_setup — Phase 0: NEURON 환경 구축 (완료)

축소형 CA1 시냅스 생리 모델(Ecker et al. 2020)을 돌리기 위한 NEURON 환경 셋업 기록.

## 결론 — 사용법 (이후 모든 단계 공통)
```powershell
conda activate ca1sim
python SourceCode/00_setup/verify_setup.py   # 환경 검증
```
`conda activate ca1sim` 하면 activate.d 스크립트가 NEURON 환경변수를 자동 설정한다. **수동 설정 불필요.**

## 환경 구성 요약
| 항목 | 값 |
|------|-----|
| Python 환경 | conda env **`ca1sim`** (Python 3.10.20) — `C:\Users\SYNAM-OFFICE\.conda\envs\ca1sim` |
| NEURON | **8.2.7** (LTS), 공식 Windows 설치본(.exe, mingw 번들) |
| NEURON 설치 위치 | `C:\Users\SYNAM-OFFICE\nrn` (`NEURONHOME`) |
| 컴파일된 메커니즘 | `01_mechanisms/nrnmech.dll` (28개 mod) |

> **중요**: NEURON 은 PyPI·conda-forge 에 **Windows 휠을 제공하지 않는다**. 반드시 공식 .exe 설치본을 사용한다(릴리스: github.com/neuronsimulator/nrn).

## 수행한 작업
1. **NEURON 설치**: `nrn-8.2.7.w64-mingw-py-39-310-311-312-313-setup.exe` 를 `C:\Users\SYNAM-OFFICE\nrn` 에 무인 설치(`/S /D=`). mingw 컴파일러(`nrnivmodl`), py3.10 확장모듈 포함.
2. **conda 연동**: `ca1sim/etc/conda/activate.d/` 에 `neuron_env.bat`·`neuron_env.ps1` 생성 → 활성화 시 `NEURONHOME`/`PATH`(bin, mingw)/`PYTHONPATH`(lib\python) 설정.
3. **mod 컴파일**: 보유 BBP 메커니즘 28개(`legacy/Workspace/455999_model_files/mechanisms/hippocampus/mod/*.mod`)를 프로젝트 `01_mechanisms/` 로 복사 후 `nrnivmodl` → `nrnmech.dll`. (빌드산물 .c/.o 는 `_archive/mechanisms_build/` 로 이동)
4. **검증**(`verify_setup.py`): NEURON 로드 + `ProbAMPANMDA_EMS`/`ProbGABAAB_EMS`(논문 §2.5 stochastic TM 시냅스) 인스턴스화 + Table 3 파라미터(Use/Dep/Fac/Nrrp) 주입 성공.

## 공용 부트스트랩
`SourceCode/common/nrn_env.py` — 어느 스크립트에서든 NEURON DLL 경로·`nrnmech.dll` 을 일관 로드.
비활성 환경에서 `python.exe` 직접 실행 시에도 동작하도록 `NEURONHOME`/`add_dll_directory` 안전장치 포함.
```python
import sys, os
sys.path.insert(0, "<SourceCode 경로>")
from common.nrn_env import h, neuron, load_project_mechanisms
load_project_mechanisms()
```

## mod 재컴파일 (01_mechanisms 수정 시)
```powershell
conda activate ca1sim
cd 01_mechanisms
nrnivmodl .
```

## 핵심 매핑 (논문 Table 3 → EMS mod RANGE 변수)
| 논문 | mod 변수 |
|------|----------|
| U_SE | `Use` |
| D (회복 시정수) | `Dep` |
| F (촉진 시정수) | `Fac` |
| N_RRP | `Nrrp` |
| τ_rise/τ_decay | `tau_r_AMPA`/`tau_d_AMPA` (NMDA·GABA 동일 패턴) |
| NMDA/AMPA 비 | `NMDA_ratio` |
| E_rev | `e` |
| ĝ (peak conductance) | NetCon weight (µS) |
