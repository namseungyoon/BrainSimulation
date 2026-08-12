# 클라우드 A6000 선검증 — 확률 시냅스 CoreNEURON GPU repro

## 무엇을 판별하려는가
로컬(WSL2 · NEURON 9.0.1 · **NVIDIA HPC SDK 26.5** · CUDA 12.6 · RTX A6000 sm_86)에서:
- ✅ BBP 확률 시냅스(ProbAMPANMDA/GABAAB_EMS)가 **CoreNEURON GPU 컴파일 통과**(#638 지연연결 self-event를 `#ifndef CORENEURON_BUILD`로 가드해 해결).
- ✅ GPU 빌드 **CPU 모드**는 일반 CoreNEURON CPU와 **비트-동일**(6 스파이크 일치) → 가드 정확.
- ✅ 채널 13종 GPU 실행 OK(단일세포 IClamp).
- ❌ **확률 시냅스 GPU 디바이스 실행 = SEGFAULT (RC=139)** ← 이 repro가 재현하는 문제.

→ 이 세그폴트가 **로컬 툴체인(HPC SDK 26.5/WSL) 특유**인지, **Random123-on-GPU 근본 한계**인지를 다른 환경(클라우드 A6000)에서 판별.

## 파일
- `ProbAMPANMDA_EMS.mod`, `ProbGABAAB_EMS.mod` — 가드 적용된 확률 시냅스(원본 BBP + `#ifndef CORENEURON_BUILD` 지연연결 가드).
- `gpu_repro_test.py` — 자체완결 테스트(아틀라스/me-model 불필요): passive 단일구획 + 확률 시냅스 20개(Random123) + 포아송 NetStim → CoreNEURON GPU 실행.

## 클라우드 셋업 (A6000 인스턴스: Lambda/RunPod/Vast 등)
1. A6000(또는 임의 NVIDIA GPU) 인스턴스 + Ubuntu.
2. **NVIDIA HPC SDK 설치** — ★핵심 변수: **로컬과 다른, CI-검증된 버전**을 권장(예: 24.x).
   최신 26.5가 로컬 세그폴트 원인일 가능성 → NEURON 저장소 CI Dockerfile(`nrn/.github` 또는 `nrn/docker`)에서 검증된 NVHPC 버전 확인 후 맞추기.
3. **NEURON+CoreNEURON GPU 소스 빌드**:
   ```
   git clone --recursive https://github.com/neuronsimulator/nrn
   cd nrn && mkdir build && cd build
   cmake .. -DNRN_ENABLE_CORENEURON=ON -DCORENRN_ENABLE_GPU=ON \
     -DNRN_ENABLE_INTERVIEWS=OFF -DNRN_ENABLE_RX3D=OFF \
     -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_COMPILER=nvc++ -DCMAKE_CUDA_COMPILER=nvcc \
     -DCMAKE_CUDA_ARCHITECTURES=86 -DNRN_ENABLE_PYTHON=ON -DCMAKE_INSTALL_PREFIX=$HOME/nrn-gpu
   cmake --build . --parallel && cmake --build . --target install
   export PATH=$HOME/nrn-gpu/bin:$PATH; export PYTHONPATH=$HOME/nrn-gpu/lib/python:$PYTHONPATH
   ```
   (NMODL 코드생성에 python 필요 시 `export NMODL_PYLIB=<libpython3.x.so 경로>`.)

## 빌드 + 실행 (이 폴더에서)
```
export NMODLHOME=$HOME/nrn-gpu
nrnivmodl -coreneuron .                                  # nvc++ GPU 컴파일 → x86_64/special
export CORENEURONLIB=$PWD/x86_64/libcorenrnmech.so
x86_64/special -python gpu_repro_test.py cpu ; echo RC=$?   # 대조(CPU 백엔드)
x86_64/special -python gpu_repro_test.py gpu ; echo RC=$?   # ★ GPU 백엔드
```

## 판정
| GPU 실행 결과 | 해석 | 다음 조치 |
| --- | --- | --- |
| **`REPRO_OK backend=GPU ...` + RC=0** | 로컬 HPC SDK 26.5/WSL 조합 문제 | 클라우드가 쓴 **NVHPC 버전을 로컬에 맞춰** 재빌드 → 해결 가능성 |
| **SEGFAULT (RC=139) 재현** | Random123-on-GPU **근본 한계**(이 CoreNEURON+HPC SDK 계열) | 대안: 결정 시냅스 GPU화(R/u/tsyn→RANGE 리팩터) 또는 GPU는 채널만·확률 시냅스는 CPU |
| 컴파일 실패 | 그 NVHPC-CoreNEURON 조합 비호환 | 다른 NVHPC 버전 |

## 참고 (로컬에서 재현하려면)
로컬 WSL에서 동일 세그폴트 재현: `~/mods_gpu_src`(가드된 20 mods)로 빌드된
`x86_64/special -python gpu_repro_test.py gpu` → RC=139. `cpu` 인자면 정상(RC=0, Vm 상승).
