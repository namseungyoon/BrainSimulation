#!/bin/bash
# 가소성 mod를 포함한 plain NEURON MPI special 빌드
# 기존 mods_full_gpu_mpi(20종) + GBPlasticitySyn(A) + GBPlasticityStpSyn(B)
#                              + GBPlasticityStpProbSyn(C, 확률방출) → ~/mods_ltp
set -e
NVHPC=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5
export PATH=$HOME/nrn-gpu-mpi/bin:$NVHPC/comm_libs/mpi/bin:$NVHPC/compilers/bin:$PATH
export PYTHONPATH=$HOME/nrn-gpu-mpi/lib/python:$PYTHONPATH
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate nrngpu
export NMODLHOME=$HOME/nrn-gpu-mpi
export NMODL_PYLIB=$HOME/miniconda3/envs/nrngpu/lib/libpython3.11.so
SRC=/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/shared/mechanisms

rm -rf $HOME/mods_ltp
mkdir -p $HOME/mods_ltp
cp $HOME/mods_full_gpu_mpi/*.mod $HOME/mods_ltp/
cp $SRC/GBPlasticitySyn.mod $SRC/GBPlasticityStpSyn.mod $SRC/GBPlasticityStpProbSyn.mod $HOME/mods_ltp/
N=$(ls $HOME/mods_ltp/*.mod | wc -l)
echo "mod 수: $N (A=GBPlasticitySyn · B=GBPlasticityStpSyn · C=GBPlasticityStpProbSyn 포함)"

cd $HOME/mods_ltp
rm -rf x86_64
nrnivmodl . 2>&1 | tail -6
if [ -x $HOME/mods_ltp/x86_64/special ]; then
  echo "BUILD OK: $HOME/mods_ltp/x86_64/special"
  # 세 POINT_PROCESS(A·B·C)가 실제로 들어갔는지 확인(0단계 통과기준 #4)
  # ⚠ POINT_PROCESS는 **섹션이 access 된 상태**여야 만들어진다. 섹션 없이
  #   `new GBPlasticitySyn(0.5)` 만 하면 이미 잘 쓰던 mod도 실패한다(2026-08-06 오진).
  #   그래서 -c 나열이 아니라 hoc 파일로 확인한다(따옴표 중첩 문제도 함께 회피).
  cat > $HOME/mods_ltp/_loadcheck.hoc <<'HOC'
create s
access s
objref a, b, cc
a = new GBPlasticitySyn(0.5)
b = new GBPlasticityStpSyn(0.5)
cc = new GBPlasticityStpProbSyn(0.5)
printf("LOADCHECK A rho0=%g gamma_p=%g b=%g\n", a.rho0, a.gamma_p, a.b)
printf("LOADCHECK B rho0=%g gamma_p=%g Use=%g Dep=%g Fac=%g norm_Pr=%g ca_stp=%g\n", \
       b.rho0, b.gamma_p, b.Use, b.Dep, b.Fac, b.norm_Pr, b.ca_stp)
// 모델 C는 setRNG 까지 확인한다 — 이 호출이 없으면 urand()가 0.0 을 돌려줘
// Nrrp=1 에서 '첫 펄스 1회 방출 후 영구 침묵'으로 조용히 죽으므로(에러 없음),
// 빌드 시점에 호출 가능 여부를 반드시 본다.
cc.setRNG(1, 2, 3)
printf("LOADCHECK C rho0=%g gamma_p=%g Use=%g Nrrp=%g norm_Pr=%g ca_stp=%g setRNG=ok\n", \
       cc.rho0, cc.gamma_p, cc.Use, cc.Nrrp, cc.norm_Pr, cc.ca_stp)
quit()
HOC
  if $HOME/mods_ltp/x86_64/special -nobanner $HOME/mods_ltp/_loadcheck.hoc 2>&1 | grep LOADCHECK; then
    echo "  LOAD OK: 세 POINT_PROCESS(A·B·C) 모두 생성됨 + C는 setRNG 호출까지 확인"
  else
    echo "  LOAD FAIL"
  fi
else
  echo "BUILD FAILED"
fi
