#!/bin/bash
# 빌드 병목이 /mnt/d 9P I/O인지 판별: models를 WSL 네이티브로 복사 후 100세포 빌드 시간 측정.
NVHPC=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5
export PATH=$HOME/nrn-gpu/bin:$NVHPC/compilers/bin:$NVHPC/cuda/bin:$PATH
export PYTHONPATH=$HOME/nrn-gpu/lib/python:$PYTHONPATH
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate nrngpu
export NMODLHOME=$HOME/nrn-gpu
export NMODL_PYLIB=$HOME/miniconda3/envs/nrngpu/lib/libpython3.11.so
export CORENEURONLIB=$HOME/mods_full_gpu/x86_64/libcorenrnmech.so
SPECIAL=$HOME/mods_full_gpu/x86_64/special
LS=/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/like_slice_CA1
echo "=== models 네이티브 복사 $(date +%T) ==="
rm -rf ~/models_native && cp -r /mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/shared/models ~/models_native
echo "복사 완료 $(date +%T)  크기: $(du -sh ~/models_native | cut -f1)"
export MODELS_DIR=$HOME/models_native
cd "$LS"
echo "=== 네이티브 FS 100세포 빌드 시간 측정 (GPU, tstop 짧게) $(date +%T) ==="
timeout 400 $SPECIAL -python 11_schaffer/sc_full_slice.py \
  --counts 60,20,10,10 --tstop 20 --seg_ms 20 --dt 0.025 --det \
  --sc_rate 150 --n_fiber 100 --sc_pc 60 --sc_int 40 --sc_g_pc 10 --sc_g_int 3 \
  --coreneuron --gpu --outdir sc_det_gpu/nativetest > ~/native_build.log 2>&1
echo "RC=$?"
grep -avE "Target stub|equivalent length" ~/native_build.log | grep -aE "1/4|2/4|3/4|4/4|완료"
echo "DONE_NATIVE $(date +%T)"
