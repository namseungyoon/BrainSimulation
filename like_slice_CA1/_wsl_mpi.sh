#!/bin/bash
# MPI 라이브러리 설치 (openmpi)
set -e
echo "[MPI] openmpi 설치..."
sudo apt-get update -y
sudo apt-get install -y libopenmpi-dev openmpi-bin
echo "===== openmpi 설치 완료 ====="
mpicc --version 2>/dev/null | head -1
mpiexec --version 2>/dev/null | head -1
