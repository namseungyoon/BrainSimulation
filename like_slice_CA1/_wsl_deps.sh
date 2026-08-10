#!/bin/bash
# 3단계: NEURON 빌드 의존성 (WSL2 Ubuntu)
set -e
echo "[deps] apt 패키지 설치..."
sudo apt-get update -y
sudo apt-get install -y build-essential cmake bison flex git \
  python3 python3-dev python3-pip python3-venv python3-numpy \
  libncurses-dev libreadline-dev
echo "===== deps 완료 ====="
cmake --version | head -1
python3 --version
