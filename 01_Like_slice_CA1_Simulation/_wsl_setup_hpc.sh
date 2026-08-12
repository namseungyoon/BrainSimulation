#!/bin/bash
# NVIDIA HPC SDK 26.5 설치 (WSL2 Ubuntu) — 2단계
set -e
echo "[1/4] GPG 키 등록..."
curl -s https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg
echo "[2/4] apt 저장소 추가..."
echo 'deb [signed-by=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' | sudo tee /etc/apt/sources.list.d/nvhpc.list
echo "[3/4] 패키지 목록 갱신..."
sudo apt-get update -y
echo "[4/4] HPC SDK 26.5 설치 (큰 다운로드, 오래 걸림)..."
sudo apt-get install -y nvhpc-26-5
echo "===== 설치 완료. 아래 nvc++ 버전 확인 ====="
ls /opt/nvidia/hpc_sdk/Linux_x86_64/ 2>/dev/null
