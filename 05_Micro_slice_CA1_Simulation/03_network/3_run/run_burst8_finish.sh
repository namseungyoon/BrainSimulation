#!/bin/bash
# tr8(8Hz) 완주 후 마무리 체인 — 3주파수 비교 + 8Hz 3D HTML. 가벼움(NEURON 불필요).
# 1) tr8 metrics 추출 → 2) 3주파수 burst 요약 그림 → 3) 8Hz 3D fEPSP HTML.
export PYTHONIOENCODING=utf-8
PY=/home/synam0216/miniconda3/envs/ca1sim/bin/python
BASE=/mnt/d/PROJECT/Project_2025_2026_HIPPO/03_Research_Development/02_BrainSimulator/05_Micro_slice_CA1_Simulation
DATA=$BASE/04_experiments/Ex3c_microSlice_burst/data
UI=$BASE/04_experiments/Ex3c_microSlice_burst/ui
cd "$BASE/03_network/3_run"
mkdir -p "$UI"

if [ ! -s "$DATA/mpi_fepsp_tr8_norm.npz" ]; then
  echo "[burst8] ✗ mpi_fepsp_tr8_norm.npz 없음 — tr8 미완주. 중단"; exit 1
fi
echo "[burst8] 1/3 tr8 metrics 추출"
$PY fepsp_metrics.py --fe "$DATA/mpi_fepsp_tr8_norm.npz" || { echo "[burst8] ✗metrics 실패"; exit 1; }

echo "[burst8] 2/3 3주파수(8/40/100Hz) 요약 그림"
$PY ex3c_burst_summary.py || { echo "[burst8] ✗summary 실패"; exit 1; }

echo "[burst8] 3/3 8Hz 3D fEPSP HTML"
$PY build_fepsp3d_full.py --fe "$DATA/mpi_fepsp_tr8_norm.npz" --base "$DATA/mpi_baseline_tr8_norm.npz" \
    --out "$UI/fepsp3d_tr8_norm.html" || { echo "[burst8] ✗HTML 실패"; exit 1; }

echo "[burst8] 완료 — 산출물:"
ls -la --time-style=+%H:%M "$DATA/metrics_tr8_norm.json" "$BASE/04_experiments/Ex3c_microSlice_burst/figures/ex3c_burst_summary.png" "$UI/fepsp3d_tr8_norm.html" 2>/dev/null
