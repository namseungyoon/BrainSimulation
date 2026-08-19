#!/usr/bin/env bash
# 05 Micro-slice CA1 — NEURON mechanism 컴파일 (3-3 구동 준비)
#
# 사전: conda env ca1sim + NEURON + 컴파일러
#   pip install neuron                                    # NEURON 9.0.2
#   conda install -y -c conda-forge c-compiler cxx-compiler make   # gcc/g++ (sudo 불필요)
#
# 컴파일 대상: ../../shared/mechanisms
#   - 시냅스: ProbAMPANMDA_EMS, ProbGABAAB_EMS (BBP EMS, cvode 비호환→고정 dt 0.025)
#   - 결정론: DetAMPANMDA, DetGABAAB · 입력: VecStim
#   - 가소성: GBPlasticity* (Graupner-Brunel, LTP/LTD용)
#   - 이온통로: na3,nax,kdr,kdrb,kad,kap,kmb,kdb,kca,cagk,cal,can,cat,hd,cacum
# 출력: scratch/mechbuild/x86_64/libnrnmech.so (gitignore)
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
MECH="$HERE/../../shared/mechanisms"
OUT="$HERE/scratch/mechbuild"
mkdir -p "$OUT" && cd "$OUT"
nrnivmodl "$MECH"
echo "완료 -> $OUT/x86_64/libnrnmech.so"
# 검증: python3 (cwd=$OUT 에서) from neuron import h; h.ProbAMPANMDA_EMS(...)
