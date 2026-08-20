# -*- coding: utf-8 -*-
"""
03_network/3_run/mpi_hello.py  —  MPI 동작 검증 (hello)

각 랭크가 자기 id/nhost를 출력. mechanism 로드도 랭크별 확인.
실행: mpirun -np 6 python 03_network/3_run/mpi_hello.py
"""
import os
from neuron import h

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MECH = os.path.join(ROOT, "scratch", "mechbuild", "x86_64", "libnrnmech.so")

h.nrnmpi_init()
pc = h.ParallelContext()
rank = int(pc.id()); nhost = int(pc.nhost())
h.nrn_load_dll(MECH.replace("\\", "/"))

# 랭크별 순서대로 출력
for r in range(nhost):
    pc.barrier()
    if r == rank:
        print(f"[rank {rank}/{nhost}] mechanism 로드 OK · ProbAMPANMDA_EMS={hasattr(h,'ProbAMPANMDA_EMS')}", flush=True)
pc.barrier()
if rank == 0:
    print(f"\n[검증] 총 {nhost}랭크 정상 · MPI 통신 OK", flush=True)
pc.done()
h.quit()
