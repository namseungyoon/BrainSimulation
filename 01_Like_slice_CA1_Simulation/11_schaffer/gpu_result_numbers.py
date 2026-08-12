# -*- coding: utf-8 -*-
"""전슬라이스 GPU 결정론 풀런(sc_det_gpu/fullscale_n4) 다관점 발화율 숫자 요약(그림 없음).
slice_cells.npz(층/mtype/EI)와 SC_spikes_all.csv 를 gid(=인덱스)로 조인."""
import os, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
CELLS = os.path.join(ROOT, "05_placement", "slice_cells.npz")
SPK = os.path.join(HERE, "sc_det_gpu", "fullscale_n4", "SC_spikes_all.csv")
TSTOP = 1000.0
dur = TSTOP / 1000.0

c = np.load(CELLS, allow_pickle=True)
mtype = c["mtype"].astype(str); layer = c["layer"].astype(str); sclass = c["sclass"].astype(str)
N = len(mtype)

gids = []
with open(SPK, encoding="utf-8") as f:
    rd = csv.reader(f); next(rd, None)
    for row in rd:
        gids.append(int(row[0]))
gids = np.array(gids, int)
nspk = np.bincount(gids, minlength=N)

is_exc = sclass == "EXC"
n_exc = int(is_exc.sum()); n_inh = N - n_exc
pc = nspk[is_exc].sum() / n_exc / dur
it = nspk[~is_exc].sum() / max(1, n_inh) / dur
fired = int((nspk > 0).sum())
print(f"[전체] {N}세포(EXC {n_exc}/INH {n_inh}) · 스파이크 {int(nspk.sum()):,} · 발화 {fired}/{N}({100*fired/N:.0f}%) · PC {pc:.2f}Hz · INT {it:.2f}Hz")

order = [l for l in ["SO", "SP", "SR", "SLM"] if (layer == l).any()]
print("[층별] " + " · ".join(f"{L} {nspk[layer==L].sum()/max(1,(layer==L).sum())/dur:.2f}Hz(n={int((layer==L).sum())})" for L in order))

mts = sorted(set(mtype), key=lambda m: -nspk[mtype == m].sum()/max(1,(mtype==m).sum()))
print("[m-type] " + " · ".join(f"{m} {nspk[mtype==m].sum()/max(1,(mtype==m).sum())/dur:.2f}Hz(n={int((mtype==m).sum())})" for m in mts))
