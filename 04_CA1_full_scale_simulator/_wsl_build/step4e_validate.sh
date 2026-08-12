#!/usr/bin/env bash
# Step 4e — validate the completed full-scale result + summarize rates/spectral.
# Pure analysis (no GPU/nestgpu needed).
set -uo pipefail

ROOT="$HOME/ca1_full_scale"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
RESULT="results/fullscale_3dtopo_baseline.h5"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

echo "== result file =="
ls -lh "$RESULT"

echo "== integrity / keys =="
"$PY" - <<'PY'
import h5py
f = h5py.File("results/fullscale_3dtopo_baseline.h5","r")
print("top-level keys:", list(f.keys()))
meta = dict(f["meta"].attrs) if "meta" in f else {}
for k in ("duration_s","scale","seed","backend","name","elapsed_s","git_sha"):
    if k in meta: print(f"  meta.{k} = {meta[k]}")
if "lfp" in f: print("  lfp samples:", f["lfp"].shape)
f.close()
PY

echo "== per-type mean firing rates =="
"$PY" - <<'PY'
import h5py
f = h5py.File("results/fullscale_3dtopo_baseline.h5","r")
dur = float(dict(f["meta"].attrs).get("duration_s", 10.0))
if "spikes" in f:
    tot_all=0; n_all=0
    for t in sorted(f["spikes"]):
        grp=f["spikes"][t]; n=len(grp); tot=sum(len(grp[c]) for c in grp)
        tot_all+=tot; n_all+=n
        print(f"  {t:16s} N={n:7d}  spikes={tot:10d}  rate={tot/max(n,1)/dur:8.3f} Hz")
    print(f"  {'TOTAL':16s} N={n_all:7d}  spikes={tot_all:10d}")
f.close()
PY

echo "== VALIDATE (tier full) =="
"$PY" -m ca1.cli validate "$RESULT" --tier full 2>&1 | tail -80

echo "== STEP4E DONE =="
