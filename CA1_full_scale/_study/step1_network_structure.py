"""STAGE 1 / step 1 — extract Bezaire network STRUCTURE from the code and compare to the paper.
Read-only extraction (no simulation). Two sources:
  (A) connectivity.json  (conndata_101, diagnostic compatibility data)
  (B) build_network_spec on the deployed config (conndata_430 = final-tier) if importable.
Prints per-type cell counts, total cells, per-projection in-degree, and total synapses,
against Bezaire 2016 (338,740 cells; 5.19 B synapses)."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER_CELLS = 338_740
PAPER_SYN = 5.19e9

# ---- (A) connectivity.json (conndata_101) ----
cj = json.loads((ROOT / "src/ca1/params/connectivity.json").read_text(encoding="utf-8"))
pops = cj["populations_used"]
ca1_types = {k: v for k, v in pops.items() if k not in ("ca3cell", "eccell")}
total_cells = sum(ca1_types.values())

print("=== (A) connectivity.json = conndata_101 (diagnostic) ===")
print(f"CA1 cell total: {total_cells:,}  (paper 338,740; match={total_cells==PAPER_CELLS})")
print("per-type:", {k: v for k, v in sorted(ca1_types.items(), key=lambda x: -x[1])})

def _sum_syn(section: dict) -> tuple[int, int]:
    n_proj, n_syn = 0, 0
    for name, rec in section.items():
        if not isinstance(rec, dict) or "total_connections_in_network" not in rec:
            continue
        n_proj += 1
        spc = int(rec.get("synapses_per_connection", 1))
        n_syn += int(rec["total_connections_in_network"]) * spc
    return n_proj, n_syn

exc_n, exc_syn = _sum_syn(cj.get("excitatory_connections", {}))
inh_n, inh_syn = _sum_syn(cj.get("inhibitory_connections", {}))
aff_n, aff_syn = _sum_syn(cj.get("afferents", {}))
rec_n, rec_syn = exc_n + inh_n, exc_syn + inh_syn
total_syn = rec_syn + aff_syn
print(f"recurrent excitatory: {exc_n} proj, {exc_syn:,} syn")
print(f"recurrent inhibitory: {inh_n} proj, {inh_syn:,} syn")
print(f"afferent (CA3+ECIII): {aff_n} proj, {aff_syn:,} syn")
print(f"TOTAL synapses: {total_syn:,}  ({total_syn/1e9:.2f} B; paper ~5.19 B)")
if "statistics" in cj:
    print("statistics:", cj["statistics"])

# ---- (B) build_network_spec on deployed config (conndata_430) ----
print("\n=== (B) build_network_spec (conndata_430, final-tier) ===")
try:
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    import yaml
    from ca1.config import build_network_spec
    cfg = yaml.safe_load((ROOT / "configs/full_scale_3dtopo.yaml").read_text())
    spec = build_network_spec(cfg)
    nc = dict(getattr(spec, "n_cells_per_type", {}) or {})
    print("spec built OK. n_cells_per_type:", nc, " total:", (sum(nc.values()) if nc else "?"))
    projs = list(getattr(spec, "projections", []) or [])
    affs = list(getattr(spec, "afferents", []) or [])
    print(f"recurrent projections: {len(projs)}, afferents: {len(affs)}")
    if projs:
        p0 = projs[0]
        print("projection attrs sample:", [a for a in dir(p0) if not a.startswith('_')][:20])
except Exception as e:
    print(f"build_network_spec unavailable/failed: {type(e).__name__}: {e}")
print("STEP1 STRUCTURE DONE")
