"""Deterministic routing and CPU primitives for the PV user_m7 morphology."""
from __future__ import annotations

import hashlib
from typing import Final, Iterable

ROUTE_SEED: Final = 0x50564D4F52504831
GOLDEN: Final = 0x9E3779B97F4A7C15
MASK64: Final = (1 << 64) - 1

# Exact native eligible-site lane order after reducing each stable HOC object
# list.  Repeated lane IDs preserve site multiplicity (not electrical area).
PV_LANE_SITES: Final = {
    "dend_50_200": (0,)*4 + (1,)*4 + (2,)*5 + (3,)*5,
    "dend_lt_50": (0, 1),
    "apical_gt_100": (0,)*12 + (1,)*12,
    "apical_gt_200": (0,)*9 + (1,)*9,
    "dend_gt_200": (0,)*9 + (1,)*9 + (2,)*3 + (3,)*3,
}
PV_ROUTE_TABLE_SHA256: Final = hashlib.sha256(
    repr(tuple((k, PV_LANE_SITES[k]) for k in sorted(PV_LANE_SITES))).encode()
).hexdigest()


def splitmix64(value: int) -> int:
    z = (value + GOLDEN) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return (z ^ (z >> 31)) & MASK64


def semantic_port_hash(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "little")


def route_contacts(source_gid: int, target_gid: int, semantic_port: str,
                   contacts: int, sites: Iterable[int]) -> tuple[int, int, int, int]:
    """Return exact per-contact lane counts using multiply-high reduction."""
    ordered = tuple(sites)
    counts = [0, 0, 0, 0]
    port_hash = semantic_port_hash(semantic_port)
    rotated_target = ((target_gid << 21) | (target_gid >> 43)) & MASK64
    for contact in range(contacts):
        u64 = splitmix64(ROUTE_SEED ^ source_gid ^ rotated_target ^ port_hash
                         ^ ((contact * GOLDEN) & MASK64))
        site_index = (u64 * len(ordered)) >> 64
        counts[ordered[site_index]] += 1
    return tuple(counts)  # type: ignore[return-value]


def branch_derivatives_per_ms(vb: float, domain_v: float, synaptic_current_pA: float,
                              h: float, n: float, params: dict[str, float],
                              region: str, lane: int) -> tuple[float, float]:
    """One lane's voltage derivative and signed domain-to-lane axial current."""
    z = max(-80.0, min(80.0, -(vb-params["Vm_half"])/params["km"]))
    m = 1.0/(1.0 + __import__("math").exp(z))
    ina = params[f"gbar_Na_{region}_{lane}"]*m**3*h*(params["E_Na"]-vb)
    ik = params[f"gbar_Kd_{region}_{lane}"]*n**4*(params["E_K"]-vb)
    axial = params[f"g_ax_b_{region}_{lane}"]*(domain_v-vb)
    rhs = (-params[f"g_leak_b_{region}_{lane}"]*(vb-params.get("E_L", -65.0))
           + axial + synaptic_current_pA + ina + ik)
    return rhs/params[f"C_b_{region}_{lane}"], axial
