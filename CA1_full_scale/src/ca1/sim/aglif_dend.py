from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from functools import lru_cache
import json
import math
import os
from pathlib import Path
from typing import cast

from ca1.params.aglif import aglif_params_for_cell_type
from ca1.params.dendritic_transfer import dendritic_transfer_for_cell_type

_PARAMS_DIR = Path(__file__).resolve().parents[1] / "params"
_SOURCE_LOCATION_TRANSFER = (
    _PARAMS_DIR / "source_location_transfer_syndata120_budget_weighted.json"
)
_SOMA_DOMAIN = 0.0
_PROX_DOMAIN = 1.0
_DIST_DOMAIN = 2.0
_DIST_C_FRAC = 0.5
_DIST_LEAK_SCALE = 1.0
_DIST_COUPLING_SCALE = 0.25

# Candidate-only Option-2 fit. V_h_half/k_h are regularized to the least-squares
# logistic reduction (-41.9515 mV, 6.9790 mV) of checked-in ch_Navcck hinf;
# all five values are fitted only to source CCK current- and conductance-f-I.
_CCK_USER_M3_H_PARAMS = {
    "V_h_half": -42.0,
    "k_h": 7.0,
    "tau_h": 66.97,
    "delta_h": 0.225,
    "h_crit": 0.35,
}

_CCK_USER_M3_SOURCE_REFIT = {
    "C_m": 161.93127473229896,
    "tau_m": 49.911433215383035,
    "V_th": -58.30251936217281,
    "A2": 108.87586824643941,
    "A1": 119.29956314076246,
    "k_adap": 0.04190609992514656,
    "t_ref": 7.067624206335223,
}

# Candidate-only active-dendrite reductions. Availability/K voltage dependence
# and kinetics are least-squares/logistic reductions of the checked-in MOD files
# at 34 C. Effective activation and conductances are bounded by the native
# density x dendritic-area totals, then constrained by the existing intrinsic
# identity gate. They are deliberately not fitted to Table-5/network rates.
_USER_M4_DENDRITIC_PARAMS = {
    "PV_Basket": {
        "gbar_Na_prox": 500.0, "gbar_Na_dist": 500.0,
        "E_Na": 55.0, "Vm_half": -30.0, "km": 2.5,
        "Vh_half": -35.0, "kh": 4.0, "tau_h": 8.711,
        "gbar_Kd_prox": 500.0, "gbar_Kd_dist": 500.0,
        "E_K": -90.0, "Vn_half": -26.5556, "kn": 7.9568,
        "tau_n": 2.744,
    },
    "Bistratified": {
        "gbar_Na_prox": 300.0, "gbar_Na_dist": 300.0,
        "E_Na": 55.0, "Vm_half": -30.0, "km": 2.5,
        "Vh_half": -41.0472, "kh": 6.9279, "tau_h": 6.154,
        "gbar_Kd_prox": 100.0, "gbar_Kd_dist": 100.0,
        "E_K": -90.0, "Vn_half": -26.5556, "kn": 7.9568,
        "tau_n": 2.744,
    },
    "O_LM": {
        "gbar_Na_prox": 50.0, "gbar_Na_dist": 50.0,
        "E_Na": 55.0, "Vm_half": -35.0, "km": 2.5,
        "Vh_half": -47.5538, "kh": 6.8537, "tau_h": 5.175,
        "gbar_Kd_prox": 300.0, "gbar_Kd_dist": 300.0,
        "E_K": -90.0, "Vn_half": -26.5556, "kn": 7.9568,
        "tau_n": 2.744,
    },
}

# Candidate-only branch-local active-dendrite reduction.  Channel kinetics and
# reversals retain the native ch_Navaxonp/ch_Navbis/ch_Nav + ch_Kdrfast fits
# used by user_m4.  C_b, branch leak and axial coupling represent one source
# dendritic branch per reduced domain and are fitted to native single/clustered
# dendritic EPSP responses, never to Table-5 rates.
_USER_M5_BRANCH_PARAMS = {
    "PV_Basket": {
        "C_b_prox": 5.0, "C_b_dist": 12.0,
        "g_leak_b_prox": 5.0, "g_leak_b_dist": 0.5656,
        "g_b_prox": 10.0, "g_b_dist": 214.20,
        "gbar_Na_prox": 500.0, "gbar_Na_dist": 471.2386,
        "gbar_Kd_prox": 75.0, "gbar_Kd_dist": 40.8407,
    },
    "Bistratified": {
        "C_b_prox": 2.9322, "C_b_dist": 50.0,
        "g_leak_b_prox": 0.1885, "g_leak_b_dist": 0.2828,
        "g_b_prox": 95.25, "g_b_dist": 214.20,
        "gbar_Na_prox": 146.6076, "gbar_Na_dist": 219.9113,
        "gbar_Kd_prox": 33.5103, "gbar_Kd_dist": 50.2654,
    },
    "O_LM": {
        "C_b_prox": 10.2231, "C_b_dist": 10.2231,
        "g_leak_b_prox": 0.0786, "g_leak_b_dist": 0.0786,
        "g_b_prox": 56.48, "g_b_dist": 56.48,
        "gbar_Na_prox": 184.0153, "gbar_Na_dist": 184.0153,
        "gbar_Kd_prox": 831.9251, "gbar_Kd_dist": 831.9251,
    },
}

# PV-only source morphology reduction for user_m7.  Lanes 0/1 are the two
# five-section apical roots; lanes 2/3 are the two three-section basal roots.
# C/leak/Na/K are direct area sums from the native template.  Axial values are
# the geometry/Ra initial reduction and are frozen before the payoff replay.
_USER_M7_PV_LANES = {
    "C_b_prox": (30.788, 30.788, 15.394, 15.394),
    "C_b_dist": (28.589, 28.589, 4.398, 4.398),
    "g_leak_b_prox": (3.959, 3.959, 1.979, 1.979),
    "g_leak_b_dist": (3.676, 3.676, 0.566, 0.566),
    "g_ax_b_prox": (100.0, 100.0, 30.0, 30.0),
    "g_ax_b_dist": (20.0, 20.0, 8.0, 8.0),
    "gbar_Na_prox": (3298.670, 3298.670, 1649.335, 1649.335),
    "gbar_Na_dist": (3063.056, 3063.056, 471.239, 471.239),
    "gbar_Kd_prox": (285.885, 285.885, 142.942, 142.942),
    "gbar_Kd_dist": (265.465, 265.465, 40.841, 40.841),
}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {value!r}") from exc


def _cell_env_suffix(cell_type: str) -> str:
    return cell_type.upper().replace("-", "_").replace(" ", "_")


def _exc_soma_cell_types() -> set[str]:
    raw = os.environ.get("CA1_AGLIF_DEND_EXC_SOMA_CELL_TYPES", "")
    return {item.strip() for item in raw.split(",") if item.strip()}


def _force_exc_soma(
    cell_type: str | None,
    receive_domain_overrides: Mapping[str, str] | None,
    receive_domain: str | None,
) -> bool:
    if receive_domain == "soma_excitatory":
        return True
    if receive_domain is not None:
        raise ValueError(
            f"unsupported AGLIF receive_domain for {cell_type!r}: "
            f"{receive_domain!r}"
        )
    if cell_type is not None and receive_domain_overrides is not None:
        mode = receive_domain_overrides.get(cell_type)
        if mode == "soma_excitatory":
            return True
        if mode is not None:
            raise ValueError(
                f"unsupported AGLIF receive-domain mode for {cell_type!r}: {mode!r}"
            )
    if os.environ.get("CA1_AGLIF_DEND_EXC_SOMA") == "1":
        return True
    if cell_type is None:
        return False
    return cell_type in _exc_soma_cell_types()


def _diagnostic_gc_scale(cell_type: str) -> float:
    global_scale = _env_float("CA1_AGLIF_DEND_GC_SCALE", 1.0)
    return _env_float(
        f"CA1_AGLIF_DEND_GC_SCALE_{_cell_env_suffix(cell_type)}",
        global_scale,
    )


def _row_domain(row: dict[str, object]) -> float:
    port = str(row["port"])
    if port.endswith("__soma"):
        return _SOMA_DOMAIN

    aglif_compartment = str(row.get("aglif_compartment", ""))
    if aglif_compartment == "soma":
        return _SOMA_DOMAIN

    loc = str(row.get("loc", "")).lower()
    if "dist" in loc or "apical" in loc:
        return _DIST_DOMAIN
    if "prox" in loc or "near_soma" in loc or "soma" in loc or "wide" in loc:
        return _PROX_DOMAIN
    raise ValueError(f"source-location row has unknown loc/domain: {row!r}")


def _source_location_table_path(table: str | Path | None = None) -> Path:
    if table is None or str(table) == "":
        return _SOURCE_LOCATION_TRANSFER
    return Path(table)


def _source_location_rows(table: str | Path | None = None) -> list[dict[str, object]]:
    path = _source_location_table_path(table)
    return _source_location_rows_cached(str(path))


def _source_location_row(raw_row: object) -> dict[str, object]:
    if not isinstance(raw_row, dict):
        raise TypeError(f"invalid source-location row: {raw_row!r}")
    row = cast(dict[object, object], raw_row)
    parsed: dict[str, object] = {}
    for key, value in row.items():
        parsed[str(key)] = value
    return parsed


@lru_cache(maxsize=None)
def _source_location_rows_cached(table: str) -> list[dict[str, object]]:
    path = Path(table)
    loaded = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(loaded, list):
        raise TypeError(f"{path} must contain a list")
    rows = cast(list[object], loaded)
    return [_source_location_row(raw_row) for raw_row in rows]


@lru_cache(maxsize=None)
def _source_location_domain_map(table: str = "") -> dict[tuple[str, str], float]:
    domains: dict[tuple[str, str], set[float]] = defaultdict(set)
    for row in _source_location_rows(table):
        domains[(str(row["post"]), str(row["port"]))].add(_row_domain(row))

    collapsed: dict[tuple[str, str], float] = {}
    for key, values in domains.items():
        non_soma = values - {_SOMA_DOMAIN}
        if not non_soma:
            collapsed[key] = _SOMA_DOMAIN
        elif _DIST_DOMAIN in non_soma and _PROX_DOMAIN not in non_soma:
            collapsed[key] = _DIST_DOMAIN
        else:
            collapsed[key] = _PROX_DOMAIN
    return collapsed


@lru_cache(maxsize=None)
def aglif_dend_mixed_domain_ports(
    source_location_transfer_table: str = "",
) -> tuple[str, ...]:
    domains: dict[tuple[str, str], set[float]] = defaultdict(set)
    for row in _source_location_rows(source_location_transfer_table):
        domains[(str(row["post"]), str(row["port"]))].add(_row_domain(row))

    mixed = [
        f"{post}:{port}"
        for (post, port), values in domains.items()
        if {_PROX_DOMAIN, _DIST_DOMAIN}.issubset(values)
    ]
    return tuple(sorted(mixed))


@lru_cache(maxsize=None)
def aglif_dend_shared_port_resolutions(
    source_location_transfer_table: str = "",
) -> tuple[str, ...]:
    resolutions: list[str] = []
    for row in _source_location_rows(source_location_transfer_table):
        loc_original = str(row.get("loc_original", row.get("loc", "")))
        loc = str(row.get("loc", ""))
        if loc_original == loc:
            continue
        resolution = str(
            row.get("shared_port_domain_resolution", "unspecified")
        )
        resolved = (
            f"{row['pre']}->{row['post']}:{row['port']}:"
            + f"{loc_original}->{loc}:{resolution}"
        )
        resolutions.append(resolved)
    return tuple(sorted(resolutions))


def aglif_dend_status(cell_type: str, gc_scale: float = 1.0) -> dict[str, float]:
    if gc_scale <= 0.0:
        raise ValueError(f"gc_scale must be positive, got {gc_scale!r}")
    params = aglif_params_for_cell_type(cell_type)
    transfer = dendritic_transfer_for_cell_type(cell_type)
    status = params.as_nest()
    membrane_conductance = params.C_m / params.tau_m
    status["dend_C_frac"] = transfer.dend_C_frac
    status["dend_leak_scale"] = transfer.dend_leak_scale
    status["g_c"] = (
        transfer.g_c_nS(membrane_conductance)
        * gc_scale
        * _diagnostic_gc_scale(cell_type)
    )
    status["dist_C_frac"] = _DIST_C_FRAC
    status["dist_leak_scale"] = _DIST_LEAK_SCALE
    status["g_c_dist"] = status["g_c"] * _DIST_COUPLING_SCALE
    return status


def cck_user_m3_status(e_l_mV: float | None = None) -> dict[str, float]:
    """Return the isolated sodium-availability fit for the CCK user_m3 model."""
    status = dict(_CCK_USER_M3_SOURCE_REFIT)
    status.update(_CCK_USER_M3_H_PARAMS)
    if e_l_mV is not None:
        exponent = (e_l_mV - status["V_h_half"]) / status["k_h"]
        status["h"] = 1.0 / (1.0 + math.exp(exponent))
    return status


def user_m4_status(cell_type: str, e_l_mV: float | None = None) -> dict[str, float]:
    """Return source-channel-fitted active dendrite status for ``user_m4``."""
    try:
        status = dict(_USER_M4_DENDRITIC_PARAMS[cell_type])
    except KeyError as exc:
        raise ValueError(f"user_m4 has no source fit for {cell_type!r}") from exc
    if e_l_mV is not None:
        h = 1.0 / (1.0 + math.exp((e_l_mV-status["Vh_half"])/status["kh"]))
        n = 1.0 / (1.0 + math.exp(-(e_l_mV-status["Vn_half"])/status["kn"]))
        status.update({"h_Na_prox": h, "h_Na_dist": h,
                       "n_Kd_prox": n, "n_Kd_dist": n})
    return status


def user_m5_status(cell_type: str, e_l_mV: float | None = None) -> dict[str, float]:
    """Return source-template-fitted private-branch status for ``user_m5``."""
    try:
        status = dict(_USER_M4_DENDRITIC_PARAMS[cell_type])
        status.update(_USER_M5_BRANCH_PARAMS[cell_type])
    except KeyError as exc:
        raise ValueError(f"user_m5 has no source fit for {cell_type!r}") from exc
    if e_l_mV is not None:
        h = 1.0 / (1.0 + math.exp((e_l_mV-status["Vh_half"])/status["kh"]))
        n = 1.0 / (1.0 + math.exp(-(e_l_mV-status["Vn_half"])/status["kn"]))
        status.update({"V_b_prox": e_l_mV, "V_b_dist": e_l_mV,
                       "h_Na_prox": h, "h_Na_dist": h,
                       "n_Kd_prox": n, "n_Kd_dist": n})
    return status


def user_m7_status(cell_type: str, e_l_mV: float | None = None) -> dict[str, float]:
    """Return the frozen native-PV heterogeneous-lane status for ``user_m7``."""
    if cell_type != "PV_Basket":
        raise ValueError(f"user_m7 has no source morphology for {cell_type!r}")
    status = {
        key: value for key, value in _USER_M4_DENDRITIC_PARAMS[cell_type].items()
        if key not in {"gbar_Na_prox", "gbar_Na_dist", "gbar_Kd_prox", "gbar_Kd_dist"}
    }
    for stem, values in _USER_M7_PV_LANES.items():
        for lane, value in enumerate(values):
            status[f"{stem}_{lane}"] = value
    if e_l_mV is not None:
        h = 1.0 / (1.0 + math.exp((e_l_mV-status["Vh_half"])/status["kh"]))
        n = 1.0 / (1.0 + math.exp(-(e_l_mV-status["Vn_half"])/status["kn"]))
        for region in ("prox", "dist"):
            for lane in range(4):
                status[f"V_b_{region}_{lane}"] = e_l_mV
                status[f"h_Na_{region}_{lane}"] = h
                status[f"n_Kd_{region}_{lane}"] = n
    return status


def aglif_dend_compartments(
    receptor_names: tuple[str, ...],
    cell_type: str | None = None,
    required_dendritic_ports: frozenset[str] | None = None,
    source_location_transfer_table: str = "",
    receive_domain_overrides: Mapping[str, str] | None = None,
    compartment_overrides: Mapping[str, Mapping[str, float]] | None = None,
    *,
    receive_domain: str | None = None,
) -> list[float]:
    compartments: list[float] = []
    force_exc_soma = _force_exc_soma(
        cell_type,
        receive_domain_overrides,
        receive_domain,
    )
    for name in receptor_names:
        configured_domain = (
            None
            if cell_type is None or compartment_overrides is None
            else compartment_overrides.get(cell_type, {}).get(name)
        )
        if configured_domain is not None:
            domain = float(configured_domain)
            if domain not in {_SOMA_DOMAIN, _PROX_DOMAIN, _DIST_DOMAIN}:
                raise ValueError(
                    f"invalid configured compartment for {cell_type}:{name}: {domain}"
                )
            compartments.append(domain)
            continue
        if force_exc_soma and name.startswith(("AMPA_fast", "AMPA_slow")):
            compartments.append(0.0)
            continue
        if name.endswith("__soma"):
            compartments.append(0.0)
            continue
        if name.endswith("__dend"):
            if cell_type is None:
                message = (
                    f"cell_type is required to resolve dendritic receptor {name!r};"
                    " refusing proximal fallback"
                )
                raise ValueError(message)
            if required_dendritic_ports is None:
                message = (
                    "required_dendritic_ports is required to resolve dendritic "
                    f"receptor {cell_type}:{name}; refusing proximal fallback"
                )
                raise ValueError(message)
            domain = _source_location_domain_map(source_location_transfer_table).get(
                (cell_type, name)
            )
            if domain is None:
                if name in required_dendritic_ports:
                    message = (
                        f"missing source-location domain for {cell_type}:{name};"
                        " refusing proximal fallback"
                    )
                    raise ValueError(message)
                compartments.append(_PROX_DOMAIN)
                continue
            compartments.append(domain)
            continue
        if name.startswith("GABA_A_fast"):
            compartments.append(_SOMA_DOMAIN)
        else:
            compartments.append(_PROX_DOMAIN)
    return compartments
