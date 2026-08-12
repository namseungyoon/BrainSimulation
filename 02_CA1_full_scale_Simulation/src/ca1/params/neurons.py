"""Single-source-of-truth loader for per-type AdEx neuron parameters.

Reads ``neuron_parameters.json`` (the authoritative correct_aeif_parameters.json).
Each key in the JSON maps to a CA1 cell type.  We expose all 9 types:
  Pyramidal, PV_Basket, CCK_Basket, Axo, Bistratified, Ivy, O_LM, SCA,
  Neurogliaform.

fit_provenance annotations
--------------------------
* 'placeholder' : a / b / tau_w are Brette & Gerstner (2005) textbook defaults
  and must be replaced before theta-critical claims can be made.
* 'analytic'    : derived from Bezaire Rin / tau_m / rheobase via standard
  AdEx identities (g_L = 1000/Rin, C_m = tau_m * g_L, V_th from rheobase).

O-LM g_L fidelity note
-----------------------
The JSON already carries the corrected value 3.735 nS
(= 1000 / 267.7 MOhm).  A previous runtime path mistakenly used ~0.56 nS
(~6x error); that path is not present here.  The value is loaded verbatim.
"""

from __future__ import annotations

from pathlib import Path

from .json_io import (
    load_json_mapping,
    mapping_field,
    numeric_field,
    reject_malformed_validation_passed,
    required_string_field,
)
from ..types import NeuronParams

# --------------------------------------------------------------------------- #
# Brette-Gerstner 2005 reference defaults used to flag provenance             #
# --------------------------------------------------------------------------- #
# Pyramidal: a=4 nS, b=80.5 pA, tau_w=144 ms are the canonical cortical PyC
# defaults from Table 1 of Brette & Gerstner (2005) J Neurophysiol.
_PLACEHOLDER_THRESHOLD = frozenset({
    ("CA1_Pyramidal", "a", 4.0),
    ("CA1_Pyramidal", "b", 80.5),
    ("CA1_Pyramidal", "tau_w", 144.0),
})

# Internal JSON key → canonical package name
_JSON_TO_NAME: dict[str, str] = {
    "CA1_Pyramidal": "Pyramidal",
    "PV_Basket":     "PV_Basket",
    "CCK_Basket":    "CCK_Basket",
    "Axo":           "Axo",
    "Bistratified":  "Bistratified",
    "Ivy":           "Ivy",
    "O_LM":          "O_LM",
    "SCA":           "SCA",
    "Neurogliaform": "Neurogliaform",
}

# Default V_peak for aeif_cond_beta_multisynapse (spike cut-off, mV)
_V_PEAK_DEFAULT: float = 0.0

# Keys in the JSON that are NOT cell-type blocks
_NON_TYPE_KEYS: frozenset[str] = frozenset({
    "comment", "formulas_used",
    "synapse_receptor_setup", "synapse_weights_from_bezaire",
})


def _is_placeholder(json_key: str, a: float, b: float, tau_w: float) -> bool:
    """Return True if a/b/tau_w match known textbook defaults for this type."""
    if json_key == "CA1_Pyramidal":
        return (
            abs(a - 4.0) < 1e-9
            and abs(b - 80.5) < 1e-9
            and abs(tau_w - 144.0) < 1e-9
        )
    return False


def _load_fitted(
    path: Path,
    analytic: dict[str, NeuronParams],
) -> dict[str, NeuronParams]:
    """Build NeuronParams from neuron_parameters_fitted.json (CMA-ES fits).

    Flat per-cell AdEx params keyed by canonical name; fit_provenance carries the
    validation verdict ('nest-validated' / 'neuron-cmaes' / 'FAILED').
    """
    raw = load_json_mapping(path, context="fitted neuron params")

    expected = set(analytic)
    unknown = set(raw) - expected
    if unknown:
        unknown_list = sorted(unknown)
        unknown_msg = (
            f"unknown cell type {unknown_list[0]!r}"
            if len(unknown_list) == 1
            else f"unknown cell types: {unknown_list}"
        )
        raise ValueError(
            f"fitted neuron params in {path} contain {unknown_msg}"
        )

    result: dict[str, NeuronParams] = {}
    for name, p in raw.items():
        if not isinstance(p, dict):
            raise TypeError(f"invalid fitted neuron params for {name!r} in {path}")
        provenance = required_string_field(
            p,
            "fit_provenance",
            context="fitted neuron params",
        )
        if provenance == "FAILED":
            raise ValueError(
                f"fitted neuron params for {name!r} in {path} are marked FAILED; "
                + "remove the record or regenerate a validated fit"
            )
        reject_malformed_validation_passed(
            p,
            context=f"fitted neuron params for {name!r}",
        )
        result[name] = NeuronParams(
            C_m=numeric_field(p, "C_m", context="fitted neuron params"),
            g_L=numeric_field(p, "g_L", context="fitted neuron params"),
            E_L=numeric_field(p, "E_L", context="fitted neuron params"),
            V_th=numeric_field(p, "V_th", context="fitted neuron params"),
            V_reset=numeric_field(p, "V_reset", context="fitted neuron params"),
            Delta_T=numeric_field(p, "Delta_T", context="fitted neuron params"),
            a=numeric_field(p, "a", context="fitted neuron params"),
            b=numeric_field(p, "b", context="fitted neuron params"),
            tau_w=numeric_field(p, "tau_w", context="fitted neuron params"),
            t_ref=numeric_field(p, "t_ref", context="fitted neuron params"),
            V_peak=numeric_field(
                p,
                "V_peak",
                context="fitted neuron params",
            ),
            I_e=0.0,
            fit_provenance=provenance,
        )
    missing = expected - set(raw)
    if missing:
        raise ValueError(
            f"missing fitted neuron params in {path}: {sorted(missing)}"
        )
    if len(result) != 9:
        raise ValueError(f"fitted params has {len(result)}/9 cell types")
    return result


def load_neuron_params(path: Path | None = None) -> dict[str, NeuronParams]:
    """Load AdEx parameters for all 9 CA1 cell types.

    Parameters
    ----------
    path:
        Optional override for the JSON file.  Defaults to the canonical
        ``neuron_parameters.json`` stored alongside this module.

    Returns
    -------
    dict[str, NeuronParams]
        Keys are canonical type names (e.g. ``"Pyramidal"``, ``"PV_Basket"``).

    Raises
    ------
    ValueError
        If any of the 9 expected cell-type keys is absent from the JSON.
    """
    if path is None:
        path = Path(__file__).parent / "neuron_parameters.json"
        result = load_neuron_params(path)
        fitted = Path(__file__).parent / "neuron_parameters_fitted.json"
        if not fitted.exists():
            raise FileNotFoundError(
                f"missing fitted neuron params: {fitted}. "
                + "Use the analytic source file explicitly when analytic "
                + "parameters are intended."
            )
        return _load_fitted(fitted, result)

    raw = load_json_mapping(path, context="neuron_parameters.json")

    result: dict[str, NeuronParams] = {}

    for json_key, canonical_name in _JSON_TO_NAME.items():
        if json_key not in raw:
            raise ValueError(
                f"neuron_parameters.json is missing expected key '{json_key}'. "
                + f"Present keys: {sorted(raw.keys())}"
            )

        block = mapping_field(raw, json_key, context="neuron_parameters.json")
        p = mapping_field(
            block,
            "aeif_parameters",
            context=f"neuron_parameters.json {json_key}",
        )

        a = numeric_field(p, "a", context=f"neuron_parameters.json {json_key}")
        b = numeric_field(p, "b", context=f"neuron_parameters.json {json_key}")
        tau_w = numeric_field(
            p,
            "tau_w",
            context=f"neuron_parameters.json {json_key}",
        )

        provenance = (
            "placeholder"
            if _is_placeholder(json_key, a, b, tau_w)
            else "analytic"
        )

        result[canonical_name] = NeuronParams(
            C_m      = numeric_field(
                p,
                "C_m",
                context=f"neuron_parameters.json {json_key}",
            ),
            g_L      = numeric_field(
                p,
                "g_L",
                context=f"neuron_parameters.json {json_key}",
            ),   # O-LM: 3.735 nS (correct)
            E_L      = numeric_field(
                p,
                "E_L",
                context=f"neuron_parameters.json {json_key}",
            ),
            V_th     = numeric_field(
                p,
                "V_th",
                context=f"neuron_parameters.json {json_key}",
            ),
            V_reset  = numeric_field(
                p,
                "V_reset",
                context=f"neuron_parameters.json {json_key}",
            ),
            Delta_T  = numeric_field(
                p,
                "Delta_T",
                context=f"neuron_parameters.json {json_key}",
            ),
            a        = a,
            b        = b,
            tau_w    = tau_w,
            t_ref    = numeric_field(
                p,
                "t_ref",
                context=f"neuron_parameters.json {json_key}",
            ),
            V_peak   = _V_PEAK_DEFAULT,
            I_e      = 0.0,
            fit_provenance = provenance,
        )

    if len(result) != 9:
        missing = set(_JSON_TO_NAME.values()) - set(result.keys())
        raise ValueError(f"Loaded {len(result)}/9 cell types; missing: {missing}")

    return result
