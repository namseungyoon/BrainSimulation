from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias, assert_never

from typing_extensions import override

from ..types import ReceptorConfig
from .receptor_port_tables import (
    COMPARTMENT_AWARE_PORT_REPLACEMENTS,
    FAST_BASKET_BISTRATIFIED_PORT,
    FAST_BASKET_DROPPED_PORT,
    PORT_REPLACEMENTS,
    REPRESENTATIVE_PORTS,
    VARIANT_COMPARTMENT_AWARE_PORT_REPLACEMENTS,
    VARIANT_PORT_REPLACEMENTS,
)
from .receptor_port_utility import utility_weighted_representative_port

BasePortCompressionStrategy: TypeAlias = Literal[
    "budget_weighted",
    "preserve_fast_basket_bistratified",
    "demix_pyramidal_olm_gabaa_slow_distal",
]
PortCompressionStrategy: TypeAlias = Literal[
    "budget_weighted",
    "preserve_fast_basket_bistratified",
    "demix_pyramidal_olm_gabaa_slow_distal",
    "utility_weighted_medoids",
    "exact",
]
PortKey = tuple[str, float, float, float]
_RECEPTOR_PORT_PROVENANCE_KEY: Final = "synapse.receptor_ports"
_RECEPTOR_PORT_SHA_PREFIX: Final = ";sha256="
_EXPECTED_FULL_RECEPTOR_PORT_SHA256_BY_LABEL: Final[Mapping[str, str]] = {
    "syndata120-compartment-aware-20port-budget_weighted": (
        "26774704b306d1bd0461fd7df69491cfacd0e1a2e6385877ece2150c9e05e46c"
    ),
    "syndata120-compartment-aware-39port-exact": (
        "7422b35a56f9c03df7bd1aa728d0a121fa86e630faff5693d59336e1f0efabd6"
    ),
}
PORT_COMPRESSION_STRATEGIES: Final[tuple[PortCompressionStrategy, ...]] = (
    "budget_weighted",
    "preserve_fast_basket_bistratified",
    "demix_pyramidal_olm_gabaa_slow_distal",
    "utility_weighted_medoids",
    "exact",
)

_PYRAMIDAL_OLM_SOURCE_PORT: Final = "GABA_A_slow__em60__tr0p11__td9p7__dend"
_PYRAMIDAL_OLM_DISTAL_PORT: Final = (
    "GABA_A_slow__em60__tr0p11__td9p7__distal__dend"
)
_CCK_OLM_SOMA_PORT: Final = "GABA_A_slow__em60__tr1__td8__soma"
_CCK_OLM_DEND_PORT: Final = "GABA_A_slow__em60__tr1__td8__dend"


@dataclass(frozen=True, slots=True)
class PortStrategyParseError(ValueError):
    raw: str
    choices: tuple[PortCompressionStrategy, ...]

    @override
    def __str__(self) -> str:
        options = ", ".join(repr(strategy) for strategy in self.choices)
        return f"receptor_port_strategy must be one of {options}, got {self.raw!r}"


@dataclass(frozen=True, slots=True)
class PortStrategyContextError(ValueError):
    strategy: PortCompressionStrategy
    message: str

    @override
    def __str__(self) -> str:
        return self.message


def parse_port_strategy(raw: str) -> PortCompressionStrategy:
    if raw in PORT_COMPRESSION_STRATEGIES:
        return raw
    raise PortStrategyParseError(raw, PORT_COMPRESSION_STRATEGIES)


def receptor_port_provenance(
    variant: int,
    compartment_aware: bool,
    strategy: PortCompressionStrategy,
    receptors: ReceptorConfig,
) -> str:
    compartment = "compartment-aware" if compartment_aware else "single-compartment"
    label = f"syndata{variant}-{compartment}-{receptors.n_ports()}port-{strategy}"
    return f"{label}{_RECEPTOR_PORT_SHA_PREFIX}{receptor_port_table_sha256(receptors)}"


def restamp_receptor_port_provenance(
    provenance: str,
    receptors: ReceptorConfig,
) -> str:
    if not provenance:
        return provenance
    label, _, _ = provenance.partition(_RECEPTOR_PORT_SHA_PREFIX)
    return f"{label}{_RECEPTOR_PORT_SHA_PREFIX}{receptor_port_table_sha256(receptors)}"


def receptor_port_table_sha256(receptors: ReceptorConfig) -> str:
    rows = [
        {
            "name": name,
            "e_rev": receptors.E_rev[index],
            "tau_rise": receptors.tau_rise[index],
            "tau_decay": receptors.tau_decay[index],
        }
        for index, name in enumerate(receptors.names)
    ]
    payload = json.dumps(rows, allow_nan=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_port_strategy_context(
    strategy: PortCompressionStrategy,
    compartment_aware: bool,
) -> None:
    if strategy == "utility_weighted_medoids" and not compartment_aware:
        raise PortStrategyContextError(
            strategy,
            "receptor_port_strategy='utility_weighted_medoids' "
            + "requires compartment_aware_synapses=True",
        )


def final_receptor_port_failures(provenance: Mapping[str, str]) -> tuple[str, ...]:
    raw = provenance.get(_RECEPTOR_PORT_PROVENANCE_KEY, "")
    if not raw:
        return (f"{_RECEPTOR_PORT_PROVENANCE_KEY}=missing",)
    label, separator, sha256 = raw.partition(_RECEPTOR_PORT_SHA_PREFIX)
    failures: list[str] = []
    expected_sha256 = _EXPECTED_FULL_RECEPTOR_PORT_SHA256_BY_LABEL.get(label)
    if expected_sha256 is None:
        accepted = ", ".join(
            repr(accepted_label)
            for accepted_label in _EXPECTED_FULL_RECEPTOR_PORT_SHA256_BY_LABEL
        )
        failures.append(
            f"{_RECEPTOR_PORT_PROVENANCE_KEY} strategy must be one of "
            + f"{accepted} for source-location-compatible full-tier receptors, "
            + f"got {label!r}"
        )
    if separator == "":
        failures.append(f"{_RECEPTOR_PORT_PROVENANCE_KEY} missing receptor port sha256")
    elif expected_sha256 is not None and sha256 != expected_sha256:
        failures.append(
            f"{_RECEPTOR_PORT_PROVENANCE_KEY} receptor port sha256 mismatch: "
            + f"expected {expected_sha256}, got {sha256}"
        )
    return tuple(failures)


def port_replacements(
    variant: int,
    compartment_aware: bool,
) -> dict[PortKey, PortKey]:
    replacements = {
        **PORT_REPLACEMENTS,
        **VARIANT_PORT_REPLACEMENTS.get(variant, {}),
    }
    if compartment_aware:
        replacements = {
            **replacements,
            **COMPARTMENT_AWARE_PORT_REPLACEMENTS,
            **VARIANT_COMPARTMENT_AWARE_PORT_REPLACEMENTS.get(variant, {}),
        }
    return replacements


def port_replacements_for_strategy(
    variant: int,
    compartment_aware: bool,
    strategy: PortCompressionStrategy,
) -> dict[PortKey, PortKey]:
    validate_port_strategy_context(strategy, compartment_aware)
    match strategy:
        case "exact" | "utility_weighted_medoids":
            return {}
        case (
            "budget_weighted"
            | "preserve_fast_basket_bistratified"
            | "demix_pyramidal_olm_gabaa_slow_distal"
        ):
            return port_replacements(variant, compartment_aware)
        case unreachable:
            assert_never(unreachable)


def representative_port(
    key: PortKey,
    strategy: PortCompressionStrategy,
) -> PortKey:
    match strategy:
        case "exact":
            return key
        case "utility_weighted_medoids":
            raise PortStrategyContextError(
                strategy,
                "utility_weighted_medoids requires compartment-aware port selection",
            )
        case (
            "budget_weighted"
            | "preserve_fast_basket_bistratified"
            | "demix_pyramidal_olm_gabaa_slow_distal"
        ):
            pass
        case unreachable:
            assert_never(unreachable)
    representatives = _representative_ports(strategy)
    if key in representatives:
        return key
    candidates = [
        candidate for candidate in representatives
        if candidate[0] == key[0] and candidate[1] == key[1]
    ]
    if not candidates:
        return key
    return min(candidates, key=lambda candidate: _tau_distance(key, candidate))


def representative_port_for_compartment(
    key: PortKey,
    strategy: PortCompressionStrategy,
    compartment: str | None,
) -> PortKey:
    match strategy:
        case "exact":
            return key
        case "utility_weighted_medoids":
            return utility_weighted_representative_port(key, compartment)
        case (
            "budget_weighted"
            | "preserve_fast_basket_bistratified"
            | "demix_pyramidal_olm_gabaa_slow_distal"
        ):
            return representative_port(key, strategy)
        case unreachable:
            assert_never(unreachable)


def pair_specific_port_name(
    *,
    pre: str,
    post: str,
    name: str,
    strategy: PortCompressionStrategy,
) -> str:
    if strategy != "demix_pyramidal_olm_gabaa_slow_distal":
        return name
    if (
        pre == "O_LM"
        and post == "Pyramidal"
        and name == _PYRAMIDAL_OLM_SOURCE_PORT
    ):
        return _PYRAMIDAL_OLM_DISTAL_PORT
    if pre == "CCK_Basket" and post == "O_LM" and name == _CCK_OLM_SOMA_PORT:
        return _CCK_OLM_DEND_PORT
    return name


def _representative_ports(strategy: BasePortCompressionStrategy) -> frozenset[PortKey]:
    if strategy != "preserve_fast_basket_bistratified":
        return REPRESENTATIVE_PORTS
    return frozenset(
        (REPRESENTATIVE_PORTS | {FAST_BASKET_BISTRATIFIED_PORT})
        - {FAST_BASKET_DROPPED_PORT}
    )


def _tau_distance(left: PortKey, right: PortKey) -> float:
    return (
        abs(math.log(left[2]) - math.log(right[2]))
        + abs(math.log(left[3]) - math.log(right[3]))
    )
