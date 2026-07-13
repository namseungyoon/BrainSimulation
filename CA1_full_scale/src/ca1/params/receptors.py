from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import cast

from ..extract.connectivity import canonical_name
from .receptor_ports import (
    PortCompressionStrategy,
    PortKey,
    pair_specific_port_name,
    port_replacements_for_strategy,
    representative_port_for_compartment,
    validate_port_strategy_context,
)
from .receptor_syndata_fields import (
    JsonDict,
    MissingSyndataKineticsError,
    UnsupportedSyndataVariantError,
    component_params,
    entry_compartment,
)
from ..types import ReceptorConfig

_PARAMS_DIR = Path(__file__).parent
_FAST_INH_PRE: frozenset[str] = frozenset({"PV_Basket", "Axo"})
_EXC_PRE: frozenset[str] = frozenset({"Pyramidal", "CA3", "ECIII"})
_ECIII_SOURCE = "ECIII"
_RECEPTOR_ALIASES: dict[str, str] = {
    "GABA_fast": "GABA_A_fast",
    "GABA_slow": "GABA_A_slow",
}

ReceptorModel = tuple[
    ReceptorConfig,
    dict[tuple[str, str], tuple[str, ...]],
    dict[tuple[str, str], tuple[str, ...]],
]
def receptor_class_for_pre(pre: str) -> str:
    if pre == _ECIII_SOURCE:
        return "AMPA_slow"
    if pre in _EXC_PRE:
        return "AMPA_fast"
    if pre in _FAST_INH_PRE:
        return "GABA_A_fast"
    return "GABA_A_slow"


def receptor_prefix(receptor: str) -> str:
    return receptor.split("__", maxsplit=1)[0]


def _canonical_receptor(receptor: str, pre: str) -> str:
    if receptor == "":
        return receptor_class_for_pre(pre)
    return _RECEPTOR_ALIASES.get(receptor, receptor)


def _float_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _dynamic_receptor_name(
    receptor_class: str,
    e_rev: float,
    tau_rise: float,
    tau_decay: float,
    compartment: str | None = None,
) -> str:
    name = (
        f"{receptor_class}__e{_float_token(e_rev)}"
        f"__tr{_float_token(tau_rise)}__td{_float_token(tau_decay)}"
    )
    return f"{name}__{compartment}" if compartment is not None else name


def _register_port(
    ports: dict[str, tuple[float, float, float]],
    receptor_class: str,
    kinetics: tuple[float, float, float],
    replacements: dict[PortKey, PortKey],
    port_strategy: PortCompressionStrategy,
    compartment: str | None = None,
) -> str:
    key = replacements.get(
        (receptor_class, *kinetics),
        (receptor_class, *kinetics),
    )
    final_class, e_rev, tau_rise, tau_decay = representative_port_for_compartment(
        key,
        port_strategy,
        compartment,
    )
    name = _dynamic_receptor_name(
        final_class, e_rev, tau_rise, tau_decay, compartment
    )
    ports[name] = (e_rev, tau_rise, tau_decay)
    return name


def _store_pair_port(
    pair_ports: dict[tuple[str, str], list[str]],
    pair: tuple[str, str],
    name: str,
    *,
    compartment_aware: bool,
) -> None:
    if compartment_aware:
        pair_ports.setdefault(pair, []).append(name)
    else:
        _ = pair_ports.setdefault(pair, [name])


@lru_cache(maxsize=None)
def syndata_receptor_model(
    variant: int,
    compartment_aware: bool = False,
    port_strategy: PortCompressionStrategy = "budget_weighted",
) -> ReceptorModel:
    if variant not in (120, 137):
        raise UnsupportedSyndataVariantError(variant)
    validate_port_strategy_context(port_strategy, compartment_aware)

    path = _PARAMS_DIR / f"syndata_{variant}.json"
    with open(path, "r", encoding="utf-8") as fh:
        data = cast(JsonDict, json.load(fh))

    ports: dict[str, tuple[float, float, float]] = {}
    replacements = port_replacements_for_strategy(
        variant,
        compartment_aware,
        port_strategy,
    )
    pair_port_lists: dict[tuple[str, str], list[str]] = {}
    pair_b_port_lists: dict[tuple[str, str], list[str]] = {}

    for entry in cast(list[JsonDict], data["entries"]):
        post = canonical_name(str(entry["postsynaptic"]))
        pre = canonical_name(str(entry["presynaptic"]))
        params = cast(JsonDict, entry["parameters"])

        primary = component_params(params, suffix="")
        if primary is None:
            primary = component_params(params, suffix="_A")
        if primary is not None:
            receptor_class = _canonical_receptor("", pre)
            name = _register_port(
                ports,
                receptor_class,
                primary,
                replacements,
                port_strategy,
                entry_compartment(entry) if compartment_aware else None,
            )
            name = _pair_specific_name(
                ports,
                name,
                pre=pre,
                post=post,
                port_strategy=port_strategy,
            )
            _store_pair_port(
                pair_port_lists,
                (post, pre),
                name,
                compartment_aware=compartment_aware,
            )

        secondary = component_params(params, suffix="_B")
        if secondary is not None:
            name_b = _register_port(
                ports,
                "GABA_B",
                secondary,
                replacements,
                port_strategy,
                entry_compartment(entry) if compartment_aware else None,
            )
            name_b = _pair_specific_name(
                ports,
                name_b,
                pre=pre,
                post=post,
                port_strategy=port_strategy,
            )
            _store_pair_port(
                pair_b_port_lists,
                (post, pre),
                name_b,
                compartment_aware=compartment_aware,
            )

    used_ports = {
        name
        for values in (*pair_port_lists.values(), *pair_b_port_lists.values())
        for name in values
    }
    names = tuple(sorted(name for name in ports if name in used_ports))
    return (
        ReceptorConfig(
            names=names,
            E_rev=tuple(ports[name][0] for name in names),
            tau_rise=tuple(ports[name][1] for name in names),
            tau_decay=tuple(ports[name][2] for name in names),
        ),
        {key: tuple(value) for key, value in pair_port_lists.items()},
        {key: tuple(value) for key, value in pair_b_port_lists.items()},
    )


def _pair_specific_name(
    ports: dict[str, tuple[float, float, float]],
    name: str,
    *,
    pre: str,
    post: str,
    port_strategy: PortCompressionStrategy,
) -> str:
    remapped = pair_specific_port_name(
        pre=pre,
        post=post,
        name=name,
        strategy=port_strategy,
    )
    if remapped != name:
        ports[remapped] = ports[name]
    return remapped


def load_receptor_config(
    variant: int = 120,
    compartment_aware: bool = False,
    port_strategy: PortCompressionStrategy = "budget_weighted",
) -> ReceptorConfig:
    config, _, _ = syndata_receptor_model(
        variant,
        compartment_aware,
        port_strategy,
    )
    return config


def pair_receptors(
    pre: str,
    post: str,
    variant: int,
    *,
    component: str = "A",
    compartment_aware: bool = False,
    port_strategy: PortCompressionStrategy = "budget_weighted",
) -> tuple[str, ...]:
    _, pair_ports, pair_b_ports = syndata_receptor_model(
        variant,
        compartment_aware,
        port_strategy,
    )
    lookup = pair_b_ports if component == "B" else pair_ports
    try:
        return lookup[(post, pre)]
    except KeyError as exc:
        raise MissingSyndataKineticsError(variant, component, pre, post) from exc


def pair_receptor(
    pre: str,
    post: str,
    variant: int,
    *,
    component: str = "A",
    compartment_aware: bool = False,
    port_strategy: PortCompressionStrategy = "budget_weighted",
) -> str:
    return pair_receptors(
        pre,
        post,
        variant,
        component=component,
        compartment_aware=compartment_aware,
        port_strategy=port_strategy,
    )[0]
