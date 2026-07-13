from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, cast

from .receptors import (
    load_receptor_config as _load_receptor_config,
    pair_receptor,
    pair_receptors,
)
from .receptor_validation import assert_declared_receptor_matches
from .receptor_ports import PortCompressionStrategy
from ..types import (
    DEFAULT_CONNECTION_DELAY_MS,
    Afferent,
    ConndataCountMode,
    Projection,
    ReceptorConfig,
)

# --------------------------------------------------------------------------- #
# Internal constants                                                           #
# --------------------------------------------------------------------------- #

_PARAMS_DIR = Path(__file__).parent

# Neurogliaform provides co-released GABA_A_slow + GABA_B
_NGF_PRE = "Neurogliaform"

# Afferent source names in connectivity.json
JsonDict = dict[str, object]
_CONNECTIVITY_AFFERENT_POP_KEYS: dict[str, str] = {
    "CA3": "ca3cell",
    "ECIII": "eccell",
}


class _ExtractConnectivity(Protocol):
    def __call__(
        self,
        modeldb_dir: Path | str | None = None,
        index: int = 101,
        cellnumbers_index: int | None = None,
        out_path: Path | str | None = None,
        count_mode: ConndataCountMode = "network_total",
    ) -> JsonDict: ...


def load_receptor_config(
    variant: int = 120,
    compartment_aware: bool = False,
    port_strategy: PortCompressionStrategy = "budget_weighted",
) -> ReceptorConfig:
    return _load_receptor_config(variant, compartment_aware, port_strategy)


def _split_weight(weight: float, receptors: tuple[str, ...]) -> float:
    if not receptors:
        raise ValueError("cannot split a synapse over zero receptors")
    return weight / len(receptors)


def _split_indegree(indegree: float, receptors: tuple[str, ...]) -> float:
    if not receptors:
        raise ValueError("cannot split a projection over zero receptors")
    return indegree / len(receptors)


def _as_float(value: object, field: str) -> float:
    if isinstance(value, str | int | float):
        return float(value)
    raise TypeError(f"connectivity field {field!r} must be numeric, got {value!r}")


def _as_int(value: object, field: str) -> int:
    if isinstance(value, str | int | float):
        return int(value)
    raise TypeError(f"connectivity field {field!r} must be integral, got {value!r}")


def _json_afferent_source_count(pre: str, populations: dict[str, int]) -> int:
    pop_key = _CONNECTIVITY_AFFERENT_POP_KEYS.get(pre)
    if pop_key is None:
        raise ValueError(f"unknown afferent source {pre!r}; expected CA3 or ECIII")
    if pop_key not in populations:
        raise ValueError(
            f"population count for afferent source {pre!r} missing as {pop_key!r}"
        )
    return int(populations[pop_key])


def _modeldb_afferent_source_count(pre: str, populations: dict[str, int]) -> int:
    if pre not in populations:
        raise ValueError(f"population count for afferent source {pre!r} missing")
    return int(populations[pre])


def _parse_projection(
    proj_data: JsonDict,
    synapse_variant: int,
    *,
    compartment_aware: bool = False,
    port_strategy: PortCompressionStrategy = "budget_weighted",
) -> list[Projection]:
    """Convert a connectivity.json projection block into Projection objects.

    Neurogliaform projections yield TWO entries: one GABA_A_slow and one
    GABA_B, so that both components of the NGF co-release are represented.
    The GABA_B weight follows ExpGABAab B-component scaling; indegree is shared.
    """
    pre = str(proj_data["presynaptic"])
    post = str(proj_data["postsynaptic"])
    indegree_key = "indegree_true" if "indegree_true" in proj_data else "indegree"
    indegree = _as_float(proj_data[indegree_key], indegree_key)
    spc = _as_int(proj_data["synapses_per_connection"], "synapses_per_connection")
    weight = _as_float(proj_data["weight_nS"], "weight_nS")
    delay = DEFAULT_CONNECTION_DELAY_MS

    primary_receptors = pair_receptors(
        pre,
        post,
        synapse_variant,
        compartment_aware=compartment_aware,
        port_strategy=port_strategy,
    )
    assert_declared_receptor_matches(
        proj_data,
        pre=pre,
        post=post,
        derived_receptors=primary_receptors,
    )

    projections: list[Projection] = []
    for primary_receptor in primary_receptors:
        projections.append(Projection(
            pre                    = pre,
            post                   = post,
            indegree               = _split_indegree(indegree, primary_receptors),
            synapses_per_connection = spc,
            weight_nS              = weight,
            receptor               = primary_receptor,
            delay_ms               = delay,
            biological_indegree    = indegree,
            release_component      = "primary",
        ))

    # Neurogliaform also co-releases GABA_B (ExpGABAab B-component)
    if pre == _NGF_PRE:
        b_receptors = pair_receptors(
            pre,
            post,
            synapse_variant,
            component="B",
            compartment_aware=compartment_aware,
            port_strategy=port_strategy,
        )
        for b_receptor in b_receptors:
            projections.append(Projection(
                pre                     = pre,
                post                    = post,
                indegree                = indegree,
                synapses_per_connection = spc,
                weight_nS               = _split_weight(weight / 3.37, b_receptors),
                receptor                = b_receptor,
                delay_ms                = delay,
                biological_indegree     = indegree,
                release_component       = "GABA_B",
            ))

    return projections


def load_projections(
    path: Path | None = None,
    conndata_index: int | None = None,
    cellnumbers_index: int = 101,
    conndata_count_mode: ConndataCountMode = "network_total",
    synapse_variant: int = 120,
    compartment_aware: bool = False,
    port_strategy: PortCompressionStrategy = "budget_weighted",
) -> list[Projection]:
    """Load all recurrent intra-CA1 projections from ``connectivity.json``.

    Covers all rows in ``excitatory_connections`` and
    ``inhibitory_connections`` (65 total in the source file).  Afferents
    (CA3/ECIII rows) are excluded here; use :func:`load_afferents`.

    Returns
    -------
    list[Projection]
        One entry per projection, with Neurogliaform projections doubled
        (GABA_A_slow + GABA_B co-release).
    """
    if path is not None and conndata_index is not None:
        raise ValueError("Pass either path or conndata_index, not both.")

    if conndata_index is not None:
        from ca1.extract.modeldb_tables import (
            extract_connectivity as raw_extract_connectivity,
        )

        extract_connectivity: _ExtractConnectivity = raw_extract_connectivity
        data = extract_connectivity(
            index=conndata_index,
            cellnumbers_index=cellnumbers_index,
            count_mode=conndata_count_mode,
        )
        modeldb_projection_section = cast(dict[str, JsonDict], data["projections"])
        modeldb_projections: list[Projection] = []
        for proj_data in modeldb_projection_section.values():
            modeldb_projections.extend(
                _parse_projection(
                    proj_data,
                    synapse_variant,
                    compartment_aware=compartment_aware,
                    port_strategy=port_strategy,
                )
            )
        if not modeldb_projections:
            message = (
                "load_projections: no projections found in conndata_"
                + f"{conndata_index}."
            )
            raise ValueError(
                message
            )
        return modeldb_projections

    if path is None:
        path = _PARAMS_DIR / "connectivity.json"

    with open(path, "r", encoding="utf-8") as fh:
        data = cast(JsonDict, json.load(fh))

    projections: list[Projection] = []

    for section_key in ("excitatory_connections", "inhibitory_connections"):
        section = cast(dict[str, JsonDict], data.get(section_key, {}))
        for proj_data in section.values():
            projections.extend(
                _parse_projection(
                    proj_data,
                    synapse_variant,
                    compartment_aware=compartment_aware,
                    port_strategy=port_strategy,
                )
            )

    if not projections:
        raise ValueError(
            "load_projections: no projections found in connectivity.json. "
            + "Check that 'excitatory_connections' and "
            + "'inhibitory_connections' keys are present."
        )

    return projections


def load_afferents(
    rate_hz: float = 0.65,
    path: Path | None = None,
    conndata_index: int | None = None,
    cellnumbers_index: int = 101,
    conndata_count_mode: ConndataCountMode = "network_total",
    synapse_variant: int = 120,
    compartment_aware: bool = False,
    port_strategy: PortCompressionStrategy = "budget_weighted",
) -> list[Afferent]:
    """Load CA3 and ECIII afferent drives from ``connectivity.json``.

    The ``indegree_true`` field in the legacy JSON path is a
    per-postsynaptic-cell synapse budget (total_synapses / N_post).  It is kept
    verbatim as ``synapses_per_cell`` and is NOT capped to the presynaptic
    population size.  Paper-faithful full-scale runs should pass
    ``conndata_index=430`` with ``conndata_count_mode="per_cell"`` instead of
    relying on the legacy JSON compatibility table.

    CA3 afferents use AMPA_fast kinetics; ECIII afferents use AMPA_slow
    kinetics.  Both use the specified Poisson rate.

    Parameters
    ----------
    rate_hz:
        Poisson firing rate for all afferent sources (default 0.65 Hz,
        the arrhythmic level at which theta emerges in Bezaire 2016 Fig 6).

    Returns
    -------
    list[Afferent]
        One entry per (source, post) pair in the ``afferents`` section.
    """
    if path is not None and conndata_index is not None:
        raise ValueError("Pass either path or conndata_index, not both.")

    if conndata_index is not None:
        from ca1.extract.modeldb_tables import (
            extract_connectivity as raw_extract_connectivity,
        )

        extract_connectivity: _ExtractConnectivity = raw_extract_connectivity
        data = extract_connectivity(
            index=conndata_index,
            cellnumbers_index=cellnumbers_index,
            count_mode=conndata_count_mode,
        )
        populations = cast(dict[str, int], data.get("populations", {}))
        modeldb_afferent_section = cast(dict[str, JsonDict], data["afferents"])
        modeldb_afferents: list[Afferent] = []
        for aff_data in modeldb_afferent_section.values():
            aff_block = aff_data
            pre = str(aff_block["presynaptic"])
            post = str(aff_block["postsynaptic"])
            receptor = pair_receptor(
                pre,
                post,
                synapse_variant,
                compartment_aware=compartment_aware,
                port_strategy=port_strategy,
            )
            assert_declared_receptor_matches(
                aff_block,
                pre=pre,
                post=post,
                derived_receptors=(receptor,),
            )
            modeldb_afferents.append(
                Afferent(
                    name=f"{pre}_to_{post}",
                    post=post,
                    n_source=_modeldb_afferent_source_count(pre, populations),
                    synapses_per_cell=_as_float(
                        aff_block["synapses_per_cell"], "synapses_per_cell"
                    ),
                    weight_nS=_as_float(aff_block["weight_nS"], "weight_nS"),
                    synapses_per_connection=_as_int(
                        aff_block["synapses_per_connection"],
                        "synapses_per_connection",
                    ),
                    receptor=receptor,
                    rate_hz=rate_hz,
                    delay_ms=DEFAULT_CONNECTION_DELAY_MS,
                )
            )
        if not modeldb_afferents:
            raise ValueError(
                f"load_afferents: no afferent entries found in conndata_{conndata_index}."
            )
        return modeldb_afferents

    if path is None:
        path = _PARAMS_DIR / "connectivity.json"

    with open(path, "r", encoding="utf-8") as fh:
        data = cast(JsonDict, json.load(fh))

    afferent_section = cast(dict[str, JsonDict], data.get("afferents", {}))
    if not afferent_section:
        raise ValueError(
            "load_afferents: 'afferents' key missing from connectivity.json."
        )

    result: list[Afferent] = []

    for aff_data in afferent_section.values():
        pre = str(aff_data["presynaptic"])
        post = str(aff_data["postsynaptic"])

        # n_source: use the canonical population count from the JSON header
        pops = cast(dict[str, int], data.get("populations_used", {}))
        n_source = _json_afferent_source_count(pre, pops)

        # synapses_per_cell = indegree_true, kept verbatim (not capped)
        synapses_per_cell = _as_float(aff_data["indegree_true"], "indegree_true")
        synapses_per_connection = _as_int(
            aff_data.get("synapses_per_connection", 1),
            "synapses_per_connection",
        )
        weight = _as_float(aff_data["weight_nS"], "weight_nS")
        receptor = pair_receptor(
            pre,
            post,
            synapse_variant,
            compartment_aware=compartment_aware,
            port_strategy=port_strategy,
        )
        assert_declared_receptor_matches(
            aff_data,
            pre=pre,
            post=post,
            derived_receptors=(receptor,),
        )

        result.append(
            Afferent(
                name             = f"{pre}_to_{post}",
                post             = post,
                n_source         = n_source,
                synapses_per_cell= synapses_per_cell,
                weight_nS        = weight,
                synapses_per_connection = synapses_per_connection,
                receptor         = receptor,
                rate_hz          = rate_hz,
                delay_ms         = DEFAULT_CONNECTION_DELAY_MS,
            )
        )

    if not result:
        raise ValueError("load_afferents: no afferent entries found.")

    return result
