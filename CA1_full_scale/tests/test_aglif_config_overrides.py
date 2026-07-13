from __future__ import annotations

import math

import pytest

from ca1.config import build_network_spec
from ca1.params.provenance import parameter_provenance_for_spec
from ca1.sim.aglif_dend import aglif_dend_compartments, aglif_dend_status


def _aglif_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "name": "aglif_config_override_test",
        "neuron_model": "aglif_dend_cond_beta",
        "compartment_aware_synapses": True,
        "receptor_port_strategy": "budget_weighted",
        "syndata_variant": 120,
        "conndata_index": 430,
        "conndata_count_mode": "per_cell",
        "cellnumbers_index": 101,
        "source_location_transfer_mode": "all_dend",
        "source_location_transfer_table": (
            "src/ca1/params/source_location_transfer_syndata120_budget_weighted.json"
        ),
    }
    config.update(overrides)
    return config


def test_aglif_receive_domain_override_routes_bistratified_ampa_to_soma() -> None:
    # Given: Bistratified has an explicit receive-domain override in config.
    spec = build_network_spec(
        _aglif_config(
            aglif_receive_domain_overrides={
                "Bistratified": "soma_excitatory",
            },
        )
    )
    receptor = next(
        name for name in spec.receptors.names
        if name.startswith("AMPA_fast") and name.endswith("__dend")
    )

    # When: AGLIF dendritic compartments are resolved from the configured spec.
    compartments = aglif_dend_compartments(
        (receptor,),
        "Bistratified",
        frozenset({receptor}),
        spec.source_location_transfer_table,
        spec.aglif_receive_domain_overrides,
    )

    # Then: the excitatory dendritic receptor is explicitly routed to soma.
    assert compartments == [0.0]


def test_aglif_gc_scale_override_routes_into_aglif_status() -> None:
    # Given: O_LM has an explicit config-backed dendritic coupling scale.
    spec = build_network_spec(
        _aglif_config(
            aglif_gc_scale_overrides={
                "O_LM": 4.0,
            },
        )
    )

    # When: AGLIF dendritic status is generated from the configured scale.
    baseline = aglif_dend_status("O_LM")
    configured = aglif_dend_status(
        "O_LM",
        spec.aglif_gc_scale_overrides["O_LM"],
    )

    # Then: g_c and its distal counterpart reflect the explicit scale.
    assert math.isclose(configured["g_c"], baseline["g_c"] * 4.0)
    assert math.isclose(configured["g_c_dist"], baseline["g_c_dist"] * 4.0)


def test_aglif_override_provenance_records_explicit_config() -> None:
    # Given: both AGLIF override families are configured.
    spec = build_network_spec(
        _aglif_config(
            aglif_receive_domain_overrides={
                "Bistratified": "soma_excitatory",
            },
            aglif_gc_scale_overrides={
                "O_LM": 4.0,
            },
        )
    )

    # When: parameter provenance is emitted.
    provenance = parameter_provenance_for_spec(spec)

    # Then: the choices are visible in parameter provenance, not hidden in env.
    assert provenance["aglif.receive_domain_overrides"] == (
        '{"Bistratified": "soma_excitatory"}'
    )
    assert provenance["aglif.gc_scale_overrides"] == '{"O_LM": 4.0}'


def test_aglif_receive_domain_override_rejects_unknown_cell() -> None:
    with pytest.raises(ValueError, match="unknown AGLIF override cell type"):
        _ = build_network_spec(
            _aglif_config(
                aglif_receive_domain_overrides={
                    "NotACell": "soma_excitatory",
                },
            )
        )


def test_aglif_receive_domain_override_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unsupported AGLIF receive-domain mode"):
        _ = build_network_spec(
            _aglif_config(
                aglif_receive_domain_overrides={
                    "Bistratified": "silent_fallback",
                },
            )
        )


@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, True])
def test_aglif_gc_scale_override_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="aglif_gc_scale_overrides"):
        _ = build_network_spec(
            _aglif_config(
                aglif_gc_scale_overrides={
                    "O_LM": value,
                },
            )
        )
