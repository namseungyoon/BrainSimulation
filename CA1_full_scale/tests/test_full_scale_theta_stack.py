from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pytest
import yaml

from ca1.config import build_network_spec
from ca1.params.provenance import parameter_provenance_for_spec
from ca1.sim.aglif_dend import aglif_dend_compartments
from ca1.sim.edge_artifact import graph_identity_digest
from ca1.sim.gpu_backend import (
    _nestgpu_model_name_for_cell,
    _neuron_status,
    _required_dendritic_ports,
)
from ca1.validation.provenance import final_tier_parameter_provenance_blockers


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/full_scale_3dtopo.yaml"
STACK = ROOT / "configs/full_scale_theta_stack.yaml"


def _row_key(row: object) -> str:
    return f"{row.pre}->{row.post}|{row.receptor}"


def test_theta_stack_config_changes_only_the_explicit_stack() -> None:
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    stack = yaml.safe_load(STACK.read_text(encoding="utf-8"))

    assert stack.pop("name") == "ca1_full_scale_theta_stack"
    assert base.pop("name") == "ca1_full_scale"
    assert stack.pop("aglif_dend_overrides") == {
        "CCK_Basket": {"model": "user_m3"},
        "Bistratified": {"model": "user_m4"},
        "O_LM": {"model": "user_m5"},
        "PV_Basket": {"model": "user_m2"},
    }
    assert stack.pop("source_grounded_stack") == {
        "refit_candidate": "../results/cck_sca_refit_candidate.json",
        "refit_cells": ["SCA"],
        "gaba_into_cck_candidate": "../results/gaba_into_cck_candidate.json",
    }
    assert stack == base


def test_theta_stack_resolves_models_candidate_params_and_honest_provenance() -> None:
    spec = build_network_spec(STACK)
    assert {
        cell: _nestgpu_model_name_for_cell(spec, cell)
        for cell in ("CCK_Basket", "Bistratified", "O_LM", "PV_Basket", "SCA")
    } == {
        "CCK_Basket": "user_m3",
        "Bistratified": "user_m4",
        "O_LM": "user_m5",
        "PV_Basket": "user_m2",
        "SCA": "user_m2",
    }

    refit = json.loads(
        (ROOT / "results/cck_sca_refit_candidate.json").read_text(encoding="utf-8")
    )
    expected_status = refit["intrinsic"]["cells"]["SCA"]["fitted_params"]
    status = _neuron_status(spec, "SCA", spec.cell_types["SCA"].params)
    assert {key: status[key] for key in expected_status} == expected_status

    weights = {_row_key(row): row.weight_nS for row in spec.projections}
    pair_weights = {f"{row.pre}->{row.post}": row.weight_nS for row in spec.projections}
    afferent_weights = {
        f"{row.name.split('_to_', 1)[0]}->{row.post}": row.weight_nS
        for row in spec.afferents
    }
    for row in refit["excitatory_transfer"]["rows"]:
        if row["contract"]["post"] != "SCA":
            continue
        key = row["row"]
        actual = pair_weights.get(key, afferent_weights.get(key))
        assert actual == pytest.approx(row["candidate_mapping"]["transferred_gmax_nS"])

    gaba = json.loads(
        (ROOT / "results/gaba_into_cck_candidate.json").read_text(encoding="utf-8")
    )
    assert len(gaba["rows"]) == 5
    for row in gaba["rows"]:
        assert weights[row["row_key"]] == pytest.approx(row["transferred_gmax_nS"])

    cck_compartments = aglif_dend_compartments(
        spec.receptors.names,
        "CCK_Basket",
        _required_dendritic_ports(spec, "CCK_Basket"),
        spec.source_location_transfer_table,
        spec.aglif_receive_domain_overrides,
        spec.aglif_compartment_overrides,
    )
    olm_row = next(row for row in gaba["rows"] if row["pre"] == "O_LM")
    olm_port = spec.receptors.port_index(olm_row["deployed_receptor"])
    assert cck_compartments[olm_port] == 0.0

    provenance = parameter_provenance_for_spec(spec)
    assert provenance["aglif_dend_override.PV_Basket.model"] == "user_m2"
    assert "cells=SCA" in provenance["source_grounded.refit"]
    assert "rows=5" in provenance["source_grounded.gaba_into_cck"]
    assert final_tier_parameter_provenance_blockers(
        provenance, spec.scaled_counts()
    ) == []


def test_theta_stack_preserves_graph_identity_and_connect_contract() -> None:
    base = build_network_spec(BASE)
    stack = build_network_spec(STACK)
    assert graph_identity_digest(base) == graph_identity_digest(stack)
    assert [
        {
            key: value
            for key, value in asdict(row).items()
            if key != "weight_nS"
        }
        for row in base.projections
    ] == [
        {
            key: value
            for key, value in asdict(row).items()
            if key != "weight_nS"
        }
        for row in stack.projections
    ]
    assert [
        {
            key: value
            for key, value in asdict(row).items()
            if key != "weight_nS"
        }
        for row in base.afferents
    ] == [
        {
            key: value
            for key, value in asdict(row).items()
            if key != "weight_nS"
        }
        for row in stack.afferents
    ]
