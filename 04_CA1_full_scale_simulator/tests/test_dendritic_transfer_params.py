from __future__ import annotations

import json
from pathlib import Path

import pytest

from ca1.analysis.location_transfer import (
    UnvalidatedLocationTransferError,
    apply_location_transfer,
)
from ca1.config import build_network_spec
from ca1.params.dendritic_transfer import (
    DendriticTransferParams,
    load_dendritic_transfer_params,
)
from ca1.params.dendritic_transfer_fit import simulate_transfer
from ca1.params.groundtruth import CELL_TEMPLATES
from ca1.params.provenance import parameter_provenance_for_spec
from ca1.validation.provenance import final_tier_parameter_provenance_blockers


_WAVE112_DISTAL_PEAK_TARGETS = {
    "Pyramidal": 0.05065020992856072,
    "PV_Basket": 0.1231117401481653,
    "CCK_Basket": 0.01388275440976371,
    "Axo": 0.12429931911363144,
    "Bistratified": 0.09785991316500323,
    "SCA": 0.06609675857727453,
}
_SOURCE_LOCATION_TABLE = Path(
    ".omo/ulw-loop/g002-continuation/evidence/fullscale_source_kinetic_ports_wave108/"
    "location_transfer_compensated_budget_with_afferents.json"
)


def _transfer_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "dend_C_frac": 0.4,
        "dend_leak_scale": 1.0,
        "g_c_scale": 1.0,
        "fit_provenance": "neuron-synaptic-transfer-fit",
    }
    record.update(overrides)
    return record


def _complete_transfer_fit(
    **overrides_by_cell: dict[str, object],
) -> dict[str, object]:
    return {
        name: _transfer_record(**overrides_by_cell.get(name, {}))
        for name in CELL_TEMPLATES
    }


def test_missing_dendritic_transfer_file_raises_not_fallback(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="missing dendritic-transfer params"):
        _ = load_dendritic_transfer_params(tmp_path / "missing.json")


def test_default_dendritic_transfer_file_covers_all_cell_types() -> None:
    params = load_dendritic_transfer_params()

    assert set(params) == {
        "Pyramidal",
        "PV_Basket",
        "CCK_Basket",
        "Axo",
        "Bistratified",
        "Ivy",
        "O_LM",
        "SCA",
        "Neurogliaform",
    }
    assert params["Bistratified"] == DendriticTransferParams(
        dend_C_frac=0.4,
        dend_leak_scale=1.0,
        g_c_scale=6.226,
        fit_provenance=(
            "neuron-epsp-location-compressed-fit-proximal-afferent-preserving"
        ),
    )


def test_full_scale_dendritic_transfer_source_domain_is_currently_validated() -> None:
    spec = build_network_spec("configs/full_scale.yaml")

    provenance = parameter_provenance_for_spec(spec)
    blockers = final_tier_parameter_provenance_blockers(
        provenance,
        spec.scaled_counts(),
    )

    assert provenance["source_location_transfer.table"].startswith(
        "source-location-transfer-m2-row-validation-passed"
    )
    assert "source_location_transfer.table=missing" not in blockers
    assert not any(blocker.startswith("dendritic_transfer.") for blocker in blockers)
    assert provenance["dendritic_transfer_source.Pyramidal"].endswith(
        "method=user_m2-row-level-source-location-response-fidelity;"
        "cell_g_c=carried-forward"
    )
    assert not any(
        blocker.startswith("dendritic_transfer_source.")
        for blocker in blockers
    )


def test_dendritic_transfer_validation_is_source_location_coupled() -> None:
    records = json.loads(
        Path("src/ca1/params/dendritic_transfer_fitted.json").read_text(
            encoding="utf-8",
        )
    )

    for record in records.values():
        validation = record["validation"]
        assert validation["passed"] is True
        assert validation["method"] == (
            "user_m2-row-level-source-location-response-fidelity"
        )
        assert validation["source_location_table"] == (
            "src/ca1/params/source_location_transfer_syndata120_budget_weighted.json"
        )


def test_source_location_transfer_candidate_is_not_final_validated() -> None:
    rows = json.loads(_SOURCE_LOCATION_TABLE.read_text(encoding="utf-8"))
    compensated_errors = [
        abs(
            row["morph_ratio_est"]
            - row["reduced_ratio_est"] * row["transfer_scale"]
        )
        for row in rows
        if "reduced_ratio_est" in row and "transfer_scale" in row
    ]
    spec = build_network_spec({
        "name": "source_location_candidate",
        "neuron_model": "aglif_dend_cond_beta",
        "compartment_aware_synapses": True,
        "conndata_index": 430,
        "conndata_count_mode": "per_cell",
        "cellnumbers_index": 101,
        "syndata_variant": 120,
    })

    assert len(rows) == 101
    assert len(compensated_errors) == 84
    assert max(compensated_errors) < 1.0e-12
    with pytest.raises(
        UnvalidatedLocationTransferError,
        match="M2 response validation",
    ):
        _ = apply_location_transfer(spec, "all_dend", _SOURCE_LOCATION_TABLE)


def test_current_single_dendrite_transfer_misses_wave112_distal_targets() -> None:
    params = load_dendritic_transfer_params()

    errors = {
        cell_type: abs(
            simulate_transfer(
                cell_type,
                params[cell_type],
                tau_rise_ms=2.0,
                tau_decay_ms=6.3,
            ).peak_ratio
            - target
        )
        for cell_type, target in _WAVE112_DISTAL_PEAK_TARGETS.items()
    }

    assert set(errors) == set(_WAVE112_DISTAL_PEAK_TARGETS)
    assert min(errors.values()) > 0.15


def test_dendritic_transfer_prefers_validated_fit_record(
    tmp_path: Path,
) -> None:
    fit_path = tmp_path / "dendritic_transfer_fitted.json"
    _ = fit_path.write_text(
        json.dumps(
            _complete_transfer_fit(
                Bistratified={
                    "dend_C_frac": 0.22,
                    "dend_leak_scale": 0.7,
                    "g_c_scale": 8.5,
                }
            )
        ),
        encoding="utf-8",
    )

    params = load_dendritic_transfer_params(fit_path)

    assert params["Bistratified"] == DendriticTransferParams(
        dend_C_frac=0.22,
        dend_leak_scale=0.7,
        g_c_scale=8.5,
        fit_provenance="neuron-synaptic-transfer-fit",
    )


def test_failed_dendritic_transfer_fit_fails_loudly(
    tmp_path: Path,
) -> None:
    fit_path = tmp_path / "dendritic_transfer_fitted.json"
    _ = fit_path.write_text(
        json.dumps(
            {
                "Ivy": {
                    "dend_C_frac": 0.05,
                    "dend_leak_scale": 9.0,
                    "g_c_scale": 20.0,
                    "fit_provenance": "FAILED",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Ivy.*marked FAILED"):
        _ = load_dendritic_transfer_params(fit_path)


def test_rejected_dendritic_transfer_validation_fails_loudly(
    tmp_path: Path,
) -> None:
    fit_path = tmp_path / "dendritic_transfer_fitted.json"
    payload = _complete_transfer_fit(
        Bistratified={"validation": {"passed": False}},
    )
    _ = fit_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Bistratified.*failed validation"):
        _ = load_dendritic_transfer_params(fit_path)


def test_missing_dendritic_transfer_fit_provenance_fails_loudly(
    tmp_path: Path,
) -> None:
    fit_path = tmp_path / "dendritic_transfer_fitted.json"
    payload = _complete_transfer_fit()
    record = _transfer_record()
    _ = record.pop("fit_provenance")
    payload["Bistratified"] = record
    _ = fit_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fit_provenance.*required"):
        _ = load_dendritic_transfer_params(fit_path)


def test_missing_dendritic_transfer_fit_fails_loudly(
    tmp_path: Path,
) -> None:
    fit_path = tmp_path / "dendritic_transfer_fitted.json"
    _ = fit_path.write_text(
        json.dumps({"Bistratified": _transfer_record()}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing dendritic-transfer fit"):
        _ = load_dendritic_transfer_params(fit_path)


def test_unknown_dendritic_transfer_cell_type_fails_loudly(
    tmp_path: Path,
) -> None:
    fit_path = tmp_path / "dendritic_transfer_fitted.json"
    _ = fit_path.write_text(
        json.dumps(
            {
                "Basket": {
                    "dend_C_frac": 0.2,
                    "dend_leak_scale": 1.0,
                    "g_c_scale": 2.0,
                    "fit_provenance": "neuron-synaptic-transfer-fit",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown cell type 'Basket'"):
        _ = load_dendritic_transfer_params(fit_path)


def test_default_path_uses_complete_explicit_transfer_file() -> None:
    params = load_dendritic_transfer_params()

    assert (
        params["Bistratified"].fit_provenance
        == "neuron-epsp-location-compressed-fit-proximal-afferent-preserving"
    )
    assert abs(params["Bistratified"].g_c_scale - 6.226) < 1e-12
    assert (
        params["SCA"].fit_provenance
        == "neuron-epsp-location-compressed-fit"
    )
