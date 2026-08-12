from __future__ import annotations

import json
from pathlib import Path

from ca1.config import build_network_spec
from ca1.params.dendritic_transfer_provenance import (
    dendritic_transfer_source_provenance,
)


def _fit_record(source_location_table: str) -> dict[str, object]:
    return {
        "dend_C_frac": 0.4,
        "dend_leak_scale": 1.0,
        "g_c_scale": 3.993,
        "fit_provenance": "neuron-epsp-location-compressed-fit",
        "note": "Wave100 conndata211/syndata120 legacy source note",
        "validation": {
            "passed": True,
            "method": "user_m2-row-level-source-location-response-fidelity",
            "source_location_table": source_location_table,
        },
    }


def _carried_forward_record(source_location_table: str) -> dict[str, object]:
    record = _fit_record(source_location_table)
    record["note"] = (
        "Wave284: conndata430 cell-level g_c carried-forward; "
        "final dendritic response validated by source-location row transfer scales."
    )
    return record


def _conndata430_refit_record(
    source_location_table: str,
    *,
    response_peak_ratio: float,
    response_area_ratio: float,
) -> dict[str, object]:
    return {
        "dend_C_frac": 0.4,
        "dend_leak_scale": 1.0,
        "g_c_scale": 3.993,
        "fit_provenance": "neuron-epsp-location-compressed-fit-conndata430-per_cell",
        "note": "Wave284 conndata430/per_cell refit",
        "validation": {
            "passed": True,
            "method": "conndata430-weighted-morphology-target-refit",
            "source_location_table": source_location_table,
            "source_budget_conndata_index": 430,
            "source_budget_count_mode": "per_cell",
            "source_budget_cellnumbers_index": 101,
            "target_peak_ratio": 0.50,
            "response_peak_ratio": response_peak_ratio,
            "target_area_ratio": 0.60,
            "response_area_ratio": response_area_ratio,
        },
    }


def test_source_location_validation_alone_keeps_stale_note_visible(
    tmp_path: Path,
) -> None:
    spec = build_network_spec("configs/full_scale.yaml")
    fit_path = tmp_path / "fit.json"
    fit_path.write_text(
        json.dumps(
            {
                "Pyramidal": _fit_record(
                    "src/ca1/params/source_location_transfer_syndata120_budget_weighted.json",
                )
            }
        ),
        encoding="utf-8",
    )

    provenance = dendritic_transfer_source_provenance(
        path=fit_path,
        expected_cells={"Pyramidal"},
        spec=spec,
    )

    assert provenance == {
        "dendritic_transfer_source.Pyramidal": (
            "source-domain-mismatch;"
            "fit=conndata211/syndata120;"
            "spec=conndata430;count_mode=per_cell;cellnumbers=101"
        )
    }


def test_stale_source_location_validation_keeps_note_mismatch_visible(
    tmp_path: Path,
) -> None:
    spec = build_network_spec("configs/full_scale.yaml")
    table_path = tmp_path / "stale_transfer.json"
    table_path.write_text(
        json.dumps(
            [
                {
                    "pre": "CA3",
                    "post": "Pyramidal",
                    "receptor": "AMPA",
                    "port": "AMPA_fast__e0__tr2__td6p3__soma",
                    "aglif_compartment": "soma",
                    "source_budget_conndata_index": 211,
                    "source_budget_count_mode": "per_cell",
                    "source_budget_cellnumbers_index": 101,
                }
            ]
        ),
        encoding="utf-8",
    )
    fit_path = tmp_path / "fit.json"
    fit_path.write_text(
        json.dumps({"Pyramidal": _fit_record(str(table_path))}),
        encoding="utf-8",
    )

    provenance = dendritic_transfer_source_provenance(
        path=fit_path,
        expected_cells={"Pyramidal"},
        spec=spec,
    )

    assert provenance == {
        "dendritic_transfer_source.Pyramidal": (
            "source-domain-mismatch;"
            "fit=conndata211/syndata120;"
            "spec=conndata430;count_mode=per_cell;cellnumbers=101"
        )
    }


def test_carried_forward_row_validation_records_current_source_domain(
    tmp_path: Path,
) -> None:
    spec = build_network_spec("configs/full_scale.yaml")
    fit_path = tmp_path / "fit.json"
    fit_path.write_text(
        json.dumps(
            {
                "Pyramidal": _carried_forward_record(
                    "src/ca1/params/source_location_transfer_syndata120_budget_weighted.json",
                )
            }
        ),
        encoding="utf-8",
    )

    provenance = dendritic_transfer_source_provenance(
        path=fit_path,
        expected_cells={"Pyramidal"},
        spec=spec,
    )

    assert provenance == {
        "dendritic_transfer_source.Pyramidal": (
            "source-location-transfer-table-validation-passed;"
            "conndata430;count_mode=per_cell;cellnumbers=101;rows=101;"
            "method=user_m2-row-level-source-location-response-fidelity;"
            "cell_g_c=carried-forward"
        )
    }


def test_conndata430_refit_validation_records_current_source_domain(
    tmp_path: Path,
) -> None:
    spec = build_network_spec("configs/full_scale.yaml")
    fit_path = tmp_path / "fit.json"
    fit_path.write_text(
        json.dumps(
            {
                "Pyramidal": _conndata430_refit_record(
                    "src/ca1/params/source_location_transfer_syndata120_budget_weighted.json",
                    response_peak_ratio=0.55,
                    response_area_ratio=0.68,
                )
            }
        ),
        encoding="utf-8",
    )

    provenance = dendritic_transfer_source_provenance(
        path=fit_path,
        expected_cells={"Pyramidal"},
        spec=spec,
    )

    assert provenance == {
        "dendritic_transfer_source.Pyramidal": (
            "source-location-transfer-table-validation-passed;"
            "conndata430;count_mode=per_cell;cellnumbers=101;rows=101;"
            "method=conndata430-weighted-morphology-target-refit"
        )
    }


def test_conndata430_refit_out_of_tolerance_stays_visible(
    tmp_path: Path,
) -> None:
    spec = build_network_spec("configs/full_scale.yaml")
    fit_path = tmp_path / "fit.json"
    fit_path.write_text(
        json.dumps(
            {
                "Pyramidal": _conndata430_refit_record(
                    "src/ca1/params/source_location_transfer_syndata120_budget_weighted.json",
                    response_peak_ratio=0.80,
                    response_area_ratio=0.95,
                )
            }
        ),
        encoding="utf-8",
    )

    provenance = dendritic_transfer_source_provenance(
        path=fit_path,
        expected_cells={"Pyramidal"},
        spec=spec,
    )

    assert provenance["dendritic_transfer_source.Pyramidal"].startswith(
        "source-domain-refit-out-of-tolerance;"
    )
