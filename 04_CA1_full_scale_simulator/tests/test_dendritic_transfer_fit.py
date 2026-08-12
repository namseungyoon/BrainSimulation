from __future__ import annotations

import pytest

from ca1.params.dendritic_transfer import DendriticTransferParams
from ca1.params.dendritic_transfer_fit import (
    CandidateResponse,
    CellDendriteParams,
    SourceResponseTarget,
    TransferTarget,
    fit_transfer_for_target,
    joint_response_loss,
    response_constraints,
    simulate_transfer,
)


def test_simulate_transfer_ratio_increases_with_coupling() -> None:
    low = simulate_transfer(
        "Ivy",
        DendriticTransferParams(
            dend_C_frac=0.4,
            dend_leak_scale=1.0,
            g_c_scale=1.0,
            fit_provenance="test",
        ),
    )
    high = simulate_transfer(
        "Ivy",
        DendriticTransferParams(
            dend_C_frac=0.4,
            dend_leak_scale=1.0,
            g_c_scale=20.0,
            fit_provenance="test",
        ),
    )

    assert low.peak_ratio < high.peak_ratio
    assert low.area_ratio < high.area_ratio


def test_fit_transfer_for_target_finds_high_coupling_for_ivy_like_target() -> None:
    target = TransferTarget(peak_ratio=0.993, area_ratio=0.996)

    fitted = fit_transfer_for_target("Ivy", target)
    response = simulate_transfer("Ivy", fitted)

    assert fitted.g_c_scale > 20.0
    assert response.peak_ratio == pytest.approx(target.peak_ratio, abs=0.08)
    assert response.area_ratio == pytest.approx(target.area_ratio, abs=0.04)


def test_joint_cell_loss_applies_hard_peak_and_charge_gates_to_every_row() -> None:
    targets = [
        SourceResponseTarget("CA3->PV", 1.0, 1.0, 1.0, 1.0),
        SourceResponseTarget("Pyr->PV", 1.0, 1.0, 1.0, 1.0),
    ]
    params = CellDendriteParams(0.4, 1.0, 3.0)

    def passing(_row: str, _params: CellDendriteParams) -> CandidateResponse:
        return CandidateResponse(1.0, 0.95, 1.0, 1.0)

    def one_row_fails(row: str, _params: CellDendriteParams) -> CandidateResponse:
        charge = 0.89 if row == "CA3->PV" else 0.95
        return CandidateResponse(1.0, charge, 1.0, 1.0)

    assert all(response_constraints(target, passing(target.row_id, params)) for target in targets)
    assert not all(
        response_constraints(target, one_row_fails(target.row_id, params))
        for target in targets
    )
    assert joint_response_loss(targets, one_row_fails, params) > joint_response_loss(
        targets, passing, params
    )


def test_cell_dendrite_params_exposes_audited_distal_overrides() -> None:
    params = CellDendriteParams(0.45, 0.8, 4.0, 0.6, 0.7, 0.3)

    assert params.as_status_overrides() == {
        "dend_C_frac": 0.45,
        "dend_leak_scale": 0.8,
        "g_c_scale": 4.0,
        "dist_C_frac": 0.6,
        "dist_leak_scale": 0.7,
        "dist_coupling_ratio": 0.3,
    }
