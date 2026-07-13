from __future__ import annotations

import pytest

from ca1.sim.intrinsic_heterogeneity import (
    IntrinsicHeterogeneity,
    MissingIntrinsicBaselineError,
    intrinsic_heterogeneity_status,
)
from ca1.types import NeuronParams


def _neuron_params() -> NeuronParams:
    return NeuronParams(
        C_m=100.0,
        g_L=5.0,
        E_L=-65.0,
        V_th=-50.0,
        V_reset=-60.0,
        Delta_T=2.0,
        a=0.0,
        b=0.0,
        tau_w=100.0,
        t_ref=2.0,
    )


@pytest.mark.parametrize(
    ("config", "baseline_status", "missing_field"),
    [
        (
            IntrinsicHeterogeneity(
                v_th_sigma_mv=0.1,
                e_l_sigma_mv=0.0,
                v_m_sigma_mv=0.0,
                clip_sigma=1.0,
            ),
            {"E_L": -63.0, "V_m": -63.0},
            "V_th",
        ),
        (
            IntrinsicHeterogeneity(
                v_th_sigma_mv=0.0,
                e_l_sigma_mv=0.1,
                v_m_sigma_mv=0.0,
                clip_sigma=1.0,
            ),
            {"V_m": -63.0, "V_th": -48.0},
            "E_L",
        ),
        (
            IntrinsicHeterogeneity(
                v_th_sigma_mv=0.0,
                e_l_sigma_mv=0.0,
                v_m_sigma_mv=0.1,
                clip_sigma=1.0,
            ),
            {"E_L": -63.0, "V_th": -48.0},
            "V_m",
        ),
    ],
)
def test_intrinsic_heterogeneity_rejects_missing_backend_baseline_field(
    config: IntrinsicHeterogeneity,
    baseline_status: dict[str, float],
    missing_field: str,
) -> None:
    with pytest.raises(MissingIntrinsicBaselineError, match=missing_field):
        intrinsic_heterogeneity_status(
            cell_type="Pyramidal",
            params=_neuron_params(),
            count=3,
            seed=123,
            config=config,
            baseline_status=baseline_status,
        )


def test_intrinsic_heterogeneity_allows_unused_e_l_to_be_absent() -> None:
    status = intrinsic_heterogeneity_status(
        cell_type="PV_Basket",
        params=_neuron_params(),
        count=2,
        seed=123,
        config=IntrinsicHeterogeneity(
            v_th_sigma_mv=0.1,
            e_l_sigma_mv=0.0,
            v_m_sigma_mv=0.0,
            clip_sigma=1.0,
        ),
        baseline_status={"V_m": -70.0, "V_th": 30.0},
    )

    assert set(status) == {"V_th"}


def test_intrinsic_heterogeneity_rejects_e_l_as_hidden_v_m_baseline() -> None:
    with pytest.raises(MissingIntrinsicBaselineError, match="V_m"):
        intrinsic_heterogeneity_status(
            cell_type="O_LM",
            params=_neuron_params(),
            count=2,
            seed=123,
            config=IntrinsicHeterogeneity(
                v_th_sigma_mv=0.0,
                e_l_sigma_mv=0.1,
                v_m_sigma_mv=0.0,
                clip_sigma=1.0,
            ),
            baseline_status={"E_L": -63.0, "V_th": -48.0},
        )
