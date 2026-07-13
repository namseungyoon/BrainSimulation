from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from ca1.types import ElectrodeRoi, SimMeta, SimResult
from ca1.validation import acceptance
from ca1.validation.targets import MODEL_PHASE_DEG, MODEL_RATES_HZ


def test_first_order_rates_use_electrode_roi_spike_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: two Pyramidal spike trains, but only the first cell is inside the ROI.
    meta = SimMeta(
        duration_s=1.0,
        dt_s=0.001,
        n_cells_per_type={cell_type: 1 for cell_type in MODEL_RATES_HZ},
        scale=1.0,
        seed=1,
        backend="test",
        config_name="roi_rates",
        crop_first_ms=0.0,
    )
    result = SimResult(
        spikes={
            **{
                cell_type: [np.asarray([0.1], dtype=np.float64)]
                for cell_type in MODEL_RATES_HZ
                if cell_type != "Pyramidal"
            },
            "Pyramidal": [
                np.asarray([0.1], dtype=np.float64),
                np.asarray([0.2], dtype=np.float64),
            ],
        },
        meta=meta,
        cell_positions_um={
            **{
                cell_type: np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)
                for cell_type in MODEL_RATES_HZ
                if cell_type != "Pyramidal"
            },
            "Pyramidal": np.asarray(
                [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
                dtype=np.float64,
            ),
        },
        analysis_roi=ElectrodeRoi(
            center_um=(0.0, 0.0, 0.0),
            radius_um=5.0,
            distance_mode="xyz",
        ),
    )

    class FakeRates:
        seen_pyramidal_count = 0

        def mean_rates(
            self,
            spikes: acceptance.Spikes,
            _duration_s: float,
            active_only: bool = False,
        ) -> dict[str, float]:
            del active_only
            self.seen_pyramidal_count = len(spikes["Pyramidal"])
            return dict(MODEL_RATES_HZ)

        def cv_isi(self, _spikes: acceptance.Spikes) -> dict[str, float]:
            return {cell_type: 1.0 for cell_type in MODEL_RATES_HZ}

    fake_rates = FakeRates()
    monkeypatch.setattr(acceptance, "_load_rates", lambda: fake_rates)

    # When: first-order validation runs.
    checks = acceptance.check_first_order(result)

    # Then: the rate module saw only ROI-selected cells.
    assert fake_rates.seen_pyramidal_count == 1
    rate_checks = [check for check in checks if check.name == "rate/Pyramidal"]
    assert rate_checks[0].metrics["analysis_scope"] == "electrode_roi"
    assert rate_checks[0].metrics["roi_cells"] == 1


def test_phase_preference_uses_electrode_roi_spike_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: marker spikes where only the first cell of each type is inside the ROI.
    cell_by_marker = {
        float(idx + 1): cell_type
        for idx, cell_type in enumerate(MODEL_PHASE_DEG)
    }
    meta = SimMeta(
        duration_s=1.0,
        dt_s=0.001,
        n_cells_per_type={cell_type: 2 for cell_type in MODEL_PHASE_DEG},
        scale=1.0,
        seed=1,
        backend="test",
        config_name="roi_phase",
        crop_first_ms=0.0,
        lfp_proxy="pyramidal_spike_density",
    )
    result = SimResult(
        spikes={
            cell_type: [
                np.full(5, float(idx + 1), dtype=np.float64),
                np.full(5, 100.0 + float(idx), dtype=np.float64),
            ]
            for idx, cell_type in enumerate(MODEL_PHASE_DEG)
        },
        meta=meta,
        cell_positions_um={
            cell_type: np.asarray(
                [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
                dtype=np.float64,
            )
            for cell_type in MODEL_PHASE_DEG
        },
        analysis_roi=ElectrodeRoi(
            center_um=(0.0, 0.0, 0.0),
            radius_um=5.0,
            distance_mode="xyz",
        ),
    )

    def fake_phase_preference(
        spike_times: npt.NDArray[np.float64],
        _lfp: npt.NDArray[np.float64],
        _fs: float,
        band: tuple[float, float],
    ) -> tuple[float, float, float]:
        del band
        marker = cast(float, spike_times.tolist()[0])
        cell_type = cell_by_marker[marker]
        return MODEL_PHASE_DEG[cell_type], 0.9, 1e-12

    fake_spectral = SimpleNamespace(
        lfp_proxy=lambda _result: (np.ones(100), 1000.0),
        phase_preference=fake_phase_preference,
    )
    monkeypatch.setattr(acceptance, "_load_spectral", lambda: fake_spectral)

    # When: phase validation runs.
    checks = acceptance.check_phase(result)

    # Then: outside-ROI marker spikes never reach phase_preference.
    per_type_checks = [check for check in checks if check.name.startswith("phase/")]
    assert per_type_checks
    assert all(check.passed for check in per_type_checks)
    assert all(
        check.metrics.get("analysis_scope") == "electrode_roi"
        for check in per_type_checks
        if "analysis_scope" in check.metrics
    )
