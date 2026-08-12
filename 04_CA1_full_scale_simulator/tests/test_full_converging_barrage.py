from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts" / "full_converging_barrage.py"
    spec = importlib.util.spec_from_file_location("_full_converging_barrage_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_configured_barrage_rows_use_full_scale_modeldb_indegrees() -> None:
    barrage = _module()
    rows = barrage.configured_rows(barrage.DEFAULT_CONFIG)
    observed = {
        row.row_id: (row.indegree_true, row.source.synapses_per_connection)
        for cell_rows in rows.values() for row in cell_rows
    }
    assert observed == {
        "CA3->Bistratified": (5782, 2),
        "CA3->PV_Basket": (6047, 2),
        "ECIII->Bistratified": (432, 2),
        "Pyramidal->Bistratified": (366, 3),
        "Pyramidal->O_LM": (2379, 3),
        "Pyramidal->PV_Basket": (424, 3),
    }


def test_poisson_schedule_is_seeded_and_preserves_contact_shape() -> None:
    barrage = _module()
    row = barrage.configured_rows(barrage.DEFAULT_CONFIG)["PV_Basket"][0]
    first = barrage.poisson_schedule(row, 100.0, 17, 11)
    replay = barrage.poisson_schedule(row, 100.0, 17, 11)
    changed = barrage.poisson_schedule(row, 100.0, 18, 11)
    assert np.array_equal(first.event_times_ms, replay.event_times_ms)
    assert np.array_equal(first.location_indices, replay.location_indices)
    assert first.location_indices.shape == (
        row.indegree_true,
        row.source.synapses_per_connection,
    )
    assert not np.array_equal(first.event_times_ms, changed.event_times_ms)


def test_gap_closed_uses_absolute_source_gap() -> None:
    barrage = _module()
    summaries = []
    for cell in barrage.TARGET_CELLS:
        summaries.extend([
            {"cell": cell, "arm": "source_neuron", "dt_ms": 0.025, "rate_mean_hz": 10.0},
            {"cell": cell, "arm": "deployed_user_m2", "dt_ms": 0.025, "rate_mean_hz": 2.0},
            {"cell": cell, "arm": "candidate_user_m2", "dt_ms": 0.025, "rate_mean_hz": 6.0},
        ])
    rows = barrage._gap_closed(summaries, 0.025)
    assert all(row["gap_closed_percent"] == pytest.approx(50.0) for row in rows)
    assert all(row["candidate_closer_to_source"] for row in rows)
