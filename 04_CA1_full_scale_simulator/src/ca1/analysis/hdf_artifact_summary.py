from __future__ import annotations

from collections.abc import Mapping


def final_tier_spike_artifact_failures(
    n_cells: Mapping[str, int],
    spike_datasets: Mapping[str, int],
) -> list[str]:
    failures: list[str] = []
    for cell_type, declared in sorted(n_cells.items()):
        datasets = spike_datasets.get(cell_type, 0)
        if datasets != declared:
            failures.append(
                f"artifact: spikes.{cell_type} has {datasets} datasets but "
                + f"n_cells_per_type.{cell_type} declares {declared}; final-tier "
                + "evidence requires one spike train per declared cell"
            )
    for cell_type in sorted(set(spike_datasets) - set(n_cells)):
        failures.append(
            f"artifact: spikes.{cell_type} has no n_cells_per_type declaration"
        )
    return failures
