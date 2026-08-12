from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest


def test_mpi_rank_size_falls_back_to_mpich_environment(monkeypatch) -> None:
    from ca1.sim.gpu_backend import _mpi_rank_size_from_env

    monkeypatch.setenv("PMI_RANK", "2")
    monkeypatch.setenv("PMI_SIZE", "3")

    assert _mpi_rank_size_from_env() == (2, 3)


def test_gpu_backend_merges_rank_spike_files_in_host_order(tmp_path: Path) -> None:
    from ca1.sim.gpu_backend import NestGpuBackend

    backend = NestGpuBackend()
    backend._mpi_size = 3
    rank_payloads = [
        {"Pyramidal": [np.array([0.1])], "O_LM": [np.array([], dtype=float)]},
        {"Pyramidal": [np.array([0.2]), np.array([0.3])], "O_LM": []},
        {"Pyramidal": [np.array([0.4])], "O_LM": [np.array([0.5])]},
    ]
    for rank, payload in enumerate(rank_payloads):
        with backend._rank_spikes_path(tmp_path, rank).open("wb") as fh:
            pickle.dump(payload, fh)

    merged = backend._merge_rank_spikes(tmp_path, {"Pyramidal": 4, "O_LM": 2})

    assert [arr.tolist() for arr in merged["Pyramidal"]] == [[0.1], [0.2], [0.3], [0.4]]
    assert [arr.tolist() for arr in merged["O_LM"]] == [[], [0.5]]


def test_gpu_backend_persists_rank_spikes_atomically(tmp_path: Path) -> None:
    from ca1.sim.gpu_backend import NestGpuBackend

    backend = NestGpuBackend()
    out = tmp_path / "spikes_raw_rank0.pkl"
    payload = {"Pyramidal": [np.array([0.1, 0.2])]}

    backend.persist_spikes(payload, out)

    with out.open("rb") as fh:
        loaded = pickle.load(fh)
    assert loaded["Pyramidal"][0].tolist() == [0.1, 0.2]
    assert not list(tmp_path.glob("*.tmp.*"))


def test_gpu_backend_merge_times_out_when_rank_file_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ca1.sim.gpu_backend import NestGpuBackend

    backend = NestGpuBackend()
    backend._mpi_size = 2
    monkeypatch.setenv("CA1_MPI_MERGE_TIMEOUT_S", "0.01")
    with backend._rank_spikes_path(tmp_path, 0).open("wb") as fh:
        pickle.dump({"Pyramidal": [np.array([], dtype=float)]}, fh)

    with pytest.raises(TimeoutError, match="rank1"):
        backend._merge_rank_spikes(tmp_path, {"Pyramidal": 1})
