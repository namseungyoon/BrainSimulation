from __future__ import annotations

import ca1.sim.gpu_backend as gpu_backend_mod
import numpy as np
import numpy.typing as npt
import pytest

from ca1.sim.gpu_backend import NestGpuBackend
from ca1.sim.nestgpu_api import NestGpuModule, nestgpu_module
from ca1.sim.source_rate_heterogeneity import source_rates_hz
from ca1.types import Afferent, CellType, NetworkSpec, NeuronParams

_StatusValue = float | int | str | list[float] | dict[str, list[float]] | None
_StatusParams = dict[str, _StatusValue]
_ConnSpec = dict[str, str | int]
_SynSpec = dict[str, float | int]


class _FakeNodeCollection:
    count: int
    label: str

    def __init__(self, count: int, label: str = "") -> None:
        self.count = count
        self.label = label

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, key: slice) -> "_FakeNodeCollection":
        del key
        return _FakeNodeCollection(1, f"{self.label}[slice]")


class _TestNestGpuBackend(NestGpuBackend):
    _ngpu: NestGpuModule | None

    def install_ngpu(self, fake_ngpu: _FakeNestGpu) -> None:
        self._ngpu = nestgpu_module(fake_ngpu)

    def set_literal_source_spike_trains(self, *, duration_s: float, seed: int) -> None:
        self._set_literal_source_spike_trains(duration_s=duration_s, seed=seed)


class _FakeNestGpu:
    def __init__(self) -> None:
        self.create_calls: list[tuple[str, int, int | None]] = []
        self.connect_calls: list[
            tuple[_FakeNodeCollection, _FakeNodeCollection, _ConnSpec, _SynSpec]
        ] = []
        self.set_status_calls: list[tuple[_FakeNodeCollection, _StatusParams]] = []

    def Create(self, model: str, count: int, n_ports: int | None = None) -> _FakeNodeCollection:  # noqa: N802
        self.create_calls.append((model, int(count), n_ports))
        return _FakeNodeCollection(int(count), model)

    def SetStatus(  # noqa: N802
        self,
        nodes: _FakeNodeCollection,
        params: _StatusParams | str,
        val: _StatusValue = None,
    ) -> None:
        status = {params: val} if isinstance(params, str) else params
        self.set_status_calls.append((nodes, status))

    def Connect(  # noqa: N802
        self,
        pre: _FakeNodeCollection,
        post: _FakeNodeCollection,
        conn_spec: _ConnSpec,
        syn_spec: _SynSpec,
    ) -> None:
        self.connect_calls.append((pre, post, conn_spec, syn_spec))


def _params() -> NeuronParams:
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


def test_source_rate_profile_is_homogeneous_when_cv_is_zero() -> None:
    # Given: the paper control condition has no per-source rate spread.
    # When: source rates are generated for one source population.
    rates = source_rates_hz(
        base_rate_hz=0.65,
        count=4,
        cv=0.0,
        seed=123,
        source="CA3",
    )

    # Then: every source has exactly the configured control rate.
    assert rates.tolist() == [0.65, 0.65, 0.65, 0.65]


def test_source_rate_profile_uses_positive_lognormal_heterogeneity() -> None:
    # Given: a nonzero CV for paper-plausible source-rate heterogeneity.
    # When: rates are generated deterministically for one source population.
    rates = source_rates_hz(
        base_rate_hz=0.65,
        count=128,
        cv=0.25,
        seed=123,
        source="ECIII",
    )

    # Then: source rates are positive and genuinely nonuniform.
    assert np.all(rates > 0.0)
    assert len(set(np.round(rates, decimals=12))) > 1
    mean_rate = float(np.mean(rates))
    assert 0.5525 <= mean_rate <= 0.7475


def test_gpu_literal_source_graph_passes_source_rate_cv_to_spike_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, float | int | str] = {}

    def fake_source_rates_hz(
        *,
        base_rate_hz: float,
        count: int,
        cv: float,
        seed: int,
        source: str,
    ) -> npt.NDArray[np.float64]:
        captured.update(
            {
                "base_rate_hz": base_rate_hz,
                "count": count,
                "cv": cv,
                "seed": seed,
                "source": source,
            }
        )
        return np.full(count, 50.0, dtype=np.float64)

    monkeypatch.setattr(gpu_backend_mod, "source_rates_hz", fake_source_rates_hz)
    backend = _TestNestGpuBackend()
    fake_ngpu = _FakeNestGpu()
    backend.install_ngpu(fake_ngpu)
    spec = NetworkSpec(
        name="gpu-literal-source-rate-heterogeneity",
        cell_types={
            "Pyramidal": CellType(
                name="Pyramidal",
                count=2,
                layers=("SP",),
                params=_params(),
            )
        },
        projections=[],
        afferents=[
            Afferent(
                name="CA3_to_Pyramidal",
                post="Pyramidal",
                n_source=4,
                synapses_per_cell=2.0,
                weight_nS=0.2,
                receptor="AMPA_fast",
                rate_hz=0.65,
            )
        ],
        afferent_topology="literal_source_graph",
        afferent_source_rate_cv=0.35,
    )
    backend.build(spec, {"Pyramidal": 2})

    # When: precomputed source spike trains are installed on NEST-GPU generators.
    backend.set_literal_source_spike_trains(duration_s=0.5, seed=7)

    # Then: the literal-source path used the configured per-source CV.
    assert captured == {
        "base_rate_hz": 0.65,
        "count": 4,
        "cv": 0.35,
        "seed": 7,
        "source": "CA3",
    }
    assert not any(
        nodes.label == "spike_generator" and "rate" in params
        for nodes, params in fake_ngpu.set_status_calls
    )
