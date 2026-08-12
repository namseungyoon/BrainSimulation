"""CA1 full-scale spiking-network model.

Public contracts live in :mod:`ca1.types`. The simulator-agnostic graph is a
:class:`ca1.types.NetworkSpec`; backends in :mod:`ca1.sim` consume it; the
validation harness in :mod:`ca1.validation` scores results against Bezaire (2016).
"""

from __future__ import annotations

from ca1.types import (
    Afferent,
    CellType,
    CheckResult,
    NetworkSpec,
    NeuronParams,
    Projection,
    ReceptorConfig,
    SimMeta,
    SimResult,
    ValidationReport,
)

__version__ = "0.2.0"

__all__ = [
    "Afferent", "CellType", "CheckResult", "NetworkSpec", "NeuronParams",
    "Projection", "ReceptorConfig", "SimMeta", "SimResult", "ValidationReport",
    "__version__",
]
