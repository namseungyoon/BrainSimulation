"""ca1.sim subpackage -- simulator backend registry.

Usage::

    from ca1.sim import get_backend
    backend = get_backend("gpu")     # NestGpuBackend (primary, multi-GPU)
    backend = get_backend("nest")    # NestBackend (CPU oracle)

Imports are lazy: neither ``nest`` nor ``nestgpu`` is imported at module level,
so this package is importable on machines where neither simulator is installed.
"""

from __future__ import annotations

from ca1.sim.backend import SimulatorBackend


def get_backend(name: str) -> SimulatorBackend:
    """Factory that returns a fresh, uninitialised backend instance.

    Parameters
    ----------
    name : str
        ``'nest'`` for the CPU NEST correctness oracle, or ``'gpu'`` /
        ``'nestgpu'`` for the primary NEST GPU multi-GPU backend.

    Returns
    -------
    SimulatorBackend
        A concrete backend instance ready to receive ``setup()`` / ``build()``.

    Raises
    ------
    ValueError
        If ``name`` is not a recognised backend identifier.

    Notes
    -----
    Both backend modules defer their simulator imports to ``setup()``, so the
    factory itself never triggers a NEST or NEST GPU import.
    """
    key = name.lower().strip()
    if key == "nest":
        from ca1.sim.nest_backend import NestBackend  # lazy  # noqa: PLC0415
        return NestBackend()
    if key in ("gpu", "nestgpu", "nest_gpu"):
        from ca1.sim.gpu_backend import NestGpuBackend  # lazy  # noqa: PLC0415
        return NestGpuBackend()
    raise ValueError(
        f"Unknown backend {name!r}. Valid choices: 'nest', 'gpu' / 'nestgpu'."
    )


__all__ = ["SimulatorBackend", "get_backend"]
