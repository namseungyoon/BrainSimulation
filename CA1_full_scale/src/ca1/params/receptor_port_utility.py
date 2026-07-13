from __future__ import annotations

from dataclasses import dataclass
from typing import Final, TypeAlias

from typing_extensions import override

PortKey: TypeAlias = tuple[str, float, float, float]
_CompartmentPortKey: TypeAlias = tuple[str, float, float, float, str]

_UTILITY_WEIGHTED_COMPARTMENT_PORTS: Final[frozenset[_CompartmentPortKey]] = (
    frozenset({
        ("AMPA_fast", 0.0, 0.07, 0.2, "dend"),
        ("AMPA_fast", 0.0, 0.1, 1.5, "dend"),
        ("AMPA_fast", 0.0, 0.11, 0.25, "dend"),
        ("AMPA_fast", 0.0, 0.3, 0.6, "dend"),
        ("AMPA_fast", 0.0, 0.5, 3.0, "dend"),
        ("AMPA_fast", 0.0, 2.0, 6.3, "dend"),
        ("AMPA_slow", 0.0, 0.5, 3.0, "dend"),
        ("AMPA_slow", 0.0, 2.0, 6.3, "dend"),
        ("GABA_A_fast", -60.0, 0.08, 4.8, "soma"),
        ("GABA_A_fast", -60.0, 0.18, 0.45, "soma"),
        ("GABA_A_fast", -60.0, 0.28, 8.4, "soma"),
        ("GABA_A_slow", -60.0, 0.11, 9.7, "dend"),
        ("GABA_A_slow", -60.0, 0.25, 7.5, "dend"),
        ("GABA_A_slow", -60.0, 0.287, 2.67, "dend"),
        ("GABA_A_slow", -60.0, 0.432, 4.49, "dend"),
        ("GABA_A_slow", -60.0, 0.432, 4.49, "soma"),
        ("GABA_A_slow", -60.0, 0.728, 20.2, "dend"),
        ("GABA_A_slow", -60.0, 1.0, 8.0, "soma"),
        ("GABA_A_slow", -60.0, 2.9, 3.1, "dend"),
        ("GABA_B", -90.0, 180.0, 200.0, "dend"),
    })
)
_UTILITY_WEIGHTED_ASSIGNMENTS: Final[dict[_CompartmentPortKey, PortKey]] = {
    ("AMPA_fast", 0.0, 0.07, 0.2, "dend"): ("AMPA_fast", 0.0, 0.07, 0.2),
    ("AMPA_fast", 0.0, 0.1, 1.5, "dend"): ("AMPA_fast", 0.0, 0.1, 1.5),
    ("AMPA_fast", 0.0, 0.11, 0.25, "dend"): ("AMPA_fast", 0.0, 0.11, 0.25),
    ("AMPA_fast", 0.0, 0.3, 0.6, "dend"): ("AMPA_fast", 0.0, 0.3, 0.6),
    ("AMPA_fast", 0.0, 0.5, 3.0, "dend"): ("AMPA_fast", 0.0, 0.5, 3.0),
    ("AMPA_fast", 0.0, 2.0, 6.3, "dend"): ("AMPA_fast", 0.0, 2.0, 6.3),
    ("AMPA_fast", 0.0, 2.0, 8.0, "dend"): ("AMPA_fast", 0.0, 2.0, 6.3),
    ("AMPA_slow", 0.0, 0.5, 3.0, "dend"): ("AMPA_slow", 0.0, 0.5, 3.0),
    ("AMPA_slow", 0.0, 2.0, 6.3, "dend"): ("AMPA_slow", 0.0, 2.0, 6.3),
    ("GABA_A_fast", -60.0, 0.08, 4.8, "soma"): ("GABA_A_fast", -60.0, 0.08, 4.8),
    ("GABA_A_fast", -60.0, 0.18, 0.45, "soma"): ("GABA_A_fast", -60.0, 0.18, 0.45),
    ("GABA_A_fast", -60.0, 0.28, 8.4, "soma"): ("GABA_A_fast", -60.0, 0.28, 8.4),
    ("GABA_A_fast", -60.0, 0.287, 2.67, "soma"): ("GABA_A_fast", -60.0, 0.08, 4.8),
    ("GABA_A_fast", -60.0, 0.3, 6.2, "soma"): ("GABA_A_fast", -60.0, 0.28, 8.4),
    ("GABA_A_fast", -60.0, 1.0, 8.0, "soma"): ("GABA_A_fast", -60.0, 0.28, 8.4),
    ("GABA_A_slow", -60.0, 0.07, 29.0, "dend"): ("GABA_A_slow", -60.0, 0.728, 20.2),
    ("GABA_A_slow", -60.0, 0.11, 9.7, "dend"): ("GABA_A_slow", -60.0, 0.11, 9.7),
    ("GABA_A_slow", -60.0, 0.13, 11.0, "dend"): ("GABA_A_slow", -60.0, 0.11, 9.7),
    ("GABA_A_slow", -60.0, 0.15, 3.9, "dend"): ("GABA_A_slow", -60.0, 0.432, 4.49),
    ("GABA_A_slow", -60.0, 0.2, 2.0, "dend"): ("GABA_A_slow", -60.0, 0.287, 2.67),
    ("GABA_A_slow", -60.0, 0.2, 4.2, "dend"): ("GABA_A_slow", -60.0, 0.432, 4.49),
    ("GABA_A_slow", -60.0, 0.2, 4.2, "soma"): ("GABA_A_slow", -60.0, 0.432, 4.49),
    ("GABA_A_slow", -60.0, 0.25, 7.5, "dend"): ("GABA_A_slow", -60.0, 0.25, 7.5),
    ("GABA_A_slow", -60.0, 0.287, 2.67, "dend"): ("GABA_A_slow", -60.0, 0.287, 2.67),
    ("GABA_A_slow", -60.0, 0.419, 4.99, "dend"): ("GABA_A_slow", -60.0, 0.432, 4.49),
    ("GABA_A_slow", -60.0, 0.432, 4.49, "dend"): ("GABA_A_slow", -60.0, 0.432, 4.49),
    ("GABA_A_slow", -60.0, 0.432, 4.49, "soma"): ("GABA_A_slow", -60.0, 0.432, 4.49),
    ("GABA_A_slow", -60.0, 0.6, 15.0, "dend"): ("GABA_A_slow", -60.0, 0.728, 20.2),
    ("GABA_A_slow", -60.0, 0.728, 10.0, "dend"): ("GABA_A_slow", -60.0, 0.11, 9.7),
    ("GABA_A_slow", -60.0, 0.728, 20.2, "dend"): ("GABA_A_slow", -60.0, 0.728, 20.2),
    ("GABA_A_slow", -60.0, 1.0, 8.0, "dend"): ("GABA_A_slow", -60.0, 0.11, 9.7),
    ("GABA_A_slow", -60.0, 1.0, 8.0, "soma"): ("GABA_A_slow", -60.0, 1.0, 8.0),
    ("GABA_A_slow", -60.0, 1.1, 11.0, "dend"): ("GABA_A_slow", -60.0, 0.11, 9.7),
    ("GABA_A_slow", -60.0, 1.3, 10.2, "dend"): ("GABA_A_slow", -60.0, 0.11, 9.7),
    ("GABA_A_slow", -60.0, 2.9, 3.1, "dend"): ("GABA_A_slow", -60.0, 2.9, 3.1),
    ("GABA_A_slow", -60.0, 3.1, 42.0, "dend"): ("GABA_A_slow", -60.0, 0.728, 20.2),
    ("GABA_A_slow", -60.0, 8.0, 39.0, "dend"): ("GABA_A_slow", -60.0, 0.728, 20.2),
    ("GABA_A_slow", -60.0, 9.0, 39.0, "dend"): ("GABA_A_slow", -60.0, 0.728, 20.2),
    ("GABA_B", -90.0, 180.0, 200.0, "dend"): ("GABA_B", -90.0, 180.0, 200.0),
}


@dataclass(frozen=True, slots=True)
class UtilityWeightedPortError(ValueError):
    message: str

    @override
    def __str__(self) -> str:
        return self.message


def utility_weighted_representative_port(
    key: PortKey,
    compartment: str | None,
) -> PortKey:
    if compartment is None:
        raise UtilityWeightedPortError(
            "utility_weighted_medoids requires compartment-aware ports"
        )
    try:
        return _UTILITY_WEIGHTED_ASSIGNMENTS[(*key, compartment)]
    except KeyError as exc:
        raise UtilityWeightedPortError(
            "utility_weighted_medoids has no representative for "
            + f"{key[0]} E_rev={key[1]:g} compartment={compartment!r}"
        ) from exc
