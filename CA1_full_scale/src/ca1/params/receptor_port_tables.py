from __future__ import annotations

from typing import Final, TypeAlias

PortKey: TypeAlias = tuple[str, float, float, float]

PORT_REPLACEMENTS: Final[dict[PortKey, PortKey]] = {
    ("AMPA_fast", 0.0, 2.0, 8.0): ("AMPA_fast", 0.0, 2.0, 6.3),
    ("GABA_A_fast", -60.0, 1.0, 8.0): ("GABA_A_fast", -60.0, 0.28, 8.4),
    ("GABA_A_slow", -60.0, 3.1, 42.0): ("GABA_A_slow", -60.0, 8.0, 39.0),
    ("GABA_A_slow", -60.0, 9.0, 39.0): ("GABA_A_slow", -60.0, 8.0, 39.0),
}
VARIANT_PORT_REPLACEMENTS: Final[dict[int, dict[PortKey, PortKey]]] = {
    137: {
        ("GABA_A_fast", -60.0, 0.08, 4.8): (
            "GABA_A_fast", -60.0, 0.3, 6.2
        ),
    },
}
VARIANT_COMPARTMENT_AWARE_PORT_REPLACEMENTS: Final[
    dict[int, dict[PortKey, PortKey]]
] = {
    137: {
        ("GABA_A_fast", -60.0, 0.3, 6.2): (
            "GABA_A_fast", -60.0, 0.287, 2.67
        ),
    },
}
COMPARTMENT_AWARE_PORT_REPLACEMENTS: Final[dict[PortKey, PortKey]] = {
    ("AMPA_fast", 0.0, 0.1, 1.5): ("AMPA_fast", 0.0, 0.3, 0.6),
    ("AMPA_fast", 0.0, 0.11, 0.25): ("AMPA_fast", 0.0, 0.07, 0.2),
    ("GABA_A_fast", -60.0, 0.08, 4.8): ("GABA_A_fast", -60.0, 0.287, 2.67),
}
REPRESENTATIVE_PORTS: Final[frozenset[PortKey]] = frozenset({
    ("AMPA_fast", 0.0, 0.07, 0.2),
    ("AMPA_fast", 0.0, 0.11, 0.25),
    ("AMPA_fast", 0.0, 0.1, 1.5),
    ("AMPA_fast", 0.0, 0.3, 0.6),
    ("AMPA_fast", 0.0, 0.5, 3.0),
    ("AMPA_fast", 0.0, 2.0, 6.3),
    ("AMPA_slow", 0.0, 0.5, 3.0),
    ("AMPA_slow", 0.0, 2.0, 6.3),
    ("GABA_A_fast", -60.0, 0.08, 4.8),
    ("GABA_A_fast", -60.0, 0.287, 2.67),
    ("GABA_A_fast", -60.0, 0.28, 8.4),
    ("GABA_A_fast", -60.0, 0.3, 6.2),
    ("GABA_A_slow", -60.0, 0.11, 9.7),
    ("GABA_A_slow", -60.0, 0.25, 7.5),
    ("GABA_A_slow", -60.0, 0.287, 2.67),
    ("GABA_A_slow", -60.0, 0.432, 4.49),
    ("GABA_A_slow", -60.0, 1.0, 8.0),
    ("GABA_A_slow", -60.0, 2.9, 3.1),
    ("GABA_A_slow", -60.0, 8.0, 39.0),
    ("GABA_A_slow", -75.0, 9.0, 39.0),
    ("GABA_B", -90.0, 180.0, 200.0),
})
FAST_BASKET_BISTRATIFIED_PORT: Final[PortKey] = (
    "GABA_A_fast", -60.0, 0.18, 0.45
)
FAST_BASKET_DROPPED_PORT: Final[PortKey] = (
    "GABA_A_slow", -60.0, 0.25, 7.5
)
