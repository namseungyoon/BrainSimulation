from __future__ import annotations

from collections.abc import Mapping, Sequence

from .receptors import receptor_prefix

_DECLARED_RECEPTOR_ALIASES: dict[str, str] = {
    "GABA_fast": "GABA_A_fast",
    "GABA_slow": "GABA_A_slow",
}


def _canonical_receptor_class(receptor: str) -> str:
    prefix = receptor_prefix(receptor)
    return _DECLARED_RECEPTOR_ALIASES.get(prefix, prefix)


def _declared_receptor_class(row: Mapping[str, object]) -> str | None:
    raw = row.get("receptor")
    if raw is None:
        return None
    declared = str(raw).strip()
    if declared == "":
        return None
    return _canonical_receptor_class(declared)


def assert_declared_receptor_matches(
    row: Mapping[str, object],
    *,
    pre: str,
    post: str,
    derived_receptors: Sequence[str],
) -> None:
    declared = _declared_receptor_class(row)
    if declared is None:
        return

    derived = {
        _canonical_receptor_class(receptor)
        for receptor in derived_receptors
    }
    if declared in derived:
        return

    raise ValueError(
        f"declared receptor {declared!r} for {pre}->{post} does not match derived receptor classes {sorted(derived)!r}"
    )
