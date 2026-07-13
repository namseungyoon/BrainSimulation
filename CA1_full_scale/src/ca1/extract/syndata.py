"""Extract synaptic kinetics from Bezaire ModelDB syndata_###.dat files.

Syndata files contain per-pathway synaptic parameters used in the HOC model.
Two mechanisms are present:

``MyExp2Sid``  (3 numeric params)
    A two-exponential synapse with a single conductance component.
    Parameters: tau_rise [ms], tau_decay [ms], e_rev [mV].

``ExpGABAab``  (6 numeric params)
    Dual-component GABA synapse encoding GABA_A + GABA_B in one entry.
    Parameters: tau_rise_A, tau_decay_A, e_rev_A [mV],
                tau_rise_B, tau_decay_B, e_rev_B [mV].

The variant index encodes the random-seed context used in the SimTracker run;
variants 120 and 137 (used in the Bezaire 2016 paper) differ only in
GABA_A E_rev: -60 mV in syndata_120, -75 mV in syndata_137.  Both are
faithfully transcribed without modification.

Paths are resolved relative to the package so no absolute ``/Users/...``
paths appear anywhere (bug #8 fix).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Default dataset directory resolved relative to this source file.
# src/ca1/extract/ -> src/ca1/ -> src/ -> project root -> bezaire_modeldb/datasets/
# ---------------------------------------------------------------------------
_DEFAULT_SYNDATA_DIR = (
    Path(__file__).resolve().parent  # .../src/ca1/extract
    / ".."                           # .../src/ca1
    / ".."                           # .../src
    / ".."                           # project root
    / "bezaire_modeldb"
    / "datasets"
)


def _default_syndata_path(variant: int) -> Path:
    return (_DEFAULT_SYNDATA_DIR / f"syndata_{variant}.dat").resolve()


# ---------------------------------------------------------------------------
# Internal parsing helpers
# ---------------------------------------------------------------------------

def _parse_numeric_fields(tokens: list[str], count: int, line_idx: int) -> list[float]:
    """Parse exactly *count* numeric fields from *tokens*.

    Raises
    ------
    ValueError
        If the token count does not match or a token is non-numeric.
    """
    numeric: list[float] = []
    for token in tokens:
        try:
            numeric.append(float(token))
        except ValueError as err:
            raise ValueError(
                f"Line {line_idx}: non-numeric field {token!r}"
            ) from err
    if len(numeric) != count:
        raise ValueError(
            f"Line {line_idx}: expected {count} numeric values, "
            f"found {len(numeric)}"
        )
    return numeric


def _parse_myexp2sid(numeric: list[float]) -> dict[str, float]:
    return {
        "tau_rise_ms":  numeric[0],
        "tau_decay_ms": numeric[1],
        "e_rev_mV":     numeric[2],
    }


def _parse_expgabaab(numeric: list[float]) -> dict[str, float]:
    return {
        "tau_rise_A_ms":  numeric[0],
        "tau_decay_A_ms": numeric[1],
        "e_rev_A_mV":     numeric[2],
        "tau_rise_B_ms":  numeric[3],
        "tau_decay_B_ms": numeric[4],
        "e_rev_B_mV":     numeric[5],
    }


_MECHANISM_PARSERS: dict[str, tuple[int, Any]] = {
    "MyExp2Sid":  (3, _parse_myexp2sid),
    "ExpGABAab":  (6, _parse_expgabaab),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_syndata(
    dat_path: Optional[Path | str] = None,
    out_path: Optional[Path | str] = None,
    *,
    variant: int = 120,
) -> dict:
    """Parse a syndata_###.dat file and return structured kinetics.

    Parameters
    ----------
    dat_path:
        Explicit path to the ``.dat`` file.  When ``None`` the variant is
        looked up in the package-bundled datasets directory.
    out_path:
        If given, write the parsed result as JSON to this path.
    variant:
        Numeric syndata variant (used only when *dat_path* is ``None``).
        Default is 120 (GABA_A E_rev = -60 mV; used in Bezaire 2016 main
        figures).  Variant 137 has GABA_A E_rev = -75 mV.

    Returns
    -------
    dict
        ``{
            "variant": int,
            "source_file": str,
            "n_entries": int,
            "entries": [ {
                "postsynaptic": str,
                "presynaptic":  str,
                "mechanism":    "MyExp2Sid" | "ExpGABAab",
                "section_list": str,
                "distance_conditions": [str, str],
                "parameters": { ... },
            }, ... ]
        }``

    Raises
    ------
    FileNotFoundError
        If the resolved path does not exist.
    ValueError
        If the declared row count does not match the parsed rows, or a line
        is malformed.
    """
    if dat_path is None:
        resolved = _default_syndata_path(variant)
    else:
        resolved = Path(dat_path).resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"syndata file not found: {resolved}")

    # Infer variant from filename stem when dat_path is explicit
    stem = resolved.stem  # e.g. "syndata_120"
    parts = stem.split("_")
    file_variant: int = int(parts[-1]) if len(parts) >= 2 and parts[-1].isdigit() else variant

    entries: list[dict] = []

    with resolved.open("r", encoding="ascii") as handle:
        header = handle.readline().strip()
        try:
            expected_rows = int(header)
        except ValueError as err:
            raise ValueError(
                f"First line of {resolved} must be row count, got {header!r}"
            ) from err

        for line_idx, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            tokens = stripped.split()
            if len(tokens) < 7:
                raise ValueError(
                    f"{resolved}:{line_idx}: expected at least 7 fields, "
                    f"got {len(tokens)}: {line!r}"
                )

            post_cell    = tokens[0]
            pre_cell     = tokens[1]
            mechanism    = tokens[2]
            section_list = tokens[3]
            dist_cond    = [tokens[4], tokens[5]]
            raw_numeric  = tokens[6:]

            if mechanism not in _MECHANISM_PARSERS:
                raise ValueError(
                    f"{resolved}:{line_idx}: unknown mechanism {mechanism!r}; "
                    f"expected one of {list(_MECHANISM_PARSERS)}"
                )

            expected_count, parser_fn = _MECHANISM_PARSERS[mechanism]
            # Filter out empty trailing tokens (trailing whitespace produces
            # empty strings after split -- already handled by split() but
            # filter defensively)
            numeric_tokens = [t for t in raw_numeric if t]
            numeric = _parse_numeric_fields(
                numeric_tokens, expected_count, line_idx
            )
            parameters = parser_fn(numeric)

            entries.append({
                "postsynaptic":       post_cell,
                "presynaptic":        pre_cell,
                "mechanism":          mechanism,
                "section_list":       section_list,
                "distance_conditions": dist_cond,
                "parameters":         parameters,
            })

    if len(entries) != expected_rows:
        raise ValueError(
            f"{resolved}: row count mismatch -- header declares "
            f"{expected_rows} rows but {len(entries)} were parsed."
        )

    result: dict = {
        "variant":     file_variant,
        "source_file": str(resolved),
        "n_entries":   len(entries),
        "entries":     entries,
    }

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result
