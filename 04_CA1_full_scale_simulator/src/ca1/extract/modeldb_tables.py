"""Extract population sizes and connectivity from Bezaire ModelDB SimTracker files.

Parses ``cellnumbers_<index>.dat`` and ``conndata_<index>.dat`` from the
``bezaire_modeldb/datasets/`` directory and returns structured dicts suitable
for downstream consumption by ``ca1.params.synapses`` and the BSB builder.

Column labelling note
---------------------
Some ModelDB conndata files use col4 as a network-wide total, while later
paper-view files such as ``conndata_430.dat`` use it as per-cell convergence.
The caller must pass ``count_mode`` explicitly.  For ``network_total`` files,
the true per-cell convergence (indegree) is therefore::

    indegree = total_connections / N_post

For ``per_cell`` files, col4 is already the convergence and must not be
divided again.  Silent auto-detection is intentionally avoided because the two
formats can coexist in the same archive.

Afferent synapse budgets
------------------------
For rows where the presynaptic population is CA3 or ECIII, ModelDB first picks
the conndata-derived source-contact convergence and then creates
``synapses_per_connection`` synapses for each contact.  The returned
``synapses_per_cell`` is that full synapse budget, and remains uncapped by the
presynaptic population size (see ``ca1.types.Afferent``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from ca1.extract.connectivity import canonical_name, is_afferent

ConndataCountMode = Literal["network_total", "per_cell"]
_PER_CELL_CONNDATA_INDICES = frozenset({430})

# ---------------------------------------------------------------------------
# Default dataset directory resolved relative to this file so no hardcoded
# absolute paths (bug #8 fix).  The bezaire_modeldb/datasets directory sits
# three levels above src/ca1/extract/.
# ---------------------------------------------------------------------------
_DEFAULT_MODELDB_DIR = (
    Path(__file__).resolve().parent  # .../src/ca1/extract
    / ".."                           # .../src/ca1
    / ".."                           # .../src
    / ".."                           # .../  (project root)
    / "bezaire_modeldb"
    / "datasets"
)


def _resolved_modeldb_dir(modeldb_dir: Optional[Path | str]) -> Path:
    """Return a resolved Path for the datasets directory."""
    if modeldb_dir is None:
        return _DEFAULT_MODELDB_DIR.resolve()
    return Path(modeldb_dir).resolve()


def _validate_count_mode_for_index(index: int, count_mode: ConndataCountMode) -> None:
    if index in _PER_CELL_CONNDATA_INDICES and count_mode != "per_cell":
        raise ValueError(
            f"conndata_{index} is a per-cell convergence table; pass "
            "count_mode='per_cell' instead of silently treating column 4 as "
            f"{count_mode!r}"
        )


# ---------------------------------------------------------------------------
# Low-level parsers (adapted from parse_modeldb_tables.py)
# ---------------------------------------------------------------------------

def _read_with_count(path: Path, column_names: list[str]) -> pd.DataFrame:
    """Read a whitespace-delimited table whose first line is the row count.

    The row count is validated against the number of rows actually read so
    truncated or malformed files are caught early.
    """
    with path.open("r", encoding="ascii") as handle:
        first_line = handle.readline().strip()
    try:
        expected_rows = int(first_line)
    except ValueError as err:
        raise ValueError(
            f"First line of {path} must be an integer row count, "
            f"found: {first_line!r}"
        ) from err

    df = pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=1,
        names=column_names,
        nrows=expected_rows,
        engine="python",
    )

    if len(df) != expected_rows:
        raise ValueError(
            f"File {path} declares {expected_rows} rows but {len(df)} were "
            "read. Check for truncated or malformed data."
        )
    return df


def _parse_cellnumbers_df(path: Path) -> pd.DataFrame:
    """Return a DataFrame with columns: cell_group, mechanism, count,
    gid_flag, external_flag."""
    columns = ["cell_group", "mechanism", "count", "gid_flag", "external_flag"]
    df = _read_with_count(path, columns)
    df["count"] = df["count"].astype(int)
    df["gid_flag"] = df["gid_flag"].astype(int)
    df["external_flag"] = df["external_flag"].astype(int)
    return df


def _parse_conndata_df(path: Path) -> pd.DataFrame:
    """Return a DataFrame with columns: pre, post, weight_uS,
    total_connections, synapses_per_connection."""
    columns = [
        "pre",
        "post",
        "weight_uS",
        "total_connections",    # network-wide total (NOT per-cell convergence)
        "synapses_per_connection",
    ]
    df = _read_with_count(path, columns)
    df["weight_uS"] = df["weight_uS"].astype(float)
    df["total_connections"] = df["total_connections"].astype(float)
    df["synapses_per_connection"] = df["synapses_per_connection"].astype(float)
    return df


def _receptor_for_pre(pre: str) -> str:
    if pre == "ECIII":
        return "AMPA_slow"
    if pre in {"CA3", "Pyramidal"}:
        return "AMPA_fast"
    if pre in {"PV_Basket", "Axo"}:
        return "GABA_A_fast"
    return "GABA_A_slow"


# ---------------------------------------------------------------------------
# Public extraction functions
# ---------------------------------------------------------------------------

def extract_cellnumbers(
    modeldb_dir: Optional[Path | str] = None,
    index: int = 101,
) -> dict[str, int]:
    """Return a dict mapping canonical population name -> cell count.

    Reads ``cellnumbers_<index>.dat`` from *modeldb_dir* (defaults to the
    project-bundled ``bezaire_modeldb/datasets/`` directory).

    Parameters
    ----------
    modeldb_dir:
        Path to the datasets directory.  ``None`` uses the package default.
    index:
        SimTracker configuration index (default 101 = full-scale Bezaire).

    Returns
    -------
    dict[str, int]
        ``{"Pyramidal": 311500, "PV_Basket": 5530, ...}`` – all populations
        including afferent sources (CA3, ECIII).

    Raises
    ------
    FileNotFoundError
        If the cellnumbers file does not exist.
    ValueError
        If the declared row count does not match the data.
    KeyError
        If any HOC cell-group name is not in the canonical alias table.
    """
    datasets = _resolved_modeldb_dir(modeldb_dir)
    path = datasets / f"cellnumbers_{index}.dat"
    if not path.exists():
        raise FileNotFoundError(f"cellnumbers file not found: {path}")

    df = _parse_cellnumbers_df(path)

    result: dict[str, int] = {}
    for _, row in df.iterrows():
        hoc_name = str(row["cell_group"])
        canonical = canonical_name(hoc_name)  # raises KeyError on unknown
        result[canonical] = int(row["count"])
    return result


def extract_connectivity(
    modeldb_dir: Optional[Path | str] = None,
    index: int = 101,
    cellnumbers_index: Optional[int] = None,
    out_path: Optional[Path | str] = None,
    count_mode: ConndataCountMode = "network_total",
) -> dict[str, object]:
    """Extract full connectivity table from a ModelDB conndata file.

    Returns a structured dict with three sections:

    ``"projections"``
        Intra-CA1 recurrent connections (pre and post both in
        :data:`~ca1.extract.connectivity.CA1_INTERNAL_TYPES`).  Each entry
        carries ``indegree``, ``weight_nS``,
        ``synapses_per_connection``, and count provenance.

    ``"afferents"``
        Rows where the presynaptic type is CA3 or ECIII.  ``synapses_per_cell``
        is the per-postsynaptic-cell synapse budget, kept uncapped (see
        module docstring).

    ``"populations"``
        The raw population counts from the companion cellnumbers file.

    Parameters
    ----------
    modeldb_dir:
        Path to the datasets directory.  ``None`` uses the package default.
    index:
        SimTracker conndata index (default 101).
    cellnumbers_index:
        SimTracker cellnumbers index.  Defaults to ``index`` for legacy paired
        extraction.  Paper Table 1 full-scale extraction uses ``index=430``
        with ``cellnumbers_index=101`` and ``count_mode="per_cell"``;
        ``ConnData=211`` is the ModelDB launcher default, not the final-tier
        paper Table 1 gate.
    out_path:
        If given, write the result as JSON to this path.
    count_mode:
        Meaning of column 4 in ``conndata_<index>.dat``. ``network_total``
        divides by the postsynaptic population. ``per_cell`` treats col4 as
        already-normalized convergence.

    Raises
    ------
    FileNotFoundError
        If either required file is missing.
    ValueError
        If row-count validation fails.
    KeyError
        If any HOC name is absent from the canonical alias table.
    """
    if count_mode not in ("network_total", "per_cell"):
        raise ValueError(
            "count_mode must be 'network_total' or 'per_cell', "
            f"got {count_mode!r}"
        )
    _validate_count_mode_for_index(index, count_mode)

    datasets = _resolved_modeldb_dir(modeldb_dir)
    population_index = index if cellnumbers_index is None else cellnumbers_index

    conndata_path = datasets / f"conndata_{index}.dat"
    cellnumbers_path = datasets / f"cellnumbers_{population_index}.dat"

    for p in (conndata_path, cellnumbers_path):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    # Parse population sizes (validates row count + name mapping)
    populations = extract_cellnumbers(modeldb_dir, population_index)

    # Parse raw connection table
    conn_df = _parse_conndata_df(conndata_path)

    projections: dict[str, dict[str, object]] = {}
    afferents: dict[str, dict[str, object]] = {}

    for _, row in conn_df.iterrows():
        pre_hoc = str(row["pre"])
        post_hoc = str(row["post"])
        weight_uS: float = float(row["weight_uS"])
        count_column: float = float(row["total_connections"])
        synapses_per_conn: int = int(round(float(row["synapses_per_connection"])))

        if count_column == 0.0 or weight_uS == 0.0 or synapses_per_conn == 0:
            continue

        pre_canonical = canonical_name(pre_hoc)
        post_canonical = canonical_name(post_hoc)

        n_post = populations.get(post_canonical, 0)
        if n_post == 0:
            raise KeyError(
                f"Postsynaptic population {post_canonical!r} has count 0 "
                "or is absent from cellnumbers; cannot compute indegree."
            )

        weight_nS = weight_uS * 1000.0

        if count_mode == "network_total":
            indegree = count_column / n_post
            estimated_total_connections = count_column
        else:
            indegree = count_column
            estimated_total_connections = count_column * n_post

        if is_afferent(pre_canonical):
            synapses_per_cell = indegree * synapses_per_conn
            key = f"{pre_canonical}_to_{post_canonical}"
            afferents[key] = {
                "presynaptic": pre_canonical,
                "postsynaptic": post_canonical,
                "weight_nS": weight_nS,
                "synapses_per_cell": synapses_per_cell,
                "synapses_per_connection": synapses_per_conn,
                "connection_count_column": count_column,
                "estimated_total_connections": int(round(estimated_total_connections)),
                "conndata_count_mode": count_mode,
                "post_population": n_post,
                "receptor": _receptor_for_pre(pre_canonical),
            }
        else:
            # Intra-CA1 recurrent projection
            key = f"{pre_canonical}_to_{post_canonical}"
            projections[key] = {
                "presynaptic": pre_canonical,
                "postsynaptic": post_canonical,
                "weight_nS": weight_nS,
                "indegree": indegree,
                "synapses_per_connection": synapses_per_conn,
                "connection_count_column": count_column,
                "estimated_total_connections": int(round(estimated_total_connections)),
                "conndata_count_mode": count_mode,
                "post_population": n_post,
                "receptor": _receptor_for_pre(pre_canonical),
            }

    result: dict[str, object] = {
        "index": index,
        "cellnumbers_index": population_index,
        "conndata_count_mode": count_mode,
        "populations": populations,
        "projections": projections,
        "afferents": afferents,
        "stats": {
            "n_projections": len(projections),
            "n_afferents": len(afferents),
        },
    }

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result
