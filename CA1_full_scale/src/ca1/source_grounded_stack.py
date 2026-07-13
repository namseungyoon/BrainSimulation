"""Opt-in deployment of source-grounded, rate-untuned candidate overlays.

The checked-in candidate JSON files remain immutable evidence artifacts.  This
module applies their validated weights, domains, and intrinsic fields to an
in-memory ``NetworkSpec`` only; it never rewrites canonical parameter files.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

from ca1.types import Afferent, NetworkSpec, Projection


_ROOT = Path(__file__).resolve().parents[2]
_DOMAIN_CODE = {"soma": 0.0, "proximal": 1.0, "distal": 2.0}
_STACK_FIELDS = frozenset(
    {"refit_candidate", "refit_cells", "gaba_into_cck_candidate"}
)


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    parsed: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{context} keys must be strings")
        parsed[key] = item
    return parsed


def _path(value: object, *, field: str, config_dir: Path) -> Path:
    if not isinstance(value, str):
        raise TypeError(f"source_grounded_stack.{field} must be a path string")
    configured = Path(value)
    candidates = (
        (config_dir / configured).resolve(),
        (_ROOT / configured).resolve(),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"source_grounded_stack.{field} does not exist: {configured}"
    )


def _load_json(path: Path) -> dict[str, Any]:
    raw = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        raise TypeError(f"source-grounded artifact must be a mapping: {path}")
    return {str(key): value for key, value in cast(dict[object, Any], raw).items()}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_common_provenance(report: Mapping[str, Any], path: Path) -> None:
    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping):
        raise TypeError(f"source-grounded artifact provenance missing: {path}")
    if provenance.get("candidate_only") is not True:
        raise ValueError(f"source-grounded artifact is not candidate-only: {path}")
    if provenance.get("table5_rate_tuning") is not False:
        raise ValueError(f"Table-5-tuned source-grounded artifact refused: {path}")


def _validate_deployed_hashes(report: Mapping[str, Any], path: Path) -> None:
    expected = report.get("immutable_deployed_file_sha256")
    if not isinstance(expected, Mapping) or not expected:
        raise ValueError(f"immutable deployed-file hashes missing: {path}")
    mismatches: dict[str, tuple[object, str | None]] = {}
    for raw_name, raw_digest in expected.items():
        name = str(raw_name)
        deployed = (_ROOT / name).resolve()
        actual = _sha256(deployed) if deployed.is_file() else None
        if actual != raw_digest:
            mismatches[name] = (raw_digest, actual)
    if mismatches:
        raise RuntimeError(
            f"deployed files differ from source-grounded snapshot: {mismatches}"
        )


def _float_mapping(value: object, context: str) -> dict[str, float]:
    raw = _mapping(value, context)
    parsed: dict[str, float] = {}
    for key, item in raw.items():
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise TypeError(f"{context}.{key} must be numeric")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{context}.{key} must be finite")
        parsed[key] = number
    return parsed


def _refit_cells(value: object, spec: NetworkSpec) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError("source_grounded_stack.refit_cells must be a list of cell names")
    cells = tuple(cast(list[str], value))
    if len(cells) != len(set(cells)):
        raise ValueError("source_grounded_stack.refit_cells contains duplicates")
    unknown = sorted(set(cells) - set(spec.cell_types))
    if unknown:
        raise ValueError(f"source-grounded refit contains unknown cells: {unknown}")
    return cells


def _set_domain(
    overrides: dict[str, dict[str, float]],
    *,
    post: str,
    receptor: str,
    domain: object,
) -> None:
    try:
        code = _DOMAIN_CODE[str(domain)]
    except KeyError as exc:
        raise ValueError(f"unknown source-grounded domain: {domain!r}") from exc
    previous = overrides.setdefault(post, {}).get(receptor)
    if previous is not None and previous != code:
        raise ValueError(
            f"conflicting source-grounded domains for {post}:{receptor}: "
            f"{previous} versus {code}"
        )
    overrides[post][receptor] = code


def _apply_refit(
    spec: NetworkSpec,
    *,
    path: Path,
    cells: tuple[str, ...],
    status_overrides: dict[str, dict[str, float]],
    compartment_overrides: dict[str, dict[str, float]],
    provenance: dict[str, str],
) -> NetworkSpec:
    report = _load_json(path)
    if report.get("schema") != "cck-sca-source-grounded-refit-candidate/v1":
        raise ValueError(f"unsupported source-grounded refit schema: {path}")
    _validate_common_provenance(report, path)
    _validate_deployed_hashes(report, path)

    intrinsic = _mapping(report.get("intrinsic"), "refit intrinsic")
    intrinsic_cells = _mapping(intrinsic.get("cells"), "refit intrinsic.cells")
    for cell in cells:
        record = _mapping(intrinsic_cells.get(cell), f"refit intrinsic.cells.{cell}")
        gate = _mapping(record.get("held_out_gate"), f"refit held-out gate {cell}")
        if gate.get("passed") is not True:
            raise ValueError(f"source-grounded intrinsic held-out gate failed: {cell}")
        status_overrides[cell] = _float_mapping(
            record.get("fitted_params"), f"refit fitted_params {cell}"
        )

    transfer = _mapping(report.get("excitatory_transfer"), "refit transfer")
    raw_rows = transfer.get("rows")
    if not isinstance(raw_rows, list):
        raise TypeError("refit excitatory_transfer.rows must be a list")
    selected: dict[tuple[str, str], dict[str, object]] = {}
    for raw_row in raw_rows:
        row = _mapping(raw_row, "refit transfer row")
        contract = _mapping(row.get("contract"), "refit transfer contract")
        post = str(contract.get("post"))
        if post not in cells:
            continue
        key = (str(contract.get("pre")), post)
        if key in selected:
            raise ValueError(f"duplicate source-grounded refit row: {key}")
        candidate_mapping = _mapping(
            row.get("candidate_mapping"), "refit candidate mapping"
        )
        selected[key] = {"contract": contract, "mapping": candidate_mapping}

    matched: set[tuple[str, str]] = set()

    def weight(pre: str, post: str, receptor: str, deployed: float) -> float:
        key = (pre, post)
        row = selected.get(key)
        if row is None:
            return deployed
        contract = cast(dict[str, object], row["contract"])
        if str(contract.get("receptor")) != receptor:
            raise ValueError(f"source-grounded refit receptor mismatch: {pre}->{post}")
        expected = float(cast(int | float, contract["deployed_gmax_nS"]))
        if not math.isclose(expected, deployed, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"source-grounded refit deployed-weight mismatch: {pre}->{post} "
                f"expected={expected} actual={deployed}"
            )
        mapping = cast(dict[str, object], row["mapping"])
        _set_domain(
            compartment_overrides,
            post=post,
            receptor=receptor,
            domain=mapping.get("domain"),
        )
        matched.add(key)
        return float(cast(int | float, mapping["transferred_gmax_nS"]))

    projections: list[Projection] = [
        replace(row, weight_nS=weight(row.pre, row.post, row.receptor, row.weight_nS))
        for row in spec.projections
    ]
    afferents: list[Afferent] = []
    for row in spec.afferents:
        pre = row.name.split("_to_", maxsplit=1)[0]
        afferents.append(
            replace(
                row,
                weight_nS=weight(pre, row.post, row.receptor, row.weight_nS),
            )
        )
    missing = sorted(set(selected) - matched)
    if missing:
        raise ValueError(f"source-grounded refit rows absent from network: {missing}")

    provenance["source_grounded.refit"] = (
        "cck-sca-source-grounded-refit-v1;"
        f"sha256={_sha256(path)};cells={','.join(cells)};"
        "source-response-only;table5-rate-tuning=false"
    )
    return replace(spec, projections=projections, afferents=afferents)


def _apply_gaba_into_cck(
    spec: NetworkSpec,
    *,
    path: Path,
    compartment_overrides: dict[str, dict[str, float]],
    provenance: dict[str, str],
) -> NetworkSpec:
    report = _load_json(path)
    if report.get("schema") != "gaba-into-cck-sca-transfer-candidate/v1":
        raise ValueError(f"unsupported GABA-into-CCK schema: {path}")
    _validate_common_provenance(report, path)
    raw_rows = report.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != 5:
        raise ValueError("GABA-into-CCK overlay requires exactly five rows")
    rows: dict[str, dict[str, object]] = {}
    for raw_row in raw_rows:
        row = _mapping(raw_row, "GABA-into-CCK row")
        key = str(row.get("row_key"))
        stability = row.get("stability_gate")
        if (
            key in rows
            or row.get("post") != "CCK_Basket"
            or row.get("source_response_gate_pass") is not True
            or not isinstance(stability, list)
            or not stability
            or any(
                not isinstance(item, Mapping) or item.get("gate_pass") is not True
                for item in stability
            )
        ):
            raise ValueError(f"invalid or unstable GABA-into-CCK row: {key}")
        rows[key] = row

    matched: set[str] = set()
    projections: list[Projection] = []
    for projection in spec.projections:
        key = f"{projection.pre}->{projection.post}|{projection.receptor}"
        row = rows.get(key)
        if row is None:
            projections.append(projection)
            continue
        if str(row.get("deployed_receptor")) != projection.receptor:
            raise ValueError(f"GABA-into-CCK receptor mismatch: {key}")
        if int(cast(int | float, row["source_contacts"])) != (
            projection.synapses_per_connection
        ):
            raise ValueError(f"GABA-into-CCK contact mismatch: {key}")
        _set_domain(
            compartment_overrides,
            post=projection.post,
            receptor=projection.receptor,
            domain=row.get("domain"),
        )
        projections.append(
            replace(
                projection,
                weight_nS=float(cast(int | float, row["transferred_gmax_nS"])),
            )
        )
        matched.add(key)
    missing = sorted(set(rows) - matched)
    if missing:
        raise ValueError(f"GABA-into-CCK rows absent from full network: {missing}")

    provenance["source_grounded.gaba_into_cck"] = (
        "gaba-into-cck-source-grounded-transfer-v1;"
        f"sha256={_sha256(path)};rows=5;all-source-response-gates-passed;"
        "table5-rate-tuning=false"
    )
    return replace(spec, projections=projections)


def apply_source_grounded_stack(
    spec: NetworkSpec,
    value: object,
    *,
    config_dir: Path,
) -> NetworkSpec:
    """Apply an explicit source-grounded stack block to ``spec`` in memory."""
    if value is None:
        return spec
    config = _mapping(value, "source_grounded_stack")
    unknown = sorted(set(config) - _STACK_FIELDS)
    if unknown:
        raise ValueError(f"source_grounded_stack has unknown fields: {unknown}")

    cells = _refit_cells(config.get("refit_cells", []), spec)
    if cells and "refit_candidate" not in config:
        raise ValueError("source_grounded_stack.refit_candidate is required for refit_cells")
    if "refit_candidate" in config and not cells:
        raise ValueError("source_grounded_stack.refit_cells cannot be empty")

    status_overrides = {
        cell: dict(values) for cell, values in spec.aglif_status_overrides.items()
    }
    compartment_overrides = {
        cell: dict(values)
        for cell, values in spec.aglif_compartment_overrides.items()
    }
    provenance = dict(spec.source_grounded_stack_provenance)
    updated = spec
    if cells:
        updated = _apply_refit(
            updated,
            path=_path(
                config["refit_candidate"],
                field="refit_candidate",
                config_dir=config_dir,
            ),
            cells=cells,
            status_overrides=status_overrides,
            compartment_overrides=compartment_overrides,
            provenance=provenance,
        )
    if "gaba_into_cck_candidate" in config:
        updated = _apply_gaba_into_cck(
            updated,
            path=_path(
                config["gaba_into_cck_candidate"],
                field="gaba_into_cck_candidate",
                config_dir=config_dir,
            ),
            compartment_overrides=compartment_overrides,
            provenance=provenance,
        )
    return replace(
        updated,
        aglif_status_overrides=status_overrides,
        aglif_compartment_overrides=compartment_overrides,
        source_grounded_stack_provenance=provenance,
    )
