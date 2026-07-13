#!/usr/bin/env python3
"""Small source-faithful closed loop for the five diagnosed interneuron types.

The mutually simulated populations are PV_Basket, Bistratified, O_LM,
CCK_Basket, and SCA.  CA3/ECIII retain their literal source graphs and 0.65 Hz
arrhythmic drive.  Pyramidal is a driven boundary population: this harness uses
the same independent held-out 1 Hz proxy as the converging-barrage diagnosis,
not recorded full-run spikes.

This working-point/recruitment test deliberately uses the ModelDB uniform
fastconn topology, not the 3-D Gaussian topology.  The topology A/B established
that 3-D and uniform preserve the same K and J and give the same rates; phase
locality is irrelevant here because this is not a theta-phase test.  Uniform is
therefore faithful to the decisive measure and avoids the 3-D spatial-extent
candidate shortage after downscaling.  Scaling always preserves in-degree
(never connection probability), and every uniform source window is audited to
contain the requested immutable K without clipping.

All graph construction and connection calls go through NestGpuBackend.build.
The only post-build operation is an arm-local SetStatus of candidate intrinsic
parameters.  Candidate transfer mappings are applied to an in-memory copy of
the NetworkSpec and a temporary copy of the source-location table.  Deployed
files are checked by SHA256 before and after every run.

The default invocation runs arms A/E serially in fresh ordinary processes;
NEST-GPU is never imported by --preflight and MPI is rejected explicitly.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from ca1.config import build_network_spec
from ca1.aglif_overrides import AglifDendOverride
from ca1.build.downscale import downscale_spec
from ca1.params.provenance import parameter_provenance_for_spec
from ca1.sim.aglif_dend import aglif_dend_compartments, aglif_dend_status
from ca1.sim.edge_artifact import _literal_source_indegree
from ca1.sim.gpu_backend import (
    NestGpuBackend,
    _projection_syn_spec,
    _record_spike_buffer_size,
    _required_dendritic_ports,
)
from ca1.sim.modeldb_topology import binned_fixed_indegree_connections
from ca1.types import NetworkSpec, SimMeta


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/full_scale_3dtopo.yaml"
DEFAULT_CANDIDATE = ROOT / "results/cck_sca_refit_candidate.json"
DEFAULT_GABA_CANDIDATE = ROOT / "results/gaba_into_cck_candidate.json"
DEFAULT_OUTPUT = ROOT / "scratchpad/closedloop_result.json"
LOOP_TYPES = ("PV_Basket", "Bistratified", "O_LM", "CCK_Basket", "SCA")
BOUNDARY_TYPE = "Pyramidal"
# Pyramidal boundary drive (Hz). Default 1.0 = conservative held-out proxy.
# Set CA1_CLOSEDLOOP_PYR_HZ to the observed full-run rate (7.82 Hz) to test the
# working point under the faithful recurrent-excitation boundary condition
# (observed, not rate-tuned to a target).
BOUNDARY_RATE_HZ = float(os.environ.get("CA1_CLOSEDLOOP_PYR_HZ", "1.0"))
ARMS = ("A", "E")
ARM_REFIT_TYPES: Mapping[str, frozenset[str]] = {
    "A": frozenset(),
    # SCA-refit was already wired in the original payoff harness.  Arm E keeps
    # it as the source-grounded SCA choice while assembling the validated model
    # stack requested here.
    "E": frozenset(("SCA",)),
}
ARM_MODELS: Mapping[str, Mapping[str, str]] = {
    "A": {},
    "E": {
        "CCK_Basket": "user_m3",
        "Bistratified": "user_m4",
        "O_LM": "user_m5",
        # PV_Basket deliberately has no override: deployed user_m2 is retained.
    },
}
# At 1/10, every target bin's uniform fastconn source interval contains at
# least the immutable row K.  Smaller tested scales fail that stronger runtime
# condition even when the total presynaptic population is nominally large enough.
MICRO_SCALE = 0.1
MICRO_COUNTS: Mapping[str, int] = {
    "PV_Basket": 553,
    "Bistratified": 221,
    "O_LM": 164,
    "CCK_Basket": 360,
    "SCA": 40,
    "Pyramidal": 31150,
}
DEPLOYED_FILES = (
    ROOT / "src/ca1/params/aglif_parameters_fitted.json",
    ROOT / "src/ca1/params/source_location_transfer_syndata120_budget_weighted.json",
    ROOT / "src/ca1/params/connectivity.json",
    ROOT / "src/ca1/params/syndata_120.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def deployed_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): _sha256(path) for path in DEPLOYED_FILES}


def _load_candidate(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "cck-sca-source-grounded-refit-candidate/v1":
        raise ValueError(f"unsupported candidate schema in {path}")
    provenance = report.get("provenance", {})
    if not provenance.get("candidate_only") or provenance.get("table5_rate_tuning"):
        raise ValueError("candidate must be candidate-only and free of Table-5 tuning")
    expected = report.get("immutable_deployed_file_sha256", {})
    actual = deployed_hashes()
    mismatch = {
        name: (expected.get(name), digest)
        for name, digest in actual.items()
        if expected.get(name) != digest
    }
    if mismatch:
        raise RuntimeError(f"deployed-file hash differs from candidate snapshot: {mismatch}")
    return report


def _load_gaba_candidate(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "gaba-into-cck-sca-transfer-candidate/v1":
        raise ValueError(f"unsupported GABA candidate schema in {path}")
    provenance = report.get("provenance", {})
    if not provenance.get("candidate_only") or provenance.get("table5_rate_tuning"):
        raise ValueError("GABA candidate must be candidate-only and free of Table-5 tuning")
    rows = report.get("rows", [])
    keys = [str(row.get("row_key")) for row in rows]
    if len(rows) != 5 or len(set(keys)) != 5:
        raise ValueError("GABA-into-CCK candidate must contain exactly five unique rows")
    if any(
        row.get("post") != "CCK_Basket"
        or not bool(row.get("source_response_gate_pass"))
        for row in rows
    ):
        raise ValueError("all GABA candidate rows must be source-gated inputs to CCK_Basket")
    return report


def _base_micro_spec(config: Path, seed: int) -> NetworkSpec:
    full = build_network_spec(config, scale=1.0, seed=seed)
    scaled = downscale_spec(full, scale=MICRO_SCALE, mode="preserve-indegree")
    retained = frozenset((*LOOP_TYPES, BOUNDARY_TYPE))
    # The scaler establishes the population sizes through the requested path.
    # Retain canonical full rows verbatim: split-port rows can have fractional
    # per-port K (for example 17.5 + 17.5), and must not be integer-truncated.
    projections = [
        row for row in full.projections
        if row.post in LOOP_TYPES and row.pre in retained
    ]
    afferents = [row for row in full.afferents if row.post in LOOP_TYPES]
    cell_types = {
        name: scaled.cell_types[name]
        for name in retained
    }
    actual_counts = {name: cell_type.count for name, cell_type in cell_types.items()}
    if actual_counts != dict(MICRO_COUNTS):
        raise RuntimeError(
            f"unexpected preserve-indegree micro counts: {actual_counts}"
        )
    target_receptors = {
        name: table for name, table in scaled.target_receptors.items()
        if name in retained
    }
    spec = replace(
        scaled,
        name="ca1_closed_loop_microcircuit",
        cell_types=cell_types,
        projections=projections,
        afferents=afferents,
        target_receptors=target_receptors,
        weight_compensation=1.0,
        working_point_mode="clamp",
        working_point_clamp_rates_hz={BOUNDARY_TYPE: BOUNDARY_RATE_HZ},
    )
    if spec.afferent_topology != "literal_source_graph":
        raise ValueError("closed loop requires literal_source_graph afferents")
    # This is a working-point/recruitment test, not a theta-phase test.  The
    # validated topology A/B gives the same rates for 3-D and uniform when K/J
    # are preserved, while uniform removes downscaled 3-D extent infeasibility.
    return replace(spec, recurrent_topology="modeldb_fastconn_binned")


def _instantiated_counts(spec: NetworkSpec) -> dict[str, int]:
    """Counts already materialized by downscale_spec; do not scale them twice."""
    return {name: int(cell_type.count) for name, cell_type in spec.cell_types.items()}


def _microcircuit_parameter_provenance(
    spec: NetworkSpec,
    config: Path,
    seed: int,
) -> dict[str, str]:
    """Validate fit files over their canonical full-cell domain."""
    standard = build_network_spec(config, scale=1.0, seed=seed)
    provenance_spec = replace(
        spec,
        cell_types={**standard.cell_types, **spec.cell_types},
    )
    provenance = parameter_provenance_for_spec(provenance_spec)
    # The extra standard cell definitions exist only to supply the exact fit-file
    # validation domain; report the network that this harness actually builds.
    provenance["network.total_cells"] = str(spec.total_cells())
    provenance["network.cell_types"] = str(len(spec.cell_types))
    return provenance


def _candidate_rows(candidate: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(record["row"]): record
        for record in candidate["excitatory_transfer"]["rows"]
    }


def _candidate_transfer_spec(
    spec: NetworkSpec,
    candidate: Mapping[str, Any],
    refit_types: frozenset[str],
) -> NetworkSpec:
    rows = _candidate_rows(candidate)

    def weight(pre: str, post: str, deployed: float) -> float:
        if post not in refit_types:
            return deployed
        record = rows.get(f"{pre}->{post}")
        if record is None:
            return deployed
        contract = record["contract"]
        if not np.isclose(float(contract["deployed_gmax_nS"]), deployed, atol=1e-12):
            raise ValueError(f"candidate/deployed transfer mismatch for {pre}->{post}")
        return float(record["candidate_mapping"]["transferred_gmax_nS"])

    projections = [
        replace(row, weight_nS=weight(row.pre, row.post, row.weight_nS))
        for row in spec.projections
    ]
    afferents = [
        replace(
            row,
            weight_nS=weight(row.name.split("_to_", 1)[0], row.post, row.weight_nS),
        )
        for row in spec.afferents
    ]
    return replace(spec, projections=projections, afferents=afferents)


def _gaba_candidate_rows(
    candidate: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    return {str(record["row_key"]): record for record in candidate["rows"]}


def _gaba_candidate_transfer_spec(
    spec: NetworkSpec,
    candidate: Mapping[str, Any],
    enabled: bool,
) -> NetworkSpec:
    if not enabled:
        return spec
    rows = _gaba_candidate_rows(candidate)
    matched: set[str] = set()
    projections = []
    for row in spec.projections:
        key = f"{row.pre}->{row.post}|{row.receptor}"
        record = rows.get(key)
        if record is None:
            projections.append(row)
            continue
        if int(record["source_contacts"]) != row.synapses_per_connection:
            raise ValueError(f"GABA candidate contact contract mismatch for {key}")
        if str(record["deployed_receptor"]) != row.receptor:
            raise ValueError(f"GABA candidate receptor contract mismatch for {key}")
        projections.append(
            replace(row, weight_nS=float(record["transferred_gmax_nS"]))
        )
        matched.add(key)
    # Ivy is intentionally outside this five-population harness.  Every other
    # candidate row must be present; adding Ivy would change the immutable graph
    # retained by the established arms.
    allowed_absent = {
        key for key, row in rows.items() if str(row["pre"]) not in LOOP_TYPES
    }
    missing = set(rows) - matched
    if missing != allowed_absent:
        raise ValueError(f"GABA candidate rows absent from retained graph: {sorted(missing)}")
    return replace(spec, projections=projections)


def _write_candidate_location_table(
    spec: NetworkSpec,
    candidate: Mapping[str, Any],
    refit_types: frozenset[str],
    path: Path,
    gaba_candidate: Mapping[str, Any] | None = None,
) -> NetworkSpec:
    source = Path(spec.source_location_transfer_table)
    table = json.loads(source.read_text(encoding="utf-8"))
    mappings = {
        str(record["row"]): str(record["candidate_mapping"]["domain"])
        for record in candidate["excitatory_transfer"]["rows"]
        if str(record["contract"]["post"]) in refit_types
    }
    seen: set[str] = set()
    for row in table:
        key = f"{row.get('pre')}->{row.get('post')}"
        domain = mappings.get(key)
        if domain is None:
            continue
        row["loc"] = {"soma": "soma", "proximal": "prox", "distal": "dist"}[domain]
        row["aglif_compartment"] = "soma" if domain == "soma" else "dend"
        row["candidate_overlay"] = "cck_sca_refit_candidate.json"
        seen.add(key)
    missing = sorted(set(mappings) - seen)
    if missing:
        raise ValueError(f"candidate location rows absent from deployed table: {missing}")
    if gaba_candidate is not None:
        gaba_mappings = {
            str(record["row_key"]): str(record["domain"])
            for record in gaba_candidate["rows"]
        }
        gaba_seen: set[str] = set()
        for row in table:
            key = f"{row.get('pre')}->{row.get('post')}|{row.get('port')}"
            domain = gaba_mappings.get(key)
            if domain is None:
                continue
            row["loc"] = {
                "soma": "soma", "proximal": "prox", "distal": "dist"
            }[domain]
            row["aglif_compartment"] = "soma" if domain == "soma" else "dend"
            row["candidate_overlay"] = "gaba_into_cck_candidate.json"
            gaba_seen.add(key)
        missing_gaba = sorted(set(gaba_mappings) - gaba_seen)
        if missing_gaba:
            raise ValueError(
                f"GABA candidate location rows absent from deployed table: {missing_gaba}"
            )
    path.write_text(json.dumps(table, separators=(",", ":")), encoding="utf-8")
    return replace(spec, source_location_transfer_table=str(path))


def build_arm_spec(
    config: Path,
    candidate: Mapping[str, Any],
    gaba_candidate: Mapping[str, Any],
    arm: str,
    seed: int,
    temporary_location_table: Path | None = None,
) -> NetworkSpec:
    if arm not in ARM_REFIT_TYPES:
        raise ValueError(f"unknown arm {arm!r}")
    spec = _candidate_transfer_spec(
        _base_micro_spec(config, seed), candidate, ARM_REFIT_TYPES[arm]
    )
    spec = _gaba_candidate_transfer_spec(spec, gaba_candidate, arm == "E")
    if ARM_MODELS[arm]:
        # Candidate-only per-cell model capability overlays.  The backend still
        # owns population creation and all connection calls.
        overrides = dict(spec.aglif_dend_overrides)
        for cell_type, model in ARM_MODELS[arm].items():
            current = overrides.get(cell_type, AglifDendOverride())
            overrides[cell_type] = replace(current, model=model)
        spec = replace(spec, aglif_dend_overrides=overrides)
    if temporary_location_table is not None:
        spec = _write_candidate_location_table(
            spec,
            candidate,
            ARM_REFIT_TYPES[arm],
            temporary_location_table,
            gaba_candidate if arm == "E" else None,
        )
    return spec


def _pair_key(row: Any) -> tuple[Any, ...]:
    return (
        row.pre, row.post, row.indegree, row.biological_indegree,
        row.synapses_per_connection, row.delay_ms, row.receptor,
        row.release_component,
    )


def preflight(
    config: Path,
    candidate_path: Path,
    gaba_candidate_path: Path,
    seed: int,
) -> dict[str, Any]:
    """CPU-build every arm through uniform edge generation, without NEST-GPU."""
    before = deployed_hashes()
    candidate = _load_candidate(candidate_path)
    gaba_candidate = _load_gaba_candidate(gaba_candidate_path)
    base = _base_micro_spec(config, seed)
    full = build_network_spec(config, scale=1.0, seed=seed)
    retained_full = [
        row for row in full.projections
        if row.post in LOOP_TYPES and row.pre in frozenset((*LOOP_TYPES, BOUNDARY_TYPE))
    ]
    if [_pair_key(row) for row in base.projections] != [_pair_key(row) for row in retained_full]:
        raise RuntimeError("retained recurrent rows differ from the canonical full spec")

    arm_reports: dict[str, Any] = {}
    identity_digests: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="ca1-closedloop-preflight-") as tmp:
        for arm in ARMS:
            spec = build_arm_spec(
                config,
                candidate,
                gaba_candidate,
                arm,
                seed,
                Path(tmp) / f"locations-{arm}.json",
            )
            counts = _instantiated_counts(spec)
            provenance = _microcircuit_parameter_provenance(spec, config, seed)
            if provenance.get("network.recurrent_topology") != "modeldb_fastconn_binned":
                raise RuntimeError(f"uniform topology missing from arm {arm} provenance")
            recurrent_rows: list[dict[str, Any]] = []
            for row_index, row in enumerate(spec.projections):
                requested = int(round(row.indegree))
                if requested < 1 or requested > counts[row.pre]:
                    raise RuntimeError(
                        f"immutable K infeasible in arm {arm} row {row_index}: "
                        f"{row.pre}->{row.post} K={requested}, Npre={counts[row.pre]}"
                    )
                calls = binned_fixed_indegree_connections(
                    pre_type=row.pre,
                    post_type=row.post,
                    pre_count=counts[row.pre],
                    post_count=counts[row.post],
                    indegree=requested,
                )
                if not calls:
                    raise RuntimeError(f"no uniform edge calls for arm {arm} row {row_index}")
                infeasible = [call for call in calls if call.source_count < call.indegree]
                realized = {call.indegree for call in calls}
                if infeasible or realized != {requested}:
                    raise RuntimeError(
                        f"uniform edge generation changed immutable K in arm {arm} "
                        f"row {row_index} {row.pre}->{row.post}: requested={requested}, "
                        f"realized={sorted(realized)}, infeasible_calls={len(infeasible)}"
                    )
                recurrent_rows.append({
                    "row_index": row_index,
                    "pre": row.pre,
                    "post": row.post,
                    "receptor": row.receptor,
                    "requested_indegree": requested,
                    "realized_indegree_per_target": requested,
                    "target_cells": counts[row.post],
                    "edge_count": requested * counts[row.post],
                    "uniform_connect_calls": len(calls),
                    "minimum_source_window": min(call.source_count for call in calls),
                    "weight_nS": row.weight_nS,
                    "synapses_per_connection": row.synapses_per_connection,
                })
            # Exercise the exact backend syn-spec constructor for every retained row.
            syn_specs = [_projection_syn_spec(spec, row) for row in spec.projections]
            afferent_degrees = {
                row.name: _literal_source_indegree(row) for row in spec.afferents
            }
            infeasible_afferents = {
                row.name: afferent_degrees[row.name]
                for row in spec.afferents
                if afferent_degrees[row.name] > row.n_source
            }
            if infeasible_afferents:
                raise RuntimeError(
                    f"literal afferent K infeasible in arm {arm}: {infeasible_afferents}"
                )
            identity = {
                "recurrent": [
                    {key: value for key, value in row.items() if key != "weight_nS"}
                    for row in recurrent_rows
                ],
                "afferent": afferent_degrees,
                "counts": counts,
                "topology": spec.recurrent_topology,
                "seed": seed,
            }
            digest = hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            identity_digests[arm] = digest
            model_by_type = {
                cell_type: ARM_MODELS[arm].get(cell_type, "user_m2")
                for cell_type in LOOP_TYPES
            }
            gaba_rows = _gaba_candidate_rows(gaba_candidate)
            realized_gaba_rows = sorted(
                key
                for key in gaba_rows
                if any(
                    key == f"{row.pre}->{row.post}|{row.receptor}"
                    for row in spec.projections
                )
            ) if arm == "E" else []
            arm_reports[arm] = {
                "refit_types": sorted(ARM_REFIT_TYPES[arm]),
                "model_by_type": model_by_type,
                "cck_model": model_by_type["CCK_Basket"],
                "gaba_candidate_rows_declared": sorted(gaba_rows) if arm == "E" else [],
                "gaba_candidate_rows_realized": realized_gaba_rows,
                "gaba_candidate_rows_outside_microcircuit": (
                    sorted(set(gaba_rows) - set(realized_gaba_rows)) if arm == "E" else []
                ),
                "parameter_provenance_checked": True,
                "parameter_provenance_sha256": hashlib.sha256(
                    json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "n_recurrent_rows": len(spec.projections),
                "recurrent_rows": recurrent_rows,
                "recurrent_edge_count": sum(row["edge_count"] for row in recurrent_rows),
                "n_afferent_rows": len(spec.afferents),
                "afferent_indegrees": afferent_degrees,
                "all_recurrent_indegrees_preserved": True,
                "all_uniform_source_windows_feasible": True,
                "backend_syn_specs_type_checked": len(syn_specs),
                "edge_identity_contract_sha256": digest,
            }
    if len(set(identity_digests.values())) != 1:
        raise RuntimeError(f"edge identity changed between arms: {identity_digests}")
    after = deployed_hashes()
    if after != before:
        raise RuntimeError("candidate overlay mutated a deployed parameter file")
    return {
        "cpu_only": True,
        "single_gpu_contract": True,
        "mpi": False,
        "downscale_mode": "preserve-indegree",
        "p_preserve_used": False,
        "recurrent_topology": "modeldb_fastconn_binned",
        "topology_rationale": (
            "working-point/recruitment (not theta phase); validated 3-D/uniform "
            "rate equivalence at fixed K/J; uniform avoids downscaled extent shortage"
        ),
        "boundary": {"type": BOUNDARY_TYPE, "kind": "held-out independent proxy", "rate_hz": BOUNDARY_RATE_HZ},
        "afferent_rate_hz": 0.65,
        "gaba_candidate": str(gaba_candidate_path),
        "gaba_candidate_five_rows_validated": len(gaba_candidate["rows"]) == 5,
        "micro_counts": dict(MICRO_COUNTS),
        "arms": arm_reports,
        "edge_identity_same_all_arms": True,
        "deployed_hashes_before": before,
        "deployed_hashes_after": after,
        "deployed_files_unchanged": before == after,
    }


def _reject_mpi() -> None:
    size = int(os.environ.get("PMI_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", "1")))
    if size != 1:
        raise RuntimeError(f"closed-loop harness is single-GPU only; MPI size={size}")


def _apply_intrinsic_overlay(
    backend: NestGpuBackend,
    candidate: Mapping[str, Any],
    refit_types: frozenset[str],
) -> dict[str, dict[str, float]]:
    if backend._ngpu is None:  # pyright: ignore[reportPrivateUsage]
        raise RuntimeError("backend was not set up")
    applied: dict[str, dict[str, float]] = {}
    for cell_type in sorted(refit_types):
        fitted = {
            str(key): float(value)
            for key, value in candidate["intrinsic"]["cells"][cell_type]["fitted_params"].items()
        }
        # This exactly matches scripts/cck_sca_refit.py::_status: fitted fields
        # replace deployed status fields, while unchanged fields (including g_c)
        # remain deployed.
        status = aglif_dend_status(cell_type)
        status.update(fitted)
        backend._ngpu.SetStatus(backend._nodes[cell_type], fitted)  # pyright: ignore[reportPrivateUsage]
        applied[cell_type] = fitted
    return applied


def _attach_trace(
    backend: NestGpuBackend,
    spec: NetworkSpec,
    sample_cells: int,
    stride: int,
) -> tuple[int | None, list[dict[str, Any]]]:
    if sample_cells <= 0:
        return None, []
    ngpu = backend._ngpu  # pyright: ignore[reportPrivateUsage]
    if ngpu is None:
        raise RuntimeError("backend was not set up")
    variables: list[str] = []
    nodes: list[int] = []
    ports: list[int] = []
    columns: list[dict[str, Any]] = []
    for cell_type in LOOP_TYPES:
        receptors = spec.receptors_for_post(cell_type)
        compartments = aglif_dend_compartments(
            receptors.names,
            cell_type,
            _required_dendritic_ports(spec, cell_type),
            spec.source_location_transfer_table,
            spec.aglif_receive_domain_overrides,
        )
        population = backend._nodes[cell_type]  # pyright: ignore[reportPrivateUsage]
        for cell_index in range(min(sample_cells, len(population))):
            node = int(population[cell_index])
            for variable in ("V_m", "V_d", "V_dist"):
                variables.append(variable)
                nodes.append(node)
                ports.append(0)
                columns.append({"cell_type": cell_type, "cell_index": cell_index,
                                "variable": variable, "port": None})
            for port, receptor in enumerate(receptors.names):
                variables.append("g")
                nodes.append(node)
                ports.append(port)
                columns.append({"cell_type": cell_type, "cell_index": cell_index,
                                "variable": "g", "port": port, "receptor": receptor,
                                "compartment": compartments[port],
                                "E_rev_mV": receptors.E_rev[port]})
    record = ngpu.CreateRecord("", variables, nodes, ports)
    ngpu.SetRecordStride(record, stride)
    return record, columns


def _save_trace(
    backend: NestGpuBackend,
    record: int | None,
    columns: Sequence[Mapping[str, Any]],
    path: Path,
) -> str | None:
    if record is None or backend._ngpu is None:  # pyright: ignore[reportPrivateUsage]
        return None
    data = np.asarray(backend._ngpu.GetRecordData(record), dtype=np.float64)  # pyright: ignore[reportPrivateUsage]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        time_ms=data[:, 0],
        values=data[:, 1:],
        columns_json=np.asarray(json.dumps(list(columns), separators=(",", ":"))),
    )
    return str(path)


def run_arm(args: argparse.Namespace) -> dict[str, Any]:
    _reject_mpi()
    before = deployed_hashes()
    candidate = _load_candidate(args.candidate)
    gaba_candidate = _load_gaba_candidate(args.gaba_candidate)
    with tempfile.TemporaryDirectory(prefix=f"ca1-closedloop-{args.arm}-") as tmp:
        spec = build_arm_spec(
            args.config,
            candidate,
            gaba_candidate,
            args.arm,
            args.seed,
            Path(tmp) / "locations.json",
        )
        counts = _instantiated_counts(spec)
        meta = SimMeta(
            duration_s=args.duration_s,
            dt_s=args.dt_ms * 1e-3,
            n_cells_per_type=counts,
            scale=MICRO_SCALE,
            seed=args.seed,
            backend="nestgpu",
            config_name=spec.name,
            crop_first_ms=args.crop_ms,
            parameter_provenance=_microcircuit_parameter_provenance(
                spec, args.config, args.seed
            ),
            diagnostic_provenance={
                "harness": "closed_loop_microcircuit",
                "arm": args.arm,
                "boundary": f"Pyramidal independent {BOUNDARY_RATE_HZ:g} Hz boundary",
                "downscale_mode": "preserve-indegree",
            },
        )
        backend = NestGpuBackend()
        backend.setup(dt_ms=args.dt_ms, seed=args.seed)
        backend.build(spec, counts)
        applied = _apply_intrinsic_overlay(backend, candidate, ARM_REFIT_TYPES[args.arm])
        backend._set_literal_source_spike_trains(duration_s=args.duration_s, seed=args.seed)  # pyright: ignore[reportPrivateUsage]
        backend._max_rec_spikes = _record_spike_buffer_size(args.duration_s)  # pyright: ignore[reportPrivateUsage]
        backend.attach_recorders(LOOP_TYPES)
        trace_record, trace_columns = _attach_trace(
            backend, spec, args.trace_cells, args.trace_stride
        )
        backend.run(args.duration_s * 1000.0)
        raw = backend.collect_spikes()
        crop_s = args.crop_ms * 1e-3
        spikes = {
            cell_type: [train[train >= crop_s] - crop_s for train in raw[cell_type]]
            for cell_type in LOOP_TYPES
        }
        measure_s = args.duration_s - crop_s
        if measure_s <= 0.0:
            raise ValueError("crop window must be shorter than duration")
        rates = {
            cell_type: float(sum(len(train) for train in trains) / (len(trains) * measure_s))
            for cell_type, trains in spikes.items()
        }
        trace_path = args.output.with_name(f"{args.output.stem}.{args.arm}.traces.npz")
        saved_trace = _save_trace(backend, trace_record, trace_columns, trace_path)
    after = deployed_hashes()
    if after != before:
        raise RuntimeError("candidate overlay mutated a deployed parameter file")
    return {
        "arm": args.arm,
        "meta": asdict(meta),
        "refit_types": sorted(ARM_REFIT_TYPES[args.arm]),
        "rates_hz": rates,
        "spike_times_s": {
            cell_type: [train.tolist() for train in trains]
            for cell_type, trains in spikes.items()
        },
        "trace_npz": saved_trace,
        "intrinsic_overlay": applied,
        "model_by_type": {
            cell_type: ARM_MODELS[args.arm].get(cell_type, "user_m2")
            for cell_type in LOOP_TYPES
        },
        "cck_model": ARM_MODELS[args.arm].get("CCK_Basket", "user_m2"),
        "gaba_candidate": str(args.gaba_candidate) if args.arm == "E" else None,
        "gaba_candidate_rows_declared": (
            sorted(_gaba_candidate_rows(gaba_candidate)) if args.arm == "E" else []
        ),
        "gaba_candidate_rows_realized": (
            sorted(
                key
                for key in _gaba_candidate_rows(gaba_candidate)
                if any(
                    key == f"{row.pre}->{row.post}|{row.receptor}"
                    for row in spec.projections
                )
            )
            if args.arm == "E" else []
        ),
        "deployed_files_unchanged": before == after,
        "n_cells": {name: counts[name] for name in LOOP_TYPES},
    }


def _arm_command(args: argparse.Namespace, arm: str, output: Path) -> list[str]:
    return [
        sys.executable, str(Path(__file__).resolve()),
        "--worker-arm", arm,
        "--config", str(args.config),
        "--candidate", str(args.candidate),
        "--gaba-candidate", str(args.gaba_candidate),
        "--output", str(output),
        "--duration-s", str(args.duration_s),
        "--crop-ms", str(args.crop_ms),
        "--dt-ms", str(args.dt_ms),
        "--seed", str(args.seed),
        "--trace-cells", str(args.trace_cells),
        "--trace-stride", str(args.trace_stride),
    ]


def _verdict(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    baseline = arms["A"]["rates_hz"]
    rates = arms["E"]["rates_hz"]
    recruited = {
        cell: bool(rates[cell] >= 1.0 and rates[cell] >= baseline[cell] + 1.0)
        for cell in ("Bistratified", "O_LM")
    }
    cck_drop = float(baseline["CCK_Basket"] - rates["CCK_Basket"])
    pv_stays_low = bool(rates["PV_Basket"] < 1.0)
    working_point_shift = bool(any(recruited.values()) and cck_drop > 1.0)
    return {
        "bistratified_and_o_lm_recruit": bool(all(recruited.values())),
        "recruited": recruited,
        "cck_drops_from_A": bool(cck_drop > 1.0),
        "cck_drop_from_A_hz": cck_drop,
        "working_point_starts_to_shift": working_point_shift,
        "pv_stays_low": pv_stays_low,
        "pv_is_sole_remaining_ping_blocker": bool(
            all(recruited.values()) and cck_drop > 1.0 and pv_stays_low
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--gaba-candidate", type=Path, default=DEFAULT_GABA_CANDIDATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--crop-ms", type=float, default=500.0)
    parser.add_argument("--dt-ms", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--trace-cells", type=int, default=2)
    parser.add_argument("--trace-stride", type=int, default=10)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--worker-arm", choices=ARMS, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.config = args.config.resolve()
    args.candidate = args.candidate.resolve()
    args.gaba_candidate = args.gaba_candidate.resolve()
    args.output = args.output.resolve()
    if args.preflight:
        report = preflight(
            args.config, args.candidate, args.gaba_candidate, args.seed
        )
        print(json.dumps(report, indent=2))
        return 0
    if args.worker_arm:
        args.arm = args.worker_arm
        report = run_arm(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return 0

    _reject_mpi()
    cpu_audit = preflight(
        args.config, args.candidate, args.gaba_candidate, args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    arm_reports: dict[str, Any] = {}
    for arm in ARMS:
        arm_path = args.output.with_name(f"{args.output.stem}.{arm}.json")
        subprocess.run(_arm_command(args, arm, arm_path), check=True)
        arm_reports[arm] = json.loads(arm_path.read_text(encoding="utf-8"))
    report = {
        "schema": "ca1-closed-loop-microcircuit/v1",
        "prediction": (
            "Arm E assembles CCK user_m3 plus the source-gated GABA-into-CCK "
            "candidate, Bistratified user_m4, O_LM user_m5, deployed PV user_m2, "
            "and the already-wired source-grounded SCA refit."
        ),
        "protocol": {
            "single_gpu": True, "mpi": False,
            "mutual_loop_types": list(LOOP_TYPES),
            "pyramidal_boundary": f"independent {BOUNDARY_RATE_HZ:g} Hz boundary",
            "afferents": "literal CA3/ECIII source graphs at 0.65 Hz",
            "downscale_mode": "preserve-indegree",
            "recurrent_topology": "modeldb_fastconn_binned",
            "table5_rate_tuning": False,
            "duration_s": args.duration_s, "crop_ms": args.crop_ms,
            "dt_ms": args.dt_ms, "seed": args.seed,
        },
        "cpu_preflight": cpu_audit,
        "arms": arm_reports,
        "verdict": _verdict(arm_reports),
        "interpretation": (
            "Bistratified and O_LM recruitment plus a CCK drop supports partial "
            "fixed-point escape. If PV alone remains low in that state, the result "
            "isolates the known user_m2 PV limitation as the remaining PING blocker."
        ),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
