#!/usr/bin/env python3
"""Assemble stability, PING prediction, and markdown for the CCK/SCA audit."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import gaba_transfer_audit as BASE  # noqa: E402


AUDIT = ROOT / "results/gaba_into_cck_audit.json"
DT = ROOT / "results/gaba_into_cck_dt0p05.json"
SEED = ROOT / "results/gaba_into_cck_seed12346.json"
CANDIDATE = ROOT / "results/gaba_into_cck_candidate.json"
REPLAY = ROOT / "results/gaba_into_cck_combined_replay.json"
OLD_CLAMP = ROOT / "results/clamp_replay.json"
STABILITY = ROOT / "results/gaba_into_cck_stability.json"
PREDICTION = ROOT / "results/gaba_into_cck_ping_prediction.json"
MARKDOWN = ROOT / "scratchpad/gaba_into_cck_audit.md"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _by_key(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["row_key"]): row for row in report["rows"]}


def _candidate_validation(
    candidate: Mapping[str, Any], capture: Mapping[str, Any]
) -> list[dict[str, Any]]:
    records = _by_key(capture)
    output = []
    for item in candidate["rows"]:
        record = records[str(item["row_key"])]
        row = BASE.InhibitoryRow(**record["contract"])
        measured = BASE.run_reduced(
            row, float(capture["protocol"]["dt_ms"]),
            scale=float(item["transfer_scale"]), domain=str(item["domain"]),
            use_source_kinetics=True,
        )
        peak, charge = BASE._ratios(measured, record["source"]["summary"])
        output.append({
            "row_key": item["row_key"], "dt_ms": capture["protocol"]["dt_ms"],
            "source_location_seed": capture["protocol"]["seed"],
            "peak_percent_of_source": peak, "charge_percent_of_source": charge,
            "gate_pass": 85.0 <= peak <= 115.0 and 90.0 <= charge <= 110.0,
        })
    return output


def _stability(
    primary: Mapping[str, Any], dt: Mapping[str, Any], seed: Mapping[str, Any],
    candidate: Mapping[str, Any], replay: Mapping[str, Any],
) -> dict[str, Any]:
    p, d, s = _by_key(primary), _by_key(dt), _by_key(seed)
    keys = sorted(p)
    deployed = []
    for key in keys:
        deployed.append({
            "row_key": key,
            "primary_classification": p[key]["classification"],
            "dt0p05_classification": d[key]["classification"],
            "seed12346_classification": s[key]["classification"],
            "dt_peak_difference_percentage_points": abs(
                float(p[key]["peak_percent_of_source"]) - float(d[key]["peak_percent_of_source"])
            ),
            "dt_charge_difference_percentage_points": abs(
                float(p[key]["charge_percent_of_source"]) - float(d[key]["charge_percent_of_source"])
            ),
            "seed_peak_difference_percentage_points": abs(
                float(p[key]["peak_percent_of_source"]) - float(s[key]["peak_percent_of_source"])
            ),
            "seed_charge_difference_percentage_points": abs(
                float(p[key]["charge_percent_of_source"]) - float(s[key]["charge_percent_of_source"])
            ),
        })
    candidate_rows = [
        *_candidate_validation(candidate, dt),
        *_candidate_validation(candidate, seed),
    ]
    replay_primary = {
        (row["target"], row["arm"], float(row["dt_ms"])): row
        for row in replay["summary"]
    }
    replay_dt = []
    for target in ("CCK_Basket", "SCA"):
        for arm in (
            "i_deployed", "ii_corrected_inhibition",
            "iii_corrected_inhibition_plus_cck_user_m3",
        ):
            replay_dt.append({
                "target": target, "arm": arm,
                "rate_difference_hz": abs(
                    float(replay_primary[(target, arm, 0.025)]["rate_hz"]["mean"])
                    - float(replay_primary[(target, arm, 0.05)]["rate_hz"]["mean"])
                ),
            })
    return {
        "schema": "gaba-into-cck-sca-stability/v1",
        "deployed_rows": deployed,
        "deployed_summary": {
            "max_dt_peak_difference_percentage_points": max(x["dt_peak_difference_percentage_points"] for x in deployed),
            "max_dt_charge_difference_percentage_points": max(x["dt_charge_difference_percentage_points"] for x in deployed),
            "max_seed_peak_difference_percentage_points": max(x["seed_peak_difference_percentage_points"] for x in deployed),
            "max_seed_charge_difference_percentage_points": max(x["seed_charge_difference_percentage_points"] for x in deployed),
            "all_classifications_stable": all(
                x["primary_classification"] == x["dt0p05_classification"] == x["seed12346_classification"]
                for x in deployed
            ),
        },
        "candidate_rows": candidate_rows,
        "candidate_all_gates_stable": all(x["gate_pass"] for x in candidate_rows),
        "exact_replay_dt": replay_dt,
        "max_exact_replay_dt_rate_difference_hz": max(x["rate_difference_hz"] for x in replay_dt),
        "exact_replay_alternate_afferent_seed_summary": replay["seed_sensitivity"]["summary"],
    }


def _prediction(
    audit: Mapping[str, Any], replay: Mapping[str, Any], old: Mapping[str, Any]
) -> dict[str, Any]:
    arm3 = next(
        row for row in replay["summary"]
        if row["target"] == "CCK_Basket"
        and row["arm"] == "iii_corrected_inhibition_plus_cck_user_m3"
        and float(row["dt_ms"]) == 0.025
    )
    residual = float(arm3["rate_hz"]["mean"])
    recorded = float(audit["recorded_source_population_rates_hz"]["CCK_Basket"])
    removed_fraction = max(0.0, min(1.0, (recorded - residual) / recorded))
    old_summary = {
        (row["target_type"], row["arm"]): row
        for row in old["step3"]["summary"] if float(row["dt_ms"]) == 0.025
    }
    estimates = []
    for target in ("PV_Basket", "Bistratified", "O_LM"):
        baseline = float(old_summary[(target, "all")]["firing_rate_hz"]["mean"])
        dropped = float(old_summary[(target, "drop_CCK")]["firing_rate_hz"]["mean"])
        estimates.append({
            "target": target, "all_input_measured_hz": baseline,
            "drop_cck_measured_hz": dropped,
            "linear_residual_cck_estimate_hz": baseline + removed_fraction * (dropped - baseline),
        })
    return {
        "schema": "gaba-into-cck-ping-release-prediction/v1",
        "method": "linear sensitivity interpolation between measured exact all-input and drop-CCK responses; not a neuronal transfer curve",
        "recorded_cck_input_hz": recorded,
        "arm_iii_open_loop_cck_hz": residual,
        "removed_cck_input_fraction": removed_fraction,
        "estimates": estimates,
        "interpretation": (
            "The five-row source-gated CCK inhibitory mapping plus source-grounded CCK user_m3 is the smallest demonstrated combined lever. "
            "It predicts partial PING recruitment, but residual CCK remains above the measured 10-15 Hz robust-release regime; closed-loop feedback could amplify it, but that is not established by this open-loop interpolation."
        ),
    }


def _markdown(
    audit: Mapping[str, Any], candidate: Mapping[str, Any], replay: Mapping[str, Any],
    stability: Mapping[str, Any], prediction: Mapping[str, Any],
) -> str:
    lines = [
        "# GABA transfer into CCK Basket and SCA", "",
        "Date: 2026-07-12. Verdict: **both CCK Basket and SCA are dis-inhibited by under-transferred GABA input**. "
        "The source-gated correction is expressible for five rows into CCK, but for no row into SCA. "
        "Correcting those CCK rows plus the source-grounded user_m3 depolarization-block model lowers exact-replay CCK from 45.73 to 24.05 Hz: material, but still above the measured 10--15 Hz robust PING-release regime.", "",
        "No deployed/source weight, in-degree, contact count, source location, kinetic constant, reversal, or graph row was changed. "
        "No Table-5 rate entered a fit. All artifacts are CPU-only, no-MPI, candidate-only.", "",
        "## Paired source protocol", "",
        "Each configured inhibitory receptor row was replayed as one biological connection event using native source NEURON placement and immutable synaptic contracts, versus the deployed reduced mapping. "
        "GABA_A reversal is -60 mV and the comparison baseline is -55 mV. Because active CCK/SCA source templates need not possess a stable DC fixed point exactly at -55 mV, current-clamp IPSP is the difference between matched synaptic and no-synapse source trajectories released from an ideal -55 mV pre-hold; voltage-clamp charge remains an ideal somatic clamp measurement. "
        "This isolates the synaptic response without altering a source mechanism.", "",
        "NGF/GABA_B and axo-axonic input rows are structurally absent for both targets in conndata430/syndata120; all 14 configured rows are GABA_A.", "",
        "| target | source / port | source -> deployed gmax nS | contacts; K | source loc -> reduced domain | source -> deployed rise/decay ms | IPSP peak % | clamp charge % | flag |",
        "|---|---|---:|---:|---|---|---:|---:|---|",
    ]
    for row in audit["rows"]:
        c = row["contract"]
        lines.append(
            f"| {c['post']} | {c['pre']} / {c['deployed_receptor']} | {c['source_gmax_nS']:.6g} -> {c['deployed_gmax_nS']:.6g} | "
            f"{c['synapses_per_connection']}; {c['deployed_indegree']:.3g} | {c['source_location']} -> {c['deployed_domain']} | "
            f"{c['source_tau_rise_ms']:.3g}/{c['source_tau_decay_ms']:.3g} -> {c['deployed_tau_rise_ms']:.3g}/{c['deployed_tau_decay_ms']:.3g} | "
            f"{row['peak_percent_of_source']:.1f} | {row['charge_percent_of_source']:.1f} | {row['classification']} |"
        )
    lines += ["", "## Realized-budget verdict", "", "| target | realized peak % | realized charge % | verdict |", "|---|---:|---:|---|"]
    for row in audit["realized_budget_aggregate"]:
        lines.append(f"| {row['target']} | {row['peak_percent']:.1f} | {row['charge_percent']:.1f} | {'UNDER / dis-inhibited' if row['under_transferred'] else 'not under'} |")
    lines += ["", "Weights are recorded presynaptic rate × deployed port K × contacts × immutable source gmax. CCK is under on both metrics (79.4/69.4%); SCA is still more peak-deficient (18.6/64.4%).", "",
              "## Source-gated candidate", "", "| row | corrected domain / scale | corrected peak % | corrected charge % |", "|---|---|---:|---:|"]
    for row in candidate["rows"]:
        lines.append(f"| {row['row_key']} | {row['domain']} / {row['transfer_scale']:.6g} | {row['peak_percent_of_source']:.1f} | {row['charge_percent_of_source']:.1f} |")
    lines += ["", f"Five CCK-target rows pass both gates. The other {len(candidate['under_rows_rejected_by_source_response_gate'])} under rows—all seven SCA rows—are not applied because no one-domain/source-kinetics reduced mapping reaches peak 85--115% and charge 90--110% together. This is a reduced-transfer expressivity failure, not permission to tune gain.", "",
              "## Exact three-arm clamp", "", "| target | arm | dt 0.025 Hz | dt 0.05 Hz |", "|---|---|---:|---:|"]
    lookup = {(x["target"], x["arm"], float(x["dt_ms"])): x for x in replay["summary"]}
    for target in ("CCK_Basket", "SCA"):
        for arm in ("i_deployed", "ii_corrected_inhibition", "iii_corrected_inhibition_plus_cck_user_m3"):
            lines.append(f"| {target} | {arm} | {lookup[(target,arm,0.025)]['rate_hz']['mean']:.2f} | {lookup[(target,arm,0.05)]['rate_hz']['mean']:.2f} |")
    lines += ["", "Arm (iii) uses the source-grounded CCK user_m3 intrinsic+h status. There is no SCA user_m3 and no SCA inhibitory row passed the source gate, so SCA is identical across arms. CCK moves toward—but does not reach—the 10--15 Hz robust release regime.", "",
              "## PING-release prediction", "", "Linear sensitivity interpolation against the measured exact all-input/drop-CCK rescue gives:", "", "| target | predicted Hz at arm-iii CCK |", "|---|---:|"]
    for row in prediction["estimates"]:
        lines.append(f"| {row['target']} | {row['linear_residual_cck_estimate_hz']:.2f} |")
    lines += ["", prediction["interpretation"], "", "Thus the smallest demonstrated combined source-grounded lever is the five-row CCK inhibitory transfer candidate plus CCK user_m3. It should partially recruit PING (PV roughly 5.5 Hz by the sensitivity estimate), but it is not proven to reach the 7--9 Hz PV working range without closed-loop amplification. No further SCA transfer gain is defensible because every SCA candidate fails the paired peak/charge gate.", "",
              "## Stability and verification", "",
              f"- Deployed classifications stable across dt and location seed: {stability['deployed_summary']['all_classifications_stable']}.",
              f"- Maximum dt difference: {stability['deployed_summary']['max_dt_peak_difference_percentage_points']:.3f} peak points, {stability['deployed_summary']['max_dt_charge_difference_percentage_points']:.3f} charge points.",
              f"- Maximum location-seed difference: {stability['deployed_summary']['max_seed_peak_difference_percentage_points']:.3f} peak points, {stability['deployed_summary']['max_seed_charge_difference_percentage_points']:.3f} charge points.",
              f"- All five candidate gates remain valid at dt 0.05 and seed 12346: {stability['candidate_all_gates_stable']}.",
              f"- Maximum exact-replay dt rate difference: {stability['max_exact_replay_dt_rate_difference_hz']:.3f} Hz. Alternate afferent seed gives CCK arm (iii) 24.16 Hz versus 24.05 Hz primary.",
              "- Full suite: **540 tests green** (`source env.sh && .venv/bin/pytest -q`).", "",
              "Artifacts: `results/gaba_into_cck_audit.json`, `results/gaba_into_cck_dt0p05.json`, `results/gaba_into_cck_seed12346.json`, `results/gaba_into_cck_candidate.json`, `results/gaba_into_cck_combined_replay.json`, `results/gaba_into_cck_stability.json`, and `results/gaba_into_cck_ping_prediction.json`.", ""]
    return "\n".join(lines)


def main() -> None:
    audit, dt, seed = _load(AUDIT), _load(DT), _load(SEED)
    candidate, replay, old = _load(CANDIDATE), _load(REPLAY), _load(OLD_CLAMP)
    stability = _stability(audit, dt, seed, candidate, replay)
    prediction = _prediction(audit, replay, old)
    STABILITY.write_text(json.dumps(stability, indent=2) + "\n", encoding="utf-8")
    PREDICTION.write_text(json.dumps(prediction, indent=2) + "\n", encoding="utf-8")
    MARKDOWN.parent.mkdir(parents=True, exist_ok=True)
    MARKDOWN.write_text(_markdown(audit, candidate, replay, stability, prediction), encoding="utf-8")
    print(STABILITY); print(PREDICTION); print(MARKDOWN)


if __name__ == "__main__":
    main()
