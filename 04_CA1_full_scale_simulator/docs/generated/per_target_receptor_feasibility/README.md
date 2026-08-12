# Per-target receptor feasibility audit

Inputs: `cellnumbers_101.dat`, `conndata_430.dat` with `count_mode=per_cell`, and original ModelDB `syndata_120/137` JSON exports.

This audit treats receptor ports as postsynaptic node-group-local channels. A target is feasible when its exact incoming `(receptor, E_rev, tau_rise, tau_decay, compartment)` table is <= 20 ports.

## Result

- SynData 120: global exact used ports=36, all-syndata exact ports=39, max used per target=13, per-target exact feasible=True.
- SynData 137: global exact used ports=36, all-syndata exact ports=39, max used per target=13, per-target exact feasible=True.

## Files

- `summary.json`: machine-readable feasibility result.
- `per_target_counts.csv`: target-local port counts for all syndata rows and paper-used pairs.
- `kinetics_components_syndata120.csv` / `kinetics_components_syndata137.csv`: exact original kinetic components after splitting ExpGABAab into GABA_A and GABA_B.

## Design implication

If NEST-GPU backend is changed from one global `spec.receptors` table to `receptor_tables[post_type]`, the original paper kinetics do not need a 39-to-20 global compression for the tested variants. Compression remains useful only for the current global-table implementation or for any future target whose local exact table exceeds 20 ports.
