# CA1 Model -- Authoritative sources and invariants for Claude Code

## Authoritative data (never regenerate without provenance)

- `src/ca1/params/connectivity.json` -- Bezaire et al. (2016) ModelDB conndata_101:
  9 populations (incl. Neurogliaform), per-projection indegree_true / weight_nS /
  synapses_per_connection / receptor; 12 afferents (CA3 Schaffer + ECIII perforant).

- `src/ca1/params/neuron_parameters.json` -- AEIF parameters derived from Bezaire
  Rin / tau_m / rheobase per cell type. O-LM g_L = 3.735 nS (authoritative).

- `src/ca1/params/interneuron_synapses.json` -- REAL nS-scale per-pair g_max +
  kinetics for all interneuron->target pairs.

- `src/ca1/params/syndata_120.json` / `syndata_137.json` -- receptor kinetics;
  GABA_A E_rev = -60 mV (120) / -75 mV (137).

## Trustworthy docs

- `docs/` -- package architecture, downscaling rationale, validation tier rules.
- `RECOVERY_PLAN.md` -- root-cause analysis of 8 confirmed bugs (read before
  touching build, sim, or downscale code).

## Goal phenomenon

Intrinsic theta ~7.8 Hz + gamma ~71 Hz driven by arrhythmic 0.65 Hz Poisson
afferents only. No rhythmic input. Reference: Bezaire et al. eLife 2016.

## Invariants -- never violate

1. **Inhibition**: POSITIVE weights routed to negative-E_rev GABA ports.
   Never use negative inhibitory weights.

2. **Afferent budget**: `Afferent.synapses_per_cell` = verbatim from conndata
   (total_synapses / N_post). Do NOT cap to presynaptic population size.
   ECIII->Neurogliaform = 58240 is correct.

3. **Downscaling**: `p-preserve` is DEBUG ONLY and silently drops in-degree.
   For all real runs use `preserve-indegree` (default) or `mean-field`.

4. **O-LM g_L**: must be ~3.735 nS. A ~6x error (0.56 nS) has been seen in
   older code paths. The authoritative value is in neuron_parameters.json.

5. **All 9 types**: build must include Neurogliaform. Never swallow exceptions
   in the build stats loop.

6. **Rate computation**: divide by actual simulated duration, not nominal chunk
   size. Old bug inflated rates ~5x.

7. **Paths**: always resolve via `Path(__file__)`. No `/Users/...` hardcodes.

## Type contracts

All modules import types from `ca1.types`. Never redefine NetworkSpec, SimResult,
CellType, etc. in other modules.

## Simulator imports

NEST and NEST-GPU imports MUST be lazy (inside methods). Neither is installed in
the dev venv -- the package must be importable without them.

## Name mapping

Cell-type name aliases must use an explicit dict. Never use
`.replace('cell', '').capitalize()` -- it silently mismaps interneurons to
pyramidal-cell parameters.
