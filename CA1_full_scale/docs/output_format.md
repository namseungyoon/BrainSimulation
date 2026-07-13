# Simulation output format and interpretation

How a `ca1 sim` result HDF5 file is laid out, what each field means, and the
exact formulas `ca1 validate` uses to turn it into the theta/gamma/phase gates.
The generative model that produces the spikes is documented separately in
[`model_equations.md`](model_equations.md).

Writer: `ca1/cli.py` (`_persist_result`, lines ~404-446). Schema:
`SimResult` in `ca1/types.py`.

---

## 1. File layout

```
<result>.h5
├─ /                         attrs: lfp_dt_s (float, s)      LFP sample interval
├─ spikes/                   group
│   └─ <CellType>/           group  (9 types)
│       └─ <cell_index>      dataset (n_spikes,) float64     spike times [s]
├─ lfp                       dataset (n_samples,) float64    LFP proxy series
├─ cell_positions/           group
│   └─ <CellType>            dataset (N, 3) float64          x,y,z [um]
├─ n_cells_per_type/         group  attrs: <CellType> -> int
└─ meta/                     group  attrs (run + provenance, see 1.4)
```

Full-scale example (`results/fullscale_theta_stack.h5`): 338,740 spike datasets
(Pyramidal 311,500 ... SCA 400), `lfp` length 9,951 at `lfp_dt_s = 0.001`.

### 1.1 `spikes/<type>/<index>`
Each dataset is one cell's spike train: **absolute spike times in seconds**,
already cropped for the startup transient (`meta.crop_first_ms`, default 50 ms).
Index is the 0-based within-type cell id (matches `cell_positions` row order).
An empty dataset = a silent cell (kept, not dropped).

### 1.2 `lfp`
The extracellular LFP proxy time series (one channel), sampled every
`lfp_dt_s` seconds. Produced by the ModelDB N-pole reduced-domain forward model
(section 4). `null`/absent only if LFP recording was disabled
(`CA1_GPU_LFP_SAMPLE_CELLS=0`), in which case analysis falls back to the
pyramidal spike-density function.

### 1.3 `cell_positions/` and `n_cells_per_type/`
Soma coordinates in micrometres (BSB placement) and the per-type counts. The
LFP forward model and the ROI selection use these positions.

### 1.4 `meta/` attributes
Run: `config_name`, `backend` (gpu|nest), `scale`, `seed`, `tier`,
`duration_s`, `dt_s` (integrator step, s), `crop_first_ms`, `lfp_proxy`.
ROI: `analysis_roi_center_um`, `analysis_roi_radius_um`,
`analysis_roi_distance_mode` (xy|xyz).
Provenance (audit trail, proves no Table-5 rate tuning):
- `parameter_provenance_json` -- per-type parameter source (e.g.
  `{"aglif.Axo": "nestgpu-fi-fit", ...}`, 75 keys).
- `diagnostic_provenance_json` -- any diagnostic overrides applied
  (`{"diagnostic.audit": "no-overrides"}` for a clean deploy run).

**Not stored**: membrane voltages, synaptic currents, adaptation states, or the
connectivity graph. Only spikes + one LFP channel + positions + metadata. The
edge graph lives separately in the edge artifact (`results/edges_fullscale.h5`).

---

## 2. Loading

Portable (raw `h5py`):

```python
import h5py, numpy as np
with h5py.File("results/fullscale_theta_stack.h5", "r") as f:
    dur = float(f["meta"].attrs["duration_s"]) - f["meta"].attrs["crop_first_ms"] * 1e-3
    # per-type list of per-cell spike-time arrays (s)
    spikes = {t: [f[f"spikes/{t}/{i}"][:] for i in range(f["n_cells_per_type"].attrs[t])]
              for t in f["spikes"]}
    lfp = f["lfp"][:]
    fs = 1.0 / float(f.attrs["lfp_dt_s"])          # LFP sample rate (Hz)
```

The scoring pipeline is invoked with `ca1 validate <result>.h5 --tier full`,
which reconstructs a `SimResult` and runs sections 3-7 below.

---

## 3. Firing rates and irregularity  (`ca1/analysis/rates.py`)

Denominator is the **actual cropped window** `T = duration_s - crop_first_ms`
(dividing by the nominal duration was a historical ~5x inflation bug).

```
rate_type          = mean_over_cells( n_spikes_cell ) / T                 [Hz]
CV_ISI_cell        = std(ISI) / mean(ISI),   ISI = diff(sort(train))       (cells >=2 spikes)
Fano_cell          = var(counts_bin) / mean(counts_bin),   bin = 10 ms
chi^2 (synchrony)  = Var_t( pop_mean(t) ) / mean_cell( Var_t(count_cell) ) (Golomb-Rinzel)
```

`chi^2 -> 0` asynchronous, `chi^2 > 1` synchronous. `CV_ISI ~ 1` is the
asynchronous-irregular regime. Targets: Bezaire Table 5 model rates
(`MODEL_RATES_HZ`), compared at `RATE_REL_TOL = 30%`.

---

## 4. LFP proxy forward model  (`ca1/sim/npole_lfp.py`)

Point-source volume conductor over pyramidal somata inside the electrode ROI:

```
d_i        = || pos_i - roi.center ||        (xyz or xy per roi.distance_mode)
w_i        = SCALE * rho / (4 * pi * d_i)     for cells in ROI, else 0
LFP(t)     = - sum_i  w_i * I_i(t)
```

with `rho = 333 Ohm.cm` (`MODELDB_NPOLE_RHO_OHM_CM`), `SCALE = 1e-4`
(`MODELDB_NPOLE_POINT_SOURCE_SCALE`), `d_i` floored at 1e-9 um. `I_i(t)` is the
recorded per-cell transmembrane current proxy; the leading minus sign follows
the extracellular convention (inward/inhibitory somatic current -> positive
field deflection). Inhibitory currents at the pyramidal soma dominate the CA1
field, so this channel carries the theta/gamma rhythm. The deployed run uses
`roi.center = (200,100,120) um`, `radius = 1000 um`, `distance_mode = xyz`.

Fallback when `lfp` is absent: population spike-density function (SDF), Gaussian
kernel `sigma = 5 ms`, pooled per-cell rate `rate(t) = count(t)/(N_cells * dt)`.

---

## 5. Oscillation gates: theta and gamma  (`ca1/analysis/spectral.py`)

**Welch PSD** (`welch_psd`): `nperseg` = smallest power of two `>= 2*fs`
(frequency resolution `<= 0.5 Hz`), capped at `n/2`; `scaling="density"`.

**Peak + prominence** (`band_power_peak`): within a band, `peak = argmax(PSD)`;
the prominence is the peak's rise over a robust aperiodic (1/f) floor:

```
fit  log(PSD) ~ a*log(f) + b   over [0.5*lo, 1.5*hi], excluding |f - f_peak| <= max(1, 3*df),
     with 3 iterations of MAD outlier rejection (keep residual <= median + 2.5*1.4826*MAD)
background(f_peak) = exp(a*log(f_peak) + b)
prominence         = PSD(f_peak) / background(f_peak)
```

A peak passes iff `prominence >= 3.0` (`SPECTRAL_PEAK_MIN_PROMINENCE_RATIO`) and
it is not the band's lowest resolvable bin (rejects a 1/f-driven band-floor
argmax). Bands and targets (`ca1/validation/targets.py`):

| Gate  | Band (Hz) | Target (Hz) | Tolerance         | Prominence |
|-------|-----------|-------------|-------------------|------------|
| theta | 5 - 10    | 7.8         | in band, not edge | `>= 3x`    |
| gamma | 25 - 80   | 71          | `+/- 20 Hz`       | `>= 3x`    |

---

## 6. Phase preference  (`ca1/analysis/spectral.py: phase_preference`)

Convention **0 degrees = LFP trough** (Klausberger / Bezaire). Per cell type:

```
filtered = sosfiltfilt(butter(4, theta_band), lfp)          # theta 5-10 Hz
phi(t)   = angle(hilbert(filtered)) + pi                    # +pi -> 0 = trough
phase_k  = phi[ round(spike_k * fs) ]                       # LFP phase at each spike
C, S     = mean(cos phase), mean(sin phase)
mean_phase = atan2(S, C)  (deg, mod 360)
R          = sqrt(C^2 + S^2)                                 # vector strength [0,1]
Rayleigh:  z = n * R^2,   p ~ exp(-z)*(1 + (2z - z^2)/(4n) - ...)   (Zar 1999)
```

Targets: Table 5 preferred phases (`MODEL_PHASE_DEG`), tolerance `45 deg`;
cell types split into a trough group (Pyr/PV/Bis/O-LM) and a rising group
(Axo/CCK/Ivy/NGF/SCA). Phase-locked iff Rayleigh `p < 0.05`. Requires a
prominence-validated theta rhythm and `>= 8` theta cycles.

---

## 7. Theta-gamma cross-frequency coupling  (Tort MI, `theta_gamma_cfc`)

```
theta_phase = angle(hilbert(bandpass(lfp, 5-10 Hz)))
gamma_amp   = abs(hilbert(bandpass(lfp, 25-80 Hz)))
bin gamma_amp by theta_phase into 18 bins -> mean amp per bin -> normalize to p
MI = KL(p || uniform) / log(18) = sum_j p_j*log(p_j / (1/18)) / log(18)     in [0,1]
```

Significance is against surrogates that permute contiguous **quarter-theta-cycle
blocks** of the gamma envelope (`block = round(fs/(4*7.5 Hz))`), breaking
phase alignment while preserving local envelope smoothness:

```
p = (1 + #{ MI_surrogate >= MI_observed }) / (N_surrogates + 1)
z = (MI_observed - mean(MI_surrogate)) / std(MI_surrogate)
```

Gate: `N_surrogates = 199`, pass iff `p <= 0.05` and `z >= 1.645`, window
`>= 1 s` (`CFC_*` in `targets.py`).

---

## 8. Gate summary

| Check          | Pass condition                                              | Constant |
|----------------|-------------------------------------------------------------|----------|
| theta_peak     | peak in 5-10 Hz, not band edge, prominence `>= 3x`          | `THETA_BAND`, `SPECTRAL_PEAK_MIN_PROMINENCE_RATIO` |
| gamma_peak     | peak in 25-80 Hz within `71 +/- 20`, prominence `>= 3x`     | `GAMMA_PEAK_HZ`, `GAMMA_PEAK_TOLERANCE_HZ` |
| theta>gamma    | theta band power `>` gamma band power                       | `THETA_DOMINATES_GAMMA` |
| theta_gamma_cfc| Tort MI, surrogate `p <= 0.05`, `z >= 1.645`                | `CFC_N_SURROGATES=199`, `CFC_MIN_Z_SCORE=1.645` |
| phase (type)   | mean phase within `45 deg` of Table 5, Rayleigh `p < 0.05`  | `MODEL_PHASE_DEG`, `PHASE_TOLERANCE_DEG` |
| rate (type)    | within `30%` of Table 5 model rate                          | `MODEL_RATES_HZ`, `RATE_REL_TOL` |

Deployed full-scale result (`full_scale_theta_stack.yaml`): theta 6.84 Hz /
prominence 3.33x, gamma 58 Hz / 9.66x, CFC MI 0.041 p=0.005 z=15.6 -- all PASS
from arrhythmic 0.65 Hz Poisson input only. See
`docs/theta_achievement_summary.md`, `docs/fullscale_theta_stack_gates.txt`.

Re-score any result with:

```bash
ca1 validate results/fullscale_theta_stack.h5 --tier full
```
