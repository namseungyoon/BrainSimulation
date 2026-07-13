"""Bezaire et al. (2016) paper targets for CA1 theta/gamma validation.

All values extracted verbatim from the published paper (eLife 2016;5:e18566).
Section / table citations are given inline.

Conventions
-----------
* Phase 0 deg = LFP trough in the pyramidal layer (trough-referenced convention,
  same as Bezaire 2016 Figure 5 and Table 5).
* Rates in Hz, bands in Hz, phases in degrees.
"""

from __future__ import annotations

# NOTE: canonical cell-type names match ca1.config / ca1.params (the spec keys).
# The axo-axonic population is "Axo" everywhere in the package (Table 5 labels it "Axo.").

# ---------------------------------------------------------------------------
# Oscillation targets  (paper Figure 4, Table 7 caption, sections p.6-7)
# ---------------------------------------------------------------------------

# Theta peak frequency (Hz) -- "theta frequency of 7.8 Hz" (p.6, Figure 4B caption)
THETA_PEAK_HZ: float = 7.8

# Theta band (Hz) -- "5-10 Hz oscillation" (p.2, abstract; p.6 Results)
THETA_BAND: tuple[float, float] = (5.0, 10.0)

# Gamma band (Hz) -- "gamma oscillations (25-80 Hz)" (p.6, Results)
GAMMA_BAND: tuple[float, float] = (25.0, 80.0)

# Gamma peak frequency (Hz) -- "gamma (71 Hz)" (p.7, Figure 4D caption)
GAMMA_PEAK_HZ: float = 71.0

# A +/-20 Hz window accepts a broad high-gamma peak (51-80 Hz within the
# published band) while rejecting a 1/f-driven argmax at the 25 Hz band floor.
GAMMA_PEAK_TOLERANCE_HZ: float = 20.0

# A spectral peak must rise at least three-fold above a robust log-log 1/f fit.
SPECTRAL_PEAK_MIN_PROMINENCE_RATIO: float = 3.0

# Afferent tonic drive (Hz) -- "afferent excitation level of 0.65 Hz" (p.9, Figure 6)
AFFERENT_HZ: float = 0.65

# CFC is accepted against block-permuted gamma-envelope surrogates, not merely
# for a positive finite-sample modulation index.
CFC_SURROGATE_ALPHA: float = 0.05
CFC_MIN_Z_SCORE: float = 1.645  # one-sided 95th percentile
CFC_N_SURROGATES: int = 199
CFC_MIN_WINDOW_S: float = 1.0

# Theta power must exceed gamma power in the LFP (qualitative criterion from paper)
THETA_DOMINATES_GAMMA: bool = True

# ---------------------------------------------------------------------------
# Phase preferences (degrees)  -- Table 5, Bezaire 2016, p.14
# "Preferred theta firing phases for each model cell type."
# Phase 0 deg = trough; columns: Cell type | Phase | Firing rate | Modulation Level | p
# ---------------------------------------------------------------------------

MODEL_PHASE_DEG: dict[str, float] = {
    # Table 5 (trough-referenced, pyramidal layer LFP)
    "Pyramidal":      339.7,   # Pyr.     -- Table 5
    "PV_Basket":      356.8,   # PV+ B.   -- Table 5
    "Bistratified":   340.0,   # Bis.     -- Table 5
    "O_LM":           334.7,   # O-LM     -- Table 5
    "Axo":     163.4,   # Axo.     -- Table 5
    "CCK_Basket":     202.8,   # CCK+ B.  -- Table 5
    "Ivy":            142.1,   # Ivy      -- Table 5
    "Neurogliaform":  176.3,   # NGF.     -- Table 5
    "SCA":            197.9,   # S.C.-A.  -- Table 5
}

# Two-group classification from paper text (pp.8-9, Figure 5 discussion):
#   "trough-group": discharge NEAR theta trough (0/360 deg); strongly driven by Pyr recurrent
#   "rising-group": discharge AWAY from trough, near ~150-200 deg (falling -> rising phase)
TROUGH_GROUP: frozenset[str] = frozenset(
    {"Pyramidal", "PV_Basket", "Bistratified", "O_LM"}
)
RISING_GROUP: frozenset[str] = frozenset(
    {"Axo", "CCK_Basket", "Ivy", "Neurogliaform", "SCA"}
)

# Tolerance for phase comparison (degrees): "shifts are small" (p.9); 45 deg used here
PHASE_TOLERANCE_DEG: float = 45.0

# ---------------------------------------------------------------------------
# Model firing rates (Hz) -- Table 5, "Firing rate (Hz)" column
# These are the INTRINSIC MODEL rates at 0.65 Hz afferent drive (full-scale).
# ---------------------------------------------------------------------------

MODEL_RATES_HZ: dict[str, float] = {
    # Table 5: Firing rate (Hz) per cell type
    "Pyramidal":      6.0,    # Pyr.     -- Table 5 (0.74 modulation)
    "PV_Basket":      0.9,    # PV+ B.   -- Table 5 (0.46 modulation)
    "Bistratified":  18.0,    # Bis.     -- Table 5 (0.76 modulation)
    "O_LM":          17.4,    # O-LM     -- Table 5 (0.76 modulation)
    "Axo":     8.9,    # Axo.     -- Table 5 (0.07 modulation)
    "CCK_Basket":    54.4,    # CCK+ B.  -- Table 5 (0.10 modulation)
    "Ivy":           43.3,    # Ivy      -- Table 5 (0.33 modulation)
    "Neurogliaform": 55.1,    # NGF.     -- Table 5 (0.07 modulation)
    "SCA":            5.2,    # S.C.-A.  -- Table 5 (0.03 modulation)
}

# Theta modulation levels from Table 5 (used for quality checks, not primary pass/fail)
MODEL_MODULATION: dict[str, float] = {
    "Pyramidal":     0.74,
    "PV_Basket":     0.46,
    "Bistratified":  0.76,
    "O_LM":          0.76,
    "Axo":    0.07,
    "CCK_Basket":    0.10,
    "Ivy":           0.33,
    "Neurogliaform": 0.07,
    "SCA":           0.03,
}

# Table 5 modulation depths are population estimates.  A 50% relative window
# with a 0.05 absolute floor avoids pretending they are exact point targets.
MODULATION_REL_TOL: float = 0.50
MODULATION_ABS_TOL: float = 0.05

# Phase estimates require a prominence-validated theta rhythm and at least
# eight cycles; the additional one-second floor prevents sub-second hard gates.
PHASE_MIN_THETA_CYCLES: float = 8.0
PHASE_MIN_WINDOW_S: float = 1.0

# ---------------------------------------------------------------------------
# Experimental (in vivo) firing rates (Hz) -- Table 6, Bezaire 2016, p.15-16
# Representative anesthetized-theta values (rat, urethane+ketamine/xylazine where noted).
# These are the BIOLOGICAL reference for comparison; the model rates above are computed.
# Format: cell_type -> (theta_rate_hz, reference_note)
# ---------------------------------------------------------------------------

EXPERIMENTAL_RATE_HZ: dict[str, tuple[float, str]] = {
    # Table 6 (anesthetized theta state, rat, unless noted)
    "Axo":     (17.10, "Klausberger et al. 2003, anesth u+k+x, rat"),
    "Bistratified":   ( 5.90, "Klausberger et al. 2004, anesth u+k+x, rat"),
    "CCK_Basket":     ( 9.40, "Klausberger et al. 2005, anesth u+k+x, rat"),
    "Ivy":            ( 0.70, "Fuentealba et al. 2008, anesth u+k+x, rat"),
    "Neurogliaform":  ( 6.00, "Fuentealba et al. 2010, anesth u+k+x, rat"),
    "O_LM":           ( 4.90, "Klausberger et al. 2003, anesth u+k+x, rat"),
    "PV_Basket":      ( 7.30, "Klausberger et al. 2003, anesth u+k+x, rat"),
    # Note: Pyramidal and SCA not listed in Table 6 with clear anesthetized theta rate;
    # Pyr is noted as "higher than ~1-2 Hz in vivo" (p.7 text) -- not used as hard target.
}

# ---------------------------------------------------------------------------
# CV(ISI) range expected for active cells during theta (biophysical plausibility)
# Not from a specific Table but consistent with irregular firing during theta (CV ~0.7-1.4)
# ---------------------------------------------------------------------------

CV_ISI_RANGE: tuple[float, float] = (0.7, 1.4)

# Relative tolerance for rate comparison (model vs model target): 30%
RATE_REL_TOL: float = 0.30

# Minimum theta modulation level to consider a cell type "phase-locked"
# (Rayleigh p < 0.05 is the paper's criterion; here we check the passed p-value)
RAYLEIGH_ALPHA: float = 0.05
