COMMENT
PairSTDPSyn.mod - classic pair-based spike-timing-dependent plasticity synapse.
Written for the 04 track (two-neuron synaptic plasticity bench), stage 5-6.

NOTE: NMODL is ASCII-only, so the full rationale (in Korean) lives next to this file
      in PairSTDPSyn.md.  Read that first.

WHAT THIS IS
    The textbook exponential STDP window (Bi & Poo 1998 form):
        dt = t_post - t_pre > 0  ->  dw = +A_p * exp(-dt/tau_p)     (LTP)
        dt < 0                   ->  dw = -A_d * exp(+dt/tau_d)     (LTD)
    Implemented with two decaying traces, which reproduces the closed form exactly
    for isolated pairs and gives the standard all-to-all sum for trains:
        dx_pre /dt = -x_pre /tau_p        pre  spike: x_pre  += 1
        dx_post/dt = -x_post/tau_d        post spike: x_post += 1
        pre  spike: w -= A_d * x_post     (all preceding post spikes)
        post spike: w += A_p * x_pre      (all preceding pre  spikes)

WHY IT EXISTS
    It is the CONTRAST to the Graupner-Brunel engines.  GB has an internal calcium
    variable, so the same dt gives different results depending on rate and history.
    Classic STDP has no such state - it sees only spike pairs.  Stage 6-3/6-4 compares
    the two, and that comparison is only meaningful if a faithful classic STDP exists.

TRANSMISSION - deliberately identical in structure to GBPlasticitySyn
        A_AMPA += weight * w * factor_AMPA          (same for NMDA, x NMDA_ratio)
        g = gmax*(B_AMPA-A_AMPA) + gmax*(B_NMDA-A_NMDA)*mggate
    Same dual-exponential kernels, same mg gate (Jahr & Stevens 1990), same nS->uS
    convention (gmax in uS, NetCon weight is the transmission flag = 1.0).
    This is required by stage 5-10: engines must be comparable at the FIRST pulse,
    which is only possible if transmission is computed the same way.

WEIGHT RANGE - mapped onto the GB efficacy scale on purpose
        w in [w_min, w_max],  default [w0, b*w0] = [1, 5.28145]
        rho = (w - w_min) / (w_max - w_min)        <- reported for cross-engine compare
    GB's transmission weight is w = w0 + rho*(b*w0 - w0), i.e. exactly this range.
    Reporting rho instead of w lets stage 5-8/6-9 put both engines on one axis.
    Bounds are HARD (clipped), which is the classic additive rule.  Soft/multiplicative
    bounds are a different model and are NOT provided here.

OUR CHOICES (none of these is prescribed by Bi & Poo 1998)
  1 all_to_all (default 1)
        1 = traces accumulate (x += 1)   -> every pre-post pair contributes
        0 = traces are reset  (x  = 1)   -> nearest-neighbour pairing only
        The two differ a lot above ~20 Hz because the windows start to overlap.
  2 Ordering at a pre spike
        Transmission uses w BEFORE the LTD update caused by that same spike.
        Rationale: the spike is delivered with the weight the synapse currently has;
        the plasticity it induces takes effect from then on.
  3 A_p / A_d units
        Absolute increments in w (not fractions of the range).  With the default
        range width 4.28145, A_p = A_d = 0.1 means one maximally-timed pair moves the
        weight by 2.3% of the full range.  Stage 5-10 calibrates these.
  4 tau_p / tau_d defaults 16.8 / 33.7 ms
        Widely cited as Bi & Poo 1998, but the ORIGINAL IS NOT IN OUR HANDS
        (docs/DECISIONS.md open item #14).  Treat as provisional.

EXACT VERIFICATION (stage 5-6)
    Set w_min very low and w_max very high to disable clipping, then the final w is
    exactly w_init + sum over all pairs of the closed-form window.  Agreement with
    lib/refs/stdp.py is then limited only by event-time quantisation, so the test
    uses spike times that are integer multiples of dt.  Measured |dw| < 1e-12.

POST SPIKE PATH
    Delivered through a weight<0 sentinel NetCon, exactly as in the GB engines, so
    lib/wiring.wire_post_sentinel() and lib/synprobe work unchanged.

INITIALISATION - w IS SET IN INITIAL, ON PURPOSE
    The GB mods compute their transmission weight only in BREAKPOINT, so a spike that
    arrives at t=0 is transmitted with w=0 and silently vanishes (measured in stage 5-2,
    docs/DECISIONS.md D22).  This mod sets w in INITIAL so that trap cannot happen.
ENDCOMMENT

NEURON {
    THREADSAFE
    POINT_PROCESS PairSTDPSyn
    RANGE tau_p, tau_d, A_p, A_d, w_min, w_max, rho0, all_to_all
    RANGE w, rho, x_pre, x_post, n_pre, n_post, dw_last
    RANGE tau_r_AMPA, tau_d_AMPA, tau_r_NMDA, tau_d_NMDA, NMDA_ratio, e, mg, gmax
    RANGE g, g_AMPA, g_NMDA, i_AMPA, i_NMDA
    NONSPECIFIC_CURRENT i
}

UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (uS) = (microsiemens)
    (mM) = (milli/liter)
}

PARAMETER {
    : ---- STDP window (Bi & Poo 1998 form; originals not verified, see .md) ----
    tau_p      = 16.8  (ms)      : LTP window time constant (pre trace)
    tau_d      = 33.7  (ms)      : LTD window time constant (post trace)
    A_p        = 0.1             : LTP amplitude, absolute increment in w
    A_d        = 0.1             : LTD amplitude, absolute decrement in w
    all_to_all = 1               : 1 = accumulate traces, 0 = nearest neighbour
    : ---- weight range, mapped onto the GB efficacy scale ----
    w_min      = 1.0             : = w0        of GBPlasticitySyn
    w_max      = 5.28145         : = b*w0      of GBPlasticitySyn
    rho0       = 0.0             : initial efficacy on [0,1] -> w = w_min + rho0*(range)
    : ---- transmission (identical structure to GBPlasticitySyn) ----
    tau_r_AMPA = 0.2   (ms)
    tau_d_AMPA = 1.7   (ms)
    tau_r_NMDA = 9.0   (ms)
    tau_d_NMDA = 61.0  (ms)
    NMDA_ratio = 0.71
    e          = 0     (mV)
    mg         = 1     (mM)
    gmax       = 0.001 (uS)      : nS -> uS
}

ASSIGNED {
    v (mV)
    i (nA)
    i_AMPA (nA)
    i_NMDA (nA)
    g_AMPA (uS)
    g_NMDA (uS)
    g (uS)
    w
    rho
    dw_last
    n_pre
    n_post
    mggate
    factor_AMPA
    factor_NMDA
}

STATE {
    A_AMPA
    B_AMPA
    A_NMDA
    B_NMDA
    x_pre
    x_post
}

INITIAL {
    LOCAL tp_AMPA, tp_NMDA
    : ---- plasticity state.  w is set HERE (see COMMENT: D22 trap) ----
    w       = w_min + rho0*(w_max - w_min)
    rho     = rho0
    x_pre   = 0
    x_post  = 0
    dw_last = 0
    n_pre   = 0
    n_post  = 0
    : ---- transmission kernels ----
    A_AMPA = 0
    B_AMPA = 0
    A_NMDA = 0
    B_NMDA = 0
    tp_AMPA = (tau_r_AMPA*tau_d_AMPA)/(tau_d_AMPA-tau_r_AMPA)*log(tau_d_AMPA/tau_r_AMPA)
    factor_AMPA = -exp(-tp_AMPA/tau_r_AMPA)+exp(-tp_AMPA/tau_d_AMPA)
    factor_AMPA = 1/factor_AMPA
    tp_NMDA = (tau_r_NMDA*tau_d_NMDA)/(tau_d_NMDA-tau_r_NMDA)*log(tau_d_NMDA/tau_r_NMDA)
    factor_NMDA = -exp(-tp_NMDA/tau_r_NMDA)+exp(-tp_NMDA/tau_d_NMDA)
    factor_NMDA = 1/factor_NMDA
}

BREAKPOINT {
    SOLVE state METHOD cnexp
    rho = (w - w_min)/(w_max - w_min)
    mggate = 1 / (1 + exp(0.062 (/mV) * -(v)) * (mg / 2.62 (mM)))
    g_AMPA = gmax*(B_AMPA - A_AMPA)
    g_NMDA = gmax*(B_NMDA - A_NMDA) * mggate
    g = g_AMPA + g_NMDA
    i_AMPA = g_AMPA*(v - e)
    i_NMDA = g_NMDA*(v - e)
    i = i_AMPA + i_NMDA
}

DERIVATIVE state {
    A_AMPA' = -A_AMPA/tau_r_AMPA
    B_AMPA' = -B_AMPA/tau_d_AMPA
    A_NMDA' = -A_NMDA/tau_r_NMDA
    B_NMDA' = -B_NMDA/tau_d_NMDA
    x_pre'  = -x_pre /tau_p
    x_post' = -x_post/tau_d
}

NET_RECEIVE(weight (uS)) {
    if (weight >= 0) {
        : ================= PRE spike =================
        n_pre = n_pre + 1
        : (1) transmission uses w BEFORE this spike's own update (OUR CHOICE 2)
        A_AMPA = A_AMPA + weight*w*factor_AMPA
        B_AMPA = B_AMPA + weight*w*factor_AMPA
        A_NMDA = A_NMDA + weight*w*NMDA_ratio*factor_NMDA
        B_NMDA = B_NMDA + weight*w*NMDA_ratio*factor_NMDA
        : (2) LTD from every preceding post spike, carried in x_post
        dw_last = -A_d * x_post
        w = w + dw_last
        if (w < w_min) { w = w_min }
        if (w > w_max) { w = w_max }
        : (3) leave a trace for future post spikes
        if (all_to_all > 0) {
            x_pre = x_pre + 1
        } else {
            x_pre = 1
        }
    } else {
        : ================= POST spike (weight<0 sentinel) =================
        n_post = n_post + 1
        : LTP from every preceding pre spike, carried in x_pre
        dw_last = A_p * x_pre
        w = w + dw_last
        if (w < w_min) { w = w_min }
        if (w > w_max) { w = w_max }
        if (all_to_all > 0) {
            x_post = x_post + 1
        } else {
            x_post = 1
        }
    }
}
