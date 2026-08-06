COMMENT
GBPlasticityStpSyn.mod - short-term (Tsodyks-Markram) + long-term (Graupner-Brunel)
plasticity merged into one synapse.  "Model B".

NOTE: NMODL is ASCII-only, so the full rationale (in Korean) lives next to this file
      in GBPlasticityStpSyn.md.  Read that first.

Built by combining two existing mods; no new physics was invented here.
  * long  term (efficacy rho, ~11.5 min) : GBPlasticitySyn.mod  (Graupner & Brunel 2012 PNAS)
  * short term (release  Pr, ~100 ms)    : TM block of DetAMPANMDA.mod
                                           (Fuhrmann et al. 2002, deterministic version)

WHY
    One TBS burst is 4 pulses at 100 Hz.  A real Schaffer collateral synapse facilitates
    over that interval (Fac = 250 ms), so pulses 2/3/4 release more than pulse 1.
    GBPlasticitySyn has no short-term plasticity at all and treats all four identically,
    so it probably UNDER-estimates the calcium a TBS burst actually produces.
    This mod exists to test that hypothesis.

COUPLING (multiplicative)
    amplitude = Pr_norm * w * weight          (Pr_norm = short term, w = long term)
    w       = w0 + rho*(b*w0 - w0)            <- Graupner, unchanged
    Pr      = u * R                           <- Fuhrmann/TM, unchanged
    Pr_norm = Pr / pr_ref

OUR CHOICE 1 - Pr normalisation (norm_Pr, default ON)
    pr_ref = Use, i.e. the Pr of the FIRST pulse into a rested synapse
    (with u0 = 0 the first event gives u = Use and R = 1).
    With this, the first pulse of a rested synapse is EXACTLY the same size as in
    model A (GBPlasticitySyn), so the only difference between the two models is the
    within-burst / repeated-stimulus dynamics.
    Turning it off (norm_Pr = 0) shrinks the first pulse by a factor Use (= 0.15),
    which changes the absolute amplitude and invalidates re-use of the stage-1
    (stimulus-intensity) result.
    This normalisation is NOT prescribed by any paper - it is our convention, adopted
    so that the model comparison is interpretable.

OUR CHOICE 2 - should calcium scale with release?  (ca_stp, default ON = 1)
    C_pre_eff = C_pre * (1 + ca_stp*(Pr_norm - 1))
      ca_stp = 0 -> fixed C_pre per spike, exactly as in Graupner & Brunel.
                    Calcium then becomes identical to model A.
      ca_stp = 1 -> more release means more glutamate and more NMDA calcium influx.
    Graupner & Brunel 2012 has no short-term plasticity, so they used a constant.
    The proportionality is OUR HYPOTHESIS, not a published result.  Setting ca_stp = 0
    recovers the original exactly, which makes the assumption directly testable.

ORDERING PROBLEM OF THE DELAYED CALCIUM, AND HOW IT IS SOLVED
    The calcium of a pre spike is injected D = 18.8 ms LATER.  In a 100 Hz burst the
    inter-pulse interval is 10 ms < D, so at least two events are always in flight.
    Holding Pr in a single variable would let a later spike overwrite an earlier one
    and inject the WRONG calcium.
    -> The amount is carried IN THE FLAG of the self event:  net_send(D, 2 + C_pre_eff).
       The receiver does  c += (flag - 2)  whenever flag >= 2.  flag is a double, so the
       value survives intact, and each event carries its own copy - no queue, no array,
       no ordering assumption.  External NetCon events arrive with flag == 0, so there
       is no collision.

MISC
    * The post-synaptic spike arrives through a weight<0 sentinel NetCon, exactly as in
      GBPlasticitySyn.  That path does NOT touch the TM state (u, R, tsyn).
    * Defaults: calcium/efficacy = Wittenberg 2006 hippocampal slice fit (Graupner
      Table S2).  Short-term Use/Dep/Fac are supplied by the caller from
      params_table3 "SC->PC (E1s)" - WARNING, those are TUNED values, they are not in
      Ecker 2020 Table 3.
    * Diagnostic RANGE variables pr_last (last Pr_norm) and ca_last (last C_pre_eff)
      let you check directly that pulses 2/3/4 of a burst really are larger.
ENDCOMMENT

NEURON {
    THREADSAFE
    POINT_PROCESS GBPlasticityStpSyn
    RANGE tau_ca, C_pre, C_post, D
    RANGE theta_d, theta_p, gamma_d, gamma_p, tau, rho_star, rho0, b, w0
    RANGE Use, Dep, Fac, u0, norm_Pr, ca_stp, pr_ref
    RANGE tau_r_AMPA, tau_d_AMPA, tau_r_NMDA, tau_d_NMDA, NMDA_ratio, e, mg, gmax
    RANGE c, rho, w, g, g_AMPA, g_NMDA, i_AMPA, i_NMDA
    RANGE pr_last, ca_last
    NONSPECIFIC_CURRENT i
}

UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (uS) = (microsiemens)
    (mM) = (milli/liter)
}

PARAMETER {
    : calcium (Wittenberg2006)
    tau_ca  = 48.8373  (ms)
    C_pre   = 1.0
    C_post  = 0.275865
    D       = 18.8008  (ms)
    : efficacy (long term)
    theta_d = 1.0
    theta_p = 1.3
    gamma_d = 313.0965
    gamma_p = 1645.59
    tau     = 688355   (ms)      : 688.355 s * 1000
    rho_star = 0.5
    rho0    = 0.0                : initial efficacy
    b       = 5.28145            : w1/w0
    w0      = 1.0
    : release (short term, Tsodyks-Markram / Fuhrmann 2002 deterministic)
    Use     = 1.0      (1)       : caller overrides with SC->PC value (0.15)
    Dep     = 100      (ms)      :   "                               (150)
    Fac     = 10       (ms)      :   "                               (250)
    u0      = 0
    norm_Pr = 1                  : 1 = scale first pulse to match model A
    ca_stp  = 1                  : 1 = calcium follows release, 0 = Graupner original
    : transmission (AMPA/NMDA)
    tau_r_AMPA = 0.2   (ms)
    tau_d_AMPA = 1.7   (ms)
    tau_r_NMDA = 9.0   (ms)
    tau_d_NMDA = 61.0  (ms)
    NMDA_ratio = 0.71
    e   = 0    (mV)
    mg  = 1    (mM)
    gmax = 0.001 (uS)            : nS -> uS
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
    pr_ref
    pr_last
    ca_last
    factor_AMPA
    factor_NMDA
    mggate
}

STATE {
    c
    rho
    A_AMPA
    B_AMPA
    A_NMDA
    B_NMDA
}

INITIAL {
    LOCAL tp_AMPA, tp_NMDA
    c = 0
    rho = rho0
    A_AMPA = 0
    B_AMPA = 0
    A_NMDA = 0
    B_NMDA = 0
    pr_last = 0
    ca_last = 0
    : first pulse into a rested synapse has Pr = Use (u0=0 -> u=Use, R=1).
    : dividing by it makes the first pulse identical to model A.
    if (norm_Pr != 0 && Use > 0) {
        pr_ref = Use
    } else {
        pr_ref = 1
    }
    tp_AMPA = (tau_r_AMPA*tau_d_AMPA)/(tau_d_AMPA-tau_r_AMPA)*log(tau_d_AMPA/tau_r_AMPA)
    factor_AMPA = -exp(-tp_AMPA/tau_r_AMPA)+exp(-tp_AMPA/tau_d_AMPA)
    factor_AMPA = 1/factor_AMPA
    tp_NMDA = (tau_r_NMDA*tau_d_NMDA)/(tau_d_NMDA-tau_r_NMDA)*log(tau_d_NMDA/tau_r_NMDA)
    factor_NMDA = -exp(-tp_NMDA/tau_r_NMDA)+exp(-tp_NMDA/tau_d_NMDA)
    factor_NMDA = 1/factor_NMDA
}

BREAKPOINT {
    SOLVE state METHOD derivimplicit
    w = w0 + rho*(b*w0 - w0)
    mggate = 1 / (1 + exp(0.062 (/mV) * -(v)) * (mg / 2.62 (mM)))
    g_AMPA = gmax*(B_AMPA - A_AMPA)
    g_NMDA = gmax*(B_NMDA - A_NMDA) * mggate
    g = g_AMPA + g_NMDA
    i_AMPA = g_AMPA*(v - e)
    i_NMDA = g_NMDA*(v - e)
    i = i_AMPA + i_NMDA
}

FUNCTION heav(x) {
    if (x > 0) {
        heav = 1
    } else {
        heav = 0
    }
}

DERIVATIVE state {
    A_AMPA' = -A_AMPA/tau_r_AMPA
    B_AMPA' = -B_AMPA/tau_d_AMPA
    A_NMDA' = -A_NMDA/tau_r_NMDA
    B_NMDA' = -B_NMDA/tau_d_NMDA
    c' = -c/tau_ca
    rho' = (-rho*(1 - rho)*(rho_star - rho) + gamma_p*(1 - rho)*heav(c - theta_p) - gamma_d*rho*heav(c - theta_d)) / tau
}

NET_RECEIVE(weight, R, Pr, u, tsyn (ms)) {
    LOCAL prn, ca_amt

    INITIAL {
        R = 1
        u = u0
        tsyn = t
    }

    if (flag >= 2) {
        : delayed calcium; the amount travelled inside the flag (see COMMENT above)
        c = c + (flag - 2)
    } else {
        if (weight >= 0) {
            : ---- PRE spike ----
            : (1) short-term update, Fuhrmann et al. 2002 Eq.2-3, same as DetAMPANMDA
            if (Fac > 0) {
                u = u*exp(-(t - tsyn)/Fac)
            } else {
                u = Use
            }
            if (Fac > 0) {
                u = u + Use*(1 - u)
            }
            R  = 1 - (1 - R) * exp(-(t - tsyn)/Dep)
            Pr = u * R
            R  = R - u * R
            tsyn = t
            prn = Pr / pr_ref
            pr_last = prn

            : (2) transmission = short term (prn) * long term (w) * NetCon weight
            A_AMPA = A_AMPA + prn*weight*w*factor_AMPA
            B_AMPA = B_AMPA + prn*weight*w*factor_AMPA
            A_NMDA = A_NMDA + prn*weight*w*NMDA_ratio*factor_NMDA
            B_NMDA = B_NMDA + prn*weight*w*NMDA_ratio*factor_NMDA

            : (3) schedule the delayed calcium; ca_stp=0 keeps C_pre fixed
            ca_amt = C_pre * (1 + ca_stp*(prn - 1))
            if (ca_amt < 0) {
                ca_amt = 0
            }
            ca_last = ca_amt
            net_send(D, 2 + ca_amt)
        } else {
            : ---- POST spike (weight<0 sentinel) ---- immediate calcium, TM state untouched
            c = c + C_post
        }
    }
}
