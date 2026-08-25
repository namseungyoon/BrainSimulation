COMMENT
GluSynapseCa.mod - spine calcium plasticity synapse where calcium is generated from the
LOCAL MEMBRANE VOLTAGE, not from spike events.  Written for the 04 track, stage 5-7.

NOTE: NMODL is ASCII-only.  The full rationale (in Korean), including exactly what is and
      is not taken from the literature, lives next to this file in GluSynapseCa.md.
      READ THAT FIRST - especially the section on what this is NOT.

WHAT THIS IS / IS NOT
    This is OUR implementation of the CONCEPT "postsynaptic calcium comes from the local
    voltage and the actual glutamate release", inspired by Chindemi et al. 2022
    (Nat Commun 13:3038).  It is NOT a reproduction of that model: we did not use their
    source and we did not fit their parameters.  Chindemi 2022 is a NEOCORTEX model; CA1
    use would be an extrapolation anyway.  Every quantitative claim from this mod must be
    reported as "our implementation", never as "Chindemi 2022 reproduced".

WHY IT EXISTS - it is the only engine that can falsify two registered gaps
    GAPS G3: the Graupner-Brunel engines treat the post-synaptic calcium contribution as a
      CONSTANT, so a synapse 670 um out on the apical trunk gets the same calcium as one
      144 um out on a basal dendrite - even though stage 3-9 measured the bAP there at
      4.2 mV vs 92.6 mV.  Here calcium follows the local voltage, so distance matters.
    GAPS G5: with the paper-original convention (ca_stp=0) the GB engines inject calcium
      even when vesicle release FAILED.  Here the NMDA branch of the calcium is
      proportional to the NMDA conductance, which is zero on a failure - no glutamate,
      no NMDA current, no NMDA calcium.

TWO CALCIUM SOURCES, DELIBERATELY SEPARATED
    (1) NMDA branch   - needs glutamate AND depolarisation (Mg block relief).
                        This is the coincidence detector.
    (2) VDCC branch   - voltage-gated, opens on a back-propagating AP alone.
                        Real spines do get calcium from a bAP even with no release, so a
                        model with only branch (1) would be too extreme in the other
                        direction.  Keeping them separate lets 5-7 report their shares.

        dc/dt = -c/tau_ca
                + k_nmda * (g_NMDA/gmax) * (e_ca - v)/norm_mV
                + k_vdcc * m_vdcc(v)    * (e_ca - v)/norm_mV
        m_vdcc(v) = 1 / (1 + exp(-(v - vh_vdcc)/slope_vdcc))

    c is DIMENSIONLESS, on the same scale as GBPlasticitySyn, so that theta_d = 1.0 and
    theta_p = 1.3 keep their meaning and the engines are directly comparable.  The gains
    k_nmda and k_vdcc are OURS, calibrated in stage 5-7 (see the .md).

EFFICACY - identical to Graupner-Brunel on purpose
        tau*drho/dt = -rho(1-rho)(rho_star-rho)
                      + gamma_p(1-rho) Heav[c-theta_p] - gamma_d rho Heav[c-theta_d]
    Keeping the efficacy equation identical isolates the ONE thing we changed: where the
    calcium comes from.  If results differ, the calcium source is why.

TRANSMISSION - identical in structure to GBPlasticitySyn and PairSTDPSyn
        A_AMPA += weight * w * factor_AMPA      (w = w0 + rho*(b*w0 - w0))
        g = gmax*(B_AMPA-A_AMPA) + gmax*(B_NMDA-A_NMDA)*mggate
    Same kernels, same Mg gate, same units (gmax in uS, NetCon weight = 1.0 flag),
    same weight range - required by stage 5-10 (first-pulse matching).

NO POST-SYNAPTIC SENTINEL NetCon
    ** Do NOT wire a weight<0 NetCon to this mod. **  The GB engines need one because they
    learn about post-synaptic spikes as events.  This mod reads the local voltage directly,
    so a sentinel would be double counting.  The registry declares post_nc=False and
    lib/synprobe refuses the connection - the contract is enforced, not just documented
    (stage 5-8, decision D29).

RELEASE FAILURE
    A failure means no glutamate: A_NMDA/B_NMDA are not incremented, g_NMDA stays 0, and the
    NMDA branch contributes nothing.  The VDCC branch still runs, because it does not depend
    on release.  That asymmetry is the point of G5.

INITIALISATION - w IS SET IN INITIAL
    Same reason as PairSTDPSyn: the GB mods compute w only in BREAKPOINT, so an event at
    t=0 is transmitted with w=0 and silently vanishes (measured in 5-2, decision D22).

SOLVER - derivimplicit, and cnexp would be SILENTLY WRONG here
    The efficacy equation is CUBIC in rho.  With METHOD cnexp, nocmodl emits no warning but
    generates the update for a homogeneous decay,
        rho += (1 - exp(J*dt)) * (-rho),      J = d(drho/dt)/drho
    which is only correct when drho/dt = J*rho, i.e. when the steady state is zero.  Here it
    is not.  Measured consequence at rho = 0 with both thresholds crossed (dt = 0.025):
        correct step  +5.976e-05      cnexp step  0.000000000
    rho would never leave 0 - the synapse would appear to have no plasticity at all, with no
    error message.  GBPlasticitySyn uses derivimplicit for exactly this reason.
    Side benefit: this mod and the GB engines now share an integrator, so the stage 5-10
    first-pulse calibration factor between them is 1.0 (decision D27 concerns cnexp vs
    derivimplicit differences between engines).
ENDCOMMENT

NEURON {
    THREADSAFE
    POINT_PROCESS GluSynapseCa
    RANGE tau_ca, k_nmda, k_vdcc, e_ca, vh_vdcc, slope_vdcc, norm_mV
    RANGE theta_d, theta_p, gamma_d, gamma_p, tau, rho_star, rho0, b, w0
    RANGE tau_r_AMPA, tau_d_AMPA, tau_r_NMDA, tau_d_NMDA, NMDA_ratio, e, mg, gmax
    RANGE c, rho, w, g, g_AMPA, g_NMDA, i_AMPA, i_NMDA
    RANGE ca_nmda, ca_vdcc, m_vdcc, n_pre
    NONSPECIFIC_CURRENT i
}

UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (uS) = (microsiemens)
    (mM) = (milli/liter)
}

PARAMETER {
    : ---- calcium (OUR gains; tau_ca kept equal to GB for comparability) ----
    tau_ca     = 48.8373 (ms)
    k_nmda     = 1.0                 : NMDA branch gain   (5-7 calibrates)
    k_vdcc     = 0.0                 : VDCC branch gain   (5-7 calibrates)
    e_ca       = 40.0    (mV)        : effective calcium driving potential
    vh_vdcc    = -30.0   (mV)        : VDCC half activation
    slope_vdcc = 7.0     (mV)        : VDCC activation slope
    norm_mV    = 100.0   (mV)        : driving-force normaliser -> c dimensionless
    : ---- efficacy: identical to GBPlasticitySyn (Wittenberg2006 fit) ----
    theta_d  = 1.0
    theta_p  = 1.3
    gamma_d  = 313.0965
    gamma_p  = 1645.59
    tau      = 688355 (ms)
    rho_star = 0.5
    rho0     = 0.0
    b        = 5.28145
    w0       = 1.0
    : ---- transmission: identical structure to GBPlasticitySyn ----
    tau_r_AMPA = 0.2  (ms)
    tau_d_AMPA = 1.7  (ms)
    tau_r_NMDA = 9.0  (ms)
    tau_d_NMDA = 61.0 (ms)
    NMDA_ratio = 0.71
    e    = 0     (mV)
    mg   = 1     (mM)
    gmax = 0.001 (uS)
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
    mggate
    m_vdcc
    ca_nmda
    ca_vdcc
    n_pre
    factor_AMPA
    factor_NMDA
}

STATE {
    A_AMPA
    B_AMPA
    A_NMDA
    B_NMDA
    c
    rho
}

INITIAL {
    LOCAL tp_AMPA, tp_NMDA
    c   = 0
    rho = rho0
    w   = w0 + rho0*(b*w0 - w0)          : set HERE (D22 trap)
    n_pre = 0
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
    SOLVE state METHOD derivimplicit
    w = w0 + rho*(b*w0 - w0)
    mggate = 1 / (1 + exp(0.062 (/mV) * -(v)) * (mg / 2.62 (mM)))
    m_vdcc = 1 / (1 + exp(-(v - vh_vdcc)/slope_vdcc))
    g_AMPA = gmax*(B_AMPA - A_AMPA)
    g_NMDA = gmax*(B_NMDA - A_NMDA) * mggate
    g = g_AMPA + g_NMDA
    i_AMPA = g_AMPA*(v - e)
    i_NMDA = g_NMDA*(v - e)
    i = i_AMPA + i_NMDA
    : diagnostics - the two calcium branches, reported separately (stage 5-7)
    ca_nmda = k_nmda * (g_NMDA/gmax) * (e_ca - v)/norm_mV
    ca_vdcc = k_vdcc * m_vdcc * (e_ca - v)/norm_mV
}

DERIVATIVE state {
    LOCAL mgg, mv, gn
    A_AMPA' = -A_AMPA/tau_r_AMPA
    B_AMPA' = -B_AMPA/tau_d_AMPA
    A_NMDA' = -A_NMDA/tau_r_NMDA
    B_NMDA' = -B_NMDA/tau_d_NMDA
    : recomputed here because ASSIGNED values from BREAKPOINT are stale inside DERIVATIVE
    mgg = 1 / (1 + exp(0.062 (/mV) * -(v)) * (mg / 2.62 (mM)))
    mv  = 1 / (1 + exp(-(v - vh_vdcc)/slope_vdcc))
    gn  = (B_NMDA - A_NMDA) * mgg
    c' = -c/tau_ca
         + k_nmda * gn * (e_ca - v)/norm_mV
         + k_vdcc * mv * (e_ca - v)/norm_mV
    rho' = (-rho*(1-rho)*(rho_star-rho)
            + gamma_p*(1-rho)*heav(c - theta_p)
            - gamma_d*rho*heav(c - theta_d)) / tau
}

FUNCTION heav(x) {
    if (x > 0) {
        heav = 1
    } else {
        heav = 0
    }
}

NET_RECEIVE(weight (uS)) {
    if (weight >= 0) {
        : PRE spike - glutamate release.  No release -> this never runs -> no NMDA calcium.
        n_pre = n_pre + 1
        A_AMPA = A_AMPA + weight*w*factor_AMPA
        B_AMPA = B_AMPA + weight*w*factor_AMPA
        A_NMDA = A_NMDA + weight*w*NMDA_ratio*factor_NMDA
        B_NMDA = B_NMDA + weight*w*NMDA_ratio*factor_NMDA
    }
    : weight<0 (post sentinel) is deliberately IGNORED - see COMMENT.
}
