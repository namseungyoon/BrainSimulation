COMMENT
GBPlasticitySyn.mod - Graupner & Brunel (2012) calcium-based long-term plasticity
synapse, deterministic (sigma=0). Ported from
papers/02_Graupner2012.../plasticity_model.py to an NMODL POINT_PROCESS.

Source: Graupner & Brunel (2012) PNAS 109(10):3991-3996.

Calcium c (Fig 1A):
    pre spike  -> after delay D:  c += C_pre
    post spike -> immediately:    c += C_post
    decay: c' = -c/tau_ca   (equivalent to the sum-of-exponentials calcium_trace)
Efficacy rho (Eq.1, noise omitted):
    tau*drho/dt = -rho(1-rho)(rho_star-rho) + gamma_p(1-rho)*Heav[c-theta_p]
                                            - gamma_d*rho*Heav[c-theta_d]
    rho in [0,1], bistable (0=DOWN, 1=UP, rho_star=0.5 boundary)
Transmission: AMPA/NMDA dual-exponential, effective conductance
    = weight * (w0 + rho*(w1-w0)),  w1 = b*w0
    (Chindemi 2022 style: deterministic rho scales transmission strength)

GPU-friendly (CoreNEURON): DERIVATIVE derivimplicit; pre delay via a single
net_send self-event (no vectors/queues -> avoids issue #638); no Random123
(deterministic); post spike delivered by a weight<0 sentinel NetCon.

Default parameters = Wittenberg2006 hippocampal slice fit (Table S2).
tau in seconds is converted to ms (x1000).
Validation (E8.1): rho(t) of this mod matches Python integrate_rho(noise=False).
ENDCOMMENT

NEURON {
    POINT_PROCESS GBPlasticitySyn
    RANGE tau_ca, C_pre, C_post, D
    RANGE theta_d, theta_p, gamma_d, gamma_p, tau, rho_star, rho0, b, w0
    RANGE tau_r_AMPA, tau_d_AMPA, tau_r_NMDA, tau_d_NMDA, NMDA_ratio, e, mg, gmax
    RANGE c, rho, w, g, g_AMPA, g_NMDA, i_AMPA, i_NMDA
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
    : efficacy
    theta_d = 1.0
    theta_p = 1.3
    gamma_d = 313.0965
    gamma_p = 1645.59
    tau     = 688355   (ms)      : 688.355 s * 1000
    rho_star = 0.5
    rho0    = 0.0                 : initial efficacy
    b       = 5.28145            : w1/w0
    w0      = 1.0
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

NET_RECEIVE(weight (uS)) {
    if (flag == 2) {
        : delayed pre-induced calcium jump (at t_pre + D)
        c = c + C_pre
    } else {
        if (weight >= 0) {
            : PRE spike - AMPA/NMDA transmission (scaled by efficacy rho) + schedule delayed calcium
            A_AMPA = A_AMPA + weight*w*factor_AMPA
            B_AMPA = B_AMPA + weight*w*factor_AMPA
            A_NMDA = A_NMDA + weight*w*NMDA_ratio*factor_NMDA
            B_NMDA = B_NMDA + weight*w*NMDA_ratio*factor_NMDA
            net_send(D, 2)
        } else {
            : POST spike (weight<0 sentinel) - immediate calcium jump
            c = c + C_post
        }
    }
}
