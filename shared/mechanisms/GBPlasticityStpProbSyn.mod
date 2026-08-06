COMMENT
GBPlasticityStpProbSyn.mod - stochastic (multi-vesicular) release
                             + short-term (Tsodyks-Markram)
                             + long-term  (Graupner-Brunel)
"Model C".

NOTE: NMODL is ASCII-only, so the full rationale (in Korean) lives next to this file
      in GBPlasticityStpProbSyn.md.  Read that first.

Built by combining three existing mods; no new physics was invented here.
  * long  term (efficacy rho, ~11.5 min) : GBPlasticitySyn.mod  (Graupner & Brunel 2012 PNAS)
  * short term (facilitation u, ~100 ms) : TM block of DetAMPANMDA.mod
                                           (Fuhrmann et al. 2002)
  * stochastic release (per-vesicle)     : MVR block of ProbAMPANMDA_EMS.mod
                                           (Blue Brain Project; Random123 stream)

WHAT IS DIFFERENT FROM MODEL B (GBPlasticityStpSyn)
    Model B computes a deterministic release amount  Pr = u * R  and scales the
    conductance by it.  That is the ENSEMBLE AVERAGE over many trials.
    Model C instead simulates the vesicles: Nrrp release sites, each either occupied
    or empty; on a pre spike every occupied site releases with probability u, and
    every empty site recovers with probability 1 - exp(-(t-tsyn)/Dep).
    The released fraction  ves/Nrrp  replaces  Pr.  Averaged over trials the two
    agree; on a SINGLE trial model C fails to release at all with probability
    (1-u)^Nrrp.  With the SC->PC (E1s) values (Use=0.15, Nrrp=1) that is 85% of pulses.

RELEASE STATE IS PER-SYNAPSE, NOT PER-NETCON
    u / tsyn / occupied / unoccupied live in ASSIGNED (same as ProbAMPANMDA_EMS),
    not in the NET_RECEIVE argument list.  This synapse is driven by exactly two
    NetCons - one presynaptic fibre (weight > 0) and one post-spike sentinel
    (weight < 0) - and the sentinel must NOT touch the release state.

COUPLING (multiplicative, same shape as model B)
    amplitude = prn * w * weight
    w   = w0 + rho*(b*w0 - w0)      <- Graupner, unchanged
    prn = (ves/Nrrp) / pr_ref       <- stochastic release, normalised

OUR CHOICE 1 - release normalisation (norm_Pr, default ON)
    pr_ref = Use.  On the first pulse into a rested synapse every site is occupied
    and u = Use, so E[ves/Nrrp] = Use and therefore E[prn] = 1.  The TRIAL-AVERAGED
    first pulse is then exactly the same size as in model A and model B, and the only
    difference between the three models is the dynamics, not the scale.
    WARNING - this holds for the AVERAGE only.  With Nrrp = 1 a single trial gives
    prn = 1/Use = 6.67 when it releases and prn = 0 when it does not.
    Same as in model B, this normalisation is OUR convention, not a published result.

OUR CHOICE 2 - should calcium follow release?  (ca_stp, default ON = 1)
    C_pre_eff = C_pre * (1 + ca_stp*(prn - 1)), clamped at 0.
      ca_stp = 0 -> fixed C_pre on EVERY pre spike, exactly as in Graupner & Brunel.
                    Transmission is stochastic but calcium is not.
      ca_stp = 1 -> calcium follows the actual release: none on a failure, and
                    1/Use = 6.67 times C_pre on a univesicular success.
    ** READ THIS BEFORE USING ca_stp = 1 WITH Nrrp = 1 **
    theta_p (potentiation threshold) is 1.3 and C_pre is 1.0.  A single successful
    release then injects 6.67, which is five times over threshold, so essentially
    every successful release potentiates.  That is a direct consequence of combining
    univesicular release with a normalisation built for the trial average - it is a
    modelling artefact of OUR construction, not a result.  Run ca_stp = 0 first.
    Graupner & Brunel 2012 has no release stochasticity at all, so neither branch is
    prescribed by the paper.  Both are ours; ca_stp = 0 is the conservative one.

ORDERING PROBLEM OF THE DELAYED CALCIUM, AND HOW IT IS SOLVED
    Same solution as model B.  The calcium of a pre spike is injected D = 18.8 ms
    later, and the 100 Hz inter-pulse interval (10 ms) is shorter than D, so several
    events are always in flight.  The amount rides IN THE FLAG of the self event:
    net_send(D, 2 + C_pre_eff), and the receiver does c += (flag - 2) for flag >= 2.
    flag is a double, each event carries its own copy, no queue and no ordering
    assumption.  External NetCon events arrive with flag == 0.

RNG
    setRNG(s1, s2, s3) installs a Random123 stream, exactly as in ProbAMPANMDA_EMS.
    ** The caller MUST call it. **  With no stream urand() returns 0.0 forever, so
    every RELEASE test (result < u) succeeds and every RECOVERY test (result > Psurv)
    fails.  With Nrrp = 1 the synapse then releases once on the very first pulse and
    is silent for the rest of the run - not "always release", but a dead synapse.
    Verified by check_gb_mods.py test 7: ves = [1, 0, 0, 0] over a 100 Hz burst.
    rng is a plain POINTER (not BBCOREPOINTER): this mod is for plain NEURON MPI, and
    CoreNEURON serialisation of the stream is deliberately not supported.

MISC
    * Defaults: calcium/efficacy = Wittenberg 2006 hippocampal slice fit (Graupner
      Table S2).  Short-term Use/Dep/Fac and Nrrp are supplied by the caller from
      params_table3 "SC->PC (E1s)" - WARNING, those are TUNED values, they are not in
      Ecker 2020 Table 3.
    * Diagnostics: pr_last (last prn), ca_last (last C_pre_eff), ves_last (vesicles
      released on the last pre spike), n_rel / n_pre (running success / spike counts).
      n_rel/n_pre should approach Use over a long run - that is the check that the
      stochastic block is wired correctly.
ENDCOMMENT

NEURON {
    THREADSAFE
    POINT_PROCESS GBPlasticityStpProbSyn
    RANGE tau_ca, C_pre, C_post, D
    RANGE theta_d, theta_p, gamma_d, gamma_p, tau, rho_star, rho0, b, w0
    RANGE Use, Dep, Fac, u0, Nrrp, norm_Pr, ca_stp, pr_ref
    RANGE u, tsyn, occupied, unoccupied
    RANGE tau_r_AMPA, tau_d_AMPA, tau_r_NMDA, tau_d_NMDA, NMDA_ratio, e, mg, gmax
    RANGE c, rho, w, g, g_AMPA, g_NMDA, i_AMPA, i_NMDA
    RANGE pr_last, ca_last, ves_last, n_pre, n_rel
    NONSPECIFIC_CURRENT i
    POINTER rng
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
    : release (short term + stochastic, Fuhrmann 2002 + BBP MVR)
    Use     = 1.0      (1)       : caller overrides with SC->PC value (0.15)
    Dep     = 100      (ms)      :   "                               (150)
    Fac     = 10       (ms)      :   "                               (250)
    Nrrp    = 1        (1)       : number of release sites (SC->PC E1s uses 1)
    u0      = 0
    norm_Pr = 1                  : 1 = scale mean first pulse to match models A/B
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

COMMENT
The Verbatim block is needed to draw uniform random numbers on [0,1) for the
release / recovery decisions.  Copied unchanged from ProbAMPANMDA_EMS.mod.
ENDCOMMENT

VERBATIM

#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#ifndef NRN_VERSION_GTEQ_8_2_0
#include "nrnran123.h"

#ifndef CORENEURON_BUILD
extern int ifarg(int iarg);
#endif

double nrn_random_pick(void* r);
void* nrn_random_arg(int argpos);
#define RANDCAST
#else
#define RANDCAST (Rand*)
#endif

ENDVERBATIM

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
    ves_last
    n_pre
    n_rel
    factor_AMPA
    factor_NMDA
    mggate
    : release state - PER SYNAPSE (see COMMENT), driven only by the weight>0 NetCon
    u (1)
    tsyn (ms)
    unoccupied (1)
    occupied   (1)
    rng
    usingR123
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
    ves_last = 0
    n_pre = 0
    n_rel = 0
    : release state: rested synapse = every site occupied
    u = u0
    tsyn = 0
    occupied = Nrrp
    unoccupied = 0
    : mean release of the first pulse into a rested synapse is Use
    : (every site occupied, each fires with probability u = Use).
    : dividing by it makes the TRIAL-AVERAGED first pulse identical to models A and B.
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

NET_RECEIVE(weight) {
    LOCAL prn, ca_amt, result, ves, Psurv

    if (flag >= 2) {
        : delayed calcium; the amount travelled inside the flag (see COMMENT above)
        c = c + (flag - 2)
    } else {
        if (weight >= 0) {
            : ---- PRE spike ----
            if (t < 0) {
                VERBATIM
                return;
                ENDVERBATIM
            }
            n_pre = n_pre + 1

            : (1) facilitation, Fuhrmann et al. 2002 Eq.2, same as DetAMPANMDA
            if (Fac > 0) {
                u = u*exp(-(t - tsyn)/Fac)
                u = u + Use*(1 - u)
            } else {
                u = Use
            }

            : (2) recovery - each empty site refills independently (BBP MVR)
            FROM counter = 0 TO (unoccupied - 1) {
                Psurv = exp(-(t - tsyn)/Dep)
                result = urand()
                if (result > Psurv) {
                    occupied = occupied + 1
                }
            }

            : (3) release - each occupied site fires with probability u
            ves = 0
            FROM counter = 0 TO (occupied - 1) {
                result = urand()
                if (result < u) {
                    ves = ves + 1
                }
            }
            occupied = occupied - ves
            unoccupied = Nrrp - occupied
            : tsyn tracks EVERY pre spike, released or not - both u and the recovery
            : clock are driven by the spike, not by the release.
            tsyn = t

            ves_last = ves
            prn = (ves/Nrrp) / pr_ref
            pr_last = prn
            if (ves > 0) {
                n_rel = n_rel + 1
            }

            : (4) transmission = release (prn) * efficacy (w) * NetCon weight
            if (ves > 0) {
                A_AMPA = A_AMPA + prn*weight*w*factor_AMPA
                B_AMPA = B_AMPA + prn*weight*w*factor_AMPA
                A_NMDA = A_NMDA + prn*weight*w*NMDA_ratio*factor_NMDA
                B_NMDA = B_NMDA + prn*weight*w*NMDA_ratio*factor_NMDA
            }

            : (5) schedule the delayed calcium; ca_stp=0 keeps C_pre fixed
            ca_amt = C_pre * (1 + ca_stp*(prn - 1))
            if (ca_amt < 0) {
                ca_amt = 0
            }
            ca_last = ca_amt
            if (ca_amt > 0) {
                net_send(D, 2 + ca_amt)
            }
        } else {
            : ---- POST spike (weight<0 sentinel) ---- immediate calcium.
            : Release state (u, tsyn, occupied) is deliberately NOT touched.
            c = c + C_post
        }
    }
}

PROCEDURE setRNG() {
VERBATIM
    #ifndef CORENEURON_BUILD
    // For compatibility, allow for either MCellRan4 or Random123
    // Distinguish by the arg types
    // Object => MCellRan4, seeds (double) => Random123
    usingR123 = 0;
    if( ifarg(1) && hoc_is_double_arg(1) ) {
        nrnran123_State** pv = (nrnran123_State**)(&_p_rng);
        uint32_t a2 = 0;
        uint32_t a3 = 0;

        if (*pv) {
            nrnran123_deletestream(*pv);
            *pv = (nrnran123_State*)0;
        }
        if (ifarg(2)) {
            a2 = (uint32_t)*getarg(2);
        }
        if (ifarg(3)) {
            a3 = (uint32_t)*getarg(3);
        }
        *pv = nrnran123_newstream3((uint32_t)*getarg(1), a2, a3);
        usingR123 = 1;
    } else if( ifarg(1) ) {   // not a double, so assume hoc object type
        void** pv = (void**)(&_p_rng);
        *pv = nrn_random_arg(1);
    } else {  // no arg, so clear pointer
        void** pv = (void**)(&_p_rng);
        *pv = (void*)0;
    }
    #endif
ENDVERBATIM
}

PROCEDURE clearRNG() {
VERBATIM
    #ifndef CORENEURON_BUILD
    if (usingR123) {
        nrnran123_State** pv = (nrnran123_State**)(&_p_rng);
        if (*pv) {
            nrnran123_deletestream(*pv);
            *pv = (nrnran123_State*)0;
        }
    } else {
        void** pv = (void**)(&_p_rng);
        if (*pv) {
            *pv = (void*)0;
        }
    }
    #endif
ENDVERBATIM
}

FUNCTION urand() {
VERBATIM
    double value = 0.0;
    if ( usingR123 ) {
        value = nrnran123_dblpick((nrnran123_State*)_p_rng);
    } else if (_p_rng) {
        #ifndef CORENEURON_BUILD
        value = nrn_random_pick(RANDCAST _p_rng);
        #endif
    } else {
        // No stream installed - see the RNG note in the header COMMENT.
        value = 0.0;
    }
    _lurand = value;
ENDVERBATIM
}
