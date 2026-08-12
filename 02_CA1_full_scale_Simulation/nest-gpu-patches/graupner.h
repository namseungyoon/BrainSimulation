/*
 *  graupner.h
 *
 *  Calcium-based plasticity synapse (Graupner & Brunel 2012) for NEST GPU.
 *  Local addition (not upstream). Tracked via nest-gpu-patches.
 *
 *  Implementation note (fidelity): the NEST-GPU synapse ABI exposes only a
 *  single mutable per-synapse float (the weight) and fires SynapseUpdate on
 *  spike PAIRS with a signed time lag Dt. We therefore implement the Graupner
 *  model as an event-driven PAIRWISE (nearest-neighbour) calcium kernel:
 *  the two calcium contributions (pre delayed by D, post) share the same decay
 *  tau_Ca, so c(t) is piecewise a single decaying exponential and the time spent
 *  above the depression / potentiation thresholds (alpha_d, alpha_p) is analytic.
 *  The synaptic efficacy rho is carried IN the weight via the linear read-out
 *  w = w0 + rho*(w1-w0) (bijective). The slow cubic double-well drift
 *  (quiescent consolidation) is NOT integrated between events -- that would need
 *  persistent per-synapse calcium state (a larger core change). This kernel is
 *  faithful for spike-timing (STDP-curve) induction and approximate for dense
 *  high-frequency bursts (calcium summation across >2 spikes is not accumulated).
 */

#ifndef GRAUPNER_H
#define GRAUPNER_H
#include <cmath>
#include <string>

namespace graupner_ns
{
enum ParamIndexes
{
  i_C_pre = 0,
  i_C_post,
  i_tau_Ca,
  i_D,
  i_theta_d,
  i_theta_p,
  i_gamma_d,
  i_gamma_p,
  i_tau_rho,
  i_w0,
  i_w1,
  N_PARAM
};

const std::string graupner_param_name[ N_PARAM ] = { "C_pre",
  "C_post",
  "tau_Ca",
  "D",
  "theta_d",
  "theta_p",
  "gamma_d",
  "gamma_p",
  "tau_rho",
  "w0",
  "w1" };

// Time a single decaying exponential c0*exp(-t/tau) spends above threshold th,
// restricted to the window [0, span]. Monotone-decreasing -> closed form.
__device__ __forceinline__ float
graupner_time_above( float c0, float th, float tau, float span )
{
  if ( c0 <= th )
  {
    return 0.0f;
  }
  float t_cross = tau * logf( c0 / th );
  return fminf( t_cross, span );
}

__device__ __forceinline__ void
GraupnerUpdate( float* weight_pt, float Dt, float* param )
{
  float C_pre = param[ i_C_pre ];
  float C_post = param[ i_C_post ];
  float tauCa = param[ i_tau_Ca ];
  float D = param[ i_D ];
  float th_d = param[ i_theta_d ];
  float th_p = param[ i_theta_p ];
  float g_d = param[ i_gamma_d ];
  float g_p = param[ i_gamma_p ];
  float tau = param[ i_tau_rho ];
  float w0 = param[ i_w0 ];
  float w1 = param[ i_w1 ];

  // Reference t_pre = 0: pre calcium jump lands at +D (amp C_pre); post jump at Dt (amp C_post).
  float t_pre = D;
  float t_post = Dt;
  float t1, t2, a1, a2;
  if ( t_pre <= t_post )
  {
    t1 = t_pre;
    a1 = C_pre;
    t2 = t_post;
    a2 = C_post;
  }
  else
  {
    t1 = t_post;
    a1 = C_post;
    t2 = t_pre;
    a2 = C_pre;
  }
  float span = t2 - t1; // >= 0

  // Phase A: [t1, t2) only the first jump is present: c = a1*exp(-(t-t1)/tauCa)
  float aA_p = graupner_time_above( a1, th_p, tauCa, span );
  float aA_d = graupner_time_above( a1, th_d, tauCa, span );
  float cA_end = a1 * expf( -span / tauCa );

  // Phase B: [t2, inf) both jumps present, single exponential from (cA_end + a2)
  float cB0 = cA_end + a2;
  float aB_p = graupner_time_above( cB0, th_p, tauCa, 1.0e30f );
  float aB_d = graupner_time_above( cB0, th_d, tauCa, 1.0e30f );

  float alpha_p = aA_p + aB_p; // total time c > theta_p (potentiation active)
  float alpha_d = aA_d + aB_d; // total time c > theta_d (depression active)

  // Recover efficacy rho from the weight (linear read-out), update, write back.
  float denom = w1 - w0;
  float rho = ( denom != 0.0f ) ? ( ( *weight_pt ) - w0 ) / denom : 0.0f;
  // Efficacy change over the (ms-scale) transient; slow cubic drift neglected (tau >> transient).
  float drho = ( g_p * ( 1.0f - rho ) * alpha_p - g_d * rho * alpha_d ) / tau;
  rho += drho;
  rho = fminf( fmaxf( rho, 0.0f ), 1.0f );
  *weight_pt = w0 + rho * denom;
}
}

#endif
