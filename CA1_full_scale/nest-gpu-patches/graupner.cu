/*
 *  graupner.cu
 *
 *  Calcium-based plasticity synapse (Graupner & Brunel 2012) for NEST GPU.
 *  Local addition (not upstream). Tracked via nest-gpu-patches.
 *
 *  Default parameters: hippocampal CA3->CA1 slice set fitted by Graupner & Brunel
 *  to Wittenberg & Wang (2006), as in papers/02 plasticity_model.py.
 */

#include "cuda_error.h"
#include "graupner.h"
#include "ngpu_exception.h"
#include "syn_model.h"
#include <config.h>
#include <iostream>
#include <stdio.h>

using namespace graupner_ns;

int
Graupner::_Init()
{
  type_ = i_graupner_model;
  n_param_ = N_PARAM;
  param_name_ = graupner_param_name;
  CUDAMALLOCCTRL( "&d_param_arr_", &d_param_arr_, n_param_ * sizeof( float ) );
  // Wittenberg & Wang (2006) CA3->CA1 hippocampal-slice fit (Graupner & Brunel 2012).
  SetParam( "C_pre", 1.0 );
  SetParam( "C_post", 0.275865 );
  SetParam( "tau_Ca", 48.8373 );  // ms
  SetParam( "D", 18.8008 );        // ms (pre calcium delay)
  SetParam( "theta_d", 1.0 );
  SetParam( "theta_p", 1.3 );
  SetParam( "gamma_d", 313.0965 );
  SetParam( "gamma_p", 1645.59 );
  SetParam( "tau_rho", 688355.0 ); // ms (very slow efficacy time constant)
  SetParam( "w0", 1.0 );           // DOWN-state weight (nS); set per-projection by the caller
  SetParam( "w1", 5.28145 );       // UP-state weight = b * w0

  return 0;
}
