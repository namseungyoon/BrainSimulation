# cython: boundscheck=False, wraparound=False, cdivision=True
"""Cython RK4 kernel for candidate-only CCK/SCA intrinsic fitting."""

import numpy as np
cimport numpy as cnp
from libc.math cimport exp, fabs


cdef inline double _h_inf(double voltage, double half, double slope):
    cdef double exponent = (voltage-half)/slope
    if exponent < -80.0: exponent = -80.0
    if exponent > 80.0: exponent = 80.0
    return 1.0/(1.0+exp(exponent))


def user_m3_current_trace_summary(
    cnp.ndarray[cnp.float64_t, ndim=1] s,
    double current_nA,
    double dt,
    double duration,
    double stim_end,
    double v_half,
    double slope,
    double tau_h,
    double delta_h,
    double h_crit,
    double recovery_test_start=-1.0,
    double recovery_test_current_nA=0.0,
):
    """CPU RK4 reference for the CUDA user_m3 hybrid state machine."""
    cdef double c=s[0], tau=s[1], el=s[2], gc=s[3], gcd=s[4]
    cdef double cd=c*s[5], cx=cd*s[6], cap0=c-cd, cap1=cd-cx, cap2=cx
    cdef double dls=s[7], xls=s[8], ie=s[9], kad=s[10], k2p=s[11], k1p=s[12]
    cdef double vth=s[13], vreset=s[14], a2=s[15], a1=s[16], tref=s[17]
    cdef double v0=el, v1=el, v2=el, ia=0.0, idep=0.0, gate=_h_inf(el,v_half,slope)
    cdef int steps=<int>round(duration/dt), refr=0, refr_steps=<int>round(tref/dt)
    cdef int i, stage, spikes=0, recovery_spikes=0, measure_count=0
    cdef double now, injected, measure_sum=0.0, measure_min=1e9, measure_max=-1e9
    cdef double tv0,tv1,tv2,tia,tidep,tgate, voltage_for_rhs
    cdef double k[4][6]
    for i in range(steps):
        now=(i+1)*dt
        if 200.0 <= now < stim_end:
            injected=current_nA*1000.0
        elif recovery_test_start >= 0.0 and now >= recovery_test_start:
            injected=recovery_test_current_nA*1000.0
        else:
            injected=0.0
        for stage in range(4):
            if stage == 0:
                tv0=v0; tv1=v1; tv2=v2; tia=ia; tidep=idep; tgate=gate
            elif stage < 3:
                tv0=v0+0.5*dt*k[stage-1][0]; tv1=v1+0.5*dt*k[stage-1][1]
                tv2=v2+0.5*dt*k[stage-1][2]; tia=ia+0.5*dt*k[stage-1][3]
                tidep=idep+0.5*dt*k[stage-1][4]; tgate=gate+0.5*dt*k[stage-1][5]
            else:
                tv0=v0+dt*k[2][0]; tv1=v1+dt*k[2][1]; tv2=v2+dt*k[2][2]
                tia=ia+dt*k[2][3]; tidep=idep+dt*k[2][4]; tgate=gate+dt*k[2][5]
            voltage_for_rhs=vreset if refr > 0 else (tv0 if tgate <= h_crit else min(tv0,0.0))
            k[stage][0]=0.0 if refr > 0 else (-(cap0/tau)*(voltage_for_rhs-el)+gc*(tv1-voltage_for_rhs)-tia+tidep+ie+injected)/cap0
            k[stage][1]=(-(cap1/tau)*dls*(tv1-el)+gc*(voltage_for_rhs-tv1)+gcd*(tv2-tv1))/cap1
            k[stage][2]=(-(cap2/tau)*xls*(tv2-el)+gcd*(tv1-tv2))/cap2
            k[stage][3]=kad*(voltage_for_rhs-el)-k2p*tia
            k[stage][4]=-k1p*tidep
            k[stage][5]=(_h_inf(voltage_for_rhs,v_half,slope)-tgate)/tau_h
        v0 += dt*(k[0][0]+2*k[1][0]+2*k[2][0]+k[3][0])/6.0
        v1 += dt*(k[0][1]+2*k[1][1]+2*k[2][1]+k[3][1])/6.0
        v2 += dt*(k[0][2]+2*k[1][2]+2*k[2][2]+k[3][2])/6.0
        ia += dt*(k[0][3]+2*k[1][3]+2*k[2][3]+k[3][3])/6.0
        idep += dt*(k[0][4]+2*k[1][4]+2*k[2][4]+k[3][4])/6.0
        gate += dt*(k[0][5]+2*k[1][5]+2*k[2][5]+k[3][5])/6.0
        if gate < 0.0: gate=0.0
        if gate > 1.0: gate=1.0
        if refr > 0:
            v0=vreset; refr-=1
        elif v0 >= vth and gate > h_crit:
            if 800.0 <= now < stim_end: spikes+=1
            if recovery_test_start >= 0.0 and now >= recovery_test_start: recovery_spikes+=1
            gate=max(0.0,gate-delta_h); v0=vreset; ia+=a2; idep=a1; refr=refr_steps
        if 800.0 <= now < stim_end:
            measure_sum+=v0; measure_count+=1
            if v0 < measure_min: measure_min=v0
            if v0 > measure_max: measure_max=v0
    return np.asarray((spikes/(max(stim_end-800.0,dt)/1000.0),
                       measure_sum/max(measure_count,1),measure_min,measure_max,gate,
                       recovery_spikes),dtype=np.float64)


def current_trace_summary(
    cnp.ndarray[cnp.float64_t, ndim=1] s,
    double current_nA,
    double dt,
    double duration,
    double stim_end,
):
    cdef double c=s[0], tau=s[1], el=s[2], gc=s[3], gcd=s[4]
    cdef double cd=c*s[5], cx=cd*s[6], cap0=c-cd, cap1=cd-cx, cap2=cx
    cdef double dls=s[7], xls=s[8], ie=s[9], kad=s[10], k2p=s[11], k1p=s[12]
    cdef double vth=s[13], vreset=s[14], a2=s[15], a1=s[16], tref=s[17]
    cdef double v0=el, v1=el, v2=el, ia=0.0, idep=0.0
    cdef int steps=<int>round(duration/dt), refr=0, refr_steps=<int>round(tref/dt)
    cdef int i, stage, spikes=0, ss_count=0, start=<int>round(200.0/dt), end=<int>round(stim_end/dt)
    cdef double now, injected, ss_sum=0.0, vss, target, tau_found=60.0
    cdef double tv0,tv1,tv2,tia,tidep
    cdef double k[4][5]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] trace=np.empty(steps,dtype=np.float64)
    for i in range(steps):
        now=(i+1)*dt
        injected=current_nA*1000.0 if 200.0 <= now < stim_end else 0.0
        for stage in range(4):
            if stage == 0:
                tv0=v0; tv1=v1; tv2=v2; tia=ia; tidep=idep
            elif stage < 3:
                tv0=v0+0.5*dt*k[stage-1][0]; tv1=v1+0.5*dt*k[stage-1][1]
                tv2=v2+0.5*dt*k[stage-1][2]; tia=ia+0.5*dt*k[stage-1][3]; tidep=idep+0.5*dt*k[stage-1][4]
            else:
                tv0=v0+dt*k[2][0]; tv1=v1+dt*k[2][1]; tv2=v2+dt*k[2][2]
                tia=ia+dt*k[2][3]; tidep=idep+dt*k[2][4]
            k[stage][0]=0.0 if refr > 0 else (-(cap0/tau)*(tv0-el)+gc*(tv1-tv0)-tia+tidep+ie+injected)/cap0
            k[stage][1]=(-(cap1/tau)*dls*(tv1-el)+gc*(tv0-tv1)+gcd*(tv2-tv1))/cap1
            k[stage][2]=(-(cap2/tau)*xls*(tv2-el)+gcd*(tv1-tv2))/cap2
            k[stage][3]=kad*(tv0-el)-k2p*tia
            k[stage][4]=-k1p*tidep
        v0 += dt*(k[0][0]+2*k[1][0]+2*k[2][0]+k[3][0])/6.0
        v1 += dt*(k[0][1]+2*k[1][1]+2*k[2][1]+k[3][1])/6.0
        v2 += dt*(k[0][2]+2*k[1][2]+2*k[2][2]+k[3][2])/6.0
        ia += dt*(k[0][3]+2*k[1][3]+2*k[2][3]+k[3][3])/6.0
        idep += dt*(k[0][4]+2*k[1][4]+2*k[2][4]+k[3][4])/6.0
        if refr > 0:
            v0=vreset; refr-=1
        elif v0 >= vth:
            if 200.0 <= now < stim_end: spikes+=1
            v0=vreset; ia+=a2; idep=a1; refr=refr_steps
        trace[i]=v0
        if stim_end-50.0 <= now < stim_end:
            ss_sum+=v0; ss_count+=1
    if current_nA >= 0.0:
        return np.asarray((spikes,0.0,0.0),dtype=np.float64)
    vss=ss_sum/ss_count
    target=el-0.632*(el-vss)
    for i in range(start,end):
        if trace[i] <= target:
            tau_found=(i+1)*dt-200.0
            break
    if tau_found < 2.0: tau_found=2.0
    if tau_found > 60.0: tau_found=60.0
    return np.asarray((spikes,fabs((vss-el)/current_nA),tau_found),dtype=np.float64)
