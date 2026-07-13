# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Compiled CPU RK4 loop for exact_network_clamp_replay.py."""

import numpy as np
cimport numpy as cnp
from libc.math cimport exp, round


def simulate_user_m2(
    cnp.ndarray[cnp.uint16_t, ndim=2] event_counts,
    cnp.ndarray[cnp.float64_t, ndim=1] amplitudes,
    cnp.ndarray[cnp.float64_t, ndim=1] tau_rise,
    cnp.ndarray[cnp.float64_t, ndim=1] tau_decay,
    cnp.ndarray[cnp.float64_t, ndim=1] e_rev,
    cnp.ndarray[cnp.int64_t, ndim=1] domain,
    cnp.ndarray[cnp.uint8_t, ndim=2, cast=True] enabled,
    double dt, double duration_ms,
    cnp.ndarray[cnp.float64_t, ndim=1] status,
    double transient_ms=0.0,
    h_params=None,
    m4_params=None,
    m5_params=None,
):
    cdef Py_ssize_t na=enabled.shape[0], nr=enabled.shape[1], ns=event_counts.shape[1]
    cdef Py_ssize_t a,r,d,step,stage,prev,measure_start=<Py_ssize_t>round(transient_ms/dt),measure_steps=ns-measure_start
    cdef double factor,v0,v1,v2,ia,idp,gg,hh,s0,s1,s2,soma,tgate,vrhs,hexponent
    cdef double hp,hd,np_,nd,mp,md,hpi,hdi,npi,ndi,inap,inad,ikp,ikd,z
    cdef double bp,bd,bpi,bdi,sbp,sbd,cbp=1,cbd=1,glbp=0,glbd=0,gbp=0,gbd=0
    cdef double c0=status[0],c1=status[1],c2=status[2],el=status[3],tm=status[4]
    cdef double gc=status[5],gcd=status[6],dls=status[7],xls=status[8],ie=status[9]
    cdef double kad=status[10],k2p=status[11],k1p=status[12],vth=status[13]
    cdef double vreset=status[14],a2=status[15],a1=status[16],tref=status[17],initial=status[18]
    cdef long refr_steps=<long>round(tref/dt)
    cdef bint use_m3=h_params is not None
    cdef bint use_m4=m4_params is not None
    cdef bint use_m5=m5_params is not None
    cdef double vh=-41.95150313, kh=6.97901741, th=50.0, dh=0.3, hc=0.2
    if use_m3:
        vh=float(h_params[0]); kh=float(h_params[1]); th=float(h_params[2])
        dh=float(h_params[3]); hc=float(h_params[4])
    cdef double gnapp=0,gnadp=0,ena=55,vmh=-30,km=7,vhh=-45,khh=7,tauh=5
    cdef double gkpp=0,gkdp=0,ek=-90,vnh=-26,kn=8,taun=3
    if use_m4:
        gnapp=float(m4_params[0]); gnadp=float(m4_params[1]); ena=float(m4_params[2])
        vmh=float(m4_params[3]); km=float(m4_params[4]); vhh=float(m4_params[5])
        khh=float(m4_params[6]); tauh=float(m4_params[7]); gkpp=float(m4_params[8])
        gkdp=float(m4_params[9]); ek=float(m4_params[10]); vnh=float(m4_params[11])
        kn=float(m4_params[12]); taun=float(m4_params[13])
    if use_m5:
        cbp=float(m5_params[0]); cbd=float(m5_params[1])
        glbp=float(m5_params[2]); glbd=float(m5_params[3])
        gbp=float(m5_params[4]); gbd=float(m5_params[5])
        gnapp=float(m5_params[6]); gnadp=float(m5_params[7]); ena=float(m5_params[8])
        vmh=float(m5_params[9]); km=float(m5_params[10]); vhh=float(m5_params[11])
        khh=float(m5_params[12]); tauh=float(m5_params[13]); gkpp=float(m5_params[14])
        gkdp=float(m5_params[15]); ek=float(m5_params[16]); vnh=float(m5_params[17])
        kn=float(m5_params[18]); taun=float(m5_params[19])
    cdef cnp.ndarray[cnp.float64_t,ndim=2] vm=np.full((na,3),initial,dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t,ndim=1] iad=np.zeros(na), idep=np.zeros(na)
    cdef cnp.ndarray[cnp.float64_t,ndim=2] g=np.zeros((na,nr)), g1=np.zeros((na,nr))
    cdef cnp.ndarray[cnp.int64_t,ndim=1] refr=np.zeros(na,dtype=np.int64), spikes=np.zeros(na,dtype=np.int64)
    cdef cnp.ndarray[cnp.float64_t,ndim=2] vsum=np.zeros((na,3))
    cdef cnp.ndarray[cnp.float64_t,ndim=1] vmax=np.full(na,-1e300)
    cdef cnp.ndarray[cnp.float64_t,ndim=2] gpeak=np.zeros((na,3))
    cdef cnp.ndarray[cnp.float64_t,ndim=3] kvm=np.zeros((4,na,3))
    cdef cnp.ndarray[cnp.float64_t,ndim=2] kiad=np.zeros((4,na)), kidep=np.zeros((4,na))
    cdef cnp.ndarray[cnp.float64_t,ndim=1] gate=np.ones(na)
    cdef cnp.ndarray[cnp.float64_t,ndim=2] kgate=np.zeros((4,na))
    cdef cnp.ndarray[cnp.float64_t,ndim=2] hm4=np.zeros((na,2)), nm4=np.zeros((na,2))
    cdef cnp.ndarray[cnp.float64_t,ndim=3] khm4=np.zeros((4,na,2)), knm4=np.zeros((4,na,2))
    cdef cnp.ndarray[cnp.float64_t,ndim=3] kg=np.zeros((4,na,nr)), kg1=np.zeros((4,na,nr))
    cdef cnp.ndarray[cnp.float64_t,ndim=2] vb=np.full((na,2),initial,dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t,ndim=3] kvb=np.zeros((4,na,2))
    # The first five columns are the historical ABI.  The appended columns are
    # diagnostic-only compartment means and total conductance peaks.
    cdef cnp.ndarray[cnp.float64_t,ndim=2] out=np.empty((na,10))
    if use_m3:
        hexponent=(initial-vh)/kh
        gate[:]=1.0/(1.0+exp(hexponent))
    if use_m4 or use_m5:
        z=(initial-vhh)/khh; hm4[:,:]=1.0/(1.0+exp(z))
        z=-(initial-vnh)/kn; nm4[:,:]=1.0/(1.0+exp(z))

    for step in range(ns):
        for a in range(na):
            for r in range(nr):
                if enabled[a,r] and event_counts[r,step]:
                    g1[a,r] += event_counts[r,step]*amplitudes[r]
        for stage in range(4):
            if stage==0:
                factor=0.0; prev=0
            elif stage<3:
                factor=0.5; prev=stage-1
            else:
                factor=1.0; prev=2
            for a in range(na):
                v0=vm[a,0]+factor*dt*kvm[prev,a,0]
                v1=vm[a,1]+factor*dt*kvm[prev,a,1]
                v2=vm[a,2]+factor*dt*kvm[prev,a,2]
                ia=iad[a]+factor*dt*kiad[prev,a]
                idp=idep[a]+factor*dt*kidep[prev,a]
                tgate=gate[a]+factor*dt*kgate[prev,a]
                hp=hm4[a,0]+factor*dt*khm4[prev,a,0]
                hd=hm4[a,1]+factor*dt*khm4[prev,a,1]
                np_=nm4[a,0]+factor*dt*knm4[prev,a,0]
                nd=nm4[a,1]+factor*dt*knm4[prev,a,1]
                bp=vb[a,0]+factor*dt*kvb[prev,a,0]
                bd=vb[a,1]+factor*dt*kvb[prev,a,1]
                vrhs=vreset if refr[a]>0 else (v0 if use_m3 and tgate<=hc else min(v0,0.0))
                s0=0.0; s1=0.0; s2=0.0; sbp=0.0; sbd=0.0
                for r in range(nr):
                    gg=g[a,r]+factor*dt*kg[prev,a,r]
                    hh=g1[a,r]+factor*dt*kg1[prev,a,r]
                    kg[stage,a,r]=hh-gg/tau_decay[r]
                    kg1[stage,a,r]=-hh/tau_rise[r]
                    if enabled[a,r]:
                        if domain[r]==0: s0 += gg*(e_rev[r]-vrhs)
                        elif domain[r]==1:
                            if use_m5: sbp += gg*(e_rev[r]-bp)
                            else: s1 += gg*(e_rev[r]-v1)
                        else:
                            if use_m5: sbd += gg*(e_rev[r]-bd)
                            else: s2 += gg*(e_rev[r]-v2)
                soma=(-(c0/tm)*(vrhs-el)+gc*(v1-vrhs)-ia+idp+ie+s0)/c0
                kvm[stage,a,0]=0.0 if refr[a]>0 else soma
                inap=0.0; inad=0.0; ikp=0.0; ikd=0.0
                if use_m4 or use_m5:
                    bpi=bp if use_m5 else v1; bdi=bd if use_m5 else v2
                    z=-(bpi-vmh)/km; z=max(-80.0,min(80.0,z)); mp=1.0/(1.0+exp(z))
                    z=-(bdi-vmh)/km; z=max(-80.0,min(80.0,z)); md=1.0/(1.0+exp(z))
                    inap=gnapp*mp*mp*mp*hp*(ena-bpi); inad=gnadp*md*md*md*hd*(ena-bdi)
                    ikp=gkpp*np_*np_*np_*np_*(ek-bpi); ikd=gkdp*nd*nd*nd*nd*(ek-bdi)
                    z=(bpi-vhh)/khh; z=max(-80.0,min(80.0,z)); hpi=1.0/(1.0+exp(z))
                    z=(bdi-vhh)/khh; z=max(-80.0,min(80.0,z)); hdi=1.0/(1.0+exp(z))
                    z=-(bpi-vnh)/kn; z=max(-80.0,min(80.0,z)); npi=1.0/(1.0+exp(z))
                    z=-(bdi-vnh)/kn; z=max(-80.0,min(80.0,z)); ndi=1.0/(1.0+exp(z))
                    khm4[stage,a,0]=(hpi-hp)/tauh; khm4[stage,a,1]=(hdi-hd)/tauh
                    knm4[stage,a,0]=(npi-np_)/taun; knm4[stage,a,1]=(ndi-nd)/taun
                if use_m5:
                    bpi=gbp*max(0.0,bp-v1); bdi=gbd*max(0.0,bd-v2)
                    kvm[stage,a,1]=(-(c1/tm)*dls*(v1-el)+gc*(vrhs-v1)+gcd*(v2-v1)+bpi)/c1
                    kvm[stage,a,2]=(-(c2/tm)*xls*(v2-el)+gcd*(v1-v2)+bdi)/c2
                    kvb[stage,a,0]=(-glbp*(bp-el)-bpi+sbp+inap+ikp)/cbp
                    kvb[stage,a,1]=(-glbd*(bd-el)-bdi+sbd+inad+ikd)/cbd
                else:
                    kvm[stage,a,1]=(-(c1/tm)*dls*(v1-el)+gc*(vrhs-v1)+gcd*(v2-v1)+s1+inap+ikp)/c1
                    kvm[stage,a,2]=(-(c2/tm)*xls*(v2-el)+gcd*(v1-v2)+s2+inad+ikd)/c2
                kiad[stage,a]=kad*(vrhs-el)-k2p*ia
                kidep[stage,a]=-k1p*idp
                if use_m3:
                    hexponent=(vrhs-vh)/kh
                    if hexponent < -80.0: hexponent=-80.0
                    if hexponent > 80.0: hexponent=80.0
                    kgate[stage,a]=(1.0/(1.0+exp(hexponent))-tgate)/th
        for a in range(na):
            for d in range(3):
                vm[a,d] += dt*(kvm[0,a,d]+2*kvm[1,a,d]+2*kvm[2,a,d]+kvm[3,a,d])/6.0
            iad[a] += dt*(kiad[0,a]+2*kiad[1,a]+2*kiad[2,a]+kiad[3,a])/6.0
            idep[a] += dt*(kidep[0,a]+2*kidep[1,a]+2*kidep[2,a]+kidep[3,a])/6.0
            if use_m3:
                gate[a] += dt*(kgate[0,a]+2*kgate[1,a]+2*kgate[2,a]+kgate[3,a])/6.0
                if gate[a]<0.0: gate[a]=0.0
                if gate[a]>1.0: gate[a]=1.0
            if use_m4 or use_m5:
                for d in range(2):
                    hm4[a,d] += dt*(khm4[0,a,d]+2*khm4[1,a,d]+2*khm4[2,a,d]+khm4[3,a,d])/6.0
                    nm4[a,d] += dt*(knm4[0,a,d]+2*knm4[1,a,d]+2*knm4[2,a,d]+knm4[3,a,d])/6.0
                    hm4[a,d]=max(0.0,min(1.0,hm4[a,d])); nm4[a,d]=max(0.0,min(1.0,nm4[a,d]))
                    if use_m5:
                        vb[a,d] += dt*(kvb[0,a,d]+2*kvb[1,a,d]+2*kvb[2,a,d]+kvb[3,a,d])/6.0
            for r in range(nr):
                g[a,r] += dt*(kg[0,a,r]+2*kg[1,a,r]+2*kg[2,a,r]+kg[3,a,r])/6.0
                g1[a,r] += dt*(kg1[0,a,r]+2*kg1[1,a,r]+2*kg1[2,a,r]+kg1[3,a,r])/6.0
            if refr[a]>0:
                vm[a,0]=vreset; refr[a]-=1
            elif vm[a,0]>=vth and (not use_m3 or gate[a]>hc):
                if step>=measure_start: spikes[a]+=1
                if use_m3: gate[a]=max(0.0,gate[a]-dh)
                vm[a,0]=vreset; iad[a]+=a2; idep[a]=a1; refr[a]=refr_steps
            if step>=measure_start:
                vsum[a,0]+=vm[a,0]
                vsum[a,1]+=vm[a,1]
                vsum[a,2]+=vm[a,2]
                if vm[a,0]>vmax[a]: vmax[a]=vm[a,0]
                for d in range(3):
                    s0=0.0
                    for r in range(nr):
                        if enabled[a,r] and domain[r]==d:
                            s0+=g[a,r]
                    if s0>gpeak[a,d]: gpeak[a,d]=s0
    for a in range(na):
        out[a,0]=spikes[a]; out[a,1]=spikes[a]/((duration_ms-transient_ms)/1000.0)
        out[a,2]=vsum[a,0]/measure_steps; out[a,3]=vmax[a]; out[a,4]=vth-vmax[a]
        out[a,5]=vsum[a,1]/measure_steps; out[a,6]=vsum[a,2]/measure_steps
        out[a,7]=gpeak[a,0]; out[a,8]=gpeak[a,1]; out[a,9]=gpeak[a,2]
    return out


def simulate_user_m7(
    cnp.ndarray[cnp.uint16_t, ndim=3] branch_event_counts,
    cnp.ndarray[cnp.uint16_t, ndim=2] soma_event_counts,
    cnp.ndarray[cnp.float64_t, ndim=1] amplitudes,
    cnp.ndarray[cnp.float64_t, ndim=1] tau_rise,
    cnp.ndarray[cnp.float64_t, ndim=1] tau_decay,
    cnp.ndarray[cnp.float64_t, ndim=1] e_rev,
    cnp.ndarray[cnp.int64_t, ndim=1] domain,
    double dt, double duration_ms,
    cnp.ndarray[cnp.float64_t, ndim=1] status,
    cnp.ndarray[cnp.float64_t, ndim=1] params,
):
    """Frozen CPU reference for PV user_m7 (Euler, used at both gate dt values)."""
    cdef Py_ssize_t nr=branch_event_counts.shape[0], ns=branch_event_counts.shape[2]
    cdef Py_ssize_t r,b,d,step
    cdef double c0=status[0],c1=status[1],c2=status[2],el=status[3],tm=status[4]
    cdef double gc=status[5],gcd=status[6],dls=status[7],xls=status[8],ie=status[9]
    cdef double kad=status[10],k2p=status[11],k1p=status[12],vth=status[13]
    cdef double vreset=status[14],a2=status[15],a1=status[16],tref=status[17],initial=status[18]
    cdef long refr=0,refr_steps=<long>round(tref/dt),spikes=0
    cdef double vm=initial,vd=initial,vx=initial,iad=0.0,idep=0.0,v,s0,syn,axial
    cdef double soma_rhs,prox_rhs,dist_rhs,m,ina,ik,z,dh,dn,vmax=-1e300
    cdef double vs0=0.0,vs1=0.0,vs2=0.0
    cdef cnp.ndarray[cnp.float64_t,ndim=1] g=np.zeros(nr),g1=np.zeros(nr)
    cdef cnp.ndarray[cnp.float64_t,ndim=2] bg=np.zeros((nr,4)),bg1=np.zeros((nr,4))
    cdef cnp.ndarray[cnp.float64_t,ndim=2] vb=np.full((2,4),initial,dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t,ndim=2] h=np.zeros((2,4)),n=np.zeros((2,4))
    cdef cnp.ndarray[cnp.float64_t,ndim=2] dvb=np.zeros((2,4)),dho=np.zeros((2,4)),dno=np.zeros((2,4))
    cdef double ena=params[40],vmh=params[41],km=params[42],vhh=params[43]
    cdef double kh=params[44],tauh=params[45],ek=params[46],vnh=params[47]
    cdef double kn=params[48],taun=params[49]
    z=(initial-vhh)/kh; h[:,:]=1.0/(1.0+exp(z))
    z=-(initial-vnh)/kn; n[:,:]=1.0/(1.0+exp(z))
    for step in range(ns):
        for r in range(nr):
            if domain[r]==0 and soma_event_counts[r,step]:
                g1[r]+=soma_event_counts[r,step]*amplitudes[r]
            elif domain[r]>0:
                for b in range(4):
                    if branch_event_counts[r,b,step]:
                        bg1[r,b]+=branch_event_counts[r,b,step]*amplitudes[r]
        for r in range(nr):
            g[r]+=dt*(g1[r]-g[r]/tau_decay[r]); g1[r]+=dt*(-g1[r]/tau_rise[r])
            for b in range(4):
                bg[r,b]+=dt*(bg1[r,b]-bg[r,b]/tau_decay[r])
                bg1[r,b]+=dt*(-bg1[r,b]/tau_rise[r])
        prox_rhs=0.0; dist_rhs=0.0
        for d in range(2):
            for b in range(4):
                syn=0.0
                for r in range(nr):
                    if domain[r]==d+1: syn+=bg[r,b]*(e_rev[r]-vb[d,b])
                axial=params[(16 if d==0 else 20)+b]*((vd if d==0 else vx)-vb[d,b])
                if d==0: prox_rhs-=axial
                else: dist_rhs-=axial
                z=-(vb[d,b]-vmh)/km; z=max(-80.0,min(80.0,z)); m=1.0/(1.0+exp(z))
                ina=params[(24 if d==0 else 28)+b]*m*m*m*h[d,b]*(ena-vb[d,b])
                ik=params[(32 if d==0 else 36)+b]*n[d,b]*n[d,b]*n[d,b]*n[d,b]*(ek-vb[d,b])
                dvb[d,b]=(-params[(8 if d==0 else 12)+b]*(vb[d,b]-el)+axial+syn+ina+ik)/params[(0 if d==0 else 4)+b]
                z=(vb[d,b]-vhh)/kh; z=max(-80.0,min(80.0,z)); dh=(1.0/(1.0+exp(z))-h[d,b])/tauh
                z=-(vb[d,b]-vnh)/kn; z=max(-80.0,min(80.0,z)); dn=(1.0/(1.0+exp(z))-n[d,b])/taun
                dho[d,b]=dh; dno[d,b]=dn
        v=vreset if refr>0 else min(vm,0.0); s0=0.0
        for r in range(nr):
            if domain[r]==0: s0+=g[r]*(e_rev[r]-v)
        soma_rhs=0.0 if refr>0 else (-(c0/tm)*(v-el)+gc*(vd-v)-iad+idep+ie+s0)/c0
        prox_rhs=(-(c1/tm)*dls*(vd-el)+gc*(v-vd)+gcd*(vx-vd)+prox_rhs)/c1
        dist_rhs=(-(c2/tm)*xls*(vx-el)+gcd*(vd-vx)+dist_rhs)/c2
        vm+=dt*soma_rhs; vd+=dt*prox_rhs; vx+=dt*dist_rhs
        iad+=dt*(kad*(v-el)-k2p*iad); idep+=dt*(-k1p*idep)
        for d in range(2):
            for b in range(4):
                vb[d,b]+=dt*dvb[d,b]; h[d,b]=max(0.0,min(1.0,h[d,b]+dt*dho[d,b]))
                n[d,b]=max(0.0,min(1.0,n[d,b]+dt*dno[d,b]))
        if refr>0: vm=vreset; refr-=1
        elif vm>=vth: spikes+=1; vm=vreset; iad+=a2; idep=a1; refr=refr_steps
        vs0+=vm; vs1+=vd; vs2+=vx; vmax=max(vmax,vm)
    return np.asarray([[spikes,spikes/(duration_ms/1000.0),vs0/ns,vmax,vth-vmax,
                        vs1/ns,vs2/ns,0.0,0.0,0.0]],dtype=np.float64)
