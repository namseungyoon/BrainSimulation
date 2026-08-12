"""Single-synapse functional smoke for the graupner syn_model, mirroring the
official NEST-GPU stdp.py pattern: the PLASTIC synapse is neuron0 -> neuron1
(both real neurons), each driven to spike by its own spike_generator at a
controlled causal lag. Repeat N pairings and read the plastic weight before/after.
Usage: step5b_smoke_graupner.py <LAG_ms> [model]   (model: graupner|stdp)"""
import sys
import nestgpu as ngpu

LAG = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
MODEL = sys.argv[2] if len(sys.argv) > 2 else "graupner"
N, ISI = 60, 50.0

ngpu.SetRandomSeed(1)
ngpu.SetTimeResolution(0.1)

pre_times = [100.0 + ISI * i for i in range(N)]
post_times = [t + LAG for t in pre_times]

sgp = ngpu.Create("spike_generator"); ngpu.SetStatus(sgp, {"spike_times": pre_times})
sgq = ngpu.Create("spike_generator"); ngpu.SetStatus(sgq, {"spike_times": post_times})
n0 = ngpu.Create("aeif_cond_beta")   # presynaptic neuron of the plastic synapse
n1 = ngpu.Create("aeif_cond_beta")   # postsynaptic neuron

if MODEL == "stdp":
    g = ngpu.CreateSynGroup("stdp", {"lambda": 0.02, "Wmax": 10.0})
    W0, W1 = 0.0, 10.0
    w_init = 2.5
else:
    g = ngpu.CreateSynGroup("graupner", {"w0": 1.0, "w1": 5.28145})
    W0, W1 = 1.0, 5.28145
    w_init = W0 + 0.5 * (W1 - W0)   # rho0 = 0.5

# drive each neuron to spike (weight 17.9 -> one spike per generator pulse, per stdp.py)
ngpu.Connect(sgp, n0, {"rule": "one_to_one"}, {"weight": 17.9, "delay": 1.0})
ngpu.Connect(sgq, n1, {"rule": "one_to_one"}, {"weight": 17.9, "delay": 1.0})
# the plastic synapse under test
ngpu.Connect(n0, n1, {"rule": "one_to_one"},
             {"weight": w_init, "delay": 1.0, "synapse_group": g})

def wnow():
    return float(ngpu.GetStatus(ngpu.GetConnections(n0, n1), "weight")[0])

for nd in (n0, n1):
    try: ngpu.ActivateRecSpikeTimes(nd, 2000)
    except Exception as e: print("rec fail", e)

ngpu.Simulate(50.0)          # calibrate before first pairing (t=100)
w_before = wnow()
ngpu.Simulate(pre_times[-1] + 400.0 - 50.0)
w_after = wnow()

def nspk(nd):
    try:
        st = ngpu.GetRecSpikeTimes(nd); return len(st[0]) if st and hasattr(st[0],"__len__") else st
    except Exception as e: return f"err:{e}"
print(f"  spikes: n0={nspk(n0)}  n1={nspk(n1)}")
print(f"MODEL={MODEL} LAG={LAG:+.1f}ms N={N}")
print(f"  weight: {w_before:.5f} -> {w_after:.5f}  (Δ={w_after-w_before:+.5f}, {100*(w_after/w_before-1) if w_before else 0:+.2f}%)")
if W1 != W0:
    print(f"  rho:    {(w_before-W0)/(W1-W0):.4f} -> {(w_after-W0)/(W1-W0):.4f}")
print("RESPONDS" if abs(w_after - w_before) > 1e-6 else "NO CHANGE")
