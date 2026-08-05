"""Diagnose why plastic weight doesn't change: check post spiking + compare
built-in stdp (known-good) vs graupner, with spike recording."""
import sys
import nestgpu as ngpu

model = sys.argv[1] if len(sys.argv) > 1 else "graupner"
ngpu.SetRandomSeed(1); ngpu.SetTimeResolution(0.1)

N, ISI, LAG = 60, 50.0, 10.0
pre_times = [100.0 + ISI * i for i in range(N)]
post_times = [t + LAG for t in pre_times]

pre = ngpu.Create("spike_generator"); ngpu.SetStatus(pre, {"spike_times": pre_times})
drv = ngpu.Create("spike_generator"); ngpu.SetStatus(drv, {"spike_times": post_times})
post = ngpu.Create("aeif_cond_beta_multisynapse", 1)
ngpu.SetStatus(post, {"E_rev": [0.0], "tau_rise": [0.3], "tau_decay": [3.0]})

g = ngpu.CreateSynGroup(model)
if model == "stdp":
    ngpu.SetSynGroupParam(g, "lambda", 0.05)
    ngpu.SetSynGroupParam(g, "Wmax", 10.0)
    winit = 2.5
else:
    ngpu.SetSynGroupParam(g, "w0", 1.0)
    ngpu.SetSynGroupParam(g, "w1", 5.28145)
    winit = 3.14

ngpu.Connect(pre, post, {"rule": "one_to_one"},
             {"weight": winit, "delay": 1.0, "receptor": 0, "synapse_group": g})
ngpu.Connect(drv, post, {"rule": "one_to_one"},
             {"weight": 400.0, "delay": 1.0, "receptor": 0})

# record post spikes
try:
    ngpu.ActivateRecSpikeTimes(post, 2000)
    rec_ok = True
except Exception as e:
    print("rec API fail:", type(e).__name__, e); rec_ok = False

def wnow():
    return float(ngpu.GetStatus(ngpu.GetConnections(pre, post), "weight")[0])

ngpu.Simulate(50.0)
wb = wnow()
ngpu.Simulate(pre_times[-1] + 400.0 - 50.0)
wa = wnow()

nspk = "?"
if rec_ok:
    try:
        st = ngpu.GetRecSpikeTimes(post)
        nspk = len(st[0]) if st and hasattr(st[0], "__len__") else st
    except Exception as e:
        nspk = f"get fail: {e}"

print(f"MODEL={model}  post_spikes={nspk}  weight {wb:.4f} -> {wa:.4f}  (Δ={wa-wb:+.4f})")
