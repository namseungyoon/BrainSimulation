"""Check plastic connection via GetConnectionStatus (reads device: syn_group + weight)."""
import sys
import nestgpu as ngpu
MODEL = sys.argv[1] if len(sys.argv) > 1 else "graupner"
ngpu.SetRandomSeed(1); ngpu.SetTimeResolution(0.1)
N, ISI, LAG = 60, 50.0, 10.0
pre_times = [100.0 + ISI*i for i in range(N)]
post_times = [t + LAG for t in pre_times]
sgp = ngpu.Create("spike_generator"); ngpu.SetStatus(sgp, {"spike_times": pre_times})
sgq = ngpu.Create("spike_generator"); ngpu.SetStatus(sgq, {"spike_times": post_times})
n0 = ngpu.Create("aeif_cond_beta"); n1 = ngpu.Create("aeif_cond_beta")
if MODEL == "stdp":
    g = ngpu.CreateSynGroup("stdp", {"lambda": 0.02, "Wmax": 10.0}); winit = 2.5
else:
    g = ngpu.CreateSynGroup("graupner", {"w0": 1.0, "w1": 5.28145}); winit = 3.14072
print("syn_group object:", g, "int:", int(g) if hasattr(g,"__int__") else "n/a")
ngpu.Connect(sgp, n0, {"rule": "one_to_one"}, {"weight": 17.9, "delay": 1.0})
ngpu.Connect(sgq, n1, {"rule": "one_to_one"}, {"weight": 17.9, "delay": 1.0})
ngpu.Connect(n0, n1, {"rule": "one_to_one"}, {"weight": winit, "delay": 1.0, "synapse_group": g})

def show(tag):
    conns = ngpu.GetConnections(n0, n1)
    st = ngpu.GetConnectionStatus(conns)
    for s in st:
        print(f"  [{tag}] syn_group={s['syn_group']} weight={s['weight']:.5f} port={s['port']} delay={s['delay']}")

ngpu.Simulate(50.0)
show("baseline")
ngpu.Simulate(pre_times[-1] + 400.0 - 50.0)
show("after")
