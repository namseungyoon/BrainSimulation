"""Headless replica of nest-gpu/python/examples/stdp.py — decisive test of whether
built-in STDP changes weights AT ALL in this build. Reports the spread of the N
plastic weights after a Delta-t sweep (should vary along the STDP curve)."""
import math
import nestgpu as ngpu

dt_step = 5.0
N = 50
tau_plus = tau_minus = 20.0
lambd = 1.0; alpha = 1.0; mu_plus = mu_minus = 1.0
Wmax = 0.001; den_delay = 0.0

syn_group = ngpu.CreateSynGroup("stdp", {"tau_plus": tau_plus, "tau_minus": tau_minus,
    "lambda": lambd, "alpha": alpha, "mu_plus": mu_plus, "mu_minus": mu_minus, "Wmax": Wmax})

sg = ngpu.Create("spike_generator")
neuron0 = ngpu.Create("aeif_cond_beta")
neuron1 = ngpu.Create("aeif_cond_beta", N)
ngpu.SetStatus(neuron1, {"t_ref": 1000.0, "den_delay": den_delay})

time_diff = 200.0
dt_list, delay_stdp_list = [], []
for i in range(N):
    dt_list.append(dt_step * (-0.5 * (N - 1) + i))
    delay_stdp_list.append(time_diff - dt_list[i])

ngpu.SetStatus(sg, {"spike_times": [50.0]})
delay0 = 1.0; delay1 = delay0 + time_diff
weight_sg = 17.9; weight_stdp = Wmax / 2

ngpu.Connect(sg, neuron0, {"rule": "one_to_one"}, {"weight": weight_sg, "delay": delay0})
ngpu.Connect(sg, neuron1, {"rule": "all_to_all"}, {"weight": weight_sg, "delay": delay1})
ngpu.Connect(neuron0, neuron1, {"rule": "all_to_all"},
             {"weight": weight_stdp, "delay_array": delay_stdp_list, "synapse_group": syn_group})

ngpu.Simulate(1000.0)

sim_w = []
for i in range(N):
    conn_id = ngpu.GetConnections(neuron0, neuron1[i])
    w = ngpu.GetStatus(conn_id, "weight")
    sim_w.append(float(w[0]))

import statistics
print(f"initial weight_stdp = {weight_stdp}")
print(f"N={N} resulting weights: min={min(sim_w):.6g} max={max(sim_w):.6g} "
      f"mean={statistics.mean(sim_w):.6g} spread={max(sim_w)-min(sim_w):.6g}")
changed = sum(1 for w in sim_w if abs(w - weight_stdp) > 1e-9)
print(f"weights changed from initial: {changed}/{N}")
print("STDP WORKS IN THIS BUILD" if changed > 0 else "STDP NOT UPDATING (build/plumbing issue)")
