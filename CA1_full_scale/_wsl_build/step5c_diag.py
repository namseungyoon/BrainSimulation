"""Diagnose GetConnections/weight-readback in this NEST-GPU build."""
import nestgpu as ngpu
ngpu.SetRandomSeed(1); ngpu.SetTimeResolution(0.1)

pre = ngpu.Create("spike_generator"); ngpu.SetStatus(pre, {"spike_times":[10.0,20.0]})
post = ngpu.Create("aeif_cond_beta_multisynapse", 1)
ngpu.SetStatus(post, {"E_rev":[0.0],"tau_rise":[0.3],"tau_decay":[3.0]})
ngpu.Connect(pre, post, {"rule":"one_to_one"}, {"weight":2.5,"delay":1.0,"receptor":0})

def try_read(tag):
    try:
        conns = ngpu.GetConnections(pre, post)
        print(f"  [{tag}] GetConnections OK, n={len(conns) if hasattr(conns,'__len__') else '?'}")
        try:
            st = ngpu.GetStatus(conns, "weight")
            print(f"  [{tag}] weight =", st)
        except Exception as e:
            print(f"  [{tag}] GetStatus(weight) FAIL: {type(e).__name__}: {e}")
    except Exception as e:
        print(f"  [{tag}] GetConnections FAIL: {type(e).__name__}: {e}")

print("STATIC synapse: Simulate FIRST, then read")
ngpu.Simulate(30.0)
try_read("after Simulate")
# also try the alternative connection-status API if present
for api in ("GetConnectionStatus",):
    fn = getattr(ngpu, api, None)
    print(f"  has {api}: {callable(fn)}")
print("DIAG DONE")
