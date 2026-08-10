"""WSL environment probe for CA1 single-cell electrophysiology reproduction.
Read-only: imports modules and reports availability. Runs nothing heavy.
Invoke in WSL with the repo venv python."""
import sys, shutil, os
print("python     :", sys.version.split()[0], "|", sys.executable)
print("cwd        :", os.getcwd())
for mod in ("numpy", "scipy", "h5py", "matplotlib"):
    try:
        m = __import__(mod)
        print(f"{mod:<11}:", getattr(m, "__version__", "?"))
    except Exception as e:  # noqa: BLE001
        print(f"{mod:<11}: MISSING ({type(e).__name__})")
try:
    import nest  # type: ignore
    print("nest (CPU) :", getattr(nest, "__version__", "?"))
except Exception as e:  # noqa: BLE001
    print("nest (CPU) : MISSING", type(e).__name__, str(e)[:140])
try:
    import nestgpu  # type: ignore
    print("nestgpu    : import OK", getattr(nestgpu, "__file__", ""))
except Exception as e:  # noqa: BLE001
    print("nestgpu    : MISSING", type(e).__name__, str(e)[:140])
print("nvidia-smi :", shutil.which("nvidia-smi"))
try:
    import ca1  # type: ignore
    print("ca1 pkg    :", ca1.__file__)
except Exception as e:  # noqa: BLE001
    print("ca1 pkg    : MISSING", type(e).__name__, str(e)[:140])
