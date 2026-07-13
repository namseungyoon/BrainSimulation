# Source this to set up the CA1 simulator environment:
#   source env.sh
# Sets up CPU NEST (nest_vars.sh), NEST GPU (NESTGPU_LIB), and PYTHONPATH.
_CA1_HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# CPU NEST (correctness oracle) -- installed into the venv by the NEST source build.
if [ -f "$_CA1_HERE/.venv/bin/nest_vars.sh" ]; then
    # shellcheck disable=SC1091
    source "$_CA1_HERE/.venv/bin/nest_vars.sh"
fi

# NEST GPU -- needs NESTGPU_LIB pointing at the compiled kernel library.
_CA1_NESTGPU_LIB="$(find "$_CA1_HERE/.venv" -name 'libnestgpukernel.so' 2>/dev/null | head -1)"
if [ -n "$_CA1_NESTGPU_LIB" ]; then
    export NESTGPU_LIB="$_CA1_NESTGPU_LIB"
fi

export PYTHONPATH="$_CA1_HERE/src:${PYTHONPATH}"

echo "ca1 env ready:"
echo "  PYTHONPATH += $_CA1_HERE/src"
echo "  NESTGPU_LIB = ${NESTGPU_LIB:-<not built>}"
echo "  run: .venv/bin/python -m ca1.cli sim --config configs/smoke_180.yaml --backend nest"
