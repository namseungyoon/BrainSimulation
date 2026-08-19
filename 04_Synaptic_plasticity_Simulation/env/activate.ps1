# 1-3  Activate the 04 environment (ASCII only: PowerShell 5.1 mangles non-BOM UTF-8)
#
# Sets NEURONHOME / PATH / PYTHONPATH for this session and points $Py04 at the
# 04-only interpreter. We do NOT use conda (see docs/DECISIONS.md D7), and we do
# NOT touch machine-wide settings -- Python was installed with PrependPath=0.
#
# DOT-SOURCE it, or the variables die with the child process:
#     . .\env\activate.ps1
#
# One-shot alternative (no dot-sourcing needed): every 04 script also works when
# launched through env\run.ps1, which dot-sources this file first.

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $here

$nrn = "$env:USERPROFILE\nrn"
$venvPy = Join-Path $root '.venv\Scripts\python.exe'

if (-not (Test-Path $nrn))    { Write-Warning "NEURON not found: $nrn  (run 1-3 install)" }
if (-not (Test-Path $venvPy)) { Write-Warning "venv not found: $venvPy  (run 1-2)" }

# NEURONHOME. shared/common/nrn_env.py reads this env var BEFORE its own hardcoded
# default, so setting it here means we never have to edit that shared file (D4).
$env:NEURONHOME = $nrn

# bin      -> nrniv.exe, nrnivmodl.bat, the core DLLs
# mingw    -> the gcc toolchain nrnivmodl shells out to; missing it fails the mod build
$binPaths = @("$nrn\bin", "$nrn\mingw\usr\bin")
foreach ($p in $binPaths) {
    if ($env:PATH -notlike "*$p*") { $env:PATH = "$p;$env:PATH" }
}

# The `neuron` Python package ships inside the NEURON install, not in the venv.
$pyPkg = "$nrn\lib\python"
if ($env:PYTHONPATH) {
    if ($env:PYTHONPATH -notlike "*$pyPkg*") { $env:PYTHONPATH = "$pyPkg;$env:PYTHONPATH" }
} else {
    $env:PYTHONPATH = $pyPkg
}

# The 04 interpreter -- single source of truth. Use $Py04 everywhere.
$global:Py04 = $venvPy
$global:Root04 = $root

Write-Output "[04] NEURONHOME = $env:NEURONHOME"
Write-Output "[04] Py04       = $global:Py04"
Write-Output "[04] PYTHONPATH = $env:PYTHONPATH"
