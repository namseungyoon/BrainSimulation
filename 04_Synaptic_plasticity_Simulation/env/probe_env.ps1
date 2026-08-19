# 1-1  Environment probe (ASCII only: PowerShell 5.1 mangles non-BOM UTF-8)
# Purpose: report what THIS machine actually has, before any Python exists.
# Run:  powershell -ExecutionPolicy Bypass -File env\probe_env.ps1
# Out:  01_env/1_probe/figures/_1-1_probe.json  + console table

$ErrorActionPreference = 'Continue'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $here                     # 04_Synaptic_plasticity_Simulation
$brain = Split-Path -Parent $root                    # 02_BrainSimulator
$out  = Join-Path $root '01_env\1_probe\figures'
if (-not (Test-Path $out)) { New-Item -ItemType Directory -Force $out | Out-Null }

$R = [ordered]@{}

function Probe($name, $block) {
    # A block that emits nothing returns AutomationNull, which ConvertTo-Json
    # renders as "{}" rather than null -- that would break `is None` checks on
    # the Python side later. Collapse to a real $null via an array wrapper.
    $v = $null
    try {
        $vals = @(& $block)
        if ($vals.Count -gt 0) { $v = $vals[0] }
    } catch { $v = $null }
    $script:R[$name] = $v
    # A WindowsApps stub exists as a file but is NOT a usable interpreter.
    # Counting it as present is how you waste an hour in 1-2. Mark it '!' , not 'O'.
    $mark = 'X'
    if ($v) { $mark = 'O' }
    if ("$v".StartsWith('STUB ')) { $mark = '!' }
    $show = $v
    if ($null -eq $show) { $show = '(none)' }
    "{0,-22} [{1}]  {2}" -f $name, $mark, $show
}

"=== 1-1 environment probe ==="
"machine user : $env:USERNAME"
"repo root    : $brain"
""

Probe 'python_on_path' {
    $c = Get-Command python -ErrorAction SilentlyContinue
    if ($c) {
        # WindowsApps stub is a 0-byte reparse point, not a real interpreter
        $len = (Get-Item $c.Source).Length
        if ($len -lt 1024) { "STUB $($c.Source)" } else { $c.Source }
    }
}
Probe 'python_version' {
    # Do NOT trust `python` on PATH: we install with PrependPath=0 on purpose, so
    # PATH keeps pointing at the WindowsApps stub even after 1-2 succeeds.
    # Check the known install target and the 04 venv explicitly.
    $cands = @(
        (Join-Path $root '.venv\Scripts\python.exe'),
        "$env:USERPROFILE\Python311\python.exe"
    )
    $hit = $cands | Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($hit) { (& $hit -V 2>$null) }
    else {
        $c = Get-Command python -ErrorAction SilentlyContinue
        if ($c -and (Get-Item $c.Source).Length -ge 1024) { (& python -V 2>$null) }
    }
}
Probe 'venv_04' {
    $p = Join-Path $root '.venv\Scripts\python.exe'
    if (Test-Path $p) { $p }
}
Probe 'conda' {
    $c = Get-Command conda -ErrorAction SilentlyContinue
    if ($c) { $c.Source }
}
Probe 'conda_dirs' {
    $p = @("$env:USERPROFILE\miniconda3","$env:USERPROFILE\anaconda3",
           "$env:LOCALAPPDATA\miniconda3","C:\ProgramData\miniconda3",
           "C:\ProgramData\anaconda3","$env:USERPROFILE\.conda\envs") |
         Where-Object { Test-Path $_ }
    if ($p) { $p -join ';' }
}
Probe 'NEURONHOME_env' { $env:NEURONHOME }
Probe 'neuron_install' {
    $p = @("$env:USERPROFILE\nrn","C:\nrn","C:\Program Files\NEURON") |
         Where-Object { Test-Path $_ }
    if ($p) { $p -join ';' }
}
Probe 'nrnivmodl' {
    $p = "$env:USERPROFILE\nrn\bin\nrnivmodl.bat"
    if (Test-Path $p) { $p }
}
Probe 'dll_local_04' {
    $p = Join-Path $root 'mechanisms\nrnmech.dll'
    if (Test-Path $p) { $p }
}
Probe 'dll_shared' {
    $p = Join-Path $brain 'shared\mechanisms\nrnmech.dll'
    if (Test-Path $p) { $p }
}
Probe 'mod_sources' {
    $d = Join-Path $brain 'shared\mechanisms'
    if (Test-Path $d) { "{0} .mod files" -f (Get-ChildItem $d -Filter *.mod).Count }
}
Probe 'pyr_bundles' {
    $d = Join-Path $brain 'Models'
    if (Test-Path $d) {
        $n = (Get-ChildItem $d -Directory | Where-Object { $_.Name -like 'CA1_pyr_*model_files' }).Count
        "{0} pyramidal bundles" -f $n
    }
}
Probe 'git' {
    $c = Get-Command git -ErrorAction SilentlyContinue
    if ($c) { (& git --version) }
}
Probe 'write_access' {
    $t = Join-Path $out '_wtest.tmp'
    Set-Content -Path $t -Value 'x' -Encoding ascii
    if (Test-Path $t) { Remove-Item $t -Force; 'writable' }
}
Probe 'free_space_GB' {
    $d = (Get-PSDrive -Name ($root.Substring(0,1)))
    [math]::Round($d.Free / 1GB, 1)
}

""
# NOTE: $R.Values on an OrderedDictionary pipes as ONE object, so a naive
#       ($R.Values | Where-Object { $_ }).Count always returns 1. Iterate keys.
# A STUB entry is truthy but unusable, so it must NOT be counted as present --
# otherwise this summary disagrees with 1-1_plot_env_probe.py (6/14 vs 5/14).
$ok = ($R.Keys | Where-Object { $R[$_] -and -not ("$($R[$_])".StartsWith('STUB ')) }).Count
"summary: {0}/{1} present" -f $ok, $R.Count

# Two outputs, deliberately separate:
#   1-1_env_probe.json      FROZEN baseline = the state 1-1 actually found, before any
#                           install. Tracked. Written only once; never overwritten, or
#                           1-1's figure would silently start describing a later machine.
#   _env_probe_latest.json  CURRENT state. Gitignored. Re-run this any time.
$body = ($R | ConvertTo-Json -Depth 3)

$latest = Join-Path $out '_env_probe_latest.json'
$body | Out-File -FilePath $latest -Encoding ascii
"saved: $latest  (current state)"

$json = Join-Path $out '1-1_env_probe.json'
if (Test-Path $json) {
    "kept : $json  (frozen 1-1 baseline, not overwritten)"
} else {
    $body | Out-File -FilePath $json -Encoding ascii
    "saved: $json  (frozen 1-1 baseline)"
}
