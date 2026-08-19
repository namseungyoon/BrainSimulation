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
    $mark = 'X'
    if ($v) { $mark = 'O' }
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
    $c = Get-Command python -ErrorAction SilentlyContinue
    if ($c) { $o = & python -V 2>$null; $o }
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
$ok = ($R.Keys | Where-Object { $R[$_] }).Count
"summary: {0}/{1} present" -f $ok, $R.Count

# No leading underscore: this JSON is 1-1's tracked result, not scratch data.
# (1-1 cannot emit a PNG -- matplotlib needs the Python that 1-1 exists to look for.)
$json = Join-Path $out '1-1_env_probe.json'
($R | ConvertTo-Json -Depth 3) | Out-File -FilePath $json -Encoding ascii
"saved: $json"
