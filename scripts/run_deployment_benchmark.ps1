# Runs each (model, device) timing pass in its own process with a cool-down in between,
# so heat from earlier passes does not throttle later ones on a laptop.
param(
    [string]$Python = "C:\Users\Legion 5\miniconda3\envs\crowd-gpu\python.exe",
    [int]$CooldownSeconds = 90
)
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$out = "reports/paper_figs/deployment_cost.json"
Remove-Item (Join-Path $root $out) -ErrorAction SilentlyContinue
foreach ($part in "gpu", "cpu") {
    foreach ($model in "SBS R50-IBN", "OSNet", "ConvNeXt-S") {
        Start-Sleep -Seconds $CooldownSeconds
        & $Python (Join-Path $root "scripts/benchmark_backbones.py") --models $model --part $part --out $out 2>&1 |
            Select-String "GPU b1|CPU b1"
    }
}
