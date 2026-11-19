# Single entry point: configure, build and run every test.
# Usage: powershell -NoProfile -File scripts/check.ps1 [-Preset debug|release]
param(
    [ValidateSet('debug', 'release')]
    [string]$Preset = 'release'
)

# Run from the repository root so the presets in CMakePresets.json are found.
Set-Location (Join-Path $PSScriptRoot '..')

# Put MSVC 14.38 on PATH; works from a plain PowerShell.
. (Join-Path $PSScriptRoot 'dev-env.ps1')
Enter-TitanDevEnv

# Each step stops the script with its own exit code on failure.
cmake --preset $Preset
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cmake --build --preset $Preset
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

ctest --preset $Preset
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Output 'check: OK'
