# Single entry point: check formatting, configure, build and run every test.
# Usage: powershell -NoProfile -File scripts/check.ps1 [-Preset debug|release]
param(
    [ValidateSet('debug', 'release')]
    [string]$Preset = 'release'
)

# Run from the repository root so the presets in CMakePresets.json are found.
Set-Location (Join-Path $PSScriptRoot '..')

# Formatting first: it is the fastest step to fail.
& (Join-Path $PSScriptRoot 'format.ps1')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Put the MSVC toolset (14.38, or TITAN_MSVC_TOOLSET) on PATH; works from a
# plain PowerShell.
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
