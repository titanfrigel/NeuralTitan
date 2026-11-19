# Checks (default) or applies (-Fix) the C++/CUDA formatting in .clang-format,
# using the clang-format pinned in tools/requirements-dev.txt.
# Usage: powershell -NoProfile -File scripts/format.ps1 [-Fix]
param(
    [switch]$Fix
)

$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$clangFormat = Join-Path $root 'tools/.venv/Scripts/clang-format.exe'
if (-not (Test-Path $clangFormat)) {
    [Console]::Error.WriteLine("clang-format not found at $clangFormat. Install it from the repository root with:`n" +
        "  python -m venv tools/.venv`n" +
        "  tools/.venv/Scripts/python -m pip install -r tools/requirements-dev.txt")
    exit 1
}

# tests/layer_check/ holds text fixtures for the layer check, not code to format.
$excluded = Join-Path $root 'tests\layer_check\'
$files = foreach ($dir in 'src', 'apps', 'tests') {
    $path = Join-Path $root $dir
    if (Test-Path $path) {
        Get-ChildItem -Path $path -Recurse -File -Include '*.hpp', '*.cpp', '*.cu', '*.cuh' |
            Where-Object { -not $_.FullName.StartsWith($excluded, [StringComparison]::OrdinalIgnoreCase) } |
            ForEach-Object { $_.FullName }
    }
}
if (-not $files) {
    Write-Output 'format: no files'
    exit 0
}

if ($Fix) {
    & $clangFormat -i --style=file $files
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Output "format: formatted $(@($files).Count) files"
} else {
    & $clangFormat --dry-run --Werror --style=file $files
    if ($LASTEXITCODE -ne 0) {
        [Console]::Error.WriteLine('format: files differ from .clang-format; run scripts/format.ps1 -Fix')
        exit $LASTEXITCODE
    }
    Write-Output "format: OK ($(@($files).Count) files)"
}
