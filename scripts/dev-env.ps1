# Dot-source this file, then call Enter-TitanDevEnv to put the MSVC 14.38
# toolset (required by CUDA 12.6) on PATH for the current PowerShell session.

function Test-TitanCl {
    $cl = Get-Command cl.exe -ErrorAction SilentlyContinue
    if (-not $cl) { return $false }
    $banner = (& cmd /c "cl.exe 2>&1") -join "`n"
    return $banner -match 'Version 19\.38\.'
}

function Enter-TitanDevEnv {
    if (Test-TitanCl) { return }

    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) {
        throw "vswhere.exe not found at $vswhere. Install Visual Studio with the C++ workload."
    }
    $vs = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if (-not $vs) {
        throw 'No Visual Studio installation with the C++ x64 tools was found.'
    }

    $vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'
    $lines = & cmd /c "call `"$vcvars`" -vcvars_ver=14.38 >nul && set"
    foreach ($line in $lines) {
        if ($line -match '^([^=]+)=(.*)$') {
            Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2]
        }
    }

    if (-not (Test-TitanCl)) {
        throw 'MSVC toolset 14.38 (cl 19.38) is required by CUDA 12.6 but was not found. Install it via Visual Studio Installer -> Individual components -> MSVC v143 14.38.'
    }
}
