# Dot-source this file, then call Enter-TitanDevEnv to put the MSVC toolset
# CUDA 12.6 needs on PATH for the current PowerShell session. The toolset is
# 14.38 by default; set TITAN_MSVC_TOOLSET (for example 14.44 in CI) to select
# another one that CUDA 12.6 supports.

function Get-TitanMsvcToolset {
    if ($env:TITAN_MSVC_TOOLSET) { return $env:TITAN_MSVC_TOOLSET }
    return '14.38'
}

# Toolset 14.xx ships cl.exe 19.xx.
function Test-TitanCl([string]$Toolset) {
    $cl = Get-Command cl.exe -ErrorAction SilentlyContinue
    if (-not $cl) { return $false }
    $minor = $Toolset.Split('.')[1]
    $banner = (& cmd /c "cl.exe 2>&1") -join "`n"
    return $banner -match "Version 19\.$minor\."
}

function Enter-TitanDevEnv {
    $toolset = Get-TitanMsvcToolset
    if (Test-TitanCl $toolset) { return }

    $installer = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer'
    $vswhere = Join-Path $installer 'vswhere.exe'
    if (-not (Test-Path $vswhere)) {
        throw "vswhere.exe not found at $vswhere. Install Visual Studio with the C++ workload."
    }
    $vs = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if (-not $vs) {
        throw 'No Visual Studio installation with the C++ x64 tools was found.'
    }

    # vcvars calls vswhere.exe by name, so the Installer directory must be on PATH.
    if (($env:PATH -split ';') -notcontains $installer) {
        $env:PATH = "$installer;$env:PATH"
    }

    $vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'
    $lines = & cmd /c "call `"$vcvars`" -vcvars_ver=$toolset >nul && set"
    foreach ($line in $lines) {
        if ($line -match '^([^=]+)=(.*)$') {
            Set-Item -LiteralPath "Env:$($Matches[1])" -Value $Matches[2]
        }
    }

    if (-not (Test-TitanCl $toolset)) {
        $minor = $toolset.Split('.')[1]
        throw "MSVC toolset $toolset (cl 19.$minor) is required by CUDA 12.6 but was not found. Install it via Visual Studio Installer -> Individual components -> MSVC v143 $toolset, or set TITAN_MSVC_TOOLSET to an installed toolset that CUDA 12.6 supports."
    }
}
