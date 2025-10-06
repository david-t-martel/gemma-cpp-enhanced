<#
.SYNOPSIS
  Configure & build Gemma with Intel oneAPI SYCL backend using Ninja.
.DESCRIPTION
  This script:
    1. Locates and sources Intel oneAPI setvars.bat (if not already in PATH).
    2. Invokes CMake configure with preset oneapi-ninja-release (override with -Preset).
    3. Builds all targets, emphasizing SYCL backend.
    4. Optionally runs tests filtered to SYCL.* or backend labels.
.PARAMETER Preset
  CMake configurePreset / buildPreset to use (default: oneapi-ninja-release)
.PARAMETER Debug
  Use the debug preset (maps to oneapi-ninja-debug)
.PARAMETER RunTests
  Execute ctest with SyclBackend filters after build.
.PARAMETER Clean
  Remove existing build directory for the chosen preset before configuring.
.EXAMPLE
  ./scripts/build_oneapi_ninja.ps1 -RunTests
.EXAMPLE
  ./scripts/build_oneapi_ninja.ps1 -Debug -Clean -RunTests
.NOTES
  Requires Intel oneAPI Base + DPC++/C++ Compiler component installed.
#>
[CmdletBinding()]
param(
  [string]$Preset = 'oneapi-ninja-release',
  # Renamed from -Debug to -BuildDebug to avoid collision with PowerShell common parameter -Debug
  [switch]$BuildDebug,
  [switch]$RunTests,
  [switch]$Clean,
  # Use legacy PowerShell env-capture path instead of single cmd chain
  [switch]$NoCmdChain
)

if ($BuildDebug) { $Preset = 'oneapi-ninja-debug' }

# ---------------- Option A: cmd chain (default) ----------------
if (-not $NoCmdChain) {
  function Write-Stage($m){ Write-Host "[oneAPI-Build] $m" -ForegroundColor Cyan }
  function Fail($m){ Write-Error $m; exit 1 }

  Write-Stage "(cmd-chain mode) Locating oneAPI setvars.bat"
  $candidateRoots = @(
    ${env:ONEAPI_ROOT},
    ${env:INTEL_ONEAPI_ROOT},
    'C:/Program Files (x86)/Intel/oneAPI'
  ) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique

  $setvars = $null
  foreach ($root in $candidateRoots) {
    $cand = Join-Path $root 'setvars.bat'
    if (Test-Path $cand) { $setvars = $cand; break }
  }
  if (-not $setvars) { Fail "Could not find setvars.bat. Install Intel oneAPI or set ONEAPI_ROOT." }

  if ($Clean) {
    $buildDir = Join-Path $PSScriptRoot "..\build-oneapi\$Preset"
    if (Test-Path $buildDir) { Write-Stage "Removing build dir: $buildDir"; Remove-Item -Recurse -Force $buildDir }
  }

  $cmdSegments = @()
  # Properly quote setvars path for spaces; then force CC/CXX to icx to steer CMake.
  $cmdSegments += "call `"$setvars`""
  $cmdSegments += "set CC=icx"
  $cmdSegments += "set CXX=icx"
  # Ensure vcpkg forwarded configure adds policy minimum for older third-party CMakeLists (sentencepiece)
  $cmdSegments += "set VCPKG_CMAKE_CONFIGURE_OPTIONS=-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
  # Disable vcpkg manifest mode to force pure FetchContent path for all deps (Option 3 fast unblock)
  $cmdSegments += "set VCPKG_FEATURE_FLAGS=-manifests"
  # Force Intel compiler selection explicitly for CMake configure
  $cmdSegments += "cmake -D CMAKE_C_COMPILER=icx -D CMAKE_CXX_COMPILER=icx --preset $Preset"
  # Option 3 temporary override: bypass system/vcpkg sentencepiece so FetchContent + patch kicks in
  $cmdSegments[-1] = $cmdSegments[-1] + " -D GEMMA_PREFER_SYSTEM_DEPS=OFF"
  $cmdSegments += "cmake --build --preset $Preset"
  if ($RunTests) { $cmdSegments += "ctest --preset oneapi-ninja-tests --output-on-failure" }
  $full = $cmdSegments -join ' && '
  Write-Stage "Invoking chained build inside cmd.exe"
  Write-Host "[oneAPI-Build] CMD> $full" -ForegroundColor DarkGray
  & cmd /c $full
  if ($LASTEXITCODE -ne 0) { Fail "Chained build failed (exit $LASTEXITCODE)" }

  # Post-build artifact check
  $artifact = Get-ChildItem (Join-Path $PSScriptRoot "..\build-oneapi\$Preset") -Recurse -Include gemma_sycl_backend.* -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($artifact) { Write-Stage "SYCL backend artifact: $($artifact.FullName)" } else { Write-Warning "SYCL backend artifact not found; inspect configure output for SYCL flags." }
  Write-Stage "Done (cmd-chain mode)."
  exit 0
}

function Write-Stage($m){ Write-Host "[oneAPI-Build] $m" -ForegroundColor Cyan }
function Fail($m){ Write-Error $m; exit 1 }

# 1. (Legacy path) Source oneAPI environment (always attempt if root present and compiler missing)
$haveCompiler = (Get-Command icpx -ErrorAction SilentlyContinue) -or (Get-Command dpcpp -ErrorAction SilentlyContinue)
if (-not $haveCompiler) {
  Write-Stage "Attempting to locate oneAPI setvars.bat (compiler not yet on PATH)"
  $candidateRoots = @(
    ${env:ONEAPI_ROOT},
    ${env:INTEL_ONEAPI_ROOT},
    'C:/Program Files (x86)/Intel/oneAPI'
  ) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique

  $setvars = $null
  foreach ($root in $candidateRoots) {
    $cand = Join-Path $root 'setvars.bat'
    if (Test-Path $cand) { $setvars = $cand; break }
  }
  if (-not $setvars) { Fail "Could not find setvars.bat. Ensure Intel oneAPI is installed (base + HPC)." }
  Write-Stage "Sourcing oneAPI environment: $setvars"
  & cmd /c "call `"$setvars`" >nul 2>&1 && set" | ForEach-Object {
    $kv = ($_ -split '=',2); if ($kv.Length -eq 2) { Set-Item -Path Env:\$($kv[0]) -Value $kv[1] }
  }
}

# Post-source sanity: look for compilers manually under common install tree if still missing
if (-not (Get-Command icpx -ErrorAction SilentlyContinue)) {
  $compilerRoot = Get-ChildItem 'C:/Program Files (x86)/Intel/oneAPI/compiler' -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
  if ($compilerRoot) {
    $binDir = Join-Path $compilerRoot.FullName 'bin'
    if (Test-Path (Join-Path $binDir 'icpx.exe')) { $env:PATH = "$binDir;$env:PATH" }
    elseif (Test-Path (Join-Path $binDir 'dpcpp.exe')) { $env:PATH = "$binDir;$env:PATH" }
  }
}

if (-not (Get-Command icpx -ErrorAction SilentlyContinue) -and -not (Get-Command dpcpp -ErrorAction SilentlyContinue)) {
  Fail "Intel oneAPI compilers (icpx/dpcpp) still not found in PATH after setvars; verify installation."
}

Write-Stage "Active C++ compiler check"
$compilerVersion = & icpx --version 2>$null
if (-not $compilerVersion) { $compilerVersion = & dpcpp --version 2>$null }
Write-Host $compilerVersion

# Hint CMake explicitly in case PATH still causes cl.exe to win detection
if (Get-Command icpx -ErrorAction SilentlyContinue) {
  $env:CXX = 'icpx'
  # icx is the C compiler front-end; fall back to icpx if icx missing
  if (Get-Command icx -ErrorAction SilentlyContinue) { $env:CC = 'icx' } else { $env:CC = 'icpx' }
  Write-Stage "Exported CC=$($env:CC) CXX=$($env:CXX)"
}

# 2. Configure
$buildDir = Join-Path $PSScriptRoot "..\build-oneapi\$Preset"
if ($Clean -and (Test-Path $buildDir)) {
  Write-Stage "Removing existing build directory: $buildDir"
  Remove-Item -Recurse -Force $buildDir
}
Write-Stage "Configuring with preset: $Preset"

# If Intel compilers discovered, build argument list with explicit overrides
$icxPath = Get-ChildItem 'C:/Program Files (x86)/Intel/oneAPI/compiler' -Recurse -Filter icx.exe -ErrorAction SilentlyContinue | Sort-Object FullName -Descending | Select-Object -First 1
$cmakeArgs = @()
if ($icxPath) {
  Write-Stage "Forcing both C and C++ compilers to MSVC-style icx.exe: $($icxPath.FullName)"
  $cmakeArgs += @('-D', "CMAKE_C_COMPILER=$($icxPath.FullName)", '-D', "CMAKE_CXX_COMPILER=$($icxPath.FullName)")
}
$cmakeArgs += @('--preset', $Preset)
# Option 3 temporary override: bypass system/vcpkg sentencepiece so FetchContent + patch kicks in
$cmakeArgs += @('-D','GEMMA_PREFER_SYSTEM_DEPS=OFF')

& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { Fail "Configure failed" }

# 3. Build
Write-Stage "Building (Ninja)"
cmake --build --preset $Preset || Fail "Build failed"

# 4. Post-build summary: check for gemma_sycl_backend
$syclLib = Get-ChildItem -Path $buildDir -Recurse -Include gemma_sycl_backend.* -ErrorAction SilentlyContinue | Select-Object -First 1
if ($syclLib) {
  Write-Stage "SYCL backend artifact found: $($syclLib.FullName)"
} else {
  Write-Warning "SYCL backend library not found – verify detection messages in configure log."
}

if ($RunTests) {
  Write-Stage "Running SYCL-focused tests"
  Push-Location $buildDir
  ctest -R SyclBackend -C Release --output-on-failure || Fail "SYCL backend tests failed"
  Pop-Location
}

Write-Stage "Done. Use -RunTests to execute SYCL tests, -Debug for debug build."
