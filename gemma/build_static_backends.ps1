# Build Gemma.cpp with Static Linking and Hardware Backends
# This script builds standalone executables with Intel oneAPI compiler

param(
    [switch]$Clean,
    [switch]$Configure,
    [switch]$Build,
    [switch]$All,
    [string]$BuildType = "Release"
)

$ErrorActionPreference = "Stop"

# Paths
$ProjectRoot = $PSScriptRoot
$BuildDir = Join-Path $ProjectRoot "build"
$OneAPIInit = "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

# Check if oneAPI is available
if (-not (Test-Path $OneAPIInit)) {
    Write-Error "Intel oneAPI not found at: $OneAPIInit"
    exit 1
}

# Helper function to run commands with oneAPI environment
function Invoke-OneAPICommand {
    param([string]$Command)
    
    $TempScript = [System.IO.Path]::GetTempFileName() + ".bat"
    @"
@echo off
call "$OneAPIInit" > nul 2>&1
$Command
"@ | Out-File -FilePath $TempScript -Encoding ASCII
    
    $Result = cmd /c $TempScript 2>&1
    Remove-Item $TempScript -Force
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Command failed: $Command`n$Result"
    }
    return $Result
}

# Clean build directory
if ($Clean -or $All) {
    Write-Host "Cleaning build directory..." -ForegroundColor Cyan
    if (Test-Path $BuildDir) {
        Remove-Item -Path "$BuildDir\CMakeCache.txt","$BuildDir\CMakeFiles" -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# Configure CMake
if ($Configure -or $All) {
    Write-Host "Configuring with Intel oneAPI and static linking..." -ForegroundColor Cyan
    
    $CMakeArgs = @(
        "-S", $ProjectRoot,
        "-B", $BuildDir,
        "-G", "Ninja",
        "-DCMAKE_C_COMPILER=icx",
        "-DCMAKE_CXX_COMPILER=icx",
        "-DCMAKE_BUILD_TYPE=$BuildType",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=sccache",
        "-DGEMMA_PREFER_SYSTEM_DEPS=OFF",
        "-DGEMMA_BUILD_ENHANCED_TESTS=OFF",
        "-DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON",
        "-DGEMMA_BUILD_BACKENDS=ON",
        "-DGEMMA_ENABLE_SYCL=ON",
        "-DGEMMA_AUTO_DETECT_BACKENDS=OFF"
    )
    
    $CMakeCommand = "cmake $($CMakeArgs -join ' ')"
    Write-Host "Running: $CMakeCommand" -ForegroundColor Yellow
    Invoke-OneAPICommand $CMakeCommand
}

# Build
if ($Build -or $All) {
    Write-Host "Building Gemma executables..." -ForegroundColor Cyan
    
    # Build main executable
    Write-Host "`n=== Building gemma.exe ===" -ForegroundColor Green
    $BuildCommand = "cmake --build $BuildDir --config $BuildType --target gemma -j"
    Invoke-OneAPICommand $BuildCommand
    
    # Check if SYCL backend was enabled
    $BuildLog = Get-Content "$BuildDir\CMakeCache.txt" -ErrorAction SilentlyContinue
    if ($BuildLog -match "GEMMA_ENABLE_SYCL:BOOL=ON") {
        Write-Host "`n=== Building gemma-sycl.exe (if available) ===" -ForegroundColor Green
        try {
            $SyclCommand = "cmake --build $BuildDir --config $BuildType --target gemma-sycl -j"
            Invoke-OneAPICommand $SyclCommand
        } catch {
            Write-Warning "SYCL backend build skipped or failed (may not be configured yet)"
        }
    }
    
    Write-Host "`nBuild complete!" -ForegroundColor Green
    Write-Host "Executables location: $BuildDir\bin\" -ForegroundColor Cyan
}

# Default action if no parameters
if (-not ($Clean -or $Configure -or $Build -or $All)) {
    Write-Host @"
Usage: build_static_backends.ps1 [-Clean] [-Configure] [-Build] [-All] [-BuildType <type>]

Options:
  -Clean       Clean build directory
  -Configure   Run CMake configuration
  -Build       Build executables
  -All         Clean, configure, and build
  -BuildType   Build type (default: Release)

Examples:
  .\build_static_backends.ps1 -All                    # Full rebuild
  .\build_static_backends.ps1 -Configure -Build       # Configure and build
  .\build_static_backends.ps1 -Build                  # Build only
  .\build_static_backends.ps1 -All -BuildType Debug   # Debug build

"@ -ForegroundColor Yellow
}
