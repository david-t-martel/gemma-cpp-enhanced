# Build Script for Gemma C++ with oneAPI + CUDA (FastDebug)
# This script configures and builds Gemma with hybrid SYCL (CPU) and CUDA (GPU) support
# Author: Generated for David's gemma project
# Date: 2025-10-13

param(
    [switch]$SkipVcpkgInstall = $false,
    [switch]$CleanBuild = $false,
    [switch]$ConfigureOnly = $false,
    [int]$Jobs = 0
)

$ErrorActionPreference = "Stop"
Set-Location "C:\codedev\llm\gemma"

Write-Host "=== Gemma oneAPI+CUDA FastDebug Build Script ===" -ForegroundColor Cyan
Write-Host "Started: $(Get-Date)" -ForegroundColor Cyan

# Helper function for parallel jobs calculation
function Get-ParallelJobs {
    param($cpuFrac = 0.7, $gbPerJob = 2.5)
    $cores = [int](Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
    $memGB = [math]::Round(((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB), 2)
    $byCpu = [math]::Floor($cores * $cpuFrac)
    $byMem = [math]::Floor($memGB / $gbPerJob)
    $jobCount = [math]::Max(1, [math]::Min($byCpu, $byMem))
    Write-Host "System: $cores cores, ${memGB}GB RAM -> Using $jobCount parallel jobs" -ForegroundColor Gray
    return $jobCount
}

# Configure environment
Write-Host "`n[1/8] Configuring environment..." -ForegroundColor Yellow

# Initialize oneAPI environment first
Write-Host "Initializing Intel oneAPI environment..." -ForegroundColor Gray
$oneapiSetvars = "C:\Program Files (x86)\Intel\oneAPI\setvars.ps1"
if (Test-Path $oneapiSetvars) {
    try {
        & $oneapiSetvars -Force 2>$null | Out-Null
        Write-Host "oneAPI environment initialized" -ForegroundColor Green
    } catch {
        Write-Host "WARNING: Could not initialize oneAPI environment automatically" -ForegroundColor Yellow
        Write-Host "Proceeding anyway - compiler may need manual environment setup" -ForegroundColor Yellow
    }
} else {
    Write-Host "WARNING: oneAPI setvars.ps1 not found" -ForegroundColor Yellow
}

$env:VCPKG_ROOT = "C:\codedev\vcpkg"
$env:VCPKG_FEATURE_FLAGS = "manifests,versions,registries,binarycaching"
$env:VCPKG_DEFAULT_TRIPLET = "x64-windows"
$env:VCPKG_DEFAULT_BINARY_CACHE = "C:\codedev\vcpkg\archives"
$env:CCACHE_DIR = "C:\Users\david\.cache\ccache"
$env:CCACHE_BASEDIR = "C:\codedev\llm\gemma"
$env:CCACHE_MAXSIZE = "20G"
$env:CCACHE_COMPILERCHECK = "content"
$env:CCACHE_LOGFILE = "C:\codedev\llm\gemma\build\oneapi-cuda-fastdebug\logs\ccache.log"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
$env:ONEAPI_ROOT = "C:\Program Files (x86)\Intel\oneAPI"

if ($Jobs -eq 0) {
    $Jobs = Get-ParallelJobs 0.7 2.5
}
$env:CMAKE_BUILD_PARALLEL_LEVEL = "$Jobs"
$env:NINJA_STATUS = "[%f/%t %o] %es "

# Update PATH
if ($env:Path -notmatch [regex]::Escape('C:\Users\david\.local\bin')) {
    $env:Path = "C:\Users\david\.local\bin;$env:Path"
}
$env:Path = "C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin;$env:Path"
$env:Path = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin;$env:Path"

# Verify compilers are accessible
Write-Host "Verifying compilers..." -ForegroundColor Gray
try {
    $icxVersion = & "C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin\icx.exe" --version 2>&1 | Select-Object -First 1
    Write-Host "  ICX: $icxVersion" -ForegroundColor Green
} catch {
    Write-Host "  WARNING: ICX compiler check failed" -ForegroundColor Yellow
}

try {
    $nvccVersion = & "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\nvcc.exe" --version 2>&1 | Select-Object -Last 1
    Write-Host "  NVCC: $nvccVersion" -ForegroundColor Green
} catch {
    Write-Host "  WARNING: NVCC compiler check failed" -ForegroundColor Yellow
}

# Create directories
Write-Host "`n[2/8] Creating build directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path ".\build\oneapi-cuda-fastdebug\logs" | Out-Null
New-Item -ItemType Directory -Force -Path ".\build\oneapi-cuda-fastdebug\reports" | Out-Null
New-Item -ItemType Directory -Force -Path $env:CCACHE_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $env:VCPKG_DEFAULT_BINARY_CACHE | Out-Null

# Install vcpkg dependencies
if (-not $SkipVcpkgInstall) {
    Write-Host "`n[3/8] Installing vcpkg dependencies via manifest..." -ForegroundColor Yellow
    Write-Host "This may take 10-30 minutes on first run..." -ForegroundColor Gray
    Write-Host "vcpkg will read from vcpkg.json manifest..." -ForegroundColor Gray
    
    # In manifest mode, just run vcpkg install without package names
    # vcpkg will automatically read from vcpkg.json
    Push-Location "C:\codedev\llm\gemma"
    & "C:\codedev\vcpkg\vcpkg.exe" install --triplet x64-windows
    $vcpkgExitCode = $LASTEXITCODE
    Pop-Location
    
    if ($vcpkgExitCode -ne 0) {
        Write-Host "WARNING: vcpkg install encountered issues" -ForegroundColor Yellow
        Write-Host "This is expected if sentencepiece is not available for Windows" -ForegroundColor Yellow
        Write-Host "Build will use FetchContent for missing dependencies" -ForegroundColor Yellow
    } else {
        Write-Host "vcpkg dependencies installed successfully" -ForegroundColor Green
    }
} else {
    Write-Host "`n[3/8] Skipping vcpkg install (--SkipVcpkgInstall specified)" -ForegroundColor Gray
}

# Clean build if requested
if ($CleanBuild -and (Test-Path ".\build\oneapi-cuda-fastdebug")) {
    Write-Host "`n[4/8] Cleaning previous build..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force ".\build\oneapi-cuda-fastdebug"
    New-Item -ItemType Directory -Force -Path ".\build\oneapi-cuda-fastdebug\logs" | Out-Null
    New-Item -ItemType Directory -Force -Path ".\build\oneapi-cuda-fastdebug\reports" | Out-Null
}

# Configure with CMake
Write-Host "`n[5/8] Configuring with CMake (preset: oneapi-cuda-fastdebug)..." -ForegroundColor Yellow
Write-Host "Log: .\build\oneapi-cuda-fastdebug\logs\configure.log" -ForegroundColor Gray

# Note: The preset needs to be added to CMakePresets.json first
# For now, we'll use manual configuration
$configCmd = @"
cmake -S . -B build/oneapi-cuda-fastdebug ``
  -G Ninja ``
  -DCMAKE_TOOLCHAIN_FILE="C:/codedev/vcpkg/scripts/buildsystems/vcpkg.cmake" ``
  -DCMAKE_BUILD_TYPE=FastDebug ``
  -DCMAKE_C_COMPILER="C:/Program Files (x86)/Intel/oneAPI/compiler/latest/bin/icx.exe" ``
  -DCMAKE_CXX_COMPILER="C:/Program Files (x86)/Intel/oneAPI/compiler/latest/bin/icx.exe" ``
  -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/nvcc.exe" ``
  -DCMAKE_CUDA_STANDARD=17 ``
  -DCMAKE_CUDA_ARCHITECTURES=89 ``
  -DGEMMA_ENABLE_SYCL=ON ``
  -DGEMMA_ENABLE_CUDA=ON ``
  -DGEMMA_FORCE_INTEL_COMPILER_FOR_SYCL=ON ``
  -DCMAKE_C_COMPILER_LAUNCHER="C:/Users/david/.local/bin/ccache.exe" ``
  -DCMAKE_CXX_COMPILER_LAUNCHER="C:/Users/david/.local/bin/ccache.exe" ``
  -DCMAKE_CUDA_COMPILER_LAUNCHER="C:/Users/david/.local/bin/ccache.exe" ``
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ``
  -DGEMMA_PREFER_SYSTEM_DEPS=ON ``
  -DGEMMA_BUILD_BACKENDS=OFF ``
  -DGEMMA_BUILD_ENHANCED_TESTS=OFF
"@

Invoke-Expression $configCmd 2>&1 | Tee-Object -FilePath ".\build\oneapi-cuda-fastdebug\logs\configure.log"

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: CMake configuration failed!" -ForegroundColor Red
    Write-Host "Check log: .\build\oneapi-cuda-fastdebug\logs\configure.log" -ForegroundColor Red
    exit 1
}

if ($ConfigureOnly) {
    Write-Host "`nConfiguration complete (--ConfigureOnly specified)" -ForegroundColor Green
    exit 0
}

# Build
Write-Host "`n[6/8] Building with Ninja ($Jobs parallel jobs)..." -ForegroundColor Yellow
Write-Host "Log: .\build\oneapi-cuda-fastdebug\logs\build.log" -ForegroundColor Gray
Write-Host "This will take 10-30 minutes depending on your system..." -ForegroundColor Gray

cmake --build build/oneapi-cuda-fastdebug -- -j $Jobs -k 0 2>&1 | Tee-Object -FilePath ".\build\oneapi-cuda-fastdebug\logs\build.log"

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nWARNING: Build had errors! Analyzing..." -ForegroundColor Yellow
    
    # Parse errors
    $errors = Select-String -Path ".\build\oneapi-cuda-fastdebug\logs\build.log" -Pattern "error:" | Select-Object -First 20
    Write-Host "`nFirst 20 errors:" -ForegroundColor Red
    $errors | ForEach-Object { Write-Host $_.Line }
    
    Write-Host "`nFull log: .\build\oneapi-cuda-fastdebug\logs\build.log" -ForegroundColor Gray
} else {
    Write-Host "`nBuild completed successfully!" -ForegroundColor Green
}

# Check ccache stats
Write-Host "`n[7/8] ccache statistics:" -ForegroundColor Yellow
& "C:\Users\david\.local\bin\ccache.exe" -s

# Validate output
Write-Host "`n[8/8] Validating build artifacts..." -ForegroundColor Yellow
$exeFiles = Get-ChildItem -Recurse ".\build\oneapi-cuda-fastdebug" -Filter "gemma.exe" -ErrorAction SilentlyContinue

if ($exeFiles) {
    Write-Host "`nFound executables:" -ForegroundColor Green
    $exeFiles | ForEach-Object {
        $size = [math]::Round($_.Length / 1MB, 2)
        Write-Host "  $($_.FullName) (${size} MB)" -ForegroundColor Cyan
    }
} else {
    Write-Host "`nWARNING: No gemma.exe found in build output!" -ForegroundColor Yellow
}

Write-Host "`n=== Build Complete ===" -ForegroundColor Cyan
Write-Host "Finished: $(Get-Date)" -ForegroundColor Cyan
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "1. Check logs in .\build\oneapi-cuda-fastdebug\logs\" -ForegroundColor Gray
Write-Host "2. Run tests or inference with the built executable" -ForegroundColor Gray
Write-Host "3. For SYCL CPU-only: Set-Item Env:\SYCL_DEVICE_FILTER 'cpu'" -ForegroundColor Gray
Write-Host "4. For CUDA GPU: Remove-Item Env:\SYCL_DEVICE_FILTER -ErrorAction SilentlyContinue" -ForegroundColor Gray
