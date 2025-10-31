# Simplified Build Script for Gemma C++ with MSVC (FastDebug)
# This is a fallback script that uses Visual Studio's MSVC compiler
# Simpler and more reliable than hybrid oneAPI+CUDA build
# Author: Generated for David's gemma project  
# Date: 2025-10-13

param(
    [switch]$CleanBuild = $false,
    [switch]$ConfigureOnly = $false,
    [int]$Jobs = 0
)

$ErrorActionPreference = "Stop"
Set-Location "C:\codedev\llm\gemma"

Write-Host "=== Gemma MSVC FastDebug Build Script ===" -ForegroundColor Cyan
Write-Host "Started: $(Get-Date)" -ForegroundColor Cyan

# Helper function for parallel jobs calculation
function Get-ParallelJobs {
    param($cpuFrac = 0.75, $gbPerJob = 2.0)
    $cores = [int](Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
    $memGB = [math]::Round(((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB), 2)
    $byCpu = [math]::Floor($cores * $cpuFrac)
    $byMem = [math]::Floor($memGB / $gbPerJob)
    $jobCount = [math]::Max(1, [math]::Min($byCpu, $byMem))
    Write-Host "System: $cores cores, ${memGB}GB RAM -> Using $jobCount parallel jobs" -ForegroundColor Gray
    return $jobCount
}

# Configure environment
Write-Host "`n[1/6] Configuring environment..." -ForegroundColor Yellow
$env:VCPKG_ROOT = "C:\codedev\vcpkg"
$env:VCPKG_FEATURE_FLAGS = "manifests,versions,registries,binarycaching"
$env:VCPKG_DEFAULT_TRIPLET = "x64-windows"
$env:VCPKG_DEFAULT_BINARY_CACHE = "C:\codedev\vcpkg\archives"
$env:CCACHE_DIR = "C:\Users\david\.cache\ccache"
$env:CCACHE_BASEDIR = "C:\codedev\llm\gemma"
$env:CCACHE_MAXSIZE = "20G"
$env:CCACHE_COMPILERCHECK = "content"

if ($Jobs -eq 0) {
    $Jobs = Get-ParallelJobs 0.75 2.0
}

# Update PATH
if ($env:Path -notmatch [regex]::Escape('C:\Users\david\.local\bin')) {
    $env:Path = "C:\Users\david\.local\bin;$env:Path"
}

# Create directories
Write-Host "`n[2/6] Creating build directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path ".\build\msvc-fastdebug\logs" | Out-Null
New-Item -ItemType Directory -Force -Path $env:CCACHE_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $env:VCPKG_DEFAULT_BINARY_CACHE | Out-Null

# Clean build if requested
if ($CleanBuild -and (Test-Path ".\build\msvc-fastdebug")) {
    Write-Host "`n[3/6] Cleaning previous build..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force ".\build\msvc-fastdebug"
    New-Item -ItemType Directory -Force -Path ".\build\msvc-fastdebug\logs" | Out-Null
}

# Install vcpkg dependencies
Write-Host "`n[4/6] Installing vcpkg dependencies (manifest mode)..." -ForegroundColor Yellow
Write-Host "This may take 10-30 minutes on first run..." -ForegroundColor Gray

Push-Location "C:\codedev\llm\gemma"
& "C:\codedev\vcpkg\vcpkg.exe" install --triplet x64-windows 2>&1 | Out-File -Append ".\build\msvc-fastdebug\logs\vcpkg-install.log"
$vcpkgExitCode = $LASTEXITCODE
Pop-Location

if ($vcpkgExitCode -ne 0) {
    Write-Host "vcpkg had some issues (exit code: $vcpkgExitCode)" -ForegroundColor Yellow
    Write-Host "Check log: .\build\msvc-fastdebug\logs\vcpkg-install.log" -ForegroundColor Yellow
    Write-Host "Continuing - FetchContent will handle missing dependencies" -ForegroundColor Yellow
} else {
    Write-Host "vcpkg dependencies installed successfully" -ForegroundColor Green
}

# Configure with CMake
Write-Host "`n[5/6] Configuring with CMake..." -ForegroundColor Yellow
Write-Host "Using Visual Studio 2022 generator" -ForegroundColor Gray
Write-Host "Log: .\build\msvc-fastdebug\logs\configure.log" -ForegroundColor Gray

$configureArgs = @(
    "-S", ".",
    "-B", "build/msvc-fastdebug",
    "-G", "Visual Studio 17 2022",
    "-A", "x64",
    "-T", "v143",
    "-DCMAKE_TOOLCHAIN_FILE=C:/codedev/vcpkg/scripts/buildsystems/vcpkg.cmake",
    "-DGEMMA_PREFER_SYSTEM_DEPS=ON",
    "-DGEMMA_BUILD_BACKENDS=OFF",
    "-DGEMMA_BUILD_ENHANCED_TESTS=OFF",
    "-DGEMMA_ENABLE_SYCL=OFF",
    "-DGEMMA_ENABLE_CUDA=OFF",
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
)

& cmake $configureArgs 2>&1 | Tee-Object -FilePath ".\build\msvc-fastdebug\logs\configure.log"

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: CMake configuration failed!" -ForegroundColor Red
    Write-Host "Check log: .\build\msvc-fastdebug\logs\configure.log" -ForegroundColor Red
    Write-Host "`nTroubleshooting:" -ForegroundColor Yellow
    Write-Host "1. Check if Visual Studio 2022 is properly installed" -ForegroundColor Gray
    Write-Host "2. Verify vcpkg dependencies are available" -ForegroundColor Gray
    Write-Host "3. Review the configuration log for specific errors" -ForegroundColor Gray
    exit 1
}

Write-Host "`nConfiguration successful!" -ForegroundColor Green

if ($ConfigureOnly) {
    Write-Host "`nConfiguration complete (--ConfigureOnly specified)" -ForegroundColor Green
    Write-Host "`nNext step: Run without -ConfigureOnly to build" -ForegroundColor Yellow
    exit 0
}

# Build
Write-Host "`n[6/6] Building with MSBuild ($Jobs parallel jobs)..." -ForegroundColor Yellow
Write-Host "Log: .\build\msvc-fastdebug\logs\build.log" -ForegroundColor Gray
Write-Host "This will take 10-30 minutes depending on your system..." -ForegroundColor Gray

$buildArgs = @(
    "--build", "build/msvc-fastdebug",
    "--config", "Release",
    "--parallel", $Jobs,
    "--", "/nologo", "/verbosity:minimal"
)

& cmake $buildArgs 2>&1 | Tee-Object -FilePath ".\build\msvc-fastdebug\logs\build.log"

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nWARNING: Build had errors! Analyzing..." -ForegroundColor Yellow
    
    # Parse errors
    $errors = Select-String -Path ".\build\msvc-fastdebug\logs\build.log" -Pattern "error " | Select-Object -First 20
    if ($errors) {
        Write-Host "`nFirst 20 errors:" -ForegroundColor Red
        $errors | ForEach-Object { Write-Host $_.Line }
    }
    
    Write-Host "`nFull log: .\build\msvc-fastdebug\logs\build.log" -ForegroundColor Gray
    Write-Host "`nBuild failed - check logs for details" -ForegroundColor Red
    exit 1
} else {
    Write-Host "`nBuild completed successfully!" -ForegroundColor Green
}

# Validate output
Write-Host "`nValidating build artifacts..." -ForegroundColor Yellow
$exeFiles = Get-ChildItem -Recurse ".\build\msvc-fastdebug" -Filter "gemma.exe" -ErrorAction SilentlyContinue

if ($exeFiles) {
    Write-Host "`nFound executables:" -ForegroundColor Green
    $exeFiles | ForEach-Object {
        $size = [math]::Round($_.Length / 1MB, 2)
        Write-Host "  $($_.FullName) (${size} MB)" -ForegroundColor Cyan
    }
    
    # Try to run basic test
    Write-Host "`nTesting executable..." -ForegroundColor Yellow
    $mainExe = $exeFiles | Where-Object { $_.Directory.Name -eq "Release" } | Select-Object -First 1
    if ($mainExe) {
        try {
            $helpOutput = & $mainExe.FullName --help 2>&1 | Select-Object -First 5
            Write-Host "  Executable responds to --help:" -ForegroundColor Green
            $helpOutput | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }
        } catch {
            Write-Host "  WARNING: Could not run executable" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "`nWARNING: No gemma.exe found in build output!" -ForegroundColor Yellow
}

Write-Host "`n=== Build Complete ===" -ForegroundColor Cyan
Write-Host "Finished: $(Get-Date)" -ForegroundColor Cyan
Write-Host "`nBuild artifacts:" -ForegroundColor Yellow
Write-Host "  Executable: .\build\msvc-fastdebug\bin\Release\gemma.exe" -ForegroundColor Gray
Write-Host "  Logs: .\build\msvc-fastdebug\logs\" -ForegroundColor Gray
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "1. Test with: .\build\msvc-fastdebug\bin\Release\gemma.exe --help" -ForegroundColor Gray
Write-Host "2. Run inference with your models" -ForegroundColor Gray
Write-Host "3. Check logs if there were any issues" -ForegroundColor Gray
