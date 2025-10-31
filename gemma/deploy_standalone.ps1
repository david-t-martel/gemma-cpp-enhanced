# Deploy Gemma.exe with Intel oneAPI Runtime DLLs
# This script copies necessary runtime DLLs to make gemma.exe standalone

param(
    [string]$BuildDir = "build\bin",
    [string]$DeployDir = "deploy",
    [string]$ExecutableName = "gemma*.exe",
    [switch]$IncludeOneAPILibs = $false,
    [switch]$Verbose = $false
)

$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
$SourceDir = Join-Path $ProjectRoot $BuildDir
$TargetDir = Join-Path $ProjectRoot $DeployDir

# Intel oneAPI paths
$OneAPIRoot = if ($env:ONEAPI_ROOT) { $env:ONEAPI_ROOT } else { "C:\Program Files (x86)\Intel\oneAPI" }

if ($Verbose) {
    Write-Host "Using oneAPI root: $OneAPIRoot" -ForegroundColor Gray
}

$CompilerBin = Join-Path $OneAPIRoot "compiler\latest\bin"
$CompilerLib = Join-Path $OneAPIRoot "compiler\latest\lib"
$MKLBin = Join-Path $OneAPIRoot "mkl\latest\bin\intel64"
$TBBBin = Join-Path $OneAPIRoot "tbb\latest\bin\intel64\vc14"
$TBBBinAlt = Join-Path $OneAPIRoot "tbb\latest\redist\intel64\vc14"
$IPPBin = Join-Path $OneAPIRoot "ipp\latest\bin\intel64"
$DNNLBin = Join-Path $OneAPIRoot "dnnl\latest\cpu_dpcpp_gpu_dpcpp\bin"

Write-Host "Deploying Gemma.exe for standalone execution..." -ForegroundColor Cyan

# Create deployment directory
if (Test-Path $TargetDir) {
    Remove-Item -Path $TargetDir -Recurse -Force
}
New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null

# Find and copy gemma executable(s)
$GemmaExes = Get-ChildItem -Path $SourceDir -Filter $ExecutableName -ErrorAction SilentlyContinue

if ($GemmaExes.Count -eq 0) {
    Write-Error "No executable matching '$ExecutableName' found in $SourceDir. Build the project first."
    exit 1
}

foreach ($exe in $GemmaExes) {
    Copy-Item $exe.FullName $TargetDir
    Write-Host "✓ Copied $($exe.Name)" -ForegroundColor Green
    
    # Detect oneAPI libs from executable name
    if ($exe.Name -match '\+(.+?)\.exe') {
        $DetectedLibs = $matches[1] -split '-'
        Write-Host "  Detected oneAPI libs: $($DetectedLibs -join ', ')" -ForegroundColor Cyan
        $IncludeOneAPILibs = $true
    }
}

# Core Intel runtime DLLs (always needed for SYCL backend)
$CoreDLLs = @(
    "libiomp5md.dll",          # OpenMP runtime
    "svml_dispmd.dll",         # Math library
    "libmmd.dll",              # Math library
    "sycl7.dll",               # SYCL runtime (oneAPI 2025)
    "pi_level_zero.dll",       # Level Zero plugin
    "pi_opencl.dll"            # OpenCL plugin
)

# oneAPI Library DLLs (when GEMMA_USE_ONEAPI_LIBS enabled)
$OneAPILibDLLs = @{
    TBB = @(
        "tbb12.dll",
        "tbbmalloc.dll"
    )
    IPP = @(
        "ippi-9.1.dll",
        "ippcore-9.1.dll",
        "ipps-9.1.dll",
        "ippvm-9.1.dll"
    )
    DNNL = @(
        "dnnl.dll"
    )
    # DPL is header-only, no DLLs needed
}

# Optional MKL DLLs (if using oneMKL)
$OptionalMKLDLLs = @(
    "mkl_core.2.dll",
    "mkl_intel_thread.2.dll",
    "mkl_sycl_blas.4.dll"
)

# Helper function to copy DLL from multiple search paths
function Copy-DLLIfFound {
    param(
        [string]$DllName,
        [string[]]$SearchPaths,
        [string]$Category = "Runtime"
    )
    
    foreach ($path in $SearchPaths) {
        $dllPath = Join-Path $path $DllName
        if (Test-Path $dllPath) {
            Copy-Item $dllPath $TargetDir -Force
            Write-Host "✓ Copied $DllName ($Category)" -ForegroundColor Green
            return $true
        }
    }
    
    if ($Verbose) {
        Write-Verbose "Not found: $DllName in $($SearchPaths.Count) search paths"
    }
    return $false
}

$CopiedCount = 0
$MissingDLLs = @()

# Copy core runtime DLLs
Write-Host "`nCopying core runtime DLLs..." -ForegroundColor Cyan
foreach ($dll in $CoreDLLs) {
    $found = Copy-DLLIfFound -DllName $dll -SearchPaths @($CompilerBin) -Category "Core Runtime"
    if ($found) {
        $CopiedCount++
    } else {
        $MissingDLLs += $dll
        Write-Warning "Missing: $dll"
    }
}

# Copy oneAPI library DLLs if requested or detected
if ($IncludeOneAPILibs) {
    Write-Host "`nCopying oneAPI library DLLs..." -ForegroundColor Cyan
    
    # TBB
    Write-Host "  Threading Building Blocks (TBB):" -ForegroundColor Yellow
    foreach ($dll in $OneAPILibDLLs.TBB) {
        $found = Copy-DLLIfFound -DllName $dll -SearchPaths @($TBBBin, $TBBBinAlt, $CompilerBin) -Category "TBB"
        if ($found) { $CopiedCount++ }
    }
    
    # IPP
    Write-Host "  Integrated Performance Primitives (IPP):" -ForegroundColor Yellow
    foreach ($dll in $OneAPILibDLLs.IPP) {
        $found = Copy-DLLIfFound -DllName $dll -SearchPaths @($IPPBin) -Category "IPP"
        if ($found) { $CopiedCount++ }
    }
    
    # DNNL
    Write-Host "  Deep Neural Network Library (DNNL):" -ForegroundColor Yellow
    foreach ($dll in $OneAPILibDLLs.DNNL) {
        $found = Copy-DLLIfFound -DllName $dll -SearchPaths @($DNNLBin) -Category "DNNL"
        if ($found) { $CopiedCount++ }
    }
    
    Write-Host "  DPL is header-only (no DLLs required)" -ForegroundColor Gray
}

# Copy optional MKL DLLs
Write-Host "`nCopying optional oneMKL DLLs..." -ForegroundColor Cyan
if (Test-Path $MKLBin) {
    foreach ($dll in $OptionalMKLDLLs) {
        $found = Copy-DLLIfFound -DllName $dll -SearchPaths @($MKLBin) -Category "oneMKL"
        if ($found) { $CopiedCount++ }
    }
} else {
    Write-Host "  oneMKL not found (skipping)" -ForegroundColor Gray
}

# Create README
$ReadmeContent = @"
# Gemma.cpp Standalone Deployment

This directory contains gemma executable(s) and all required Intel oneAPI runtime DLLs
for standalone execution without needing to load the oneAPI environment.

## Contents

$($GemmaExes | ForEach-Object { "- $($_.Name): Gemma executable`n" } | Out-String)
- *.dll: Intel oneAPI runtime libraries

## Usage

Simply run the desired executable from this directory:

``````
.\gemma_std.exe --help
.\gemma_hw-sycl.exe --weights model.sbs --tokenizer tokenizer.spm
``````

Or add this directory to your PATH for system-wide access.

## Executable Naming Convention

- gemma_std.exe: Standard CPU build
- gemma_hw-<backend>.exe: Hardware accelerated build
- gemma_std+<libs>.exe: CPU build with oneAPI performance libraries

Examples:
- gemma_std+tbb-ipp.exe: TBB threading + IPP vector operations
- gemma_hw-sycl.exe: SYCL GPU acceleration
- gemma_hw-sycl+dnnl.exe: SYCL + DNNL matrix operations

## Runtime Libraries

### Core Runtime (SYCL Backend)
$($CoreDLLs | ForEach-Object { "- $_`n" } | Out-String)

### oneAPI Performance Libraries
$(if ($IncludeOneAPILibs) {
@"
- TBB (Threading): $($OneAPILibDLLs.TBB -join ', ')
- IPP (Vector Ops): $($OneAPILibDLLs.IPP -join ', ')
- DNNL (Matrix Ops): $($OneAPILibDLLs.DNNL -join ', ')
- DPL (Algorithms): Header-only, no DLLs
"@
} else {
"Not included in this deployment"
})

## System Requirements

- Windows 10/11 x64
- Intel CPU with AVX2 support recommended
- Intel GPU (Arc, Iris Xe, UHD) for SYCL acceleration (optional)
- 8GB RAM minimum, 16GB+ recommended for larger models

## Notes

- If you encounter "DLL not found" errors, ensure Intel GPU drivers are installed
- For SYCL GPU acceleration, install Intel Arc/Iris graphics drivers from:
  https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html
- The executable will fall back to CPU execution if GPU is not available
- oneAPI library optimizations improve CPU performance without requiring GPU

## Performance Tips

- Use TBB-enabled builds for better multi-core scaling
- IPP accelerates vector operations (dot products, activations)
- DNNL provides optimized matrix multiplication on CPU
- SYCL enables GPU acceleration for maximum performance

Built with Intel oneAPI $((Get-Date).ToString('yyyy-MM-dd'))
Deployment script: deploy_standalone.ps1
"@

$ReadmeContent | Out-File -FilePath (Join-Path $TargetDir "README.txt") -Encoding UTF8

Write-Host "`nDeployment complete!" -ForegroundColor Green
Write-Host "Location: $TargetDir" -ForegroundColor Cyan
Write-Host "Copied $CopiedCount files" -ForegroundColor Cyan

if ($MissingDLLs.Count -gt 0) {
    Write-Warning "`nMissing DLLs (may cause runtime errors):"
    $MissingDLLs | ForEach-Object { Write-Warning "  - $_" }
    Write-Host "`nInstall Intel oneAPI Base Toolkit to ensure all DLLs are available." -ForegroundColor Yellow
}

# Test the deployed executable
Write-Host "`nTesting deployed executable..." -ForegroundColor Cyan
Push-Location $TargetDir
try {
    $testResult = & ".\gemma.exe" --help 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Deployed gemma.exe runs successfully!" -ForegroundColor Green
    } else {
        Write-Warning "Deployed gemma.exe returned exit code $LASTEXITCODE"
        Write-Host "Test output:`n$testResult" -ForegroundColor Yellow
    }
} catch {
    Write-Error "Failed to run deployed gemma.exe: $_"
} finally {
    Pop-Location
}
