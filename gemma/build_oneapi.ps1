# Build Gemma.cpp with oneAPI
# Usage: .\build_oneapi.ps1 [-Config <std|tbb-ipp|perfpack>] [-Clean]

param(
    [ValidateSet("std", "tbb-ipp", "perfpack")]
    [string]$Config = "std",
    [int]$Jobs = 10,
    [switch]$Clean,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

# Configuration mapping
$BuildConfigs = @{
    "std" = @{
        Name = "Standard"
        BuildDir = "build_std"
        CMakeArgs = @()
    }
    "tbb-ipp" = @{
        Name = "TBB + IPP"
        BuildDir = "build_tbb_ipp"
        CMakeArgs = @(
            "-DGEMMA_USE_ONEAPI_LIBS=ON",
            "-DGEMMA_USE_TBB=ON",
            "-DGEMMA_USE_IPP=ON"
        )
    }
    "perfpack" = @{
        Name = "Performance Pack"
        BuildDir = "build_perfpack"
        CMakeArgs = @(
            "-DGEMMA_USE_ONEAPI_LIBS=ON",
            "-DGEMMA_ONEAPI_PERFORMANCE_PACK=ON"
        )
    }
}

$SelectedConfig = $BuildConfigs[$Config]
$BuildDir = $SelectedConfig.BuildDir

Write-Host "=== Building Gemma.cpp: $($SelectedConfig.Name) ===" -ForegroundColor Cyan
Write-Host "Build directory: $BuildDir" -ForegroundColor Gray

# Clean if requested
if ($Clean -and (Test-Path $BuildDir)) {
    Write-Host "Cleaning $BuildDir..." -ForegroundColor Yellow
    Remove-Item -Path $BuildDir -Recurse -Force
}

# oneAPI paths
$OneAPIRoot = "C:\Program Files (x86)\Intel\oneAPI"
$SetvarsScript = Join-Path $OneAPIRoot "setvars.bat"

if (-not (Test-Path $SetvarsScript)) {
    Write-Error "oneAPI setvars.bat not found at: $SetvarsScript"
    exit 1
}

Write-Host "`nInitializing oneAPI environment..." -ForegroundColor Yellow

# Create temporary batch script
$TempBatch = [System.IO.Path]::GetTempFileName() + ".bat"

$BatchContent = @"
@echo off
call "$SetvarsScript"
if %ERRORLEVEL% neq 0 (
    echo Failed to initialize oneAPI environment
    exit /b 1
)

echo.
echo === CMake Configuration ===
cmake -S . -B "$BuildDir" -G "Ninja" ^
  -DCMAKE_C_COMPILER=icx ^
  -DCMAKE_CXX_COMPILER=icx ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON ^
  -DGEMMA_PREFER_SYSTEM_DEPS=OFF ^
  $($SelectedConfig.CMakeArgs -join ' ^`n  ')

if %ERRORLEVEL% neq 0 (
    echo CMake configuration failed
    exit /b 1
)

echo.
echo === Building ===
cmake --build "$BuildDir" --config Release --parallel $Jobs

if %ERRORLEVEL% neq 0 (
    echo Build failed
    exit /b 1
)

echo.
echo === Build Complete ===
dir "$BuildDir\bin\gemma*.exe"
"@

$BatchContent | Out-File -FilePath $TempBatch -Encoding ASCII

try {
    # Execute batch script
    $Process = Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$TempBatch`"" -NoNewWindow -Wait -PassThru
    
    if ($Process.ExitCode -ne 0) {
        Write-Error "Build process failed with exit code $($Process.ExitCode)"
        exit $Process.ExitCode
    }
    
    # Verify output
    $Executables = Get-ChildItem "$BuildDir\bin\gemma*.exe" -ErrorAction SilentlyContinue
    
    if ($Executables) {
        Write-Host "`n=== Build Successful ===" -ForegroundColor Green
        Write-Host "Executables:" -ForegroundColor Cyan
        foreach ($exe in $Executables) {
            $SizeMB = [math]::Round($exe.Length / 1MB, 2)
            Write-Host "  $($exe.Name) ($SizeMB MB)" -ForegroundColor Green
        }
        
        # Quick smoke test
        Write-Host "`nRunning smoke test..." -ForegroundColor Yellow
        $TestExe = $Executables[0].FullName
        & $TestExe --help 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  Smoke test PASSED" -ForegroundColor Green
        } else {
            Write-Warning "Smoke test returned exit code $LASTEXITCODE"
        }
    } else {
        Write-Error "Build completed but no executables found in $BuildDir\bin"
        exit 1
    }
}
finally {
    Remove-Item $TempBatch -Force -ErrorAction SilentlyContinue
}

Write-Host "`nBuild artifacts location: $BuildDir\bin" -ForegroundColor Cyan
