# Build Gemma with MSVC using local dependencies
# Bypasses FetchContent git issues

param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

Write-Host "=== Building Gemma.cpp with MSVC (Local Dependencies) ===" -ForegroundColor Cyan

$BuildDir = "build_msvc"

if ($Clean -and (Test-Path $BuildDir)) {
    Write-Host "Cleaning $BuildDir..." -ForegroundColor Yellow
    Remove-Item -Path $BuildDir -Recurse -Force
}

# Step 1: Configure with MSVC
Write-Host "`nConfiguring with MSVC..." -ForegroundColor Yellow

$ConfigArgs = @(
    "-S", ".",
    "-B", $BuildDir,
    "-G", "Visual Studio 17 2022",
    "-A", "x64",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_CONFIGURATION_TYPES=Release",
    "-DGEMMA_PREFER_SYSTEM_DEPS=ON",
    "-DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON"
)

& cmake @ConfigArgs

if ($LASTEXITCODE -ne 0) {
    Write-Error "CMake configuration failed"
    exit 1
}

Write-Host "`nConfiguration successful!" -ForegroundColor Green

# Step 2: Build
Write-Host "`nBuilding..." -ForegroundColor Yellow

& cmake --build $BuildDir --config Release --parallel

if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed"
    exit 1
}

Write-Host "`nBuild successful!" -ForegroundColor Green

# Step 3: Verify
$Executables = Get-ChildItem "$BuildDir\bin\Release\gemma*.exe" -ErrorAction SilentlyContinue

if (-not $Executables) {
    # Try alternate location
    $Executables = Get-ChildItem "$BuildDir\Release\gemma*.exe" -ErrorAction SilentlyContinue
}

if ($Executables) {
    Write-Host "`n=== Build Output ===" -ForegroundColor Cyan
    foreach ($exe in $Executables) {
        $SizeMB = [math]::Round($exe.Length / 1MB, 2)
        Write-Host "  $($exe.Name) ($SizeMB MB)" -ForegroundColor Green
        
        # Smoke test
        Write-Host "  Testing..." -NoNewline
        & $exe.FullName --help > $null 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host " OK" -ForegroundColor Green
        } else {
            Write-Host " FAILED (exit code $LASTEXITCODE)" -ForegroundColor Red
        }
    }
} else {
    Write-Warning "No executables found. Check build output."
}

Write-Host "`nDone! Executables in: $BuildDir\bin\Release" -ForegroundColor Cyan
