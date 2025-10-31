# build-simple.ps1 - Simplified accelerated build script
param(
    [string]$Config = "Release",
    [int]$Jobs = 12,
    [switch]$Clean
)

Write-Host "Gemma.cpp Accelerated Build" -ForegroundColor Cyan
Write-Host "Configuration: $Config, Jobs: $Jobs" -ForegroundColor Gray
Write-Host ""

# Setup environment
Write-Host "Setting up environment..." -ForegroundColor Yellow
$env:CMAKE_C_COMPILER_LAUNCHER = "sccache"
$env:CMAKE_CXX_COMPILER_LAUNCHER = "sccache"
$env:CMAKE_GENERATOR = "Ninja Multi-Config"

# Clean if requested
if ($Clean -and (Test-Path "build-ninja")) {
    Write-Host "Cleaning build directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "build-ninja"
}

# Configure
Write-Host "Configuring with CMake..." -ForegroundColor Yellow
cmake --preset ninja-accelerated
if ($LASTEXITCODE -ne 0) {
    Write-Host "Configuration failed!" -ForegroundColor Red
    exit 1
}

# Build
Write-Host "Building (this may take 12-15 minutes)..." -ForegroundColor Yellow
cmake --build build-ninja --config $Config -j $Jobs
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Build complete!" -ForegroundColor Green
Write-Host "Executable: .\build-ninja\$Config\gemma.exe" -ForegroundColor White
Write-Host ""

# Show cache stats
if (Get-Command sccache -ErrorAction SilentlyContinue) {
    Write-Host "Cache statistics:" -ForegroundColor Cyan
    sccache --show-stats
}
