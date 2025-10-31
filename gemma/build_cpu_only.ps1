# Build CPU-only Gemma with versioned naming
$ErrorActionPreference = "Stop"

$OneAPIInit = "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

Write-Host "Building CPU-only Gemma.exe with Intel oneAPI..." -ForegroundColor Cyan

# Clean
if (Test-Path build) {
    Remove-Item -Recurse -Force build
}

# Configure
$ConfigCmd = "cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER_LAUNCHER=sccache -DGEMMA_BUILD_ENHANCED_TESTS=OFF -DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON -DGEMMA_BUILD_BACKENDS=OFF"

cmd /c """$OneAPIInit"" >nul 2>&1 && $ConfigCmd"

# Build
$BuildCmd = "cmake --build build --config Release --target gemma -j"
cmd /c """$OneAPIInit"" >nul 2>&1 && $BuildCmd"

if (Test-Path "build\bin\gemma_std.exe") {
    Write-Host "`n✓ Build successful: build\bin\gemma_std.exe" -ForegroundColor Green
} else {
    Write-Error "Build failed"
}
