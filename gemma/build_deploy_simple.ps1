# Build script for deployment
# Initialize Visual Studio environment and build gemma.exe

Write-Host "Initializing Visual Studio 2022 environment..." -ForegroundColor Cyan

# Import VS Developer Shell
Import-Module "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
Enter-VsDevShell -VsInstallPath "C:\Program Files\Microsoft Visual Studio\2022\Community" -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64"

Write-Host "Visual Studio environment loaded" -ForegroundColor Green
Write-Host ""

# Configure
Write-Host "Configuring CMake..." -ForegroundColor Cyan
& "C:\Program Files\CMake\bin\cmake.exe" -B build_deploy -G "Visual Studio 17 2022" -A x64 -T v143

if ($LASTEXITCODE -ne 0) {
    Write-Host "Configuration failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Configuration complete" -ForegroundColor Green
Write-Host ""

# Build
Write-Host "Building gemma.exe (this may take 10-15 minutes)..." -ForegroundColor Cyan
& "C:\Program Files\CMake\bin\cmake.exe" --build build_deploy --config Release -j 10 --target gemma

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Build complete!" -ForegroundColor Green
Write-Host "Binary location: build_deploy\Release\gemma.exe" -ForegroundColor White
