#!/usr/bin/env powershell
# Direct build script for gemma.exe
$ErrorActionPreference = "Stop"

Write-Host "=== Clean Build of Gemma.cpp ===" -ForegroundColor Cyan

# Set up environment
$env:PATH = "C:\Program Files\CMake\bin;$env:PATH"

# Clean build directory
if (Test-Path "build") {
    Write-Host "Removing old build directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force build
}

# Configure
Write-Host "Configuring with CMake..." -ForegroundColor Yellow
& "C:\Program Files\CMake\bin\cmake.exe" -B build -G "Visual Studio 17 2022" -T v143 -A x64

if ($LASTEXITCODE -ne 0) {
    Write-Host "Configuration failed!" -ForegroundColor Red
    exit 1
}

# Build
Write-Host "Building Release configuration..." -ForegroundColor Yellow
& "C:\Program Files\CMake\bin\cmake.exe" --build build --config Release -j 10

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

# Check for output
Write-Host ""
Write-Host "=== Build Complete ===" -ForegroundColor Green

$exePath = "build\bin\Release\gemma.exe"
if (Test-Path $exePath) {
    $size = (Get-Item $exePath).Length / 1MB
    $timestamp = (Get-Item $exePath).LastWriteTime
    Write-Host "Binary: $exePath" -ForegroundColor White
    Write-Host "Size: $([math]::Round($size, 2)) MB" -ForegroundColor White
    Write-Host "Timestamp: $timestamp" -ForegroundColor White

    # Test help output
    Write-Host ""
    Write-Host "Testing --help flag..." -ForegroundColor Yellow
    & $exePath --help

    # Copy to deploy directory
    if (-not (Test-Path "deploy")) {
        New-Item -ItemType Directory -Path "deploy" | Out-Null
    }
    Copy-Item $exePath "deploy\gemma.exe" -Force
    Write-Host ""
    Write-Host "Binary copied to deploy\gemma.exe" -ForegroundColor Green
} else {
    Write-Host "ERROR: Binary not found at $exePath" -ForegroundColor Red
    exit 1
}
