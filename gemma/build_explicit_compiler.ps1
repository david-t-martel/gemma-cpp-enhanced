#!/usr/bin/env powershell
# Build with explicit compiler paths
$ErrorActionPreference = "Stop"

Write-Host "=== Building Gemma.cpp (Explicit Compiler) ===" -ForegroundColor Cyan

$msvcVersion = "14.44.35207"
$msvcRoot = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\$msvcVersion"
$clPath = "$msvcRoot\bin\Hostx64\x64\cl.exe"

if (-not (Test-Path $clPath)) {
    Write-Host "ERROR: Compiler not found at $clPath" -ForegroundColor Red
    exit 1
}

Write-Host "Found compiler: $clPath" -ForegroundColor Green

# Clean build directory
if (Test-Path "build") {
    Write-Host "Removing old build directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force build
}

# Set compiler environment variables
$env:CC = $clPath
$env:CXX = $clPath

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
$exePath = "build\bin\Release\gemma.exe"
if (Test-Path $exePath) {
    $size = (Get-Item $exePath).Length / 1MB
    $timestamp = (Get-Item $exePath).LastWriteTime
    Write-Host ""
    Write-Host "=== Build Success ===" -ForegroundColor Green
    Write-Host "Binary: $exePath" -ForegroundColor White
    Write-Host "Size: $([math]::Round($size, 2)) MB" -ForegroundColor White
    Write-Host "Timestamp: $timestamp" -ForegroundColor White

    # Test for session flags
    Write-Host ""
    Write-Host "Checking for session management flags..." -ForegroundColor Yellow
    $helpOutput = & $exePath --help 2>&1 | Out-String
    if ($helpOutput -match "--session|--load_session|--save_on_exit") {
        Write-Host "Session management flags FOUND" -ForegroundColor Green
    } else {
        Write-Host "Session management flags NOT FOUND" -ForegroundColor Yellow
    }

    # Copy to deploy
    if (-not (Test-Path "deploy")) {
        New-Item -ItemType Directory -Path "deploy" | Out-Null
    }
    Copy-Item $exePath "deploy\gemma.exe" -Force
    Write-Host "Binary copied to deploy\gemma.exe" -ForegroundColor Green
} else {
    Write-Host "ERROR: Binary not found" -ForegroundColor Red
    exit 1
}
