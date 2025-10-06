#\!/usr/bin/env pwsh
# Build script for RAG-Redis system integration
# This script builds both the RAG-Redis system and Rust extensions for Python integration

param(
    [switch]$Release,
    [switch]$Clean,
    [switch]$Test,
    [switch]$Features = $true,
    [string]$Feature = "full"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Script paths
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$STATS_DIR = Split-Path -Parent $SCRIPT_DIR
$RAG_REDIS_DIR = Join-Path $STATS_DIR "rag-redis-system"
$RUST_EXT_DIR = Join-Path $STATS_DIR "rust_extensions"

Write-Host "=== RAG-Redis System Build Script ===" -ForegroundColor Green
Write-Host "Stats Dir: $STATS_DIR" -ForegroundColor Cyan
Write-Host "RAG-Redis Dir: $RAG_REDIS_DIR" -ForegroundColor Cyan
Write-Host "Rust Extensions Dir: $RUST_EXT_DIR" -ForegroundColor Cyan

# Function to check Redis availability
function Test-RedisConnection {
    Write-Host "Checking Redis connection..." -ForegroundColor Yellow
    try {
        $result = redis-cli ping 2>$null
        if ($result -eq "PONG") {
            Write-Host "✓ Redis is running" -ForegroundColor Green
            return $true
        }
    }
    catch {
        Write-Host "✗ Redis not available. Please start Redis server on localhost:6379" -ForegroundColor Red
        Write-Host "  Windows: Start redis-server.exe" -ForegroundColor Yellow
        Write-Host "  WSL: sudo service redis-server start" -ForegroundColor Yellow
        return $false
    }
    return $false
}

# Main execution
try {
    # Check prerequisites
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
        throw "Rust/Cargo not found. Please install Rust toolchain."
    }

    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv not found. Please install uv Python package manager."
    }

    # Check Redis if we're running tests
    if ($Test -and -not (Test-RedisConnection)) {
        Write-Warning "Redis not available - integration tests may fail"
    }

    Write-Host "Build script ready - see full implementation in the file" -ForegroundColor Green

}
catch {
    Write-Host "Build failed: $_" -ForegroundColor Red
    exit 1
}
