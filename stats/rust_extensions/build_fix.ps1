#!/usr/bin/env powershell
# Script to fix PyO3 binding warnings in rust_extensions

# Set the PYO3_PYTHON environment variable
$env:PYO3_PYTHON = "C:\Users\david\AppData\Roaming\uv\python\cpython-3.13.3-windows-x86_64-none\python.exe"

Write-Host "Setting PYO3_PYTHON to: $env:PYO3_PYTHON"

# Change to the rust_extensions directory
Set-Location "C:\codedev\llm\stats\rust_extensions"

Write-Host "Current directory: $(Get-Location)"

# Verify Python executable exists
if (Test-Path $env:PYO3_PYTHON) {
    Write-Host "✓ Python executable found"
    & $env:PYO3_PYTHON --version
} else {
    Write-Host "✗ Python executable not found at: $env:PYO3_PYTHON"
    exit 1
}

Write-Host "`n=== Running cargo check ==="
try {
    cargo check
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ cargo check passed"
    } else {
        Write-Host "✗ cargo check failed with exit code: $LASTEXITCODE"
    }
} catch {
    Write-Host "✗ cargo check failed with error: $_"
}

Write-Host "`n=== Running cargo clippy with warnings as errors ==="
try {
    cargo clippy --all-targets --all-features -- -D warnings
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ cargo clippy passed"
    } else {
        Write-Host "✗ cargo clippy failed with exit code: $LASTEXITCODE"
    }
} catch {
    Write-Host "✗ cargo clippy failed with error: $_"
}

Write-Host "`n=== Building with maturin ==="
try {
    uv run maturin develop --release
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ maturin develop passed"
    } else {
        Write-Host "✗ maturin develop failed with exit code: $LASTEXITCODE"
    }
} catch {
    Write-Host "✗ maturin develop failed with error: $_"
}

Write-Host "`nScript completed."
