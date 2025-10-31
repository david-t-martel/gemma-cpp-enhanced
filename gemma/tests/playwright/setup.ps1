# Setup script for terminal UI test automation (Windows PowerShell)

Write-Host "Setting up Terminal UI Test Automation..." -ForegroundColor Cyan

# Check Python version
$pythonVersion = python --version 2>&1
Write-Host "✓ Python version: $pythonVersion" -ForegroundColor Green

# Install dependencies
Write-Host "`nInstalling dependencies..." -ForegroundColor Cyan
pip install -r playwright_requirements.txt

# Optional: Check for asciinema
if (-not (Get-Command asciinema -ErrorAction SilentlyContinue)) {
    Write-Host "`n⚠ asciinema not found (optional for terminal recording)" -ForegroundColor Yellow
    Write-Host "  Install with: scoop install asciinema" -ForegroundColor Gray
}

# Create output directories
Write-Host "`nCreating output directories..." -ForegroundColor Cyan
@("screenshots", "videos", "recordings", "snapshots") | ForEach-Object {
    if (-not (Test-Path $_)) {
        New-Item -ItemType Directory -Path $_ | Out-Null
    }
}
Write-Host "✓ Directories created" -ForegroundColor Green

# Test installation
Write-Host "`nTesting installation..." -ForegroundColor Cyan
python -c "import pytest; import pyte; import rich; print('✓ All core dependencies available')"

Write-Host "`n✓ Setup complete!" -ForegroundColor Green
Write-Host "`nRun tests with:" -ForegroundColor Cyan
Write-Host "  pytest tests/playwright/ -v" -ForegroundColor White
Write-Host "  python run_tests.py" -ForegroundColor White
Write-Host "`nSee README.md for more usage examples." -ForegroundColor Gray
