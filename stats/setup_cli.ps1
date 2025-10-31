#!/usr/bin/env pwsh
# Setup script for Gemma CLI refactoring

Write-Host "Gemma CLI Setup - Click Framework Migration" -ForegroundColor Cyan
Write-Host "=" * 60

# Check if uv is available
if (!(Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] uv not found. Install with: pip install uv" -ForegroundColor Red
    exit 1
}

# Navigate to project directory
$PROJECT_ROOT = "C:\codedev\llm\stats"
Set-Location $PROJECT_ROOT

Write-Host "`n[1/5] Installing dependencies..." -ForegroundColor Yellow
uv pip install -e .

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to install dependencies" -ForegroundColor Red
    exit 1
}

Write-Host "`n[2/5] Creating configuration..." -ForegroundColor Yellow
if (Test-Path "config/config.toml") {
    Write-Host "  Config already exists: config/config.toml" -ForegroundColor Green
} else {
    uv run gemma config init
}

Write-Host "`n[3/5] Verifying installation..." -ForegroundColor Yellow
uv run gemma --version
uv run gemma --help | Select-Object -First 20

Write-Host "`n[4/5] Running health check..." -ForegroundColor Yellow
uv run gemma health

Write-Host "`n[5/5] Validating configuration..." -ForegroundColor Yellow
uv run gemma config validate

Write-Host "`n" + ("=" * 60)
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "`nQuick Start:" -ForegroundColor Cyan
Write-Host "  uv run gemma chat interactive" -ForegroundColor White
Write-Host "  uv run gemma memory stats" -ForegroundColor White
Write-Host "  uv run gemma model list" -ForegroundColor White
Write-Host "`nFor help:" -ForegroundColor Cyan
Write-Host "  uv run gemma --help" -ForegroundColor White
Write-Host "  uv run gemma chat --help" -ForegroundColor White
Write-Host "`nShell Completion:" -ForegroundColor Cyan
Write-Host "  PowerShell: Add to `$PROFILE" -ForegroundColor White
Write-Host '    Invoke-Expression (& gemma completion bash)' -ForegroundColor Dim
