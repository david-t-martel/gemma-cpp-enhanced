# Setup environment for MCP Gemma server
param(
    [string]$Environment = "development"
)

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

Write-Host "Setting up MCP Gemma environment..." -ForegroundColor Green

# Check Python
try {
    $pythonVersion = python --version
    Write-Host "Found Python: $pythonVersion" -ForegroundColor Blue
} catch {
    Write-Error "Python not found. Please install Python 3.8 or higher."
    exit 1
}

# Setup virtual environment
Push-Location $ProjectRoot
try {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv

    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    .\.venv\Scripts\python.exe -m pip install aiohttp aiohttp-cors websockets redis pyyaml python-dotenv

    Write-Host "Setup completed successfully!" -ForegroundColor Green
    Write-Host "To start server: .\.venv\Scripts\python.exe server\main.py --model <path-to-model>" -ForegroundColor Cyan

} finally {
    Pop-Location
}
