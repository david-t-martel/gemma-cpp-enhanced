# Start MCP Gemma server
param(
    [string]$ModelPath = "C:\codedev\llm\.models\gemma2-2b-it-sfp.sbs",
    [string]$Mode = "all",
    [int]$Port = 8080,
    [int]$WSPort = 8081,
    [switch]$Debug
)

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$ServerScript = Join-Path $ProjectRoot "server\main.py"

Write-Host "Starting MCP Gemma Server..." -ForegroundColor Green
Write-Host "Model: $ModelPath" -ForegroundColor Blue
Write-Host "Mode: $Mode" -ForegroundColor Blue
Write-Host "HTTP Port: $Port" -ForegroundColor Blue
Write-Host "WebSocket Port: $WSPort" -ForegroundColor Blue

$args = @(
    $ServerScript,
    "--mode", $Mode,
    "--host", "localhost",
    "--port", $Port,
    "--ws-port", $WSPort,
    "--model", $ModelPath
)

if ($Debug) {
    $args += "--debug"
}

Push-Location $ProjectRoot
try {
    if (Test-Path ".venv\Scripts\python.exe") {
        & .\.venv\Scripts\python.exe @args
    } else {
        & python @args
    }
} finally {
    Pop-Location
}
