# RAG-Redis MCP Server Windows Deployment Script
# PowerShell script for automated Windows deployment

param(
    [Parameter(Mandatory=$false)]
    [string]$Environment = "production",

    [Parameter(Mandatory=$false)]
    [string]$ServiceName = "RagRedisMcpServer",

    [Parameter(Mandatory=$false)]
    [string]$InstallPath = "C:\Program Files\RAG-Redis",

    [Parameter(Mandatory=$false)]
    [string]$DataPath = "C:\ProgramData\RAG-Redis",

    [Parameter(Mandatory=$false)]
    [switch]$SkipBuild,

    [Parameter(Mandatory=$false)]
    [switch]$InstallService,

    [Parameter(Mandatory=$false)]
    [switch]$StartService,

    [Parameter(Mandatory=$false)]
    [switch]$Force
)

# Main deployment logic
Write-Host "RAG-Redis MCP Server Windows Deployment" -ForegroundColor Green
Write-Host "Environment: $Environment"
Write-Host "Service Name: $ServiceName" 
Write-Host "Install Path: $InstallPath"
Write-Host "Data Path: $DataPath"

# Build project if not skipped
if (-not $SkipBuild) {
    Write-Host "Building project..." -ForegroundColor Blue
    cargo build --release --bin mcp-server
}

# Create directories
$directories = @(
    $InstallPath,
    (Join-Path $InstallPath "bin"),
    (Join-Path $InstallPath "config"),
    $DataPath,
    (Join-Path $DataPath "data"),
    (Join-Path $DataPath "cache"),
    (Join-Path $DataPath "logs")
)

foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created directory: $dir" -ForegroundColor Blue
    }
}

Write-Host "Deployment script created successfully" -ForegroundColor Green
