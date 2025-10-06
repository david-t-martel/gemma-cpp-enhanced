# RAG-Redis MCP Server Windows Service Installation Script
param(
    [string]$ServiceName = "RagRedisMcpServer",
    [string]$BinaryPath = "C:\Program Files\RAG-Redis\bin\mcp-server.exe",
    [switch]$StartService
)

Write-Host "Installing Windows Service: $ServiceName" -ForegroundColor Green
Write-Host "Binary Path: $BinaryPath"

# Create service
sc.exe create $ServiceName binPath= "`"$BinaryPath`"" start= auto DisplayName= "RAG-Redis MCP Server"

if ($LASTEXITCODE -eq 0) {
    Write-Host "Service created successfully" -ForegroundColor Green
    
    if ($StartService) {
        Start-Service -Name $ServiceName
        Write-Host "Service started" -ForegroundColor Green
    }
} else {
    Write-Error "Failed to create service"
}
