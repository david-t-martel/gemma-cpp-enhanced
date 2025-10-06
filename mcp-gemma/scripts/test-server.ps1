# Test MCP Gemma server functionality
param(
    [string]$BaseURL = "http://localhost:8080",
    [string]$WSUrl = "ws://localhost:8081"
)

Write-Host "Testing MCP Gemma Server..." -ForegroundColor Green

# Test HTTP endpoint
Write-Host "Testing HTTP health endpoint..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$BaseURL/health" -Method Get
    Write-Host "✓ Health check passed: $($response.status)" -ForegroundColor Green
} catch {
    Write-Host "✗ Health check failed: $_" -ForegroundColor Red
}

# Test tools endpoint
Write-Host "Testing tools endpoint..." -ForegroundColor Yellow
try {
    $tools = Invoke-RestMethod -Uri "$BaseURL/tools" -Method Get
    Write-Host "✓ Tools endpoint working. Found $($tools.tools.Count) tools" -ForegroundColor Green
    foreach ($tool in $tools.tools) {
        Write-Host "  - $($tool.name): $($tool.description)" -ForegroundColor Cyan
    }
} catch {
    Write-Host "✗ Tools endpoint failed: $_" -ForegroundColor Red
}

# Test text generation
Write-Host "Testing text generation..." -ForegroundColor Yellow
try {
    $genRequest = @{
        prompt = "Hello, how are you?"
        max_tokens = 50
    }
    $response = Invoke-RestMethod -Uri "$BaseURL/generate" -Method Post -Body ($genRequest | ConvertTo-Json) -ContentType "application/json"
    Write-Host "✓ Text generation working" -ForegroundColor Green
    Write-Host "  Response: $($response.text.Substring(0, [Math]::Min(100, $response.text.Length)))..." -ForegroundColor Cyan
} catch {
    Write-Host "✗ Text generation failed: $_" -ForegroundColor Red
}

Write-Host "Testing completed." -ForegroundColor Green
