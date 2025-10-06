# Quick benchmark script for immediate testing

Write-Host "`n=== Gemma.cpp Quick Performance Test ===" -ForegroundColor Cyan
Write-Host "This will run a quick benchmark to verify your setup" -ForegroundColor Gray
Write-Host ""

# Configuration
$binaryPath = "C:\codedev\llm\gemma\gemma.cpp\build_wsl\gemma"
$model2B = "C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\2b-it.sbs"
$tokenizer2B = "C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\tokenizer.spm"

# Check prerequisites
Write-Host "Checking prerequisites..." -ForegroundColor Yellow

if (-not (Get-Command wsl -ErrorAction SilentlyContinue)) {
    Write-Error "WSL is not installed or not in PATH"
    exit 1
}
Write-Host "  WSL found" -ForegroundColor Green

if (-not (Test-Path $binaryPath)) {
    Write-Error "Gemma binary not found at: $binaryPath"
    exit 1
}
Write-Host "  Binary found" -ForegroundColor Green

if (-not (Test-Path $model2B)) {
    Write-Error "2B model not found at: $model2B"
    exit 1
}
Write-Host "  Model found" -ForegroundColor Green

# Convert paths for WSL
$wslBinary = $binaryPath -replace '\', '/' -replace 'C:', '/mnt/c'
$wslWeights = $model2B -replace '\', '/' -replace 'C:', '/mnt/c'
$wslTokenizer = $tokenizer2B -replace '\', '/' -replace 'C:', '/mnt/c'

Write-Host "`nRunning quick benchmark..." -ForegroundColor Cyan

# Test 1: Simple generation
Write-Host "Test 1: Simple generation (50 tokens)" -ForegroundColor White
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$output = wsl bash -c "echo 'What is machine learning?' | $wslBinary --model $wslWeights --tokenizer $wslTokenizer --max_tokens 50 --temperature 0.7 2>&1"
$stopwatch.Stop()

Write-Host "  Time: $([math]::Round($stopwatch.Elapsed.TotalSeconds, 2))s" -ForegroundColor Gray
if ($output -match "(\d+\.?\d*)\s*tokens/sec") {
    Write-Host "  Speed: $($matches[1]) tokens/sec" -ForegroundColor Green
}

# Test 2: Longer generation
Write-Host "`nTest 2: Longer generation (200 tokens)" -ForegroundColor White
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$output = wsl bash -c "echo 'Explain how neural networks work' | $wslBinary --model $wslWeights --tokenizer $wslTokenizer --max_tokens 200 --temperature 0.7 2>&1"
$stopwatch.Stop()

Write-Host "  Time: $([math]::Round($stopwatch.Elapsed.TotalSeconds, 2))s" -ForegroundColor Gray
if ($output -match "(\d+\.?\d*)\s*tokens/sec") {
    Write-Host "  Speed: $($matches[1]) tokens/sec" -ForegroundColor Green
}

Write-Host "`n=== Quick Test Complete ===" -ForegroundColor Green
