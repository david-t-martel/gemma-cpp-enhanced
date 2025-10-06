# Gemma.cpp Performance Benchmarking Suite
# Main benchmark orchestrator script

param(
    [string]$ModelPath = "C:\codedev\llm\.models",
    [string]$BinaryPath = "C:\codedev\llm\gemma\gemma.cpp\build_wsl\gemma",
    [string]$OutputDir = "C:\codedev\llm\benchmarks\results",
    [switch]$QuickTest,
    [switch]$FullBenchmark,
    [switch]$GenerateReport
)

# Ensure WSL is available
if (-not (Get-Command wsl -ErrorAction SilentlyContinue)) {
    Write-Error "WSL is not installed or not in PATH"
    exit 1
}

# Create output directory
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

Write-Host "`n=== Gemma.cpp Performance Benchmark Suite ===" -ForegroundColor Yellow
Write-Host "This script benchmarks token generation performance" -ForegroundColor Gray

# Quick implementation for immediate testing
$models = @(
    @{Name="2B"; Weights="$ModelPath\gemma-gemmacpp-2b-it-v3\2b-it.sbs"; Tokenizer="$ModelPath\gemma-gemmacpp-2b-it-v3\tokenizer.spm"}
)

if (Test-Path "$ModelPath\gemma-3-gemmaCpp-3.0-4b-it-sfp-v1\4b-it-sfp.sbs") {
    $models += @{Name="4B"; Weights="$ModelPath\gemma-3-gemmaCpp-3.0-4b-it-sfp-v1\4b-it-sfp.sbs"; Tokenizer="$ModelPath\gemma-3-gemmaCpp-3.0-4b-it-sfp-v1\tokenizer.spm"}
}

foreach ($model in $models) {
    Write-Host "`nTesting $($model.Name) model..." -ForegroundColor Cyan
    
    $wslBinary = $BinaryPath -replace '\', '/' -replace 'C:', '/mnt/c'
    $wslWeights = $model.Weights -replace '\', '/' -replace 'C:', '/mnt/c'
    $wslTokenizer = $model.Tokenizer -replace '\', '/' -replace 'C:', '/mnt/c'
    
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    $output = wsl bash -c "echo 'What is machine learning?' | $wslBinary --model $wslWeights --tokenizer $wslTokenizer --max_tokens 50 2>&1"
    $stopwatch.Stop()
    
    Write-Host "Execution time: $($stopwatch.Elapsed.TotalSeconds) seconds" -ForegroundColor Green
    
    if ($output -match "(\d+\.?\d*)\s*tokens/sec") {
        Write-Host "Performance: $($matches[1]) tokens/sec" -ForegroundColor Green
    }
}

Write-Host "`n=== Benchmark Complete ===" -ForegroundColor Green
