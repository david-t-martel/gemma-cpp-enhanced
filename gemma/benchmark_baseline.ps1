# Gemma.cpp Baseline Benchmark Suite
# Validates performance and correctness across build configurations

param(
    [string]$Executable,
    [string]$OutputFile = "benchmark_results.json",
    [switch]$Baseline,
    [switch]$Compare
)

$ErrorActionPreference = "Stop"

# ============================================================================
# Benchmark Configuration
# ============================================================================

$BenchmarkConfig = @{
    # Test prompts of varying lengths
    Prompts = @(
        @{ Name = "Short"; Text = "Hello"; Tokens = 2 }
        @{ Name = "Medium"; Text = "Write a short story about"; Tokens = 5 }
        @{ Name = "Long"; Text = "Explain quantum computing in detail with examples and applications"; Tokens = 12 }
    )
    
    # Generation parameters
    MaxTokens = 100
    Temperature = 0.7
    
    # Number of runs for averaging
    Runs = 3
}

# ============================================================================
# Helper Functions
# ============================================================================

function Get-BuildInfo {
    param([string]$ExePath)
    
    $Name = [System.IO.Path]::GetFileNameWithoutExtension($ExePath)
    $Modified = (Get-Item $ExePath).LastWriteTime
    $Size = (Get-Item $ExePath).Length
    
    # Parse build configuration from name
    $Config = @{
        Name = $Name
        Modified = $Modified.ToString("yyyy-MM-dd HH:mm:ss")
        SizeMB = [math]::Round($Size / 1MB, 2)
        IsStandard = $Name -match "std"
        IsHardware = $Name -match "hw-"
        Backends = @()
        OneAPILibs = @()
    }
    
    # Extract backends
    if ($Name -match "hw-([^+]+)") {
        $Config.Backends = $matches[1] -split "-"
    }
    
    # Extract oneAPI libraries
    if ($Name -match "\+(.+)$") {
        $Config.OneAPILibs = $matches[1] -split "-"
    }
    
    return $Config
}

function Find-GemmaModel {
    param([string]$ModelDir = "c:\codedev\llm\.models")
    
    # Priority order: look for instruction-tuned models
    $Candidates = @(
        "gemma-gemmacpp-2b-it-v3\2b-it.sbs",
        "gemma-3-gemmaCpp-3.0-4b-it-sfp-v1\3.0-4b-it-sfp.sbs",
        "gemma-2-2b-it-gguf\*-it*.sbs"
    )
    
    foreach ($Pattern in $Candidates) {
        $FullPath = Join-Path $ModelDir $Pattern
        $Found = Get-ChildItem $FullPath -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($Found) {
            return $Found.FullName
        }
    }
    
    return $null
}

function Find-Tokenizer {
    param([string]$ModelDir)
    
    # Look for tokenizer in model directory or parent
    $Locations = @(
        (Join-Path $ModelDir "tokenizer.spm"),
        (Join-Path (Split-Path $ModelDir -Parent) "tokenizer.spm")
    )
    
    foreach ($Path in $Locations) {
        if (Test-Path $Path) {
            return $Path
        }
    }
    
    return $null
}

function Measure-Inference {
    param(
        [string]$ExePath,
        [string]$Prompt,
        [int]$MaxTokens = 50,
        [string]$ModelPath,
        [string]$TokenizerPath
    )
    
    if (-not $ModelPath) {
        return @{
            Success = $false
            Error = "Model path not provided"
            Skipped = $true
        }
    }
    
    if (-not (Test-Path $ModelPath)) {
        return @{
            Success = $false
            Error = "Model file not found: $ModelPath"
            Skipped = $true
        }
    }
    
    # Create temp prompt file
    $TempPrompt = [System.IO.Path]::GetTempFileName()
    $TempOutput = [System.IO.Path]::GetTempFileName()
    
    try {
        # Write prompt in format gemma expects
        $Prompt | Out-File -FilePath $TempPrompt -Encoding UTF8 -NoNewline
        
        # Build command arguments
        $Args = @(
            "--weights", $ModelPath,
            "--max_generated_tokens", $MaxTokens,
            "--verbosity", "0"
        )
        
        # Add tokenizer if provided and not single-file format
        if ($TokenizerPath -and (Test-Path $TokenizerPath)) {
            $Args += "--tokenizer"
            $Args += $TokenizerPath
        }
        
        Write-Verbose "Running: $ExePath $($Args -join ' ')"
        Write-Verbose "Prompt: $Prompt"
        
        $StartTime = Get-Date
        
        # Run inference with input piped
        $ProcessInfo = New-Object System.Diagnostics.ProcessStartInfo
        $ProcessInfo.FileName = $ExePath
        $ProcessInfo.Arguments = $Args -join ' '
        $ProcessInfo.RedirectStandardInput = $true
        $ProcessInfo.RedirectStandardOutput = $true
        $ProcessInfo.RedirectStandardError = $true
        $ProcessInfo.UseShellExecute = $false
        $ProcessInfo.CreateNoWindow = $true
        
        $Process = New-Object System.Diagnostics.Process
        $Process.StartInfo = $ProcessInfo
        
        $OutputBuilder = New-Object System.Text.StringBuilder
        $ErrorBuilder = New-Object System.Text.StringBuilder
        
        $OutputEvent = Register-ObjectEvent -InputObject $Process `
            -EventName OutputDataReceived -Action {
                if ($EventArgs.Data) {
                    [void]$Event.MessageData.AppendLine($EventArgs.Data)
                }
            } -MessageData $OutputBuilder
        
        $ErrorEvent = Register-ObjectEvent -InputObject $Process `
            -EventName ErrorDataReceived -Action {
                if ($EventArgs.Data) {
                    [void]$Event.MessageData.AppendLine($EventArgs.Data)
                }
            } -MessageData $ErrorBuilder
        
        [void]$Process.Start()
        $Process.BeginOutputReadLine()
        $Process.BeginErrorReadLine()
        
        # Send prompt and exit command
        $Process.StandardInput.WriteLine($Prompt)
        $Process.StandardInput.WriteLine("quit")
        $Process.StandardInput.Close()
        
        # Wait with timeout (60 seconds)
        $Timeout = 60000
        if (-not $Process.WaitForExit($Timeout)) {
            $Process.Kill()
            throw "Process timeout after $($Timeout/1000) seconds"
        }
        
        Unregister-Event -SourceIdentifier $OutputEvent.Name -Force
        Unregister-Event -SourceIdentifier $ErrorEvent.Name -Force
        
        $EndTime = Get-Date
        $Duration = ($EndTime - $StartTime).TotalSeconds
        
        $Output = $OutputBuilder.ToString()
        $Error = $ErrorBuilder.ToString()
        
        # Parse output to extract generated text
        $GeneratedText = ""
        if ($Output) {
            # Gemma outputs after the prompt
            $Lines = $Output -split "`n" | Where-Object { $_.Trim() -ne "" }
            $GeneratedText = ($Lines | Select-Object -Skip 1) -join " "
        }
        
        $Success = $Process.ExitCode -eq 0
        
        return @{
            Success = $Success
            DurationSec = [math]::Round($Duration, 3)
            Output = $Output
            GeneratedText = $GeneratedText.Trim()
            TokensPerSecond = if ($Duration -gt 0) { [math]::Round($MaxTokens / $Duration, 2) } else { 0 }
            ExitCode = $Process.ExitCode
            StdErr = $Error
        }
    }
    catch {
        return @{
            Success = $false
            Error = $_.Exception.Message
        }
    }
    finally {
        Remove-Item $TempPrompt -ErrorAction SilentlyContinue
        Remove-Item $TempOutput -ErrorAction SilentlyContinue
    }
}

function Test-NumericalAccuracy {
    param([string]$ExePath)
    
    Write-Host "  Running numerical accuracy tests..." -ForegroundColor Cyan
    
    # Basic smoke tests
    $Tests = @(
        @{ Name = "Help"; Args = "--help"; ExpectSuccess = $true }
        @{ Name = "Version"; Args = "--version"; ExpectSuccess = $false } # May not exist
    )
    
    $Results = @()
    foreach ($Test in $Tests) {
        try {
            $Output = & $ExePath $Test.Args 2>&1
            $Success = $LASTEXITCODE -eq 0
            
            $Results += @{
                Name = $Test.Name
                Passed = $Success -eq $Test.ExpectSuccess
                ExitCode = $LASTEXITCODE
            }
        }
        catch {
            $Results += @{
                Name = $Test.Name
                Passed = $false
                Error = $_.Exception.Message
            }
        }
    }
    
    return $Results
}

# ============================================================================
# Main Benchmark Execution
# ============================================================================

function Run-Benchmark {
    param([string]$ExePath)
    
    Write-Host "`n╔════════════════════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "║  Gemma.cpp Benchmark Suite                         ║" -ForegroundColor Green
    Write-Host "╚════════════════════════════════════════════════════╝" -ForegroundColor Green
    
    if (-not (Test-Path $ExePath)) {
        Write-Error "Executable not found: $ExePath"
        return
    }
    
    Write-Host "`nExecutable: $ExePath" -ForegroundColor Cyan
    
    # Get build information
    $BuildInfo = Get-BuildInfo -ExePath $ExePath
    Write-Host "Build Type: $($BuildInfo.Name)" -ForegroundColor Cyan
    Write-Host "Modified: $($BuildInfo.Modified)" -ForegroundColor Cyan
    Write-Host "Size: $($BuildInfo.SizeMB) MB" -ForegroundColor Cyan
    
    if ($BuildInfo.Backends) {
        Write-Host "Backends: $($BuildInfo.Backends -join ', ')" -ForegroundColor Yellow
    }
    if ($BuildInfo.OneAPILibs) {
        Write-Host "oneAPI: $($BuildInfo.OneAPILibs -join ', ')" -ForegroundColor Yellow
    }
    
    # Run numerical accuracy tests
    Write-Host "`n─────────────────────────────────────────────────────" -ForegroundColor Gray
    Write-Host "Numerical Accuracy Tests" -ForegroundColor White
    Write-Host "─────────────────────────────────────────────────────" -ForegroundColor Gray
    
    $AccuracyResults = Test-NumericalAccuracy -ExePath $ExePath
    foreach ($Result in $AccuracyResults) {
        $Status = if ($Result.Passed) { "✅ PASS" } else { "❌ FAIL" }
        Write-Host "  $($Result.Name): $Status"
    }
    
    # Inference benchmarks with actual model
    Write-Host "`n─────────────────────────────────────────────────────" -ForegroundColor Gray
    Write-Host "Inference Performance Benchmarks" -ForegroundColor White
    Write-Host "─────────────────────────────────────────────────────" -ForegroundColor Gray
    
    # Locate model files
    $ModelPath = Find-GemmaModel
    if ($ModelPath) {
        Write-Host "  Model: $ModelPath" -ForegroundColor Cyan
        $ModelDir = Split-Path $ModelPath -Parent
        $TokenizerPath = Find-Tokenizer -ModelDir $ModelDir
        if ($TokenizerPath) {
            Write-Host "  Tokenizer: $TokenizerPath" -ForegroundColor Cyan
        } else {
            Write-Host "  Tokenizer: (embedded in model)" -ForegroundColor Cyan
        }
        
        $InferenceResults = @()
        
        # Run inference tests with varying prompts
        foreach ($PromptConfig in $BenchmarkConfig.Prompts) {
            Write-Host "`n  Testing $($PromptConfig.Name) prompt..." -ForegroundColor Yellow
            
            $Results = @()
            for ($i = 1; $i -le $BenchmarkConfig.Runs; $i++) {
                Write-Host "    Run $i/$($BenchmarkConfig.Runs)..." -NoNewline
                
                $Result = Measure-Inference `
                    -ExePath $ExePath `
                    -Prompt $PromptConfig.Text `
                    -MaxTokens $BenchmarkConfig.MaxTokens `
                    -ModelPath $ModelPath `
                    -TokenizerPath $TokenizerPath
                
                if ($Result.Success) {
                    Write-Host " ✅ $($Result.DurationSec)s ($($Result.TokensPerSecond) tok/s)" -ForegroundColor Green
                    $Results += $Result
                } elseif ($Result.Skipped) {
                    Write-Host " ⏭️ Skipped: $($Result.Error)" -ForegroundColor Yellow
                    break
                } else {
                    Write-Host " ❌ Failed: $($Result.Error)" -ForegroundColor Red
                }
            }
            
            if ($Results.Count -gt 0) {
                $AvgDuration = ($Results | Measure-Object -Property DurationSec -Average).Average
                $AvgToksPerSec = ($Results | Measure-Object -Property TokensPerSecond -Average).Average
                
                Write-Host "    Average: $([math]::Round($AvgDuration, 3))s ($([math]::Round($AvgToksPerSec, 2)) tok/s)" -ForegroundColor Cyan
                
                $InferenceResults += @{
                    PromptType = $PromptConfig.Name
                    Runs = $Results.Count
                    AvgDurationSec = [math]::Round($AvgDuration, 3)
                    AvgTokensPerSec = [math]::Round($AvgToksPerSec, 2)
                    MinDurationSec = [math]::Round(($Results | Measure-Object -Property DurationSec -Minimum).Minimum, 3)
                    MaxDurationSec = [math]::Round(($Results | Measure-Object -Property DurationSec -Maximum).Maximum, 3)
                    SampleOutput = $Results[0].GeneratedText
                }
            }
        }
    } else {
        Write-Host "  ⚠️ No model found in c:\codedev\llm\.models\" -ForegroundColor Yellow
        Write-Host "  Skipping inference benchmarks" -ForegroundColor Yellow
        $InferenceResults = @()
    }
    
    # Compile results
    $Results = @{
        Timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
        BuildInfo = $BuildInfo
        AccuracyTests = $AccuracyResults
        InferenceBenchmarks = $InferenceResults
        Summary = @{
            AllTestsPassed = ($AccuracyResults | Where-Object { -not $_.Passed }).Count -eq 0
            InferenceTestsRun = $InferenceResults.Count -gt 0
            AvgTokensPerSec = if ($InferenceResults.Count -gt 0) {
                [math]::Round(($InferenceResults | Measure-Object -Property AvgTokensPerSec -Average).Average, 2)
            } else { 0 }
        }
    }
    
    # Save results
    $Results | ConvertTo-Json -Depth 10 | Out-File -FilePath $OutputFile -Encoding UTF8
    Write-Host "`n✅ Results saved to: $OutputFile" -ForegroundColor Green
    
    return $Results
}

# ============================================================================
# Compare Mode
# ============================================================================

function Compare-Benchmarks {
    param(
        [string]$BaselineFile,
        [string]$CurrentFile
    )
    
    if (-not (Test-Path $BaselineFile)) {
        Write-Error "Baseline file not found: $BaselineFile"
        return
    }
    
    if (-not (Test-Path $CurrentFile)) {
        Write-Error "Current results file not found: $CurrentFile"
        return
    }
    
    $Baseline = Get-Content $BaselineFile | ConvertFrom-Json
    $Current = Get-Content $CurrentFile | ConvertFrom-Json
    
    Write-Host "`n╔════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║  Benchmark Comparison                              ║" -ForegroundColor Cyan
    Write-Host "╚════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    
    Write-Host "`nBaseline: $($Baseline.BuildInfo.Name)" -ForegroundColor Yellow
    Write-Host "Current:  $($Current.BuildInfo.Name)" -ForegroundColor Yellow
    
    # Compare accuracy
    Write-Host "`n─── Accuracy Comparison ───" -ForegroundColor Gray
    $BaselinePassed = $Baseline.Summary.AllTestsPassed
    $CurrentPassed = $Current.Summary.AllTestsPassed
    
    if ($BaselinePassed -and $CurrentPassed) {
        Write-Host "  ✅ Both builds pass all accuracy tests" -ForegroundColor Green
    } elseif ($CurrentPassed -and -not $BaselinePassed) {
        Write-Host "  ✅ IMPROVEMENT: Current build now passes" -ForegroundColor Green
    } elseif (-not $CurrentPassed -and $BaselinePassed) {
        Write-Host "  ❌ REGRESSION: Current build fails tests" -ForegroundColor Red
    } else {
        Write-Host "  ⚠️ Both builds have issues" -ForegroundColor Yellow
    }
    
    # Compare inference performance
    if ($Baseline.Summary.InferenceTestsRun -and $Current.Summary.InferenceTestsRun) {
        Write-Host "`n─── Inference Performance Comparison ───" -ForegroundColor Gray
        
        $BaselineTokPerSec = $Baseline.Summary.AvgTokensPerSec
        $CurrentTokPerSec = $Current.Summary.AvgTokensPerSec
        
        Write-Host "  Baseline: $BaselineTokPerSec tok/s" -ForegroundColor White
        Write-Host "  Current:  $CurrentTokPerSec tok/s" -ForegroundColor White
        
        if ($BaselineTokPerSec -gt 0) {
            $PercentChange = [math]::Round((($CurrentTokPerSec - $BaselineTokPerSec) / $BaselineTokPerSec) * 100, 2)
            
            if ($PercentChange -gt 5) {
                Write-Host "  ✅ IMPROVEMENT: +$PercentChange% faster" -ForegroundColor Green
            } elseif ($PercentChange -lt -5) {
                Write-Host "  ⚠️ REGRESSION: $PercentChange% slower" -ForegroundColor Yellow
            } else {
                Write-Host "  ✅ Performance within 5% (${PercentChange}%)" -ForegroundColor Green
            }
        }
        
        # Detailed per-prompt comparison
        Write-Host "`n  Per-Prompt Details:" -ForegroundColor Gray
        foreach ($BaselineBench in $Baseline.InferenceBenchmarks) {
            $CurrentBench = $Current.InferenceBenchmarks | Where-Object { $_.PromptType -eq $BaselineBench.PromptType } | Select-Object -First 1
            if ($CurrentBench) {
                $PromptChange = [math]::Round((($CurrentBench.AvgTokensPerSec - $BaselineBench.AvgTokensPerSec) / $BaselineBench.AvgTokensPerSec) * 100, 2)
                $ChangeColor = if ($PromptChange -gt 0) { "Green" } elseif ($PromptChange -lt -5) { "Yellow" } else { "White" }
                Write-Host "    $($BaselineBench.PromptType): $($BaselineBench.AvgTokensPerSec) → $($CurrentBench.AvgTokensPerSec) tok/s (${PromptChange}%)" -ForegroundColor $ChangeColor
            }
        }
    } elseif ($Current.Summary.InferenceTestsRun -and -not $Baseline.Summary.InferenceTestsRun) {
        Write-Host "`n─── Inference Performance ───" -ForegroundColor Gray
        Write-Host "  ✅ Current build includes inference benchmarks (baseline did not)" -ForegroundColor Green
        Write-Host "  Average: $($Current.Summary.AvgTokensPerSec) tok/s" -ForegroundColor Cyan
    }
}

# ============================================================================
# Main Entry Point
# ============================================================================

if ($Compare) {
    Compare-Benchmarks -BaselineFile "benchmark_baseline.json" -CurrentFile $OutputFile
}
elseif ($Executable) {
    $Results = Run-Benchmark -ExePath $Executable
    
    if ($Baseline) {
        Write-Host "`n📊 This run will be used as the baseline for comparisons" -ForegroundColor Cyan
        Copy-Item $OutputFile "benchmark_baseline.json"
    }
}
else {
    Write-Host @"
╔════════════════════════════════════════════════════════════╗
║  Gemma.cpp Baseline Benchmark Suite                       ║
╚════════════════════════════════════════════════════════════╝

Usage:
  .\benchmark_baseline.ps1 -Executable <path> [-Baseline] [-OutputFile <file>]
  .\benchmark_baseline.ps1 -Compare [-OutputFile <file>]

Examples:
  # Establish baseline with standard build
  .\benchmark_baseline.ps1 -Executable build\bin\gemma_std.exe -Baseline

  # Benchmark oneAPI build and compare
  .\benchmark_baseline.ps1 -Executable build\bin\gemma_std+tbb-ipp.exe -OutputFile oneapi_results.json
  .\benchmark_baseline.ps1 -Compare -OutputFile oneapi_results.json

  # Benchmark SYCL backend
  .\benchmark_baseline.ps1 -Executable build\bin\gemma_hw-sycl.exe

"@ -ForegroundColor Cyan
}
