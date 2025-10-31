# Gemma.cpp Build Comparison Tool
# Automatically builds multiple configurations and compares performance

param(
    [string[]]$Configurations = @("std", "tbb", "tbb-ipp", "tbb-ipp-dnnl", "sycl"),
    [string]$BuildDir = "build",
    [string]$ResultsDir = "build_comparison_results",
    [switch]$SkipBuild,
    [switch]$SkipBenchmark,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

# ============================================================================
# Configuration Definitions
# ============================================================================

$BuildConfigs = @{
    "std" = @{
        Name = "Standard CPU"
        CMakeArgs = @()
        Description = "Baseline build without oneAPI optimizations"
    }
    "tbb" = @{
        Name = "TBB Threading"
        CMakeArgs = @(
            "-DGEMMA_USE_ONEAPI_LIBS=ON",
            "-DGEMMA_USE_TBB=ON"
        )
        Description = "Intel TBB for parallel threading"
    }
    "ipp" = @{
        Name = "IPP Vector Ops"
        CMakeArgs = @(
            "-DGEMMA_USE_ONEAPI_LIBS=ON",
            "-DGEMMA_USE_IPP=ON"
        )
        Description = "Intel IPP for accelerated vector operations"
    }
    "tbb-ipp" = @{
        Name = "TBB + IPP"
        CMakeArgs = @(
            "-DGEMMA_USE_ONEAPI_LIBS=ON",
            "-DGEMMA_USE_TBB=ON",
            "-DGEMMA_USE_IPP=ON"
        )
        Description = "Combined threading and vector optimizations"
    }
    "tbb-ipp-dnnl" = @{
        Name = "Performance Pack"
        CMakeArgs = @(
            "-DGEMMA_USE_ONEAPI_LIBS=ON",
            "-DGEMMA_ONEAPI_PERFORMANCE_PACK=ON"
        )
        Description = "Full oneAPI performance library stack"
    }
    "sycl" = @{
        Name = "SYCL GPU"
        CMakeArgs = @(
            "-DGEMMA_ENABLE_SYCL=ON"
        )
        Description = "SYCL backend for GPU acceleration"
    }
    "sycl-tbb" = @{
        Name = "SYCL + TBB"
        CMakeArgs = @(
            "-DGEMMA_ENABLE_SYCL=ON",
            "-DGEMMA_USE_ONEAPI_LIBS=ON",
            "-DGEMMA_USE_TBB=ON"
        )
        Description = "GPU acceleration with CPU threading optimizations"
    }
}

# ============================================================================
# Helper Functions
# ============================================================================

function Write-SectionHeader {
    param([string]$Title)
    Write-Host "`n╔$('═' * 78)╗" -ForegroundColor Cyan
    Write-Host "║ $Title$(' ' * (77 - $Title.Length))║" -ForegroundColor Cyan
    Write-Host "╚$('═' * 78)╝" -ForegroundColor Cyan
}

function Invoke-BuildConfiguration {
    param(
        [string]$ConfigName,
        [hashtable]$Config,
        [string]$BuildRoot
    )
    
    Write-SectionHeader "Building: $($Config.Name)"
    Write-Host "Description: $($Config.Description)" -ForegroundColor Gray
    Write-Host "CMake Args: $($Config.CMakeArgs -join ' ')" -ForegroundColor Gray
    
    $ConfigBuildDir = Join-Path $BuildRoot "build_$ConfigName"
    
    try {
        # Configure
        Write-Host "`nConfiguring..." -ForegroundColor Yellow
        $CMakeCmd = "cmake -B `"$ConfigBuildDir`" -DCMAKE_BUILD_TYPE=Release $($Config.CMakeArgs -join ' ')"
        Write-Verbose "Running: $CMakeCmd"
        
        $ConfigureResult = Invoke-Expression $CMakeCmd 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "CMake configuration failed with exit code $LASTEXITCODE"
        }
        Write-Host "✅ Configuration successful" -ForegroundColor Green
        
        # Build
        Write-Host "`nBuilding..." -ForegroundColor Yellow
        $BuildCmd = "cmake --build `"$ConfigBuildDir`" --config Release --parallel"
        $BuildStart = Get-Date
        
        $BuildResult = Invoke-Expression $BuildCmd 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "Build failed with exit code $LASTEXITCODE"
        }
        
        $BuildDuration = ((Get-Date) - $BuildStart).TotalSeconds
        Write-Host "✅ Build successful ($([math]::Round($BuildDuration, 1))s)" -ForegroundColor Green
        
        # Find executable
        $Executable = Get-ChildItem -Path "$ConfigBuildDir\bin" -Filter "gemma*.exe" -ErrorAction Stop | Select-Object -First 1
        
        return @{
            Success = $true
            ConfigName = $ConfigName
            DisplayName = $Config.Name
            BuildDir = $ConfigBuildDir
            Executable = $Executable.FullName
            BuildTime = $BuildDuration
            ErrorMessage = $null
        }
    }
    catch {
        Write-Host "❌ Build failed: $_" -ForegroundColor Red
        return @{
            Success = $false
            ConfigName = $ConfigName
            DisplayName = $Config.Name
            BuildDir = $ConfigBuildDir
            Executable = $null
            BuildTime = 0
            ErrorMessage = $_.Exception.Message
        }
    }
}

function Invoke-BenchmarkConfiguration {
    param(
        [hashtable]$BuildResult,
        [string]$ResultsDir
    )
    
    if (-not $BuildResult.Success) {
        Write-Host "⏭️  Skipping benchmark (build failed)" -ForegroundColor Yellow
        return $null
    }
    
    Write-Host "`nRunning benchmarks for $($BuildResult.DisplayName)..." -ForegroundColor Cyan
    
    $BenchmarkScript = Join-Path $PSScriptRoot "benchmark_baseline.ps1"
    if (-not (Test-Path $BenchmarkScript)) {
        Write-Warning "Benchmark script not found: $BenchmarkScript"
        return $null
    }
    
    $OutputFile = Join-Path $ResultsDir "$($BuildResult.ConfigName)_results.json"
    
    try {
        & $BenchmarkScript -Executable $BuildResult.Executable -OutputFile $OutputFile -ErrorAction Stop
        
        if (Test-Path $OutputFile) {
            $Results = Get-Content $OutputFile | ConvertFrom-Json
            Write-Host "✅ Benchmark completed" -ForegroundColor Green
            return $Results
        } else {
            Write-Warning "Benchmark results file not created"
            return $null
        }
    }
    catch {
        Write-Host "❌ Benchmark failed: $_" -ForegroundColor Red
        return $null
    }
}

function Compare-BuildResults {
    param(
        [array]$BuildResults,
        [array]$BenchmarkResults
    )
    
    Write-SectionHeader "Build Comparison Summary"
    
    # Build time comparison
    Write-Host "`n📊 Build Times:" -ForegroundColor Yellow
    $BuildResults | Where-Object { $_.Success } | Sort-Object BuildTime | ForEach-Object {
        $Bar = "█" * [math]::Min(([int]($_.BuildTime / 2)), 50)
        Write-Host ("  {0,-20} {1,6}s {2}" -f $_.DisplayName, [math]::Round($_.BuildTime, 1), $Bar) -ForegroundColor Cyan
    }
    
    # Failed builds
    $Failed = $BuildResults | Where-Object { -not $_.Success }
    if ($Failed.Count -gt 0) {
        Write-Host "`n❌ Failed Builds:" -ForegroundColor Red
        $Failed | ForEach-Object {
            Write-Host "  - $($_.DisplayName): $($_.ErrorMessage)" -ForegroundColor Red
        }
    }
    
    # Benchmark comparison
    if ($BenchmarkResults.Count -gt 0) {
        Write-Host "`n📈 Inference Performance (tokens/sec):" -ForegroundColor Yellow
        
        # Find baseline (std) for percentage calculations
        $Baseline = $BenchmarkResults | Where-Object { $_.ConfigName -eq "std" } | Select-Object -First 1
        
        $Sorted = $BenchmarkResults | Where-Object { $_.Summary.InferenceTestsRun } | 
                  Sort-Object { $_.Summary.AvgTokensPerSec } -Descending
        
        foreach ($Result in $Sorted) {
            $TokPerSec = $Result.Summary.AvgTokensPerSec
            $Bar = "█" * [math]::Min(([int]($TokPerSec / 2)), 50)
            
            $PercentText = ""
            if ($Baseline -and $Result.ConfigName -ne "std") {
                $BaselineTok = $Baseline.Summary.AvgTokensPerSec
                if ($BaselineTok -gt 0) {
                    $Percent = [math]::Round((($TokPerSec - $BaselineTok) / $BaselineTok) * 100, 1)
                    $Sign = if ($Percent -gt 0) { "+" } else { "" }
                    $PercentText = " ($Sign$Percent%)"
                }
            }
            
            $ConfigName = $BuildResults | Where-Object { $_.ConfigName -eq $Result.ConfigName } | Select-Object -First 1 -ExpandProperty DisplayName
            Write-Host ("  {0,-20} {1,6} tok/s {2}{3}" -f $ConfigName, [math]::Round($TokPerSec, 2), $Bar, $PercentText) -ForegroundColor Cyan
        }
    }
    
    # Export detailed comparison
    $ComparisonFile = Join-Path $ResultsDir "comparison_summary.json"
    $ComparisonData = @{
        Timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
        BuildResults = $BuildResults
        BenchmarkResults = $BenchmarkResults
    }
    $ComparisonData | ConvertTo-Json -Depth 10 | Out-File $ComparisonFile -Encoding UTF8
    Write-Host "`n📄 Detailed comparison saved to: $ComparisonFile" -ForegroundColor Green
}

# ============================================================================
# Main Execution
# ============================================================================

Write-SectionHeader "Gemma.cpp Build Comparison Tool"

# Create results directory
if (-not (Test-Path $ResultsDir)) {
    New-Item -ItemType Directory -Path $ResultsDir -Force | Out-Null
}

# Filter configurations
$SelectedConfigs = $BuildConfigs.GetEnumerator() | Where-Object { $Configurations -contains $_.Key }

if ($SelectedConfigs.Count -eq 0) {
    Write-Error "No valid configurations selected. Available: $($BuildConfigs.Keys -join ', ')"
    exit 1
}

Write-Host "Selected configurations: $($SelectedConfigs.Key -join ', ')" -ForegroundColor Cyan
Write-Host "Results directory: $ResultsDir" -ForegroundColor Gray

$BuildResults = @()
$BenchmarkResults = @()

# Clean builds if requested
if ($Clean) {
    Write-Host "`n🧹 Cleaning previous builds..." -ForegroundColor Yellow
    foreach ($Config in $SelectedConfigs) {
        $ConfigBuildDir = Join-Path $BuildDir "build_$($Config.Key)"
        if (Test-Path $ConfigBuildDir) {
            Remove-Item -Path $ConfigBuildDir -Recurse -Force
            Write-Host "  Cleaned $($Config.Key)" -ForegroundColor Gray
        }
    }
}

# Build each configuration
if (-not $SkipBuild) {
    foreach ($Config in $SelectedConfigs) {
        $Result = Invoke-BuildConfiguration -ConfigName $Config.Key -Config $Config.Value -BuildRoot $BuildDir
        $BuildResults += $Result
        
        # Save individual build result
        $BuildResultFile = Join-Path $ResultsDir "$($Config.Key)_build.json"
        $Result | ConvertTo-Json -Depth 10 | Out-File $BuildResultFile -Encoding UTF8
    }
} else {
    Write-Host "`n⏭️  Skipping builds (using existing executables)" -ForegroundColor Yellow
    
    # Load existing build results
    foreach ($Config in $SelectedConfigs) {
        $ConfigBuildDir = Join-Path $BuildDir "build_$($Config.Key)"
        $Executable = Get-ChildItem -Path "$ConfigBuildDir\bin" -Filter "gemma*.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
        
        if ($Executable) {
            $BuildResults += @{
                Success = $true
                ConfigName = $Config.Key
                DisplayName = $Config.Value.Name
                BuildDir = $ConfigBuildDir
                Executable = $Executable.FullName
                BuildTime = 0
                ErrorMessage = $null
            }
        } else {
            Write-Warning "No executable found for $($Config.Key)"
        }
    }
}

# Benchmark each configuration
if (-not $SkipBenchmark) {
    Write-SectionHeader "Running Benchmarks"
    
    foreach ($BuildResult in $BuildResults) {
        $BenchResult = Invoke-BenchmarkConfiguration -BuildResult $BuildResult -ResultsDir $ResultsDir
        if ($BenchResult) {
            $BenchResult | Add-Member -NotePropertyName "ConfigName" -NotePropertyValue $BuildResult.ConfigName
            $BenchmarkResults += $BenchResult
        }
    }
} else {
    Write-Host "`n⏭️  Skipping benchmarks" -ForegroundColor Yellow
}

# Generate comparison report
Compare-BuildResults -BuildResults $BuildResults -BenchmarkResults $BenchmarkResults

Write-Host "`n✅ Build comparison complete!" -ForegroundColor Green
Write-Host "Results saved to: $ResultsDir" -ForegroundColor Cyan
