# build-accelerated.ps1
# One-command build script for gemma.cpp with full acceleration
# Handles environment setup, configuration, building, and monitoring

[CmdletBinding()]
param(
    [ValidateSet("Release", "FastDebug", "Debug", "RelWithSymbols")]
    [string]$Config = "Release",

    [ValidateRange(1, 24)]
    [int]$Parallel = 12,

    [ValidateSet("ninja-accelerated", "ninja-accelerated-release", "ninja-accelerated-fast-debug")]
    [string]$Preset = "ninja-accelerated",

    [switch]$Clean,
    [switch]$Monitor,
    [switch]$SkipTests,
    [switch]$Verbose,
    [switch]$Persistent
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# ============================================================================
# Banner
# ============================================================================

function Write-Banner {
    Write-Host ""
    Write-Host "╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║           Gemma.cpp Accelerated Build System                ║" -ForegroundColor Cyan
    Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Configuration: $Config" -ForegroundColor White
    Write-Host "  Preset:        $Preset" -ForegroundColor White
    Write-Host "  Parallel:      $Parallel jobs" -ForegroundColor White
    Write-Host "  Monitor:       $(if($Monitor){'Enabled'}else{'Disabled'})" -ForegroundColor White
    Write-Host ""
}

# ============================================================================
# Environment Setup
# ============================================================================

function Invoke-EnvironmentSetup {
    Write-Host "=== Step 1: Environment Setup ===" -ForegroundColor Yellow
    Write-Host ""

    $setupScript = Join-Path $PSScriptRoot "scripts\setup-build-env.ps1"
    if (-not (Test-Path $setupScript)) {
        Write-Host "✗ Error: setup-build-env.ps1 not found" -ForegroundColor Red
        exit 1
    }

    & $setupScript -CacheSize "10G" -Persistent:$Persistent

    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Environment setup failed" -ForegroundColor Red
        exit 1
    }

    Write-Host "✓ Environment configured successfully" -ForegroundColor Green
    Write-Host ""
}

# ============================================================================
# Clean Build Directory
# ============================================================================

function Invoke-CleanBuild {
    param([string]$BuildDir)

    Write-Host "=== Cleaning Build Directory ===" -ForegroundColor Yellow
    Write-Host ""

    if (Test-Path $BuildDir) {
        Write-Host "Removing: $BuildDir" -ForegroundColor Gray
        Remove-Item -Path $BuildDir -Recurse -Force -ErrorAction SilentlyContinue
    }

    Write-Host "✓ Clean complete" -ForegroundColor Green
    Write-Host ""
}

# ============================================================================
# CMake Configuration
# ============================================================================

function Invoke-CMakeConfigure {
    param([string]$Preset)

    Write-Host "=== Step 2: CMake Configuration ===" -ForegroundColor Yellow
    Write-Host ""

    $configureArgs = @(
        "--preset", $Preset
    )

    if ($Verbose) {
        $configureArgs += @("--log-level=DEBUG")
    }

    Write-Host "Running: cmake $($configureArgs -join ' ')" -ForegroundColor Gray
    Write-Host ""

    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

    & cmake @configureArgs

    $stopwatch.Stop()

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "✗ CMake configuration failed" -ForegroundColor Red
        Write-Host ""
        Write-Host "Troubleshooting tips:" -ForegroundColor Yellow
        Write-Host "  • Check CMake version: cmake --version (need 3.20+)" -ForegroundColor Gray
        Write-Host "  • Verify Ninja is installed: ninja --version" -ForegroundColor Gray
        Write-Host "  • Check vcpkg: echo `$env:VCPKG_ROOT" -ForegroundColor Gray
        Write-Host "  • Try: .\build-accelerated.ps1 -Clean" -ForegroundColor Gray
        exit 1
    }

    Write-Host ""
    Write-Host "✓ Configuration complete ($('{0:F1}s' -f $stopwatch.Elapsed.TotalSeconds))" -ForegroundColor Green
    Write-Host ""
}

# ============================================================================
# Build Monitoring (Background)
# ============================================================================

function Start-BuildMonitoring {
    param([string]$BuildDir)

    if (-not $Monitor) {
        return $null
    }

    $monitorScript = Join-Path $PSScriptRoot "scripts\monitor-build.ps1"
    if (-not (Test-Path $monitorScript)) {
        Write-Host "⚠ Warning: monitor-build.ps1 not found, skipping monitoring" -ForegroundColor Yellow
        return $null
    }

    Write-Host "Starting build monitor..." -ForegroundColor Cyan

    $job = Start-Job -ScriptBlock {
        param($Script, $BuildDir)
        & $Script -BuildDir $BuildDir -Continuous
    } -ArgumentList $monitorScript, $BuildDir

    return $job
}

function Stop-BuildMonitoring {
    param($MonitorJob)

    if ($MonitorJob) {
        Write-Host ""
        Write-Host "Stopping build monitor..." -ForegroundColor Cyan
        Stop-Job -Job $MonitorJob -ErrorAction SilentlyContinue
        Remove-Job -Job $MonitorJob -Force -ErrorAction SilentlyContinue
    }
}

# ============================================================================
# CMake Build
# ============================================================================

function Invoke-CMakeBuild {
    param(
        [string]$Preset,
        [string]$Config,
        [int]$Jobs
    )

    Write-Host "=== Step 3: Build ===" -ForegroundColor Yellow
    Write-Host ""

    $buildArgs = @(
        "--build", "build-ninja",
        "--config", $Config,
        "-j", $Jobs
    )

    if ($Verbose) {
        $buildArgs += @("--verbose")
    }

    Write-Host "Running: cmake $($buildArgs -join ' ')" -ForegroundColor Gray
    Write-Host ""
    Write-Host "⏳ Building gemma.cpp (this may take 12-15 minutes on first build)..." -ForegroundColor Cyan
    Write-Host ""

    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

    & cmake @buildArgs

    $stopwatch.Stop()

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "✗ Build failed" -ForegroundColor Red
        Write-Host ""
        Write-Host "Check build logs for errors." -ForegroundColor Yellow
        Write-Host "Common issues:" -ForegroundColor Yellow
        Write-Host "  • Missing dependencies (Highway, SentencePiece)" -ForegroundColor Gray
        Write-Host "  • Compiler version too old (need C++20)" -ForegroundColor Gray
        Write-Host "  • Out of memory (close other applications)" -ForegroundColor Gray
        return $false
    }

    Write-Host ""
    Write-Host "✓ Build complete ($('{0:F1}s' -f $stopwatch.Elapsed.TotalSeconds))" -ForegroundColor Green
    Write-Host ""

    # Display build time breakdown
    $minutes = [math]::Floor($stopwatch.Elapsed.TotalMinutes)
    $seconds = $stopwatch.Elapsed.Seconds

    Write-Host "Build Time: ${minutes}m ${seconds}s" -ForegroundColor White
    Write-Host ""

    return $true
}

# ============================================================================
# Cache Statistics
# ============================================================================

function Show-CacheStatistics {
    Write-Host "=== Step 4: Cache Statistics ===" -ForegroundColor Yellow
    Write-Host ""

    # Try sccache first
    $sccache = Get-Command sccache -ErrorAction SilentlyContinue
    if ($sccache) {
        Write-Host "sccache Statistics:" -ForegroundColor Cyan
        & sccache --show-stats 2>$null
        Write-Host ""
        return
    }

    # Fallback to ccache
    $ccache = Get-Command ccache -ErrorAction SilentlyContinue
    if ($ccache) {
        Write-Host "ccache Statistics:" -ForegroundColor Cyan
        & ccache -s 2>$null
        Write-Host ""
        return
    }

    Write-Host "No compiler cache available" -ForegroundColor DarkGray
    Write-Host ""
}

# ============================================================================
# Run Tests
# ============================================================================

function Invoke-Tests {
    param([string]$BuildDir, [string]$Config)

    if ($SkipTests) {
        Write-Host "Skipping tests (SkipTests parameter specified)" -ForegroundColor DarkGray
        Write-Host ""
        return $true
    }

    Write-Host "=== Step 5: Running Tests ===" -ForegroundColor Yellow
    Write-Host ""

    Push-Location $BuildDir
    try {
        $testArgs = @(
            "-C", $Config,
            "-j", $Parallel,
            "--output-on-failure"
        )

        if ($Verbose) {
            $testArgs += @("-V")
        }

        Write-Host "Running: ctest $($testArgs -join ' ')" -ForegroundColor Gray
        Write-Host ""

        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

        & ctest @testArgs

        $stopwatch.Stop()

        if ($LASTEXITCODE -ne 0) {
            Write-Host ""
            Write-Host "⚠ Some tests failed (see output above)" -ForegroundColor Yellow
            Write-Host ""
            return $false
        }

        Write-Host ""
        Write-Host "✓ All tests passed ($('{0:F1}s' -f $stopwatch.Elapsed.TotalSeconds))" -ForegroundColor Green
        Write-Host ""
        return $true

    } finally {
        Pop-Location
    }
}

# ============================================================================
# Summary
# ============================================================================

function Show-Summary {
    param(
        [timespan]$TotalTime,
        [bool]$BuildSuccess,
        [bool]$TestsSuccess
    )

    Write-Host ""
    Write-Host "╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "║                    Build Summary                            ║" -ForegroundColor Green
    Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Build:        $(if($BuildSuccess){'✓ Success'}else{'✗ Failed'})" -ForegroundColor $(if($BuildSuccess){'Green'}else{'Red'})
    Write-Host "  Tests:        $(if($TestsSuccess){'✓ Passed'}elseif($SkipTests){'- Skipped'}else{'✗ Failed'})" -ForegroundColor $(if($TestsSuccess){'Green'}elseif($SkipTests){'Gray'}else{'Red'})
    Write-Host "  Total Time:   $('{0:mm}m {0:ss}s' -f $TotalTime)" -ForegroundColor White
    Write-Host ""

    if ($BuildSuccess) {
        Write-Host "Executables:" -ForegroundColor Cyan
        Write-Host "  gemma.exe:    .\build-ninja\$Config\gemma.exe" -ForegroundColor Gray
        Write-Host "  benchmark:    .\build-ninja\$Config\single_benchmark.exe" -ForegroundColor Gray
        Write-Host ""
    }

    Write-Host "Next Steps:" -ForegroundColor Cyan
    if ($BuildSuccess) {
        Write-Host "  - Run inference: .\build-ninja\$Config\gemma.exe --help" -ForegroundColor White
        Write-Host "  - Benchmark: .\build-ninja\$Config\single_benchmark.exe --weights MODEL_PATH" -ForegroundColor White
        if (-not $TestsSuccess -and -not $SkipTests) {
            Write-Host "  - Fix test failures: ctest --rerun-failed -C $Config -V" -ForegroundColor White
        }
    } else {
        Write-Host "  - Check build logs for errors" -ForegroundColor White
        Write-Host "  - Try: .\build-accelerated.ps1 -Clean -Verbose" -ForegroundColor White
    }
    Write-Host ""
}

# ============================================================================
# Main Execution
# ============================================================================

$totalStopwatch = [System.Diagnostics.Stopwatch]::StartNew()

try {
    Write-Banner

    # Step 1: Environment setup
    Invoke-EnvironmentSetup

    # Step 1.5: Clean if requested
    if ($Clean) {
        Invoke-CleanBuild -BuildDir "build-ninja"
    }

    # Step 2: Configure
    Invoke-CMakeConfigure -Preset $Preset

    # Step 3: Start monitoring (optional)
    $monitorJob = Start-BuildMonitoring -BuildDir "build-ninja"

    # Step 4: Build
    $buildSuccess = Invoke-CMakeBuild -Preset $Preset -Config $Config -Jobs $Parallel

    # Step 5: Stop monitoring
    Stop-BuildMonitoring -MonitorJob $monitorJob

    # Step 6: Show cache statistics
    Show-CacheStatistics

    # Step 7: Run tests
    $testsSuccess = $false
    if ($buildSuccess) {
        $testsSuccess = Invoke-Tests -BuildDir "build-ninja" -Config $Config
    }

    # Step 8: Summary
    $totalStopwatch.Stop()
    Show-Summary -TotalTime $totalStopwatch.Elapsed -BuildSuccess $buildSuccess -TestsSuccess $testsSuccess

    # Exit with appropriate code
    if ($buildSuccess -and ($testsSuccess -or $SkipTests)) {
        exit 0
    } else {
        exit 1
    }

} catch {
    Write-Host ""
    Write-Host "Fatal Error: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Stack Trace:" -ForegroundColor DarkGray
    Write-Host $_.ScriptStackTrace -ForegroundColor DarkGray
    Write-Host ""

    # Stop monitoring if running
    Stop-BuildMonitoring -MonitorJob $monitorJob

    exit 1
}
