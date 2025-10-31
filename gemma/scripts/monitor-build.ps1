# monitor-build.ps1
# Real-time build monitoring for gemma.cpp with cache statistics
# Tracks compilation progress, cache hit rate, and estimated completion

[CmdletBinding()]
param(
    [Parameter(Mandatory=$false)]
    [string]$BuildDir = "build",

    [Parameter(Mandatory=$false)]
    [int]$RefreshInterval = 2,

    [Parameter(Mandatory=$false)]
    [switch]$Continuous
)

# ============================================================================
# Global State
# ============================================================================

$script:StartTime = Get-Date
$script:LastStats = $null
$script:PreviousTargetCount = 0
$script:EstimatedTotal = 0

# ============================================================================
# Cache Statistics Functions
# ============================================================================

function Get-SccacheStats {
    try {
        $output = & sccache --show-stats 2>$null
        if ($LASTEXITCODE -eq 0) {
            $stats = @{}

            # Parse key metrics
            foreach ($line in $output -split "`n") {
                if ($line -match "Compile requests\s+(\d+)") {
                    $stats.Requests = [int]$matches[1]
                }
                if ($line -match "Cache hits\s+(\d+)") {
                    $stats.Hits = [int]$matches[1]
                }
                if ($line -match "Cache misses\s+(\d+)") {
                    $stats.Misses = [int]$matches[1]
                }
                if ($line -match "Cache hit rate\s+([\d.]+)") {
                    $stats.HitRate = [double]$matches[1]
                }
            }

            return $stats
        }
    } catch {
        # sccache not available
    }
    return $null
}

function Get-CcacheStats {
    try {
        $output = & ccache -s 2>$null
        if ($LASTEXITCODE -eq 0) {
            $stats = @{}

            # Parse key metrics
            foreach ($line in $output -split "`n") {
                if ($line -match "cache hit.*:\s+(\d+)") {
                    $stats.Hits = [int]$matches[1]
                }
                if ($line -match "cache miss.*:\s+(\d+)") {
                    $stats.Misses = [int]$matches[1]
                }
                if ($line -match "Hits:\s+([\d.]+)\s*%") {
                    $stats.HitRate = [double]$matches[1]
                }
            }

            if ($stats.Hits -or $stats.Misses) {
                $total = $stats.Hits + $stats.Misses
                if ($total -gt 0) {
                    $stats.HitRate = ($stats.Hits / $total) * 100
                }
                $stats.Requests = $total
            }

            return $stats
        }
    } catch {
        # ccache not available
    }
    return $null
}

function Get-CompilerCacheStats {
    # Try sccache first (preferred)
    $stats = Get-SccacheStats
    if ($stats) {
        $stats.Type = "sccache"
        return $stats
    }

    # Fallback to ccache
    $stats = Get-CcacheStats
    if ($stats) {
        $stats.Type = "ccache"
        return $stats
    }

    return $null
}

# ============================================================================
# Build Progress Tracking
# ============================================================================

function Get-BuildProgress {
    param([string]$BuildDir)

    $progress = @{
        Targets = 0
        Completed = 0
        Failed = 0
        InProgress = 0
    }

    # Check for Ninja build system
    $ninjaBuild = Join-Path $BuildDir ".ninja_log"
    if (Test-Path $ninjaBuild) {
        try {
            $log = Get-Content $ninjaBuild -ErrorAction SilentlyContinue
            $progress.Completed = ($log | Where-Object { $_ -match "^\d+\s+\d+\s+\d+\s+" }).Count
        } catch {
            # Ninja log unavailable
        }
    }

    # Check for MSBuild logs (Visual Studio)
    $msbuildLogs = Get-ChildItem -Path $BuildDir -Filter "*.log" -Recurse -ErrorAction SilentlyContinue
    if ($msbuildLogs) {
        foreach ($log in $msbuildLogs) {
            $content = Get-Content $log.FullName -ErrorAction SilentlyContinue
            $progress.Completed += ($content | Select-String -Pattern "Compiling|Linking" -AllMatches).Matches.Count
        }
    }

    return $progress
}

function Get-EstimatedCompletion {
    param(
        [int]$Completed,
        [int]$Total,
        [datetime]$StartTime
    )

    if ($Completed -eq 0 -or $Total -eq 0) {
        return "Calculating..."
    }

    $elapsed = (Get-Date) - $StartTime
    $rate = $Completed / $elapsed.TotalSeconds

    if ($rate -eq 0) {
        return "Calculating..."
    }

    $remaining = $Total - $Completed
    $eta = [TimeSpan]::FromSeconds($remaining / $rate)

    return "{0:mm}m {0:ss}s" -f $eta
}

# ============================================================================
# Display Functions
# ============================================================================

function Write-BuildHeader {
    Clear-Host
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  Gemma.cpp Build Monitor" -ForegroundColor Cyan
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Build Directory: $BuildDir" -ForegroundColor Gray
    Write-Host "  Started: $($script:StartTime.ToString('HH:mm:ss'))" -ForegroundColor Gray
    Write-Host "  Elapsed: $((Get-Date) - $script:StartTime | ForEach-Object { '{0:mm}:{0:ss}' -f $_ })" -ForegroundColor Gray
    Write-Host ""
}

function Write-CacheStatistics {
    param($Stats)

    if (-not $Stats) {
        Write-Host "─────────────────────────────────────────────────────────────────" -ForegroundColor Gray
        Write-Host "Compiler Cache: Not Available" -ForegroundColor Yellow
        Write-Host ""
        return
    }

    Write-Host "─────────────────────────────────────────────────────────────────" -ForegroundColor Gray
    Write-Host "Compiler Cache ($($Stats.Type)):" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Total Requests:  $($Stats.Requests)" -ForegroundColor White
    Write-Host "  Cache Hits:      $($Stats.Hits)" -ForegroundColor Green
    Write-Host "  Cache Misses:    $($Stats.Misses)" -ForegroundColor Yellow

    # Color code hit rate
    $hitRate = "{0:F1}%" -f $Stats.HitRate
    $hitRateColor = "Red"
    if ($Stats.HitRate -ge 80) { $hitRateColor = "Green" }
    elseif ($Stats.HitRate -ge 50) { $hitRateColor = "Yellow" }

    Write-Host "  Hit Rate:        $hitRate" -ForegroundColor $hitRateColor

    # Calculate incremental stats since last check
    if ($script:LastStats) {
        $newRequests = $Stats.Requests - $script:LastStats.Requests
        $newHits = $Stats.Hits - $script:LastStats.Hits
        $newMisses = $Stats.Misses - $script:LastStats.Misses

        if ($newRequests -gt 0) {
            $incrementalHitRate = ($newHits / $newRequests) * 100
            Write-Host ""
            Write-Host "  Recent Activity:" -ForegroundColor DarkCyan
            Write-Host "    New Requests:  $newRequests" -ForegroundColor Gray
            Write-Host "    Recent Hits:   $newHits" -ForegroundColor Gray
            Write-Host "    Recent Rate:   $("{0:F1}%" -f $incrementalHitRate)" -ForegroundColor Gray
        }
    }

    Write-Host ""
}

function Write-BuildProgress {
    param($Progress, $Estimated)

    Write-Host "─────────────────────────────────────────────────────────────────" -ForegroundColor Gray
    Write-Host "Build Progress:" -ForegroundColor Cyan
    Write-Host ""

    if ($Progress.Completed -eq 0) {
        Write-Host "  Initializing build..." -ForegroundColor Yellow
        Write-Host ""
        return
    }

    Write-Host "  Completed Targets: $($Progress.Completed)" -ForegroundColor Green

    if ($script:EstimatedTotal -gt 0) {
        $percentage = ($Progress.Completed / $script:EstimatedTotal) * 100
        Write-Host "  Progress:          $("{0:F1}%" -f $percentage)" -ForegroundColor White
        Write-Host "  Estimated Total:   $script:EstimatedTotal targets" -ForegroundColor Gray
        Write-Host "  ETA:               $Estimated" -ForegroundColor Yellow

        # Progress bar
        $barWidth = 50
        $filled = [math]::Floor(($percentage / 100) * $barWidth)
        $empty = $barWidth - $filled
        $bar = "[" + ("█" * $filled) + ("░" * $empty) + "]"
        Write-Host ""
        Write-Host "  $bar" -ForegroundColor Green
    } else {
        Write-Host "  (Estimating total targets...)" -ForegroundColor DarkGray
    }

    Write-Host ""
}

function Write-BuildTips {
    Write-Host "─────────────────────────────────────────────────────────────────" -ForegroundColor Gray
    Write-Host "Tips:" -ForegroundColor Cyan
    Write-Host "  • First build: Cache warm-up (slower)" -ForegroundColor Gray
    Write-Host "  • Second build: 4-5x faster with cache hits" -ForegroundColor Gray
    Write-Host "  • Incremental: 5-10x faster after changes" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Press Ctrl+C to exit monitor" -ForegroundColor DarkGray
    Write-Host ""
}

# ============================================================================
# Main Monitoring Loop
# ============================================================================

function Start-BuildMonitoring {
    Write-Host "Starting build monitor..." -ForegroundColor Green
    Write-Host "Refresh interval: $RefreshInterval seconds" -ForegroundColor Gray
    Write-Host ""
    Start-Sleep -Seconds 1

    # Estimate total targets on first run
    $script:EstimatedTotal = 250  # Rough estimate for gemma.cpp

    do {
        # Collect statistics
        $cacheStats = Get-CompilerCacheStats
        $buildProgress = Get-BuildProgress -BuildDir $BuildDir
        $eta = Get-EstimatedCompletion -Completed $buildProgress.Completed -Total $script:EstimatedTotal -StartTime $script:StartTime

        # Update estimated total if we have better info
        if ($buildProgress.Completed -gt $script:EstimatedTotal) {
            $script:EstimatedTotal = $buildProgress.Completed + 50
        }

        # Display dashboard
        Write-BuildHeader
        Write-CacheStatistics -Stats $cacheStats
        Write-BuildProgress -Progress $buildProgress -Estimated $eta
        Write-BuildTips

        # Store stats for next iteration
        $script:LastStats = $cacheStats
        $script:PreviousTargetCount = $buildProgress.Completed

        # Wait before next refresh
        Start-Sleep -Seconds $RefreshInterval

    } while ($Continuous -or (Test-Path (Join-Path $BuildDir "CMakeFiles")))

    Write-Host "Build monitoring complete." -ForegroundColor Green
}

# ============================================================================
# Entry Point
# ============================================================================

if (-not (Test-Path $BuildDir)) {
    Write-Host "Error: Build directory not found: $BuildDir" -ForegroundColor Red
    Write-Host "Run cmake configure first." -ForegroundColor Yellow
    exit 1
}

Start-BuildMonitoring
