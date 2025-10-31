# setup-build-env.ps1
# Configure environment variables for accelerated gemma.cpp builds
# Run this before building to ensure optimal compiler cache and build settings

[CmdletBinding()]
param(
    [string]$CacheDir = "C:\codedev\build-cache",
    [string]$CacheSize = "10G",
    [ValidateSet("sccache", "ccache", "auto")]
    [string]$CacheType = "auto",
    [switch]$Persistent
)

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Gemma.cpp Build Environment Setup" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# Detect Compiler Cache
# ============================================================================

function Find-CompilerCache {
    param([string]$PreferredType)

    $sccache = Get-Command sccache -ErrorAction SilentlyContinue
    $ccache = Get-Command ccache -ErrorAction SilentlyContinue

    if ($PreferredType -eq "sccache" -and $sccache) {
        return @{Type = "sccache"; Path = $sccache.Source}
    } elseif ($PreferredType -eq "ccache" -and $ccache) {
        return @{Type = "ccache"; Path = $ccache.Source}
    } elseif ($PreferredType -eq "auto") {
        if ($sccache) {
            return @{Type = "sccache"; Path = $sccache.Source}
        } elseif ($ccache) {
            return @{Type = "ccache"; Path = $ccache.Source}
        }
    }

    return $null
}

$cache = Find-CompilerCache -PreferredType $CacheType

if (-not $cache) {
    Write-Host "⚠ WARNING: No compiler cache found (sccache/ccache)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Install options:" -ForegroundColor Yellow
    Write-Host "  sccache: cargo install sccache" -ForegroundColor Gray
    Write-Host "  ccache:  choco install ccache" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Continuing without compiler cache (builds will be slower)..." -ForegroundColor Yellow
    Start-Sleep -Seconds 2
} else {
    Write-Host "✓ Compiler Cache: $($cache.Type)" -ForegroundColor Green
    Write-Host "  Location: $($cache.Path)" -ForegroundColor Gray
}

# ============================================================================
# Configure Compiler Cache Environment Variables
# ============================================================================

# Create cache directory if it doesn't exist
$cacheDir = Join-Path $CacheDir $cache.Type
if (-not (Test-Path $cacheDir)) {
    New-Item -ItemType Directory -Path $cacheDir -Force | Out-Null
    Write-Host "✓ Created cache directory: $cacheDir" -ForegroundColor Green
}

# Set environment variables based on cache type
if ($cache.Type -eq "sccache") {
    $env:SCCACHE_DIR = $cacheDir
    $env:SCCACHE_CACHE_SIZE = $CacheSize
    $env:SCCACHE_LOG = "warn"  # Less verbose logging
    $env:CMAKE_C_COMPILER_LAUNCHER = "sccache"
    $env:CMAKE_CXX_COMPILER_LAUNCHER = "sccache"

    Write-Host "✓ Environment Variables:" -ForegroundColor Green
    Write-Host "  SCCACHE_DIR = $env:SCCACHE_DIR" -ForegroundColor Gray
    Write-Host "  SCCACHE_CACHE_SIZE = $env:SCCACHE_CACHE_SIZE" -ForegroundColor Gray
    Write-Host "  CMAKE_C_COMPILER_LAUNCHER = sccache" -ForegroundColor Gray
    Write-Host "  CMAKE_CXX_COMPILER_LAUNCHER = sccache" -ForegroundColor Gray

    # Make persistent if requested
    if ($Persistent) {
        [System.Environment]::SetEnvironmentVariable("SCCACHE_DIR", $cacheDir, "User")
        [System.Environment]::SetEnvironmentVariable("SCCACHE_CACHE_SIZE", $CacheSize, "User")
        [System.Environment]::SetEnvironmentVariable("CMAKE_C_COMPILER_LAUNCHER", "sccache", "User")
        [System.Environment]::SetEnvironmentVariable("CMAKE_CXX_COMPILER_LAUNCHER", "sccache", "User")
        Write-Host "  ✓ Made persistent (User environment variables)" -ForegroundColor Green
    }

} elseif ($cache.Type -eq "ccache") {
    $env:CCACHE_DIR = $cacheDir
    $env:CCACHE_MAXSIZE = $CacheSize
    $env:CCACHE_COMPRESS = "true"
    $env:CCACHE_COMPRESSLEVEL = "6"
    $env:CMAKE_C_COMPILER_LAUNCHER = "ccache"
    $env:CMAKE_CXX_COMPILER_LAUNCHER = "ccache"

    Write-Host "✓ Environment Variables:" -ForegroundColor Green
    Write-Host "  CCACHE_DIR = $env:CCACHE_DIR" -ForegroundColor Gray
    Write-Host "  CCACHE_MAXSIZE = $env:CCACHE_MAXSIZE" -ForegroundColor Gray
    Write-Host "  CCACHE_COMPRESS = true" -ForegroundColor Gray
    Write-Host "  CMAKE_C_COMPILER_LAUNCHER = ccache" -ForegroundColor Gray
    Write-Host "  CMAKE_CXX_COMPILER_LAUNCHER = ccache" -ForegroundColor Gray

    # Make persistent if requested
    if ($Persistent) {
        [System.Environment]::SetEnvironmentVariable("CCACHE_DIR", $cacheDir, "User")
        [System.Environment]::SetEnvironmentVariable("CCACHE_MAXSIZE", $CacheSize, "User")
        [System.Environment]::SetEnvironmentVariable("CCACHE_COMPRESS", "true", "User")
        [System.Environment]::SetEnvironmentVariable("CMAKE_C_COMPILER_LAUNCHER", "ccache", "User")
        [System.Environment]::SetEnvironmentVariable("CMAKE_CXX_COMPILER_LAUNCHER", "ccache", "User")
        Write-Host "  ✓ Made persistent (User environment variables)" -ForegroundColor Green
    }
}

# ============================================================================
# Configure CMake Generator Preference
# ============================================================================

# Prefer Ninja for fastest builds
$ninja = Get-Command ninja -ErrorAction SilentlyContinue
if ($ninja) {
    $env:CMAKE_GENERATOR = "Ninja Multi-Config"
    Write-Host "✓ CMake Generator: Ninja Multi-Config (fast)" -ForegroundColor Green
    if ($Persistent) {
        [System.Environment]::SetEnvironmentVariable("CMAKE_GENERATOR", "Ninja Multi-Config", "User")
    }
} else {
    Write-Host "  CMake Generator: Visual Studio (Ninja not found)" -ForegroundColor Gray
    Write-Host "  Install Ninja for faster builds: choco install ninja" -ForegroundColor Yellow
}

# ============================================================================
# Configure vcpkg Integration
# ============================================================================

$vcpkgRoot = $env:VCPKG_ROOT
if ($vcpkgRoot -and (Test-Path $vcpkgRoot)) {
    Write-Host "✓ vcpkg: $vcpkgRoot" -ForegroundColor Green

    # Set triplet for optimized builds
    $env:VCPKG_DEFAULT_TRIPLET = "x64-windows-release"
    Write-Host "  VCPKG_DEFAULT_TRIPLET = x64-windows-release" -ForegroundColor Gray

    if ($Persistent) {
        [System.Environment]::SetEnvironmentVariable("VCPKG_DEFAULT_TRIPLET", "x64-windows-release", "User")
    }
} else {
    Write-Host "  vcpkg: Not configured (VCPKG_ROOT not set)" -ForegroundColor Gray
}

# ============================================================================
# Display Cache Statistics (if available)
# ============================================================================

Write-Host ""
Write-Host "─────────────────────────────────────────────────────────────────" -ForegroundColor Gray

if ($cache.Type -eq "sccache") {
    Write-Host "Current sccache Statistics:" -ForegroundColor Cyan
    try {
        & sccache --show-stats 2>$null | Write-Host -ForegroundColor Gray
    } catch {
        Write-Host "  (sccache stats unavailable)" -ForegroundColor DarkGray
    }
} elseif ($cache.Type -eq "ccache") {
    Write-Host "Current ccache Statistics:" -ForegroundColor Cyan
    try {
        & ccache -s 2>$null | Write-Host -ForegroundColor Gray
    } catch {
        Write-Host "  (ccache stats unavailable)" -ForegroundColor DarkGray
    }
}

# ============================================================================
# Additional Optimizations
# ============================================================================

Write-Host ""
Write-Host "─────────────────────────────────────────────────────────────────" -ForegroundColor Gray
Write-Host "Additional Optimizations:" -ForegroundColor Cyan

# Increase file handle limits for parallel builds
$env:_CL_ = "/MP12"
Write-Host "  ✓ MSVC multi-processor compilation: /MP12" -ForegroundColor Green

# Disable CMake file API for faster configuration
$env:CMAKE_SUPPRESS_REGENERATION = "1"
Write-Host "  ✓ CMake file API suppressed (faster configure)" -ForegroundColor Green

# Enable colored diagnostics for better output
$env:CLICOLOR_FORCE = "1"
Write-Host "  ✓ Colored compiler diagnostics enabled" -ForegroundColor Green

# ============================================================================
# Success Summary
# ============================================================================

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host "  Build Environment Configured Successfully!" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""

if (-not $Persistent) {
    Write-Host "NOTE: These settings are temporary for this session." -ForegroundColor Yellow
    Write-Host "      Run with -Persistent to make them permanent:" -ForegroundColor Yellow
    Write-Host "      .\setup-build-env.ps1 -Persistent" -ForegroundColor Gray
    Write-Host ""
}

Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Run: .\build-accelerated.ps1" -ForegroundColor White
Write-Host "  2. Or: cmake --preset ninja-accelerated" -ForegroundColor White
Write-Host "  3. Or: cmake -B build && cmake --build build -j 12" -ForegroundColor White
Write-Host ""

# ============================================================================
# Export Functions for Build Scripts
# ============================================================================

# Export cache info for use in other scripts
$global:GemmaCacheType = $cache.Type
$global:GemmaCacheDir = $cacheDir

Write-Host "Build environment ready. Happy building! 🚀" -ForegroundColor Green
Write-Host ""
