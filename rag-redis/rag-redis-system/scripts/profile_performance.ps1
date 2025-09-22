# RAG-Redis Performance Profiling Script (PowerShell)
# This script uses cargo-flamegraph to identify hotspots and performance bottlenecks

param(
    [switch]$Comprehensive,
    [string]$OutputDir = "performance_profiles"
)

Write-Host "=== RAG-Redis Performance Profiling ===" -ForegroundColor Cyan
Write-Host "This script will profile the RAG-Redis system using flame graphs"
Write-Host ""

# Check prerequisites
function Test-Prerequisites {
    Write-Host "Checking prerequisites..." -ForegroundColor Yellow

    # Check if cargo-flamegraph is installed
    if (-not (Get-Command cargo-flamegraph -ErrorAction SilentlyContinue)) {
        Write-Host "cargo-flamegraph not found. Installing..." -ForegroundColor Red
        cargo install flamegraph
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Failed to install cargo-flamegraph" -ForegroundColor Red
            exit 1
        }
    }

    # Check if cargo is available
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
        Write-Host "Rust/Cargo not found. Please install Rust toolchain." -ForegroundColor Red
        exit 1
    }

    Write-Host "Prerequisites satisfied" -ForegroundColor Green
}

# Create output directory
function New-OutputDirectory {
    if (-not (Test-Path $OutputDir)) {
        New-Item -ItemType Directory -Path $OutputDir | Out-Null
    }
    Write-Host "Output directory: $OutputDir"
}

# Profile baseline vector operations
function Invoke-VectorOperationsProfile {
    Write-Host "Profiling vector operations (baseline)..." -ForegroundColor Yellow

    try {
        $outputFile = Join-Path $OutputDir "vector_operations_baseline.svg"
        cargo flamegraph --test performance_test --output=$outputFile -- --exact performance_test::tests::test_baseline_performance_medium

        if ($LASTEXITCODE -eq 0) {
            Write-Host "Vector operations profile saved to $outputFile" -ForegroundColor Green
            return $true
        } else {
            Write-Host "Vector operations profiling failed" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "Error during vector operations profiling: $_" -ForegroundColor Red
        return $false
    }
}

# Profile SIMD optimizations
function Invoke-SIMDOperationsProfile {
    Write-Host "Profiling SIMD distance calculations..." -ForegroundColor Yellow

    try {
        $outputFile = Join-Path $OutputDir "simd_operations.svg"
        cargo flamegraph --test performance_test --output=$outputFile -- --exact performance_test::tests::test_distance_metric_comparison

        if ($LASTEXITCODE -eq 0) {
            Write-Host "SIMD operations profile saved to $outputFile" -ForegroundColor Green
            return $true
        } else {
            Write-Host "SIMD operations profiling failed" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "Error during SIMD operations profiling: $_" -ForegroundColor Red
        return $false
    }
}

# Main execution
function Main {
    Write-Host "Starting performance profiling for RAG-Redis system..."

    Test-Prerequisites
    New-OutputDirectory

    Write-Host "Running profiling tests..."

    $results = @{}

    # Profile individual components
    $results["vector"] = Invoke-VectorOperationsProfile
    if ($results["vector"]) {
        Write-Host "✓ Vector operations profiled" -ForegroundColor Green
    } else {
        Write-Host "✗ Vector operations profiling failed" -ForegroundColor Red
    }

    $results["simd"] = Invoke-SIMDOperationsProfile
    if ($results["simd"]) {
        Write-Host "✓ SIMD operations profiled" -ForegroundColor Green
    } else {
        Write-Host "✗ SIMD operations profiling failed" -ForegroundColor Red
    }

    Write-Host ""
    Write-Host "=== Performance Profiling Complete ===" -ForegroundColor Green
    Write-Host "Flame graphs generated in: $OutputDir\"
}

# Execute main function
Main
