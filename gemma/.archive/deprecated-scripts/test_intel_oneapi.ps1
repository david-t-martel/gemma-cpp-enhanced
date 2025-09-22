<#
.SYNOPSIS
    Tests Intel OneAPI installation for Gemma.cpp compatibility.

.DESCRIPTION
    This script verifies that Intel OneAPI components required for building
    Gemma.cpp are properly installed and accessible.

.PARAMETER Detailed
    Show detailed information about each component

.PARAMETER ComponentPath
    Custom path to Intel OneAPI installation directory

.PARAMETER ExportReport
    Export test results to a JSON file

.PARAMETER Quiet
    Suppress non-essential output, only show results

.EXAMPLE
    .\test_intel_oneapi.ps1
    Performs basic Intel OneAPI installation test

.EXAMPLE
    .\test_intel_oneapi.ps1 -Detailed -ExportReport
    Performs detailed test and exports results to JSON

.NOTES
    Author: Converted from batch script
    Requires: Intel OneAPI Base Toolkit (optional for detection)
#>

[CmdletBinding()]
param(
    [switch]$Detailed,
    [string]$ComponentPath = "C:\Program Files (x86)\Intel\oneAPI",
    [switch]$ExportReport,
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"

function Write-ColoredOutput {
    param([string]$Message, [ConsoleColor]$Color = [ConsoleColor]::White)
    if (-not $Quiet) {
        $originalColor = $Host.UI.RawUI.ForegroundColor
        $Host.UI.RawUI.ForegroundColor = $Color
        Write-Host $Message
        $Host.UI.RawUI.ForegroundColor = $originalColor
    }
}

function Test-ComponentPath {
    param([string]$Path, [string]$ComponentName)
    $exists = Test-Path $Path
    $status = if ($exists) { "[OK]" } else { "[MISSING]" }
    $color = if ($exists) { [ConsoleColor]::Green } else { [ConsoleColor]::Red }
    Write-ColoredOutput "$status $ComponentName" -Color $color
    if ($Detailed -and $exists) {
        $item = Get-Item $Path -ErrorAction SilentlyContinue
        if ($item) {
            Write-ColoredOutput "      Path: $Path" -Color Gray
            if ($item.PSIsContainer) {
                $children = Get-ChildItem $Path -ErrorAction SilentlyContinue | Measure-Object
                Write-ColoredOutput "      Contains: $($children.Count) items" -Color Gray
            } else {
                Write-ColoredOutput "      Size: $([math]::Round($item.Length/1MB, 2)) MB" -Color Gray
                Write-ColoredOutput "      Modified: $($item.LastWriteTime)" -Color Gray
            }
        }
    }
    return @{ Name = $ComponentName; Path = $Path; Exists = $exists }
}

function Test-EnvironmentSetup {
    param([string]$OneAPIPath)
    $setvarsPath = Join-Path $OneAPIPath "setvars.bat"
    if (-not (Test-Path $setvarsPath)) {
        return @{ Success = $false; Message = "setvars.bat not found" }
    }

    try {
        $tempBat = [System.IO.Path]::GetTempFileName() + ".bat"
        $tempOut = [System.IO.Path]::GetTempFileName() + ".out"
        $batchContent = "@echo off`ncall `"$setvarsPath`" intel64 vs2022 >nul 2>&1`necho ONEAPI_SUCCESS > `"$tempOut`""
        Set-Content -Path $tempBat -Value $batchContent
        $result = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", $tempBat -Wait -PassThru
        $success = (Test-Path $tempOut) -and ($result.ExitCode -eq 0)
        Remove-Item $tempBat, $tempOut -ErrorAction SilentlyContinue
        return @{ Success = $success; Message = if ($success) { "Environment setup successful" } else { "Environment setup failed" } }
    }
    catch {
        return @{ Success = $false; Message = "Environment test error: $($_.Exception.Message)" }
    }
}

function Get-ComponentVersion {
    param([string]$ComponentPath)
    try {
        $versionFile = Get-ChildItem -Path $ComponentPath -Filter "*version*" -File -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($versionFile) {
            $content = Get-Content $versionFile.FullName -ErrorAction SilentlyContinue
            return ($content | Where-Object { $_ -match "\d+\.\d+" } | Select-Object -First 1)
        }
        return "Unknown"
    }
    catch { return "Unknown" }
}

try {
    if (-not $Quiet) {
        Write-ColoredOutput "==============================================================" -Color Cyan
        Write-ColoredOutput "Intel OneAPI Installation Test for Gemma.cpp (PowerShell Edition)" -Color Cyan
        Write-ColoredOutput "==============================================================" -Color Cyan
        Write-Host ""
    }

    $testResults = @()
    $overallSuccess = $true

    # Test main OneAPI installation
    Write-ColoredOutput "Checking Intel OneAPI installation..." -Color Yellow
    $setvarsPath = Join-Path $ComponentPath "setvars.bat"
    $mainTest = Test-ComponentPath -Path $setvarsPath -ComponentName "Intel OneAPI setvars.bat"
    $testResults += $mainTest
    $overallSuccess = $overallSuccess -and $mainTest.Exists

    if ($mainTest.Exists) {
        # Test environment setup
        Write-ColoredOutput "Testing environment setup..." -Color Yellow
        $envTest = Test-EnvironmentSetup -OneAPIPath $ComponentPath
        $envStatus = if ($envTest.Success) { "[OK]" } else { "[FAILED]" }
        $envColor = if ($envTest.Success) { [ConsoleColor]::Green } else { [ConsoleColor]::Red }
        Write-ColoredOutput "$envStatus Environment Configuration" -Color $envColor
        if ($Detailed) { Write-ColoredOutput "      $($envTest.Message)" -Color Gray }
        $overallSuccess = $overallSuccess -and $envTest.Success

        # Test required components
        Write-ColoredOutput "Checking required components..." -Color Yellow
        $components = @(
            @{ Name = "Intel MKL"; Path = (Join-Path $ComponentPath "mkl") }
            @{ Name = "Intel TBB"; Path = (Join-Path $ComponentPath "tbb") }
            @{ Name = "Intel IPP"; Path = (Join-Path $ComponentPath "ipp") }
            @{ Name = "Intel DNNL"; Path = (Join-Path $ComponentPath "dnnl") }
            @{ Name = "Intel Compiler"; Path = (Join-Path $ComponentPath "compiler") }
        )

        foreach ($component in $components) {
            $test = Test-ComponentPath -Path $component.Path -ComponentName $component.Name
            $testResults += $test
            if ($component.Name -in @("Intel MKL", "Intel TBB")) {
                $overallSuccess = $overallSuccess -and $test.Exists
            }
        }
    } else {
        Write-ColoredOutput "[ERROR] Intel OneAPI not found\!" -Color Red
        $overallSuccess = $false
    }

    # Summary
    Write-Host ""
    if ($overallSuccess) {
        Write-ColoredOutput "=== TEST RESULT: PASSED ===" -Color Green
        Write-ColoredOutput "Intel OneAPI is properly installed and ready for Gemma.cpp development\!" -Color Green
        Write-Host ""
        Write-ColoredOutput "Next steps:" -Color Yellow
        Write-ColoredOutput "1. Run: .\build_intel_oneapi.ps1" -Color White
        Write-ColoredOutput "2. Wait for build completion" -Color White
        Write-ColoredOutput "3. Test the built executable" -Color White
    } else {
        Write-ColoredOutput "=== TEST RESULT: FAILED ===" -Color Red
        Write-ColoredOutput "Intel OneAPI installation issues detected\!" -Color Red
        Write-Host ""
        Write-ColoredOutput "Required actions:" -Color Yellow
        Write-ColoredOutput "1. Download Intel OneAPI Base Toolkit from:" -Color White
        Write-ColoredOutput "   https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html" -Color White
        Write-ColoredOutput "2. Install with default settings" -Color White
        Write-ColoredOutput "3. Restart your terminal/PowerShell session" -Color White
        Write-ColoredOutput "4. Re-run this test script" -Color White
    }

    # Export report if requested
    if ($ExportReport) {
        $reportPath = "intel_oneapi_test_report.json"
        $report = @{
            Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            OverallSuccess = $overallSuccess
            ComponentPath = $ComponentPath
            TestResults = $testResults
            PowerShellVersion = $PSVersionTable.PSVersion.ToString()
            OSVersion = [System.Environment]::OSVersion.VersionString
        }
        $report | ConvertTo-Json -Depth 3 | Set-Content -Path $reportPath -Encoding UTF8
        Write-ColoredOutput "Test report exported to: $reportPath" -Color Magenta
    }

    # Exit with appropriate code
    if (-not $overallSuccess) { exit 1 }
    Write-Host ""
    Write-ColoredOutput "Press any key to continue..." -Color Gray
    if (-not $Quiet) { $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") }
}
catch {
    Write-ColoredOutput "ERROR: $($_.Exception.Message)" -Color Red
    Write-Host ""
    Write-ColoredOutput "Script execution failed. Please check:" -Color Yellow
    Write-ColoredOutput "1. PowerShell execution policy (Run as Administrator: Set-ExecutionPolicy RemoteSigned)" -Color White
    Write-ColoredOutput "2. File permissions on Intel OneAPI directory" -Color White
    Write-ColoredOutput "3. Available disk space and system resources" -Color White
    exit 1
}
