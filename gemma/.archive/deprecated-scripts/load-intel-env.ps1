<#
.SYNOPSIS
    Loads Intel OneAPI environment variables into the current PowerShell session.

.DESCRIPTION
    This script initializes Intel OneAPI environment variables for development
    with Intel optimized tools including compiler, MKL, TBB, and other components.

.PARAMETER OneAPIPath
    Path to Intel OneAPI installation directory

.PARAMETER Architecture
    Target architecture (intel64, ia32). Default: intel64

.PARAMETER VisualStudioVersion
    Visual Studio version integration (vs2019, vs2022). Default: vs2022

.PARAMETER Components
    Specific components to load (mkl, tbb, ipp, dnnl, compiler, all). Default: all

.PARAMETER Persistent
    Make environment changes persistent for current user (modifies user environment)

.PARAMETER Verbose
    Show detailed information about loaded components

.PARAMETER Validate
    Validate that components are properly loaded after initialization

.EXAMPLE
    .\load-intel-env.ps1
    Loads Intel OneAPI environment with default settings

.EXAMPLE
    .\load-intel-env.ps1 -Components mkl,tbb -Verbose
    Loads only MKL and TBB components with verbose output

.EXAMPLE
    .\load-intel-env.ps1 -Persistent -Validate
    Loads environment persistently and validates all components

.NOTES
    Author: PowerShell implementation of Intel environment loader
    Requires: Intel OneAPI Base Toolkit
    Version: 1.0
#>

[CmdletBinding()]
param(
    [string]$OneAPIPath = "C:\Program Files (x86)\Intel\oneAPI",
    [ValidateSet("intel64", "ia32")]
    [string]$Architecture = "intel64",
    [ValidateSet("vs2019", "vs2022")]
    [string]$VisualStudioVersion = "vs2022",
    [ValidateSet("mkl", "tbb", "ipp", "dnnl", "compiler", "all")]
    [string[]]$Components = @("all"),
    [switch]$Persistent,
    [switch]$Verbose,
    [switch]$Validate
)

$ErrorActionPreference = "Stop"

function Write-ColoredOutput {
    param([string]$Message, [ConsoleColor]$Color = [ConsoleColor]::White)
    $originalColor = $Host.UI.RawUI.ForegroundColor
    $Host.UI.RawUI.ForegroundColor = $Color
    Write-Host $Message
    $Host.UI.RawUI.ForegroundColor = $originalColor
}

function Test-IntelComponent {
    param([string]$ComponentPath, [string]$ComponentName)
    $exists = Test-Path $ComponentPath
    if ($Verbose) {
        $status = if ($exists) { "[OK]" } else { "[MISSING]" }
        $color = if ($exists) { [ConsoleColor]::Green } else { [ConsoleColor]::Yellow }
        Write-ColoredOutput "  $status $ComponentName" -Color $color
    }
    return $exists
}

function Set-IntelEnvironment {
    param([string]$OneAPIPath, [string]$Arch, [string]$VSVersion, [string[]]$ComponentList)

    # Main setvars.bat path
    $setvarsPath = Join-Path $OneAPIPath "setvars.bat"
    if (-not (Test-Path $setvarsPath)) {
        throw "Intel OneAPI setvars.bat not found at: $setvarsPath"
    }

    if ($Verbose) { Write-ColoredOutput "Loading Intel OneAPI environment..." -Color Yellow }

    try {
        # Create temporary files for environment capture
        $tempBat = [System.IO.Path]::GetTempFileName() + ".bat"
        $tempEnvBefore = [System.IO.Path]::GetTempFileName() + ".env.before"
        $tempEnvAfter = [System.IO.Path]::GetTempFileName() + ".env.after"

        # Create batch script to capture environment before and after
        $batchContent = @"
@echo off
set > "$tempEnvBefore"
call "$setvarsPath" $Arch $VSVersion >nul 2>&1
if errorlevel 1 (
    echo INTEL_ENV_LOAD_FAILED
    exit /b 1
)
set > "$tempEnvAfter"
echo INTEL_ENV_LOAD_SUCCESS
"@

        Set-Content -Path $tempBat -Value $batchContent

        # Execute the batch file
        $result = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", $tempBat -Wait -PassThru -RedirectStandardOutput $tempEnvAfter

        if ($result.ExitCode -ne 0) {
            throw "Failed to initialize Intel OneAPI environment (exit code: $($result.ExitCode))"
        }

        # Read environment variables after Intel setup
        if (Test-Path $tempEnvAfter) {
            $envAfter = @{}
            Get-Content $tempEnvAfter | ForEach-Object {
                if ($_ -match "^([^=]+)=(.*)$") {
                    $envAfter[$matches[1]] = $matches[2]
                }
            }

            # Read environment variables before Intel setup
            $envBefore = @{}
            if (Test-Path $tempEnvBefore) {
                Get-Content $tempEnvBefore | ForEach-Object {
                    if ($_ -match "^([^=]+)=(.*)$") {
                        $envBefore[$matches[1]] = $matches[2]
                    }
                }
            }

            # Apply new or changed environment variables
            $changedVars = @()
            foreach ($key in $envAfter.Keys) {
                if (-not $envBefore.ContainsKey($key) -or $envBefore[$key] -ne $envAfter[$key]) {
                    $envScope = if ($Persistent) { "User" } else { "Process" }
                    [Environment]::SetEnvironmentVariable($key, $envAfter[$key], $envScope)
                    $changedVars += $key
                    if ($Verbose -and $key -match "(INTEL|MKL|TBB|IPP|DNNL)") {
                        $value = if ($envAfter[$key].Length -gt 50) { $envAfter[$key].Substring(0, 47) + "..." } else { $envAfter[$key] }
                        Write-ColoredOutput "  Set $key = $value" -Color Gray
                    }
                }
            }

            if ($Verbose) {
                Write-ColoredOutput "Applied $($changedVars.Count) environment variables" -Color Green
            }
        }

        # Cleanup temporary files
        Remove-Item $tempBat, $tempEnvBefore, $tempEnvAfter -ErrorAction SilentlyContinue

        if ($Verbose) {
            Write-ColoredOutput "[OK] Intel OneAPI environment loaded successfully" -Color Green
        }
    }
    catch {
        # Cleanup on error
        Remove-Item $tempBat, $tempEnvBefore, $tempEnvAfter -ErrorAction SilentlyContinue
        throw "Failed to load Intel OneAPI environment: $($_.Exception.Message)"
    }
}

function Test-LoadedEnvironment {
    param([string]$OneAPIPath)

    if ($Verbose) { Write-ColoredOutput "Validating loaded environment..." -Color Yellow }

    $validationResults = @()
    $allValid = $true

    # Test key environment variables
    $requiredVars = @("ONEAPI_ROOT", "MKLROOT", "TBBROOT")
    foreach ($var in $requiredVars) {
        $value = [Environment]::GetEnvironmentVariable($var)
        $isValid = -not [string]::IsNullOrEmpty($value) -and (Test-Path $value -ErrorAction SilentlyContinue)
        $validationResults += @{ Variable = $var; Valid = $isValid; Value = $value }
        $allValid = $allValid -and $isValid
        if ($Verbose) {
            $status = if ($isValid) { "[OK]" } else { "[FAILED]" }
            $color = if ($isValid) { [ConsoleColor]::Green } else { [ConsoleColor]::Red }
            Write-ColoredOutput "  $status $var" -Color $color
        }
    }

    # Test component directories
    $components = @(
        @{ Name = "Intel MKL"; Path = (Join-Path $OneAPIPath "mkl") }
        @{ Name = "Intel TBB"; Path = (Join-Path $OneAPIPath "tbb") }
        @{ Name = "Intel IPP"; Path = (Join-Path $OneAPIPath "ipp") }
        @{ Name = "Intel DNNL"; Path = (Join-Path $OneAPIPath "dnnl") }
    )

    foreach ($component in $components) {
        $isValid = Test-IntelComponent -ComponentPath $component.Path -ComponentName $component.Name
        $validationResults += @{ Component = $component.Name; Valid = $isValid; Path = $component.Path }
        if ($component.Name -in @("Intel MKL", "Intel TBB")) {
            $allValid = $allValid -and $isValid
        }
    }

    return @{ Success = $allValid; Results = $validationResults }
}

# Main execution
try {
    Write-ColoredOutput "==============================================================" -Color Cyan
    Write-ColoredOutput "Intel OneAPI Environment Loader (PowerShell Edition)" -Color Cyan
    Write-ColoredOutput "==============================================================" -Color Cyan
    Write-Host ""

    # Validate Intel OneAPI installation
    if (-not (Test-Path $OneAPIPath)) {
        throw "Intel OneAPI installation not found at: $OneAPIPath`nPlease install Intel OneAPI Base Toolkit or specify correct path with -OneAPIPath parameter."
    }

    if ($Verbose) {
        Write-ColoredOutput "Configuration:" -Color Magenta
        Write-ColoredOutput "  OneAPI Path: $OneAPIPath" -Color White
        Write-ColoredOutput "  Architecture: $Architecture" -Color White
        Write-ColoredOutput "  Visual Studio: $VisualStudioVersion" -Color White
        Write-ColoredOutput "  Components: $($Components -join ", ")" -Color White
        Write-ColoredOutput "  Persistent: $Persistent" -Color White
        Write-Host ""
    }

    # Check available components
    if ($Verbose) {
        Write-ColoredOutput "Checking available components..." -Color Yellow
        Test-IntelComponent -ComponentPath (Join-Path $OneAPIPath "mkl") -ComponentName "Intel MKL"
        Test-IntelComponent -ComponentPath (Join-Path $OneAPIPath "tbb") -ComponentName "Intel TBB"
        Test-IntelComponent -ComponentPath (Join-Path $OneAPIPath "ipp") -ComponentName "Intel IPP"
        Test-IntelComponent -ComponentPath (Join-Path $OneAPIPath "dnnl") -ComponentName "Intel DNNL"
        Test-IntelComponent -ComponentPath (Join-Path $OneAPIPath "compiler") -ComponentName "Intel Compiler"
        Write-Host ""
    }

    # Load Intel environment
    Set-IntelEnvironment -OneAPIPath $OneAPIPath -Arch $Architecture -VSVersion $VisualStudioVersion -ComponentList $Components

    # Validate if requested
    if ($Validate) {
        Write-Host ""
        $validation = Test-LoadedEnvironment -OneAPIPath $OneAPIPath
        if ($validation.Success) {
            Write-ColoredOutput "=== VALIDATION: PASSED ===" -Color Green
            Write-ColoredOutput "Intel OneAPI environment is properly loaded and ready!" -Color Green
        } else {
            Write-ColoredOutput "=== VALIDATION: FAILED ===" -Color Red
            Write-ColoredOutput "Some components may not be properly loaded." -Color Yellow
        }
    } else {
        Write-ColoredOutput "Intel OneAPI environment loaded successfully!" -Color Green
    }

    Write-Host ""
    Write-ColoredOutput "Usage examples:" -Color Yellow
    Write-ColoredOutput "  Build with CMake: cmake -DGEMMA_USE_INTEL_MKL=ON -DGEMMA_USE_INTEL_TBB=ON .." -Color White
    Write-ColoredOutput "  Check MKL path:   echo `$env:MKLROOT" -Color White
    Write-ColoredOutput "  Check TBB path:   echo `$env:TBBROOT" -Color White

    if ($Persistent) {
        Write-ColoredOutput "NOTE: Environment variables have been made persistent for the current user." -Color Magenta
        Write-ColoredOutput "      New PowerShell sessions will automatically have Intel OneAPI available." -Color Magenta
    } else {
        Write-ColoredOutput "NOTE: Environment is loaded for this session only." -Color Magenta
        Write-ColoredOutput "      Run with -Persistent to make changes permanent." -Color Magenta
    }
}
catch {
    Write-ColoredOutput "ERROR: $($_.Exception.Message)" -Color Red
    Write-Host ""
    Write-ColoredOutput "Troubleshooting:" -Color Yellow
    Write-ColoredOutput "1. Verify Intel OneAPI is installed: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html" -Color White
    Write-ColoredOutput "2. Check installation path: dir `"$OneAPIPath`"" -Color White
    Write-ColoredOutput "3. Run as Administrator if permission errors occur" -Color White
    Write-ColoredOutput "4. Use -Verbose for detailed diagnostic output" -Color White
    exit 1
}
