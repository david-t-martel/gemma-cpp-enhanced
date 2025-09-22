<#
.SYNOPSIS
    Runs gemma.cpp in WSL Ubuntu environment.

.DESCRIPTION
    This script executes gemma.cpp in a WSL Ubuntu distribution, automatically
    forwarding all command-line arguments to the WSL environment.

.PARAMETER Arguments
    All arguments passed to this script will be forwarded to the gemma.cpp executable in WSL.

.PARAMETER Distribution
    The WSL distribution to use. Defaults to 'Ubuntu'.

.PARAMETER Verbose
    Enable verbose output showing detailed execution information.

.PARAMETER WhatIf
    Show what would be executed without actually running the command.

.EXAMPLE
    .\run_gemma_wsl.ps1 --model gemma-2b --prompt "Hello"
    Runs gemma.cpp in WSL with the specified model and prompt.

.EXAMPLE
    .\run_gemma_wsl.ps1 -Distribution "Ubuntu-20.04" --help
    Runs gemma.cpp help in a specific Ubuntu distribution.

.NOTES
    Author: Converted from batch script
    Requires: WSL with Ubuntu distribution and built gemma.cpp
#>

[CmdletBinding(SupportsShouldProcess)]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments = @(),

    [Parameter()]
    [string]$Distribution = "Ubuntu",

    [Parameter()]
    [switch]$Verbose
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Function to write colored output
function Write-ColoredOutput {
    param(
        [string]$Message,
        [ConsoleColor]$Color = [ConsoleColor]::White
    )

    $originalColor = $Host.UI.RawUI.ForegroundColor
    $Host.UI.RawUI.ForegroundColor = $Color
    Write-Host $Message
    $Host.UI.RawUI.ForegroundColor = $originalColor
}

# Function to test WSL availability
function Test-WSLAvailability {
    try {
        $wslInfo = wsl --list --verbose 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw "WSL command failed"
        }
        return $true
    }
    catch {
        return $false
    }
}

# Function to test specific distribution
function Test-WSLDistribution {
    param([string]$DistName)

    try {
        $distributions = wsl --list --quiet
        $distExists = $distributions -contains $DistName
        return $distExists
    }
    catch {
        return $false
    }
}

# Function to test if gemma build exists in WSL
function Test-GemmaBuildInWSL {
    param([string]$DistName)

    try {
        $testResult = wsl -d $DistName bash -c "test -f /mnt/c/codedev/llm/gemma/gemma.cpp/build_wsl/run_gemma.sh && echo 'exists' || echo 'missing'"
        return $testResult.Trim() -eq "exists"
    }
    catch {
        return $false
    }
}

# Main execution
try {
    Write-ColoredOutput "==============================================================" -Color Cyan
    Write-ColoredOutput "Gemma.cpp WSL Runner (PowerShell Edition)" -Color Cyan
    Write-ColoredOutput "==============================================================" -Color Cyan
    Write-Host ""

    # Test WSL availability
    if ($Verbose) { Write-ColoredOutput "Checking WSL availability..." -Color Yellow }
    if (-not (Test-WSLAvailability)) {
        throw "WSL is not available or not properly installed. Please install WSL and try again."
    }
    if ($Verbose) { Write-ColoredOutput "[OK] WSL is available" -Color Green }

    # Test distribution availability
    if ($Verbose) { Write-ColoredOutput "Checking distribution '$Distribution'..." -Color Yellow }
    if (-not (Test-WSLDistribution -DistName $Distribution)) {
        throw "WSL distribution '$Distribution' is not available. Available distributions:`n$(wsl --list)"
    }
    if ($Verbose) { Write-ColoredOutput "[OK] Distribution '$Distribution' is available" -Color Green }

    # Test gemma build
    if ($Verbose) { Write-ColoredOutput "Checking gemma.cpp build in WSL..." -Color Yellow }
    if (-not (Test-GemmaBuildInWSL -DistName $Distribution)) {
        throw "Gemma.cpp build not found in WSL. Expected: /mnt/c/codedev/llm/gemma/gemma.cpp/build_wsl/run_gemma.sh"
    }
    if ($Verbose) { Write-ColoredOutput "[OK] Gemma.cpp build found" -Color Green }

    # Prepare command
    $wslPath = "/mnt/c/codedev/llm/gemma/gemma.cpp/build_wsl"
    $argumentString = ($Arguments -join " ")
    $bashCommand = "cd $wslPath && ./run_gemma.sh $argumentString"

    if ($Verbose) {
        Write-ColoredOutput "WSL Distribution: $Distribution" -Color Magenta
        Write-ColoredOutput "Working Directory: $wslPath" -Color Magenta
        Write-ColoredOutput "Command: ./run_gemma.sh $argumentString" -Color Magenta
        Write-Host ""
    }

    if ($PSCmdlet.ShouldProcess("WSL $Distribution", "Execute gemma.cpp with arguments: $argumentString")) {
        Write-ColoredOutput "Starting gemma.cpp in WSL..." -Color Green
        Write-Host ""

        # Execute the command
        wsl -d $Distribution bash -c $bashCommand

        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-ColoredOutput "Gemma.cpp execution completed successfully!" -Color Green
        } else {
            Write-Host ""
            Write-ColoredOutput "Gemma.cpp execution failed with exit code: $LASTEXITCODE" -Color Red
            exit $LASTEXITCODE
        }
    }
}
catch {
    Write-ColoredOutput "ERROR: $($_.Exception.Message)" -Color Red
    Write-Host ""
    Write-ColoredOutput "Troubleshooting tips:" -Color Yellow
    Write-ColoredOutput "1. Ensure WSL is installed: wsl --install" -Color White
    Write-ColoredOutput "2. Ensure Ubuntu is installed: wsl --install -d Ubuntu" -Color White
    Write-ColoredOutput "3. Build gemma.cpp in WSL first" -Color White
    Write-ColoredOutput "4. Check the build path exists: /mnt/c/codedev/llm/gemma/gemma.cpp/build_wsl/" -Color White
    exit 1
}
