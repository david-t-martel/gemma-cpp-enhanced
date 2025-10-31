# =============================================================================
# PHASE 1: PROFILE SYSTEM FOUNDATION
# =============================================================================
# Basic profile management system for LLM agent integration.
#
# Version: 1.0.0
# Created: 2025-01-15
# =============================================================================

# Global profile storage
$global:ProfileConfig = $null
$global:ProfileConfigPath = $null

#region Core Profile Functions

<#
.SYNOPSIS
    Sets the active LLM profile.

.DESCRIPTION
    Creates or activates an LLM profile with working directory and configuration.

.PARAMETER ProfileName
    Name of the profile to create or activate.

.PARAMETER WorkingDirectory
    Working directory for this profile.

.EXAMPLE
    Set-LLMProfile -ProfileName "gemma-dev" -WorkingDirectory "C:\codedev\llm\gemma"
#>
function Set-LLMProfile {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$ProfileName,

        [Parameter(Mandatory = $true)]
        [string]$WorkingDirectory
    )

    # Ensure profile directory exists
    $profileDir = "$env:USERPROFILE\.llm-profile\profiles"
    if (-not (Test-Path $profileDir)) {
        New-Item -ItemType Directory -Path $profileDir -Force | Out-Null
    }

    # Profile config path
    $global:ProfileConfigPath = "$profileDir\$ProfileName.json"

    # Load or create profile
    if (Test-Path $global:ProfileConfigPath) {
        $global:ProfileConfig = Get-Content $global:ProfileConfigPath | ConvertFrom-Json -AsHashtable
        Write-Host "✓ Loaded profile: $ProfileName" -ForegroundColor Green
    } else {
        $global:ProfileConfig = @{
            ProfileName = $ProfileName
            WorkingDirectory = $WorkingDirectory
            CreatedAt = Get-Date -Format "o"
            LastModified = Get-Date -Format "o"
        }
        Write-Host "✓ Created profile: $ProfileName" -ForegroundColor Green
    }

    # Update working directory
    $global:ProfileConfig.WorkingDirectory = $WorkingDirectory
    $global:ProfileConfig.LastModified = Get-Date -Format "o"

    # Save profile
    $global:ProfileConfig | ConvertTo-Json -Depth 10 | Set-Content -Path $global:ProfileConfigPath -Force

    # Change to working directory if it exists
    if (Test-Path $WorkingDirectory) {
        Set-Location $WorkingDirectory
    }

    Write-Host "  Working Directory: $WorkingDirectory" -ForegroundColor Cyan
}

<#
.SYNOPSIS
    Gets the current LLM profile configuration.

.DESCRIPTION
    Returns the active profile configuration.

.EXAMPLE
    Get-LLMProfile
#>
function Get-LLMProfile {
    [CmdletBinding()]
    param()

    if ($null -eq $global:ProfileConfig) {
        Write-Warning "No active profile. Use Set-LLMProfile to create or activate a profile."
        return $null
    }

    return $global:ProfileConfig
}

<#
.SYNOPSIS
    Lists all available LLM profiles.

.DESCRIPTION
    Shows all profiles stored in the profile directory.

.EXAMPLE
    Get-LLMProfiles
#>
function Get-LLMProfiles {
    [CmdletBinding()]
    param()

    $profileDir = "$env:USERPROFILE\.llm-profile\profiles"
    if (-not (Test-Path $profileDir)) {
        Write-Host "No profiles found" -ForegroundColor Yellow
        return @()
    }

    $profiles = Get-ChildItem -Path $profileDir -Filter "*.json" | ForEach-Object {
        $config = Get-Content $_.FullName | ConvertFrom-Json
        [PSCustomObject]@{
            ProfileName = $config.ProfileName
            WorkingDirectory = $config.WorkingDirectory
            CreatedAt = $config.CreatedAt
            LastModified = $config.LastModified
        }
    }

    return $profiles
}

<#
.SYNOPSIS
    Removes an LLM profile.

.DESCRIPTION
    Deletes a profile configuration file.

.PARAMETER ProfileName
    Name of the profile to remove.

.EXAMPLE
    Remove-LLMProfile -ProfileName "test"
#>
function Remove-LLMProfile {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$ProfileName
    )

    $profilePath = "$env:USERPROFILE\.llm-profile\profiles\$ProfileName.json"
    if (Test-Path $profilePath) {
        Remove-Item -Path $profilePath -Force
        Write-Host "✓ Removed profile: $ProfileName" -ForegroundColor Green

        # Clear global if this was the active profile
        if ($global:ProfileConfig -and $global:ProfileConfig.ProfileName -eq $ProfileName) {
            $global:ProfileConfig = $null
            $global:ProfileConfigPath = $null
        }
    } else {
        Write-Warning "Profile not found: $ProfileName"
    }
}

#endregion

Write-Host "✓ Phase 1 profile system loaded" -ForegroundColor Cyan
