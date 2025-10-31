# =============================================================================
# PHASE 2A: AUTO-CLAUDE PROFILE-AWARE INTEGRATION
# =============================================================================
# Provides PowerShell functions to integrate with auto-claude CLI tool
# with profile-specific contexts and configurations.
#
# Dependencies:
# - auto-claude.exe in PATH or ~/.local/bin
# - Profile context from $ProfileConfig global variable
#
# Version: 1.0.0
# Created: $(Get-Date -Format 'yyyy-MM-dd')
# =============================================================================

#region Auto-Claude Integration Functions

<#
.SYNOPSIS
    Invokes auto-claude with profile-aware context.

.DESCRIPTION
    Executes auto-claude CLI with automatic profile context injection,
    secure credential passing, and output capture.

.PARAMETER Prompt
    The prompt/query to send to auto-claude.

.PARAMETER Context
    Additional context files or directories to include.

.PARAMETER Model
    Model to use (defaults to profile preference or claude-3-5-sonnet).

.PARAMETER MaxTokens
    Maximum tokens for response (default: 4096).

.PARAMETER Temperature
    Temperature for generation (default: 0.7).

.PARAMETER IncludeProfileContext
    Whether to automatically include profile context (default: $true).

.PARAMETER PassThrough
    Pass additional arguments directly to auto-claude.

.EXAMPLE
    Invoke-AutoClaude "Explain this codebase"

.EXAMPLE
    Invoke-AutoClaude "Review these changes" -Context "./src" -Model "claude-3-opus"

.EXAMPLE
    Invoke-AutoClaude "Generate tests" -IncludeProfileContext $false -PassThrough @("--format", "json")
#>
function Invoke-AutoClaude {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true, Position = 0)]
        [string]$Prompt,

        [Parameter()]
        [string[]]$Context = @(),

        [Parameter()]
        [string]$Model = "",

        [Parameter()]
        [int]$MaxTokens = 4096,

        [Parameter()]
        [double]$Temperature = 0.7,

        [Parameter()]
        [bool]$IncludeProfileContext = $true,

        [Parameter()]
        [string[]]$PassThrough = @()
    )

    # Validate auto-claude is available
    $autoClaudePath = Get-Command "auto-claude" -ErrorAction SilentlyContinue
    if (-not $autoClaudePath) {
        Write-Error "auto-claude not found in PATH. Install from: https://github.com/skydeckai/auto-claude"
        return $null
    }

    # Build arguments
    $args = @()
    
    # Add prompt
    $args += @("--prompt", $Prompt)

    # Add model (use profile preference or default)
    $effectiveModel = $Model
    if ([string]::IsNullOrWhiteSpace($effectiveModel)) {
        if ($global:ProfileConfig -and $global:ProfileConfig.ContainsKey("PreferredModel")) {
            $effectiveModel = $global:ProfileConfig.PreferredModel
        } else {
            $effectiveModel = "claude-3-5-sonnet-20241022"
        }
    }
    $args += @("--model", $effectiveModel)

    # Add generation parameters
    $args += @("--max-tokens", $MaxTokens.ToString())
    $args += @("--temperature", $Temperature.ToString())

    # Add profile context if enabled
    if ($IncludeProfileContext -and $global:ProfileConfig) {
        # Add profile working directory as context
        if ($global:ProfileConfig.ContainsKey("WorkingDirectory") -and 
            (Test-Path $global:ProfileConfig.WorkingDirectory)) {
            $args += @("--context", $global:ProfileConfig.WorkingDirectory)
        }

        # Add profile-specific context files
        if ($global:ProfileConfig.ContainsKey("ContextFiles")) {
            foreach ($file in $global:ProfileConfig.ContextFiles) {
                if (Test-Path $file) {
                    $args += @("--context", $file)
                }
            }
        }
    }

    # Add explicit context paths
    foreach ($ctx in $Context) {
        if (Test-Path $ctx) {
            $args += @("--context", $ctx)
        } else {
            Write-Warning "Context path not found: $ctx"
        }
    }

    # Add passthrough arguments
    if ($PassThrough.Count -gt 0) {
        $args += $PassThrough
    }

    # Invoke auto-claude
    try {
        Write-Verbose "Invoking auto-claude with model: $effectiveModel"
        Write-Verbose "Arguments: $($args -join ' ')"
        
        $result = & auto-claude @args 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            return $result
        } else {
            Write-Error "auto-claude failed with exit code: $LASTEXITCODE"
            Write-Error $result
            return $null
        }
    }
    catch {
        Write-Error "Failed to invoke auto-claude: $_"
        return $null
    }
}

<#
.SYNOPSIS
    Gets current auto-claude configuration.

.DESCRIPTION
    Retrieves and displays current auto-claude settings including
    API key status, model preferences, and profile integration.

.EXAMPLE
    Get-AutoClaudeConfig
#>
function Get-AutoClaudeConfig {
    [CmdletBinding()]
    param()

    $config = @{
        AutoClaudeInstalled = $null -ne (Get-Command "auto-claude" -ErrorAction SilentlyContinue)
        ApiKeyConfigured = $null -ne $env:ANTHROPIC_API_KEY
        ProfileIntegration = $null -ne $global:ProfileConfig
        PreferredModel = $null
        ContextPaths = @()
    }

    # Get preferred model
    if ($global:ProfileConfig -and $global:ProfileConfig.ContainsKey("PreferredModel")) {
        $config.PreferredModel = $global:ProfileConfig.PreferredModel
    } else {
        $config.PreferredModel = "claude-3-5-sonnet-20241022 (default)"
    }

    # Get context paths
    if ($global:ProfileConfig) {
        if ($global:ProfileConfig.ContainsKey("WorkingDirectory")) {
            $config.ContextPaths += $global:ProfileConfig.WorkingDirectory
        }
        if ($global:ProfileConfig.ContainsKey("ContextFiles")) {
            $config.ContextPaths += $global:ProfileConfig.ContextFiles
        }
    }

    return $config
}

<#
.SYNOPSIS
    Sets auto-claude configuration for current profile.

.DESCRIPTION
    Updates profile-specific auto-claude settings including model
    preferences and context paths.

.PARAMETER PreferredModel
    Default model to use for this profile.

.PARAMETER AddContextPath
    Add a context path to profile auto-claude configuration.

.PARAMETER RemoveContextPath
    Remove a context path from profile configuration.

.EXAMPLE
    Set-AutoClaudeConfig -PreferredModel "claude-3-opus-20240229"

.EXAMPLE
    Set-AutoClaudeConfig -AddContextPath "C:\Projects\MyApp\docs"
#>
function Set-AutoClaudeConfig {
    [CmdletBinding()]
    param(
        [Parameter()]
        [string]$PreferredModel,

        [Parameter()]
        [string]$AddContextPath,

        [Parameter()]
        [string]$RemoveContextPath
    )

    if (-not $global:ProfileConfig) {
        Write-Error "No active profile. Use Set-LLMProfile first."
        return
    }

    # Set preferred model
    if (-not [string]::IsNullOrWhiteSpace($PreferredModel)) {
        $global:ProfileConfig.PreferredModel = $PreferredModel
        Write-Host "✓ Set preferred model to: $PreferredModel" -ForegroundColor Green
    }

    # Add context path
    if (-not [string]::IsNullOrWhiteSpace($AddContextPath)) {
        if (-not (Test-Path $AddContextPath)) {
            Write-Warning "Path does not exist: $AddContextPath"
        }
        
        if (-not $global:ProfileConfig.ContainsKey("ContextFiles")) {
            $global:ProfileConfig.ContextFiles = @()
        }
        
        if ($global:ProfileConfig.ContextFiles -notcontains $AddContextPath) {
            $global:ProfileConfig.ContextFiles += $AddContextPath
            Write-Host "✓ Added context path: $AddContextPath" -ForegroundColor Green
        } else {
            Write-Warning "Context path already configured: $AddContextPath"
        }
    }

    # Remove context path
    if (-not [string]::IsNullOrWhiteSpace($RemoveContextPath)) {
        if ($global:ProfileConfig.ContainsKey("ContextFiles")) {
            $global:ProfileConfig.ContextFiles = $global:ProfileConfig.ContextFiles | 
                Where-Object { $_ -ne $RemoveContextPath }
            Write-Host "✓ Removed context path: $RemoveContextPath" -ForegroundColor Green
        }
    }

    # Persist profile config
    if ($global:ProfileConfigPath) {
        try {
            $global:ProfileConfig | ConvertTo-Json -Depth 10 | 
                Set-Content -Path $global:ProfileConfigPath -Force
            Write-Verbose "Profile configuration saved"
        }
        catch {
            Write-Warning "Failed to save profile configuration: $_"
        }
    }
}

#endregion

#region Helper Aliases

# Convenient aliases for auto-claude integration
Set-Alias -Name "ac" -Value "Invoke-AutoClaude" -Scope Global -ErrorAction SilentlyContinue
Set-Alias -Name "claude" -Value "Invoke-AutoClaude" -Scope Global -ErrorAction SilentlyContinue

#endregion

Write-Host "✓ Auto-Claude profile integration loaded" -ForegroundColor Cyan
