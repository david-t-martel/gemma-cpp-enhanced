# =============================================================================
# PHASE 2A: TOOL USE HOOKS WITH AUDITING
# =============================================================================
# Provides PowerShell hooks for agent tool invocations with automatic
# logging, security scanning, and profile-aware context injection.
#
# Version: 1.0.0
# Created: $(Get-Date -Format 'yyyy-MM-dd')
# =============================================================================

#region Tool Hook Configuration

# Global tool audit log
$script:ToolAuditLog = @()
$script:ToolAuditLogPath = "$env:USERPROFILE\.llm-profile\tool-audit.jsonl"

# Security scanning configuration
$script:DangerousPatterns = @(
    '\brm\s+-rf\s+/',
    '\bdel\s+/s\s+/q',
    'format\s+[cdefg]:',
    'Remove-Item.*-Recurse.*-Force',
    '\b(sudo|su)\s+',
    'curl.*\|\s*sh',
    'wget.*\|\s*bash'
)

<#
.SYNOPSIS
    Registers a tool use hook for auditing and security.

.DESCRIPTION
    Intercepts agent tool invocations to log usage, scan for
    dangerous commands, and inject profile context automatically.

.PARAMETER ToolName
    Name of the tool to hook.

.PARAMETER PreHook
    ScriptBlock to execute before tool invocation.

.PARAMETER PostHook
    ScriptBlock to execute after tool invocation.

.EXAMPLE
    Register-ToolHook -ToolName "Invoke-LLMCommand" -PreHook {
        param($ToolName, $Arguments)
        Write-Host "About to run: $ToolName"
    }
#>
function Register-ToolHook {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$ToolName,

        [Parameter()]
        [scriptblock]$PreHook,

        [Parameter()]
        [scriptblock]$PostHook
    )

    if (-not $global:ToolHooks) {
        $global:ToolHooks = @{}
    }

    $global:ToolHooks[$ToolName] = @{
        PreHook = $PreHook
        PostHook = $PostHook
        RegisteredAt = Get-Date -Format "o"
    }

    Write-Verbose "Registered hooks for tool: $ToolName"
}

<#
.SYNOPSIS
    Invokes a tool with registered hooks.

.DESCRIPTION
    Executes a tool command with pre/post hooks for auditing,
    security scanning, and context injection.

.PARAMETER ToolName
    Name of the tool to invoke.

.PARAMETER Arguments
    Arguments to pass to the tool.

.PARAMETER SkipSecurity
    Skip security scanning (use with caution).

.EXAMPLE
    Invoke-ToolWithHooks -ToolName "git" -Arguments @("status")
#>
function Invoke-ToolWithHooks {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$ToolName,

        [Parameter()]
        [hashtable]$Arguments = @{},

        [Parameter()]
        [switch]$SkipSecurity
    )

    $auditEntry = @{
        Timestamp = Get-Date -Format "o"
        ToolName = $ToolName
        Arguments = $Arguments
        Profile = if ($global:ProfileConfig) { $global:ProfileConfig.ProfileName } else { "none" }
        SecurityScan = $null
        Result = $null
        Duration = $null
    }

    $startTime = Get-Date

    try {
        # Security scan
        if (-not $SkipSecurity) {
            $securityResult = Test-CommandSecurity -Command $ToolName -Arguments $Arguments
            $auditEntry.SecurityScan = $securityResult

            if ($securityResult.Dangerous) {
                Write-Warning "SECURITY WARNING: Dangerous command detected!"
                Write-Warning "Command: $ToolName"
                Write-Warning "Reason: $($securityResult.Reason)"
                
                $confirmation = Read-Host "Proceed anyway? (yes/no)"
                if ($confirmation -ne "yes") {
                    $auditEntry.Result = "Blocked by security scan"
                    Write-ToolAuditEntry -Entry $auditEntry
                    return $null
                }
            }
        }

        # Execute pre-hook
        if ($global:ToolHooks -and $global:ToolHooks.ContainsKey($ToolName)) {
            $hook = $global:ToolHooks[$ToolName]
            if ($hook.PreHook) {
                & $hook.PreHook $ToolName $Arguments
            }
        }

        # Inject profile context if available
        if ($global:ProfileConfig) {
            if (-not $Arguments.ContainsKey("WorkingDirectory") -and 
                $global:ProfileConfig.ContainsKey("WorkingDirectory")) {
                $Arguments["WorkingDirectory"] = $global:ProfileConfig.WorkingDirectory
            }
        }

        # Execute the actual tool
        Write-Verbose "Executing tool: $ToolName"
        $result = & $ToolName @Arguments

        $auditEntry.Result = "Success"
        $auditEntry.Duration = (Get-Date) - $startTime

        # Execute post-hook
        if ($global:ToolHooks -and $global:ToolHooks.ContainsKey($ToolName)) {
            $hook = $global:ToolHooks[$ToolName]
            if ($hook.PostHook) {
                & $hook.PostHook $ToolName $Arguments $result
            }
        }

        Write-ToolAuditEntry -Entry $auditEntry
        return $result
    }
    catch {
        $auditEntry.Result = "Error: $_"
        $auditEntry.Duration = (Get-Date) - $startTime
        Write-ToolAuditEntry -Entry $auditEntry
        throw
    }
}

<#
.SYNOPSIS
    Tests command for security issues.

.DESCRIPTION
    Scans command and arguments for known dangerous patterns
    like recursive deletes, format commands, etc.

.PARAMETER Command
    Command to test.

.PARAMETER Arguments
    Command arguments.

.EXAMPLE
    Test-CommandSecurity -Command "rm" -Arguments @{Path = "/"; Recurse = $true}
#>
function Test-CommandSecurity {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,

        [Parameter()]
        [hashtable]$Arguments = @{}
    )

    $result = @{
        Dangerous = $false
        Reason = $null
        Patterns = @()
    }

    # Build full command string
    $fullCommand = $Command + " " + ($Arguments.GetEnumerator() | ForEach-Object {
        if ($_.Value -is [switch] -and $_.Value) {
            "-$($_.Key)"
        } elseif ($_.Value) {
            "-$($_.Key) $($_.Value)"
        }
    }) -join " "

    # Check against dangerous patterns
    foreach ($pattern in $script:DangerousPatterns) {
        if ($fullCommand -match $pattern) {
            $result.Dangerous = $true
            $result.Patterns += $pattern
            $result.Reason = "Matches dangerous pattern: $pattern"
            break
        }
    }

    # Check specific dangerous scenarios
    if ($Command -match '^(Remove-Item|rm|del)$' -and 
        $Arguments.Recurse -and 
        $Arguments.Force -and
        $Arguments.Path -match '^[/\\]$|^C:\\$') {
        $result.Dangerous = $true
        $result.Reason = "Recursive forced deletion of root directory"
    }

    return $result
}

<#
.SYNOPSIS
    Gets tool usage audit log.

.DESCRIPTION
    Retrieves recent tool invocation audit entries with optional filtering.

.PARAMETER Last
    Number of recent entries to return (default: 50).

.PARAMETER ToolName
    Filter by specific tool name.

.PARAMETER Profile
    Filter by profile name.

.PARAMETER Since
    Filter by entries after this timestamp.

.EXAMPLE
    Get-ToolAuditLog -Last 10

.EXAMPLE
    Get-ToolAuditLog -ToolName "git" -Profile "gemma-dev"
#>
function Get-ToolAuditLog {
    [CmdletBinding()]
    param(
        [Parameter()]
        [int]$Last = 50,

        [Parameter()]
        [string]$ToolName,

        [Parameter()]
        [string]$Profile,

        [Parameter()]
        [datetime]$Since
    )

    # Load audit log from file
    if (Test-Path $script:ToolAuditLogPath) {
        $entries = Get-Content $script:ToolAuditLogPath | 
            ForEach-Object { $_ | ConvertFrom-Json }
    } else {
        $entries = @()
    }

    # Apply filters
    if ($ToolName) {
        $entries = $entries | Where-Object { $_.ToolName -eq $ToolName }
    }

    if ($Profile) {
        $entries = $entries | Where-Object { $_.Profile -eq $Profile }
    }

    if ($Since) {
        $entries = $entries | Where-Object { 
            [datetime]::Parse($_.Timestamp) -gt $Since 
        }
    }

    # Return last N entries
    return $entries | Select-Object -Last $Last
}

<#
.SYNOPSIS
    Clears tool audit log.

.DESCRIPTION
    Removes audit log entries with optional filtering.

.PARAMETER All
    Clear all entries.

.PARAMETER Before
    Clear entries before this timestamp.

.PARAMETER Confirm
    Confirm before clearing (default: prompt).

.EXAMPLE
    Clear-ToolAuditLog -All -Confirm:$false

.EXAMPLE
    Clear-ToolAuditLog -Before (Get-Date).AddDays(-30)
#>
function Clear-ToolAuditLog {
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [Parameter()]
        [switch]$All,

        [Parameter()]
        [datetime]$Before
    )

    if (-not (Test-Path $script:ToolAuditLogPath)) {
        Write-Host "No audit log exists" -ForegroundColor Yellow
        return
    }

    if ($All) {
        if ($PSCmdlet.ShouldProcess("all entries", "Clear tool audit log")) {
            Remove-Item -Path $script:ToolAuditLogPath -Force
            Write-Host "✓ Cleared all tool audit entries" -ForegroundColor Green
        }
    }
    elseif ($Before) {
        $entries = Get-Content $script:ToolAuditLogPath | 
            ForEach-Object { $_ | ConvertFrom-Json }

        $kept = $entries | Where-Object { 
            [datetime]::Parse($_.Timestamp) -ge $Before 
        }

        if ($PSCmdlet.ShouldProcess("entries before $Before", "Clear tool audit log")) {
            $kept | ConvertTo-Json -Compress | 
                Set-Content -Path $script:ToolAuditLogPath -Force
            
            $removedCount = $entries.Count - $kept.Count
            Write-Host "✓ Removed $removedCount audit entries" -ForegroundColor Green
        }
    }
}

#endregion

#region Helper Functions

function Write-ToolAuditEntry {
    param([hashtable]$Entry)

    # Ensure audit directory exists
    $auditDir = Split-Path -Parent $script:ToolAuditLogPath
    if (-not (Test-Path $auditDir)) {
        New-Item -ItemType Directory -Path $auditDir -Force | Out-Null
    }

    # Append to JSONL file
    $Entry | ConvertTo-Json -Compress | 
        Add-Content -Path $script:ToolAuditLogPath -Encoding UTF8

    # Also keep in memory for this session
    $script:ToolAuditLog += $Entry
}

#endregion

Write-Host "✓ Tool use hooks with auditing loaded" -ForegroundColor Cyan
