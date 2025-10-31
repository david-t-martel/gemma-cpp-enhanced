# Phase 2B Core Profile Integration
# Add this section to core_profile.ps1 after the GCP Profile Management section (around line 710)

#region Phase 2B - LLM Profile System Integration

# --- Phase 2B: RAG-Redis MCP Integration ---
Write-Debug '[core_profile] Initializing Phase 2B RAG-Redis MCP Integration (Lazy Load)'

$script:Phase2BLoaded = $false
$script:Phase2BModulesPath = "C:\codedev\llm\gemma"

# Lazy load Phase 2B modules
function global:Initialize-Phase2B {
    <#
    .SYNOPSIS
        Lazy load Phase 2B modules and tools
    
    .DESCRIPTION
        Loads Phase 1, Phase 2A, and Phase 2B modules on demand
    #>
    if ($script:Phase2BLoaded) { return $true }
    
    Write-Debug '[core_profile] Loading Phase 2B modules...'
    
    try {
        # Load modules in order
        $modules = @(
            'PHASE1_IMPLEMENTATION.ps1',
            'PHASE2A_AUTO_CLAUDE.ps1',
            'PHASE2A_RAG_REDIS.ps1',
            'PHASE2A_TOOL_HOOKS.ps1',
            'PHASE2B_RAG_INTEGRATION.ps1'
        )
        
        foreach ($module in $modules) {
            $modulePath = Join-Path $script:Phase2BModulesPath $module
            if (Test-Path $modulePath) {
                . $modulePath
                Write-Debug "[core_profile] Loaded Phase 2B module: $module"
            } else {
                Write-Warning "[core_profile] Phase 2B module not found: $module"
                return $false
            }
        }
        
        $script:Phase2BLoaded = $true
        Write-Verbose 'Phase 2B modules loaded successfully'
        return $true
    } catch {
        Write-Error "Failed to load Phase 2B modules: $_"
        return $false
    }
}

# Create lazy-loaded wrapper functions for Phase 2B tools
function global:Test-Phase2BInstallation {
    <#
    .SYNOPSIS
        Verify Phase 2B installation
    
    .DESCRIPTION
        Quick validation of Phase 2B components
        
    .PARAMETER Detailed
        Show detailed information for each check
    
    .EXAMPLE
        Test-Phase2BInstallation
        
    .EXAMPLE
        Test-Phase2BInstallation -Detailed
    #>
    param([switch]$Detailed)
    
    # Load verification script and run test
    $verifyScript = Join-Path $script:Phase2BModulesPath "PHASE2B_VERIFY.ps1"
    if (Test-Path $verifyScript) {
        . $verifyScript
        Test-Phase2BInstallation @PSBoundParameters
    } else {
        Write-Error "Phase 2B verification script not found: $verifyScript"
    }
}

function global:Invoke-Phase2BTests {
    <#
    .SYNOPSIS
        Run comprehensive Phase 2B test suite
    
    .DESCRIPTION
        Executes all tests for Phase 2B RAG-Redis MCP integration
    
    .PARAMETER SkipPrerequisites
        Skip prerequisite checks
    
    .PARAMETER SkipServerTests
        Skip MCP server tests
    
    .PARAMETER SkipIntegration
        Skip integration tests
    
    .PARAMETER ExportReport
        Export test results to JSON
    
    .EXAMPLE
        Invoke-Phase2BTests
        
    .EXAMPLE
        Invoke-Phase2BTests -SkipPrerequisites -ExportReport
    #>
    [CmdletBinding()]
    param(
        [switch]$SkipPrerequisites,
        [switch]$SkipServerTests,
        [switch]$SkipIntegration,
        [switch]$SkipCleanup,
        [switch]$ExportReport
    )
    
    if (-not (Initialize-Phase2B)) {
        Write-Error "Failed to initialize Phase 2B"
        return
    }
    
    $testScript = Join-Path $script:Phase2BModulesPath "PHASE2B_TESTS.ps1"
    if (Test-Path $testScript) {
        . $testScript
        Invoke-Phase2BTests @PSBoundParameters
    } else {
        Write-Error "Phase 2B test script not found: $testScript"
    }
}

function global:Invoke-Phase2BDeploy {
    <#
    .SYNOPSIS
        Deploy Phase 2B RAG-Redis MCP Integration
    
    .DESCRIPTION
        Automated deployment of Phase 2B components including:
        - Compilation of MCP server
        - Binary installation
        - Configuration updates
        - Integration testing
    
    .PARAMETER SkipBackup
        Skip creating backup of existing installation
    
    .PARAMETER SkipTests
        Skip running tests after deployment
    
    .PARAMETER SkipBuild
        Skip building (use existing binary)
    
    .PARAMETER ForceRollback
        Force rollback to previous version
    
    .EXAMPLE
        Invoke-Phase2BDeploy
        
    .EXAMPLE
        Invoke-Phase2BDeploy -SkipTests
    #>
    [CmdletBinding()]
    param(
        [switch]$SkipBackup,
        [switch]$SkipTests,
        [switch]$SkipBuild,
        [switch]$ForceRollback
    )
    
    $deployScript = Join-Path $script:Phase2BModulesPath "PHASE2B_DEPLOY.ps1"
    if (Test-Path $deployScript) {
        . $deployScript
        Invoke-Phase2BDeploy @PSBoundParameters
    } else {
        Write-Error "Phase 2B deployment script not found: $deployScript"
    }
}

# Create lazy-loaded LLM Profile commands
function global:Set-LLMProfile {
    <#
    .SYNOPSIS
        Create or switch to an LLM profile
    
    .DESCRIPTION
        Manages profile-specific context for LLM interactions
    
    .PARAMETER ProfileName
        Name of the profile to create or activate
    
    .PARAMETER WorkingDirectory
        Working directory for the profile
    
    .PARAMETER PreferredModel
        Preferred LLM model for this profile
    
    .EXAMPLE
        Set-LLMProfile -ProfileName "my-project" -WorkingDirectory C:\projects\myapp
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$ProfileName,
        
        [Parameter(Mandatory)]
        [string]$WorkingDirectory,
        
        [string]$PreferredModel
    )
    
    if (-not (Initialize-Phase2B)) {
        Write-Error "Failed to initialize Phase 2B"
        return
    }
    
    # Forward to Phase 2A implementation
    Microsoft.PowerShell.Core\Set-LLMProfile @PSBoundParameters
}

function global:Get-LLMProfile {
    <#
    .SYNOPSIS
        Get current or specific LLM profile
    
    .PARAMETER ProfileName
        Name of profile to retrieve (optional, returns current if omitted)
    
    .EXAMPLE
        Get-LLMProfile
        
    .EXAMPLE
        Get-LLMProfile -ProfileName "my-project"
    #>
    [CmdletBinding()]
    param([string]$ProfileName)
    
    if (-not (Initialize-Phase2B)) {
        Write-Error "Failed to initialize Phase 2B"
        return
    }
    
    Microsoft.PowerShell.Core\Get-LLMProfile @PSBoundParameters
}

function global:Initialize-RagRedisMcp {
    <#
    .SYNOPSIS
        Initialize RAG-Redis MCP for current profile
    
    .DESCRIPTION
        Configures RAG-Redis MCP server integration for the active LLM profile
    
    .PARAMETER RedisHost
        Redis server hostname (default: localhost)
    
    .PARAMETER RedisPort
        Redis server port (default: 6380)
    
    .EXAMPLE
        Initialize-RagRedisMcp
        
    .EXAMPLE
        Initialize-RagRedisMcp -RedisHost "localhost" -RedisPort 6380
    #>
    [CmdletBinding()]
    param(
        [string]$RedisHost = "localhost",
        [int]$RedisPort = 6380
    )
    
    if (-not (Initialize-Phase2B)) {
        Write-Error "Failed to initialize Phase 2B"
        return
    }
    
    Microsoft.PowerShell.Core\Initialize-RagRedisMcp @PSBoundParameters
}

# Convenience aliases for Phase 2B
Set-Alias -Name 'llm-profile' -Value Set-LLMProfile -Description 'Create/switch LLM profile'
Set-Alias -Name 'llm-test' -Value Test-Phase2BInstallation -Description 'Test Phase 2B installation'
Set-Alias -Name 'llm-deploy' -Value Invoke-Phase2BDeploy -Description 'Deploy Phase 2B'
Set-Alias -Name 'rag-init' -Value Initialize-RagRedisMcp -Description 'Initialize RAG-Redis MCP'

# Display Phase 2B status if available
function global:Get-Phase2BStatus {
    <#
    .SYNOPSIS
        Show Phase 2B integration status
    
    .DESCRIPTION
        Displays status of Phase 2B modules, MCP server, and current profile
    #>
    Write-Host "`nPhase 2B Status:" -ForegroundColor Cyan
    
    # Check if modules are loaded
    Write-Host "  Modules Loaded: " -NoNewline
    Write-Host $(if ($script:Phase2BLoaded) { "Yes" } else { "Not yet (lazy loaded)" }) -ForegroundColor $(if ($script:Phase2BLoaded) { "Green" } else { "Yellow" })
    
    # Check MCP server binary
    $mcpServer = "$env:USERPROFILE\.local\bin\rag-redis-mcp-server.exe"
    Write-Host "  MCP Server: " -NoNewline
    if (Test-Path $mcpServer) {
        $serverInfo = Get-Item $mcpServer
        Write-Host "Installed ($([math]::Round($serverInfo.Length / 1MB, 1)) MB)" -ForegroundColor Green
    } else {
        Write-Host "Not installed" -ForegroundColor Red
    }
    
    # Check current profile
    if ($script:Phase2BLoaded) {
        try {
            $currentProfile = Get-LLMProfile -ErrorAction SilentlyContinue
            if ($currentProfile) {
                Write-Host "  Active Profile: $($currentProfile.ProfileName)" -ForegroundColor Green
                Write-Host "    Working Dir: $($currentProfile.WorkingDirectory)" -ForegroundColor Gray
                if ($currentProfile.RagRedisMcp -and $currentProfile.RagRedisMcp.UseMcp) {
                    Write-Host "    RAG-Redis: Configured" -ForegroundColor Green
                } else {
                    Write-Host "    RAG-Redis: Not configured" -ForegroundColor Yellow
                }
            } else {
                Write-Host "  Active Profile: None" -ForegroundColor Yellow
            }
        } catch {
            Write-Host "  Active Profile: Unable to determine" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  Active Profile: Not loaded yet" -ForegroundColor Gray
    }
    
    Write-Host "`n  Quick Start:" -ForegroundColor Cyan
    Write-Host "    Test installation:  Test-Phase2BInstallation" -ForegroundColor White
    Write-Host "    Create profile:     Set-LLMProfile -ProfileName 'test' -WorkingDirectory \$PWD" -ForegroundColor White
    Write-Host "    Initialize RAG:     Initialize-RagRedisMcp" -ForegroundColor White
    Write-Host "    Run tests:          Invoke-Phase2BTests" -ForegroundColor White
    Write-Host ""
}

Set-Alias -Name 'llm-status' -Value Get-Phase2BStatus -Description 'Show Phase 2B status'

Write-Debug '[core_profile] Phase 2B integration initialized (lazy load ready)'

#endregion Phase 2B Integration

# To integrate into core_profile.ps1:
# 1. Copy this entire section
# 2. Paste after line 710 (after GCP Profile Management section)
# 3. The functions will be available immediately, modules load on first use
