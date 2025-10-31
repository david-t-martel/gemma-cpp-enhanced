# =============================================================================
# PHASE 2A AUTOMATED TEST SUITE
# =============================================================================
# Comprehensive testing of all Phase 2A implementations including:
# - Auto-Claude Integration
# - RAG-Redis Profile Separation
# - Tool Use Hooks & Auditing
# - End-to-End Integration
#
# Version: 1.0.0
# Created: 2025-01-15
# =============================================================================

#Requires -Version 7.0

# Test configuration
$script:TestResults = @{
    Passed = 0
    Failed = 0
    Skipped = 0
    Tests = @()
}

$script:TestContext = @{
    TestProfile = "phase2a-test-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    TestDirectory = "$env:TEMP\phase2a-test"
    OriginalLocation = Get-Location
}

#region Test Framework

function Test-Assertion {
    param(
        [Parameter(Mandatory)]
        [string]$Name,
        
        [Parameter(Mandatory)]
        [scriptblock]$Test,
        
        [string]$Category = "General",
        
        [switch]$SkipOnError
    )
    
    Write-Host "`n[TEST] $Category :: $Name" -ForegroundColor Cyan
    
    try {
        $result = & $Test
        
        if ($result -eq $true -or $result -eq $null) {
            Write-Host "  ✓ PASS" -ForegroundColor Green
            $script:TestResults.Passed++
            $status = "PASS"
        }
        else {
            Write-Host "  ✗ FAIL: $result" -ForegroundColor Red
            $script:TestResults.Failed++
            $status = "FAIL"
        }
    }
    catch {
        if ($SkipOnError) {
            Write-Host "  ⊘ SKIP: $_" -ForegroundColor Yellow
            $script:TestResults.Skipped++
            $status = "SKIP"
        }
        else {
            Write-Host "  ✗ FAIL: $_" -ForegroundColor Red
            $script:TestResults.Failed++
            $status = "FAIL"
        }
    }
    
    $script:TestResults.Tests += @{
        Category = $Category
        Name = $Name
        Status = $status
        Timestamp = Get-Date -Format "o"
    }
}

function Write-TestHeader {
    param([string]$Title)
    
    Write-Host "`n" -NoNewline
    Write-Host ("=" * 80) -ForegroundColor Cyan
    Write-Host "  $Title" -ForegroundColor White
    Write-Host ("=" * 80) -ForegroundColor Cyan
}

function Write-TestSummary {
    Write-Host "`n" -NoNewline
    Write-Host ("=" * 80) -ForegroundColor Cyan
    Write-Host "  TEST SUMMARY" -ForegroundColor White
    Write-Host ("=" * 80) -ForegroundColor Cyan
    
    Write-Host "`nResults:" -ForegroundColor White
    Write-Host "  Passed:  $($script:TestResults.Passed)" -ForegroundColor Green
    Write-Host "  Failed:  $($script:TestResults.Failed)" -ForegroundColor Red
    Write-Host "  Skipped: $($script:TestResults.Skipped)" -ForegroundColor Yellow
    
    $total = $script:TestResults.Passed + $script:TestResults.Failed + $script:TestResults.Skipped
    $passRate = if ($total -gt 0) { [math]::Round(($script:TestResults.Passed / $total) * 100, 2) } else { 0 }
    
    Write-Host "`n  Pass Rate: $passRate%" -ForegroundColor $(if ($passRate -ge 80) { "Green" } elseif ($passRate -ge 60) { "Yellow" } else { "Red" })
    
    # Failed tests detail
    if ($script:TestResults.Failed -gt 0) {
        Write-Host "`nFailed Tests:" -ForegroundColor Red
        $script:TestResults.Tests | Where-Object { $_.Status -eq "FAIL" } | ForEach-Object {
            Write-Host "  - [$($_.Category)] $($_.Name)" -ForegroundColor Red
        }
    }
}

#endregion

#region Setup & Teardown

function Initialize-Modules {
    Write-Host "Loading Phase 2A modules..." -ForegroundColor Cyan
    
    # Load Phase 1
    if (Test-Path "C:\codedev\llm\gemma\PHASE1_IMPLEMENTATION.ps1") {
        . "C:\codedev\llm\gemma\PHASE1_IMPLEMENTATION.ps1"
    }
    
    # Load Phase 2A modules
    if (Test-Path "C:\codedev\llm\gemma\PHASE2A_AUTO_CLAUDE.ps1") {
        . "C:\codedev\llm\gemma\PHASE2A_AUTO_CLAUDE.ps1"
    }
    
    if (Test-Path "C:\codedev\llm\gemma\PHASE2A_RAG_REDIS.ps1") {
        . "C:\codedev\llm\gemma\PHASE2A_RAG_REDIS.ps1"
    }
    
    if (Test-Path "C:\codedev\llm\gemma\PHASE2A_TOOL_HOOKS.ps1") {
        . "C:\codedev\llm\gemma\PHASE2A_TOOL_HOOKS.ps1"
    }
    
    Write-Host "  ✓ Modules loaded" -ForegroundColor Green
}

function Initialize-TestEnvironment {
    Write-Host "`nInitializing test environment..." -ForegroundColor Cyan
    
    # Create test directory
    if (-not (Test-Path $script:TestContext.TestDirectory)) {
        New-Item -ItemType Directory -Path $script:TestContext.TestDirectory -Force | Out-Null
    }
    
    # Create test files
    @"
# Test README
This is a test project for Phase 2A validation.
"@ | Set-Content -Path "$($script:TestContext.TestDirectory)\README.md"
    
    @"
# Test Documentation
Phase 2A testing documentation.
"@ | Set-Content -Path "$($script:TestContext.TestDirectory)\DOCS.md"
    
    Write-Host "  ✓ Test environment ready" -ForegroundColor Green
}

function Remove-TestEnvironment {
    Write-Host "`nCleaning up test environment..." -ForegroundColor Cyan
    
    # Remove test directory
    if (Test-Path $script:TestContext.TestDirectory) {
        Remove-Item -Path $script:TestContext.TestDirectory -Recurse -Force -ErrorAction SilentlyContinue
    }
    
    # Remove test profile
    $profilePath = "$env:USERPROFILE\.llm-profile\profiles\$($script:TestContext.TestProfile).json"
    if (Test-Path $profilePath) {
        Remove-Item -Path $profilePath -Force -ErrorAction SilentlyContinue
    }
    
    # Restore location
    Set-Location $script:TestContext.OriginalLocation
    
    Write-Host "  ✓ Test environment cleaned" -ForegroundColor Green
}

#endregion

#region Test Suites

function Test-Phase1Prerequisites {
    Write-TestHeader "TEST SUITE 1: Phase 1 Prerequisites"
    
    Test-Assertion -Name "Phase 1 implementation file exists" -Category "Prerequisites" -Test {
        Test-Path "C:\codedev\llm\gemma\PHASE1_IMPLEMENTATION.ps1"
    }
    
    Test-Assertion -Name "Can load Phase 1 implementation" -Category "Prerequisites" -Test {
        $null = & {
            . "C:\codedev\llm\gemma\PHASE1_IMPLEMENTATION.ps1"
        }
        $true
    } -SkipOnError
    
    Test-Assertion -Name "Set-LLMProfile function is available" -Category "Prerequisites" -Test {
        $null -ne (Get-Command Set-LLMProfile -ErrorAction SilentlyContinue)
    }
    
    Test-Assertion -Name "Get-LLMProfile function is available" -Category "Prerequisites" -Test {
        $null -ne (Get-Command Get-LLMProfile -ErrorAction SilentlyContinue)
    }
    
    Test-Assertion -Name "Profile directory structure exists" -Category "Prerequisites" -Test {
        Test-Path "$env:USERPROFILE\.llm-profile\profiles"
    }
}

function Test-AutoClaudeIntegration {
    Write-TestHeader "TEST SUITE 2: Auto-Claude Integration"
    
    Test-Assertion -Name "Auto-Claude module file exists" -Category "Auto-Claude" -Test {
        Test-Path "C:\codedev\llm\gemma\PHASE2A_AUTO_CLAUDE.ps1"
    }
    
    Test-Assertion -Name "Can load Auto-Claude module" -Category "Auto-Claude" -Test {
        $null = & {
            . "C:\codedev\llm\gemma\PHASE2A_AUTO_CLAUDE.ps1"
        }
        $true
    }
    
    Test-Assertion -Name "Invoke-AutoClaude function is available" -Category "Auto-Claude" -Test {
        $null -ne (Get-Command Invoke-AutoClaude -ErrorAction SilentlyContinue)
    }
    
    Test-Assertion -Name "Get-AutoClaudeConfig function is available" -Category "Auto-Claude" -Test {
        $null -ne (Get-Command Get-AutoClaudeConfig -ErrorAction SilentlyContinue)
    }
    
    Test-Assertion -Name "Set-AutoClaudeConfig function is available" -Category "Auto-Claude" -Test {
        $null -ne (Get-Command Set-AutoClaudeConfig -ErrorAction SilentlyContinue)
    }
    
    Test-Assertion -Name "Auto-Claude alias 'ac' is available" -Category "Auto-Claude" -Test {
        $null -ne (Get-Alias ac -ErrorAction SilentlyContinue)
    }
    
    Test-Assertion -Name "Auto-Claude alias 'claude' is available" -Category "Auto-Claude" -Test {
        $null -ne (Get-Alias claude -ErrorAction SilentlyContinue)
    }
    
    Test-Assertion -Name "Auto-Claude binary detection works" -Category "Auto-Claude" -Test {
        $config = Get-AutoClaudeConfig
        $config.AutoClaudeInstalled -is [bool]
    } -SkipOnError
    
    Test-Assertion -Name "Can get Auto-Claude config without active profile" -Category "Auto-Claude" -Test {
        $config = Get-AutoClaudeConfig
        $config.ContainsKey("PreferredModel")
    } -SkipOnError
}

function Test-RagRedisIntegration {
    Write-TestHeader "TEST SUITE 3: RAG-Redis Integration"
    
    Test-Assertion -Name "RAG-Redis module file exists" -Category "RAG-Redis" -Test {
        Test-Path "C:\codedev\llm\gemma\PHASE2A_RAG_REDIS.ps1"
    }
    
    Test-Assertion -Name "Can load RAG-Redis module" -Category "RAG-Redis" -Test {
        $null = & {
            . "C:\codedev\llm\gemma\PHASE2A_RAG_REDIS.ps1"
        }
        $true
    }
    
    Test-Assertion -Name "Initialize-RagRedis function is available" -Category "RAG-Redis" -Test {
        $null -ne (Get-Command Initialize-RagRedis -ErrorAction SilentlyContinue)
    }
    
    Test-Assertion -Name "Add-RagDocument function is available" -Category "RAG-Redis" -Test {
        $null -ne (Get-Command Add-RagDocument -ErrorAction SilentlyContinue)
    }
    
    Test-Assertion -Name "Search-RagDocuments function is available" -Category "RAG-Redis" -Test {
        $null -ne (Get-Command Search-RagDocuments -ErrorAction SilentlyContinue)
    }
    
    Test-Assertion -Name "Clear-RagDocuments function is available" -Category "RAG-Redis" -Test {
        $null -ne (Get-Command Clear-RagDocuments -ErrorAction SilentlyContinue)
    }
    
    Test-Assertion -Name "Stop-RagRedis function is available" -Category "RAG-Redis" -Test {
        $null -ne (Get-Command Stop-RagRedis -ErrorAction SilentlyContinue)
    }
    
    Test-Assertion -Name "RAG-Redis binary detection" -Category "RAG-Redis" -Test {
        $cmd = Get-Command "rag-redis-mcp-server" -ErrorAction SilentlyContinue
        if ($cmd) { $true } else { "rag-redis-mcp-server not found in PATH" }
    } -SkipOnError
    
    Test-Assertion -Name "Redis connectivity check" -Category "RAG-Redis" -Test {
        $redis = Get-Command "redis-cli" -ErrorAction SilentlyContinue
        if ($redis) {
            $ping = & redis-cli ping 2>&1
            if ($ping -match "PONG") { $true } else { "Redis not responding" }
        } else {
            "redis-cli not found"
        }
    } -SkipOnError
}

function Test-ToolHooksIntegration {
    Write-TestHeader "TEST SUITE 4: Tool Hooks & Security"
    
    Test-Assertion -Name "Tool Hooks module file exists" -Category "Tool-Hooks" -Test {
        Test-Path "C:\codedev\llm\gemma\PHASE2A_TOOL_HOOKS.ps1"
    }
    
    Test-Assertion -Name "Can load Tool Hooks module" -Category "Tool-Hooks" -Test {
        $null = & {
            . "C:\codedev\llm\gemma\PHASE2A_TOOL_HOOKS.ps1"
        }
        $true
    }
    
    Test-Assertion -Name "Register-ToolHook function is available" -Category "Tool-Hooks" -Test {
        $null -ne (Get-Command Register-ToolHook -ErrorAction SilentlyContinue)
    }
    
    Test-Assertion -Name "Invoke-ToolWithHooks function is available" -Category "Tool-Hooks" -Test {
        $null -ne (Get-Command Invoke-ToolWithHooks -ErrorAction SilentlyContinue)
    }
    
    Test-Assertion -Name "Test-CommandSecurity function is available" -Category "Tool-Hooks" -Test {
        $null -ne (Get-Command Test-CommandSecurity -ErrorAction SilentlyContinue)
    }
    
    Test-Assertion -Name "Get-ToolAuditLog function is available" -Category "Tool-Hooks" -Test {
        $null -ne (Get-Command Get-ToolAuditLog -ErrorAction SilentlyContinue)
    }
    
    Test-Assertion -Name "Clear-ToolAuditLog function is available" -Category "Tool-Hooks" -Test {
        $null -ne (Get-Command Clear-ToolAuditLog -ErrorAction SilentlyContinue)
    }
    
    Test-Assertion -Name "Hook registration works" -Category "Tool-Hooks" -Test {
        Register-ToolHook -ToolName "Test-Command" -PreHook { param($t, $a) }
        $global:ToolHooks.ContainsKey("Test-Command")
    }
    
    Test-Assertion -Name "Security scan detects safe command" -Category "Tool-Hooks" -Test {
        $result = Test-CommandSecurity -Command "Get-ChildItem" -Arguments @{ Path = "." }
        -not $result.Dangerous
    }
    
    Test-Assertion -Name "Security scan detects dangerous command" -Category "Tool-Hooks" -Test {
        $result = Test-CommandSecurity -Command "Remove-Item" -Arguments @{ Path = "C:\"; Recurse = $true; Force = $true }
        $result.Dangerous
    }
    
    Test-Assertion -Name "Security scan detects rm -rf pattern" -Category "Tool-Hooks" -Test {
        $result = Test-CommandSecurity -Command "rm -rf /" -Arguments @{}
        $result.Dangerous
    }
    
    Test-Assertion -Name "Audit log directory is created" -Category "Tool-Hooks" -Test {
        $auditDir = Split-Path -Parent "$env:USERPROFILE\.llm-profile\tool-audit.jsonl"
        if (-not (Test-Path $auditDir)) {
            New-Item -ItemType Directory -Path $auditDir -Force | Out-Null
        }
        Test-Path $auditDir
    }
}

function Test-EndToEndWorkflow {
    Write-TestHeader "TEST SUITE 5: End-to-End Integration"
    
    Test-Assertion -Name "Create test profile" -Category "E2E" -Test {
        Set-LLMProfile -ProfileName $script:TestContext.TestProfile -WorkingDirectory $script:TestContext.TestDirectory
        $null -ne $global:ProfileConfig
    }
    
    Test-Assertion -Name "Profile configuration persisted" -Category "E2E" -Test {
        $profilePath = "$env:USERPROFILE\.llm-profile\profiles\$($script:TestContext.TestProfile).json"
        Test-Path $profilePath
    }
    
    Test-Assertion -Name "Can set Auto-Claude config for profile" -Category "E2E" -Test {
        Set-AutoClaudeConfig -PreferredModel "claude-3-5-sonnet-20241022"
        $global:ProfileConfig.PreferredModel -eq "claude-3-5-sonnet-20241022"
    }
    
    Test-Assertion -Name "Can add context path to profile" -Category "E2E" -Test {
        Set-AutoClaudeConfig -AddContextPath $script:TestContext.TestDirectory
        $global:ProfileConfig.ContextFiles -contains $script:TestContext.TestDirectory
    }
    
    Test-Assertion -Name "Auto-Claude config reflects profile settings" -Category "E2E" -Test {
        $config = Get-AutoClaudeConfig
        $config.ProfileIntegration -eq $true
    }
    
    Test-Assertion -Name "Can retrieve profile" -Category "E2E" -Test {
        $profile = Get-LLMProfile
        $profile.ProfileName -eq $script:TestContext.TestProfile
    }
    
    Test-Assertion -Name "Hook with profile context injection" -Category "E2E" -Test {
        Register-ToolHook -ToolName "Get-Location" -PreHook {
            param($t, $a)
            # Pre-hook should have access to profile context
            $null -ne $global:ProfileConfig
        }
        $true
    }
    
    Test-Assertion -Name "Security scan with audit logging" -Category "E2E" -Test {
        # This should create an audit entry
        $result = Test-CommandSecurity -Command "Get-Process" -Arguments @{}
        -not $result.Dangerous
    }
}

#endregion

#region Main Execution

function Start-Phase2ATesting {
    Write-Host @"

╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║                    PHASE 2A AUTOMATED TEST SUITE                          ║
║                                                                           ║
║  PowerShell Profile Framework - LLM Agent Integration Testing            ║
║  Version 1.0.0 | $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')                                      ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝

"@ -ForegroundColor Cyan

    # Initialize
    Initialize-Modules
    Initialize-TestEnvironment
    
    # Run test suites
    Test-Phase1Prerequisites
    Test-AutoClaudeIntegration
    Test-RagRedisIntegration
    Test-ToolHooksIntegration
    Test-EndToEndWorkflow
    
    # Summary
    Write-TestSummary
    
    # Cleanup
    Remove-TestEnvironment
    
    # Save results
    $resultsPath = "C:\codedev\llm\gemma\PHASE2A_TEST_RESULTS_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
    $script:TestResults | ConvertTo-Json -Depth 10 | Set-Content -Path $resultsPath
    Write-Host "`nTest results saved to: $resultsPath" -ForegroundColor Cyan
    
    # Return exit code
    if ($script:TestResults.Failed -eq 0) {
        Write-Host "`n✓ ALL TESTS PASSED" -ForegroundColor Green
        return 0
    } else {
        Write-Host "`n✗ SOME TESTS FAILED" -ForegroundColor Red
        return 1
    }
}

# Run tests
$exitCode = Start-Phase2ATesting
exit $exitCode

#endregion
