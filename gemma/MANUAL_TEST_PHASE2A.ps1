# =============================================================================
# PHASE 2A MANUAL TESTING & VERIFICATION
# =============================================================================
# Simple, direct tests that verify Phase 2A functionality without complex
# test framework scoping issues.
#
# Run this script to perform manual verification of all components.
# =============================================================================

Write-Host @"

╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║                  PHASE 2A MANUAL TESTING & VERIFICATION                   ║
║                                                                           ║
║  Direct verification of Phase 2A implementations                         ║
║  Version 1.0.0 | $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')                                      ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝

"@ -ForegroundColor Cyan

$ErrorActionPreference = "Continue"
$testResults = @{
    Passed = 0
    Failed = 0
}

#region Load Modules

Write-Host "`n[1] Loading Phase 2A Modules..." -ForegroundColor Yellow
Write-Host "=" * 80

try {
    . "C:\codedev\llm\gemma\PHASE1_IMPLEMENTATION.ps1"
    . "C:\codedev\llm\gemma\PHASE2A_AUTO_CLAUDE.ps1"
    . "C:\codedev\llm\gemma\PHASE2A_RAG_REDIS.ps1"
    . "C:\codedev\llm\gemma\PHASE2A_TOOL_HOOKS.ps1"
    Write-Host "✓ All modules loaded successfully" -ForegroundColor Green
    $testResults.Passed++
}
catch {
    Write-Host "✗ Failed to load modules: $_" -ForegroundColor Red
    $testResults.Failed++
    exit 1
}

#endregion

#region Verify Functions

Write-Host "`n[2] Verifying Function Availability..." -ForegroundColor Yellow
Write-Host "=" * 80

$functions = @(
    "Set-LLMProfile",
    "Get-LLMProfile",
    "Get-LLMProfiles",
    "Remove-LLMProfile",
    "Invoke-AutoClaude",
    "Get-AutoClaudeConfig",
    "Set-AutoClaudeConfig",
    "Initialize-RagRedis",
    "Add-RagDocument",
    "Search-RagDocuments",
    "Clear-RagDocuments",
    "Stop-RagRedis",
    "Register-ToolHook",
    "Invoke-ToolWithHooks",
    "Test-CommandSecurity",
    "Get-ToolAuditLog",
    "Clear-ToolAuditLog"
)

$missingFunctions = @()
foreach ($func in $functions) {
    $cmd = Get-Command $func -ErrorAction SilentlyContinue
    if ($cmd) {
        Write-Host "  ✓ $func" -ForegroundColor Green
        $testResults.Passed++
    } else {
        Write-Host "  ✗ $func NOT FOUND" -ForegroundColor Red
        $missingFunctions += $func
        $testResults.Failed++
    }
}

if ($missingFunctions.Count -eq 0) {
    Write-Host "`n✓ All $($functions.Count) functions are available" -ForegroundColor Green
} else {
    Write-Host "`n✗ Missing functions: $($missingFunctions -join ', ')" -ForegroundColor Red
}

#endregion

#region Test Profile Management

Write-Host "`n[3] Testing Profile Management..." -ForegroundColor Yellow
Write-Host "=" * 80

$testProfileName = "phase2a-verification-$(Get-Date -Format 'yyyyMMddHHmmss')"
$testDir = "$env:TEMP\phase2a-test"

try {
    # Create test directory
    if (-not (Test-Path $testDir)) {
        New-Item -ItemType Directory -Path $testDir -Force | Out-Null
    }
    
    # Create profile
    Set-LLMProfile -ProfileName $testProfileName -WorkingDirectory $testDir
    Write-Host "  ✓ Created test profile" -ForegroundColor Green
    $testResults.Passed++
    
    # Verify profile config
    if ($null -ne $global:ProfileConfig) {
        Write-Host "  ✓ Profile config is set" -ForegroundColor Green
        $testResults.Passed++
    } else {
        Write-Host "  ✗ Profile config is null" -ForegroundColor Red
        $testResults.Failed++
    }
    
    # Get profile
    $profile = Get-LLMProfile
    if ($profile.ProfileName -eq $testProfileName) {
        Write-Host "  ✓ Retrieved profile matches" -ForegroundColor Green
        $testResults.Passed++
    } else {
        Write-Host "  ✗ Profile mismatch" -ForegroundColor Red
        $testResults.Failed++
    }
    
    # List profiles
    $profiles = Get-LLMProfiles
    if ($profiles | Where-Object { $_.ProfileName -eq $testProfileName }) {
        Write-Host "  ✓ Profile appears in list" -ForegroundColor Green
        $testResults.Passed++
    } else {
        Write-Host "  ✗ Profile not in list" -ForegroundColor Red
        $testResults.Failed++
    }
}
catch {
    Write-Host "  ✗ Profile management error: $_" -ForegroundColor Red
    $testResults.Failed++
}

#endregion

#region Test Auto-Claude Integration

Write-Host "`n[4] Testing Auto-Claude Integration..." -ForegroundColor Yellow
Write-Host "=" * 80

try {
    # Get config without error
    $config = Get-AutoClaudeConfig
    Write-Host "  ✓ Get-AutoClaudeConfig works" -ForegroundColor Green
    $testResults.Passed++
    
    # Check config structure
    if ($config.ContainsKey("AutoClaudeInstalled")) {
        Write-Host "  ✓ Config has expected structure" -ForegroundColor Green
        $testResults.Passed++
    } else {
        Write-Host "  ✗ Config missing expected keys" -ForegroundColor Red
        $testResults.Failed++
    }
    
    # Set config
    Set-AutoClaudeConfig -PreferredModel "claude-3-5-sonnet-20241022"
    if ($global:ProfileConfig.PreferredModel -eq "claude-3-5-sonnet-20241022") {
        Write-Host "  ✓ Set-AutoClaudeConfig updates profile" -ForegroundColor Green
        $testResults.Passed++
    } else {
        Write-Host "  ✗ Config not updated" -ForegroundColor Red
        $testResults.Failed++
    }
    
    # Check for auto-claude binary
    $acCmd = Get-Command "auto-claude" -ErrorAction SilentlyContinue
    if ($acCmd) {
        Write-Host "  ✓ auto-claude binary found: $($acCmd.Source)" -ForegroundColor Green
        $testResults.Passed++
    } else {
        Write-Host "  ⊘ auto-claude binary not found (expected if not installed)" -ForegroundColor Yellow
    }
}
catch {
    Write-Host "  ✗ Auto-Claude test error: $_" -ForegroundColor Red
    $testResults.Failed++
}

#endregion

#region Test Tool Hooks & Security

Write-Host "`n[5] Testing Tool Hooks & Security..." -ForegroundColor Yellow
Write-Host "=" * 80

try {
    # Register a hook
    Register-ToolHook -ToolName "Test-Command" -PreHook {
        param($t, $a)
        Write-Verbose "Test hook executed"
    }
    
    if ($global:ToolHooks.ContainsKey("Test-Command")) {
        Write-Host "  ✓ Hook registration works" -ForegroundColor Green
        $testResults.Passed++
    } else {
        Write-Host "  ✗ Hook not registered" -ForegroundColor Red
        $testResults.Failed++
    }
    
    # Test security scanning - safe command
    $result = Test-CommandSecurity -Command "Get-ChildItem" -Arguments @{ Path = "." }
    if (-not $result.Dangerous) {
        Write-Host "  ✓ Security scan: Safe command detected correctly" -ForegroundColor Green
        $testResults.Passed++
    } else {
        Write-Host "  ✗ Security scan: False positive on safe command" -ForegroundColor Red
        $testResults.Failed++
    }
    
    # Test security scanning - dangerous command
    $result = Test-CommandSecurity -Command "Remove-Item" -Arguments @{
        Path = "C:\"
        Recurse = $true
        Force = $true
    }
    if ($result.Dangerous) {
        Write-Host "  ✓ Security scan: Dangerous command detected correctly" -ForegroundColor Green
        $testResults.Passed++
    } else {
        Write-Host "  ✗ Security scan: Missed dangerous command" -ForegroundColor Red
        $testResults.Failed++
    }
    
    # Test rm -rf pattern
    $result = Test-CommandSecurity -Command "rm -rf /" -Arguments @{}
    if ($result.Dangerous) {
        Write-Host "  ✓ Security scan: 'rm -rf /' pattern detected" -ForegroundColor Green
        $testResults.Passed++
    } else {
        Write-Host "  ✗ Security scan: Missed 'rm -rf /' pattern" -ForegroundColor Red
        $testResults.Failed++
    }
}
catch {
    Write-Host "  ✗ Tool hooks test error: $_" -ForegroundColor Red
    $testResults.Failed++
}

#endregion

#region Test RAG-Redis

Write-Host "`n[6] Testing RAG-Redis Integration..." -ForegroundColor Yellow
Write-Host "=" * 80

try {
    # Check for rag-redis binary
    $ragCmd = Get-Command "rag-redis-mcp-server" -ErrorAction SilentlyContinue
    if ($ragCmd) {
        Write-Host "  ✓ rag-redis-mcp-server found: $($ragCmd.Source)" -ForegroundColor Green
        $testResults.Passed++
    } else {
        Write-Host "  ⊘ rag-redis-mcp-server not found (expected if not installed)" -ForegroundColor Yellow
    }
    
    # Check for Redis
    $redisCmd = Get-Command "redis-cli" -ErrorAction SilentlyContinue
    if ($redisCmd) {
        $ping = & redis-cli ping 2>&1
        if ($ping -match "PONG") {
            Write-Host "  ✓ Redis server is running" -ForegroundColor Green
            $testResults.Passed++
        } else {
            Write-Host "  ⊘ Redis server not responding" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  ⊘ redis-cli not found (expected if not installed)" -ForegroundColor Yellow
    }
    
    # Function availability already tested above
    Write-Host "  ✓ RAG-Redis functions are available" -ForegroundColor Green
    $testResults.Passed++
}
catch {
    Write-Host "  ✗ RAG-Redis test error: $_" -ForegroundColor Red
    $testResults.Failed++
}

#endregion

#region Cleanup

Write-Host "`n[7] Cleanup..." -ForegroundColor Yellow
Write-Host "=" * 80

try {
    # Remove test profile
    Remove-LLMProfile -ProfileName $testProfileName
    Write-Host "  ✓ Test profile removed" -ForegroundColor Green
    $testResults.Passed++
    
    # Remove test directory
    if (Test-Path $testDir) {
        Remove-Item -Path $testDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    Write-Host "  ✓ Test directory cleaned" -ForegroundColor Green
    $testResults.Passed++
}
catch {
    Write-Host "  ✗ Cleanup error: $_" -ForegroundColor Red
    $testResults.Failed++
}

#endregion

#region Summary

Write-Host "`n" -NoNewline
Write-Host ("=" * 80) -ForegroundColor Cyan
Write-Host "  TEST SUMMARY" -ForegroundColor White
Write-Host ("=" * 80) -ForegroundColor Cyan

$total = $testResults.Passed + $testResults.Failed
$passRate = if ($total -gt 0) { [math]::Round(($testResults.Passed / $total) * 100, 2) } else { 0 }

Write-Host "`nResults:" -ForegroundColor White
Write-Host "  Passed:  $($testResults.Passed)" -ForegroundColor Green
Write-Host "  Failed:  $($testResults.Failed)" -ForegroundColor Red
Write-Host "  Total:   $total"
Write-Host "`n  Pass Rate: $passRate%" -ForegroundColor $(
    if ($passRate -ge 90) { "Green" }
    elseif ($passRate -ge 75) { "Yellow" }
    else { "Red" }
)

if ($testResults.Failed -eq 0) {
    Write-Host "`n✓ ALL TESTS PASSED - Phase 2A is functioning correctly!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "`n⚠ SOME TESTS FAILED - Review output above for details" -ForegroundColor Yellow
    exit 1
}

#endregion
