#Requires -Version 7.0
<#
.SYNOPSIS
    Quick installation verification for Phase 2B

.DESCRIPTION
    Validates Phase 2B installation and configuration
    
.EXAMPLE
    . C:\codedev\llm\gemma\PHASE2B_VERIFY.ps1
    Test-Phase2BInstallation
    
.EXAMPLE
    Test-Phase2BInstallation -Detailed
#>

function Test-Phase2BInstallation {
    <#
    .SYNOPSIS
        Verify Phase 2B installation
    
    .PARAMETER Detailed
        Show detailed information for each check
    #>
    param([switch]$Detailed)
    
    Write-Host "`n=== Phase 2B Installation Verification ===" -ForegroundColor Cyan
    Write-Host "  Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
    
    $results = @{
        Total = 0
        Passed = 0
        Failed = 0
    }
    
    function Test-Component {
        param($Name, $Test, $Details = "")
        $results.Total++
        if ($Test) {
            $results.Passed++
            Write-Host "  ✓ $Name" -ForegroundColor Green
            if ($Detailed -and $Details) { 
                Write-Host "    $Details" -ForegroundColor DarkGray 
            }
        } else {
            $results.Failed++
            Write-Host "  ✗ $Name" -ForegroundColor Red
            if ($Details) { 
                Write-Host "    $Details" -ForegroundColor Red 
            }
        }
    }
    
    Write-Host "`n  Prerequisites:" -ForegroundColor Cyan
    
    # PowerShell version
    Test-Component "PowerShell 7+" `
        ($PSVersionTable.PSVersion.Major -ge 7) `
        "Version: $($PSVersionTable.PSVersion)"
    
    # Rust toolchain
    try {
        $rustc = & rustc --version 2>&1
        Test-Component "Rust Toolchain" ($LASTEXITCODE -eq 0) $rustc
    } catch {
        Test-Component "Rust Toolchain" $false "Not installed"
    }
    
    Write-Host "`n  Phase Modules:" -ForegroundColor Cyan
    
    # Phase 1
    $phase1 = Get-Command "Get-LLMProfile" -ErrorAction SilentlyContinue
    Test-Component "Phase 1 Module" ($null -ne $phase1) "Get-LLMProfile available"
    
    # Phase 2A
    $phase2a = Get-Command "Set-LLMProfile" -ErrorAction SilentlyContinue
    Test-Component "Phase 2A Module" ($null -ne $phase2a) "Set-LLMProfile available"
    
    # Phase 2B Integration
    $phase2b = Get-Command "Initialize-RagRedisMcp" -ErrorAction SilentlyContinue
    Test-Component "Phase 2B Integration" ($null -ne $phase2b) "Initialize-RagRedisMcp available"
    
    Write-Host "`n  Files & Directories:" -ForegroundColor Cyan
    
    # MCP Server Binary
    $binary = "$env:USERPROFILE\.local\bin\rag-redis-mcp-server.exe"
    $binaryExists = Test-Path $binary
    if ($binaryExists) {
        $binaryInfo = Get-Item $binary
        Test-Component "MCP Server Binary" $true "Size: $([math]::Round($binaryInfo.Length / 1MB, 2)) MB"
    } else {
        Test-Component "MCP Server Binary" $false $binary
    }
    
    # MCP Config
    $config = "C:\codedev\llm\rag-redis\mcp.json"
    Test-Component "MCP Configuration" (Test-Path $config) $config
    
    # Embedding Model
    $model = "C:\codedev\llm\rag-redis\models\all-MiniLM-L6-v2"
    Test-Component "Embedding Model" (Test-Path $model) $model
    
    # Profile Directory
    $profiles = "C:\codedev\llm\gemma\profiles"
    Test-Component "Profile Directory" (Test-Path $profiles) $profiles
    
    Write-Host "`n  Tools & Scripts:" -ForegroundColor Cyan
    
    # Test Scripts
    $tests = "C:\codedev\llm\gemma\PHASE2B_TESTS.ps1"
    Test-Component "Test Suite" (Test-Path $tests) $tests
    
    # Deploy Script
    $deploy = "C:\codedev\llm\gemma\PHASE2B_DEPLOY.ps1"
    Test-Component "Deploy Script" (Test-Path $deploy) $deploy
    
    # Integration Script
    $integration = "C:\codedev\llm\gemma\PHASE2B_RAG_INTEGRATION.ps1"
    Test-Component "Integration Script" (Test-Path $integration) $integration
    
    # Summary
    Write-Host "`n  Summary:" -ForegroundColor Cyan
    Write-Host "    Total Checks:  $($results.Total)" -ForegroundColor White
    Write-Host "    Passed:        " -NoNewline
    Write-Host $results.Passed -ForegroundColor Green
    Write-Host "    Failed:        " -NoNewline
    Write-Host $results.Failed -ForegroundColor $(if ($results.Failed -eq 0) { "Green" } else { "Red" })
    
    $passRate = [math]::Round(($results.Passed / $results.Total) * 100, 1)
    Write-Host "    Pass Rate:     " -NoNewline
    Write-Host "${passRate}%" -ForegroundColor $(if ($passRate -ge 90) { "Green" } elseif ($passRate -ge 70) { "Yellow" } else { "Red" })
    
    if ($results.Failed -gt 0) {
        Write-Host "`n  ⚠ Some checks failed. Run with -Detailed for more information" -ForegroundColor Yellow
        Write-Host "  To fix issues, run: Invoke-Phase2BDeploy" -ForegroundColor Yellow
        return $false
    }
    
    Write-Host "`n  ✓ Phase 2B installation verified successfully!" -ForegroundColor Green
    Write-Host "`n  Next Steps:" -ForegroundColor Cyan
    Write-Host "    1. Create profile:  Set-LLMProfile -ProfileName 'test' -WorkingDirectory \$PWD" -ForegroundColor White
    Write-Host "    2. Initialize RAG:  Initialize-RagRedisMcp" -ForegroundColor White
    Write-Host "    3. Test document:   Add-RagDocumentMcp -Content 'test'" -ForegroundColor White
    Write-Host "    4. Run full tests:  Invoke-Phase2BTests" -ForegroundColor White
    Write-Host ""
    
    return $true
}

Export-ModuleMember -Function 'Test-Phase2BInstallation'
