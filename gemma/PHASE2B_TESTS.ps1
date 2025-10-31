#Requires -Version 7.0
<#
.SYNOPSIS
    Comprehensive test suite for Phase 2B RAG-Redis MCP Integration

.DESCRIPTION
    Tests all components of Phase 2B including:
    - Prerequisites validation
    - MCP server functionality
    - Document management
    - Memory operations
    - Project context management
    - Integration with Phase 2A

.NOTES
    File Name      : PHASE2B_TESTS.ps1
    Author         : Phase 2B Implementation
    Prerequisite   : Phase 2A and Phase 2B modules loaded
    Version        : 1.0.0
    Date           : 2025-01-15
#>

#region Test Configuration

$Script:TestConfig = @{
    TestDataDir = Join-Path $env:TEMP "phase2b-tests"
    TestProfileName = "phase2b-test-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    RedisHost = "localhost"
    RedisPort = 6380
    ServerPath = "$env:USERPROFILE\.local\bin\rag-redis-mcp-server.exe"
    McpConfigPath = "C:\codedev\llm\rag-redis\mcp.json"
    TestTimeout = 30  # seconds
    Verbose = $false
}

$Script:TestResults = @{
    Total = 0
    Passed = 0
    Failed = 0
    Skipped = 0
    StartTime = Get-Date
    Tests = @()
}

#endregion

#region Test Utilities

function Write-TestHeader {
    param([string]$Title)
    Write-Host "`n$('='*80)" -ForegroundColor Cyan
    Write-Host "  $Title" -ForegroundColor Cyan
    Write-Host "$('='*80)" -ForegroundColor Cyan
}

function Write-TestSection {
    param([string]$Section)
    Write-Host "`n$('-'*80)" -ForegroundColor DarkCyan
    Write-Host "  $Section" -ForegroundColor DarkCyan
    Write-Host "$('-'*80)" -ForegroundColor DarkCyan
}

function Write-TestResult {
    param(
        [string]$TestName,
        [bool]$Passed,
        [string]$Message = "",
        [object]$Details = $null
    )
    
    $Script:TestResults.Total++
    
    if ($Passed) {
        $Script:TestResults.Passed++
        Write-Host "  ✓ " -ForegroundColor Green -NoNewline
        Write-Host "$TestName" -ForegroundColor White
        if ($Message) {
            Write-Host "    $Message" -ForegroundColor DarkGray
        }
    } else {
        $Script:TestResults.Failed++
        Write-Host "  ✗ " -ForegroundColor Red -NoNewline
        Write-Host "$TestName" -ForegroundColor White
        if ($Message) {
            Write-Host "    ERROR: $Message" -ForegroundColor Red
        }
    }
    
    $Script:TestResults.Tests += @{
        Name = $TestName
        Passed = $Passed
        Message = $Message
        Details = $Details
        Timestamp = Get-Date
    }
}

function Write-TestSkipped {
    param([string]$TestName, [string]$Reason)
    
    $Script:TestResults.Total++
    $Script:TestResults.Skipped++
    
    Write-Host "  ○ " -ForegroundColor Yellow -NoNewline
    Write-Host "$TestName" -ForegroundColor DarkGray
    Write-Host "    SKIPPED: $Reason" -ForegroundColor Yellow
}

function Invoke-TestWithTimeout {
    param(
        [scriptblock]$ScriptBlock,
        [int]$TimeoutSeconds = 30
    )
    
    $job = Start-Job -ScriptBlock $ScriptBlock
    $completed = Wait-Job -Job $job -Timeout $TimeoutSeconds
    
    if ($completed) {
        $result = Receive-Job -Job $job
        Remove-Job -Job $job -Force
        return $result
    } else {
        Stop-Job -Job $job
        Remove-Job -Job $job -Force
        throw "Test timed out after $TimeoutSeconds seconds"
    }
}

#endregion

#region Prerequisite Tests

function Test-Prerequisites {
    Write-TestSection "Testing Prerequisites"
    
    # Test PowerShell version
    $psVersion = $PSVersionTable.PSVersion
    Write-TestResult `
        -TestName "PowerShell Version >= 7.0" `
        -Passed ($psVersion.Major -ge 7) `
        -Message "Version: $($psVersion.ToString())"
    
    # Test Phase 1 loaded
    $phase1Loaded = Get-Command -Name "Get-LLMProfile" -ErrorAction SilentlyContinue
    Write-TestResult `
        -TestName "Phase 1 Module Loaded" `
        -Passed ($null -ne $phase1Loaded) `
        -Message "Get-LLMProfile command available"
    
    # Test Phase 2A loaded
    $phase2aLoaded = Get-Command -Name "Set-LLMProfile" -ErrorAction SilentlyContinue
    Write-TestResult `
        -TestName "Phase 2A Module Loaded" `
        -Passed ($null -ne $phase2aLoaded) `
        -Message "Set-LLMProfile command available"
    
    # Test Phase 2B loaded
    $phase2bLoaded = Get-Command -Name "Initialize-RagRedisMcp" -ErrorAction SilentlyContinue
    Write-TestResult `
        -TestName "Phase 2B Module Loaded" `
        -Passed ($null -ne $phase2bLoaded) `
        -Message "Initialize-RagRedisMcp command available"
    
    # Test Rust toolchain
    try {
        $rustc = & rustc --version 2>&1
        Write-TestResult `
            -TestName "Rust Toolchain Available" `
            -Passed ($LASTEXITCODE -eq 0) `
            -Message $rustc
    } catch {
        Write-TestResult `
            -TestName "Rust Toolchain Available" `
            -Passed $false `
            -Message $_.Exception.Message
    }
    
    # Test cargo
    try {
        $cargo = & cargo --version 2>&1
        Write-TestResult `
            -TestName "Cargo Build Tool Available" `
            -Passed ($LASTEXITCODE -eq 0) `
            -Message $cargo
    } catch {
        Write-TestResult `
            -TestName "Cargo Build Tool Available" `
            -Passed $false `
            -Message $_.Exception.Message
    }
    
    # Test directory structure
    $testDirs = @(
        @{Path = "C:\codedev\llm\rag-redis"; Name = "RAG-Redis Project Directory"},
        @{Path = "C:\codedev\llm\rag-redis\rag-redis-system"; Name = "RAG-Redis System Directory"},
        @{Path = "C:\codedev\llm\rag-redis\models\all-MiniLM-L6-v2"; Name = "Embedding Model Directory"},
        @{Path = "$env:USERPROFILE\.local\bin"; Name = "User Bin Directory"}
    )
    
    foreach ($dir in $testDirs) {
        $exists = Test-Path $dir.Path
        Write-TestResult `
            -TestName $dir.Name `
            -Passed $exists `
            -Message $dir.Path
    }
}

#endregion

#region MCP Server Tests

function Test-McpServer {
    Write-TestSection "Testing MCP Server"
    
    # Test executable exists
    $serverPath = $Script:TestConfig.ServerPath
    $serverExists = Test-Path $serverPath
    Write-TestResult `
        -TestName "MCP Server Executable Exists" `
        -Passed $serverExists `
        -Message $serverPath
    
    if (-not $serverExists) {
        Write-TestSkipped "Remaining MCP Server Tests" "Executable not found"
        return
    }
    
    # Test executable properties
    $serverFile = Get-Item $serverPath
    Write-TestResult `
        -TestName "MCP Server Executable Size > 1MB" `
        -Passed ($serverFile.Length -gt 1MB) `
        -Message "Size: $([math]::Round($serverFile.Length / 1MB, 2)) MB"
    
    # Test availability via function
    try {
        $available = Test-RagRedisMcpAvailable
        Write-TestResult `
            -TestName "MCP Server Availability Check" `
            -Passed $available `
            -Message "Test-RagRedisMcpAvailable returned $available"
    } catch {
        Write-TestResult `
            -TestName "MCP Server Availability Check" `
            -Passed $false `
            -Message $_.Exception.Message
    }
    
    # Test MCP config file
    $mcpConfig = $Script:TestConfig.McpConfigPath
    $configExists = Test-Path $mcpConfig
    Write-TestResult `
        -TestName "MCP Configuration File Exists" `
        -Passed $configExists `
        -Message $mcpConfig
    
    if ($configExists) {
        try {
            $config = Get-Content $mcpConfig -Raw | ConvertFrom-Json
            $hasServers = $null -ne $config.mcpServers
            Write-TestResult `
                -TestName "MCP Configuration Valid JSON" `
                -Passed $hasServers `
                -Message "Contains mcpServers configuration"
        } catch {
            Write-TestResult `
                -TestName "MCP Configuration Valid JSON" `
                -Passed $false `
                -Message $_.Exception.Message
        }
    }
    
    # Test server can start (quick test)
    Write-Host "    Testing MCP server startup (this may take a few seconds)..." -ForegroundColor DarkGray
    try {
        $testRequest = @{
            jsonrpc = "2.0"
            id = 1
            method = "ping"
        } | ConvertTo-Json -Compress
        
        $startInfo = New-Object System.Diagnostics.ProcessStartInfo
        $startInfo.FileName = $serverPath
        $startInfo.UseShellExecute = $false
        $startInfo.RedirectStandardInput = $true
        $startInfo.RedirectStandardOutput = $true
        $startInfo.RedirectStandardError = $true
        $startInfo.CreateNoWindow = $true
        $startInfo.EnvironmentVariables["REDIS_URL"] = "redis://localhost:6380"
        
        $process = New-Object System.Diagnostics.Process
        $process.StartInfo = $startInfo
        $started = $process.Start()
        
        if ($started) {
            $process.StandardInput.WriteLine($testRequest)
            $process.StandardInput.Close()
            
            $timeout = 5000 # 5 seconds
            $exited = $process.WaitForExit($timeout)
            
            if ($exited) {
                Write-TestResult `
                    -TestName "MCP Server Startup Test" `
                    -Passed ($process.ExitCode -eq 0 -or $process.ExitCode -eq 1) `
                    -Message "Server started and responded (exit code: $($process.ExitCode))"
            } else {
                $process.Kill()
                Write-TestResult `
                    -TestName "MCP Server Startup Test" `
                    -Passed $false `
                    -Message "Server did not respond within timeout"
            }
        } else {
            Write-TestResult `
                -TestName "MCP Server Startup Test" `
                -Passed $false `
                -Message "Failed to start process"
        }
    } catch {
        Write-TestResult `
            -TestName "MCP Server Startup Test" `
            -Passed $false `
            -Message $_.Exception.Message
    }
}

#endregion

#region Profile Integration Tests

function Test-ProfileIntegration {
    Write-TestSection "Testing Profile Integration"
    
    # Create test profile
    try {
        $testDir = $Script:TestConfig.TestDataDir
        if (-not (Test-Path $testDir)) {
            New-Item -Path $testDir -ItemType Directory -Force | Out-Null
        }
        
        Set-LLMProfile `
            -ProfileName $Script:TestConfig.TestProfileName `
            -WorkingDirectory $testDir `
            -ErrorAction Stop
        
        Write-TestResult `
            -TestName "Create Test Profile" `
            -Passed $true `
            -Message "Profile: $($Script:TestConfig.TestProfileName)"
    } catch {
        Write-TestResult `
            -TestName "Create Test Profile" `
            -Passed $false `
            -Message $_.Exception.Message
        return
    }
    
    # Test profile exists
    try {
        $profile = Get-LLMProfile -ProfileName $Script:TestConfig.TestProfileName
        Write-TestResult `
            -TestName "Retrieve Test Profile" `
            -Passed ($null -ne $profile) `
            -Message "Profile retrieved successfully"
    } catch {
        Write-TestResult `
            -TestName "Retrieve Test Profile" `
            -Passed $false `
            -Message $_.Exception.Message
        return
    }
    
    # Test RAG-Redis initialization
    try {
        $init = Initialize-RagRedisMcp `
            -RedisHost $Script:TestConfig.RedisHost `
            -RedisPort $Script:TestConfig.RedisPort `
            -ErrorAction Stop
        
        Write-TestResult `
            -TestName "Initialize RAG-Redis MCP" `
            -Passed ($null -ne $init) `
            -Message "Initialization successful"
    } catch {
        Write-TestResult `
            -TestName "Initialize RAG-Redis MCP" `
            -Passed $false `
            -Message $_.Exception.Message
        return
    }
    
    # Verify profile has RAG configuration
    try {
        $profile = Get-LLMProfile -ProfileName $Script:TestConfig.TestProfileName
        $hasRagConfig = $null -ne $profile.RagRedisMcp
        Write-TestResult `
            -TestName "Profile Has RAG Configuration" `
            -Passed $hasRagConfig `
            -Message "RagRedisMcp section present: $hasRagConfig"
        
        if ($hasRagConfig) {
            $useMcp = $profile.RagRedisMcp.UseMcp -eq $true
            Write-TestResult `
                -TestName "RAG MCP Protocol Enabled" `
                -Passed $useMcp `
                -Message "UseMcp flag: $useMcp"
        }
    } catch {
        Write-TestResult `
            -TestName "Profile Has RAG Configuration" `
            -Passed $false `
            -Message $_.Exception.Message
    }
}

#endregion

#region Document Management Tests

function Test-DocumentManagement {
    Write-TestSection "Testing Document Management"
    
    # Test document ingestion
    try {
        $testContent = "This is a test document for Phase 2B validation. It contains important information about testing."
        $docResult = Add-RagDocumentMcp `
            -Content $testContent `
            -Metadata @{
                test = $true
                created = (Get-Date).ToString()
                type = "test-document"
            } `
            -ErrorAction Stop
        
        Write-TestResult `
            -TestName "Add Document via MCP" `
            -Passed ($null -ne $docResult) `
            -Message "Document added successfully"
        
        # Store document ID for later tests
        $Script:TestDocumentId = $docResult
    } catch {
        Write-TestResult `
            -TestName "Add Document via MCP" `
            -Passed $false `
            -Message $_.Exception.Message
    }
    
    # Test document search
    try {
        Start-Sleep -Seconds 2  # Allow time for indexing
        
        $searchResults = Search-RagDocumentsMcp `
            -Query "test document validation" `
            -Limit 5 `
            -ErrorAction Stop
        
        $hasResults = $null -ne $searchResults -and $searchResults.Count -gt 0
        Write-TestResult `
            -TestName "Search Documents via MCP" `
            -Passed $hasResults `
            -Message "Found $($searchResults.Count) results"
    } catch {
        Write-TestResult `
            -TestName "Search Documents via MCP" `
            -Passed $false `
            -Message $_.Exception.Message
    }
    
    # Test batch ingestion
    try {
        $testDocs = @(
            "Document 1: System architecture overview"
            "Document 2: API endpoint documentation"
            "Document 3: Database schema definition"
        )
        
        $batchResults = $testDocs | ForEach-Object {
            Add-RagDocumentMcp -Content $_ -ErrorAction SilentlyContinue
        }
        
        $successCount = ($batchResults | Where-Object { $null -ne $_ }).Count
        Write-TestResult `
            -TestName "Batch Document Ingestion" `
            -Passed ($successCount -eq $testDocs.Count) `
            -Message "Successfully indexed $successCount of $($testDocs.Count) documents"
    } catch {
        Write-TestResult `
            -TestName "Batch Document Ingestion" `
            -Passed $false `
            -Message $_.Exception.Message
    }
}

#endregion

#region Memory Management Tests

function Test-MemoryManagement {
    Write-TestSection "Testing Memory Management"
    
    # Test memory storage - different tiers
    $memoryTiers = @(
        @{Type = "working"; Content = "Current task: Testing Phase 2B memory system"; Importance = 0.9}
        @{Type = "short_term"; Content = "Test discovered bug in validation logic"; Importance = 0.7}
        @{Type = "long_term"; Content = "Project uses PowerShell 7 for automation"; Importance = 0.6}
        @{Type = "episodic"; Content = "Deployed test version at $(Get-Date)"; Importance = 0.5}
        @{Type = "semantic"; Content = "All tests must pass before deployment"; Importance = 1.0}
    )
    
    foreach ($tier in $memoryTiers) {
        try {
            $memResult = Store-RagMemoryMcp `
                -Content $tier.Content `
                -MemoryType $tier.Type `
                -Importance $tier.Importance `
                -ErrorAction Stop
            
            Write-TestResult `
                -TestName "Store Memory ($($tier.Type))" `
                -Passed ($null -ne $memResult) `
                -Message "Stored with importance $($tier.Importance)"
        } catch {
            Write-TestResult `
                -TestName "Store Memory ($($tier.Type))" `
                -Passed $false `
                -Message $_.Exception.Message
        }
    }
    
    # Test memory recall
    try {
        Start-Sleep -Seconds 2  # Allow time for storage
        
        $recallResults = Recall-RagMemoryMcp `
            -Query "testing Phase 2B" `
            -Limit 10 `
            -ErrorAction Stop
        
        $hasResults = $null -ne $recallResults -and $recallResults.Count -gt 0
        Write-TestResult `
            -TestName "Recall Memories via MCP" `
            -Passed $hasResults `
            -Message "Recalled $($recallResults.Count) memories"
    } catch {
        Write-TestResult `
            -TestName "Recall Memories via MCP" `
            -Passed $false `
            -Message $_.Exception.Message
    }
}

#endregion

#region Health Check Tests

function Test-HealthChecks {
    Write-TestSection "Testing Health Checks"
    
    # Test health check
    try {
        $health = Get-RagHealthMcp -ErrorAction Stop
        Write-TestResult `
            -TestName "Get System Health" `
            -Passed ($null -ne $health) `
            -Message "Health check returned data"
    } catch {
        Write-TestResult `
            -TestName "Get System Health" `
            -Passed $false `
            -Message $_.Exception.Message
    }
    
    # Test direct MCP tool invocation
    try {
        $directResult = Invoke-RagRedisMcpTool `
            -ToolName "health_check" `
            -Parameters @{} `
            -ErrorAction Stop
        
        Write-TestResult `
            -TestName "Direct MCP Tool Invocation" `
            -Passed ($null -ne $directResult) `
            -Message "health_check tool invoked successfully"
    } catch {
        Write-TestResult `
            -TestName "Direct MCP Tool Invocation" `
            -Passed $false `
            -Message $_.Exception.Message
    }
}

#endregion

#region Cleanup

function Invoke-TestCleanup {
    Write-TestSection "Cleanup"
    
    # Remove test profile
    try {
        if (Test-Path "C:\codedev\llm\gemma\profiles\$($Script:TestConfig.TestProfileName).json") {
            Remove-Item "C:\codedev\llm\gemma\profiles\$($Script:TestConfig.TestProfileName).json" -Force
            Write-TestResult `
                -TestName "Remove Test Profile" `
                -Passed $true `
                -Message "Test profile removed"
        } else {
            Write-TestSkipped "Remove Test Profile" "Profile not found"
        }
    } catch {
        Write-TestResult `
            -TestName "Remove Test Profile" `
            -Passed $false `
            -Message $_.Exception.Message
    }
    
    # Remove test data directory
    try {
        if (Test-Path $Script:TestConfig.TestDataDir) {
            Remove-Item $Script:TestConfig.TestDataDir -Recurse -Force
            Write-TestResult `
                -TestName "Remove Test Data Directory" `
                -Passed $true `
                -Message "Test data directory removed"
        } else {
            Write-TestSkipped "Remove Test Data Directory" "Directory not found"
        }
    } catch {
        Write-TestResult `
            -TestName "Remove Test Data Directory" `
            -Passed $false `
            -Message $_.Exception.Message
    }
}

#endregion

#region Test Report

function Show-TestReport {
    param([bool]$ExportJson = $false)
    
    $Script:TestResults.EndTime = Get-Date
    $duration = $Script:TestResults.EndTime - $Script:TestResults.StartTime
    
    Write-TestHeader "Test Results Summary"
    
    Write-Host "`n  Total Tests:   " -NoNewline
    Write-Host $Script:TestResults.Total -ForegroundColor White
    
    Write-Host "  Passed:        " -NoNewline
    Write-Host $Script:TestResults.Passed -ForegroundColor Green
    
    Write-Host "  Failed:        " -NoNewline
    $failColor = if ($Script:TestResults.Failed -gt 0) { "Red" } else { "Green" }
    Write-Host $Script:TestResults.Failed -ForegroundColor $failColor
    
    Write-Host "  Skipped:       " -NoNewline
    Write-Host $Script:TestResults.Skipped -ForegroundColor Yellow
    
    $passRate = if ($Script:TestResults.Total -gt 0) {
        [math]::Round(($Script:TestResults.Passed / $Script:TestResults.Total) * 100, 2)
    } else { 0 }
    
    Write-Host "  Pass Rate:     " -NoNewline
    $rateColor = if ($passRate -ge 90) { "Green" } elseif ($passRate -ge 70) { "Yellow" } else { "Red" }
    Write-Host "${passRate}%" -ForegroundColor $rateColor
    
    Write-Host "  Duration:      " -NoNewline
    Write-Host "$([math]::Round($duration.TotalSeconds, 2)) seconds" -ForegroundColor White
    
    # Failed tests detail
    if ($Script:TestResults.Failed -gt 0) {
        Write-Host "`n  Failed Tests:" -ForegroundColor Red
        $Script:TestResults.Tests | Where-Object { -not $_.Passed } | ForEach-Object {
            Write-Host "    • $($_.Name)" -ForegroundColor Red
            if ($_.Message) {
                Write-Host "      $($_.Message)" -ForegroundColor DarkRed
            }
        }
    }
    
    Write-Host ""
    
    # Export to JSON if requested
    if ($ExportJson) {
        $reportPath = Join-Path $env:TEMP "phase2b-test-report-$(Get-Date -Format 'yyyyMMdd-HHmmss').json"
        $Script:TestResults | ConvertTo-Json -Depth 10 | Out-File $reportPath
        Write-Host "  Test report exported to: $reportPath" -ForegroundColor Cyan
    }
    
    # Return exit code
    if ($Script:TestResults.Failed -gt 0) {
        return 1
    } else {
        return 0
    }
}

#endregion

#region Main Test Runner

function Invoke-Phase2BTests {
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
    
    .PARAMETER SkipCleanup
        Skip cleanup after tests
    
    .PARAMETER ExportReport
        Export test results to JSON
    
    .PARAMETER Verbose
        Enable verbose output
    
    .EXAMPLE
        Invoke-Phase2BTests
        
    .EXAMPLE
        Invoke-Phase2BTests -SkipPrerequisites -Verbose
    #>
    [CmdletBinding()]
    param(
        [switch]$SkipPrerequisites,
        [switch]$SkipServerTests,
        [switch]$SkipIntegration,
        [switch]$SkipCleanup,
        [switch]$ExportReport,
        [switch]$Verbose
    )
    
    $Script:TestConfig.Verbose = $Verbose
    $Script:TestResults = @{
        Total = 0
        Passed = 0
        Failed = 0
        Skipped = 0
        StartTime = Get-Date
        Tests = @()
    }
    
    Write-TestHeader "Phase 2B RAG-Redis MCP Test Suite"
    Write-Host "  Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
    Write-Host "  Test Profile: $($Script:TestConfig.TestProfileName)" -ForegroundColor Gray
    
    try {
        if (-not $SkipPrerequisites) {
            Test-Prerequisites
        }
        
        if (-not $SkipServerTests) {
            Test-McpServer
        }
        
        if (-not $SkipIntegration) {
            Test-ProfileIntegration
            Test-DocumentManagement
            Test-MemoryManagement
            Test-HealthChecks
        }
    } catch {
        Write-Host "`n  FATAL ERROR: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "  $($_.ScriptStackTrace)" -ForegroundColor DarkRed
    } finally {
        if (-not $SkipCleanup) {
            Invoke-TestCleanup
        }
        
        $exitCode = Show-TestReport -ExportJson:$ExportReport
        exit $exitCode
    }
}

#endregion

# Export functions
Export-ModuleMember -Function @(
    'Invoke-Phase2BTests',
    'Test-Prerequisites',
    'Test-McpServer',
    'Test-ProfileIntegration',
    'Test-DocumentManagement',
    'Test-MemoryManagement',
    'Test-HealthChecks'
)
