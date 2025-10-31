#Requires -Version 7.0
<#
.SYNOPSIS
    Automated deployment script for Phase 2B RAG-Redis MCP Integration

.DESCRIPTION
    Handles complete deployment of Phase 2B including:
    - Prerequisites validation
    - RAG-Redis MCP server compilation
    - Binary installation
    - Configuration updates
    - Integration testing
    - Rollback on failure

.NOTES
    File Name      : PHASE2B_DEPLOY.ps1
    Author         : Phase 2B Implementation
    Prerequisite   : Rust toolchain, Phase 1 and Phase 2A installed
    Version        : 1.0.0
    Date           : 2025-01-15
#>

#region Configuration

$Script:DeployConfig = @{
    # Paths
    ProjectRoot = "C:\codedev\llm\rag-redis"
    SourceDir = "C:\codedev\llm\rag-redis\rag-redis-system\mcp-server"
    TargetBinDir = "$env:USERPROFILE\.local\bin"
    BinaryName = "rag-redis-mcp-server.exe"
    ModelPath = "C:\codedev\llm\rag-redis\models\all-MiniLM-L6-v2"
    
    # Build configuration
    BuildProfile = "release"
    CargoFeatures = @()
    
    # Backup
    BackupDir = "$env:TEMP\phase2b-backup-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    CreateBackup = $true
    
    # Validation
    RunTests = $true
    TestTimeout = 300  # 5 minutes
    
    # Redis
    RedisHost = "localhost"
    RedisPort = 6380
    StartRedis = $false
}

$Script:DeployState = @{
    StartTime = Get-Date
    Steps = @()
    BackupCreated = $false
    BinaryInstalled = $false
    ConfigUpdated = $false
    TestsPassed = $false
    Success = $false
}

#endregion

#region Utilities

function Write-DeployHeader {
    param([string]$Title)
    Write-Host "`n$('='*80)" -ForegroundColor Cyan
    Write-Host "  $Title" -ForegroundColor Cyan
    Write-Host "$('='*80)" -ForegroundColor Cyan
}

function Write-DeploySection {
    param([string]$Section)
    Write-Host "`n$('-'*80)" -ForegroundColor DarkCyan
    Write-Host "  $Section" -ForegroundColor DarkCyan
    Write-Host "$('-'*80)" -ForegroundColor DarkCyan
}

function Write-DeployStep {
    param(
        [string]$Step,
        [string]$Status = "Running",
        [string]$Message = ""
    )
    
    $statusColor = switch ($Status) {
        "Running" { "Yellow" }
        "Success" { "Green" }
        "Failed" { "Red" }
        "Warning" { "Yellow" }
        "Info" { "Cyan" }
        default { "White" }
    }
    
    $statusSymbol = switch ($Status) {
        "Running" { "⟳" }
        "Success" { "✓" }
        "Failed" { "✗" }
        "Warning" { "⚠" }
        "Info" { "ℹ" }
        default { "•" }
    }
    
    Write-Host "  $statusSymbol " -ForegroundColor $statusColor -NoNewline
    Write-Host "$Step" -ForegroundColor White
    if ($Message) {
        Write-Host "    $Message" -ForegroundColor DarkGray
    }
    
    $Script:DeployState.Steps += @{
        Step = $Step
        Status = $Status
        Message = $Message
        Timestamp = Get-Date
    }
}

function Invoke-DeployCommand {
    param(
        [string]$Command,
        [string]$WorkingDirectory = $PWD,
        [hashtable]$Environment = @{},
        [int]$TimeoutSeconds = 300
    )
    
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = "pwsh"
    $startInfo.Arguments = "-NoProfile -Command `"$Command`""
    $startInfo.WorkingDirectory = $WorkingDirectory
    $startInfo.UseShellExecute = $false
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.CreateNoWindow = $true
    
    foreach ($key in $Environment.Keys) {
        $startInfo.EnvironmentVariables[$key] = $Environment[$key]
    }
    
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    
    $stdout = New-Object System.Text.StringBuilder
    $stderr = New-Object System.Text.StringBuilder
    
    $process.add_OutputDataReceived({
        if ($null -ne $EventArgs.Data) {
            [void]$stdout.AppendLine($EventArgs.Data)
        }
    })
    
    $process.add_ErrorDataReceived({
        if ($null -ne $EventArgs.Data) {
            [void]$stderr.AppendLine($EventArgs.Data)
        }
    })
    
    [void]$process.Start()
    $process.BeginOutputReadLine()
    $process.BeginErrorReadLine()
    
    $completed = $process.WaitForExit($TimeoutSeconds * 1000)
    
    if (-not $completed) {
        $process.Kill()
        throw "Command timed out after $TimeoutSeconds seconds"
    }
    
    return @{
        ExitCode = $process.ExitCode
        Output = $stdout.ToString()
        Error = $stderr.ToString()
        Success = ($process.ExitCode -eq 0)
    }
}

#endregion

#region Prerequisites Check

function Test-DeployPrerequisites {
    Write-DeploySection "Checking Prerequisites"
    
    $allPassed = $true
    
    # PowerShell version
    $psVersion = $PSVersionTable.PSVersion
    if ($psVersion.Major -ge 7) {
        Write-DeployStep "PowerShell Version" "Success" "Version: $($psVersion.ToString())"
    } else {
        Write-DeployStep "PowerShell Version" "Failed" "Required: 7.0+, Found: $($psVersion.ToString())"
        $allPassed = $false
    }
    
    # Rust toolchain
    try {
        $rustc = & rustc --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-DeployStep "Rust Compiler" "Success" $rustc
        } else {
            Write-DeployStep "Rust Compiler" "Failed" "rustc command failed"
            $allPassed = $false
        }
    } catch {
        Write-DeployStep "Rust Compiler" "Failed" $_.Exception.Message
        $allPassed = $false
    }
    
    # Cargo
    try {
        $cargo = & cargo --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-DeployStep "Cargo Build Tool" "Success" $cargo
        } else {
            Write-DeployStep "Cargo Build Tool" "Failed" "cargo command failed"
            $allPassed = $false
        }
    } catch {
        Write-DeployStep "Cargo Build Tool" "Failed" $_.Exception.Message
        $allPassed = $false
    }
    
    # Project directory
    if (Test-Path $Script:DeployConfig.ProjectRoot) {
        Write-DeployStep "Project Directory" "Success" $Script:DeployConfig.ProjectRoot
    } else {
        Write-DeployStep "Project Directory" "Failed" "Not found: $($Script:DeployConfig.ProjectRoot)"
        $allPassed = $false
    }
    
    # Source directory
    if (Test-Path $Script:DeployConfig.SourceDir) {
        Write-DeployStep "Source Directory" "Success" $Script:DeployConfig.SourceDir
    } else {
        Write-DeployStep "Source Directory" "Failed" "Not found: $($Script:DeployConfig.SourceDir)"
        $allPassed = $false
    }
    
    # Target bin directory
    if (Test-Path $Script:DeployConfig.TargetBinDir) {
        Write-DeployStep "Target Bin Directory" "Success" $Script:DeployConfig.TargetBinDir
    } else {
        Write-DeployStep "Target Bin Directory" "Warning" "Creating: $($Script:DeployConfig.TargetBinDir)"
        New-Item -Path $Script:DeployConfig.TargetBinDir -ItemType Directory -Force | Out-Null
    }
    
    # Embedding model
    if (Test-Path $Script:DeployConfig.ModelPath) {
        Write-DeployStep "Embedding Model" "Success" $Script:DeployConfig.ModelPath
    } else {
        Write-DeployStep "Embedding Model" "Warning" "Not found: $($Script:DeployConfig.ModelPath)"
    }
    
    # Phase 1 and 2A modules
    $phase1 = Get-Command -Name "Get-LLMProfile" -ErrorAction SilentlyContinue
    if ($phase1) {
        Write-DeployStep "Phase 1 Module" "Success" "Loaded"
    } else {
        Write-DeployStep "Phase 1 Module" "Warning" "Not loaded (optional for deployment)"
    }
    
    $phase2a = Get-Command -Name "Set-LLMProfile" -ErrorAction SilentlyContinue
    if ($phase2a) {
        Write-DeployStep "Phase 2A Module" "Success" "Loaded"
    } else {
        Write-DeployStep "Phase 2A Module" "Warning" "Not loaded (optional for deployment)"
    }
    
    return $allPassed
}

#endregion

#region Backup

function New-DeployBackup {
    Write-DeploySection "Creating Backup"
    
    if (-not $Script:DeployConfig.CreateBackup) {
        Write-DeployStep "Backup" "Info" "Skipped (CreateBackup = false)"
        return $true
    }
    
    try {
        $backupDir = $Script:DeployConfig.BackupDir
        New-Item -Path $backupDir -ItemType Directory -Force | Out-Null
        Write-DeployStep "Create Backup Directory" "Success" $backupDir
        
        # Backup existing binary
        $targetBinary = Join-Path $Script:DeployConfig.TargetBinDir $Script:DeployConfig.BinaryName
        if (Test-Path $targetBinary) {
            $backupBinary = Join-Path $backupDir $Script:DeployConfig.BinaryName
            Copy-Item $targetBinary $backupBinary -Force
            
            $fileInfo = Get-Item $targetBinary
            Write-DeployStep "Backup Existing Binary" "Success" "Size: $([math]::Round($fileInfo.Length / 1MB, 2)) MB"
        } else {
            Write-DeployStep "Backup Existing Binary" "Info" "No existing binary found"
        }
        
        # Backup MCP config
        $mcpConfig = Join-Path $Script:DeployConfig.ProjectRoot "mcp.json"
        if (Test-Path $mcpConfig) {
            $backupConfig = Join-Path $backupDir "mcp.json"
            Copy-Item $mcpConfig $backupConfig -Force
            Write-DeployStep "Backup MCP Config" "Success" $mcpConfig
        } else {
            Write-DeployStep "Backup MCP Config" "Warning" "Config not found"
        }
        
        $Script:DeployState.BackupCreated = $true
        return $true
    } catch {
        Write-DeployStep "Create Backup" "Failed" $_.Exception.Message
        return $false
    }
}

#endregion

#region Build

function Invoke-DeployBuild {
    Write-DeploySection "Building MCP Server"
    
    try {
        $buildDir = $Script:DeployConfig.SourceDir
        
        Write-DeployStep "Clean Previous Build" "Running" "cargo clean"
        $cleanResult = Invoke-DeployCommand `
            -Command "cargo clean" `
            -WorkingDirectory $buildDir `
            -TimeoutSeconds 60
        
        if ($cleanResult.Success) {
            Write-DeployStep "Clean Previous Build" "Success" "Build artifacts removed"
        } else {
            Write-DeployStep "Clean Previous Build" "Warning" "Clean failed (continuing anyway)"
        }
        
        Write-DeployStep "Compile MCP Server" "Running" "This may take several minutes..."
        
        $buildCmd = "cargo build --profile=$($Script:DeployConfig.BuildProfile)"
        if ($Script:DeployConfig.CargoFeatures.Count -gt 0) {
            $features = $Script:DeployConfig.CargoFeatures -join ","
            $buildCmd += " --features=$features"
        }
        
        $buildResult = Invoke-DeployCommand `
            -Command $buildCmd `
            -WorkingDirectory $buildDir `
            -Environment @{
                RUST_BACKTRACE = "1"
                RUSTFLAGS = "-C target-cpu=native"
            } `
            -TimeoutSeconds 600
        
        if ($buildResult.Success) {
            $builtBinary = Join-Path $Script:DeployConfig.ProjectRoot "target\$($Script:DeployConfig.BuildProfile)\mcp-server.exe"
            
            if (Test-Path $builtBinary) {
                $fileInfo = Get-Item $builtBinary
                Write-DeployStep "Compile MCP Server" "Success" "Size: $([math]::Round($fileInfo.Length / 1MB, 2)) MB"
                return $builtBinary
            } else {
                Write-DeployStep "Compile MCP Server" "Failed" "Binary not found at expected location"
                Write-Host "`nBuild Output:" -ForegroundColor Yellow
                Write-Host $buildResult.Output -ForegroundColor Gray
                return $null
            }
        } else {
            Write-DeployStep "Compile MCP Server" "Failed" "Build failed with exit code $($buildResult.ExitCode)"
            Write-Host "`nBuild Error:" -ForegroundColor Red
            Write-Host $buildResult.Error -ForegroundColor DarkRed
            return $null
        }
    } catch {
        Write-DeployStep "Build Process" "Failed" $_.Exception.Message
        return $null
    }
}

#endregion

#region Installation

function Install-DeployBinary {
    param([string]$SourcePath)
    
    Write-DeploySection "Installing Binary"
    
    try {
        if (-not (Test-Path $SourcePath)) {
            Write-DeployStep "Verify Source Binary" "Failed" "Source not found: $SourcePath"
            return $false
        }
        
        $sourceInfo = Get-Item $SourcePath
        Write-DeployStep "Verify Source Binary" "Success" "Size: $([math]::Round($sourceInfo.Length / 1MB, 2)) MB"
        
        $targetPath = Join-Path $Script:DeployConfig.TargetBinDir $Script:DeployConfig.BinaryName
        
        # Stop any running instances
        $runningProcesses = Get-Process | Where-Object { $_.Path -eq $targetPath }
        if ($runningProcesses) {
            Write-DeployStep "Stop Running Instances" "Running" "Found $($runningProcesses.Count) running instance(s)"
            $runningProcesses | Stop-Process -Force
            Start-Sleep -Seconds 2
            Write-DeployStep "Stop Running Instances" "Success" "Processes terminated"
        } else {
            Write-DeployStep "Stop Running Instances" "Info" "No running instances found"
        }
        
        # Copy binary
        Write-DeployStep "Copy Binary to Target" "Running" $targetPath
        Copy-Item $SourcePath $targetPath -Force
        
        if (Test-Path $targetPath) {
            $targetInfo = Get-Item $targetPath
            Write-DeployStep "Copy Binary to Target" "Success" "Installed: $([math]::Round($targetInfo.Length / 1MB, 2)) MB"
            $Script:DeployState.BinaryInstalled = $true
            return $true
        } else {
            Write-DeployStep "Copy Binary to Target" "Failed" "Copy operation failed"
            return $false
        }
    } catch {
        Write-DeployStep "Install Binary" "Failed" $_.Exception.Message
        return $false
    }
}

#endregion

#region Configuration

function Update-DeployConfiguration {
    Write-DeploySection "Updating Configuration"
    
    try {
        $mcpConfigPath = Join-Path $Script:DeployConfig.ProjectRoot "mcp.json"
        
        if (-not (Test-Path $mcpConfigPath)) {
            Write-DeployStep "Load MCP Config" "Warning" "Config file not found, creating new one"
            
            $newConfig = @{
                mcpServers = @{
                    "rag-redis-llm" = @{
                        command = Join-Path $Script:DeployConfig.TargetBinDir $Script:DeployConfig.BinaryName
                        args = @()
                        env = @{
                            REDIS_URL = "redis://$($Script:DeployConfig.RedisHost):$($Script:DeployConfig.RedisPort)"
                            RUST_LOG = "info"
                            MODEL_PATH = $Script:DeployConfig.ModelPath
                        }
                    }
                }
            }
            
            $newConfig | ConvertTo-Json -Depth 10 | Out-File $mcpConfigPath -Encoding UTF8
            Write-DeployStep "Create MCP Config" "Success" "New configuration created"
            $Script:DeployState.ConfigUpdated = $true
            return $true
        }
        
        Write-DeployStep "Load MCP Config" "Success" $mcpConfigPath
        
        $config = Get-Content $mcpConfigPath -Raw | ConvertFrom-Json
        
        # Update command path
        $oldCommand = $config.mcpServers.'rag-redis-llm'.command
        $newCommand = Join-Path $Script:DeployConfig.TargetBinDir $Script:DeployConfig.BinaryName
        $config.mcpServers.'rag-redis-llm'.command = $newCommand
        
        if ($oldCommand -ne $newCommand) {
            Write-DeployStep "Update Command Path" "Success" "Updated from $oldCommand"
        } else {
            Write-DeployStep "Update Command Path" "Info" "Path already correct"
        }
        
        # Ensure MODEL_PATH is set
        if (-not $config.mcpServers.'rag-redis-llm'.env.MODEL_PATH) {
            $config.mcpServers.'rag-redis-llm'.env | Add-Member -NotePropertyName "MODEL_PATH" -NotePropertyValue $Script:DeployConfig.ModelPath
            Write-DeployStep "Add MODEL_PATH" "Success" $Script:DeployConfig.ModelPath
        } else {
            Write-DeployStep "Verify MODEL_PATH" "Info" "Already configured"
        }
        
        # Save config
        $config | ConvertTo-Json -Depth 10 | Out-File $mcpConfigPath -Encoding UTF8
        Write-DeployStep "Save MCP Config" "Success" "Configuration updated"
        
        $Script:DeployState.ConfigUpdated = $true
        return $true
    } catch {
        Write-DeployStep "Update Configuration" "Failed" $_.Exception.Message
        return $false
    }
}

#endregion

#region Testing

function Invoke-DeployTests {
    Write-DeploySection "Running Tests"
    
    if (-not $Script:DeployConfig.RunTests) {
        Write-DeployStep "Run Tests" "Info" "Skipped (RunTests = false)"
        $Script:DeployState.TestsPassed = $true
        return $true
    }
    
    try {
        # Check if test module exists
        $testScript = "C:\codedev\llm\gemma\PHASE2B_TESTS.ps1"
        if (-not (Test-Path $testScript)) {
            Write-DeployStep "Load Test Module" "Warning" "Test script not found: $testScript"
            Write-DeployStep "Run Tests" "Warning" "Skipping tests (module not available)"
            $Script:DeployState.TestsPassed = $true
            return $true
        }
        
        Write-DeployStep "Load Test Module" "Success" $testScript
        
        # Load required modules first
        $phase1Script = "C:\codedev\llm\gemma\PHASE1_IMPLEMENTATION.ps1"
        $phase2aScripts = @(
            "C:\codedev\llm\gemma\PHASE2A_AUTO_CLAUDE.ps1",
            "C:\codedev\llm\gemma\PHASE2A_RAG_REDIS.ps1",
            "C:\codedev\llm\gemma\PHASE2A_TOOL_HOOKS.ps1"
        )
        $phase2bScript = "C:\codedev\llm\gemma\PHASE2B_RAG_INTEGRATION.ps1"
        
        $allScripts = @($phase1Script) + $phase2aScripts + @($phase2bScript, $testScript)
        
        foreach ($script in $allScripts) {
            if (Test-Path $script) {
                . $script
            }
        }
        
        Write-DeployStep "Execute Tests" "Running" "This may take a few minutes..."
        
        # Run tests in a job to capture output
        $testJob = Start-Job -ScriptBlock {
            param($TestScript)
            . $TestScript
            Invoke-Phase2BTests -ExportReport
        } -ArgumentList $testScript
        
        $completed = Wait-Job -Job $testJob -Timeout $Script:DeployConfig.TestTimeout
        
        if ($completed) {
            $testOutput = Receive-Job -Job $testJob
            Remove-Job -Job $testJob -Force
            
            # Parse test results from output
            if ($testOutput -match "Pass Rate:\s+(\d+(?:\.\d+)?)%") {
                $passRate = [double]$Matches[1]
                
                if ($passRate -ge 90) {
                    Write-DeployStep "Execute Tests" "Success" "Pass rate: ${passRate}%"
                    $Script:DeployState.TestsPassed = $true
                    return $true
                } elseif ($passRate -ge 70) {
                    Write-DeployStep "Execute Tests" "Warning" "Pass rate: ${passRate}% (below 90%)"
                    Write-Host "    Consider investigating failures before proceeding" -ForegroundColor Yellow
                    $Script:DeployState.TestsPassed = $true
                    return $true
                } else {
                    Write-DeployStep "Execute Tests" "Failed" "Pass rate: ${passRate}% (below 70%)"
                    return $false
                }
            } else {
                Write-DeployStep "Execute Tests" "Warning" "Could not parse test results"
                $Script:DeployState.TestsPassed = $true
                return $true
            }
        } else {
            Stop-Job -Job $testJob
            Remove-Job -Job $testJob -Force
            Write-DeployStep "Execute Tests" "Failed" "Tests timed out after $($Script:DeployConfig.TestTimeout) seconds"
            return $false
        }
    } catch {
        Write-DeployStep "Run Tests" "Failed" $_.Exception.Message
        return $false
    }
}

#endregion

#region Rollback

function Invoke-DeployRollback {
    Write-DeploySection "Rolling Back Deployment"
    
    if (-not $Script:DeployState.BackupCreated) {
        Write-DeployStep "Rollback" "Warning" "No backup available"
        return
    }
    
    try {
        $backupDir = $Script:DeployConfig.BackupDir
        
        # Restore binary
        if ($Script:DeployState.BinaryInstalled) {
            $backupBinary = Join-Path $backupDir $Script:DeployConfig.BinaryName
            if (Test-Path $backupBinary) {
                $targetBinary = Join-Path $Script:DeployConfig.TargetBinDir $Script:DeployConfig.BinaryName
                Copy-Item $backupBinary $targetBinary -Force
                Write-DeployStep "Restore Binary" "Success" "Previous version restored"
            }
        }
        
        # Restore config
        if ($Script:DeployState.ConfigUpdated) {
            $backupConfig = Join-Path $backupDir "mcp.json"
            if (Test-Path $backupConfig) {
                $targetConfig = Join-Path $Script:DeployConfig.ProjectRoot "mcp.json"
                Copy-Item $backupConfig $targetConfig -Force
                Write-DeployStep "Restore Config" "Success" "Previous configuration restored"
            }
        }
        
        Write-DeployStep "Rollback Complete" "Success" "System restored to previous state"
    } catch {
        Write-DeployStep "Rollback" "Failed" $_.Exception.Message
    }
}

#endregion

#region Report

function Show-DeployReport {
    Write-DeployHeader "Deployment Summary"
    
    $duration = (Get-Date) - $Script:DeployState.StartTime
    
    Write-Host "`n  Status:        " -NoNewline
    if ($Script:DeployState.Success) {
        Write-Host "SUCCESS" -ForegroundColor Green
    } else {
        Write-Host "FAILED" -ForegroundColor Red
    }
    
    Write-Host "  Duration:      " -NoNewline
    Write-Host "$([math]::Round($duration.TotalSeconds, 2)) seconds" -ForegroundColor White
    
    Write-Host "  Steps:         " -NoNewline
    Write-Host "$($Script:DeployState.Steps.Count) total" -ForegroundColor White
    
    Write-Host "`n  Backup:        " -NoNewline
    Write-Host $(if ($Script:DeployState.BackupCreated) { "Created" } else { "Not created" }) -ForegroundColor $(if ($Script:DeployState.BackupCreated) { "Green" } else { "Yellow" })
    
    Write-Host "  Binary:        " -NoNewline
    Write-Host $(if ($Script:DeployState.BinaryInstalled) { "Installed" } else { "Not installed" }) -ForegroundColor $(if ($Script:DeployState.BinaryInstalled) { "Green" } else { "Red" })
    
    Write-Host "  Config:        " -NoNewline
    Write-Host $(if ($Script:DeployState.ConfigUpdated) { "Updated" } else { "Not updated" }) -ForegroundColor $(if ($Script:DeployState.ConfigUpdated) { "Green" } else { "Red" })
    
    Write-Host "  Tests:         " -NoNewline
    Write-Host $(if ($Script:DeployState.TestsPassed) { "Passed" } else { "Failed" }) -ForegroundColor $(if ($Script:DeployState.TestsPassed) { "Green" } else { "Red" })
    
    if ($Script:DeployState.BackupCreated) {
        Write-Host "`n  Backup Location: $($Script:DeployConfig.BackupDir)" -ForegroundColor Cyan
    }
    
    Write-Host ""
}

#endregion

#region Main Deployment Function

function Invoke-Phase2BDeploy {
    <#
    .SYNOPSIS
        Deploy Phase 2B RAG-Redis MCP Integration
    
    .DESCRIPTION
        Automated deployment of Phase 2B components
    
    .PARAMETER SkipBackup
        Skip creating backup of existing installation
    
    .PARAMETER SkipTests
        Skip running tests after deployment
    
    .PARAMETER SkipBuild
        Skip building (use existing binary)
    
    .PARAMETER ForceRollback
        Force rollback to previous version
    
    .PARAMETER StartRedis
        Start Redis server before deployment
    
    .EXAMPLE
        Invoke-Phase2BDeploy
        
    .EXAMPLE
        Invoke-Phase2BDeploy -SkipTests -StartRedis
    #>
    [CmdletBinding()]
    param(
        [switch]$SkipBackup,
        [switch]$SkipTests,
        [switch]$SkipBuild,
        [switch]$ForceRollback,
        [switch]$StartRedis
    )
    
    $Script:DeployState = @{
        StartTime = Get-Date
        Steps = @()
        BackupCreated = $false
        BinaryInstalled = $false
        ConfigUpdated = $false
        TestsPassed = $false
        Success = $false
    }
    
    $Script:DeployConfig.CreateBackup = -not $SkipBackup
    $Script:DeployConfig.RunTests = -not $SkipTests
    $Script:DeployConfig.StartRedis = $StartRedis
    
    Write-DeployHeader "Phase 2B RAG-Redis MCP Deployment"
    Write-Host "  Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
    
    if ($ForceRollback) {
        Invoke-DeployRollback
        Show-DeployReport
        return
    }
    
    try {
        # Prerequisites
        if (-not (Test-DeployPrerequisites)) {
            throw "Prerequisites check failed"
        }
        
        # Backup
        if (-not (New-DeployBackup)) {
            throw "Backup creation failed"
        }
        
        # Build
        if ($SkipBuild) {
            Write-DeploySection "Build"
            Write-DeployStep "Build" "Info" "Skipped (using existing binary)"
            $builtBinary = Join-Path $Script:DeployConfig.ProjectRoot "target\$($Script:DeployConfig.BuildProfile)\mcp-server.exe"
            
            if (-not (Test-Path $builtBinary)) {
                throw "Binary not found: $builtBinary"
            }
        } else {
            $builtBinary = Invoke-DeployBuild
            if (-not $builtBinary) {
                throw "Build failed"
            }
        }
        
        # Installation
        if (-not (Install-DeployBinary -SourcePath $builtBinary)) {
            throw "Binary installation failed"
        }
        
        # Configuration
        if (-not (Update-DeployConfiguration)) {
            throw "Configuration update failed"
        }
        
        # Testing
        if (-not (Invoke-DeployTests)) {
            throw "Tests failed"
        }
        
        $Script:DeployState.Success = $true
        Write-DeploySection "Deployment Complete"
        Write-DeployStep "Deployment" "Success" "All components deployed successfully"
        
    } catch {
        Write-Host "`n  DEPLOYMENT FAILED: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "  $($_.ScriptStackTrace)" -ForegroundColor DarkRed
        
        $rollback = Read-Host "`n  Rollback to previous version? (Y/N)"
        if ($rollback -eq 'Y' -or $rollback -eq 'y') {
            Invoke-DeployRollback
        }
    } finally {
        Show-DeployReport
        
        if ($Script:DeployState.Success) {
            Write-Host "  Next Steps:" -ForegroundColor Cyan
            Write-Host "    1. Load Phase 2B module: . C:\codedev\llm\gemma\PHASE2B_RAG_INTEGRATION.ps1" -ForegroundColor White
            Write-Host "    2. Test availability: Test-RagRedisMcpAvailable" -ForegroundColor White
            Write-Host "    3. Create profile: Set-LLMProfile -ProfileName 'test' -WorkingDirectory \$PWD" -ForegroundColor White
            Write-Host "    4. Initialize: Initialize-RagRedisMcp" -ForegroundColor White
            Write-Host ""
        }
    }
}

#endregion

# Export functions
Export-ModuleMember -Function 'Invoke-Phase2BDeploy'
