# =============================================================================
# PHASE 2A: RAG-REDIS PROFILE SEPARATION
# =============================================================================
# Provides PowerShell functions to manage profile-specific RAG-Redis
# vector stores with automatic isolation and context management.
#
# Dependencies:
# - rag-redis-mcp-server in PATH or ~/.local/bin
# - Redis server (local or remote)
# - Profile context from $ProfileConfig global variable
#
# Version: 1.0.0
# Created: $(Get-Date -Format 'yyyy-MM-dd')
# =============================================================================

#region RAG-Redis Configuration

# Default Redis connection settings
$script:DefaultRedisHost = "localhost"
$script:DefaultRedisPort = 6379
$script:RagRedisServerPath = $null

<#
.SYNOPSIS
    Initializes RAG-Redis server for current profile.

.DESCRIPTION
    Starts or verifies RAG-Redis MCP server with profile-specific
    vector store namespace for isolated context management.

.PARAMETER RedisHost
    Redis server host (default: localhost).

.PARAMETER RedisPort
    Redis server port (default: 6379).

.PARAMETER RedisPassword
    Redis password if authentication is required.

.PARAMETER Force
    Force restart of RAG-Redis server.

.EXAMPLE
    Initialize-RagRedis

.EXAMPLE
    Initialize-RagRedis -RedisHost "redis.example.com" -RedisPort 6380
#>
function Initialize-RagRedis {
    [CmdletBinding()]
    param(
        [Parameter()]
        [string]$RedisHost = $script:DefaultRedisHost,

        [Parameter()]
        [int]$RedisPort = $script:DefaultRedisPort,

        [Parameter()]
        [securestring]$RedisPassword,

        [Parameter()]
        [switch]$Force
    )

    # Verify profile is active
    if (-not $global:ProfileConfig) {
        Write-Error "No active profile. Use Set-LLMProfile first."
        return $false
    }

    $profileName = $global:ProfileConfig.ProfileName
    if ([string]::IsNullOrWhiteSpace($profileName)) {
        Write-Error "Profile name is not configured."
        return $false
    }

    # Find RAG-Redis server
    $ragRedisCmd = Get-Command "rag-redis-mcp-server" -ErrorAction SilentlyContinue
    if (-not $ragRedisCmd) {
        Write-Error "rag-redis-mcp-server not found in PATH."
        Write-Host "Install from: https://github.com/amanr-dev/rag-redis-mcp-server" -ForegroundColor Yellow
        return $false
    }

    $script:RagRedisServerPath = $ragRedisCmd.Path

    # Check if server is already running for this profile
    $existingProcess = Get-RagRedisProcess -ProfileName $profileName

    if ($existingProcess -and -not $Force) {
        Write-Host "✓ RAG-Redis server already running for profile: $profileName" -ForegroundColor Green
        Write-Host "  PID: $($existingProcess.Id)" -ForegroundColor Cyan
        return $true
    }

    if ($existingProcess -and $Force) {
        Write-Host "Stopping existing RAG-Redis server (PID: $($existingProcess.Id))..." -ForegroundColor Yellow
        Stop-Process -Id $existingProcess.Id -Force
        Start-Sleep -Milliseconds 500
    }

    # Build Redis connection string
    $redisUrl = "redis://${RedisHost}:${RedisPort}"
    if ($RedisPassword) {
        $plainPassword = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto(
            [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($RedisPassword)
        )
        $redisUrl = "redis://:${plainPassword}@${RedisHost}:${RedisPort}"
    }

    # Create profile-specific namespace
    $namespace = "profile:$profileName"

    # Start RAG-Redis server
    try {
        Write-Host "Starting RAG-Redis server for profile: $profileName..." -ForegroundColor Cyan

        $processArgs = @(
            "--redis-url", $redisUrl,
            "--namespace", $namespace,
            "--port", "0"  # Auto-assign port
        )

        $process = Start-Process -FilePath $script:RagRedisServerPath `
            -ArgumentList $processArgs `
            -NoNewWindow `
            -PassThru `
            -RedirectStandardOutput "$env:TEMP\rag-redis-$profileName.out.log" `
            -RedirectStandardError "$env:TEMP\rag-redis-$profileName.err.log"

        # Store process info in profile config
        $global:ProfileConfig.RagRedisProcess = @{
            Pid = $process.Id
            Namespace = $namespace
            RedisHost = $RedisHost
            RedisPort = $RedisPort
            StartedAt = Get-Date -Format "o"
        }

        # Give server time to start
        Start-Sleep -Milliseconds 1000

        if (-not $process.HasExited) {
            Write-Host "✓ RAG-Redis server started successfully" -ForegroundColor Green
            Write-Host "  PID: $($process.Id)" -ForegroundColor Cyan
            Write-Host "  Namespace: $namespace" -ForegroundColor Cyan
            Write-Host "  Redis: ${RedisHost}:${RedisPort}" -ForegroundColor Cyan
            return $true
        } else {
            Write-Error "RAG-Redis server exited unexpectedly"
            $errorLog = Get-Content "$env:TEMP\rag-redis-$profileName.err.log" -Raw -ErrorAction SilentlyContinue
            if ($errorLog) {
                Write-Error $errorLog
            }
            return $false
        }
    }
    catch {
        Write-Error "Failed to start RAG-Redis server: $_"
        return $false
    }
}

<#
.SYNOPSIS
    Adds documents to profile-specific RAG vector store.

.DESCRIPTION
    Indexes documents or directories into the profile's isolated
    RAG-Redis vector store for semantic retrieval.

.PARAMETER Path
    Path to document(s) or directory to index.

.PARAMETER Recursive
    Recursively index subdirectories.

.PARAMETER FileFilter
    File extension filter (e.g., "*.md", "*.py").

.EXAMPLE
    Add-RagDocument -Path "./docs" -Recursive

.EXAMPLE
    Add-RagDocument -Path "./src" -FileFilter "*.ts" -Recursive
#>
function Add-RagDocument {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true, ValueFromPipeline = $true)]
        [string[]]$Path,

        [Parameter()]
        [switch]$Recursive,

        [Parameter()]
        [string]$FileFilter = "*"
    )

    begin {
        if (-not $global:ProfileConfig -or -not $global:ProfileConfig.RagRedisProcess) {
            Write-Error "RAG-Redis not initialized. Run Initialize-RagRedis first."
            return
        }
    }

    process {
        foreach ($p in $Path) {
            if (-not (Test-Path $p)) {
                Write-Warning "Path not found: $p"
                continue
            }

            $resolvedPath = Resolve-Path $p

            if (Test-Path -Path $resolvedPath -PathType Container) {
                # Directory - index all matching files
                $files = if ($Recursive) {
                    Get-ChildItem -Path $resolvedPath -Filter $FileFilter -Recurse -File
                } else {
                    Get-ChildItem -Path $resolvedPath -Filter $FileFilter -File
                }

                Write-Host "Indexing $($files.Count) files from: $resolvedPath" -ForegroundColor Cyan

                foreach ($file in $files) {
                    Index-SingleFile -FilePath $file.FullName
                }
            }
            else {
                # Single file
                Index-SingleFile -FilePath $resolvedPath
            }
        }
    }
}

<#
.SYNOPSIS
    Queries profile-specific RAG vector store.

.DESCRIPTION
    Performs semantic search against the profile's RAG-Redis vector
    store and returns relevant document chunks.

.PARAMETER Query
    Search query text.

.PARAMETER TopK
    Number of results to return (default: 5).

.PARAMETER MinScore
    Minimum similarity score (0.0 - 1.0, default: 0.7).

.EXAMPLE
    Search-RagDocuments "authentication implementation"

.EXAMPLE
    Search-RagDocuments "error handling" -TopK 10 -MinScore 0.8
#>
function Search-RagDocuments {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true, Position = 0)]
        [string]$Query,

        [Parameter()]
        [int]$TopK = 5,

        [Parameter()]
        [double]$MinScore = 0.7
    )

    if (-not $global:ProfileConfig -or -not $global:ProfileConfig.RagRedisProcess) {
        Write-Error "RAG-Redis not initialized. Run Initialize-RagRedis first."
        return $null
    }

    # TODO: Implement actual RAG query via MCP protocol
    # This is a placeholder that would call the rag-redis-mcp-server
    Write-Host "Searching RAG store: $Query" -ForegroundColor Cyan
    Write-Host "  Top K: $TopK" -ForegroundColor Gray
    Write-Host "  Min Score: $MinScore" -ForegroundColor Gray

    Write-Warning "RAG search not yet implemented - requires MCP client integration"
    
    return @()
}

<#
.SYNOPSIS
    Clears profile-specific RAG vector store.

.DESCRIPTION
    Removes all indexed documents from the current profile's
    RAG-Redis namespace.

.PARAMETER Confirm
    Confirm before clearing (default: prompt).

.EXAMPLE
    Clear-RagDocuments -Confirm:$false
#>
function Clear-RagDocuments {
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [Parameter()]
        [switch]$Force
    )

    if (-not $global:ProfileConfig) {
        Write-Error "No active profile."
        return
    }

    $profileName = $global:ProfileConfig.ProfileName

    if (-not $Force -and -not $PSCmdlet.ShouldProcess($profileName, "Clear RAG documents")) {
        return
    }

    Write-Host "Clearing RAG documents for profile: $profileName" -ForegroundColor Yellow

    # TODO: Implement actual RAG clear via MCP protocol
    Write-Warning "RAG clear not yet implemented - requires MCP client integration"
}

<#
.SYNOPSIS
    Stops RAG-Redis server for current profile.

.DESCRIPTION
    Gracefully stops the RAG-Redis MCP server process.

.EXAMPLE
    Stop-RagRedis
#>
function Stop-RagRedis {
    [CmdletBinding()]
    param()

    if (-not $global:ProfileConfig -or -not $global:ProfileConfig.RagRedisProcess) {
        Write-Warning "RAG-Redis not running for current profile."
        return
    }

    $pid = $global:ProfileConfig.RagRedisProcess.Pid

    try {
        $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
        if ($process) {
            Write-Host "Stopping RAG-Redis server (PID: $pid)..." -ForegroundColor Yellow
            Stop-Process -Id $pid -Force
            Start-Sleep -Milliseconds 500

            Write-Host "✓ RAG-Redis server stopped" -ForegroundColor Green
        } else {
            Write-Warning "RAG-Redis process (PID: $pid) not found"
        }
    }
    catch {
        Write-Error "Failed to stop RAG-Redis server: $_"
    }
    finally {
        $global:ProfileConfig.Remove("RagRedisProcess")
    }
}

#endregion

#region Helper Functions

function Get-RagRedisProcess {
    param([string]$ProfileName)

    if ($global:ProfileConfig -and $global:ProfileConfig.RagRedisProcess) {
        $pid = $global:ProfileConfig.RagRedisProcess.Pid
        return Get-Process -Id $pid -ErrorAction SilentlyContinue
    }

    return $null
}

function Index-SingleFile {
    param([string]$FilePath)

    # TODO: Implement actual file indexing via MCP protocol
    Write-Host "  Indexing: $FilePath" -ForegroundColor Gray
}

#endregion

Write-Host "✓ RAG-Redis profile separation loaded" -ForegroundColor Cyan
