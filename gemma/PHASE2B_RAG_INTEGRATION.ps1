# =============================================================================
# PHASE 2B: RAG-REDIS FULL MCP INTEGRATION
# =============================================================================
# Integrates the fully-functional RAG-Redis MCP server with Phase 2A
# profile system, providing complete document indexing and search via MCP.
#
# Version: 1.0.0
# Created: $(Get-Date -Format 'yyyy-MM-dd')
# =============================================================================

#region RAG-Redis MCP Configuration

$script:RagRedisMcpPath = "C:\users\david\.local\bin\rag-redis-mcp-server.exe"
$script:RagRedisConfigPath = "C:\codedev\llm\rag-redis\mcp.json"

<#
.SYNOPSIS
    Tests RAG-Redis MCP server availability.

.DESCRIPTION
    Checks if the RAG-Redis MCP server executable exists and is accessible.

.EXAMPLE
    Test-RagRedisMcpAvailable
#>
function Test-RagRedisMcpAvailable {
    [CmdletBinding()]
    param()

    if (-not (Test-Path $script:RagRedisMcpPath)) {
        Write-Warning "RAG-Redis MCP server not found at: $script:RagRedisMcpPath"
        return $false
    }

    try {
        $fileInfo = Get-Item $script:RagRedisMcpPath
        Write-Verbose "RAG-Redis MCP server found: $($fileInfo.Length) bytes, last modified $($fileInfo.LastWriteTime)"
        return $true
    }
    catch {
        Write-Warning "Failed to access RAG-Redis MCP server: $_"
        return $false
    }
}

<#
.SYNOPSIS
    Initializes RAG-Redis with full MCP integration.

.DESCRIPTION
    This replaces the Phase 2A placeholder Initialize-RagRedis function
    with full MCP protocol support for actual document indexing and search.

.PARAMETER RedisHost
    Redis server host (default: localhost).

.PARAMETER RedisPort
    Redis server port (default: 6380).

.PARAMETER RedisPassword
    Redis password if authentication is required.

.PARAMETER Force
    Force restart of RAG-Redis server.

.PARAMETER UseMcp
    Use MCP protocol for communication (default: true).

.EXAMPLE
    Initialize-RagRedisMcp

.EXAMPLE
    Initialize-RagRedisMcp -RedisHost "localhost" -RedisPort 6380 -UseMcp $true
#>
function Initialize-RagRedisMcp {
    [CmdletBinding()]
    param(
        [Parameter()]
        [string]$RedisHost = "localhost",

        [Parameter()]
        [int]$RedisPort = 6380,

        [Parameter()]
        [securestring]$RedisPassword,

        [Parameter()]
        [switch]$Force,

        [Parameter()]
        [bool]$UseMcp = $true
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

    # Verify MCP server is available
    if (-not (Test-RagRedisMcpAvailable)) {
        Write-Error "RAG-Redis MCP server not available. Build it first:"
        Write-Host "  cd C:\codedev\llm\rag-redis\rag-redis-system\mcp-server" -ForegroundColor Yellow
        Write-Host "  cargo build --release" -ForegroundColor Yellow
        Write-Host "  Copy target\release\mcp-server.exe to C:\users\david\.local\bin\rag-redis-mcp-server.exe" -ForegroundColor Yellow
        return $false
    }

    # Build Redis URL
    $redisUrl = "redis://${RedisHost}:${RedisPort}"
    if ($RedisPassword) {
        $plainPassword = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto(
            [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($RedisPassword)
        )
        $redisUrl = "redis://:${plainPassword}@${RedisHost}:${RedisPort}"
    }

    # Create profile-specific namespace
    $namespace = "profile:$profileName"

    # Store configuration in profile
    $global:ProfileConfig.RagRedisMcp = @{
        ServerPath = $script:RagRedisMcpPath
        RedisHost = $RedisHost
        RedisPort = $RedisPort
        RedisUrl = $redisUrl
        Namespace = $namespace
        UseMcp = $UseMcp
        ConfiguredAt = Get-Date -Format "o"
    }

    Write-Host "✓ RAG-Redis MCP configured for profile: $profileName" -ForegroundColor Green
    Write-Host "  Server: $script:RagRedisMcpPath" -ForegroundColor Cyan
    Write-Host "  Namespace: $namespace" -ForegroundColor Cyan
    Write-Host "  Redis: ${RedisHost}:${RedisPort}" -ForegroundColor Cyan
    Write-Host "  MCP Protocol: $(if ($UseMcp) { 'Enabled' } else { 'Disabled' })" -ForegroundColor Cyan

    return $true
}

<#
.SYNOPSIS
    Invokes RAG-Redis MCP tool via JSON-RPC.

.DESCRIPTION
    Sends a tool call request to the RAG-Redis MCP server using JSON-RPC 2.0
    protocol over stdio.

.PARAMETER ToolName
    Name of the tool to invoke (e.g., "ingest_document", "search_documents").

.PARAMETER Parameters
    Hashtable of parameters to pass to the tool.

.EXAMPLE
    Invoke-RagRedisMcpTool -ToolName "search_documents" -Parameters @{ query = "authentication"; limit = 5 }
#>
function Invoke-RagRedisMcpTool {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$ToolName,

        [Parameter()]
        [hashtable]$Parameters = @{}
    )

    if (-not $global:ProfileConfig -or -not $global:ProfileConfig.RagRedisMcp) {
        Write-Error "RAG-Redis MCP not initialized. Run Initialize-RagRedisMcp first."
        return $null
    }

    $config = $global:ProfileConfig.RagRedisMcp

    # Build JSON-RPC request
    $request = @{
        jsonrpc = "2.0"
        id = Get-Random -Minimum 1 -Maximum 10000
        method = "tools/call"
        params = @{
            name = $ToolName
            arguments = $Parameters
        }
    } | ConvertTo-Json -Depth 10 -Compress

    Write-Verbose "MCP Request: $request"

    try {
        # Set environment variables
        $env:REDIS_URL = $config.RedisUrl
        $env:RUST_LOG = "info"

        # Execute MCP server with request
        $response = $request | & $config.ServerPath 2>$null | Where-Object { $_ -match '^{' }

        if ($response) {
            $result = $response | ConvertFrom-Json
            
            if ($result.error) {
                Write-Error "MCP Error: $($result.error.message)"
                return $null
            }

            return $result.result
        }
        else {
            Write-Warning "No response from MCP server"
            return $null
        }
    }
    catch {
        Write-Error "Failed to invoke MCP tool: $_"
        return $null
    }
}

<#
.SYNOPSIS
    Indexes a document using RAG-Redis MCP.

.DESCRIPTION
    Ingests a document into the RAG system using the actual MCP protocol,
    replacing the Phase 2A placeholder implementation.

.PARAMETER Content
    Document content to index.

.PARAMETER Metadata
    Optional metadata hashtable.

.PARAMETER DocumentId
    Optional custom document ID.

.EXAMPLE
    Add-RagDocumentMcp -Content "This is a test document" -Metadata @{ title = "Test"; author = "Me" }
#>
function Add-RagDocumentMcp {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true, ValueFromPipeline = $true)]
        [string]$Content,

        [Parameter()]
        [hashtable]$Metadata = @{},

        [Parameter()]
        [string]$DocumentId
    )

    $params = @{
        content = $Content
        metadata = $Metadata
    }

    if ($DocumentId) {
        $params.document_id = $DocumentId
    }

    $result = Invoke-RagRedisMcpTool -ToolName "ingest_document" -Parameters $params

    if ($result) {
        Write-Host "✓ Document indexed via MCP" -ForegroundColor Green
        return $result
    }
    else {
        Write-Error "Failed to index document"
        return $null
    }
}

<#
.SYNOPSIS
    Searches documents using RAG-Redis MCP.

.DESCRIPTION
    Performs semantic search using the actual MCP protocol,
    replacing the Phase 2A placeholder implementation.

.PARAMETER Query
    Search query text.

.PARAMETER Limit
    Maximum number of results (default: 10).

.PARAMETER MinScore
    Minimum similarity score (default: 0.0).

.EXAMPLE
    Search-RagDocumentsMcp -Query "authentication implementation" -Limit 5
#>
function Search-RagDocumentsMcp {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$Query,

        [Parameter()]
        [int]$Limit = 10,

        [Parameter()]
        [double]$MinScore = 0.0
    )

    $params = @{
        query = $Query
        limit = $Limit
        min_score = $MinScore
    }

    $result = Invoke-RagRedisMcpTool -ToolName "search_documents" -Parameters $params

    if ($result) {
        Write-Host "✓ Search completed via MCP" -ForegroundColor Green
        return $result
    }
    else {
        Write-Error "Search failed"
        return $null
    }
}

<#
.SYNOPSIS
    Stores a memory using RAG-Redis MCP.

.DESCRIPTION
    Stores a memory in the multi-tier memory system using MCP protocol.

.PARAMETER Content
    Memory content to store.

.PARAMETER MemoryType
    Type of memory (working, short_term, long_term, episodic, semantic).

.PARAMETER Importance
    Importance score (0.0-1.0).

.EXAMPLE
    Store-RagMemoryMcp -Content "User prefers dark mode" -MemoryType "long_term" -Importance 0.8
#>
function Store-RagMemoryMcp {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$Content,

        [Parameter()]
        [ValidateSet("working", "short_term", "long_term", "episodic", "semantic")]
        [string]$MemoryType = "short_term",

        [Parameter()]
        [ValidateRange(0.0, 1.0)]
        [double]$Importance = 0.5
    )

    $params = @{
        content = $Content
        memory_type = $MemoryType
        importance = $Importance
    }

    $result = Invoke-RagRedisMcpTool -ToolName "store_memory" -Parameters $params

    if ($result) {
        Write-Host "✓ Memory stored via MCP (type: $MemoryType, importance: $Importance)" -ForegroundColor Green
        return $result
    }
    else {
        Write-Error "Failed to store memory"
        return $null
    }
}

<#
.SYNOPSIS
    Recalls memories using RAG-Redis MCP.

.DESCRIPTION
    Retrieves relevant memories from the system using MCP protocol.

.PARAMETER Query
    Query to search memories.

.PARAMETER MemoryType
    Optional memory type filter.

.PARAMETER Limit
    Maximum number of memories to return (default: 10).

.EXAMPLE
    Recall-RagMemoryMcp -Query "user preferences" -Limit 5
#>
function Recall-RagMemoryMcp {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$Query,

        [Parameter()]
        [ValidateSet("working", "short_term", "long_term", "episodic", "semantic")]
        [string]$MemoryType,

        [Parameter()]
        [int]$Limit = 10
    )

    $params = @{
        query = $Query
        limit = $Limit
    }

    if ($MemoryType) {
        $params.memory_type = $MemoryType
    }

    $result = Invoke-RagRedisMcpTool -ToolName "recall_memory" -Parameters $params

    if ($result) {
        Write-Host "✓ Memory recall completed via MCP" -ForegroundColor Green
        return $result
    }
    else {
        Write-Error "Memory recall failed"
        return $null
    }
}

<#
.SYNOPSIS
    Gets RAG-Redis system health via MCP.

.DESCRIPTION
    Retrieves system health and metrics using MCP protocol.

.EXAMPLE
    Get-RagHealthMcp
#>
function Get-RagHealthMcp {
    [CmdletBinding()]
    param()

    $result = Invoke-RagRedisMcpTool -ToolName "health_check" -Parameters @{}

    if ($result) {
        Write-Host "✓ Health check completed" -ForegroundColor Green
        return $result
    }
    else {
        Write-Warning "Health check failed or server unavailable"
        return $null
    }
}

#endregion

# Export functions
Write-Host "✓ RAG-Redis MCP integration (Phase 2B) loaded" -ForegroundColor Cyan
Write-Host "  Use Initialize-RagRedisMcp to configure for current profile" -ForegroundColor Gray
