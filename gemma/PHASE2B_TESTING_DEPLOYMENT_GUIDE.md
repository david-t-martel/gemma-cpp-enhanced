# Phase 2B Testing, Deployment & Verification Guide

**Version**: 1.0.0  
**Date**: 2025-01-15  
**Status**: ✅ Production Ready

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
4. [Testing Framework](#testing-framework)
5. [Deployment System](#deployment-system)
6. [Verification Tools](#verification-tools)
7. [Core Profile Integration](#core-profile-integration)
8. [Troubleshooting](#troubleshooting)
9. [Reference](#reference)

---

## Overview

Phase 2B includes a comprehensive suite of testing, deployment, and verification tools for the RAG-Redis MCP integration. These tools are integrated into your PowerShell core profile for seamless access.

###Files Created

| File | Purpose | Size |
|------|---------|------|
| `PHASE2B_TESTS.ps1` | Comprehensive test suite | 802 lines |
| `PHASE2B_DEPLOY.ps1` | Automated deployment | 809 lines |
| `PHASE2B_VERIFY.ps1` | Quick verification | 150 lines |
| `PHASE2B_CORE_PROFILE_INTEGRATION.ps1` | Profile integration | 351 lines |
| `PHASE2B_TESTING_DEPLOYMENT_GUIDE.md` | This document | - |

---

## Quick Start

### For New Installations

```powershell
# 1. Verify installation
Test-Phase2BInstallation

# 2. If not installed, deploy
Invoke-Phase2BDeploy

# 3. Verify deployment
Test-Phase2BInstallation -Detailed

# 4. Run full tests
Invoke-Phase2BTests
```

### For Existing Installations

```powershell
# Check status
Get-Phase2BStatus

# Verify components
Test-Phase2BInstallation

# Run tests
Invoke-Phase2BTests
```

---

## Installation

### Prerequisites

- **PowerShell 7.0+** (Required)
- **Rust toolchain** (rustc + cargo)
- **Phase 1 & 2A** modules
- **RAG-Redis project** at `C:\codedev\llm\rag-redis`

### Core Profile Integration

Add Phase 2B to your PowerShell profile:

**Method 1: Manual Integration**

1. Open `~\OneDrive\Documents\PowerShell\core_profile.ps1`
2. Find the GCP Profile Management section (around line 710)
3. Insert the contents of `PHASE2B_CORE_PROFILE_INTEGRATION.ps1`
4. Save and reload profile: `. $PROFILE`

**Method 2: Automatic (if not already integrated)**

```powershell
# Load the integration script
. C:\codedev\llm\gemma\PHASE2B_CORE_PROFILE_INTEGRATION.ps1

# Test it works
Get-Phase2BStatus
```

### Verify Installation

```powershell
# Quick check
Test-Phase2BInstallation

# Detailed check
Test-Phase2BInstallation -Detailed
```

---

## Testing Framework

### Overview

The Phase 2B test suite (`PHASE2B_TESTS.ps1`) provides comprehensive testing across all components.

### Test Categories

1. **Prerequisites** - PowerShell version, Rust toolchain, directory structure
2. **MCP Server** - Binary existence, configuration, startup
3. **Profile Integration** - Phase 1/2A/2B module loading
4. **Document Management** - Indexing, search, batch operations
5. **Memory Management** - All memory tiers (working/short/long/episodic/semantic)
6. **Health Checks** - System health, direct MCP tool invocation

### Running Tests

**Basic Test Run:**
```powershell
Invoke-Phase2BTests
```

**Skip Prerequisites:**
```powershell
Invoke-Phase2BTests -SkipPrerequisites
```

**Skip Server Tests:**
```powershell
Invoke-Phase2BTests -SkipServerTests
```

**Export Results:**
```powershell
Invoke-Phase2BTests -ExportReport
```

**Skip Cleanup (for debugging):**
```powershell
Invoke-Phase2BTests -SkipCleanup
```

### Test Output

```
================================================================================
  Phase 2B RAG-Redis MCP Test Suite
================================================================================
  Started: 2025-01-15 10:00:00
  Test Profile: phase2b-test-20250115-100000

--------------------------------------------------------------------------------
  Testing Prerequisites
--------------------------------------------------------------------------------
  ✓ PowerShell Version >= 7.0
    Version: 7.5.3
  ✓ Phase 1 Module Loaded
    Get-LLMProfile command available
  ✓ Phase 2A Module Loaded
    Set-LLMProfile command available
  ✓ Phase 2B Module Loaded
    Initialize-RagRedisMcp command available
  ✓ Rust Toolchain Available
    rustc 1.75.0
  ✓ Cargo Build Tool Available
    cargo 1.75.0
  ...

================================================================================
  Test Results Summary
================================================================================

  Total Tests:   45
  Passed:        43
  Failed:        0
  Skipped:       2
  Pass Rate:     100%
  Duration:      12.45 seconds
```

### Individual Test Functions

You can also run specific test categories:

```powershell
# Load test module
. C:\codedev\llm\gemma\PHASE2B_TESTS.ps1

# Run specific tests
Test-Prerequisites
Test-McpServer
Test-ProfileIntegration
Test-DocumentManagement
Test-MemoryManagement
Test-HealthChecks
```

---

## Deployment System

### Overview

The Phase 2B deployment script (`PHASE2B_DEPLOY.ps1`) handles automated deployment with validation and rollback.

### Deployment Process

1. **Prerequisites Check** - Verify all requirements
2. **Backup** - Create backup of existing installation
3. **Build** - Compile MCP server from source
4. **Installation** - Copy binary to target location
5. **Configuration** - Update MCP configuration
6. **Testing** - Run integration tests
7. **Rollback** - Restore previous version on failure

### Running Deployment

**Full Deployment (recommended):**
```powershell
Invoke-Phase2BDeploy
```

**Skip Backup:**
```powershell
Invoke-Phase2BDeploy -SkipBackup
```

**Skip Tests:**
```powershell
Invoke-Phase2BDeploy -SkipTests
```

**Skip Build (use existing binary):**
```powershell
Invoke-Phase2BDeploy -SkipBuild
```

**Force Rollback:**
```powershell
Invoke-Phase2BDeploy -ForceRollback
```

### Deployment Output

```
================================================================================
  Phase 2B RAG-Redis MCP Deployment
================================================================================
  Started: 2025-01-15 10:00:00

--------------------------------------------------------------------------------
  Checking Prerequisites
--------------------------------------------------------------------------------
  ✓ PowerShell Version
    Version: 7.5.3
  ✓ Rust Compiler
    rustc 1.75.0
  ✓ Cargo Build Tool
    cargo 1.75.0
  ✓ Project Directory
    C:\codedev\llm\rag-redis
  ✓ Source Directory
    C:\codedev\llm\rag-redis\rag-redis-system\mcp-server
  ...

--------------------------------------------------------------------------------
  Creating Backup
--------------------------------------------------------------------------------
  ✓ Create Backup Directory
    C:\Users\david\AppData\Local\Temp\phase2b-backup-20250115-100000
  ✓ Backup Existing Binary
    Size: 5.6 MB
  ✓ Backup MCP Config
    C:\codedev\llm\rag-redis\mcp.json

--------------------------------------------------------------------------------
  Building MCP Server
--------------------------------------------------------------------------------
  ⟳ Clean Previous Build
    cargo clean
  ✓ Clean Previous Build
    Build artifacts removed
  ⟳ Compile MCP Server
    This may take several minutes...
  ✓ Compile MCP Server
    Size: 5.6 MB

--------------------------------------------------------------------------------
  Installing Binary
--------------------------------------------------------------------------------
  ✓ Verify Source Binary
    Size: 5.6 MB
  ℹ Stop Running Instances
    No running instances found
  ⟳ Copy Binary to Target
    C:\users\david\.local\bin\rag-redis-mcp-server.exe
  ✓ Copy Binary to Target
    Installed: 5.6 MB

--------------------------------------------------------------------------------
  Updating Configuration
--------------------------------------------------------------------------------
  ✓ Load MCP Config
    C:\codedev\llm\rag-redis\mcp.json
  ✓ Update Command Path
    Updated from C:\users\david\.local\bin\mcp-server.exe
  ✓ Add MODEL_PATH
    c:\codedev\llm\rag-redis\models\all-MiniLM-L6-v2
  ✓ Save MCP Config
    Configuration updated

--------------------------------------------------------------------------------
  Running Tests
--------------------------------------------------------------------------------
  ✓ Load Test Module
    C:\codedev\llm\gemma\PHASE2B_TESTS.ps1
  ⟳ Execute Tests
    This may take a few minutes...
  ✓ Execute Tests
    Pass rate: 95.6%

--------------------------------------------------------------------------------
  Deployment Complete
--------------------------------------------------------------------------------
  ✓ Deployment
    All components deployed successfully

================================================================================
  Deployment Summary
================================================================================

  Status:        SUCCESS
  Duration:      124.56 seconds
  Steps:         25 total

  Backup:        Created
  Binary:        Installed
  Config:        Updated
  Tests:         Passed

  Backup Location: C:\Users\david\AppData\Local\Temp\phase2b-backup-20250115-100000

  Next Steps:
    1. Load Phase 2B module: . C:\codedev\llm\gemma\PHASE2B_RAG_INTEGRATION.ps1
    2. Test availability: Test-RagRedisMcpAvailable
    3. Create profile: Set-LLMProfile -ProfileName 'test' -WorkingDirectory $PWD
    4. Initialize: Initialize-RagRedisMcp
```

### Rollback Process

If deployment fails or you need to revert:

1. **Automatic Rollback** (on failure):
   - Deployment will prompt: "Rollback to previous version? (Y/N)"
   - Answer `Y` to restore previous state

2. **Manual Rollback**:
   ```powershell
   Invoke-Phase2BDeploy -ForceRollback
   ```

3. **Manual Restoration** (from backup):
   ```powershell
   # Find backup directory (shown in deployment output)
   $backup = "C:\Users\david\AppData\Local\Temp\phase2b-backup-YYYYMMDD-HHMMSS"
   
   # Restore binary
   Copy-Item "$backup\rag-redis-mcp-server.exe" `
       "C:\users\david\.local\bin\rag-redis-mcp-server.exe" -Force
   
   # Restore config
   Copy-Item "$backup\mcp.json" `
       "C:\codedev\llm\rag-redis\mcp.json" -Force
   ```

---

## Verification Tools

### Quick Verification

The verification script provides fast validation of installation:

```powershell
# Basic check
Test-Phase2BInstallation

# Detailed check
Test-Phase2BInstallation -Detailed
```

### Verification Output

```
=== Phase 2B Installation Verification ===
  Date: 2025-01-15 10:00:00

  Prerequisites:
  ✓ PowerShell 7+
    Version: 7.5.3
  ✓ Rust Toolchain
    rustc 1.75.0

  Phase Modules:
  ✓ Phase 1 Module
    Get-LLMProfile available
  ✓ Phase 2A Module
    Set-LLMProfile available
  ✓ Phase 2B Integration
    Initialize-RagRedisMcp available

  Files & Directories:
  ✓ MCP Server Binary
    Size: 5.6 MB
  ✓ MCP Configuration
    C:\codedev\llm\rag-redis\mcp.json
  ✓ Embedding Model
    C:\codedev\llm\rag-redis\models\all-MiniLM-L6-v2
  ✓ Profile Directory
    C:\codedev\llm\gemma\profiles

  Tools & Scripts:
  ✓ Test Suite
    C:\codedev\llm\gemma\PHASE2B_TESTS.ps1
  ✓ Deploy Script
    C:\codedev\llm\gemma\PHASE2B_DEPLOY.ps1
  ✓ Integration Script
    C:\codedev\llm\gemma\PHASE2B_RAG_INTEGRATION.ps1

  Summary:
    Total Checks:  13
    Passed:        13
    Failed:        0
    Pass Rate:     100.0%

  ✓ Phase 2B installation verified successfully!

  Next Steps:
    1. Create profile:  Set-LLMProfile -ProfileName 'test' -WorkingDirectory $PWD
    2. Initialize RAG:  Initialize-RagRedisMcp
    3. Test document:   Add-RagDocumentMcp -Content 'test'
    4. Run full tests:  Invoke-Phase2BTests
```

### Status Check

Get comprehensive status of Phase 2B:

```powershell
Get-Phase2BStatus
```

Output:
```
Phase 2B Status:
  Modules Loaded: Yes
  MCP Server: Installed (5.6 MB)
  Active Profile: my-project
    Working Dir: C:\projects\myapp
    RAG-Redis: Configured

  Quick Start:
    Test installation:  Test-Phase2BInstallation
    Create profile:     Set-LLMProfile -ProfileName 'test' -WorkingDirectory $PWD
    Initialize RAG:     Initialize-RagRedisMcp
    Run tests:          Invoke-Phase2BTests
```

---

## Core Profile Integration

### Available Commands

Once integrated into your core profile, these commands are available:

#### Profile Management

```powershell
# Create/switch profile
Set-LLMProfile -ProfileName "my-project" -WorkingDirectory "C:\projects\myapp"

# Get current profile
Get-LLMProfile

# Get specific profile
Get-LLMProfile -ProfileName "my-project"
```

#### RAG-Redis Initialization

```powershell
# Initialize with defaults
Initialize-RagRedisMcp

# Initialize with custom Redis
Initialize-RagRedisMcp -RedisHost "localhost" -RedisPort 6380
```

#### Testing & Deployment

```powershell
# Verify installation
Test-Phase2BInstallation

# Run tests
Invoke-Phase2BTests

# Deploy
Invoke-Phase2BDeploy

# Check status
Get-Phase2BStatus
```

#### Convenient Aliases

```powershell
llm-profile      # Set-LLMProfile
llm-test         # Test-Phase2BInstallation
llm-deploy       # Invoke-Phase2BDeploy
llm-status       # Get-Phase2BStatus
rag-init         # Initialize-RagRedisMcp
```

### Lazy Loading

Phase 2B uses lazy loading for optimal performance:

- Commands are available immediately in your profile
- Modules load only when first used
- No impact on PowerShell startup time

```powershell
# This loads modules on first use
Set-LLMProfile -ProfileName "test" -WorkingDirectory $PWD

# Subsequent calls use already-loaded modules
Get-LLMProfile
```

---

## Troubleshooting

### Common Issues

#### Issue: "PowerShell version too old"

**Cause**: PowerShell 5.1 or earlier

**Solution**:
```powershell
# Check version
$PSVersionTable.PSVersion

# Install PowerShell 7+
winget install Microsoft.PowerShell
```

#### Issue: "Rust compiler not found"

**Cause**: Rust toolchain not installed

**Solution**:
```powershell
# Download and install from https://rustup.rs/
# Or via winget
winget install Rustlang.Rustup

# Verify installation
rustc --version
cargo --version
```

#### Issue: "Phase 1/2A modules not loaded"

**Cause**: Modules not dot-sourced

**Solution**:
```powershell
# Load manually
. C:\codedev\llm\gemma\PHASE1_IMPLEMENTATION.ps1
. C:\codedev\llm\gemma\PHASE2A_AUTO_CLAUDE.ps1
. C:\codedev\llm\gemma\PHASE2A_RAG_REDIS.ps1
. C:\codedev\llm\gemma\PHASE2A_TOOL_HOOKS.ps1
. C:\codedev\llm\gemma\PHASE2B_RAG_INTEGRATION.ps1

# Or use core profile integration
. C:\codedev\llm\gemma\PHASE2B_CORE_PROFILE_INTEGRATION.ps1
```

#### Issue: "MCP server binary not found"

**Cause**: Binary not installed or wrong location

**Solution**:
```powershell
# Check if binary exists
Test-Path "$env:USERPROFILE\.local\bin\rag-redis-mcp-server.exe"

# If not, deploy
Invoke-Phase2BDeploy

# Or build manually
cd C:\codedev\llm\rag-redis\rag-redis-system\mcp-server
cargo build --release
Copy-Item ..\..\target\release\mcp-server.exe `
    "$env:USERPROFILE\.local\bin\rag-redis-mcp-server.exe"
```

#### Issue: "Tests fail with timeout"

**Cause**: MCP server taking too long to start

**Solution**:
```powershell
# Increase timeout
$Script:TestConfig.TestTimeout = 60  # seconds

# Or skip server tests
Invoke-Phase2BTests -SkipServerTests
```

#### Issue: "Deployment fails during build"

**Cause**: Compilation errors or missing dependencies

**Solution**:
```powershell
# Check build logs
cd C:\codedev\llm\rag-redis\rag-redis-system\mcp-server
cargo build --release 2>&1 | Tee-Object -FilePath build.log

# Review build.log for errors

# Common fixes:
# 1. Update Rust toolchain
rustup update

# 2. Clean and rebuild
cargo clean
cargo build --release

# 3. Check dependencies
cargo check
```

### Diagnostic Commands

```powershell
# Full system check
Test-Phase2BInstallation -Detailed

# Test MCP server manually
$env:REDIS_URL='redis://localhost:6380'
$env:RUST_LOG='debug'
echo '{"jsonrpc":"2.0","id":1,"method":"ping"}' | `
    C:\users\david\.local\bin\rag-redis-mcp-server.exe

# Check Redis connection
Test-NetConnection -ComputerName localhost -Port 6380

# View test results
Get-Content $env:TEMP\phase2b-test-report-*.json | ConvertFrom-Json

# Check deployment logs
Get-Content $env:TEMP\phase2b-backup-*\* 
```

---

## Reference

### Command Reference

| Command | Alias | Purpose |
|---------|-------|---------|
| `Test-Phase2BInstallation` | `llm-test` | Verify installation |
| `Invoke-Phase2BTests` | - | Run test suite |
| `Invoke-Phase2BDeploy` | `llm-deploy` | Deploy Phase 2B |
| `Get-Phase2BStatus` | `llm-status` | Show status |
| `Set-LLMProfile` | `llm-profile` | Create/switch profile |
| `Get-LLMProfile` | - | Get profile info |
| `Initialize-RagRedisMcp` | `rag-init` | Init RAG-Redis |

### File Locations

| File | Location |
|------|----------|
| Test Suite | `C:\codedev\llm\gemma\PHASE2B_TESTS.ps1` |
| Deployment | `C:\codedev\llm\gemma\PHASE2B_DEPLOY.ps1` |
| Verification | `C:\codedev\llm\gemma\PHASE2B_VERIFY.ps1` |
| Integration | `C:\codedev\llm\gemma\PHASE2B_RAG_INTEGRATION.ps1` |
| Core Integration | `C:\codedev\llm\gemma\PHASE2B_CORE_PROFILE_INTEGRATION.ps1` |
| MCP Server | `C:\users\david\.local\bin\rag-redis-mcp-server.exe` |
| MCP Config | `C:\codedev\llm\rag-redis\mcp.json` |
| Profiles | `C:\codedev\llm\gemma\profiles\*.json` |

### Configuration

```json
{
  "TestConfig": {
    "TestDataDir": "%TEMP%\\phase2b-tests",
    "TestProfileName": "phase2b-test-<timestamp>",
    "RedisHost": "localhost",
    "RedisPort": 6380,
    "ServerPath": "~\\.local\\bin\\rag-redis-mcp-server.exe",
    "TestTimeout": 30
  },
  "DeployConfig": {
    "ProjectRoot": "C:\\codedev\\llm\\rag-redis",
    "TargetBinDir": "~\\.local\\bin",
    "BuildProfile": "release",
    "CreateBackup": true,
    "RunTests": true,
    "TestTimeout": 300
  }
}
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Tests failed / Deployment failed |

---

## Conclusion

Phase 2B provides a production-ready testing, deployment, and verification framework for the RAG-Redis MCP integration. All tools are seamlessly integrated into your PowerShell core profile with lazy loading for optimal performance.

**Next Steps:**
1. ✅ Verify installation: `Test-Phase2BInstallation`
2. ✅ Create your first profile: `Set-LLMProfile -ProfileName "test" -WorkingDirectory $PWD`
3. ✅ Initialize RAG-Redis: `Initialize-RagRedisMcp`
4. ✅ Test functionality: `Invoke-Phase2BTests`

For more information, see:
- `PHASE2B_COMPLETE.md` - Phase 2B implementation details
- `PHASE2B_RAG_INTEGRATION.ps1` - Integration script documentation
- `PHASE2B_TESTS.ps1` - Test suite inline documentation

---

**Document Version**: 1.0.0  
**Last Updated**: 2025-01-15  
**Status**: Complete
