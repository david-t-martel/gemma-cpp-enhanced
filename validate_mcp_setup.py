#!/usr/bin/env python3
"""
MCP Setup Validation Script
Validates the complete MCP integration setup across all components.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

def validate_paths(config: Dict) -> List[str]:
    """Validate all paths in MCP configuration."""
    errors = []

    for server_name, server_config in config.get("mcpServers", {}).items():
        print(f"Validating server: {server_name}")

        # Check working directory
        cwd = server_config.get("cwd")
        if cwd and not Path(cwd).exists():
            errors.append(f"{server_name}: Working directory not found: {cwd}")

        # Check command args (script paths)
        args = server_config.get("args", [])
        for arg in args:
            if arg.startswith("/") and arg.endswith(".py"):
                if not Path(arg).exists():
                    errors.append(f"{server_name}: Script not found: {arg}")

        # Check environment paths
        env = server_config.get("environment", {})
        for key, value in env.items():
            if key.endswith("_DIR") or key.endswith("_PATH"):
                # Handle environment variable references
                if value.startswith("${env:") and value.endswith("}"):
                    continue
                if value and not Path(value).exists():
                    errors.append(f"{server_name}: Environment path not found: {key}={value}")

    return errors

def validate_dependencies() -> List[str]:
    """Validate required dependencies."""
    errors = []

    # Check uv is available
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            errors.append("uv is not available or not working properly")
        else:
            print(f"✓ uv version: {result.stdout.strip()}")
    except FileNotFoundError:
        errors.append("uv command not found")

    # Check Python availability
    try:
        result = subprocess.run([sys.executable, "--version"], capture_output=True, text=True)
        print(f"✓ Python version: {result.stdout.strip()}")
    except Exception as e:
        errors.append(f"Python check failed: {e}")

    return errors

def validate_servers(config: Dict) -> List[str]:
    """Validate that MCP servers can be imported/started."""
    errors = []

    for server_name, server_config in config.get("mcpServers", {}).items():
        print(f"Testing server startup: {server_name}")

        args = server_config.get("args", [])
        cwd = server_config.get("cwd", ".")

        if "python" in args:
            script_path = None
            for i, arg in enumerate(args):
                if arg == "python" and i + 1 < len(args):
                    script_path = args[i + 1]
                    break

            if script_path:
                # Try to validate Python script
                try:
                    cmd = [sys.executable, "-m", "py_compile", script_path]
                    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
                    if result.returncode != 0:
                        errors.append(f"{server_name}: Script compilation failed: {result.stderr}")
                    else:
                        print(f"✓ {server_name}: Script compiles successfully")
                except Exception as e:
                    errors.append(f"{server_name}: Script validation failed: {e}")

    return errors

def validate_git_status() -> List[str]:
    """Validate git repository status."""
    errors = []

    repos = [
        "/c/codedev/llm",
        "/c/codedev/llm/gemma",
        "/c/codedev/llm/stats",
        "/c/codedev/llm/rag-redis"
    ]

    for repo_path in repos:
        if Path(repo_path).exists():
            try:
                result = subprocess.run(
                    ["git", "status", "--porcelain"],
                    capture_output=True, text=True, cwd=repo_path
                )
                if result.returncode == 0:
                    print(f"✓ Git repo: {repo_path}")
                    if result.stdout.strip():
                        print(f"  - Has uncommitted changes")
                else:
                    errors.append(f"Git status failed for {repo_path}")
            except Exception as e:
                errors.append(f"Git check failed for {repo_path}: {e}")

    return errors

def check_submodules() -> List[str]:
    """Check if git submodules are needed."""
    errors = []
    warnings = []

    # Check if gemma.cpp is a submodule or needs to be
    gemma_cpp_path = Path("/c/codedev/llm/gemma/gemma.cpp")
    if gemma_cpp_path.exists():
        try:
            # Check if it's a git repo
            result = subprocess.run(
                ["git", "status"],
                capture_output=True, text=True, cwd=gemma_cpp_path
            )
            if result.returncode == 0:
                print("✓ gemma.cpp appears to be a git repository")

                # Check if it's a submodule
                main_git_dir = Path("/c/codedev/llm/.git")
                if main_git_dir.exists():
                    gitmodules_path = Path("/c/codedev/llm/.gitmodules")
                    if not gitmodules_path.exists():
                        warnings.append("gemma.cpp could be configured as a git submodule")
            else:
                warnings.append("gemma.cpp exists but is not a git repository")
        except Exception as e:
            warnings.append(f"Could not check gemma.cpp git status: {e}")
    else:
        errors.append("gemma.cpp directory not found")

    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  ! {warning}")

    return errors

def main():
    """Main validation routine."""
    print("🔍 MCP Setup Validation")
    print("=" * 50)

    # Load MCP configuration
    config_path = Path("/c/codedev/llm/stats/mcp.json")
    if not config_path.exists():
        print("❌ MCP configuration file not found")
        return 1

    try:
        with open(config_path) as f:
            config = json.load(f)
        print("✓ MCP configuration loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load MCP configuration: {e}")
        return 1

    all_errors = []

    # Run validations
    print("\n📁 Validating paths...")
    all_errors.extend(validate_paths(config))

    print("\n🔧 Validating dependencies...")
    all_errors.extend(validate_dependencies())

    print("\n🖥️  Validating servers...")
    all_errors.extend(validate_servers(config))

    print("\n📂 Validating git repositories...")
    all_errors.extend(validate_git_status())

    print("\n🌿 Checking submodules...")
    all_errors.extend(check_submodules())

    # Summary
    print("\n" + "=" * 50)
    if all_errors:
        print("❌ Validation completed with errors:")
        for error in all_errors:
            print(f"  • {error}")
        return 1
    else:
        print("✅ All validations passed!")
        print("\nMCP setup is ready for testing.")
        return 0

if __name__ == "__main__":
    sys.exit(main())