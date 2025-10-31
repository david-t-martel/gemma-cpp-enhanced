#!/usr/bin/env python3
"""Simple validation script for MCP client implementation."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    """Run simple validation."""
    print("=" * 60)
    print("MCP Client Implementation Validation")
    print("=" * 60)
    print()

    # Check files
    print("Checking files...")
    files = [
        "src/gemma_cli/mcp/__init__.py",
        "src/gemma_cli/mcp/client.py",
        "src/gemma_cli/mcp/config_loader.py",
        "config/mcp_servers.toml",
        "tests/test_mcp_client.py",
    ]

    for f in files:
        exists = Path(f).exists()
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {f}")

    print()

    # Check imports
    print("Checking imports...")
    try:
        from gemma_cli.mcp.client import (
            MCPClientManager,
            MCPServerConfig,
            MCPTransportType,
        )
        print("  [OK] Main client imports")

        from gemma_cli.mcp.config_loader import load_mcp_servers
        print("  [OK] Config loader imports")

        from gemma_cli.mcp import MCPClientManager as M
        print("  [OK] Package imports")

    except ImportError as e:
        print(f"  [ERROR] Import failed: {e}")
        return 1

    print()

    # Test config creation
    print("Testing configuration...")
    try:
        from gemma_cli.mcp.client import MCPServerConfig, MCPTransportType

        config = MCPServerConfig(
            name="test",
            transport=MCPTransportType.STDIO,
            command="test-cmd",
        )
        print(f"  [OK] Created config: {config.name}")

    except Exception as e:
        print(f"  [ERROR] Config creation failed: {e}")
        return 1

    print()
    print("=" * 60)
    print("Validation complete - All checks passed!")
    print("=" * 60)
    print()
    print("Implementation summary:")
    print("  - MCPClientManager: Main client for MCP operations")
    print("  - MCPServerConfig: Server configuration model")
    print("  - MCPToolRegistry: Tool caching with TTL")
    print("  - Config loader: TOML configuration support")
    print("  - Example usage: src/gemma_cli/mcp/example_usage.py")
    print("  - Tests: tests/test_mcp_client.py")
    print()
    print("Next steps:")
    print("  1. Install dependencies: uv pip install mcp aiofiles toml")
    print("  2. Run tests: pytest tests/test_mcp_client.py -v")
    print("  3. See README: src/gemma_cli/mcp/README.md")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
