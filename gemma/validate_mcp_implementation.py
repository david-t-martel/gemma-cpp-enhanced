#!/usr/bin/env python3
"""Validation script for MCP client implementation.

This script verifies:
- All required files exist
- Imports work correctly
- Type hints are valid
- Basic configuration can be loaded
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def validate_files() -> bool:
    """Check all required files exist."""
    print("Validating files...")

    required_files = [
        "src/gemma_cli/mcp/__init__.py",
        "src/gemma_cli/mcp/client.py",
        "src/gemma_cli/mcp/config_loader.py",
        "src/gemma_cli/mcp/example_usage.py",
        "src/gemma_cli/mcp/README.md",
        "config/mcp_servers.toml",
        "tests/test_mcp_client.py",
    ]

    missing = []
    for file_path in required_files:
        path = Path(file_path)
        if not path.exists():
            missing.append(file_path)
            print(f"  ✗ Missing: {file_path}")
        else:
            print(f"  ✓ Found: {file_path}")

    if missing:
        print(f"\nError: {len(missing)} required files missing")
        return False

    print("✓ All required files present\n")
    return True


def validate_imports() -> bool:
    """Check all imports work."""
    print("Validating imports...")

    try:
        # Import main client
        from gemma_cli.mcp.client import (
            MCPClientManager,
            MCPServerConfig,
            MCPTransportType,
            MCPServerStatus,
            MCPError,
            MCPConnectionError,
            MCPToolExecutionError,
            MCPResourceError,
            MCPToolRegistry,
            CachedTool,
            ServerConnection,
        )
        print("  ✓ Main client imports")

        # Import config loader
        from gemma_cli.mcp.config_loader import (
            MCPConfigLoader,
            load_mcp_servers,
            validate_mcp_config,
        )
        print("  ✓ Config loader imports")

        # Import from package
        from gemma_cli.mcp import (
            MCPClientManager as MCPClientManager2,
            MCPServerConfig as MCPServerConfig2,
        )
        print("  ✓ Package-level imports")

        print("✓ All imports successful\n")
        return True

    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        return False


def validate_types() -> bool:
    """Check type definitions."""
    print("Validating types...")

    try:
        from gemma_cli.mcp.client import MCPServerConfig, MCPTransportType

        # Test creating config
        config = MCPServerConfig(
            name="test",
            transport=MCPTransportType.STDIO,
            command="test-command",
            args=["--test"],
        )

        assert config.name == "test"
        assert config.transport == MCPTransportType.STDIO
        assert config.command == "test-command"
        print("  ✓ MCPServerConfig creation")

        # Test enum
        assert MCPTransportType.STDIO.value == "stdio"
        assert MCPTransportType.HTTP.value == "http"
        print("  ✓ MCPTransportType enum")

        print("✓ Type validation successful\n")
        return True

    except Exception as e:
        print(f"  ✗ Type validation error: {e}")
        return False


def validate_config_file() -> bool:
    """Check configuration file."""
    print("Validating config file...")

    try:
        import toml

        config_path = Path("config/mcp_servers.toml")
        if not config_path.exists():
            print("  ✗ Config file not found")
            return False

        with open(config_path, encoding="utf-8") as f:
            config_data = toml.load(f)

        print(f"  ✓ Config file loaded ({len(config_data)} servers)")

        for name, server_config in config_data.items():
            required_fields = ["enabled", "transport"]
            missing = [f for f in required_fields if f not in server_config]
            if missing:
                print(f"  ✗ Server '{name}' missing fields: {missing}")
                return False
            print(f"  ✓ Server '{name}' valid")

        print("✓ Config file validation successful\n")
        return True

    except Exception as e:
        print(f"  ✗ Config validation error: {e}")
        return False


def validate_documentation() -> bool:
    """Check documentation exists."""
    print("Validating documentation...")

    readme_path = Path("src/gemma_cli/mcp/README.md")
    if not readme_path.exists():
        print("  ✗ README.md not found")
        return False

    content = readme_path.read_text()
    required_sections = [
        "# MCP Client Manager",
        "## Features",
        "## Installation",
        "## Quick Start",
        "## Configuration Reference",
        "## API Reference",
        "## Error Handling",
        "## Examples",
    ]

    missing = []
    for section in required_sections:
        if section not in content:
            missing.append(section)

    if missing:
        print(f"  ✗ Missing sections: {missing}")
        return False

    print(f"  ✓ README.md complete ({len(content)} chars)")
    print("✓ Documentation validation successful\n")
    return True


def main() -> int:
    """Run all validations."""
    print("=" * 60)
    print("MCP Client Implementation Validation")
    print("=" * 60)
    print()

    results = {
        "Files": validate_files(),
        "Imports": validate_imports(),
        "Types": validate_types(),
        "Config": validate_config_file(),
        "Documentation": validate_documentation(),
    }

    print("=" * 60)
    print("Validation Results")
    print("=" * 60)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:<20} {status}")

    print("=" * 60)

    if all(results.values()):
        print("\n✓ All validations passed!")
        print("\nThe MCP client implementation is ready for use.")
        print("\nNext steps:")
        print("  1. Run tests: uv run pytest tests/test_mcp_client.py -v")
        print("  2. Try examples: uv run python -m gemma_cli.mcp.example_usage")
        print("  3. Integrate with Gemma CLI")
        return 0
    else:
        print("\n✗ Some validations failed")
        print("\nPlease fix the issues above before using the MCP client.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
