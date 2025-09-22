#!/usr/bin/env python3
"""
Validation script for RAG Redis MCP setup.
Checks if all components are properly configured and accessible.
"""
import json
import os
import sys
from pathlib import Path
import subprocess
import asyncio

def validate_paths():
    """Validate that all required paths exist."""
    print("🔍 Validating paths...")
    
    paths_to_check = [
        "C:/codedev/llm/stats/mcp.json",
        "C:/codedev/llm/rag-redis/python-bridge",
        "C:/codedev/llm/rag-redis/data/rag",
        "C:/codedev/llm/rag-redis/cache/embeddings",
        "C:/codedev/llm/rag-redis/logs",
        "C:/codedev/llm/rag-redis/rag-redis-system",
    ]
    
    missing_paths = []
    for path in paths_to_check:
        if not Path(path).exists():
            missing_paths.append(path)
        else:
            print(f"  ✅ {path}")
    
    if missing_paths:
        print(f"  ❌ Missing paths:")
        for path in missing_paths:
            print(f"    - {path}")
        return False
    
    return True

def validate_mcp_config():
    """Validate MCP configuration file."""
    print("\n🔍 Validating MCP configuration...")
    
    config_path = "C:/codedev/llm/stats/mcp.json"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Check if rag-redis server is configured
        if "rag-redis" not in config.get("mcpServers", {}):
            print("  ❌ rag-redis server not found in MCP configuration")
            return False
        
        server_config = config["mcpServers"]["rag-redis"]
        
        # Validate required fields
        required_fields = ["command", "args", "cwd"]
        for field in required_fields:
            if field not in server_config:
                print(f"  ❌ Missing required field: {field}")
                return False
        
        # Check working directory
        cwd = server_config["cwd"]
        if not Path(cwd).exists():
            print(f"  ❌ Working directory does not exist: {cwd}")
            return False
        
        print("  ✅ MCP configuration is valid")
        print(f"  ✅ Working directory: {cwd}")
        print(f"  ✅ Command: {server_config['command']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to validate MCP config: {e}")
        return False

def validate_python_module():
    """Validate that the Python module can be imported."""
    print("\n🔍 Validating Python module...")
    
    # Change to the correct directory
    bridge_dir = Path("C:/codedev/llm/rag-redis/python-bridge")
    if not bridge_dir.exists():
        print(f"  ❌ Bridge directory not found: {bridge_dir}")
        return False
    
    # Check if the module file exists
    module_file = bridge_dir / "rag_redis_mcp" / "mcp_main.py"
    if not module_file.exists():
        print(f"  ❌ Module file not found: {module_file}")
        return False
    
    print(f"  ✅ Module file exists: {module_file}")
    
    # Try to validate the Python syntax
    try:
        with open(module_file, 'r', encoding='utf-8') as f:
            code = f.read()
        compile(code, str(module_file), 'exec')
        print("  ✅ Python module syntax is valid")
        return True
    except SyntaxError as e:
        print(f"  ❌ Python syntax error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Error validating Python module: {e}")
        return False

def validate_environment():
    """Validate environment variables and dependencies."""
    print("\n🔍 Validating environment...")
    
    # Check for uv command
    try:
        result = subprocess.run(["uv", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"  ✅ uv is available: {result.stdout.strip()}")
        else:
            print("  ❌ uv command failed")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("  ❌ uv command not found")
        return False
    
    # Check Python version
    try:
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        print(f"  ✅ Python version: {python_version}")
        
        if sys.version_info < (3, 8):
            print("  ⚠️  Python 3.8+ recommended")
    except Exception as e:
        print(f"  ❌ Error checking Python version: {e}")
        return False
    
    return True

def validate_rust_binary():
    """Check if Rust binary exists (optional)."""
    print("\n🔍 Validating Rust binary...")
    
    rust_binary = "C:/codedev/llm/rag-redis/rag-redis-system/mcp-server/target/release/mcp-server.exe"
    
    if Path(rust_binary).exists():
        print(f"  ✅ Rust binary found: {rust_binary}")
        return True
    else:
        print(f"  ⚠️  Rust binary not found: {rust_binary}")
        print("     This is optional - the system can work without it")
        return True  # Not a failure

def print_summary(results):
    """Print validation summary."""
    print("\n" + "="*60)
    print("📋 VALIDATION SUMMARY")
    print("="*60)
    
    all_passed = all(results.values())
    
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {check}")
    
    print("-"*60)
    if all_passed:
        print("🎉 All validations passed! Setup is ready.")
        print("\nNext steps:")
        print("1. Ensure Redis is running on port 6380")
        print("2. Test the MCP server connection")
        print("3. Try ingesting a test document")
    else:
        print("❌ Some validations failed. Please fix the issues above.")
        return False
    
    return True

def main():
    """Main validation function."""
    print("🔧 RAG Redis MCP Setup Validation")
    print("="*50)
    
    # Run all validations
    results = {
        "Paths": validate_paths(),
        "MCP Configuration": validate_mcp_config(),
        "Python Module": validate_python_module(),
        "Environment": validate_environment(),
        "Rust Binary": validate_rust_binary(),
    }
    
    # Print summary and exit with appropriate code
    success = print_summary(results)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()