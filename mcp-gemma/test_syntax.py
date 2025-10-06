#!/usr/bin/env python3
"""
Simple syntax validation test for consolidated MCP server.
Tests that all modules can be imported without execution.
"""

import ast
import sys
from pathlib import Path


def validate_python_syntax(file_path):
    """Validate Python syntax of a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()

        # Parse the AST to check syntax
        ast.parse(source, filename=str(file_path))
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def main():
    """Main test function."""
    project_root = Path(__file__).parent
    server_dir = project_root / "server"

    # Files to validate
    python_files = [
        server_dir / "chat_handler.py",
        server_dir / "consolidated_server.py",
        server_dir / "base.py",
        server_dir / "handlers.py",
        server_dir / "transports.py",
        server_dir / "main.py",
        project_root / "build.py",
    ]

    print("🧪 Testing Python syntax validation...")

    all_valid = True
    for file_path in python_files:
        if file_path.exists():
            valid, error = validate_python_syntax(file_path)
            if valid:
                print(f"✅ {file_path.name}: Syntax OK")
            else:
                print(f"❌ {file_path.name}: {error}")
                all_valid = False
        else:
            print(f"⚠️  {file_path.name}: File not found")
            all_valid = False

    # Test C++ files exist
    cpp_dir = project_root / "cpp-server"
    cpp_files = [
        "mcp_server.h",
        "mcp_server.cpp",
        "inference_handler.h",
        "inference_handler.cpp",
        "model_manager.h",
        "model_manager.cpp",
        "main.cpp",
        "CMakeLists.txt",
    ]

    print("\n🔧 Testing C++ files existence...")
    for cpp_file in cpp_files:
        file_path = cpp_dir / cpp_file
        if file_path.exists():
            print(f"✅ {cpp_file}: Present")
        else:
            print(f"❌ {cpp_file}: Missing")
            all_valid = False

    # Test directory structure
    print("\n📁 Testing directory structure...")
    required_dirs = [
        server_dir,
        cpp_dir,
        project_root / "client",
        project_root / "tests",
        project_root / ".archive",
    ]

    for dir_path in required_dirs:
        if dir_path.exists():
            print(f"✅ {dir_path.name}/: Present")
        else:
            print(f"❌ {dir_path.name}/: Missing")

    # Test archive contents
    print("\n📦 Testing archive contents...")
    archive_dir = project_root / ".archive" / "original-implementations"
    if archive_dir.exists():
        archived_files = list(archive_dir.glob("*"))
        print(f"✅ Archive contains {len(archived_files)} files")
        for archived_file in archived_files:
            print(f"   📄 {archived_file.name}")
    else:
        print("⚠️  Archive directory not found")

    # Summary
    print(f"\n{'🎉 All tests passed!' if all_valid else '❌ Some tests failed'}")

    if all_valid:
        print("\n📋 Next steps:")
        print("1. Install Python dependencies: pip install mcp asyncio redis aiohttp websockets pydantic")
        print("2. Build C++ server: cd cpp-server && mkdir build && cd build && cmake .. && cmake --build .")
        print("3. Test with a model: python server/consolidated_server.py --model /path/to/model.sbs")

    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())